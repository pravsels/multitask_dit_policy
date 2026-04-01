#!/usr/bin/env python

# Copyright 2025 Bryson Jones and the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Training script for Multi-Task Diffusion Transformer (DiT) policy.

Credit is given to the https://github.com/huggingface/lerobot project
for which this training script is adapted from.
"""

import logging
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import wandb
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from robocandywrapper import make_dataset_without_config
from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig
from multitask_dit_policy.utils.dataset_adapter import (
    DEFAULT_ACTION_KEYS,
    DEFAULT_STATE_KEYS,
    ROT6D_END,
    ROT6D_START,
    adapt_batch,
    compute_adapted_features,
    detect_sub_features,
    select_default_keys,
)
from multitask_dit_policy.utils.ramen_normalization import (
    build_norm_mask,
    compute_ramen_stats,
    ramen_normalize_batch,
)
from multitask_dit_policy.utils.utils import move_to_device, save_policy

# Suppress Pydantic warnings from draccus ChoiceRegistry union types
# This is an interaction with draccus that we can't control
warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402


@dataclass
class TrainConfig:
    # Dataset parameters
    dataset_path: str
    run_name: str = "default"
    state_keys: list[str] = field(default_factory=lambda: list(DEFAULT_STATE_KEYS))
    action_keys: list[str] = field(default_factory=lambda: list(DEFAULT_ACTION_KEYS))
    rot6d_slice: tuple[int, int] = (ROT6D_START, ROT6D_END)

    # Training parameters
    batch_size: int = 16
    num_workers: int = 2
    train_steps: int = 10_000
    save_freq: int = 500
    log_freq: int = 25
    output_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    seed: int = 17
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    lr_scheduler_min_lr_scale: float = 0.1

    # Checkpoint loading
    checkpoint_path: str | None = None

    # Policy parameters
    policy: MultiTaskDiTConfig = field(default_factory=MultiTaskDiTConfig)

    def __post_init__(self):
        valid_schedulers = {"constant", "cosine"}
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(f"lr_scheduler must be one of {sorted(valid_schedulers)}, got {self.lr_scheduler}")
        if self.lr_warmup_steps < 0:
            raise ValueError(f"lr_warmup_steps must be non-negative, got {self.lr_warmup_steps}")
        if not 0.0 <= self.lr_scheduler_min_lr_scale <= 1.0:
            raise ValueError(
                "lr_scheduler_min_lr_scale must be in [0, 1], "
                f"got {self.lr_scheduler_min_lr_scale}"
            )


def compute_lr_scale(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> float:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if not 0.0 <= min_lr_scale <= 1.0:
        raise ValueError(f"min_lr_scale must be in [0, 1], got {min_lr_scale}")

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps

    if total_steps <= warmup_steps:
        return 1.0

    decay_steps = max(1, total_steps - warmup_steps - 1)
    decay_progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine_scale


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.lr_scheduler == "constant":
        return None

    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: compute_lr_scale(
                step,
                total_steps=cfg.train_steps,
                warmup_steps=cfg.lr_warmup_steps,
                min_lr_scale=cfg.lr_scheduler_min_lr_scale,
            ),
        )

    raise ValueError(f"Unsupported lr_scheduler: {cfg.lr_scheduler}")


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg.device == "cpu" or cfg.device.startswith("cpu"):
        logging.warning("=" * 80)
        logging.warning("WARNING: Config device is set to CPU")
        logging.warning("Training will be significantly slower than on GPU")

    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = "WANDB_API_KEY" in os.environ
    if use_wandb:
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "pravsels"),
            project="multitask-dit-policy",
            name=cfg.run_name,
            config=asdict(cfg),
            dir=str(run_dir),
        )

    # Resolve dataset location: only treat absolute paths as local datasets
    dataset_path = Path(cfg.dataset_path)
    if dataset_path.is_absolute() and dataset_path.is_dir():
        repo_id = dataset_path.name
        root = str(dataset_path)
    else:
        repo_id = cfg.dataset_path
        root = None

    # Load metadata for feature detection and episode boundaries
    ds_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

    # Build task_index → task_text lookup for CLIP conditioning
    task_index_to_text = {row.task_index: task for task, row in ds_metadata.tasks.iterrows()}
    logging.info(f"Task descriptions: {task_index_to_text}")

    # Detect sub-features and narrow to pos + eef_pose
    detected_state, detected_action = detect_sub_features(ds_metadata.features)
    state_keys = cfg.state_keys or select_default_keys(detected_state, detected_action)[0]
    action_keys = cfg.action_keys or select_default_keys(detected_state, detected_action)[1]
    logging.info(f"Sub-feature keys — state: {state_keys}, action: {action_keys}")

    # Build norm mask: True for dims that get delta + normalization, False for 6D rotation
    rot6d_start, rot6d_end = cfg.rot6d_slice
    input_features, output_features = compute_adapted_features(
        ds_metadata.features, state_keys, action_keys,
    )
    state_dim = input_features["observation.state"].shape[0]
    norm_mask = build_norm_mask(state_dim, rot6d_start, rot6d_end)

    cfg.policy.input_features = input_features
    cfg.policy.output_features = output_features

    # Auto-detect latest checkpoint in run_dir for resume
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None:
        ckpt_dirs = sorted(run_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
        if ckpt_dirs:
            checkpoint_path = str(ckpt_dirs[-1])
            logging.info(f"Auto-detected latest checkpoint: {checkpoint_path}")

    # Load or create policy
    if checkpoint_path is not None:
        logging.info(f"Loading policy from checkpoint: {checkpoint_path}")
        policy = MultiTaskDiTPolicy.load(checkpoint_path)
        policy_config = policy.config
    else:
        policy_config = cfg.policy
        policy = MultiTaskDiTPolicy(policy_config)

    policy_config.device = cfg.device
    policy.to(cfg.device)
    policy.train()

    # Create dataset via RoboCandyWrapper (handles delta_timestamps automatically)
    dataset = make_dataset_without_config(
        repo_id=repo_id,
        action_delta_indices=list(policy_config.action_delta_indices),
        observation_delta_indices=list(policy_config.observation_delta_indices),
        root=root,
        video_backend="pyav",
        use_imagenet_stats=True,
    )

    # Compute Ramen per-timestep percentile stats (cached to disk)
    stats_cache = run_dir / "ramen_stats.pt"
    ramen_stats = compute_ramen_stats(
        dataset,
        state_keys=state_keys,
        action_keys=action_keys,
        norm_mask=norm_mask,
        cache_path=stats_cache,
        device=cfg.device,
    )
    norm_mask = norm_mask.to(cfg.device)

    sampler = EpisodeAwareSampler(
        ds_metadata.episodes["dataset_from_index"],
        ds_metadata.episodes["dataset_to_index"],
        drop_n_last_frames=cfg.policy.drop_n_last_frames,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    optimizer_config = policy_config.get_optimizer_preset()
    optimizer_params = policy.get_optim_params()
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, cfg)

    use_amp = cfg.use_amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logging.info("Using Automatic Mixed Precision (AMP) for training")

    step = 0

    # Resume training state from checkpoint
    if checkpoint_path is not None:
        train_state_path = Path(checkpoint_path) / "train_state.pt"
        if train_state_path.exists():
            train_state = torch.load(train_state_path, map_location=cfg.device, weights_only=True)
            step = train_state["step"]
            optimizer.load_state_dict(train_state["optimizer"])
            scaler.load_state_dict(train_state["scaler"])
            if scheduler is not None and train_state.get("scheduler") is not None:
                scheduler.load_state_dict(train_state["scheduler"])
            logging.info(f"Resumed training state from step {step}")
        else:
            logging.warning(f"No train_state.pt in checkpoint, starting optimizer from scratch")

    progress_bar = tqdm(total=cfg.train_steps, initial=step)
    dataloader_iter = cycle(dataloader)

    while step < cfg.train_steps:
        batch = next(dataloader_iter)
        batch = move_to_device(batch, cfg.device, non_blocking=True)
        batch = adapt_batch(batch, state_keys, action_keys, norm_mask)

        # Ensure task text exists for CLIP conditioning
        if "task" not in batch and "task_index" in batch:
            batch["task"] = [task_index_to_text[idx.item()] for idx in batch["task_index"]]

        normalized_batch = ramen_normalize_batch(batch, ramen_stats, norm_mask)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=cfg.device, enabled=use_amp):
            loss, _ = policy(normalized_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        step += 1

        if step % cfg.save_freq == 0:
            save_path = run_dir / f"checkpoint_{step}"
            save_policy(policy, save_path)
            torch.save(ramen_stats, save_path / "ramen_stats.pt")
            torch.save({
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
            }, save_path / "train_state.pt")
            logging.info(f"Saved checkpoint to {save_path}")

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())

        if step % cfg.log_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Step {step}: loss={loss.item():.6f} grad_norm={grad_norm:.4f} lr={lr:.2e}")
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/lr": lr,
                }, step=step)

    final_dir = run_dir / "final_model"
    save_policy(policy, final_dir)
    torch.save(ramen_stats, final_dir / "ramen_stats.pt")
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
    }, final_dir / "train_state.pt")
    logging.info("Training finished.")
    if use_wandb:
        wandb.finish()


@draccus.wrap()
def main(cfg: TrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
