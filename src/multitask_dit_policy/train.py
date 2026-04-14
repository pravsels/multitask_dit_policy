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
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from lerobot.datasets.utils import cycle
from robocandywrapper import make_dataset_without_config
from robocandywrapper.plugins import ControlModePlugin
from torch.utils.data import DataLoader
from tqdm import tqdm

from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.utils.configuration import DatasetSchema, MultiTaskDiTConfig
from multitask_dit_policy.utils.dataset_adapter import (
    adapt_batch,
    compute_adapted_features,
)
from multitask_dit_policy.utils.distributed_sampler import DistributedIndexSampler
from multitask_dit_policy.utils.valid_indices import (
    VALID_INDICES_FILENAME,
    compute_valid_indices,
    write_filtering_report,
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


@dataclass(frozen=True)
class RuntimeContext:
    use_ddp: bool
    world_size: int
    rank: int
    local_rank: int
    device: str
    autocast_device_type: str
    is_main_process: bool


@dataclass
class TrainConfig:
    # Dataset parameters
    dataset_path: str
    run_name: str = "default"
    dataset_schema: DatasetSchema = field(default_factory=DatasetSchema)

    # Training parameters
    batch_size: int = 16
    num_workers: int = 2
    prefetch_factor: int = 2
    train_steps: int = 10_000
    save_freq: int = 500
    keep_freq: int | None = None
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
        if self.keep_freq is not None and self.keep_freq <= 0:
            raise ValueError(f"keep_freq must be positive when set, got {self.keep_freq}")
        if self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive, got {self.prefetch_factor}")


def get_runtime_context(configured_device: str) -> RuntimeContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    use_ddp = world_size > 1

    if torch.cuda.is_available() and configured_device.startswith("cuda"):
        device = f"cuda:{local_rank}" if use_ddp else f"cuda:{local_rank}"
        autocast_device_type = "cuda"
    else:
        device = configured_device
        autocast_device_type = "cpu" if configured_device.startswith("cpu") else configured_device.split(":", 1)[0]

    return RuntimeContext(
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        autocast_device_type=autocast_device_type,
        is_main_process=rank == 0,
    )


def setup_distributed(runtime_context: RuntimeContext) -> None:
    if torch.cuda.is_available() and runtime_context.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(runtime_context.device))

    if runtime_context.use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=30))


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def load_or_compute_ramen_stats(
    *,
    dataset,
    schema: DatasetSchema,
    norm_mask: torch.Tensor,
    cache_path: str | Path | None,
    device: str,
    runtime_context: RuntimeContext,
) -> dict[str, torch.Tensor]:
    kwargs = dict(schema=schema, norm_mask=norm_mask, cache_path=cache_path, device=device)
    if not runtime_context.use_ddp:
        return compute_ramen_stats(dataset, **kwargs)

    if runtime_context.is_main_process:
        stats = compute_ramen_stats(dataset, **kwargs)
        dist.barrier()
        return stats

    dist.barrier()
    return compute_ramen_stats(dataset, **kwargs)


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


def get_amp_settings(policy: torch.nn.Module, *, device: str, use_amp: bool) -> tuple[bool, torch.dtype | None, bool]:
    """Choose autocast dtype and whether gradient scaling is safe."""
    amp_enabled = use_amp and device.startswith("cuda")
    if not amp_enabled:
        return False, None, False

    param_dtypes = {param.dtype for param in policy.parameters() if param.is_floating_point()}
    if torch.bfloat16 in param_dtypes:
        return True, torch.bfloat16, False

    return True, torch.float16, True


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
    runtime_context = get_runtime_context(cfg.device)
    setup_distributed(runtime_context)

    try:
        seed = cfg.seed + runtime_context.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if runtime_context.device == "cpu" or runtime_context.device.startswith("cpu"):
            if runtime_context.is_main_process:
                logging.warning("=" * 80)
                logging.warning("WARNING: Config device is set to CPU")
                logging.warning("Training will be significantly slower than on GPU")

        run_dir = Path(cfg.output_dir) / cfg.run_name
        if runtime_context.is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)
        if runtime_context.use_ddp:
            dist.barrier()

        use_wandb = runtime_context.is_main_process and "WANDB_API_KEY" in os.environ
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

        # Create dataset — robocandywrapper handles multi-dataset configs
        # ("[repo1, repo2, ...]") and provides combined metadata.
        dataset = make_dataset_without_config(
            repo_id=repo_id,
            action_delta_indices=list(cfg.policy.action_delta_indices),
            observation_delta_indices=list(cfg.policy.observation_delta_indices),
            root=root,
            video_backend="pyav",
            use_imagenet_stats=cfg.policy.observation_encoder.use_imagenet_stats,
            plugins=[ControlModePlugin()],
        )

        # All metadata comes from dataset.meta (robocandywrapper's combined
        # metadata for multi-dataset, or LeRobot metadata for single-dataset).
        ds_meta = dataset.meta

        # Task text lookup for CLIP conditioning.
        # .tasks is a plain dict (multi-dataset) or pandas DataFrame (single).
        tasks = ds_meta.tasks
        if isinstance(tasks, dict):
            task_index_to_text = {int(k): str(v) for k, v in tasks.items()}
        else:
            task_index_to_text = {row.task_index: task for task, row in tasks.iterrows()}
        
        if runtime_context.is_main_process:
            logging.info(f"Task descriptions: {task_index_to_text}")

        schema = cfg.dataset_schema
        if runtime_context.is_main_process:
            logging.info(
                f"Dataset schema — state keys: {schema.state_keys} ({schema.state_dim}D), "
                f"action keys: {schema.action_keys} ({schema.action_dim}D)"
            )

        # Build features and norm mask.
        rot6d_start, rot6d_end = schema.rot6d_slice
        input_features, output_features = compute_adapted_features(
            ds_meta.features, schema,
        )
        norm_mask = build_norm_mask(
            max(schema.state_dim, schema.action_dim), rot6d_start, rot6d_end,
        )

        cfg.policy.input_features = input_features
        cfg.policy.output_features = output_features
        cfg.policy.dataset_schema = schema

        # Auto-detect latest checkpoint in run_dir for resume
        checkpoint_path = cfg.checkpoint_path
        if checkpoint_path is None:
            ckpt_dirs = sorted(run_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
            if ckpt_dirs:
                checkpoint_path = str(ckpt_dirs[-1])
                if runtime_context.is_main_process:
                    logging.info(f"Auto-detected latest checkpoint: {checkpoint_path}")

        # Load or create policy
        if checkpoint_path is not None:
            if runtime_context.is_main_process:
                logging.info(f"Loading policy from checkpoint: {checkpoint_path}")
            policy = MultiTaskDiTPolicy.load(checkpoint_path)
            policy_config = policy.config
        else:
            policy_config = cfg.policy
            policy = MultiTaskDiTPolicy(policy_config)

        policy_config.device = runtime_context.device
        policy.to(runtime_context.device)
        policy.train()

        # Compute Ramen per-timestep percentile stats (cached to disk).
        # Stats are computed by bulk-reading numerical columns directly from
        # the underlying parquet files, bypassing video decoding entirely.
        stats_cache = run_dir / "ramen_stats.pt"
        ramen_stats = load_or_compute_ramen_stats(
            dataset=dataset,
            schema=schema,
            norm_mask=norm_mask,
            cache_path=stats_cache,
            device=runtime_context.device,
            runtime_context=runtime_context,
        )
        norm_mask = norm_mask.to(runtime_context.device)

        # Compute valid indices: filters out policy frames from DAgger datasets
        # and drops the last N frames per episode for action horizon safety.
        valid_indices, filtering_report = compute_valid_indices(
            dataset, drop_n_last_frames=cfg.policy.drop_n_last_frames,
        )
        total_frames = len(dataset)
        if runtime_context.is_main_process:
            if len(valid_indices) < total_frames:
                logging.info(
                    "DAgger filtering: keeping %d/%d frames (%.1f%%)",
                    len(valid_indices), total_frames,
                    100 * len(valid_indices) / max(total_frames, 1),
                )
            else:
                logging.info("No DAgger policy frames detected, using all %d frames", total_frames)
            write_filtering_report(filtering_report, run_dir / VALID_INDICES_FILENAME)

        sampler = DistributedIndexSampler(
            indices=valid_indices,
            num_replicas=runtime_context.world_size,
            rank=runtime_context.rank,
            shuffle=True,
            drop_last=False,
            seed=cfg.seed,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=runtime_context.autocast_device_type == "cuda",
            persistent_workers=cfg.num_workers > 0,
            drop_last=False,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
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

        amp_enabled, amp_dtype, use_grad_scaler = get_amp_settings(
            policy,
            device=runtime_context.device,
            use_amp=cfg.use_amp,
        )
        scaler = torch.amp.GradScaler(enabled=use_grad_scaler)
        if amp_enabled and runtime_context.is_main_process:
            scaler_mode = "enabled" if use_grad_scaler else "disabled"
            logging.info(f"Using Automatic Mixed Precision (dtype={amp_dtype}, grad_scaler={scaler_mode})")

        if runtime_context.use_ddp:
            policy = torch.nn.parallel.DistributedDataParallel(
                policy,
                device_ids=[runtime_context.local_rank] if runtime_context.autocast_device_type == "cuda" else None,
            )

        step = 0

        # Resume training state from checkpoint
        if checkpoint_path is not None:
            train_state_path = Path(checkpoint_path) / "train_state.pt"
            if train_state_path.exists():
                # Deserialize to CPU — optimizer state is ~2x model size
                # (Adam momentum + variance), loading to GPU would briefly
                # hold two copies and OOM.
                train_state = torch.load(train_state_path, map_location="cpu", weights_only=True)
                step = train_state["step"]
                # load_state_dict copies each tensor to param.device (GPU),
                # leaving the CPU originals in train_state for cleanup below
                optimizer.load_state_dict(train_state["optimizer"])
                if use_grad_scaler and train_state.get("scaler"):
                    scaler.load_state_dict(train_state["scaler"])
                if scheduler is not None and train_state.get("scheduler") is not None:
                    scheduler.load_state_dict(train_state["scheduler"])
                # Free the CPU copies now that optimizer holds GPU state
                del train_state
                if runtime_context.is_main_process:
                    logging.info(f"Resumed training state from step {step}")
            elif runtime_context.is_main_process:
                logging.warning("No train_state.pt in checkpoint, starting optimizer from scratch")

        progress_bar = tqdm(total=cfg.train_steps, initial=step, disable=not runtime_context.is_main_process)
        epoch = 0
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)

        while step < cfg.train_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                epoch += 1
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            batch = move_to_device(batch, runtime_context.device, non_blocking=True)
            batch = adapt_batch(batch, schema, norm_mask)

            # Ensure task text exists for CLIP conditioning
            if "task" not in batch and "task_index" in batch:
                batch["task"] = [task_index_to_text[idx.item()] for idx in batch["task_index"]]

            normalized_batch = ramen_normalize_batch(batch, ramen_stats, norm_mask)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=runtime_context.autocast_device_type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                loss, _ = policy(normalized_batch)

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(policy).parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(policy).parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            step += 1

            if runtime_context.is_main_process and step % cfg.save_freq == 0:
                save_path = run_dir / f"checkpoint_{step}"
                save_policy(unwrap_model(policy), save_path)
                torch.save(ramen_stats, save_path / "ramen_stats.pt")
                torch.save({
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(),
                }, save_path / "train_state.pt")
                logging.info(f"Saved checkpoint to {save_path}")
                should_keep_checkpoint = (
                    cfg.keep_freq is None
                    or step % cfg.keep_freq == 0
                    or step == cfg.train_steps
                )
                if not should_keep_checkpoint:
                    shutil.rmtree(save_path)
                    logging.info(
                        "Pruned checkpoint %s (keep_freq=%s)",
                        save_path.name,
                        cfg.keep_freq,
                    )
            if runtime_context.use_ddp and step % cfg.save_freq == 0:
                dist.barrier()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

            if runtime_context.is_main_process and step % cfg.log_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                logging.info(f"Step {step}: loss={loss.item():.6f} grad_norm={grad_norm:.4f} lr={lr:.2e}")
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/lr": lr,
                    }, step=step)

        if runtime_context.is_main_process:
            final_dir = run_dir / "final_model"
            save_policy(unwrap_model(policy), final_dir)
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
        if runtime_context.use_ddp:
            dist.barrier()
    finally:
        cleanup_distributed()


@draccus.wrap()
def main(cfg: TrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
