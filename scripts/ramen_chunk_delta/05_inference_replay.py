"""Run the trained checkpoint on a real frame and compare to the recorded chunk.

Phase 2.2b of the Ramen chunk-delta investigation.

For a chosen frame t* in a chosen episode (default: ep 166 t=25, the worst
joint_1 saturation case from task 2.2a), this script:

  1. Loads the policy from `pravsels/dit_coffee_capsules_config_fix/checkpoint_30000`
     (downloads model.safetensors via huggingface_hub if needed).
  2. Loads the buggy `(1, 17)` ramen_stats.pt from the same checkpoint.
  3. Builds the proper input batch via LeRobotDataset:
       observation.state.{pos,eef_pose}: 2 frames at t* and t*-1
       observation.images.{front,wrist}: 2 frames decoded from MP4
       action.{pos,eef_pose}: next 32 frames (the recorded chunk)
       task: language description
  4. Runs assemble_vector + chunk-relative delta exactly like training
     (`adapt_batch`), then ramen_normalize, then `_generate_actions`.
  5. Reports per-(k, dim) tables with:
       gt_raw   = act[t*+k] - obs[t*]                  (raw chunk delta)
       gt_norm  = clamp(normalize(gt_raw))             (training target)
       gt_round_trip = unnormalize(gt_norm)            (lossy floor)
       pred_norm = model output (in normalized space)
       pred_unnorm = unnormalize(pred_norm)            (deployed action delta)
       gap_norm   = gt_raw - gt_round_trip             (model-independent floor)
       gap_model  = gt_round_trip - pred_unnorm        (model fit error)
       total_gap  = gt_raw - pred_unnorm               (sum of A + B)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from multitask_dit_policy.model.model import MultiTaskDiTPolicy
from multitask_dit_policy.utils.dataset_adapter import adapt_batch
from multitask_dit_policy.utils.configuration import DatasetSchema, SchemaEntry
from multitask_dit_policy.utils.ramen_normalization import (
    load_ramen_stats,
    ramen_normalize,
    ramen_unnormalize,
)


REPO = "pravsels/dit_coffee_capsules_config_fix"
CHECKPOINT_DIR = "checkpoint_30000"
DATASET_REPO = "villekuosmanen/bin_pick_pack_coffee_capsules"
DATASET_ROOT = Path("data/lerobot/villekuosmanen/bin_pick_pack_coffee_capsules")

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = 32
N_OBS = 2
FPS = 20
ROT6D_SLICE = (10, 16)
DIM_LABELS = (
    [f"joint_{i}" for i in range(6)]
    + ["joint_gripper"]
    + ["eef_xyz_x", "eef_xyz_y", "eef_xyz_z"]
    + [f"rot6d_{i}" for i in range(6)]
    + ["eef_gripper"]
)


def build_schema() -> DatasetSchema:
    """Hardcode the schema from config/train_coffee_capsules.yaml."""
    return DatasetSchema(
        state=[
            SchemaEntry(key="observation.state.pos", dim=7, convert_rotation=False),
            SchemaEntry(key="observation.state.eef_pose", dim=7, convert_rotation=True),
        ],
        action=[
            SchemaEntry(key="action.pos", dim=7, convert_rotation=False),
            SchemaEntry(key="action.eef_pose", dim=7, convert_rotation=True),
        ],
        rot6d_slice=(10, 16),
    )


def episode_global_start(dataset: LeRobotDataset, episode_id: int) -> int:
    """Return the global frame index where the requested episode starts."""
    return int(dataset.meta.episodes[episode_id]["dataset_from_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=166, help="Episode id (default 166: worst joint_1 saturation)")
    parser.add_argument("--t", type=int, default=25, help="Frame index inside the episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Snapshot-downloading {REPO}/{CHECKPOINT_DIR} ...")
    ckpt_path = Path(snapshot_download(REPO, allow_patterns=[f"{CHECKPOINT_DIR}/*"])) / CHECKPOINT_DIR
    print(f"  ckpt at {ckpt_path}")

    print("Loading policy ...")
    policy = MultiTaskDiTPolicy.load(ckpt_path)
    policy.to(args.device).eval()

    stats_path = ckpt_path / "ramen_stats.json"
    if not stats_path.exists():
        stats_path = ckpt_path / "ramen_stats.pt"
    print(f"Loading {stats_path.name} ...")
    stats = load_ramen_stats(stats_path, device=args.device)
    obs_q02, obs_q98 = stats["obs_q02"], stats["obs_q98"]
    act_q02, act_q98 = stats["action_q02"], stats["action_q98"]
    norm_mask = stats["norm_mask"]

    print("Loading LeRobotDataset for image decode + state assembly ...")
    delta_obs_seconds = [-(N_OBS - 1 - i) / FPS for i in range(N_OBS)]   # [-0.05, 0.0]
    delta_act_seconds = [k / FPS for k in range(H)]                       # [0.0, 0.05, ..., 1.55]
    delta_timestamps = {
        "observation.state.pos": delta_obs_seconds,
        "observation.state.eef_pose": delta_obs_seconds,
        "observation.images.front": delta_obs_seconds,
        "observation.images.wrist": delta_obs_seconds,
        "action.pos": delta_act_seconds,
        "action.eef_pose": delta_act_seconds,
    }
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO, root=DATASET_ROOT,
        video_backend="pyav",
        delta_timestamps=delta_timestamps,
    )

    global_idx = episode_global_start(dataset, args.episode) + args.t
    print(f"  episode={args.episode}, local t={args.t}  -> global frame index {global_idx}")
    sample = dataset[global_idx]

    # ---- Build the model batch (same recipe as training: assemble + delta) ----
    schema = build_schema()
    batch_keys = [
        "observation.state.pos", "observation.state.eef_pose",
        "observation.images.front", "observation.images.wrist",
        "action.pos", "action.eef_pose",
        "task",
    ]
    batch: dict = {}
    for k in batch_keys:
        v = sample[k]
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(args.device)         # add batch dim
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    # Convert images to float in [0, 1] if they're uint8.
    for k in ("observation.images.front", "observation.images.wrist"):
        if batch[k].dtype == torch.uint8:
            batch[k] = batch[k].float() / 255.0

    # `adapt_batch` mutates a copy; pass norm_mask to subtract current obs on shared dims.
    adapted = adapt_batch(dict(batch), schema, norm_mask)
    # Now adapted has:
    #   observation.state: (1, 2, 17)
    #   action: (1, 32, 17) - chunk-relative deltas (same as training target)
    print(f"  observation.state shape: {tuple(adapted['observation.state'].shape)}")
    print(f"  action shape (chunk-relative deltas): {tuple(adapted['action'].shape)}")

    # ---- Compute ground-truth in all relevant spaces ----
    gt_raw = adapted["action"][0]                     # (32, 17), raw chunk-relative deltas
    gt_norm_clamped = ramen_normalize(gt_raw, act_q02, act_q98, norm_mask)
    gt_round_trip = ramen_unnormalize(gt_norm_clamped, act_q02, act_q98, norm_mask)

    denom = (act_q98 - act_q02).clamp_min(1e-8)
    gt_norm_unclamped = (2 * (gt_raw - act_q02) / denom) - 1   # may exceed ±1.5

    # ---- Run the model ----
    # Build the input batch the model expects:
    #   observation.state (1, 2, 17), normalized
    #   observation.images.* (1, 2, C, H, W), un-normalized RGB in [0, 1]; the model's
    #     observation_encoder applies any required image normalization internally.
    #   task: list[str]
    obs_state_norm = ramen_normalize(adapted["observation.state"], obs_q02, obs_q98, norm_mask)
    model_batch = {
        "observation.state": obs_state_norm,
        "observation.images.front": batch["observation.images.front"],
        "observation.images.wrist": batch["observation.images.wrist"],
        "task": batch["task"],
    }
    # Stack image features into observation.images as the model's forward path expects.
    from lerobot.utils.constants import OBS_IMAGES
    model_batch[OBS_IMAGES] = torch.stack(
        [model_batch[k] for k in policy.config.image_features], dim=-4
    )

    print("\nRunning policy.objective.conditional_sample ...")
    with torch.no_grad():
        cond = policy.observation_encoder.encode(model_batch)
        # full (B, horizon=32, action_dim) — same shape as the training target
        pred_norm = policy.objective.conditional_sample(policy.noise_predictor, 1, cond)[0]
    assert pred_norm.shape == gt_raw.shape, (pred_norm.shape, gt_raw.shape)
    pred_unnorm = ramen_unnormalize(pred_norm, act_q02, act_q98, norm_mask)

    # ---- Compute gaps ----
    gap_norm_total = gt_raw - gt_round_trip
    gap_model = gt_round_trip - pred_unnorm
    total_gap = gt_raw - pred_unnorm

    # ---- Per-dim summary across the 32-step chunk ----
    print(f"\n--- Per-dim summary across 32 chunk positions  (ep={args.episode}, t={args.t}) ---")
    print(
        f"{'dim':>3}  {'label':<14}  {'mask':>4}  "
        f"{'gt_raw_p2p':>10}  {'sat. cnt':>9}  "
        f"{'Σ|gap_norm|':>13}  {'Σ|gap_model|':>13}  {'Σ|total_gap|':>13}"
    )
    rows = []
    D = gt_raw.shape[-1]
    for d in range(D):
        masked = bool(norm_mask[d].item())
        p2p = float(gt_raw[:, d].max() - gt_raw[:, d].min())
        sat_cnt = int((gt_norm_unclamped[:, d].abs() > 1.5).sum())
        sgn = float(gap_norm_total[:, d].abs().sum())
        sgm = float(gap_model[:, d].abs().sum())
        stg = float(total_gap[:, d].abs().sum())
        rows.append({
            "dim": d, "label": DIM_LABELS[d], "norm_masked": masked,
            "gt_raw_p2p": p2p, "n_chunk_positions_hard_clamped": sat_cnt,
            "sum_abs_gap_norm": sgn, "sum_abs_gap_model": sgm, "sum_abs_total_gap": stg,
        })
        print(
            f"{d:>3}  {DIM_LABELS[d]:<14}  {str(masked):>4}  "
            f"{p2p:>10.4f}  {sat_cnt:>3} / {H}  "
            f"{sgn:>13.4f}  {sgm:>13.4f}  {stg:>13.4f}"
        )

    # ---- Per-(k, dim) tables for the 3 dims with the worst gap_norm ----
    masked_rows = [r for r in rows if r["norm_masked"]]
    worst_dims = sorted(masked_rows, key=lambda r: r["sum_abs_gap_norm"], reverse=True)[:3]
    csv_paths = []
    for r in worst_dims:
        d = r["dim"]
        path = OUT_DIR / f"05_replay_ep{args.episode:03d}_t{args.t:04d}_dim{d:02d}_{DIM_LABELS[d]}.csv"
        with path.open("w") as fh:
            fh.write("k,gt_raw,gt_norm_unclamped,gt_norm_clamped,gt_round_trip,pred_norm,pred_unnorm,gap_norm,gap_model,total_gap\n")
            for k in range(H):
                fh.write(
                    f"{k},{float(gt_raw[k, d]):+.6f},{float(gt_norm_unclamped[k, d]):+.4f},"
                    f"{float(gt_norm_clamped[k, d]):+.4f},{float(gt_round_trip[k, d]):+.6f},"
                    f"{float(pred_norm[k, d]):+.4f},{float(pred_unnorm[k, d]):+.6f},"
                    f"{float(gap_norm_total[k, d]):+.6f},{float(gap_model[k, d]):+.6f},"
                    f"{float(total_gap[k, d]):+.6f}\n"
                )
        csv_paths.append(str(path.relative_to(Path(__file__).resolve().parents[2])))
        print(f"\n  wrote {path.name} (dim {d}, label={DIM_LABELS[d]})")

    summary = {
        "episode": args.episode, "t_in_episode": args.t,
        "global_index": int(global_idx),
        "task": batch["task"],
        "horizon": H,
        "per_dim": rows,
        "worst_dim_csvs": csv_paths,
    }
    json_path = OUT_DIR / f"05_replay_ep{args.episode:03d}_t{args.t:04d}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path.name}")


if __name__ == "__main__":
    main()
