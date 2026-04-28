"""Validate recomputed chunk-aware ramen stats on coffee_capsules.

Phase 4.2 of the ramen chunk-delta investigation.
"""

from __future__ import annotations

import json
from pathlib import Path

import warnings

import torch
from huggingface_hub import hf_hub_download
from robocandywrapper import make_dataset_without_config
from robocandywrapper.plugins import ControlModePlugin

# Match train.py/configuration.py warning suppression before importing draccus.
warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402

from multitask_dit_policy.train import TrainConfig
from multitask_dit_policy.utils.dataset_adapter import adapt_batch
from multitask_dit_policy.utils.ramen_normalization import (
    build_norm_mask,
    load_ramen_stats,
    ramen_normalize,
    ramen_unnormalize,
)


REPO = "pravsels/dit_coffee_capsules_config_fix"
CHECKPOINT_DIR = "checkpoint_30000"
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "train_coffee_capsules.yaml"
OUT_DIR = ROOT / "scripts" / "ramen_chunk_delta" / "out"
STATS_PATH = OUT_DIR / "ramen_stats_coffee_capsules_H32_obs2.json"
OUT_JSON = OUT_DIR / "06_validate_stats.json"
ROUND_TRIP_SAMPLES = [0, 100, 1000]
MOTION_DIMS = [0, 1, 2, 3, 4, 5, 6, 16]
ARM_DIMS = [0, 1, 2, 3, 4, 5]
GRIP_DIMS = [6, 16]


def load_train_config(path: Path) -> TrainConfig:
    with path.open() as f, draccus.config_type("yaml"):
        return draccus.load(TrainConfig, f)


def load_checkpoint_stats() -> dict[str, torch.Tensor]:
    stats_path = hf_hub_download(REPO, f"{CHECKPOINT_DIR}/ramen_stats.pt")
    return load_ramen_stats(stats_path)


def build_dataset(cfg: TrainConfig):
    dataset_path = Path(cfg.dataset_path)
    if dataset_path.is_absolute() and dataset_path.is_dir():
        repo_id = dataset_path.name
        root = str(dataset_path)
    else:
        repo_id = cfg.dataset_path
        root = None

    return make_dataset_without_config(
        repo_id=repo_id,
        action_delta_indices=list(cfg.policy.action_delta_indices),
        observation_delta_indices=list(cfg.policy.observation_delta_indices),
        root=root,
        video_backend="pyav",
        use_imagenet_stats=cfg.policy.observation_encoder.use_imagenet_stats,
        plugins=[ControlModePlugin()],
    )


def pass_fail(name: str, ok: bool, detail: str) -> dict[str, object]:
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {name} - {detail}")
    return {"name": name, "ok": ok, "detail": detail}


def build_batch_from_sample(sample: dict, device: str = "cpu") -> dict:
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, str):
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def main() -> None:
    cfg = load_train_config(CONFIG_PATH)
    schema = cfg.dataset_schema
    rot6d_start, rot6d_end = schema.rot6d_slice
    norm_mask = build_norm_mask(max(schema.state_dim, schema.action_dim), rot6d_start, rot6d_end)
    stats = load_ramen_stats(STATS_PATH)
    old_stats = load_checkpoint_stats()
    dataset = build_dataset(cfg)

    results: list[dict[str, object]] = []

    results.append(
        pass_fail(
            "shapes",
            tuple(stats["action_q02"].shape) == (32, 17)
            and tuple(stats["action_q98"].shape) == (32, 17)
            and tuple(stats["obs_q02"].shape) == (1, 17)
            and tuple(stats["obs_q98"].shape) == (1, 17),
            (
                f"action_q02={tuple(stats['action_q02'].shape)}, "
                f"action_q98={tuple(stats['action_q98'].shape)}, "
                f"obs_q02={tuple(stats['obs_q02'].shape)}, "
                f"obs_q98={tuple(stats['obs_q98'].shape)}"
            ),
        )
    )

    new_span = stats["action_q98"][0] - stats["action_q02"][0]
    old_span = old_stats["action_q98"][0] - old_stats["action_q02"][0]
    rel_err = ((new_span - old_span).abs() / old_span.clamp_min(1e-8))
    masked_rel_err = rel_err[norm_mask.bool()]
    results.append(
        pass_fail(
            "single-frame-span-match",
            masked_rel_err.max().item() <= 0.25,
            (
                "masked max relative error at position 0 vs old (1,D) span = "
                f"{masked_rel_err.max().item():.4f}, "
                f"masked mean = {masked_rel_err.mean().item():.4f}"
            ),
        )
    )

    per_dim_ratios = {}
    arm_ok = True
    grip_ok = True
    for d in MOTION_DIMS:
        late = float(stats["action_q98"][31, d] - stats["action_q02"][31, d])
        early = float(stats["action_q98"][0, d] - stats["action_q02"][0, d])
        ratio = late / max(early, 1e-8)
        per_dim_ratios[d] = ratio
        if d in ARM_DIMS:
            arm_ok = arm_ok and ratio >= 5.0
        else:
            grip_ok = grip_ok and ratio >= 1.5
    results.append(
        pass_fail(
            "late-vs-early-span-growth",
            arm_ok and grip_ok,
            (
                "arm dims expect >=5x, grip dims expect >=1.5x; "
                + ", ".join(f"d{d}={ratio:.2f}x" for d, ratio in per_dim_ratios.items())
            ),
        )
    )

    grip_range = float(stats["action_q98"][31, 6] - stats["action_q02"][31, 6])
    results.append(
        pass_fail(
            "grip-range-physical-scale",
            0.5 * 0.0780 <= grip_range <= 2.0 * 0.0780,
            f"position-31 grip span={grip_range:.4f}, Phase-1 physical range~=0.0780",
        )
    )

    finite_ok = all(torch.isfinite(stats[key]).all().item() for key in ("obs_q02", "obs_q98", "action_q02", "action_q98"))
    results.append(pass_fail("finite-values", finite_ok, "all four stats tensors finite"))

    rot_q02 = stats["action_q02"][:, 10:16]
    rot_q98 = stats["action_q98"][:, 10:16]
    rot_ok = torch.isfinite(rot_q02).all().item() and torch.isfinite(rot_q98).all().item()
    results.append(
        pass_fail(
            "rot6d-stats-finite",
            rot_ok,
            f"rot6d q02 range=({float(rot_q02.min()):+.4f}, {float(rot_q02.max()):+.4f}), "
            f"q98 range=({float(rot_q98.min()):+.4f}, {float(rot_q98.max()):+.4f})",
        )
    )

    round_trip_details = []
    round_trip_ok = True
    for idx in ROUND_TRIP_SAMPLES:
        sample = dataset[idx]
        batch = build_batch_from_sample(sample)
        adapted = adapt_batch(dict(batch), schema, norm_mask)
        action = adapted["action"]
        action_norm = ramen_normalize(action, stats["action_q02"], stats["action_q98"], norm_mask)
        action_rt = ramen_unnormalize(action_norm, stats["action_q02"], stats["action_q98"], norm_mask)
        max_err = float((action_rt - action).abs().max())
        round_trip_details.append({"dataset_index": idx, "max_abs_err": max_err})
        round_trip_ok = round_trip_ok and max_err <= 1e-5
    results.append(
        pass_fail(
            "dataset-round-trip",
            round_trip_ok,
            ", ".join(f"idx {row['dataset_index']} max_err={row['max_abs_err']:.2e}" for row in round_trip_details),
        )
    )

    summary = {
        "stats_path": str(STATS_PATH.relative_to(ROOT)),
        "results": results,
        "late_vs_early_ratios": per_dim_ratios,
        "round_trip_details": round_trip_details,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")

    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
