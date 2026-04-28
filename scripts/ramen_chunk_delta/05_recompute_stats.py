"""Recompute chunk-aware Ramen stats on coffee_capsules.

Phase 4.1 of the ramen chunk-delta investigation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import warnings
from huggingface_hub import hf_hub_download
from robocandywrapper import make_dataset_without_config
from robocandywrapper.plugins import ControlModePlugin

# Match train.py/configuration.py warning suppression before importing draccus.
warnings.filterwarnings("ignore", message=".*Field.*attribute.*repr.*")
warnings.filterwarnings("ignore", message=".*Field.*attribute.*frozen.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import draccus  # noqa: E402

from multitask_dit_policy.train import TrainConfig
from multitask_dit_policy.utils.ramen_normalization import (
    build_norm_mask,
    compute_ramen_stats,
    load_ramen_stats,
)


REPO = "pravsels/dit_coffee_capsules_config_fix"
CHECKPOINT_DIR = "checkpoint_30000"
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "train_coffee_capsules.yaml"
OUT_DIR = ROOT / "scripts" / "ramen_chunk_delta" / "out"
OUT_STATS = OUT_DIR / "ramen_stats_coffee_capsules_H32_obs2.json"
OUT_MD = OUT_DIR / "05_recompute.md"
OUT_JSON = OUT_DIR / "05_recompute.json"

DIMS = [0, 1, 2, 6, 16]
POSITIONS = [0, 1, 4, 8, 16, 24, 31]
DIM_LABELS = {
    0: "joint_0",
    1: "joint_1",
    2: "joint_2",
    6: "joint_gripper",
    16: "eef_gripper",
}


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


def render_markdown(
    *,
    cfg: TrainConfig,
    stats: dict[str, torch.Tensor],
    old_stats: dict[str, torch.Tensor],
) -> str:
    lines: list[str] = []
    lines.append("# Phase 4.1 Recomputed (H, D) Ramen Stats")
    lines.append("")
    lines.append("## Shapes")
    lines.append("")
    for key in ("obs_q02", "obs_q98", "action_q02", "action_q98"):
        lines.append(f"- `{key}`: `{tuple(stats[key].shape)}`")
    lines.append("")
    lines.append("## Selected Dims")
    lines.append("")
    lines.append(
        "| dim | label | pos | offset | new q02 | new q98 | old q02 | old q98 | "
        "new span / old span |"
    )
    lines.append("|---:|:------|---:|---:|---:|---:|---:|---:|---:|")
    for dim in DIMS:
        old_q02 = float(old_stats["action_q02"][0, dim])
        old_q98 = float(old_stats["action_q98"][0, dim])
        old_span = old_q98 - old_q02
        for pos in POSITIONS:
            offset = pos
            new_q02 = float(stats["action_q02"][pos, dim])
            new_q98 = float(stats["action_q98"][pos, dim])
            new_span = new_q98 - new_q02
            ratio = new_span / max(old_span, 1e-8)
            lines.append(
                f"| {dim} | {DIM_LABELS[dim]} | {pos} | {offset:+d} | "
                f"{new_q02:+.4f} | {new_q98:+.4f} | {old_q02:+.4f} | {old_q98:+.4f} | "
                f"{ratio:.2f}x |"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `i=0` is now offset 0, so it should be the closest row to the old `(1, D)` stats."
    )
    lines.append(
        "- Later positions should show larger action spans on motion dims if the chunk-aware stats are doing the right thing."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_train_config(CONFIG_PATH)
    schema = cfg.dataset_schema
    rot6d_start, rot6d_end = schema.rot6d_slice
    norm_mask = build_norm_mask(max(schema.state_dim, schema.action_dim), rot6d_start, rot6d_end)

    print(f"Loading dataset via training config: {cfg.dataset_path}")
    dataset = build_dataset(cfg)
    print(f"  dataset length: {len(dataset)}")

    print("Recomputing chunk-aware ramen stats ...")
    stats = compute_ramen_stats(
        dataset=dataset,
        schema=schema,
        norm_mask=norm_mask,
        cache_path=OUT_STATS,
        horizon=cfg.policy.horizon,
        n_obs_steps=cfg.policy.n_obs_steps,
        drop_n_last_frames=cfg.policy.drop_n_last_frames,
        device="cpu",
    )
    for key in ("obs_q02", "obs_q98", "action_q02", "action_q98"):
        print(f"  {key}: shape={tuple(stats[key].shape)}")

    old_stats = load_checkpoint_stats()
    md = render_markdown(cfg=cfg, stats=stats, old_stats=old_stats)
    OUT_MD.write_text(md)

    summary = {
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "dataset_path": cfg.dataset_path,
        "horizon": cfg.policy.horizon,
        "n_obs_steps": cfg.policy.n_obs_steps,
        "drop_n_last_frames": cfg.policy.drop_n_last_frames,
        "stats_path": str(OUT_STATS.relative_to(ROOT)),
        "stats_shapes": {k: list(stats[k].shape) for k in ("obs_q02", "obs_q98", "action_q02", "action_q98")},
        "selected_dims": [],
    }
    for dim in DIMS:
        old_q02 = float(old_stats["action_q02"][0, dim])
        old_q98 = float(old_stats["action_q98"][0, dim])
        selected = []
        for pos in POSITIONS:
            selected.append(
                {
                    "position": pos,
                    "offset": pos,
                    "new_q02": float(stats["action_q02"][pos, dim]),
                    "new_q98": float(stats["action_q98"][pos, dim]),
                    "old_q02": old_q02,
                    "old_q98": old_q98,
                }
            )
        summary["selected_dims"].append({"dim": dim, "label": DIM_LABELS[dim], "rows": selected})
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {OUT_STATS.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print("")
    print(md)


if __name__ == "__main__":
    main()
