"""Fetch the coffee-capsules checkpoint's baked Ramen stats and inspect them.

Phase 2.1 of the Ramen chunk-delta investigation.

Goals:
  - Download `ramen_stats.pt` and `config.json` from
    `pravsels/dit_coffee_capsules_config_fix` (small files; we don't pull
    the model weights here).
  - Confirm `obs_q02 / obs_q98 / action_q02 / action_q98` have shape
    `(1, D)` — i.e. the buggy single-frame stats, not `(H, D)`.
  - Cross-check that the stored `action_q02 / action_q98` match the
    single-frame delta percentiles we computed from the parquet files in
    task 1.1 (sanity: same dataset, same recipe).
  - Print the model config's chunk horizon (so we know what H to use
    for the (H, D) fix later).

Outputs:
  - Prints a side-by-side per-dim table.
  - Saves a JSON summary at out/03_checkpoint_stats.json.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from multitask_dit_policy.utils.ramen_normalization import load_ramen_stats

REPO = "pravsels/dit_coffee_capsules_config_fix"
CHECKPOINT_DIR = "checkpoint_30000"
OUT_DIR = Path(__file__).parent / "out"
OUT_JSON = OUT_DIR / "03_checkpoint_stats.json"
DATASET_SUMMARY_JSON = OUT_DIR / "01_summary.json"

DIM_LABELS = (
    [f"joint_{i}" for i in range(6)]
    + ["joint_gripper"]
    + ["eef_xyz_x", "eef_xyz_y", "eef_xyz_z"]
    + [f"rot6d_{i}" for i in range(6)]
    + ["eef_gripper"]
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {REPO}/{CHECKPOINT_DIR} ...")
    try:
        stats_path = hf_hub_download(REPO, f"{CHECKPOINT_DIR}/ramen_stats.json")
    except EntryNotFoundError:
        stats_path = hf_hub_download(REPO, f"{CHECKPOINT_DIR}/ramen_stats.pt")
    config_path = hf_hub_download(REPO, f"{CHECKPOINT_DIR}/config.json")
    print(f"  {Path(stats_path).name} -> {stats_path}")
    print(f"  config.json    -> {config_path}")

    cfg = json.loads(Path(config_path).read_text())
    horizon = cfg.get("horizon") or cfg.get("chunk_size") or cfg.get("n_action_steps")
    n_action_steps = cfg.get("n_action_steps")
    print(f"\nConfig: horizon={horizon}, n_action_steps={n_action_steps}")
    print(f"  full keys: {sorted(cfg.keys())}")

    stats = load_ramen_stats(stats_path)
    print(f"\nLoaded {Path(stats_path).name} — keys: {sorted(stats.keys())}")
    for k, v in stats.items():
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else None
        dtype = v.dtype if isinstance(v, torch.Tensor) else type(v).__name__
        print(f"  {k:>14}  shape={shape}  dtype={dtype}")

    obs_q02 = stats["obs_q02"]
    obs_q98 = stats["obs_q98"]
    act_q02 = stats["action_q02"]
    act_q98 = stats["action_q98"]
    norm_mask = stats["norm_mask"]

    # Stats stored as (T, D); training wraps in a leading T=1 dim.
    assert act_q02.shape == act_q98.shape, (act_q02.shape, act_q98.shape)
    is_buggy_1d = act_q02.shape[0] == 1
    print(
        f"\nVERDICT: action stats shape = {tuple(act_q02.shape)} "
        f"-> {'(1, D) — BUGGY single-frame stats' if is_buggy_1d else '(H, D) — already chunk-aware'}"
    )

    # Cross-check vs task 1.1 single-frame percentiles
    print("\nPer-dim action stats (this checkpoint) vs single-frame delta stats (task 1.1, recomputed):")
    summary = json.loads(DATASET_SUMMARY_JSON.read_text())
    sf_rows = summary["single_frame_delta_stats"]
    print(
        f"{'dim':>3}  {'label':<14}  {'mask':>4}  "
        f"{'ckpt q02':>10}  {'recomp q02':>11}  {'Δ':>8}    "
        f"{'ckpt q98':>10}  {'recomp q98':>11}  {'Δ':>8}"
    )
    rows = []
    D = act_q02.shape[-1]
    for d in range(D):
        ck_lo = float(act_q02[0, d])
        ck_hi = float(act_q98[0, d])
        rc_lo = float(sf_rows[d]["p02"])
        rc_hi = float(sf_rows[d]["p98"])
        masked = bool(norm_mask[d].item())
        rows.append({
            "dim": d, "label": DIM_LABELS[d], "norm_masked": masked,
            "ckpt_action_q02": ck_lo, "ckpt_action_q98": ck_hi,
            "recomputed_single_frame_q02": rc_lo, "recomputed_single_frame_q98": rc_hi,
            "diff_q02": ck_lo - rc_lo, "diff_q98": ck_hi - rc_hi,
        })
        print(
            f"{d:>3}  {DIM_LABELS[d]:<14}  {str(masked):>4}  "
            f"{ck_lo:>+10.4f}  {rc_lo:>+11.4f}  {ck_lo-rc_lo:>+8.4f}    "
            f"{ck_hi:>+10.4f}  {rc_hi:>+11.4f}  {ck_hi-rc_hi:>+8.4f}"
        )

    # Sanity: max abs diff on the masked dims (where Ramen is actually applied)
    max_diff = max(
        max(abs(r["diff_q02"]), abs(r["diff_q98"])) for r in rows if r["norm_masked"]
    )
    print(f"\nMax |ckpt - recomputed| on masked dims: {max_diff:.6f}")
    if max_diff > 1e-3:
        print("  -> Mismatch larger than expected; double-check the checkpoint was trained on this dataset version.")
    else:
        print("  -> Match within tolerance; checkpoint stats are the buggy single-frame deltas computed from this dataset.")

    out = {
        "repo": REPO,
        "checkpoint_dir": CHECKPOINT_DIR,
        "config_horizon": horizon,
        "config_n_action_steps": n_action_steps,
        "stats_shapes": {k: tuple(v.shape) if isinstance(v, torch.Tensor) else None for k, v in stats.items()},
        "is_buggy_1d": is_buggy_1d,
        "per_dim_compare": rows,
        "max_diff_on_masked": max_diff,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
