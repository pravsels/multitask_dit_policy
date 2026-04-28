"""Quantify the model-independent normalization gap.

Phase 2.2a of the Ramen chunk-delta investigation.

For every valid anchor frame t and chunk position k in the coffee-capsules dataset:
    gt_raw[t, k]        = act[t + k] - obs[t]
    gt_norm[t, k]       = ramen_normalize(gt_raw[t, k], q02, q98)  # clipped
    gt_round_trip[t, k] = ramen_unnormalize(gt_norm[t, k], q02, q98)
    gap_norm[t, k]      = gt_raw[t, k] - gt_round_trip[t, k]

`gap_norm` is the physical motion the buggy `(1, D)` normalization
*throws away* — i.e. the floor a model trained on this scheme can never
recover, regardless of how well it fits the (clipped) target.

Outputs:
  - Per-dim summary: mean |gap_norm|, max |gap_norm|, p98 |gap_norm|,
    fraction of (t, k) pairs hitting the soft (|y|>1.0) and hard
    (|y|>1.5) saturation bounds.
  - For the 3 worst (episode, t) windows on each of joint_1 and
    eef_xyz_z (representative joint and eef dims), dump CSVs over all
    32 chunk positions: k, gt_raw, gt_norm, gt_round_trip, gap_norm.
  - JSON summary at out/04_gap_norm.json.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from multitask_dit_policy.utils.ramen_normalization import (
    load_ramen_stats,
    ramen_normalize,
    ramen_unnormalize,
)
from multitask_dit_policy.utils.rotation import convert_eef_pose


DATA_GLOB = "data/lerobot/villekuosmanen/bin_pick_pack_coffee_capsules/data/chunk-*/file-*.parquet"

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "out"
DEFAULT_STATS_PATH = OUT_DIR / "ramen_stats_coffee_capsules_H32_obs2.json"
ROT6D_SLICE = (10, 16)
DIM_LABELS = (
    [f"joint_{i}" for i in range(6)]
    + ["joint_gripper"]
    + ["eef_xyz_x", "eef_xyz_y", "eef_xyz_z"]
    + [f"rot6d_{i}" for i in range(6)]
    + ["eef_gripper"]
)
TRACE_DIMS = [1, 9]  # joint_1 (worst joint), eef_xyz_z (representative eef)
N_TRACES_PER_DIM = 3


def parquet_files() -> list[Path]:
    files = list(ROOT.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {ROOT / DATA_GLOB}")

    def sort_key(p: Path) -> tuple[int, int]:
        chunk = int(re.search(r"chunk-(\d+)", str(p)).group(1))
        idx = int(re.search(r"file-(\d+)\.parquet$", p.name).group(1))
        return (chunk, idx)

    return sorted(files, key=sort_key)


def load_assembled() -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Returns (episode_index, all_obs (N,17), all_act (N,17))."""
    files = parquet_files()
    keys = [
        "observation.state.pos", "observation.state.eef_pose",
        "action.pos", "action.eef_pose",
        "episode_index",
    ]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for f in files:
        t = pq.read_table(f, columns=keys)
        for k in keys:
            col = t.column(k).to_numpy(zero_copy_only=False)
            if col.dtype == object:
                col = np.stack([np.asarray(x, dtype=np.float32) for x in col])
            parts[k].append(col)
    cols = {k: np.concatenate(v, axis=0) for k, v in parts.items()}

    def assemble(pos_key: str, eef_key: str) -> torch.Tensor:
        pos = torch.from_numpy(cols[pos_key]).float()
        eef = convert_eef_pose(torch.from_numpy(cols[eef_key]).float())
        return torch.cat([pos, eef], dim=-1)

    return cols["episode_index"], assemble("observation.state.pos", "observation.state.eef_pose"), assemble("action.pos", "action.eef_pose")


def build_chunk_targets(
    ep_idx: np.ndarray,
    all_obs: torch.Tensor,
    all_act: torch.Tensor,
    *,
    horizon: int,
    n_obs_steps: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Return (targets (M, H, D), episode_id_per_t (M,), local_t_per_t (M,)).

    targets[i, k, d] = act[t + k, d] - obs[t, d]
    for delta dims, raw act for rot6d dims.
    Iterates per-episode so chunks don't cross episode boundaries.
    """
    delta_indices = list(range(0, horizon))
    k_min = delta_indices[0]
    k_max = delta_indices[-1]
    offsets = torch.tensor(delta_indices, dtype=torch.long)
    starts = np.concatenate(([0], np.where(np.diff(ep_idx) != 0)[0] + 1, [len(ep_idx)]))
    targets_list: list[torch.Tensor] = []
    ep_ids: list[np.ndarray] = []
    local_ts: list[np.ndarray] = []
    for s, e in zip(starts[:-1], starts[1:]):
        L = e - s
        t_lo = max(0, -k_min)
        t_hi = L - k_max
        if t_hi <= t_lo:
            continue
        valid = t_hi - t_lo
        local_anchors = torch.arange(t_lo, t_hi, dtype=torch.long)
        obs_t = all_obs[s + local_anchors]  # (valid, D)
        rows = local_anchors[:, None] + offsets[None, :]
        act_chunks = all_act[s + rows]  # (valid, H, D)
        tgt = act_chunks - obs_t.unsqueeze(1)
        # rot6d dims pass through (no delta)
        tgt[..., ROT6D_SLICE[0]:ROT6D_SLICE[1]] = act_chunks[..., ROT6D_SLICE[0]:ROT6D_SLICE[1]]
        targets_list.append(tgt)
        ep_ids.append(np.full(valid, ep_idx[s], dtype=np.int64))
        local_ts.append(local_anchors.numpy())
    targets = torch.cat(targets_list, dim=0)
    return targets, np.concatenate(ep_ids), np.concatenate(local_ts)

def sanitize_tag(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_tag = sanitize_tag(args.stats_path)
    trace_dir = OUT_DIR / f"04_traces_{stats_tag}"
    out_json = OUT_DIR / f"04_gap_norm_{stats_tag}.json"
    trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading stats from {args.stats_path} ...")
    if args.stats_path.exists():
        stats = load_ramen_stats(args.stats_path)
    else:
        if args.stats_path == DEFAULT_STATS_PATH:
            raise FileNotFoundError(f"Expected recomputed stats at {args.stats_path}")
        raise FileNotFoundError(args.stats_path)
    q02 = stats["action_q02"]
    q98 = stats["action_q98"]
    norm_mask = stats["norm_mask"]
    horizon = int(stats.get("horizon", torch.tensor(args.horizon)).item()) if hasattr(stats.get("horizon", None), "item") else args.horizon
    n_obs_steps = int(stats.get("n_obs_steps", torch.tensor(args.n_obs_steps)).item()) if hasattr(stats.get("n_obs_steps", None), "item") else args.n_obs_steps
    print(
        f"  action_q02 shape={tuple(q02.shape)}, norm_mask sum={int(norm_mask.sum())} / {len(norm_mask)}, "
        f"horizon={horizon}, n_obs_steps={n_obs_steps}"
    )

    print("\nLoading dataset (parquet only) ...")
    ep_idx, all_obs, all_act = load_assembled()
    print(f"  frames={len(all_obs)}, dim={all_obs.shape[-1]}")

    print(f"\nBuilding chunk targets (H={horizon}, n_obs_steps={n_obs_steps}) ...")
    targets, ep_per_t, local_t = build_chunk_targets(
        ep_idx,
        all_obs,
        all_act,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )
    M, _, D = targets.shape
    print(f"  targets shape: ({M}, {horizon}, {D})")

    # --- Compute the gap on the masked (delta) dims only -----------------
    # ramen_normalize/unnormalize already broadcast over leading dims, and
    # leave non-masked dims (rot6d) untouched.
    print("\nApplying normalize -> unnormalize round trip ...")
    gt_norm_clamped = ramen_normalize(targets, q02, q98, norm_mask)
    gt_round_trip = ramen_unnormalize(gt_norm_clamped, q02, q98, norm_mask)
    gap = (targets - gt_round_trip).abs()  # (M, H, D)

    # Unclamped normalized values, used to count saturation rates per dim.
    denom = (q98 - q02).clamp_min(1e-8)
    gt_norm_unclamped = (2 * (targets - q02) / denom) - 1
    gap_flat = gap.reshape(M * horizon, D)
    gt_norm_unclamped_flat = gt_norm_unclamped.reshape(M * horizon, D)
    saturate_soft = (gt_norm_unclamped_flat.abs() > 1.0).float().mean(dim=0)
    saturate_hard = (gt_norm_unclamped_flat.abs() > 1.5).float().mean(dim=0)
    saturate_extreme = (gt_norm_unclamped_flat.abs() > 3.0).float().mean(dim=0)

    print("\n--- Per-dim gap_norm summary (over all (t, k) chunk targets) ---")
    print(
        f"{'dim':>3}  {'label':<14}  {'mask':>4}  "
        f"{'mean |gap|':>11}  {'p98 |gap|':>11}  {'max |gap|':>10}  "
        f"{'frac>1.0':>9}  {'frac>1.5':>9}  {'frac>3.0':>9}"
    )
    rows = []
    for d in range(D):
        masked = bool(norm_mask[d].item())
        mean_g = float(gap_flat[:, d].mean())
        p98_g = float(np.percentile(gap_flat[:, d].numpy(), 98))
        max_g = float(gap_flat[:, d].max())
        s_soft = float(saturate_soft[d])
        s_hard = float(saturate_hard[d])
        s_extreme = float(saturate_extreme[d])
        rows.append({
            "dim": d, "label": DIM_LABELS[d], "norm_masked": masked,
            "mean_abs_gap": mean_g, "p98_abs_gap": p98_g, "max_abs_gap": max_g,
            "frac_norm_above_1.0": s_soft, "frac_norm_above_1.5": s_hard,
            "frac_norm_above_3.0": s_extreme,
        })
        print(
            f"{d:>3}  {DIM_LABELS[d]:<14}  {str(masked):>4}  "
            f"{mean_g:>11.4f}  {p98_g:>11.4f}  {max_g:>10.4f}  "
            f"{s_soft:>9.2%}  {s_hard:>9.2%}  {s_extreme:>9.2%}"
        )

    # --- Per-dim: pick worst (t, k) windows and dump traces --------------
    targets_3d = targets
    gap_3d = gap
    gt_round_trip_3d = gt_round_trip
    gt_norm_unclamped_3d = gt_norm_unclamped

    print(f"\nDumping CSV traces for the {N_TRACES_PER_DIM} worst (episode, t) windows on dims {TRACE_DIMS} ...")
    trace_index = []
    for d in TRACE_DIMS:
        per_t_max = gap_3d[:, :, d].max(dim=1).values.numpy()  # (M,)
        worst_idx = np.argsort(-per_t_max)[:N_TRACES_PER_DIM]
        for rank, i in enumerate(worst_idx):
            ep_id = int(ep_per_t[i])
            t = int(local_t[i])
            csv_path = trace_dir / f"dim{d:02d}_{DIM_LABELS[d]}_ep{ep_id:03d}_t{t:04d}.csv"
            with csv_path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow([
                    "k", "offset", "gt_raw", "gt_norm_unclamped", "gt_round_trip",
                    "gap_norm", "abs_gap_norm", "clamped",
                ])
                for k in range(horizon):
                    offset = k
                    raw = float(targets_3d[i, k, d])
                    nu = float(gt_norm_unclamped_3d[i, k, d])
                    rt = float(gt_round_trip_3d[i, k, d])
                    gp = raw - rt
                    clamped = abs(nu) > 1.5
                    w.writerow([k, offset, f"{raw:+.6f}", f"{nu:+.4f}", f"{rt:+.6f}", f"{gp:+.6f}", f"{abs(gp):.6f}", int(clamped)])
            trace_index.append({
                "dim": d, "label": DIM_LABELS[d], "rank": int(rank), "episode_id": ep_id,
                "t_in_episode": t, "max_abs_gap_in_chunk": float(per_t_max[i]),
                "csv": str(csv_path.relative_to(ROOT)),
            })
            print(f"  dim {d} ({DIM_LABELS[d]}) rank {rank}: ep={ep_id:>3} t={t:>4}  max|gap|={per_t_max[i]:.4f}  -> {csv_path.relative_to(ROOT)}")

    out = {
        "stats_path": str(args.stats_path.relative_to(ROOT) if args.stats_path.is_absolute() and args.stats_path.is_relative_to(ROOT) else args.stats_path),
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_chunk_targets": int(M),
        "per_dim_gap_summary": rows,
        "trace_index": trace_index,
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_json.relative_to(ROOT)}")

    # --- Headline ---
    masked_dims = [r for r in rows if r["norm_masked"]]
    worst_mean = max(masked_dims, key=lambda r: r["mean_abs_gap"])
    worst_max = max(masked_dims, key=lambda r: r["max_abs_gap"])
    print("\n--- Headline ---")
    print(
        f"Worst dim by mean |gap_norm|: {worst_mean['label']:<14}  "
        f"mean={worst_mean['mean_abs_gap']:.4f}  hard-clamped frac={worst_mean['frac_norm_above_1.5']:.2%}"
    )
    print(
        f"Worst dim by max  |gap_norm|: {worst_max['label']:<14}  "
        f"max ={worst_max['max_abs_gap']:.4f}  hard-clamped frac={worst_max['frac_norm_above_1.5']:.2%}"
    )


if __name__ == "__main__":
    main()
