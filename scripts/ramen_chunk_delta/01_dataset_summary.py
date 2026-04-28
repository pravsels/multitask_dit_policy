"""Per-dim summary of the coffee-capsules dataset (parquet-only, no video).

Phase 1 of the Ramen chunk-delta investigation.

Reads parquet directly with pyarrow from the local HF lerobot cache:
  data/lerobot/villekuosmanen/bin_pick_pack_coffee_capsules/data/chunk-*/file-*.parquet

No LeRobotDataset, no torchcodec, no video decode. Then applies the same
RPY->rot6d conversion the training pipeline uses to assemble 17D state and
action vectors.

Prints per-dim min/max/mean/std/p02/p98 for state, action, and the
single-frame delta (act - obs). Saves the same data to
scripts/ramen_chunk_delta/out/01_summary.json.
"""

import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from multitask_dit_policy.utils.rotation import convert_eef_pose


REPO_ID = "villekuosmanen/bin_pick_pack_coffee_capsules"
DATA_GLOB = "data/lerobot/villekuosmanen/bin_pick_pack_coffee_capsules/data/chunk-*/file-*.parquet"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "out"
OUT_JSON = OUT_DIR / "01_summary.json"

H = 32  # action chunk horizon (matches train_coffee_capsules*.yaml)

DIM_LABELS = (
    [f"joint_{i}" for i in range(6)]
    + ["joint_gripper"]
    + ["eef_xyz_x", "eef_xyz_y", "eef_xyz_z"]
    + [f"rot6d_{i}" for i in range(6)]
    + ["eef_gripper"]
)
HIGHLIGHT_DIMS = {6, 16}

ROT6D_SLICE = (10, 16)
RAW_KEYS = [
    "observation.state.pos",
    "observation.state.eef_pose",
    "action.pos",
    "action.eef_pose",
    "episode_index",
]


def parquet_files() -> list[Path]:
    files = list(ROOT.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {ROOT / DATA_GLOB}")

    def sort_key(p: Path) -> tuple[int, int]:
        chunk = int(re.search(r"chunk-(\d+)", str(p)).group(1))
        idx = int(re.search(r"file-(\d+)\.parquet$", p.name).group(1))
        return (chunk, idx)

    return sorted(files, key=sort_key)


def read_columns(files: list[Path], keys: list[str]) -> dict[str, np.ndarray]:
    """Read the requested columns across all parquet files, concatenated."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for f in files:
        table = pq.read_table(f, columns=keys)
        for k in keys:
            col = table.column(k).to_numpy(zero_copy_only=False)
            if col.dtype == object:
                col = np.stack([np.asarray(x, dtype=np.float32) for x in col])
            parts[k].append(col)
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def assemble(cols: dict[str, np.ndarray], state_keys: list[tuple[str, bool]]) -> torch.Tensor:
    """Assemble (N, D) tensor; (key, convert_rotation) pairs in order."""
    parts: list[torch.Tensor] = []
    for key, convert in state_keys:
        t = torch.from_numpy(cols[key]).float()
        if convert:
            t = convert_eef_pose(t)
        parts.append(t)
    return torch.cat(parts, dim=-1)


def per_dim_stats(x: torch.Tensor) -> list[dict]:
    x = x.float()
    mn = x.min(dim=0).values
    mx = x.max(dim=0).values
    mu = x.mean(dim=0)
    sd = x.std(dim=0)
    p02 = torch.quantile(x, 0.02, dim=0)
    p98 = torch.quantile(x, 0.98, dim=0)
    rows = []
    for d in range(x.shape[-1]):
        rows.append({
            "dim": d,
            "label": DIM_LABELS[d],
            "min": float(mn[d]),
            "max": float(mx[d]),
            "mean": float(mu[d]),
            "std": float(sd[d]),
            "p02": float(p02[d]),
            "p98": float(p98[d]),
        })
    return rows


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'dim':>3}  {'label':<14}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}  {'p02':>10}  {'p98':>10}")
    print("-" * 96)
    for r in rows:
        prefix = ">>> " if r["dim"] in HIGHLIGHT_DIMS else "    "
        print(
            f"{prefix}{r['dim']:>1}  {r['label']:<14}  "
            f"{r['min']:>10.4f}  {r['max']:>10.4f}  {r['mean']:>10.4f}  "
            f"{r['std']:>10.4f}  {r['p02']:>10.4f}  {r['p98']:>10.4f}"
        )


def episode_length_stats(episode_index: np.ndarray) -> dict:
    _, counts = np.unique(episode_index, return_counts=True)
    L = torch.tensor(counts, dtype=torch.float32)
    return {
        "n_episodes": int(L.numel()),
        "min": int(L.min().item()),
        "p10": float(torch.quantile(L, 0.10).item()),
        "median": float(torch.quantile(L, 0.50).item()),
        "mean": float(L.mean().item()),
        "p90": float(torch.quantile(L, 0.90).item()),
        "max": int(L.max().item()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = parquet_files()
    print(f"Reading {len(files)} parquet files for {REPO_ID}")
    cols = read_columns(files, RAW_KEYS)
    print(f"Loaded columns: " + ", ".join(f"{k}{cols[k].shape}" for k in RAW_KEYS))

    all_obs = assemble(cols, [
        ("observation.state.pos", False),
        ("observation.state.eef_pose", True),
    ])  # (N, 17)
    all_act = assemble(cols, [
        ("action.pos", False),
        ("action.eef_pose", True),
    ])  # (N, 17)
    print(f"Assembled tensors: obs {tuple(all_obs.shape)}, act {tuple(all_act.shape)}")

    # Build delta only on non-rot6d dims, mirroring build_norm_mask.
    norm_mask = torch.ones(all_obs.shape[-1], dtype=torch.bool)
    norm_mask[ROT6D_SLICE[0]:ROT6D_SLICE[1]] = False
    shared = min(all_obs.shape[-1], all_act.shape[-1])
    m = norm_mask[:shared]
    all_delta = all_act.clone()
    all_delta[..., :shared][..., m] = (
        all_act[..., :shared][..., m] - all_obs[..., :shared][..., m]
    )

    ep_stats = episode_length_stats(cols["episode_index"])
    print(f"\nFrames: {all_obs.shape[0]}   Episodes: {ep_stats['n_episodes']}")
    print(
        f"Episode length — min={ep_stats['min']}, p10={ep_stats['p10']:.0f}, "
        f"median={ep_stats['median']:.0f}, mean={ep_stats['mean']:.1f}, "
        f"p90={ep_stats['p90']:.0f}, max={ep_stats['max']}"
    )

    obs_rows = per_dim_stats(all_obs)
    act_rows = per_dim_stats(all_act)
    delta_rows = per_dim_stats(all_delta)

    # Chunk-relative delta target[t, k] = act[t+k] - obs[t] for k in [0, H-1],
    # respecting episode boundaries. Per-dim p02/p98 over the *flattened* (t, k)
    # distribution gives the closest one-number proxy to what the buggy (1, D)
    # stats see vs what training actually demands.
    ep_idx = cols["episode_index"]
    starts = np.concatenate(([0], np.where(np.diff(ep_idx) != 0)[0] + 1, [len(ep_idx)]))
    chunk_targets: list[torch.Tensor] = []
    for s, e in zip(starts[:-1], starts[1:]):
        L = e - s
        if L < H:
            continue
        valid = L - H + 1
        obs_t = all_obs[s:s + valid]              # (valid, D)
        act_chunks = torch.stack([all_act[s + k:s + k + valid] for k in range(H)], dim=1)  # (valid, H, D)
        tgt = act_chunks - obs_t.unsqueeze(1)
        tgt[..., ROT6D_SLICE[0]:ROT6D_SLICE[1]] = act_chunks[..., ROT6D_SLICE[0]:ROT6D_SLICE[1]]
        chunk_targets.append(tgt.reshape(-1, all_obs.shape[-1]))
    chunk_targets_flat = torch.cat(chunk_targets, dim=0)
    chunk_rows = per_dim_stats(chunk_targets_flat)

    print_table("observation.state (17D)", obs_rows)
    print_table("action (17D, raw)", act_rows)
    print_table("single-frame delta (act[t] - obs[t])", delta_rows)
    print_table(f"CHUNK delta (act[t+k] - obs[t], H={H}, all k flattened)", chunk_rows)

    print(f"\nChunk-vs-single span ratio per dim (chunk_p98 - chunk_p02) / (single_p98 - single_p02):")
    print(f"{'dim':>3}  {'label':<14}  {'single span':>12}  {'chunk span':>12}  {'ratio':>8}")
    for d in range(all_obs.shape[-1]):
        ss = delta_rows[d]["p98"] - delta_rows[d]["p02"]
        cs = chunk_rows[d]["p98"] - chunk_rows[d]["p02"]
        if d in range(*ROT6D_SLICE):
            continue  # rot6d pass-through is identical
        ratio = cs / ss if ss > 1e-12 else float("inf")
        marker = " <<<" if ratio >= 2.0 else ""
        print(
            f"{d:>3}  {DIM_LABELS[d]:<14}  {ss:>12.4f}  {cs:>12.4f}  {ratio:>8.2f}x{marker}"
        )

    print("\nGripper highlights:")
    for d in sorted(HIGHLIGHT_DIMS):
        print(
            f"  dim {d} ({DIM_LABELS[d]}): "
            f"obs[{obs_rows[d]['min']:.4f}, {obs_rows[d]['max']:.4f}], "
            f"act[{act_rows[d]['min']:.4f}, {act_rows[d]['max']:.4f}], "
            f"delta p02={delta_rows[d]['p02']:.4f} p98={delta_rows[d]['p98']:.4f}"
        )

    out = {
        "repo_id": REPO_ID,
        "n_frames": int(all_obs.shape[0]),
        "n_episodes": ep_stats["n_episodes"],
        "episode_length_stats": ep_stats,
        "horizon": H,
        "obs_stats": obs_rows,
        "action_stats": act_rows,
        "single_frame_delta_stats": delta_rows,
        "chunk_delta_stats_flat": chunk_rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
