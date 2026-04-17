"""Find real gripper-motion events and quantify chunk-relative deltas.

Phase 1.2 of the Ramen chunk-delta investigation.

The training pipeline computes the model's target at chunk step k as
    target[t, k] = action[t + k] - obs[t]   for k in [0, H-1]
(see `dataset_adapter.adapt_batch`). This is what gets normalized.

For each episode we:
  - Compute target[t, k] across all valid (t, k).
  - Find the (t*, k*) with the largest |target| on the gripper joint.
  - Also report the largest single-frame delta (action[t] - obs[t]),
    which is what the *current* (buggy) stats are computed from.

For the top 3 episodes we dump a 64-frame window centred on t*:
  t, obs_grip, act_grip, single_frame_delta=act[t]-obs[t],
  chunk_delta_at_k* = act[t+k*] - obs[t].

Summary lands in `out/02_summary.json`; CSVs under `out/02_traces/`.
"""

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REPO_ID = "villekuosmanen/bin_pick_pack_coffee_capsules"
ROOT = Path(__file__).resolve().parents[2]
DATA_GLOB = "data/lerobot/villekuosmanen/bin_pick_pack_coffee_capsules/data/chunk-*/file-*.parquet"
OUT_DIR = Path(__file__).parent / "out"
TRACE_DIR = OUT_DIR / "02_traces"
SUMMARY_JSON = OUT_DIR / "02_summary.json"

H = 32  # chunk horizon (matches train config)
GRIP_DIM_IN_POS = 6  # joint_6 inside observation.state.pos / action.pos

# From task 1.1 (single-frame delta on dim 6):
SINGLE_FRAME_Q02 = -0.0395
SINGLE_FRAME_Q98 = +0.0362
SATURATION_CEILING_NORMAL = SINGLE_FRAME_Q98 - SINGLE_FRAME_Q02  # at y in [-1, +1]
SATURATION_CEILING_CLAMP = 1.5 * (SINGLE_FRAME_Q98 - SINGLE_FRAME_Q02)  # at y in [-1.5, +1.5]


def load_gripper_signals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (episode_index, obs_grip, act_grip) concatenated across all parquet files."""
    files = list(ROOT.glob(DATA_GLOB))

    def sort_key(p: Path) -> tuple[int, int]:
        chunk = int(re.search(r"chunk-(\d+)", str(p)).group(1))
        idx = int(re.search(r"file-(\d+)\.parquet$", p.name).group(1))
        return (chunk, idx)

    files = sorted(files, key=sort_key)
    ep_parts, obs_parts, act_parts = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "observation.state.pos", "action.pos"])
        ep_parts.append(t.column("episode_index").to_numpy())
        obs_parts.append(np.stack([np.asarray(x) for x in t.column("observation.state.pos").to_numpy(zero_copy_only=False)])[:, GRIP_DIM_IN_POS])
        act_parts.append(np.stack([np.asarray(x) for x in t.column("action.pos").to_numpy(zero_copy_only=False)])[:, GRIP_DIM_IN_POS])
    return (
        np.concatenate(ep_parts),
        np.concatenate(obs_parts).astype(np.float64),
        np.concatenate(act_parts).astype(np.float64),
    )


def episode_chunk_motion(
    obs_grip_ep: np.ndarray, act_grip_ep: np.ndarray
) -> tuple[int, int, float, np.ndarray, np.ndarray]:
    """Find (t*, k*) maximising |action[t+k] - obs[t]| on the gripper.

    Builds target matrix M[t, k] = act[t+k] - obs[t] for t in [0, n-H],
    k in [0, H-1]. Returns (t_peak, k_peak, signed_value, per_t_max_abs, M).
    per_t_max_abs[t] = max over k of |M[t, k]|.
    """
    n = len(act_grip_ep)
    if n < H:
        return -1, -1, 0.0, np.array([]), np.empty((0, H))
    valid = n - H + 1
    M = np.empty((valid, H), dtype=np.float64)
    for k in range(H):
        M[:, k] = act_grip_ep[k:k + valid] - obs_grip_ep[:valid]
    flat = int(np.argmax(np.abs(M)))
    t_peak, k_peak = divmod(flat, H)
    per_t = np.max(np.abs(M), axis=1)
    return int(t_peak), int(k_peak), float(M[t_peak, k_peak]), per_t, M


def write_trace_csv(
    path: Path,
    ep_id: int,
    t_peak: int,
    k_peak: int,
    obs_grip_ep: np.ndarray,
    act_grip_ep: np.ndarray,
) -> None:
    """Dump a 64-frame window centred on t_peak.

    For each t in the window:
      - obs_grip[t], act_grip[t]
      - single_frame_delta = act_grip[t] - obs_grip[t]
      - chunk_delta_at_kpeak = act_grip[t + k_peak] - obs_grip[t]
        (the dim the model would have to express at chunk position k_peak)
    """
    n = len(obs_grip_ep)
    window_lo = max(0, t_peak - 32)
    window_hi = min(n, t_peak + 32 + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "episode_id", "t", "obs_grip", "act_grip",
            "single_frame_delta", f"chunk_delta_at_k{k_peak}",
        ])
        for t in range(window_lo, window_hi):
            single = act_grip_ep[t] - obs_grip_ep[t]
            tk = t + k_peak
            chunk = act_grip_ep[tk] - obs_grip_ep[t] if tk < n else float("nan")
            w.writerow([
                ep_id, t,
                f"{obs_grip_ep[t]:.6f}", f"{act_grip_ep[t]:.6f}",
                f"{single:.6f}", f"{chunk:.6f}",
            ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading gripper signals from {REPO_ID} ...")
    ep_idx, obs_grip, act_grip = load_gripper_signals()
    ep_ids, ep_starts, ep_counts = np.unique(ep_idx, return_index=True, return_counts=True)
    print(f"  total frames: {len(act_grip)},  episodes: {len(ep_ids)}")

    per_ep = []
    for ep_id, s, c in zip(ep_ids.tolist(), ep_starts.tolist(), ep_counts.tolist()):
        obs_ep = obs_grip[s:s + c]
        act_ep = act_grip[s:s + c]
        t_peak, k_peak, signed_peak, _per_t, _M = episode_chunk_motion(obs_ep, act_ep)
        if t_peak < 0:
            continue
        single_peak = float(np.max(np.abs(act_ep - obs_ep)))
        per_ep.append({
            "episode_id": int(ep_id),
            "ep_start": int(s),
            "ep_len": int(c),
            "t_peak": int(t_peak),
            "k_peak": int(k_peak),
            "chunk_delta_peak_signed": float(signed_peak),
            "chunk_delta_peak_abs": float(abs(signed_peak)),
            "single_frame_delta_peak_abs": single_peak,
        })

    per_ep.sort(key=lambda r: r["chunk_delta_peak_abs"], reverse=True)
    top = per_ep[:3]

    print(f"\nTop 3 episodes by |chunk gripper delta = act[t+k] - obs[t]| (out of {len(per_ep)}):")
    print(f"{'ep_id':>6}  {'len':>5}  {'t*':>5}  {'k*':>3}  {'chunk Δ':>10}  {'|chunk Δ|':>10}  {'|single Δ|':>11}  {'chunk/single':>13}")
    for r in top:
        ratio = r["chunk_delta_peak_abs"] / r["single_frame_delta_peak_abs"] if r["single_frame_delta_peak_abs"] > 0 else float("inf")
        print(
            f"{r['episode_id']:>6}  {r['ep_len']:>5}  {r['t_peak']:>5}  {r['k_peak']:>3}  "
            f"{r['chunk_delta_peak_signed']:>+10.4f}  {r['chunk_delta_peak_abs']:>10.4f}  "
            f"{r['single_frame_delta_peak_abs']:>11.4f}  {ratio:>13.2f}"
        )

    print(f"\nReference: single-frame delta saturation ceilings on gripper")
    print(f"  q98 - q02 = {SATURATION_CEILING_NORMAL:.4f} m (max in-chunk span at norm in [-1, +1])")
    print(f"  1.5 * (q98 - q02) = {SATURATION_CEILING_CLAMP:.4f} m (max at clamp ±1.5)")

    print("\nDumping 64-frame windows for top 3 episodes ...")
    for r in top:
        s, c = r["ep_start"], r["ep_len"]
        obs_ep = obs_grip[s:s + c]
        act_ep = act_grip[s:s + c]
        path = TRACE_DIR / f"ep{r['episode_id']:03d}_t{r['t_peak']:04d}_k{r['k_peak']:02d}.csv"
        write_trace_csv(path, r["episode_id"], r["t_peak"], r["k_peak"], obs_ep, act_ep)
        print(
            f"  wrote {path.relative_to(ROOT)}  "
            f"(t*={r['t_peak']}, k*={r['k_peak']}, chunk Δ={r['chunk_delta_peak_signed']:+.4f} m)"
        )

    summary = {
        "repo_id": REPO_ID,
        "horizon": H,
        "single_frame_q02": SINGLE_FRAME_Q02,
        "single_frame_q98": SINGLE_FRAME_Q98,
        "saturation_ceiling_normal": SATURATION_CEILING_NORMAL,
        "saturation_ceiling_clamp": SATURATION_CEILING_CLAMP,
        "all_episodes_sorted_by_chunk_delta": per_ep,
        "top_3": top,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {SUMMARY_JSON.relative_to(ROOT)}")

    print("\n--- Verdict ---")
    worst = max(per_ep, key=lambda r: r["chunk_delta_peak_abs"])
    ratio_norm = worst["chunk_delta_peak_abs"] / SATURATION_CEILING_NORMAL
    ratio_clamp = worst["chunk_delta_peak_abs"] / SATURATION_CEILING_CLAMP
    print(
        f"Worst case observed: ep {worst['episode_id']} needs "
        f"|act[t*+k*] - obs[t*]| = {worst['chunk_delta_peak_abs']:.4f} m on the gripper.\n"
        f"  Buggy single-frame stats can express at most {SATURATION_CEILING_NORMAL:.4f} m at norm in [-1,+1] "
        f"({ratio_norm:.2f}× over budget),\n"
        f"  or {SATURATION_CEILING_CLAMP:.4f} m at the ±1.5 clamp ceiling "
        f"({ratio_clamp:.2f}× over budget)."
    )

    # Also report aggregate chunk-delta percentiles vs single-frame percentiles
    all_chunk = []
    all_single = []
    for ep_id, s, c in zip(ep_ids.tolist(), ep_starts.tolist(), ep_counts.tolist()):
        obs_ep = obs_grip[s:s + c]
        act_ep = act_grip[s:s + c]
        if c < H:
            continue
        valid = c - H + 1
        for k in range(H):
            all_chunk.append(act_ep[k:k + valid] - obs_ep[:valid])
        all_single.append(act_ep - obs_ep)
    all_chunk = np.concatenate(all_chunk)
    all_single = np.concatenate(all_single)
    print(
        f"\nDataset-wide percentiles on the gripper joint:\n"
        f"  single-frame delta act[t]-obs[t]:  q02={np.percentile(all_single,2):+.4f}  "
        f"q98={np.percentile(all_single,98):+.4f}  span={np.percentile(all_single,98)-np.percentile(all_single,2):.4f}\n"
        f"  chunk delta     act[t+k]-obs[t]:  q02={np.percentile(all_chunk,2):+.4f}  "
        f"q98={np.percentile(all_chunk,98):+.4f}  span={np.percentile(all_chunk,98)-np.percentile(all_chunk,2):.4f}"
    )
    summary["dataset_percentiles"] = {
        "single_frame_q02": float(np.percentile(all_single, 2)),
        "single_frame_q98": float(np.percentile(all_single, 98)),
        "chunk_q02": float(np.percentile(all_chunk, 2)),
        "chunk_q98": float(np.percentile(all_chunk, 98)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
