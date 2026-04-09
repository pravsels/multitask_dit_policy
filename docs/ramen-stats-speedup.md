# Ramen Stats Computation Speedup

## Problem

Computing Ramen normalization statistics required iterating every sample in the dataset via `dataset[i]`. Each call triggered LeRobot's `__getitem__`, which decodes a video frame (via pyav) even though Ramen stats only need numerical state/action columns.

On the block tower 6-mix config (~341k frames), this took **~9.5 hours** at ~10 samples/s on the HPC.

## Root cause

LeRobot stores images as mp4 video files and numerical data as parquet. The `__getitem__` path always decodes the video frame for that index — there's no way to request "just the numbers" through the standard API.

`robocandywrapper`'s `load_videos=False` flag only controls whether video files are *downloaded*; if they're already cached, frames are still decoded on access.

## Solution

Bypass `__getitem__` entirely. Each `LeRobotDataset` exposes `.hf_dataset`, which is a HuggingFace `datasets.Dataset` backed by the parquet files. Reading columns from it gives direct access to numerical data without touching video files.

The new `compute_ramen_stats` in `ramen_normalization.py`:

1. **Bulk-reads** all schema keys from `.hf_dataset` across all inner datasets (handles multi-dataset configs via `WrappedRobotDataset._datasets`)
2. **Converts** to tensors in one `np.array()` → `torch.tensor()` call per column
3. **Assembles** state/action vectors with RPY→rot6d conversion as batch tensor ops
4. **Computes** delta actions and percentiles on the full (N, D) tensors

```
_bulk_read_columns(dataset, keys)
    └─ for ds in dataset._datasets:
           hf = ds.hf_dataset
           subset = hf.select_columns(keys)
           torch.tensor(np.array(subset[k]))

_assemble_bulk(data, schema_entries)
    └─ concatenate columns, apply convert_eef_pose where declared
```

## Results

| Dataset | Frames | Before | After | Speedup |
|---------|--------|--------|-------|---------|
| eval_build_block_tower_dino_test_set | 6,534 | ~10 min | **1.4s** | ~440x |
| block_tower 6-mix (estimated) | 341,704 | ~9.5 hrs | **~30-60s** | ~500x |

The bottleneck shifts from video I/O to `torch.quantile` on the full tensor, which is negligible.

## Caching

Stats are still cached to `{run_dir}/ramen_stats.pt` after first computation. Subsequent runs load from cache instantly regardless of method.
