# Block Tower Baseline v1

**Date**: 2026-04-09
**Branch**: `stage1-multimodal-abstraction`
**Config**: `config/train_block_tower.yaml`
**Cluster**: Isambard AIP2, 1 node, 4x GPUs

## Job History

| Job ID | Submitted (UTC) | Status | Notes |
|--------|-----------------|--------|-------|
| 3705708 | ~Apr 8 | Failed | `KeyError: 'observation.state.pos'` — hardcoded key names didn't match block tower dataset schema |
| 3707280 | Apr 9 12:54 | Cancelled | Ramen stats stuck at ~10 it/s over 341k samples (~9.5h ETA), video decoding bottleneck |
| 3707596 | Apr 9 13:00 | Cancelled | Bulk parquet stats worked, but unauthenticated HF requests hit 429 rate limits |
| 3707697 | Apr 9 13:00 | **Timeout** | Clean run, 35k/50k steps completed before 24h walltime. Loss decreasing at cutoff. |

## Dataset

6-dataset mix (1 base + 5 DAgger rounds), ~341k total frames:

- `villekuosmanen/build_block_tower`
- `villekuosmanen/dAgger_build_block_tower_1.0.0`
- `villekuosmanen/dAgger_build_block_tower_1.1.0`
- `villekuosmanen/dAgger_build_block_tower_1.2.0`
- `villekuosmanen/dAgger_build_block_tower_1.3.0`
- `villekuosmanen/dAgger_build_block_tower_1.4.0`

DAgger policy frames are filtered out via `ControlModePlugin` (only human-control frames used for training).

## Schema

Block tower datasets use a different key layout than coffee capsules:

```yaml
dataset_schema:
  state:                              # 16D total (7 + 9)
    - key: observation.state          # joint positions (7D)
      dim: 7
    - key: observation.eef_6d_pose    # xyz + rpy → xyz + rot6d (6D → 9D)
      dim: 6
      convert_rotation: true
  action:                             # 17D total (7 + 10)
    - key: action                     # joint commands (7D)
      dim: 7
    - key: action.eef_pose            # xyz + rpy + gripper → xyz + rot6d + gripper (7D → 10D)
      dim: 7
      convert_rotation: true
  rot6d_slice: [10, 16]
```

State (16D) and action (17D) are asymmetric. Delta actions are computed on the shared 16D prefix.

## Training Config

| Parameter | Value |
|-----------|-------|
| Batch size | 64 per GPU (256 global) |
| Train steps | 50,000 |
| Learning rate | 2e-5, cosine schedule |
| Warmup | 500 steps |
| Horizon | 100 |
| Action steps | 50 |
| Obs steps | 2 |
| Diffusion steps | 100 (DDPM, squaredcos_cap_v2) |
| Vision encoder | CLIP ViT-B/16 (per-camera, lr_mult=0.1) |
| Text encoder | CLIP ViT-B/16 |
| AMP | enabled |
| Save freq | every 5,000 steps |

## Key Changes This Session

1. **Task-based DatasetSchema** (`7d6be2a`): Replaced hardcoded key names with per-task YAML schema declaring keys, dimensions, and RPY→rot6d conversion. Supports asymmetric state/action dims.

2. **Bulk parquet Ramen stats** (`97bb353`): Reads numerical columns directly from `hf_dataset` (parquet), bypassing video decoding. ~440x speedup (9.5h → ~30s for 341k frames). See `docs/ramen-stats-speedup.md`.

3. **HF token passthrough** (`120228f`): Reads `~/.hf_token` on HPC and passes as `HF_TOKEN` env var to the Apptainer container, avoiding 429 rate limits.

## Storage (scratch)

Checked Apr 9:

| Directory | Size |
|-----------|------|
| `openpi` | 959 GB |
| `huggingface_cache` | 151 GB |
| `multitask_dit_policy` | 41 GB |
| **Total (measured)** | **~1.15 TB / 5 TB quota** |

## W&B
- local: `wandb/offline-run-20260409_130038-pv8q64et`
- synced: https://wandb.ai/pravsels/dit_block_tower/runs/pv8q64et

## HuggingFace
- repo: https://huggingface.co/pravsels/dit_block_tower_baseline
- uploaded checkpoints: step 35000, params only
- includes: README, TRAINING_LOG, assets (ramen_stats.pt, valid_indices.json)

## Next Steps

- Resume from checkpoint_35000 for remaining 15k steps
- Check `valid_indices.json` report for DAgger filtering correctness
- Evaluate checkpoints on `villekuosmanen/eval_build_block_tower_dino_test_set` (5 episodes)
- Consider cleaning `openpi` scratch if storage pressure increases
