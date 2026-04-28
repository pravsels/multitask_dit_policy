# Block Tower Norm Fix

**Date**: 2026-04-17
**Branch**: `stage1-multimodal-abstraction`
**Config**: `config/train_block_tower.yaml`
**Cluster**: Isambard AIP2, 1 node, 4x GPUs

## Mode
- run_type: replication
- objective: retrain block tower from scratch with per-timestep (H,D) RAMEN action stats and semantic cleanup (action chunk starts at current action)

## Config
- script: `slurm/train_block_tower_bs320_lr3e4.sh`
- config: `config/train_block_tower.yaml`
- dataset: HF mix (`villekuosmanen/build_block_tower` + DAgger rounds 1.0.0-1.4.0)
- key settings: batch_size=80 per GPU (320 global), train_steps=50000, optimizer_lr=3e-4, warmup=500, save_freq=1000, keep_freq=5000, num_workers=8, prefetch_factor=2, horizon=32, n_action_steps=32, DDIM, resize_shape=[224,224], crop_shape=null
- what changed vs prior run (`2026-04-14_block_tower_baseline_v2`):
  - `compute_ramen_stats` now emits (H=32, D=17) action stats instead of (1, 17)
  - action chunk semantic cleanup: slot 0 = act[t] - obs[t] (first executable action), no look-back prefix
  - config consolidated from `train_block_tower_bs320_lr3e4.yaml` into `train_block_tower.yaml`
  - fresh training from step 0 (old checkpoints semantically incompatible)

## Job
- execution_id: 3880995
- submitted: 2026-04-17T16:55:00Z

## Status
- 2026-04-17 16:55 UTC - submitted as Slurm job 3880995, pending (Priority)

## Results

## W&B

## Next
