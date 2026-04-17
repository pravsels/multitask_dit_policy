# Coffee Capsules Norm Fix

**Date**: 2026-04-17
**Branch**: `stage1-multimodal-abstraction`
**Config**: `config/train_coffee_capsules.yaml`
**Cluster**: Isambard AIP2, 1 node, 4x GPUs

## Mode
- run_type: replication
- objective: retrain coffee capsules from scratch with per-timestep (H,D) RAMEN action stats and semantic cleanup (action chunk starts at current action)

## Config
- script: `slurm/train_coffee_capsules_bs320_lr3e4.sh`
- config: `config/train_coffee_capsules.yaml`
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules`
- key settings: batch_size=80 per GPU (320 global), train_steps=50000, optimizer_lr=3e-4, warmup=500, save_freq=1000, keep_freq=5000, num_workers=8, prefetch_factor=2, horizon=32, n_action_steps=32, DDIM, resize_shape=[224,224], crop_shape=null
- what changed vs prior run (`2026-04-14_coffee_capsules_bs320_lr3e4`):
  - `compute_ramen_stats` now emits (H=32, D=17) action stats instead of (1, 17)
  - action chunk semantic cleanup: slot 0 = act[t] - obs[t] (first executable action), no look-back prefix
  - fresh training from step 0 (old checkpoints semantically incompatible)

## Job
- execution_id: 3880994
- submitted: 2026-04-17T16:55:00Z

## Status
- 2026-04-17 16:55 UTC - submitted as Slurm job 3880994, pending (Priority)

## Results

## W&B

## Next
