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
- submitted/start: 2026-04-17T17:48:05Z
- start_human: Thursday, Apr 17, 2026 17:48 UTC
- end: 2026-04-18T17:48:34Z
- end_human: Friday, Apr 18, 2026 17:48 UTC
- runtime: 1-00:00:29
- node: nid010340

## Status
- 2026-04-17 16:55 UTC - submitted as Slurm job 3880995, pending (Priority)
- 2026-04-17 17:48 UTC - running on nid010340
- 2026-04-18 17:48 UTC - TIMEOUT at step ~29588/50000 (walltime 1 day reached)

## Results
- runtime: 1-00:00:29 (walltime limit)
- final step: ~29588/50000
- start_train_loss: 1.04
- end_train_loss: 0.0047
- start_val_loss: n/a
- end_val_loss: n/a
- loss_one_liner: Loss dropped steadily from 1.04 to 0.0047 over ~29.5k steps; healthy progression, no sign of plateau or overfitting.
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/block_tower_norm_fix/checkpoint_29000`
- config_snapshot: pending
- MaxRSS: ~117GB — no OOM with workers=8

## W&B
- local: `outputs/block_tower_norm_fix/wandb/offline-run-20260417_174833-ksuxe451`
- synced: `https://wandb.ai/pravsels/dit_block_tower_norm_fix/runs/ksuxe451`
- notes: pending — review dashboard with user

## HuggingFace
- repo: https://huggingface.co/pravsels/dit_block_tower_norm_fix
- uploaded checkpoints: step 29000 (params only)
- includes: model.safetensors, config.json, ramen_stats.json

## Next
- resume from checkpoint_29000 to complete remaining ~21k steps
- review W&B dashboard
