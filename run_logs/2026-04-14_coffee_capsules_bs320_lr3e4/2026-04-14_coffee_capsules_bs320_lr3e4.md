# Coffee Capsules bs320 lr3e-4

**Date**: 2026-04-14
**Branch**: `stage1-multimodal-abstraction`
**Config**: `config/train_coffee_capsules.yaml`
**Cluster**: Isambard AIP2, 1 node, 4x GPUs

## Mode
- run_type: replication
- objective: replicate coffee capsules training with higher global batch size (320) and learning rate (3e-4)

## Config
- script: `slurm/train_coffee_capsules_bs320_lr3e4.sh`
- config: `config/train_coffee_capsules.yaml`
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules`
- key settings: batch_size=80 per GPU (320 global), train_steps=50000, optimizer_lr=3e-4, warmup=500, save_freq=5000, keep_freq=10000, num_workers=20, prefetch_factor=1, horizon=32, n_action_steps=32, DDIM, resize_shape=[224,224], crop_shape=null

## Job
- execution_id: 3819693
- submitted/start: 2026-04-14T21:30:00Z
- start_human: Monday, Apr 14, 2026 21:30 UTC
- end: pending
- end_human: pending
- runtime: pending
- node: pending

## Status
- 2026-04-14 21:30 UTC - submitted as Slurm job 3819693 (`workers=20`, `prefetch_factor=1`), pending
- prior job 3818885 (`workers=24`) cancelled before running — preemptive fix based on block tower OOM at step 1164 with same worker count

## Results
- runtime: pending
- final step: pending
- start_train_loss: pending
- end_train_loss: pending
- start_val_loss: pending
- end_val_loss: pending
- loss_one_liner: pending
- checkpoint: pending
- config_snapshot: pending

## W&B
- local: pending
- synced: pending
- notes: pending

## Next
- monitor 3819693 for OOM stability
- if OOM recurs, reduce `num_workers` to 16
- if stable, run to completion and compare against prior coffee capsules baselines
