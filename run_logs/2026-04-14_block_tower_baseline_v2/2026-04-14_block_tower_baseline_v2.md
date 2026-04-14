# Block Tower Baseline v2 (bs320 lr3e-4)

**Date**: 2026-04-14
**Branch**: `stage1-multimodal-abstraction`
**Config**: `config/train_block_tower_bs320_lr3e4.yaml`
**Cluster**: Isambard AIP2, 1 node, 4x GPUs

## Mode
- run_type: replication
- objective: validate reliable learning signal using higher global batch and learning rate suggested by prior baseline feedback

## Config
- script: `slurm/train_block_tower_bs320_lr3e4.sh`
- config: `config/train_block_tower_bs320_lr3e4.yaml`
- dataset: HF mix (`villekuosmanen/build_block_tower` + DAgger rounds 1.0.0-1.4.0)
- key settings: batch_size=80 per GPU (320 global), train_steps=50000, optimizer_lr=3e-4, warmup=500, save_freq=5000, keep_freq=10000, num_workers=28, prefetch_factor=1, horizon=32, n_action_steps=32, DDIM, resize_shape=[224,224], crop_shape=null

## Job
- execution_id: 3812891
- submitted/start: 2026-04-14T09:03:00Z
- start_human: Tuesday, Apr 14, 2026 09:03 UTC
- end: pending
- end_human: pending
- runtime: pending
- node: nid011269

## Status
- 2026-04-14 07:19 UTC - submitted as Slurm job 3811446
- 2026-04-14 07:24 UTC - job 3811446 running on nid010685; observed bursty utilization and periodic stalls
- 2026-04-14 07:45 UTC - resubmitted as 3812038 after throughput tuning (`cpus-per-task=288`, workers increased)
- 2026-04-14 07:49 UTC - job 3812038 OUT_OF_MEMORY (`ReqMem=460000M`, full-node RAM exhausted)
- 2026-04-14 08:19 UTC - resubmitted as 3812244 with updated policy settings and workers=16; training progressed but remained bursty with long stalls
- 2026-04-14 08:50 UTC - resubmitted as 3812607 (`workers=32`, `prefetch_factor=1`); OOM at step 34
- 2026-04-14 09:03 UTC - resubmitted as 3812891 (`workers=28`, `prefetch_factor=1`), currently RUNNING
- 2026-04-14 09:17 UTC - live check: step progression continues (not hard-stuck) with intermittent one-rank lag and periodic stalls

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
- local: `outputs/block_tower_baseline_v2_bs320_lr3e4/wandb/offline-run-*`
- synced: pending
- notes: run has multiple OOM and relaunch attempts while tuning host-side data pipeline; current run is stable so far but throughput remains bursty

## HuggingFace
- repo: pending
- uploaded checkpoints: pending
- includes: pending

## Next
- continue monitoring 3812891 for OOM/stall behavior and step cadence
- if OOM recurs, reduce `num_workers` by small increments (e.g. 24) while keeping `prefetch_factor=1`
- if stable, run to checkpoint_10000 and compare throughput/loss trajectory to v1 and prior v2 attempts
