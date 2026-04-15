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
- key settings: batch_size=80 per GPU (320 global), train_steps=50000, optimizer_lr=3e-4, warmup=500, save_freq=5000, keep_freq=10000, num_workers=8, prefetch_factor=2, horizon=32, n_action_steps=32, DDIM, resize_shape=[224,224], crop_shape=null

## Job
- execution_id: 3828939
- submitted: 2026-04-15T~11:00:00Z
- start: pending
- start_human: pending
- end: pending
- end_human: pending
- runtime: pending
- node: pending

## Status
- 2026-04-14 07:19 UTC - submitted as Slurm job 3811446
- 2026-04-14 07:24 UTC - job 3811446 running on nid010685; observed bursty utilization and periodic stalls
- 2026-04-14 07:45 UTC - resubmitted as 3812038 after throughput tuning (`cpus-per-task=288`, workers increased)
- 2026-04-14 07:49 UTC - job 3812038 OUT_OF_MEMORY (`ReqMem=460000M`, full-node RAM exhausted)
- 2026-04-14 08:19 UTC - resubmitted as 3812244 with updated policy settings and workers=16; training progressed but remained bursty with long stalls
- 2026-04-14 08:50 UTC - resubmitted as 3812607 (`workers=32`, `prefetch_factor=1`); OOM at step 34
- 2026-04-14 09:03 UTC - resubmitted as 3812891 (`workers=28`, `prefetch_factor=1`)
- 2026-04-14 09:17 UTC - live check on 3812891: step progression continues with intermittent one-rank lag and periodic stalls
- 2026-04-14 18:48 UTC - resubmitted as 3815464 (`workers=24`, `prefetch_factor=1`), ran on nid011127
- 2026-04-14 19:42 UTC - job 3815464 OUT_OF_MEMORY at step 1164/50000 (loss=0.0417, ~1.66 it/s, 51 min runtime, MaxRSS=247GB)
- 2026-04-14 21:30 UTC - resubmitted as 3819692 (`workers=20`, `prefetch_factor=1`), pending
- 2026-04-15 04:28 UTC - job 3819692 FAILED on nid010634 after 00:34:14 (Slurm exit 1:0); stderr shows NCCL watchdog collective timeout at sequence 214 (rank 1/2/3), torchrun aborted with `ChildFailedError`
- 2026-04-15 ~11:00 UTC - resubmitted as 3828939 (`workers=8`, `prefetch_factor=2`, `TORCH_NCCL_TRACE_BUFFER_SIZE=1000`, `NCCL_ASYNC_ERROR_HANDLING=1`); reduced workers from 20→8 to cut Lustre I/O contention, increased prefetch to 2 to maintain pipeline depth

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

## HuggingFace
- repo: pending
- uploaded checkpoints: pending
- includes: pending

## Next
- monitor 3828939 for stable throughput past step 1000 (no bursty stalls)
- if NCCL timeout recurs, flight recorder traces will show the stuck collective's stack trace
- if stable, run to checkpoint at step 5000/10000 and compare loss trajectory to v1
