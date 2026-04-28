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
- start: 2026-04-15T11:31:35Z
- start_human: Tuesday, Apr 15, 2026 11:31 UTC
- end: 2026-04-16T11:31:54Z
- end_human: Wednesday, Apr 16, 2026 11:31 UTC
- runtime: 1-00:00:19 (walltime limit)
- node: nid010251

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
- 2026-04-15 11:31 UTC - job 3828939 running on nid010251, stable throughput ~1.2 it/s
- 2026-04-16 11:31 UTC - job TIMEOUT at 24h walltime, step 28,974/50,000; loss ~0.009–0.02, throughput ~1.2 it/s with periodic bursty stalls; checkpoint_25000 was saved but pruned by keep_freq=10000 logic (save-then-delete), leaving checkpoint_20000 as latest on disk
- 2026-04-16 ~12:00 UTC - resubmitted as 3856137 to resume from checkpoint_20000; updated code with checkpoint pruning fix (`397136e`): save_freq=1000, keep_freq=5000, rolling window retained until milestone; pending (Priority)

## Results
- runtime: 24h (walltime limit)
- final step: 28,974/50,000
- start_train_loss: ~1.07
- end_train_loss: ~0.009–0.02
- start_val_loss: n/a
- end_val_loss: n/a
- loss_one_liner: 1.07 → ~0.01 by step 29k, noisy due to rank-0-only logging
- checkpoint: checkpoint_20000 (latest on disk; checkpoint_25000 pruned, 30000 never reached)
- config_snapshot: `config/train_block_tower_bs320_lr3e4.yaml`

## W&B
- local (run 1): `outputs/block_tower_baseline_v2_bs320_lr3e4/wandb/offline-run-20260415_113202-c00fb0ai`
- synced (run 1): https://wandb.ai/pravsels/dit_block_tower_config_fix/runs/c00fb0ai
- local (run 2 — resume): `outputs/block_tower_baseline_v2_bs320_lr3e4/wandb/offline-run-20260416_121837-a8ql2mse`
- synced (run 2): https://wandb.ai/pravsels/dit_block_tower_config_fix/runs/a8ql2mse
- notes: periodic bursty stalls visible in throughput; loss curve noisy due to rank-0-only logging

## HuggingFace
- repo: https://huggingface.co/pravsels/dit_block_tower_config_fix
- uploaded checkpoints: checkpoint_40000 (model.safetensors + config.json + ramen_stats.pt)
- sha256 (model.safetensors, verified twice): `455f0f6fe4032461848de181414dab2315cd2e7433aaa1828ff362d8891f7c29`

## Next
- monitor 3856137 for resume from checkpoint_20000 (auto-detect), confirm training continues from step 20000
- expect ~30k remaining steps; at ~1.2 it/s should reach ~50k within 24h walltime
- if walltime hit again, latest checkpoint will be at most 1k steps old (rolling window fix)
