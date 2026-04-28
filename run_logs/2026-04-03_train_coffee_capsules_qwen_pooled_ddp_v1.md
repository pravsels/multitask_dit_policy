# Multi-Task DiT — coffee capsules pooled Qwen DDP (v1)

## Mode
- run_type: experiment
- objective: Run the pooled `Qwen/Qwen3-VL-4B-Instruct` coffee-capsules trainer on `stage1-multimodal-abstraction` with single-node `torchrun` DDP across 4 GPUs.

## Config
- script: `slurm/train_qwen_pooled.sh`
- config: `config/train_coffee_capsules_qwen_pooled.yaml`
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules`
- repo checkout on HPC: `/home/u6cr/pravsels.u6cr/multitask_dit_policy_stage1_multimodal_abstraction`
- run_name: `coffee_capsules_qwen_pooled_ddp_v1`
- key settings: `batch_size=16` per GPU, effective global batch `64`, `train_steps=50000`, `use_amp=true`, `num_workers=8`
- distributed launch: `torchrun --standalone --nnodes=1 --nproc_per_node=4 -m multitask_dit_policy.train`

## Ramen Stats
- trainer cache path: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_qwen_pooled_ddp_v1/ramen_stats.pt`
- seeded from: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_qwen_pooled_v1/ramen_stats.pt`

## Commits
- `4bbafcb` `add single-node ddp training support`
- `142bd48` `fix ddp checkpoint save nccl timeout`
- `90244f7` `fix grad scaler resume for disabled scaler (bf16)`
- `70ed076` `fix OOM on checkpoint resume by deserializing optimizer state to CPU`

## Jobs
- long run: `3605472` (FAILED — NCCL timeout at step 20000)
- resume attempt 1: `3617900` (FAILED — GradScaler resume crash)
- resume attempt 2: `3619886` (FAILED — OOM on first forward pass after resume)
- resume attempt 3: `3633130`

## Status
- 2026-04-03 14:07 UTC — submitted DDP run `3605472` with pre-seeded `ramen_stats.pt`
- 2026-04-03 14:13 UTC — confirmed Slurm allocated `4` GPUs on `nid010208`
- 2026-04-03 14:13 UTC — confirmed the trainer is running multi-process DDP from NCCL `rank0` startup logs
- 2026-04-03 21:31 UTC — job `3605472` FAILED after ~7h24m at step ~20000 with NCCL BROADCAST timeout (exit code 1)
- 2026-04-04 — root cause: checkpoint save (~26GB) on rank 0 blocked training while ranks 1-3 raced ahead; NCCL watchdog timed out after 600s. `checkpoint_20000` was incomplete (missing `ramen_stats.pt` and `train_state.pt`).
- 2026-04-04 — fix: added `dist.barrier()` after checkpoint saves + increased NCCL timeout from 10min to 30min (`142bd48`)
- 2026-04-04 — deleted corrupted `checkpoint_20000` and stale checkpoints; resuming from `checkpoint_15000`
- 2026-04-04 — submitted resume run `3617900`
- 2026-04-04 — job `3617900` FAILED immediately: `GradScaler.load_state_dict` rejected empty state dict saved by disabled scaler (bf16 AMP uses no grad scaling)
- 2026-04-04 — fix: skip scaler state restore when grad scaling is disabled (`90244f7`)
- 2026-04-04 — submitted resume run `3619886`
- 2026-04-05 — job `3619886` FAILED: OOM on all 4 GPUs during first forward pass. `torch.load(map_location=device)` put ~18.6GB optimizer state on GPU, then `load_state_dict` created a second GPU copy internally — double copy exceeded 95GB.
- 2026-04-05 — fix: deserialize train state to CPU; `load_state_dict` handles CPU→GPU transfer internally (`70ed076`)
- 2026-04-05 — submitted resume run `3633130`

## Current Long-Run State
- job `3633130` state: `PENDING`
- output dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_qwen_pooled_ddp_v1`
- resuming from: `checkpoint_15000`
- currently present artifacts:
  - `ramen_stats.pt`
  - `checkpoint_15000/`
  - `wandb/`

## Notes
- This run is intentionally parallel to the earlier 1-GPU run `3604515`.
- Early step time is about `~1.25s/it`; with DDP the more relevant metric is throughput because each optimizer step now covers global batch `64`.
- The NCCL timeout failure was caused by missing `dist.barrier()` around checkpoint saves — non-saving DDP ranks raced ahead into the next training step while rank 0 was still writing ~26GB to Lustre scratch.
