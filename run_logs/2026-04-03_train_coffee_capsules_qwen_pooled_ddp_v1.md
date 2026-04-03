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

## Jobs
- long run: `3605472`

## Status
- 2026-04-03 14:07 UTC — submitted DDP run `3605472` with pre-seeded `ramen_stats.pt`
- 2026-04-03 14:13 UTC — confirmed Slurm allocated `4` GPUs on `nid010208`
- 2026-04-03 14:13 UTC — confirmed the trainer is running multi-process DDP from NCCL `rank0` startup logs

## Current Long-Run State
- job `3605472` state: `RUNNING`
- node: `nid010208`
- output dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_qwen_pooled_ddp_v1`
- currently present artifacts:
  - `ramen_stats.pt`
  - `wandb/`

## Notes
- This run is intentionally parallel to the earlier 1-GPU run `3604515`.
- Early step time is about `~1.25s/it`; with DDP the more relevant metric is throughput because each optimizer step now covers global batch `64`.
