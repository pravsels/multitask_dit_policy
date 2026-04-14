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
- key settings: batch_size=80 per GPU (320 global), train_steps=40000, optimizer_lr=3e-4, warmup=500, save_freq=5000

## Job
- execution_id: 3811446
- submitted/start: 2026-04-14T07:19:42Z
- start_human: Tuesday, Apr 14, 2026 07:19 UTC
- end: pending
- end_human: pending
- runtime: pending
- node: pending (queue)

## Status
- 2026-04-14 07:19 UTC - submitted as Slurm job 3811446
- 2026-04-14 07:19 UTC - queue state: PENDING (Priority)

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
- submit this run with `sbatch slurm/train_block_tower_bs320_lr3e4.sh`
- monitor first hour for step/loss progression and any HF/W&B auth warnings
- if healthy, let it run to 40k and compare against v1 trajectory at equivalent steps
