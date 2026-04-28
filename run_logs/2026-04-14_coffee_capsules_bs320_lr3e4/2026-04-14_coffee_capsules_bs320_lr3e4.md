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
- submitted: 2026-04-14T21:30:00Z
- start: 2026-04-15T03:54:45Z
- start_human: Tuesday, Apr 15, 2026 03:54 UTC
- end: 2026-04-16T03:54:57Z
- end_human: Wednesday, Apr 16, 2026 03:54 UTC
- runtime: 1-00:00:12 (walltime limit)
- node: nid010386

## Status
- 2026-04-14 21:30 UTC - submitted as Slurm job 3819693 (`workers=20`, `prefetch_factor=1`), pending
- prior job 3818885 (`workers=24`) cancelled before running — preemptive fix based on block tower OOM at step 1164 with same worker count
- 2026-04-15 03:54 UTC - job running on nid010386
- 2026-04-16 03:54 UTC - job TIMEOUT at 24h walltime, step 30,530/50,000; loss ~0.004-0.006, throughput ~1.7 it/s with periodic bursty stalls (same pattern as block tower)

## Results
- runtime: 24h (walltime limit)
- final step: 30,530/50,000
- start_train_loss: ~0.15
- end_train_loss: ~0.004-0.006
- start_val_loss: n/a
- end_val_loss: n/a
- loss_one_liner: 0.15 → ~0.005 by step 30k, noisy due to rank-0-only logging
- checkpoint: checkpoint_30000
- config_snapshot: `config/train_coffee_capsules.yaml`

## W&B
- local: `outputs/coffee_capsules_bs320_lr3e4_v1/wandb/offline-run-20260415_035512-6pobxw3c`
- synced: https://wandb.ai/pravsels/dit_coffee_capsules_config_fix/runs/6pobxw3c
- notes: periodic bursty stalls visible in throughput; loss curve noisy due to rank-0-only logging (fixed in `0a46894`)

## HuggingFace
- repo: https://huggingface.co/pravsels/dit_coffee_capsules_config_fix
- uploaded checkpoints: checkpoint_30000 (model.safetensors + config.json + ramen_stats.pt)
- sha256 (model.safetensors, verified twice): `1e0fa3278d7bc8e8a578822e03c613c9fe24f8c99ddacc0f0c416686cfe627b9`

## Next
- resume from checkpoint_30000 with updated config (`workers=8`, `prefetch_factor=2`) to finish remaining ~20k steps
- compare final loss and eval against prior coffee capsules baselines
