# Multi-Task DiT — coffee capsules pooled Qwen GCloud (v1)

## Mode
- run_type: experiment
- objective: Train the coffee-capsules diffusion policy with pooled `Qwen/Qwen3-VL-4B-Instruct` conditioning on a GCloud A100 80GB, while the Isambard DDP run (`3633130`) remains stuck in `PENDING`.

## Config
- config: `config/train_coffee_capsules_qwen_pooled_gcloud.yaml`
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules`
- platform: GCloud `a2-ultragpu-1g` (1× A100 80GB, 12 vCPUs, 170GB RAM) in `us-central1-c`
- runtime: Docker (`multitask-dit-policy:latest`), native amd64 build on VM
- run_name: `coffee_capsules_qwen_pooled_gcloud_v1`
- key settings: `batch_size=16`, `train_steps=50000`, `save_freq=5000`, `log_freq=100`, `use_amp=true`, `num_workers=8`
- objective: `DDIM`, `num_train_timesteps=100`, `num_inference_steps=20`
- transformer: `use_rope=true`
- multimodal encoder: pooled `Qwen/Qwen3-VL-4B-Instruct`, `output_dim=2048`, `max_text_length=200`, `freeze_backbone=false`, `lr_multiplier=0.1`, `gradient_checkpointing=true`
- optimizer: base `optimizer_lr=2e-5`, Qwen backbone at `0.1x` multiplier

## Ramen Stats
- trainer cache path: `/home/ps/outputs/coffee_capsules_qwen_pooled_gcloud_v1/coffee_capsules_qwen_pooled_gcloud_v1/ramen_stats.pt`
- seeded from: `pravsels/multitask-dit-coffee-capsules-core-diffusion-sweep` (HuggingFace, `baseline/checkpoints/50000/ramen_stats.pt`)

## Commits
- `a6c46e1` `add gcloud training config with batch_size=32 for A100 80GB`
- `dcee184` `fix cuda device index for single-GPU runs`
- `1a43489` `reduce gcloud batch_size to 16 after OOM at 32`
- `ea08d72` `add gradient checkpointing support for multimodal encoder`

## Jobs
- long run: started 2026-04-05 on GCloud VM `a100-training`

## Status
- 2026-04-05 — Isambard DDP job `3633130` stuck in `PENDING`; decided to deploy to GCloud
- 2026-04-05 — provisioned `a2-ultragpu-1g` in `us-central1-c` (initial `a2-ultragpu-4g` failed with `ZONE_RESOURCE_POOL_EXHAUSTED`)
- 2026-04-05 — Docker image built natively on VM, repo cloned, `ramen_stats.pt` pre-seeded
- 2026-04-05 — first attempt with `batch_size=32`: OOM during backward pass
- 2026-04-05 — reduced to `batch_size=16`: still OOM without gradient checkpointing
- 2026-04-05 — enabled `gradient_checkpointing=true` on multimodal encoder; `batch_size=16` fits comfortably
- 2026-04-05 — training running stably: VRAM ~47GB (57%), step time ~2.5s/it, loss dropping from ~0.78 → ~0.15 by step 119
- 2026-04-05 — estimated completion: ~35 hours, cost ~$177 ($5.07/hr on-demand)

## Current Long-Run State
- VM: `a100-training` (`a2-ultragpu-1g`, `us-central1-c`)
- output dir: `/home/ps/outputs/coffee_capsules_qwen_pooled_gcloud_v1`
- GPU metrics at step ~119: 47018 MiB VRAM (57%), temp cycling normally, power spikes to 400W
- currently present artifacts:
  - `ramen_stats.pt`
- not yet present at last check:
  - `checkpoint_5000/`
  - `final_model/`

## Notes
- Originally planned 4× A100 DDP on GCloud, but `a2-ultragpu-4g` was unavailable due to capacity limits. Fell back to single-GPU `a2-ultragpu-1g`.
- `batch_size=32` OOM'd even though the Isambard GH200 ran `batch_size=64` at 92GB — the A100 has only 81GB and different memory characteristics.
- Gradient checkpointing cut VRAM from ~80GB (OOM) to ~47GB at `batch_size=16`, at the cost of ~20-30% slower step time. Only the Qwen backbone layers are checkpointed; re-forward computes ~10 layers at a time instead of caching all 40.
- Training log is written to `/home/ps/outputs/coffee_capsules_qwen_pooled_gcloud_v1/train.log` on the VM via `tee`.
- VM must be deleted after training completes to stop billing.
