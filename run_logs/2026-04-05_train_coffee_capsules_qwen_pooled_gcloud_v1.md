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
- 2026-04-07 — **training complete**: 50000/50000 steps in 35h17m, final loss 0.00672, step time ~2.54s/it
- 2026-04-07 — container exited cleanly (exit code 0), VM still running
- 2026-04-07 — uploaded checkpoints 35k, 45k, 50k (inference-only) to HF `pravsels/multitask-dit-coffee-capsules-qwen-pooled`
- 2026-04-07 — uploaded model card README to HF repo
- 2026-04-07 — VM `a100-training` deleted

## Results
- total training time: 35h 17m
- final loss: 0.00672
- step time: ~2.54s/it (consistent throughout)
- W&B: `WANDB_MODE=offline` was set but no `WANDB_API_KEY` — wandb did not initialize, no offline run to sync

## Checkpoints
All saved to `/home/ps/outputs/coffee_capsules_qwen_pooled_gcloud_v1/coffee_capsules_qwen_pooled_gcloud_v1/`:

| checkpoint | timestamp | loss |
|---|---|---|
| `checkpoint_5000/` | Apr 5 23:37 | 0.0151 |
| `checkpoint_10000/` | Apr 6 03:08 | 0.0148 |
| `checkpoint_15000/` | Apr 6 06:40 | 0.0134 |
| `checkpoint_20000/` | Apr 6 10:12 | 0.0176 |
| `checkpoint_25000/` | Apr 6 13:43 | 0.0107 |
| `checkpoint_30000/` | Apr 6 17:15 | 0.0109 |
| `checkpoint_35000/` | Apr 6 20:46 | 0.00816 |
| `checkpoint_40000/` | Apr 7 00:18 | 0.00974 |
| `checkpoint_45000/` | Apr 7 03:50 | 0.00727 |
| `checkpoint_50000/` | Apr 7 07:22 | 0.00672 |
| `final_model/` | Apr 7 07:23 | — |

`final_model/` contents: `config.json` (2.6K), `model.safetensors` (8.7G), `ramen_stats.pt` (17K), `train_state.pt` (18G)

## HuggingFace Upload
- repo: [`pravsels/multitask-dit-coffee-capsules-qwen-pooled`](https://huggingface.co/pravsels/multitask-dit-coffee-capsules-qwen-pooled)
- uploaded: `checkpoint_35000/`, `checkpoint_45000/`, `checkpoint_50000/` (inference-only: `model.safetensors`, `config.json`, `ramen_stats.pt`)
- checkpoint hashes (sha256 manifest, verified twice):

| checkpoint | sha256 |
|---|---|
| `checkpoint_35000` | `365922a3e53e7f0a6c7bb36356eeb43373dac5b5993f1f2efe66a047c5e1002f` |
| `checkpoint_45000` | `3771f8e6fdf7810988fde0f44d9c6132db1bc187fc7fbc8ed07c9c6b57621c58` |
| `checkpoint_50000` | `e218da1ee6fdd6d15d44d0244356b9836700eae50683028a60b220e4ed12b737` |

Reproduce with: `find <ckpt_dir> -type f \( -name "model.safetensors" -o -name "config.json" -o -name "ramen_stats.pt" \) | sort | xargs sha256sum | sha256sum`

## VM State
- VM `a100-training` **deleted** 2026-04-07
- GCloud project: `gen-lang-client-0388971498`

## Notes
- Originally planned 4× A100 DDP on GCloud, but `a2-ultragpu-4g` was unavailable due to capacity limits. Fell back to single-GPU `a2-ultragpu-1g`.
- `batch_size=32` OOM'd even though the Isambard GH200 ran `batch_size=64` at 92GB — the A100 has only 81GB and different memory characteristics.
- Gradient checkpointing cut VRAM from ~80GB (OOM) to ~47GB at `batch_size=16`, at the cost of ~20-30% slower step time. Only the Qwen backbone layers are checkpointed; re-forward computes ~10 layers at a time instead of caching all 40.
- Training log at `/home/ps/outputs/coffee_capsules_qwen_pooled_gcloud_v1/coffee_capsules_qwen_pooled_gcloud_v1/train.log` (8MB, mostly progress bar output with loss values).
- W&B was not functional — `WANDB_MODE=offline` without `WANDB_API_KEY` means wandb never initialized. No offline run directory exists to sync.
- VM deleted after uploading checkpoints to HuggingFace.
