# Multi-Task DiT — coffee capsules pooled Qwen (v1)

## Mode
- run_type: experiment
- objective: Train the coffee-capsules diffusion policy with pooled `Qwen/Qwen3-VL-4B-Instruct` conditioning on the `stage1-multimodal-abstraction` branch, using the `ddim_rope_20` recipe as the base and a reduced LR on the multimodal backbone.

## Config
- script: `slurm/train_qwen_pooled.sh`
- config: `config/train_coffee_capsules_qwen_pooled.yaml`
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules`
- repo checkout on HPC: `/home/u6cr/pravsels.u6cr/multitask_dit_policy_stage1_multimodal_abstraction`
- container: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/container/multitask-dit-policy_arm64.sif`
- key settings: `batch_size=16`, `train_steps=50000`, `save_freq=5000`, `log_freq=100`, `use_amp=true`, `num_workers=8`
- objective: `DDIM`, `num_train_timesteps=100`, `num_inference_steps=20`
- transformer: `use_rope=true`
- multimodal encoder: pooled `Qwen/Qwen3-VL-4B-Instruct`, `output_dim=2048`, `max_text_length=200`, `freeze_backbone=false`, `lr_multiplier=0.1`
- optimizer: base `optimizer_lr=2e-5`, Qwen backbone at `0.1x` multiplier
- state/action: 17D adapted state/action with Ramen normalization and 6D rotation exemption

## Ramen Stats
- trainer cache path: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_qwen_pooled_v1/ramen_stats.pt`
- seeded from: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/weights/ramen_stats/ddim_20/checkpoints/50000/ramen_stats.pt`
- reason: avoid recomputing dataset-wide Ramen stats before each retry

## Commits
- `8734b5e` `prepare pooled qwen train config`
- `0e72e25` `add qwen pooled submit script`
- `bae5aa7` `reduce qwen pooled batch size`
- `92d2167` `reduce qwen pooled batch size again`
- `f5070f2` `fix bf16 amp scaler`

## Jobs
- long run: `3604515`

## Status
- 2026-04-03 12:26 UTC — submitted long run `3604515` with pre-seeded `ramen_stats.pt`
- 2026-04-03 12:38 UTC — long run `3604515` observed healthy in progress, with cached stats reuse and training advancing past startup

## Current Long-Run State
- job `3604515` state: `RUNNING`
- node: `nid011209`
- observed elapsed at last check: `00:04:05`
- current output dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_qwen_pooled_v1`
- currently present artifacts:
  - `ramen_stats.pt`
  - `wandb/`
- not yet present at last check:
  - `checkpoint_5000/`
  - `final_model/`

## Notes
- `batch_size=16` is the first setting that survived both model load and real backward/optimizer steps on the 95 GB GPU.
- The critical training-loop fix was to keep AMP enabled but disable `GradScaler` when the active parameter set includes `torch.bfloat16`.
- The repeated torchvision video deprecation warning is noisy but non-fatal; training progressed through it.
- `output_dim=2048` was not the first-order cause of the original OOMs; the failures occurred inside Qwen forward before the small projection head mattered much.
