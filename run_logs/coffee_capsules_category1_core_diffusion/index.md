# coffee_capsules_category1_core_diffusion

## Goal
- Category 1 core diffusion ablations around the cleaned coffee-capsules baseline.

## Mode
- run_type: `experiment`
- objective: Compare DDPM vs DDIM, 20 vs 50 inference steps, and RoPE vs no-RoPE on the cleaned coffee-capsules setup.

## Baseline
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Shared Overrides
- none

## Variants
| variant | run_name | job_id | status | description |
| --- | --- | --- | --- | --- |
| `baseline` | `coffee_capsules_category1_core_diffusion-baseline` | `3564785` | `completed` | Clean baseline config as-is. |
| `ddpm_20` | `coffee_capsules_category1_core_diffusion-ddpm_20` | `3564786` | `completed` | Keep DDPM and reduce inference steps to 20. |
| `ddpm_50` | `coffee_capsules_category1_core_diffusion-ddpm_50` | `3564787` | `completed` | Keep DDPM and reduce inference steps to 50. |
| `ddpm_rope` | `coffee_capsules_category1_core_diffusion-ddpm_rope` | `3564788` | `completed` | Keep DDPM baseline sampler and enable RoPE. |
| `ddim_20` | `coffee_capsules_category1_core_diffusion-ddim_20` | `3564789` | `completed` | Switch sampler to DDIM with 20 inference steps. |
| `ddim_50` | `coffee_capsules_category1_core_diffusion-ddim_50` | `3564790` | `completed` | Switch sampler to DDIM with 50 inference steps. |
| `ddim_rope_20` | `coffee_capsules_category1_core_diffusion-ddim_rope_20` | `3564791` | `completed` | Enable RoPE and use DDIM with 20 inference steps. |
| `ddim_rope_50` | `coffee_capsules_category1_core_diffusion-ddim_rope_50` | `3564792` | `completed` | Enable RoPE and use DDIM with 50 inference steps. |

## Sweep Evaluation
| variant | end_loss | runtime | canonical_wandb | takeaway |
| --- | --- | --- | --- | --- |
| `baseline` | `0.0030173` | `14:22:12` | `8xhewlqu` | Healthy baseline; tied with the non-RoPE group. |
| `ddpm_20` | `0.0030173` | `13:57:16` | `leamkw9q` | No training-loss gain over baseline from reducing DDPM steps to 20. |
| `ddpm_50` | `0.0030173` | `14:54:13` | `ay6nbct0` | No training-loss gain over baseline from DDPM 50-step inference. |
| `ddpm_rope` | `0.0029481` | `13:33:13` | `3695rhof` | Best-loss group; RoPE gives a mild edge without instability. |
| `ddim_20` | `0.0030173` | `15:47:49` | `innf1jzx` | Stable, but no loss advantage over the other non-RoPE runs. |
| `ddim_50` | `0.0030173` | `14:12:37` | `266664n6` | Stable tie with the rest of the non-RoPE group. |
| `ddim_rope_20` | `0.0029481` | `14:04:40` | `qh7lwdtq` | Best-loss group; good shortlist candidate for eval. |
| `ddim_rope_50` | `0.0029481` | `13:16:37` | `1j441mvr` | Best-loss group; similar to the other RoPE runs on train loss. |

## W&B
- project: `https://wandb.ai/pravsels/multitask-dit-policy`
- canonical sync rule: sync only the large completed offline run for each variant, not the tiny startup stub run.

## Hugging Face
- model_repo: `https://huggingface.co/pravsels/multitask-dit-coffee-capsules-core-diffusion-sweep`
- published_artifacts: params-only `checkpoint_40000` and `checkpoint_50000` for all 8 variants
- upload_mode: uploaded directly from Isambard scratch via `multitask-dit-policy_arm64.sif`, without a separate local staging copy
- integrity: the repo root `README.md` records per-variant and per-step `sha256` values for `model.safetensors`, `config.json`, and `ramen_stats.pt`, plus a manifest hash for each published checkpoint directory
- checksum_process: hashes were generated twice on the original scratch files and only recorded after both passes matched exactly
- checksum_source: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/hf_publish/coffee_capsules_core_diffusion_sweep/checksums.json`

## Summary
- All eight variants completed successfully with exit code `0` and produced `checkpoint_50000` plus `final_model`.
- The `40000` and `50000` params-only checkpoints for every variant are now published in the Hugging Face repo above, with a single root `README.md` containing the sweep summary and integrity data.
- The loss curves are all healthy and tightly clustered; there is no obvious bad configuration from training stability alone.
- Compared against the older `coffee_capsules_v1` baseline at step `50000` (`train/loss ~= 0.011694`), every run in this sweep performed substantially better, finishing around `0.00295` to `0.00302`.
- The only consistent split is RoPE vs non-RoPE: RoPE variants finished slightly lower (`0.0029481`) than the non-RoPE variants (`0.0030173`).
- `DDPM` vs `DDIM` and `20` vs `50` inference steps did not separate meaningfully on train loss in this sweep.

## Recommended Eval Order
- `ddpm_rope`
- `ddim_rope_20`
- `ddim_rope_50`
- `baseline` as the control comparison
