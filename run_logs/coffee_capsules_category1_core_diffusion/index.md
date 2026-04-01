# coffee_capsules_category1_core_diffusion

## Goal
- Category 1 core diffusion ablations around the cleaned coffee-capsules baseline.

## Baseline
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Shared Overrides
- none

## Variants
| variant | run_name | job_id | status | description |
| --- | --- | --- | --- | --- |
| `baseline` | `coffee_capsules_category1_core_diffusion-baseline` | `-` | `planned` | Clean baseline config as-is. |
| `ddpm_20` | `coffee_capsules_category1_core_diffusion-ddpm_20` | `-` | `planned` | Keep DDPM and reduce inference steps to 20. |
| `ddpm_50` | `coffee_capsules_category1_core_diffusion-ddpm_50` | `-` | `planned` | Keep DDPM and reduce inference steps to 50. |
| `ddpm_rope` | `coffee_capsules_category1_core_diffusion-ddpm_rope` | `-` | `planned` | Keep DDPM baseline sampler and enable RoPE. |
| `ddim_20` | `coffee_capsules_category1_core_diffusion-ddim_20` | `-` | `planned` | Switch sampler to DDIM with 20 inference steps. |
| `ddim_50` | `coffee_capsules_category1_core_diffusion-ddim_50` | `-` | `planned` | Switch sampler to DDIM with 50 inference steps. |
| `ddim_rope_20` | `coffee_capsules_category1_core_diffusion-ddim_rope_20` | `-` | `planned` | Enable RoPE and use DDIM with 20 inference steps. |
| `ddim_rope_50` | `coffee_capsules_category1_core_diffusion-ddim_rope_50` | `-` | `planned` | Enable RoPE and use DDIM with 50 inference steps. |
