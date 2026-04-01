# coffee_capsules_category1_core_diffusion-ddpm_20

## Variant
- variant: `ddpm_20`
- description: Keep DDPM and reduce inference steps to 20.
- status: `planned`
- job_id: `pending`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddpm_20`
- `--policy.objective.noise_scheduler_type=DDPM`
- `--policy.objective.num_inference_steps=20`

## Status
- created
