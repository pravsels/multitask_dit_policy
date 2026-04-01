# coffee_capsules_category1_core_diffusion-ddim_50

## Variant
- variant: `ddim_50`
- description: Switch sampler to DDIM with 50 inference steps.
- status: `planned`
- job_id: `pending`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddim_50`
- `--policy.objective.noise_scheduler_type=DDIM`
- `--policy.objective.num_inference_steps=50`

## Status
- created
