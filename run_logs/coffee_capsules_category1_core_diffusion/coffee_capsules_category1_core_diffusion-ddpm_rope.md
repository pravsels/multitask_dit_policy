# coffee_capsules_category1_core_diffusion-ddpm_rope

## Variant
- variant: `ddpm_rope`
- description: Keep DDPM baseline sampler and enable RoPE.
- status: `planned`
- job_id: `pending`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddpm_rope`
- `--policy.transformer.use_rope=true`

## Status
- created
