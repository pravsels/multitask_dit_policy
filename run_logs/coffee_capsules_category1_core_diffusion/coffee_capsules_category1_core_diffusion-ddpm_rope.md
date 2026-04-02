# coffee_capsules_category1_core_diffusion-ddpm_rope

## Variant
- variant: `ddpm_rope`
- description: Keep DDPM baseline sampler and enable RoPE.
- status: `completed`
- job_id: `3564788`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddpm_rope`
- `--policy.transformer.use_rope=true`

## Job
- job_id: `3564788`
- start: `2026-04-01T20:55:29+00:00`
- end: `2026-04-02T10:28:34+00:00`
- runtime: `13:33:13`
- node: `nid010644`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010644`
- 2026-04-02 10:28 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0029481`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_rope/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_rope/final_model/train_state.pt`
- loss_one_liner: Loss converged smoothly and finished in the best-loss group, suggesting a mild RoPE advantage over the non-RoPE variants.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_rope/wandb/offline-run-20260401_205544-3695rhof`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/3695rhof`
- notes: One of the three best-loss runs; strongest DDPM candidate to carry into checkpoint eval.

## Next
- prioritize for downstream checkpoint eval
- compare directly against `ddim_rope_20`, `ddim_rope_50`, and `baseline`
