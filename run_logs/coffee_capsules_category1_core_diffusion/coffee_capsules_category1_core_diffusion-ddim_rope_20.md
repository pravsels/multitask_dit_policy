# coffee_capsules_category1_core_diffusion-ddim_rope_20

## Variant
- variant: `ddim_rope_20`
- description: Enable RoPE and use DDIM with 20 inference steps.
- status: `completed`
- job_id: `3564791`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddim_rope_20`
- `--policy.objective.noise_scheduler_type=DDIM`
- `--policy.objective.num_inference_steps=20`
- `--policy.transformer.use_rope=true`

## Job
- job_id: `3564791`
- start: `2026-04-01T20:55:28+00:00`
- end: `2026-04-02T11:00:01+00:00`
- runtime: `14:04:40`
- node: `nid010715`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010715`
- 2026-04-02 11:00 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0029481`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_rope_20/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_rope_20/final_model/train_state.pt`
- loss_one_liner: Loss converged smoothly and landed in the best-loss group, matching the other RoPE variants.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_rope_20/wandb/offline-run-20260401_205541-qh7lwdtq`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/qh7lwdtq`
- notes: Strong shortlist candidate; combines the RoPE loss gain with the 20-step DDIM setting.

## Next
- prioritize for downstream checkpoint eval
- compare directly against `ddpm_rope`, `ddim_rope_50`, and `baseline`
