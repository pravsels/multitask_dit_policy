# coffee_capsules_category1_core_diffusion-ddim_rope_50

## Variant
- variant: `ddim_rope_50`
- description: Enable RoPE and use DDIM with 50 inference steps.
- status: `completed`
- job_id: `3564792`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddim_rope_50`
- `--policy.objective.noise_scheduler_type=DDIM`
- `--policy.objective.num_inference_steps=50`
- `--policy.transformer.use_rope=true`

## Job
- job_id: `3564792`
- start: `2026-04-01T20:55:28+00:00`
- end: `2026-04-02T10:11:58+00:00`
- runtime: `13:16:37`
- node: `nid010718`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010718`
- 2026-04-02 10:11 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0029481`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_rope_50/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_rope_50/final_model/train_state.pt`
- loss_one_liner: Loss converged cleanly into the best-loss group, with no visible instability relative to the other RoPE runs.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_rope_50/wandb/offline-run-20260401_205540-1j441mvr`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/1j441mvr`
- notes: Strong shortlist candidate; RoPE helps on train loss, but this run should be judged against `ddim_rope_20` on checkpoint eval quality and inference cost.

## Next
- prioritize for downstream checkpoint eval
- compare directly against `ddim_rope_20`, `ddpm_rope`, and `baseline`
