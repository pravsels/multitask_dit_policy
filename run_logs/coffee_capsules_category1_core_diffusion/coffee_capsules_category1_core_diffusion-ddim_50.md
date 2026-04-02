# coffee_capsules_category1_core_diffusion-ddim_50

## Variant
- variant: `ddim_50`
- description: Switch sampler to DDIM with 50 inference steps.
- status: `completed`
- job_id: `3564790`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddim_50`
- `--policy.objective.noise_scheduler_type=DDIM`
- `--policy.objective.num_inference_steps=50`

## Job
- job_id: `3564790`
- start: `2026-04-01T20:55:28+00:00`
- end: `2026-04-02T11:07:58+00:00`
- runtime: `14:12:37`
- node: `nid010713`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010713`
- 2026-04-02 11:07 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0030173`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_50/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_50/final_model/train_state.pt`
- loss_one_liner: Loss remained healthy and stable but ended tied with the rest of the non-RoPE group.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_50/wandb/offline-run-20260401_205541-266664n6`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/266664n6`
- notes: DDIM 50-step training behaved well, but the sweep does not show a training-loss edge for this setting.

## Next
- lower priority than the RoPE variants for checkpoint eval
- retain mainly as a stable DDIM non-RoPE reference point
