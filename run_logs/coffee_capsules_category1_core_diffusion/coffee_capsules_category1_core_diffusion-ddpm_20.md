# coffee_capsules_category1_core_diffusion-ddpm_20

## Variant
- variant: `ddpm_20`
- description: Keep DDPM and reduce inference steps to 20.
- status: `completed`
- job_id: `3564786`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddpm_20`
- `--policy.objective.noise_scheduler_type=DDPM`
- `--policy.objective.num_inference_steps=20`

## Job
- job_id: `3564786`
- start: `2026-04-01T20:55:28+00:00`
- end: `2026-04-02T10:52:37+00:00`
- runtime: `13:57:16`
- node: `nid010634`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010634`
- 2026-04-02 10:52 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0030173`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_20/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_20/final_model/train_state.pt`
- loss_one_liner: Loss converged cleanly but matched the baseline/non-RoPE group rather than improving on it.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_20/wandb/offline-run-20260401_205541-leamkw9q`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/leamkw9q`
- notes: Reducing DDPM inference steps to 20 did not produce a training-loss gain over baseline in this sweep.

## Next
- lower priority than the RoPE variants for checkpoint eval
- keep only if 20-step DDPM inference is materially simpler downstream
