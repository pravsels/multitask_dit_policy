# coffee_capsules_category1_core_diffusion-ddpm_50

## Variant
- variant: `ddpm_50`
- description: Keep DDPM and reduce inference steps to 50.
- status: `completed`
- job_id: `3564787`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddpm_50`
- `--policy.objective.noise_scheduler_type=DDPM`
- `--policy.objective.num_inference_steps=50`

## Job
- job_id: `3564787`
- start: `2026-04-01T20:55:28+00:00`
- end: `2026-04-02T11:49:34+00:00`
- runtime: `14:54:13`
- node: `nid010642`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010642`
- 2026-04-02 11:49 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0030173`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_50/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_50/final_model/train_state.pt`
- loss_one_liner: Loss stayed healthy throughout training but finished tied with the other non-RoPE runs.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddpm_50/wandb/offline-run-20260401_205541-ay6nbct0`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/ay6nbct0`
- notes: DDPM with 50 inference steps was stable but did not separate from baseline on training loss.

## Next
- lower priority than the RoPE variants for checkpoint eval
- keep as a reference only if DDPM 50-step behavior is useful at inference time
