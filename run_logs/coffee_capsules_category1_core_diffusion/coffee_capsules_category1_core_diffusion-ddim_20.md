# coffee_capsules_category1_core_diffusion-ddim_20

## Variant
- variant: `ddim_20`
- description: Switch sampler to DDIM with 20 inference steps.
- status: `completed`
- job_id: `3564789`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-ddim_20`
- `--policy.objective.noise_scheduler_type=DDIM`
- `--policy.objective.num_inference_steps=20`

## Job
- job_id: `3564789`
- start: `2026-04-01T20:55:29+00:00`
- end: `2026-04-02T12:43:10+00:00`
- runtime: `15:47:49`
- node: `nid010686`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010686`
- 2026-04-02 12:43 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0030173`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_20/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_20/final_model/train_state.pt`
- loss_one_liner: Loss converged cleanly but stayed in the same non-RoPE cluster as baseline and the DDPM step ablations.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-ddim_20/wandb/offline-run-20260401_205546-innf1jzx`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/innf1jzx`
- notes: Stable DDIM run, but no visible training-loss advantage over the baseline family.

## Next
- lower priority than the RoPE variants for checkpoint eval
- keep only if DDIM 20-step inference has downstream practical benefits
