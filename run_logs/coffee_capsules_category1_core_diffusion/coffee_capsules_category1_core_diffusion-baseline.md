# coffee_capsules_category1_core_diffusion-baseline

## Variant
- variant: `baseline`
- description: Clean baseline config as-is.
- status: `completed`
- job_id: `3564785`

## Config
- base_config: `/workspace/config/train_coffee_capsules.yaml`
- output_dir: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs`
- slurm_script: `/workspace/slurm/train.sh`

## Overrides
- `--run_name=coffee_capsules_category1_core_diffusion-baseline`

## Job
- job_id: `3564785`
- start: `2026-04-01T20:55:27+00:00`
- end: `2026-04-02T11:17:33+00:00`
- runtime: `14:22:12`
- node: `nid010624`
- exit_code: `0`

## Status
- 2026-04-01 20:55 UTC — submitted and started on `nid010624`
- 2026-04-02 11:17 UTC — completed successfully at step `50000`

## Results
- final_step: `50000`
- end_train_loss: `0.0030173`
- checkpoint: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-baseline/checkpoint_50000/train_state.pt`
- final_model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-baseline/final_model/train_state.pt`
- loss_one_liner: Loss dropped smoothly and converged in the healthy non-RoPE cluster, ending slightly above the RoPE variants.

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_category1_core_diffusion-baseline/wandb/offline-run-20260401_205540-8xhewlqu`
- synced: `https://wandb.ai/pravsels/multitask-dit-policy/runs/8xhewlqu`
- notes: Stable training curve with no obvious pathologies; use as the control comparison for the sweep.

## Next
- keep as the control checkpoint in downstream evals
- compare directly against `ddpm_rope`, `ddim_rope_20`, and `ddim_rope_50`
