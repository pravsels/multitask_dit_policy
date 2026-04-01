# Coffee Capsules Sweep Ideas

**Goal:** Capture high-value config-only sweep ideas for the cleaned coffee-capsules baseline so studies can be run one category at a time on Slurm.

**Baseline:** `config/train_coffee_capsules.yaml`

**Principle:** Keep the first studies interpretable. Sweep one category at a time instead of mixing objective, temporal, and architecture changes into one giant grid.

---

## Recommended Order

1. Core diffusion ablations
2. Temporal control ablations
3. Architecture and conditioning ablations
4. Optimization and training dynamics
5. Objective family comparison

This order keeps early studies close to the current baseline and pushes larger regime changes later.

## Category 1: Core Diffusion Ablations

These are the highest-signal "cool" variants that still stay close to the current baseline.

- `DDPM vs DDIM`
  - `--policy.objective.noise_scheduler_type=DDPM`
  - `--policy.objective.noise_scheduler_type=DDIM`
  - Why: isolates the sampler family while keeping the rest of the policy intact.

- `Inference step count`
  - `--policy.objective.num_inference_steps=10`
  - `--policy.objective.num_inference_steps=20`
  - `--policy.objective.num_inference_steps=50`
  - `--policy.objective.num_inference_steps=100`
  - Why: measures the quality vs compute trade-off at sampling time.

- `RoPE on/off`
  - `--policy.transformer.use_rope=true`
  - `--policy.transformer.use_rope=false`
  - Why: tests whether rotary position embeddings help temporal structure in this policy.

- `RoPE with and without learned positional encoding`
  - `--policy.transformer.use_rope=true --policy.transformer.use_positional_encoding=false`
  - `--policy.transformer.use_rope=true --policy.transformer.use_positional_encoding=true`
  - Why: distinguishes "RoPE replaces positional signal" from "RoPE adds another positional signal."

- `Prediction target`
  - `--policy.objective.prediction_type=epsilon`
  - `--policy.objective.prediction_type=sample`
  - Why: changes the regression target in diffusion training without changing the overall model family.

- `Diffusion timestep grid`
  - `--policy.objective.num_train_timesteps=50`
  - `--policy.objective.num_train_timesteps=100`
  - `--policy.objective.num_train_timesteps=200`
  - Why: tests whether a coarser or finer diffusion discretization helps.

### Suggested first study

- baseline
- `DDIM + 20 steps`
- `DDIM + 50 steps`
- `DDPM + RoPE`
- `DDIM + RoPE + 20 steps`

This is the best first real sweep because it is interesting, still close to the current baseline, and easy to interpret.

## Category 2: Temporal Control Ablations

These affect how much history the policy sees and how much future it predicts/executed per query.

- `Horizon`
  - `--policy.horizon=32`
  - `--policy.horizon=64`
  - `--policy.horizon=100`
  - Why: the README explicitly calls horizon sensitive and recommends starting closer to a one-second horizon.

- `Action chunk length`
  - `--policy.n_action_steps=24`
  - `--policy.n_action_steps=32`
  - `--policy.n_action_steps=50`
  - Why: directly changes the replan frequency vs stability trade-off.

- `Observation history`
  - `--policy.n_obs_steps=1`
  - `--policy.n_obs_steps=2`
  - `--policy.n_obs_steps=3`
  - Why: tests how much short-term temporal context is actually useful.

- `Windowing policy`
  - strict full-window trimming:
    - `--policy.drop_n_last_frames=49` for the current `horizon=100`, `n_action_steps=50`, `n_obs_steps=2`
    - or rely on auto-calculation if the field is omitted in a generated config
  - permissive padded windows:
    - `--policy.drop_n_last_frames=0 --policy.do_mask_loss_for_padding=true`
  - Why: tests whether keeping more end-of-episode windows is worth padded-loss handling.

- `Padding-mask safety check`
  - `--policy.do_mask_loss_for_padding=true`
  - `--policy.do_mask_loss_for_padding=false`
  - Why: only meaningful when compared alongside a permissive windowing setup.

## Category 3: Architecture And Conditioning Ablations

These are larger capacity or representation changes, but still config-only.

- `Separate encoder per camera`
  - `--policy.observation_encoder.vision.use_separate_encoder_per_camera=true`
  - `--policy.observation_encoder.vision.use_separate_encoder_per_camera=false`
  - Why: tests view-specific specialization vs parameter sharing.

- `Transformer depth`
  - `--policy.transformer.num_layers=6`
  - `--policy.transformer.num_layers=8`
  - Why: straightforward capacity increase with clean interpretation.

- `Transformer width and heads`
  - `--policy.transformer.hidden_dim=512 --policy.transformer.num_heads=8`
  - `--policy.transformer.hidden_dim=768 --policy.transformer.num_heads=12`
  - Why: increases model capacity while keeping head dimension valid.

- `Dropout`
  - `--policy.transformer.dropout=0.0`
  - `--policy.transformer.dropout=0.1`
  - `--policy.transformer.dropout=0.2`
  - Why: checks whether the current baseline is under- or over-regularized.

- `Diffusion timestep embedding size`
  - `--policy.transformer.diffusion_step_embed_dim=128`
  - `--policy.transformer.diffusion_step_embed_dim=256`
  - `--policy.transformer.diffusion_step_embed_dim=512`
  - Why: tests whether time-conditioning capacity is currently a bottleneck.

- `Vision backbone update rate`
  - `--policy.observation_encoder.vision.lr_multiplier=0.05`
  - `--policy.observation_encoder.vision.lr_multiplier=0.1`
  - `--policy.observation_encoder.vision.lr_multiplier=0.2`
  - Why: often matters more than raw base LR when fine-tuning pretrained vision encoders.

- `Crop policy`
  - `--policy.observation_encoder.vision.crop_is_random=false`
  - `--policy.observation_encoder.vision.crop_is_random=true`
  - Why: worth testing later, but deterministic center crop is a good baseline starting point.

## Category 4: Optimization And Training Dynamics

These are useful, but less exciting than the objective/temporal/architecture studies.

- `Constant vs cosine LR`
  - `--lr_scheduler=constant`
  - `--lr_scheduler=cosine`
  - Why: checks whether the new cosine baseline is actually helping.

- `Warmup length`
  - `--lr_warmup_steps=0`
  - `--lr_warmup_steps=500`
  - `--lr_warmup_steps=2000`
  - Why: warmup interacts with the shorter `train_steps=50000` baseline.

- `Cosine floor`
  - `--lr_scheduler_min_lr_scale=0.0`
  - `--lr_scheduler_min_lr_scale=0.05`
  - `--lr_scheduler_min_lr_scale=0.1`
  - Why: controls how aggressively the LR anneals at the end.

- `Base optimizer LR`
  - `--policy.optimizer_lr=1e-5`
  - `--policy.optimizer_lr=2e-5`
  - `--policy.optimizer_lr=3e-5`
  - `--policy.optimizer_lr=5e-5`
  - Why: still an important tuning knob, but should not be the first study.

- `Adam betas`
  - `--policy.optimizer_betas='[0.9,0.999]'`
  - `--policy.optimizer_betas='[0.95,0.999]'`
  - `--policy.optimizer_betas='[0.99,0.999]'`
  - Why: changes how quickly the optimizer reacts to noisy updates.

- `Weight decay`
  - `--policy.optimizer_weight_decay=0.0`
  - `--policy.optimizer_weight_decay=1e-4`
  - Why: simple regularization study once more important axes are understood.

## Category 5: Objective Family Comparison

This should be its own study because it is a bigger regime change than the others.

- `Diffusion vs flow matching`
  - diffusion baseline:
    - `--policy.objective.type=diffusion`
  - flow matching:
    - `--policy.objective.type=flow_matching`
  - Why: this is a meaningful model-family comparison, not a small ablation.

- `Flow matching timestep sampling`
  - `--policy.objective.type=flow_matching --policy.objective.timestep_sampling.type=uniform`
  - `--policy.objective.type=flow_matching --policy.objective.timestep_sampling.type=beta`
  - Why: beta sampling is already exposed and may matter more than the family switch alone.

- `Flow matching integration method`
  - `--policy.objective.type=flow_matching --policy.objective.integration_method=euler`
  - `--policy.objective.type=flow_matching --policy.objective.integration_method=rk4`
  - Why: changes inference compute/quality trade-off within the flow-matching family.

- `Flow matching integration steps`
  - `--policy.objective.type=flow_matching --policy.objective.num_integration_steps=50`
  - `--policy.objective.type=flow_matching --policy.objective.num_integration_steps=100`
  - Why: the flow-matching analogue of diffusion inference-step sweeps.

## Guardrails

- Do not mix too many categories into one study. One study should answer one question.
- Keep `run_name` unique per variant so outputs and checkpoints never collide.
- When sweeping `horizon`, `n_action_steps`, or `n_obs_steps`, double-check `drop_n_last_frames`.
- When using `drop_n_last_frames=0`, keep `do_mask_loss_for_padding=true`.
- Treat `flow_matching` as a separate family, not just another small variant in a diffusion study.

## Recommended Study Backlog

1. `core_diffusion_ablation`
2. `temporal_control_ablation`
3. `architecture_conditioning_ablation`
4. `optimizer_schedule_ablation`
5. `objective_family_ablation`
