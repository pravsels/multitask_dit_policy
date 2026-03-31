# Multi-Task DiT — bin-pick-pack coffee capsules (v1)

## Mode
- run_type: experiment
- objective: Train Multi-Task DiT diffusion policy on single-task bin-pick-pack coffee capsules dataset with 17D state/action (joint_pos + eef_pose w/ 6D rotation), Ramen normalization, and CLIP task conditioning.

## Config
- script: `slurm/train.sh`
- config: `config/train_coffee_capsules.yaml`
- dataset: `villekuosmanen/bin_pick_pack_coffee_capsules` from shared cache at `/scratch/u6cr/pravsels.u6cr/huggingface_cache/lerobot/villekuosmanen/bin_pick_pack_coffee_capsules`
- container: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/container/multitask-dit-policy_arm64.sif` (6.2GB)
- key settings: `batch_size=32`, `train_steps=100000`, `save_freq=5000`, `log_freq=100`, `use_amp=true`, `num_workers=8`, `video_backend=pyav`
- state/action: 17D — `[joint_pos(7), eef_xyz(3), rot6d(6), gripper(1)]`
- delta actions: all dims except 6D rotation (dims 10–15), which stays absolute
- normalization: Ramen (q02/q98 percentile, per-timestep, per-dim, clipped to [-1.5, 1.5]); 6D rotation exempt; ImageNet mean/std for images
- task text: "pick a single coffee capsule from the cardboard tray and drop it inside the brown cardboard container holding a plastic bag"
- dataset size: 47,865 samples, 200 episodes (~2,991 steps/epoch, ~33 epochs total)
- model: DiT with CLIP ViT-B/16 vision encoder, CLIP text encoder, flow matching objective
- optimizer: AdamW (default lr), gradient clipping at 1.0
- wandb: offline mode, entity `pravsels`, project `multitask-dit-policy`

## Ramen Stats
- computed over full dataset (47,865 samples)
- runtime: ~35–40 min on GH200
- cached at `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_v1/ramen_stats.pt`

## Job
- job_id: `3536573`
- submitted: `2026-03-31T16:34:27+00:00`
- node: `nid010098`
- partition: `workq`, 1 GPU, 16 CPUs, exclusive, 24h walltime

### Failed attempts
- job `3531959` — `ImportError: torchcodec is required but not available.` — arm64 container doesn't have torchcodec. Fixed by switching `video_backend` to `pyav`.
- job `3532487` — same torchcodec error. Container was running installed package code, not the updated repo code. Fixed by adding `PYTHONPATH=${repo_dir}/src` to the slurm script so live repo code takes precedence.

## Status
- 2026-03-31 16:34 UTC — submitted
- 2026-03-31 16:34 UTC — Ramen stats computation started
- 2026-03-31 ~17:10 UTC — stats complete, training started
- 2026-03-31 ~18:33 UTC — step 5,812/100,000, loss ~0.01–0.03, ~5 it/s, checkpoint_5000 saved

## Results
- (in progress)

## W&B
- local: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_v1/wandb/offline-run-20260331_155412-s9pa741j`
- synced: (pending `wandb sync`)

## Artifacts
- ramen_stats: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_v1/ramen_stats.pt`
- checkpoints: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_v1/checkpoint_*/`
- final model: `/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs/coffee_capsules_v1/final_model/` (when complete)

## Next
- Monitor training to completion
- `wandb sync` the offline run
- Evaluate on `villekuosmanen/bin_pick_pack_coffee_capsules_eval`
- Add validation loss loop
- Consider learning rate schedule (cosine annealing)

## Notes
- Container has packages installed at `/usr/local/lib/python3.12/dist-packages/`, but `PYTHONPATH=${repo_dir}/src` in the slurm script overrides with live repo code — no rebuild needed for code changes, just `git pull`.
- Shared HF caches at `/scratch/u6cr/pravsels.u6cr/huggingface_cache/` — dataset and model weights are reused across projects.
- `run_name` in config scopes all outputs (stats, checkpoints, wandb) to a subdirectory — different experiments don't collide.
- The 6D rotation representation (Zhou et al. 2019) replaces RPY Euler angles for continuity. The 7D eef_pose (xyz + rpy + gripper) becomes 10D (xyz + rot6d + gripper), giving the full 17D vector.
- Ramen normalization skips 6D rotation dims to avoid corrupting the rotation matrix structure.
