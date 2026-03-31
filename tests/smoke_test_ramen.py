"""Smoke test for the Ramen normalization + 6D rotation pipeline.

Run inside Docker with GPU:
    docker run --rm --gpus all -v "$PWD/src:/workspace/src" \
        -v "$PWD/tests:/workspace/tests" \
        -e HF_DATASETS_CACHE=/tmp/hf_datasets \
        multitask-dit-policy:amd64 python3 tests/smoke_test_ramen.py
"""

import torch

from multitask_dit_policy.utils.rotation import (
    convert_eef_pose,
    rot6d_to_rpy,
    rpy_to_rot6d,
    unconvert_eef_pose,
)


def test_rpy_roundtrip():
    rpy = torch.tensor([
        [0.1, 0.2, 0.3],
        [1.0, -0.5, 2.0],
        [0.0, 0.0, 0.0],
        [3.14, 1.57, -1.57],
    ], dtype=torch.float64)

    rot6d = rpy_to_rot6d(rpy)
    rpy_back = rot6d_to_rpy(rot6d)
    max_err = (rpy - rpy_back).abs().max().item()
    print(f"RPY roundtrip max error: {max_err:.2e}")
    assert max_err < 1e-6, f"Roundtrip error too large: {max_err}"
    assert rot6d.abs().max() <= 1.0 + 1e-7, "rot6d values outside [-1, 1]"
    print(f"rot6d range: [{rot6d.min():.4f}, {rot6d.max():.4f}]")


def test_eef_pose_roundtrip():
    eef7 = torch.randn(2, 5, 7)
    eef10 = convert_eef_pose(eef7)
    assert eef10.shape == (2, 5, 10), f"Expected (2,5,10), got {eef10.shape}"
    eef7_back = unconvert_eef_pose(eef10)
    assert eef7_back.shape == (2, 5, 7), f"Expected (2,5,7), got {eef7_back.shape}"

    # xyz and gripper should be exact
    assert torch.allclose(eef7[..., :3], eef7_back[..., :3], atol=1e-6)
    assert torch.allclose(eef7[..., 6:], eef7_back[..., 6:], atol=1e-6)

    # RPY may differ by equivalent representations, so compare via rot6d
    rot6d_orig = rpy_to_rot6d(eef7[..., 3:6])
    rot6d_back = rpy_to_rot6d(eef7_back[..., 3:6])
    max_err = (rot6d_orig - rot6d_back).abs().max().item()
    print(f"eef_pose rotation roundtrip max error: {max_err:.2e}")
    assert max_err < 1e-5, f"eef rotation roundtrip error too large: {max_err}"


def test_full_pipeline():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from robocandywrapper import make_dataset_without_config

    from multitask_dit_policy.model.model import MultiTaskDiTPolicy
    from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig
    from multitask_dit_policy.utils.dataset_adapter import (
        ROT6D_END,
        ROT6D_START,
        adapt_batch,
        compute_adapted_features,
        detect_sub_features,
        select_default_keys,
    )
    from multitask_dit_policy.utils.ramen_normalization import (
        build_norm_mask,
        compute_ramen_stats,
        ramen_normalize_batch,
        ramen_unnormalize,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = "villekuosmanen/bin_pick_pack_coffee_capsules"

    # Feature detection
    meta = LeRobotDatasetMetadata(repo_id=repo)
    detected_s, detected_a = detect_sub_features(meta.features)
    state_keys, action_keys = select_default_keys(detected_s, detected_a)
    print(f"state_keys: {state_keys}")
    print(f"action_keys: {action_keys}")

    inp, out = compute_adapted_features(meta.features, state_keys, action_keys)
    state_dim = inp["observation.state"].shape[0]
    print(f"State dim: {state_dim}")
    assert state_dim == 17, f"Expected 17, got {state_dim}"
    assert out["action"].shape[0] == 17

    norm_mask = build_norm_mask(state_dim, ROT6D_START, ROT6D_END)
    print(f"norm_mask: {norm_mask}")

    # Load 1 episode
    cfg = MultiTaskDiTConfig()
    cfg.input_features = inp
    cfg.output_features = out

    ds = make_dataset_without_config(
        repo_id=repo,
        action_delta_indices=list(cfg.action_delta_indices),
        observation_delta_indices=list(cfg.observation_delta_indices),
        video_backend="torchcodec",
        episodes=[0],
        use_imagenet_stats=True,
    )
    print(f"Dataset length: {len(ds)}")

    # Ramen stats
    ramen_stats = compute_ramen_stats(ds, state_keys, action_keys, norm_mask, device=device)
    print(f"obs_q02 shape: {ramen_stats['obs_q02'].shape}")
    print(f"action_q02 shape: {ramen_stats['action_q02'].shape}")

    # Adapt batch
    sample = ds[0]
    batch = {
        k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample.items()
    }
    batch = adapt_batch(batch, state_keys, action_keys, norm_mask.to(device))

    print(f"obs shape: {batch['observation.state'].shape}")
    print(f"action shape: {batch['action'].shape}")
    assert batch["observation.state"].shape[-1] == 17
    assert batch["action"].shape[-1] == 17

    rot_vals = batch["action"][..., ROT6D_START:ROT6D_END]
    print(f"action rot6d range: [{rot_vals.min():.4f}, {rot_vals.max():.4f}]")

    # Normalize
    norm_batch = ramen_normalize_batch(batch, ramen_stats, norm_mask.to(device))
    m = norm_mask.to(device)
    ns = norm_batch["observation.state"]
    print(f"Norm state range (non-rot): [{ns[..., m].min():.3f}, {ns[..., m].max():.3f}]")
    print(
        f"Rot6d unchanged: "
        f"{torch.allclose(ns[..., ROT6D_START:ROT6D_END], batch['observation.state'][..., ROT6D_START:ROT6D_END])}"
    )

    # Image normalization
    for ik in (k for k in norm_batch if k.startswith("observation.image")):
        print(f"{ik} range: [{norm_batch[ik].min():.3f}, {norm_batch[ik].max():.3f}]")

    # Unnormalize roundtrip
    unnorm = ramen_unnormalize(ns, ramen_stats["obs_q02"], ramen_stats["obs_q98"], m)
    max_err = (unnorm - batch["observation.state"]).abs().max().item()
    print(f"Unnormalize roundtrip max error: {max_err:.2e}")

    # Forward + backward
    print("Creating policy...")
    policy = MultiTaskDiTPolicy(cfg)
    policy.to(device)
    policy.train()
    print(f"Params: {sum(p.numel() for p in policy.parameters()):,}")

    loss, _ = policy(norm_batch)
    print(f"Loss: {loss.item():.6f}")
    loss.backward()
    print("Backward OK")


if __name__ == "__main__":
    print("=== Rotation tests ===")
    test_rpy_roundtrip()
    test_eef_pose_roundtrip()
    print("Rotation tests passed!\n")

    print("=== Full pipeline test ===")
    test_full_pipeline()
    print("\nAll tests passed!")
