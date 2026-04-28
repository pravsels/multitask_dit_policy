"""Unit tests for the unified Ramen clip value.

`MultiTaskDiTConfig.ramen_clip_value` should be the single source of
truth for two clipping operations on Ramen-normalized actions/states:

  1. `ramen_normalize` clamps the normalized output to ±clip_value.
  2. The DDIM noise scheduler's `clip_sample_range` is set to the same
     value (so inference-time sampling can reach the full trained range).

This keeps the training-time saturation boundary and the inference-time
clip in lockstep — there is no way to silently desync them.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from multitask_dit_policy.utils.configuration import (
    DiffusionConfig,
    MultiTaskDiTConfig,
)
from multitask_dit_policy.utils.ramen_normalization import (
    build_norm_mask,
    ramen_normalize,
    ramen_normalize_batch,
    ramen_unnormalize,
)


# ---- ramen_normalize: clip_value parameter ----------------------------------


def _stats(D: int = 4):
    """Tiny per-dim stats: q02=-1, q98=+1 → normalize is identity (·1)."""
    q02 = torch.full((1, D), -1.0)
    q98 = torch.full((1, D), +1.0)
    mask = torch.ones(D, dtype=torch.bool)
    return q02, q98, mask


def test_ramen_normalize_default_clip_is_1_5():
    """Backward-compat: the default clip is the historical ±1.5."""
    q02, q98, mask = _stats(D=2)
    big = torch.tensor([[10.0, -10.0]])

    y = ramen_normalize(big, q02, q98, mask)

    assert torch.allclose(y, torch.tensor([[1.5, -1.5]]))


def test_ramen_normalize_respects_clip_value():
    """Override the clip and the saturation boundary moves with it."""
    q02, q98, mask = _stats(D=2)
    big = torch.tensor([[10.0, -10.0]])

    y = ramen_normalize(big, q02, q98, mask, clip_value=2.0)

    assert torch.allclose(y, torch.tensor([[2.0, -2.0]]))


def test_ramen_normalize_clip_value_does_not_affect_in_range_values():
    """Values that already fit in [-clip_value, +clip_value] are untouched."""
    q02, q98, mask = _stats(D=3)  # q02=-1, q98=+1 → normalize is identity
    x = torch.tensor([[0.0, 0.5, -0.7]])

    y = ramen_normalize(x, q02, q98, mask, clip_value=1.5)

    assert torch.allclose(y, x, atol=1e-6)


def test_ramen_unnormalize_round_trip_at_boundary_uses_clip_value():
    """A value clamped at the new boundary unnormalizes to the matching physical bound."""
    q02 = torch.tensor([[0.0]])
    q98 = torch.tensor([[10.0]])
    mask = torch.ones(1, dtype=torch.bool)
    huge = torch.tensor([[1000.0]])

    # With clip_value=2.0, normalized output saturates at +2.0
    # → unnormalize: ((2 + 1) / 2) * (10 - 0) + 0 = 15.0
    y = ramen_normalize(huge, q02, q98, mask, clip_value=2.0)
    x_back = ramen_unnormalize(y, q02, q98, mask)

    assert torch.allclose(y, torch.tensor([[2.0]]))
    assert torch.allclose(x_back, torch.tensor([[15.0]]))


def test_ramen_normalize_batch_forwards_clip_value():
    """`ramen_normalize_batch` should pass clip_value through to `ramen_normalize`."""
    D = 4
    q02, q98, mask = _stats(D=D)
    stats = {
        "obs_q02": q02, "obs_q98": q98,
        "action_q02": q02, "action_q98": q98,
        "norm_mask": mask,
    }
    batch = {
        "observation.state": torch.full((1, 1, D), 10.0),
        "action":            torch.full((1, 1, D), 10.0),
    }

    out = ramen_normalize_batch(batch, stats, mask, clip_value=2.5)

    assert torch.allclose(out["observation.state"], torch.full((1, 1, D), 2.5))
    assert torch.allclose(out["action"],            torch.full((1, 1, D), 2.5))


# ---- MultiTaskDiTConfig: ramen_clip_value field ----------------------------


def test_multitask_dit_config_default_ramen_clip_value():
    """The new top-level knob defaults to 1.5 (matches the historical clamp)."""
    cfg = MultiTaskDiTConfig()
    assert cfg.ramen_clip_value == 1.5


def test_multitask_dit_config_ramen_clip_value_is_overridable():
    cfg = MultiTaskDiTConfig(ramen_clip_value=2.0)
    assert cfg.ramen_clip_value == 2.0


def test_multitask_dit_config_rejects_non_positive_clip_value():
    with pytest.raises(ValueError):
        MultiTaskDiTConfig(ramen_clip_value=0.0)
    with pytest.raises(ValueError):
        MultiTaskDiTConfig(ramen_clip_value=-1.0)


# ---- Wiring into the diffusion noise scheduler ------------------------------


def _build_diffusion_objective(parent_clip_value: float = 1.5):
    """Build a MultiTaskDiTPolicy with a tiny config and return its scheduler."""
    from multitask_dit_policy.model.objectives import DiffusionObjective
    objective_cfg = DiffusionConfig(noise_scheduler_type="DDIM")
    obj = DiffusionObjective(
        objective_cfg,
        action_dim=4,
        horizon=8,
        ramen_clip_value=parent_clip_value,
    )
    return obj.noise_scheduler


def test_diffusion_scheduler_uses_ramen_clip_value():
    """The HF DDIM scheduler's clip_sample_range must equal ramen_clip_value."""
    sched = _build_diffusion_objective(parent_clip_value=1.5)
    assert sched.config.clip_sample is True
    assert sched.config.clip_sample_range == 1.5


def test_diffusion_scheduler_tracks_overridden_ramen_clip_value():
    sched = _build_diffusion_objective(parent_clip_value=2.0)
    assert sched.config.clip_sample is True
    assert sched.config.clip_sample_range == 2.0


def test_legacy_diffusion_config_clip_sample_range_is_ignored_with_warning():
    """Old-format configs with clip_sample_range set must still load, but the
    value is ignored in favour of the unified `ramen_clip_value`."""
    from multitask_dit_policy.model.objectives import DiffusionObjective
    legacy = DiffusionConfig(
        noise_scheduler_type="DDIM",
        clip_sample=True,
        clip_sample_range=1.0,  # legacy buggy default
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = DiffusionObjective(
            legacy, action_dim=4, horizon=8, ramen_clip_value=1.5,
        )
        assert any("clip_sample_range" in str(warn.message) for warn in w), (
            f"expected a deprecation warning about clip_sample_range; got {[str(x.message) for x in w]}"
        )

    assert obj.noise_scheduler.config.clip_sample_range == 1.5


# ---- End-to-end: model construction picks up the parent config -------------


def test_policy_diffusion_objective_uses_parent_ramen_clip_value():
    """Building a real MultiTaskDiTPolicy threads ramen_clip_value through to the scheduler."""
    pytest.importorskip("transformers")  # observation encoder loads CLIP weights
    from lerobot.configs.types import FeatureType, PolicyFeature
    from multitask_dit_policy.model.model import MultiTaskDiTPolicy

    cfg = MultiTaskDiTConfig(
        n_obs_steps=1, horizon=4, n_action_steps=4,
        ramen_clip_value=1.7,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
    )
    policy = MultiTaskDiTPolicy(cfg, load_pretrained_backbones=False)

    sched = policy.objective.noise_scheduler
    assert sched.config.clip_sample is True
    assert sched.config.clip_sample_range == 1.7
