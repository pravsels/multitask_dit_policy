from __future__ import annotations

import torch
import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature

from multitask_dit_policy.model import model as model_module
from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig, PooledMultimodalEncoderConfig


class FakeObservationEncoder(nn.Module):
    def __init__(self, config, load_pretrained_backbones: bool = True):
        super().__init__()
        self.multimodal_encoder = nn.Module()
        self.multimodal_encoder.model = nn.Linear(3, 3)
        self.multimodal_encoder.projection = nn.Linear(3, 3)
        self.vision_encoder = None
        self.conditioning_dim = 8

    def encode(self, batch):
        batch_size = batch["observation.state"].shape[0]
        return torch.zeros(batch_size, self.conditioning_dim)


class FakeDiffusionTransformer(nn.Module):
    def __init__(self, config, conditioning_dim: int):
        super().__init__()
        self.output = nn.Linear(conditioning_dim, config.action_feature.shape[0])


class FakeObjective:
    def __init__(self, *args, **kwargs):
        self.horizon = kwargs["horizon"]
        self.action_dim = kwargs["action_dim"]

    def conditional_sample(self, noise_predictor, batch_size, conditioning_vec):
        base = torch.arange(
            batch_size * self.horizon * self.action_dim,
            dtype=torch.float32,
        ).reshape(batch_size, self.horizon, self.action_dim)
        return base


def test_get_optim_params_uses_multimodal_lr_multiplier(monkeypatch):
    monkeypatch.setattr(model_module, "ObservationEncoder", FakeObservationEncoder)
    monkeypatch.setattr(model_module, "DiffusionTransformer", FakeDiffusionTransformer)
    monkeypatch.setattr(model_module, "DiffusionObjective", FakeObjective)

    cfg = MultiTaskDiTConfig(n_obs_steps=2)
    cfg.input_features = {
        "observation.image.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    cfg.observation_encoder.multimodal = PooledMultimodalEncoderConfig(
        model="Qwen/Qwen3-VL-4B-Instruct",
        output_dim=64,
        max_text_length=200,
        freeze_backbone=False,
        lr_multiplier=0.1,
    )

    policy = model_module.MultiTaskDiTPolicy(cfg, load_pretrained_backbones=False)

    optim_groups = policy.get_optim_params()

    assert len(optim_groups) == 2

    base_group, multimodal_group = optim_groups
    assert "lr" not in base_group
    assert multimodal_group["lr"] == cfg.optimizer_lr * cfg.observation_encoder.multimodal.lr_multiplier

    base_param_ids = {id(param) for param in base_group["params"]}
    multimodal_param_ids = {id(param) for param in multimodal_group["params"]}

    backbone_param_ids = {id(param) for param in policy.observation_encoder.multimodal_encoder.model.parameters()}
    projection_param_ids = {id(param) for param in policy.observation_encoder.multimodal_encoder.projection.parameters()}

    assert backbone_param_ids == multimodal_param_ids
    assert projection_param_ids <= base_param_ids


def test_generate_actions_returns_from_chunk_slot_zero(monkeypatch):
    monkeypatch.setattr(model_module, "ObservationEncoder", FakeObservationEncoder)
    monkeypatch.setattr(model_module, "DiffusionTransformer", FakeDiffusionTransformer)
    monkeypatch.setattr(model_module, "DiffusionObjective", FakeObjective)

    cfg = MultiTaskDiTConfig(n_obs_steps=2, horizon=6, n_action_steps=3)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }

    policy = model_module.MultiTaskDiTPolicy(cfg, load_pretrained_backbones=False)

    batch = {
        "observation.state": torch.zeros(1, 2, 3),
    }
    actions = policy._generate_actions(batch)

    expected = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]])
    assert torch.equal(actions, expected)
