from __future__ import annotations

import torch
import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature

from multitask_dit_policy.model import observation_encoder as observation_encoder_module
from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig


class FakeVisionEncoder(nn.Module):
    def get_output_shape(self):
        return (4, 1, 1)

    def forward(self, x):
        return torch.zeros((x.shape[0], 4, 1, 1), dtype=x.dtype, device=x.device)


class FakeTextEncoder(nn.Module):
    def __init__(self, model_name: str, projection_dim: int, pretrained: bool = True):
        super().__init__()
        self.projection_dim = projection_dim

    def forward(self, texts):
        return torch.zeros((len(texts), self.projection_dim))


def test_observation_encoder_supports_separate_encoder_per_camera(monkeypatch):
    monkeypatch.setattr(observation_encoder_module, "create_vision_encoder", lambda config, pretrained=True: FakeVisionEncoder())
    monkeypatch.setattr(observation_encoder_module, "CLIPTextEncoder", FakeTextEncoder)

    cfg = MultiTaskDiTConfig(n_obs_steps=2)
    cfg.transformer.hidden_dim = 8
    cfg.observation_encoder.vision.use_separate_encoder_per_camera = True
    cfg.input_features = {
        "observation.image.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.image.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }

    encoder = observation_encoder_module.ObservationEncoder(cfg, load_pretrained_backbones=False)

    assert encoder.num_cameras == 2
    assert encoder.conditioning_dim == 38
