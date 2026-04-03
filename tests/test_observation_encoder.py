from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature

from multitask_dit_policy.model import observation_encoder as observation_encoder_module
from multitask_dit_policy.utils.configuration import MultiTaskDiTConfig, PooledMultimodalEncoderConfig


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


class FailIfConstructedTextEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise AssertionError("legacy text encoder should be skipped in multimodal mode")


class FakeMultimodalEncoder(nn.Module):
    def __init__(self, config=None, pretrained: bool = True):
        super().__init__()
        self.output_dim = 5

    def get_output_dim(self):
        return self.output_dim

    def forward(self, batch):
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        return torch.ones((batch_size, n_obs_steps, self.output_dim))


class FakeProcessor:
    last_call: dict | None = None

    @classmethod
    def from_pretrained(cls, model_name: str):
        instance = cls()
        instance.model_name = model_name
        return instance

    def __call__(self, text, images, padding, truncation, max_length, return_tensors):
        type(self).last_call = {
            "text": text,
            "images": images,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "return_tensors": return_tensors,
        }
        batch_size = len(text)
        return {
            "input_ids": torch.ones((batch_size, 3), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]] * batch_size, dtype=torch.long),
            "pixel_values": torch.zeros((batch_size, 1), dtype=torch.float32),
        }


class FakeChatTemplateProcessor:
    last_call: dict | None = None

    @classmethod
    def from_pretrained(cls, model_name: str):
        instance = cls()
        instance.model_name = model_name
        return instance

    def __call__(self, *args, **kwargs):
        raise AssertionError("Qwen multimodal path should use apply_chat_template, not plain processor(text=..., images=...)")

    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_tensors,
        processor_kwargs,
    ):
        type(self).last_call = {
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": return_dict,
            "return_tensors": return_tensors,
            "processor_kwargs": processor_kwargs,
        }
        batch_size = len(messages)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]] * batch_size, dtype=torch.long),
            "pixel_values": torch.zeros((batch_size, 1), dtype=torch.float32),
        }


class FakeHFMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=5)
        self.backbone_weight = nn.Parameter(torch.ones(1))

    @classmethod
    def from_pretrained(cls, model_name: str):
        instance = cls()
        instance.model_name = model_name
        return instance

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, output_hidden_states=False, **kwargs):
        batch_size = input_ids.shape[0]
        hidden = torch.zeros((batch_size, 3, self.config.hidden_size), dtype=torch.float32)
        for idx in range(batch_size):
            hidden[idx, 0] = idx + 1
            hidden[idx, 1] = idx + 3
            hidden[idx, 2] = 999
        return SimpleNamespace(hidden_states=[hidden])


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


def test_observation_encoder_uses_multimodal_encoder_when_configured(monkeypatch):
    monkeypatch.setattr(
        observation_encoder_module,
        "create_multimodal_encoder",
        lambda config, pretrained=True: FakeMultimodalEncoder(config, pretrained),
        raising=False,
    )
    monkeypatch.setattr(observation_encoder_module, "CLIPTextEncoder", FailIfConstructedTextEncoder)

    cfg = MultiTaskDiTConfig(n_obs_steps=2)
    cfg.observation_encoder.vision.crop_shape = None
    cfg.observation_encoder.multimodal = PooledMultimodalEncoderConfig(model="fake-qwen", output_dim=5)
    cfg.input_features = {
        "observation.image.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }

    encoder = observation_encoder_module.ObservationEncoder(cfg, load_pretrained_backbones=False)

    batch = {
        "observation.state": torch.zeros((2, 2, 3)),
        "observation.images": torch.zeros((2, 2, 1, 3, 16, 16)),
        "task": ["pick", "place"],
    }
    encoded = encoder.encode(batch)

    assert encoder.conditioning_dim == 16
    assert encoded.shape == (2, 16)


def test_observation_encoder_keeps_legacy_text_and_vision_path_by_default(monkeypatch):
    monkeypatch.setattr(observation_encoder_module, "create_vision_encoder", lambda config, pretrained=True: FakeVisionEncoder())
    monkeypatch.setattr(observation_encoder_module, "CLIPTextEncoder", FakeTextEncoder)

    cfg = MultiTaskDiTConfig(n_obs_steps=2)
    cfg.transformer.hidden_dim = 8
    cfg.input_features = {
        "observation.image.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }

    encoder = observation_encoder_module.ObservationEncoder(cfg, load_pretrained_backbones=False)

    assert encoder.multimodal_encoder is None
    assert encoder.text_encoder is not None
    assert encoder.conditioning_dim == 30


def test_create_multimodal_encoder_returns_pooled_hf_encoder(monkeypatch):
    monkeypatch.setattr(observation_encoder_module, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(observation_encoder_module, "AutoModelForImageTextToText", FakeHFMultimodalModel)

    config = PooledMultimodalEncoderConfig(model="fake-qwen", output_dim=7, freeze_backbone=True)

    encoder = observation_encoder_module.create_multimodal_encoder(config, pretrained=True)

    assert isinstance(encoder, observation_encoder_module.PooledHuggingFaceMultimodalEncoder)
    assert encoder.get_output_dim() == 7
    assert not encoder.model.backbone_weight.requires_grad
    assert encoder.projection.weight.requires_grad


def test_pooled_multimodal_encoder_forward_pools_hidden_states(monkeypatch):
    monkeypatch.setattr(observation_encoder_module, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(observation_encoder_module, "AutoModelForImageTextToText", FakeHFMultimodalModel)

    config = PooledMultimodalEncoderConfig(model="fake-qwen", output_dim=5, freeze_backbone=True, max_text_length=32)

    encoder = observation_encoder_module.create_multimodal_encoder(config, pretrained=True)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(5))
        encoder.projection.bias.zero_()

    batch = {
        "observation.state": torch.zeros((2, 2, 3)),
        "observation.images": torch.zeros((2, 2, 2, 3, 8, 8)),
        "task": ["pick", "place"],
    }

    features = encoder(batch)

    assert features.shape == (2, 2, 5)
    assert torch.allclose(features[0, 0], torch.full((5,), 2.0))
    assert torch.allclose(features[1, 1], torch.full((5,), 5.0))
    assert FakeProcessor.last_call["text"] == ["pick", "pick", "place", "place"]
    assert len(FakeProcessor.last_call["images"]) == 4
    assert all(len(sample_images) == 2 for sample_images in FakeProcessor.last_call["images"])
    assert FakeProcessor.last_call["max_length"] == 32


def test_pooled_multimodal_encoder_uses_chat_template_for_qwen_processors(monkeypatch):
    monkeypatch.setattr(observation_encoder_module, "AutoProcessor", FakeChatTemplateProcessor)
    monkeypatch.setattr(observation_encoder_module, "AutoModelForImageTextToText", FakeHFMultimodalModel)

    config = PooledMultimodalEncoderConfig(model="Qwen/Qwen3-VL-4B-Instruct", output_dim=5, freeze_backbone=True)

    encoder = observation_encoder_module.create_multimodal_encoder(config, pretrained=True)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.eye(5))
        encoder.projection.bias.zero_()

    batch = {
        "observation.state": torch.zeros((2, 2, 3)),
        "observation.images": torch.zeros((2, 2, 2, 3, 8, 8)),
        "task": ["pick", "place"],
    }

    features = encoder(batch)

    messages = FakeChatTemplateProcessor.last_call["messages"]
    assert features.shape == (2, 2, 5)
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert [item["type"] for item in messages[0]["content"]] == ["image", "image", "text"]
    assert messages[0]["content"][-1]["text"] == "pick"
    assert messages[-1]["content"][-1]["text"] == "place"
    assert FakeChatTemplateProcessor.last_call["add_generation_prompt"] is False
    assert FakeChatTemplateProcessor.last_call["processor_kwargs"] == {"padding": True}
