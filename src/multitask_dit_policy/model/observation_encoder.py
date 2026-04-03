#!/usr/bin/env python

# Copyright 2025 Bryson Jones. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observation encoding for Multi-Task DiT policy.

Handles vision encoding, text encoding, robot state, and environment state.
"""

from abc import ABC, abstractmethod

import einops
import timm
import torch
import torch.nn as nn
import torchvision
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
)


class BaseVisionEncoder(ABC):
    """Abstract base class for vision encoders."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to feature maps."""
        pass

    @abstractmethod
    def get_output_shape(self) -> tuple:
        """Get the output shape (C', H', W')."""
        pass


class BaseMultimodalEncoder(ABC):
    """Abstract base class for unified multimodal encoders."""

    @abstractmethod
    def forward(self, batch: dict) -> Tensor:
        """Encode batch observations into timestep-aligned multimodal features."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the per-timestep multimodal feature dimension."""
        pass


class DinoV3Encoder(nn.Module, BaseVisionEncoder):
    """DinoV3 vision encoder using the CLS token for global image representation."""

    def __init__(self, config, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.model_name = config.backbone

        # Create the timm model
        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        self.num_non_spatial_tokens = 5  # 1 CLS + 4 register
        self.embed_dim = self.model.embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to feature maps."""
        # Extract all features
        features = self.model.forward_features(x)  # (B, total_tokens, embed_dim)

        # Use only the CLS token (first token)
        cls_token = features[:, 0]  # (B, embed_dim)
        b, embed_dim = cls_token.shape

        # Reshape to spatial format (B, C, H, W) with H=W=1 for compatibility
        cls_features = cls_token.reshape(b, embed_dim, 1, 1)
        return cls_features

    def get_output_shape(self) -> tuple:
        return (self.embed_dim, 1, 1)


class CLIPEncoder(nn.Module, BaseVisionEncoder):
    """CLIP vision encoder using the CLS token for global image representation."""

    def __init__(self, config, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.model_name = config.backbone

        # Create the timm model
        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, we want features
        )

        # CLIP models have 1 CLS token (no register tokens like DinoV3)
        self.num_non_spatial_tokens = 1

        # Get embed_dim from model config
        self.embed_dim = self.model.embed_dim

    def forward(self, x: Tensor) -> Tensor:
        """Encode RGB image to CLS token.

        Preprocessing (resize, crop) is handled by ObservationEncoder
        """
        # Extract all features
        features = self.model.forward_features(x)  # (B, total_tokens, embed_dim)

        # Use only the CLS token (first token)
        cls_token = features[:, 0]  # (B, embed_dim)
        b, embed_dim = cls_token.shape

        # Reshape to spatial format (B, C, H, W) with H=W=1 for compatibility
        cls_features = cls_token.reshape(b, embed_dim, 1, 1)
        return cls_features

    def get_output_shape(self) -> tuple:
        return (self.embed_dim, 1, 1)


def create_vision_encoder(config, pretrained: bool = True) -> BaseVisionEncoder:
    """Create a vision encoder from config.

    Supports any timm model with "clip" or "dinov3" in the backbone name.
    The encoder type is automatically detected based on the backbone name.
    """
    backbone_name = config.backbone.lower()

    # Check if it's a CLIP model
    if "clip" in backbone_name:
        return CLIPEncoder(config, pretrained=pretrained)

    # Check if it's a DinoV3 model
    elif "dinov3" in backbone_name:
        return DinoV3Encoder(config, pretrained=pretrained)

    else:
        raise ValueError(
            f"Unsupported vision backbone: {config.backbone}. "
            f"Currently supported: any timm model with 'dinov3' or 'clip' in the name"
        )


# Registry for easy extension
VISION_ENCODER_REGISTRY: dict[str, type] = {
    "dinov3": DinoV3Encoder,
    "clip": CLIPEncoder,
}


def register_vision_encoder(name: str, encoder_class: type):
    """Register a new vision encoder type."""
    VISION_ENCODER_REGISTRY[name] = encoder_class


def get_registered_encoders() -> dict[str, type]:
    """Get all registered vision encoder types."""
    return VISION_ENCODER_REGISTRY.copy()


class CLIPTextEncoder(nn.Module):
    """Supports any HuggingFace CLIP model. The encoder weights are frozen,
    and a learnable projection layer maps the CLIP embeddings to the desired dimension.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", projection_dim: int = 512, pretrained: bool = True):
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim

        # Load CLIP text encoder and tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        if pretrained:
            self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        else:
            text_config = CLIPTextConfig.from_pretrained(model_name)
            self.text_encoder = CLIPTextModel(text_config)

        # Freeze all CLIP text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_embed_dim = self.text_encoder.config.hidden_size

        # Learnable projection layer (always present, only trainable component)
        self.projection = nn.Linear(self.text_embed_dim, projection_dim)

    def forward(self, text: str | list[str]) -> Tensor:
        """Encode text to feature vectors.

        Args:
            text: Single string or list of strings

        Returns:
            Text features of shape (B, projection_dim)
        """
        # handle single string input
        if isinstance(text, str):
            text = [text]

        text_inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        text_inputs = {k: v.to(next(self.parameters()).device) for k, v in text_inputs.items()}

        # encode text through CLIP (frozen)
        with torch.no_grad():
            outputs = self.text_encoder(**text_inputs)
            # Extract pooled output (EOS token embedding)
            clip_features = outputs.pooler_output  # (B, text_embed_dim)

        # project to desired dimension (trainable)
        projected_features = self.projection(clip_features)  # (B, projection_dim)

        return projected_features


class PooledHuggingFaceMultimodalEncoder(nn.Module, BaseMultimodalEncoder):
    """Pooled multimodal encoder backed by a Hugging Face image-text model."""

    def __init__(self, config, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.output_dim = config.output_dim
        self.processor = AutoProcessor.from_pretrained(config.model)
        if pretrained:
            self.model = AutoModelForImageTextToText.from_pretrained(config.model)
        else:
            model_config = AutoConfig.from_pretrained(config.model)
            self.model = AutoModelForImageTextToText.from_config(model_config)

        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        hidden_size = self._get_hidden_size()
        self.projection = nn.Linear(hidden_size, self.output_dim)

    def _get_hidden_size(self) -> int:
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            return self.model.config.text_config.hidden_size
        raise ValueError(f"Could not determine hidden size for multimodal model config: {self.model.config}")

    def get_output_dim(self) -> int:
        return self.output_dim

    def _expand_text_per_timestep(self, task: str | list[str] | None, n_obs_steps: int, batch_size: int) -> list[str]:
        if task is None:
            task_text = [""] * batch_size
        elif isinstance(task, str):
            task_text = [task] * batch_size
        else:
            task_text = list(task)

        if len(task_text) != batch_size:
            raise ValueError(f"Expected {batch_size} task strings, got {len(task_text)}")

        return [text for text in task_text for _ in range(n_obs_steps)]

    def _convert_images_for_processor(self, images: Tensor) -> list[list]:
        images_cpu = images.detach().cpu()
        return [
            [torchvision.transforms.functional.to_pil_image(image) for image in sample_images]
            for sample_images in images_cpu
        ]

    def _build_chat_template_messages(self, text: list[str], images: list[list]) -> list[dict]:
        messages = []
        for sample_text, sample_images in zip(text, images, strict=True):
            content = [{"type": "image", "image": image} for image in sample_images]
            content.append({"type": "text", "text": sample_text})
            messages.append({"role": "user", "content": content})
        return messages

    def _prepare_processor_inputs(self, text: list[str], images: list[list]) -> dict:
        if hasattr(self.processor, "apply_chat_template"):
            messages = self._build_chat_template_messages(text, images)
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )

        return self.processor(
            text=text,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        )

    def _pool_hidden_states(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        if attention_mask is None or attention_mask.shape[:2] != hidden_states.shape[:2]:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def forward(self, batch: dict) -> Tensor:
        images = batch.get(OBS_IMAGES)
        if images is None:
            raise ValueError("Multimodal encoder requires observation images")
        if len(images.shape) != 6:
            raise ValueError(f"Expected observation images with shape (B, T, N, C, H, W), got {tuple(images.shape)}")

        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        flat_text = self._expand_text_per_timestep(batch.get("task"), n_obs_steps=n_obs_steps, batch_size=batch_size)
        flat_images = einops.rearrange(images, "b s n c h w -> (b s) n c h w")
        processor_images = self._convert_images_for_processor(flat_images)

        processor_inputs = self._prepare_processor_inputs(flat_text, processor_images)

        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in processor_inputs.items()
        }
        outputs = self.model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        pooled = self._pool_hidden_states(hidden_states, model_inputs.get("attention_mask"))
        pooled = pooled.to(self.projection.weight.dtype)
        projected = self.projection(pooled)
        return einops.rearrange(projected, "(b s) f -> b s f", b=batch_size, s=n_obs_steps)


def create_text_encoder(config, projection_dim: int, pretrained: bool = True) -> nn.Module:
    """Create a legacy text encoder from config."""
    model_name = getattr(config, "model", "")
    if "clip" in model_name.lower():
        return CLIPTextEncoder(
            model_name=model_name,
            projection_dim=projection_dim,
            pretrained=pretrained,
        )

    raise ValueError(f"Unsupported text encoder config: {config}")


def create_multimodal_encoder(config, pretrained: bool = True) -> BaseMultimodalEncoder:
    """Create a unified multimodal encoder from config.

    Stage 1 only establishes the integration seam. Concrete encoders such as
    Qwen-backed implementations will plug into this factory later.
    """
    from multitask_dit_policy.utils.configuration import PooledMultimodalEncoderConfig

    if isinstance(config, PooledMultimodalEncoderConfig):
        return PooledHuggingFaceMultimodalEncoder(config, pretrained=pretrained)

    raise NotImplementedError(
        f"Multimodal encoder type '{getattr(config, 'type', type(config).__name__)}' is not implemented yet"
    )


class ObservationEncoder(nn.Module):
    """Handles all observation processing for the conditioning vector."""

    def __init__(self, config, load_pretrained_backbones: bool = True):
        super().__init__()
        self.config = config
        vision_config = config.observation_encoder.vision
        self.text_dim = config.transformer.hidden_dim
        self.multimodal_encoder = None

        self._setup_preprocessing(vision_config)

        if config.image_features:
            self.num_cameras = len(config.image_features)
            self.camera_names = list(config.image_features.keys())  # Preserve ordering
        else:
            self.camera_names = []
            self.num_cameras = 0

        self.vision_encoder = None
        self.vision_encoders = None
        self.text_encoder = None

        if config.observation_encoder.uses_multimodal_encoder:
            self.multimodal_encoder = create_multimodal_encoder(
                config.observation_encoder.multimodal,
                pretrained=load_pretrained_backbones,
            )
        elif config.image_features:
            if vision_config.use_separate_encoder_per_camera:
                self.vision_encoders = nn.ModuleList(
                    [create_vision_encoder(vision_config, pretrained=load_pretrained_backbones) for _ in self.camera_names]
                )
            else:
                self.vision_encoder = create_vision_encoder(vision_config, pretrained=load_pretrained_backbones)

        if hasattr(config, "robot_state_feature") and config.robot_state_feature:
            self.robot_state_dim = config.robot_state_feature.shape[0]
        else:
            self.robot_state_dim = 0

        if hasattr(config, "env_state_feature") and config.env_state_feature:
            self.env_state_dim = config.env_state_feature.shape[0]
        else:
            self.env_state_dim = 0

        if not config.observation_encoder.uses_multimodal_encoder:
            text_config = config.observation_encoder.text
            self.text_encoder = create_text_encoder(
                text_config,
                projection_dim=self.text_dim,
                pretrained=load_pretrained_backbones,
            )

        self._setup_vector_output()

    def _apply_preprocessing(self, images: Tensor) -> Tensor:
        """Apply preprocessing transforms to images."""
        if self.do_resize:
            images = self.resize(images)
        if self.do_crop:
            images = self.maybe_random_crop(images) if self.training else self.center_crop(images)

        return images

    def _setup_preprocessing(self, vision_config):
        """Setup image preprocessing transforms."""
        if vision_config.resize_shape is not None:
            self.do_resize = True
            self.resize = torchvision.transforms.Resize(
                size=vision_config.resize_shape,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            self.do_resize = False
        if vision_config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(vision_config.crop_shape)
            if vision_config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(vision_config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

    def _setup_vector_output(self):
        """Setup for vector output."""
        total_dim = 0

        if self.multimodal_encoder is not None:
            total_dim += self.multimodal_encoder.get_output_dim()

        # Vision features - get CLS token feature dimension
        elif self.vision_encoder is not None or self.vision_encoders is not None:
            encoder_to_check = self.vision_encoder or self.vision_encoders[0]

            # Get output shape from encoder (deterministic for CLS tokens)
            feature_map_shape = encoder_to_check.get_output_shape()
            c, h, w = feature_map_shape
            spatial_feature_dim = c * h * w  # For CLS token: embed_dim * 1 * 1 = embed_dim

            total_dim += spatial_feature_dim * self.num_cameras

        # State features
        total_dim += self.robot_state_dim
        total_dim += self.env_state_dim

        # Text features
        if self.text_encoder is not None:
            total_dim += self.text_dim

        # Account for temporal stacking
        self.conditioning_dim = total_dim * self.config.n_obs_steps

    def encode(self, batch: dict) -> Tensor:
        """Encode observations to vector format."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        conditioning_feats = []

        conditioning_feats.append(batch[OBS_STATE])

        if self.multimodal_encoder is not None:
            multimodal_batch = dict(batch)

            if OBS_IMAGES in batch:
                images = batch[OBS_IMAGES]
                if len(images.shape) == 5:
                    images = images.unsqueeze(1)

                images_shape = images.shape
                images_flat = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
                images_flat = self._apply_preprocessing(images_flat)
                multimodal_batch[OBS_IMAGES] = einops.rearrange(
                    images_flat,
                    "(b s n) c h w -> b s n c h w",
                    b=images_shape[0],
                    s=images_shape[1],
                    n=images_shape[2],
                )

            multimodal_features = self.multimodal_encoder(multimodal_batch)
            conditioning_feats.append(multimodal_features)

        elif self.vision_encoder is not None or self.vision_encoders is not None:
            images = batch[OBS_IMAGES]  # (B, n_obs_steps, num_cameras, C, H, W)

            # Handle case when n_obs=1 and time dimension might be squeezed
            if len(images.shape) == 5:
                # Shape is (B, N, C, H, W) - add time dimension
                images = images.unsqueeze(1)  # (B, 1, N, C, H, W)

            vision_config = self.config.observation_encoder.vision
            if vision_config.use_separate_encoder_per_camera:
                # Process each camera with its own encoder
                camera_features = []

                for cam_idx in range(self.num_cameras):
                    # Extract images for this camera: (B, n_obs_steps, C, H, W)
                    cam_images = images[:, :, cam_idx]

                    # Rearrange to: (B*n_obs_steps, C, H, W)
                    cam_images_flat = einops.rearrange(cam_images, "b s c h w -> (b s) c h w")

                    # Apply preprocessing
                    cam_images_flat = self._apply_preprocessing(cam_images_flat)

                    # Process with camera-specific encoder (direct index access)
                    cam_features = self.vision_encoders[cam_idx](cam_images_flat)

                    # Apply spatial vectorization (flatten CLS token features)
                    cam_visual_features = cam_features.flatten(start_dim=1)

                    # Reshape back: (B*n_obs_steps, feature_dim) → (B, n_obs_steps, feature_dim)
                    cam_features_reshaped = einops.rearrange(
                        cam_visual_features, "(b s) f -> b s f", b=batch_size, s=n_obs_steps
                    )
                    camera_features.append(cam_features_reshaped)

                # Concatenate features from all cameras: (B, n_obs_steps, total_feature_dim)
                img_features = torch.cat(camera_features, dim=-1)
                conditioning_feats.append(img_features)

            else:
                # Shared encoder for all cameras
                # Rearrange to: (B*n_obs_steps*num_cameras, C, H, W)
                images_flat = einops.rearrange(images, "b s n c h w -> (b s n) c h w")

                images_flat = self._apply_preprocessing(images_flat)

                visual_features = self.vision_encoder(images_flat).flatten(start_dim=1)

                # Reshape back and concatenate camera features
                # (B*n_obs_steps*num_cameras, feature_dim) → (B, n_obs_steps, num_cameras*feature_dim)
                img_features = einops.rearrange(
                    visual_features, "(b s n) f -> b s (n f)", b=batch_size, s=n_obs_steps, n=self.num_cameras
                )

                conditioning_feats.append(img_features)

        if self.env_state_dim > 0 and OBS_ENV_STATE in batch:
            conditioning_feats.append(batch[OBS_ENV_STATE])

        if self.text_encoder is not None and "task" in batch:
            text_features = self.text_encoder(batch["task"])  # (B, text_dim)
            # Expand across temporal dimension to match other features
            text_features = text_features.unsqueeze(1).expand(-1, n_obs_steps, -1)  # (B, T, text_dim)
            # print("Text features shape after unsqueeze and expand:", text_features.shape)
            conditioning_feats.append(text_features)

        # for vec in conditioning_feats:
        #     print(f"Conditioning feature shape: {vec.shape}")
        combined_features = torch.cat(conditioning_feats, dim=-1)  # (B, n_obs_steps, total_feature_dim)

        return combined_features.flatten(start_dim=1)  # (B, n_obs_steps * total_feature_dim)
