"""Adapter for LeRobot datasets with configurable sub-features.

Handles datasets where state and action are split into sub-features
by selecting, optionally converting RPY -> 6D rotation, computing
delta actions, and concatenating into flat vectors the model expects.

The key layout is declared per-task via DatasetSchema (YAML config),
so different datasets can have different key names and dimensions.
"""

import torch
from torch import Tensor
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

from multitask_dit_policy.utils.configuration import DatasetSchema, SchemaEntry
from multitask_dit_policy.utils.rotation import convert_eef_pose


def assemble_vector(
    tensors: dict[str, Tensor],
    entries: list[SchemaEntry],
    *,
    pop: bool = True,
) -> Tensor | None:
    """Concatenate sub-feature tensors, applying RPY->rot6d where declared.

    Args:
        tensors: mapping of key -> tensor.
        entries: schema entries declaring keys and conversion.
        pop: if True, remove consumed keys from *tensors* in-place.

    Returns:
        Concatenated tensor, or None if no matching keys found.
    """
    parts: list[Tensor] = []
    for entry in entries:
        val = tensors.pop(entry.key, None) if pop else tensors.get(entry.key)
        if val is None:
            continue
        if entry.convert_rotation:
            val = convert_eef_pose(val)
        parts.append(val)
    if not parts:
        return None
    return torch.cat(parts, dim=-1)


def compute_adapted_features(
    features: dict,
    schema: DatasetSchema,
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """Build input_features and output_features from dataset metadata + schema.

    Image features are discovered from the *features* dict.
    State/action dims come from the schema (no sub-feature lookup needed).
    """
    input_features: dict[str, PolicyFeature] = {}
    output_features: dict[str, PolicyFeature] = {}

    for key, ft in features.items():
        if key == "index":
            continue
        shape = tuple(ft["shape"])
        if key.startswith(OBS_IMAGE):
            h, w, c = shape
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))

    if schema.state:
        input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE, shape=(schema.state_dim,),
        )
    elif OBS_STATE in features:
        shape = tuple(features[OBS_STATE]["shape"])
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=shape)

    if schema.action:
        output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(schema.action_dim,),
        )
    elif ACTION in features:
        shape = tuple(features[ACTION]["shape"])
        output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=shape)

    return input_features, output_features


def adapt_batch(
    batch: dict[str, Tensor],
    schema: DatasetSchema,
    norm_mask: Tensor,
) -> dict[str, Tensor]:
    """Transform a batch: assemble sub-features, RPY->6D, delta actions.

    Args:
        batch: raw batch from DataLoader with sub-feature keys.
        schema: dataset schema declaring keys and conversions.
        norm_mask: bool tensor — True for dims that get delta'd.
            Length must be >= min(state_dim, action_dim).

    Returns:
        Batch with flat observation.state and action tensors.
    """
    adapted = dict(batch)

    obs = assemble_vector(adapted, schema.state, pop=True)
    if obs is not None:
        adapted[OBS_STATE] = obs

    act = assemble_vector(adapted, schema.action, pop=True)
    if act is not None:
        adapted[ACTION] = act

    # Delta actions: subtract on the shared prefix of state/action dims.
    if OBS_STATE in adapted and ACTION in adapted:
        current_obs = adapted[OBS_STATE][:, -1:, :]
        shared = min(current_obs.shape[-1], adapted[ACTION].shape[-1])
        m = norm_mask[:shared].to(adapted[ACTION].device)
        adapted[ACTION][..., :shared][..., m] = (
            adapted[ACTION][..., :shared][..., m] - current_obs[..., :shared][..., m]
        )

    # Merge _is_pad flags (take the first per category).
    for entries, target in [(schema.state, OBS_STATE), (schema.action, ACTION)]:
        pad_key = f"{target}_is_pad"
        first_pad = None
        for entry in entries:
            pk = f"{entry.key}_is_pad"
            if pk in adapted:
                if first_pad is None:
                    first_pad = adapted.pop(pk)
                else:
                    adapted.pop(pk)
        if first_pad is not None:
            adapted[pad_key] = first_pad

    return adapted
