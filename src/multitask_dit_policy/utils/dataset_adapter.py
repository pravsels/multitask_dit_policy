"""Adapter for LeRobot datasets with sub-features and Ramen-style transforms.

Handles datasets where state and action are split into sub-features
(e.g., observation.state.pos, observation.state.eef_pose) by selecting,
converting RPY -> 6D rotation, computing delta actions, and concatenating
into the flat 17D vectors the model expects.

17D layout: [joint_pos(7), eef_xyz(3), rot6d(6), gripper(1)]
"""

import torch
from torch import Tensor
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

from multitask_dit_policy.utils.rotation import convert_eef_pose

# Default sub-feature keys for pos + eef_pose datasets
DEFAULT_STATE_KEYS = ["observation.state.pos", "observation.state.eef_pose"]
DEFAULT_ACTION_KEYS = ["action.pos", "action.eef_pose"]

# After RPY->6D conversion: joint_pos(7) + eef_xyz(3) + rot6d(6) + gripper(1) = 17
EEF_CONVERTED_DIM = 10  # eef_pose 7D -> 10D after RPY->6D
ROT6D_START = 10  # index in the 17D vector where 6D rotation starts
ROT6D_END = 16    # index where 6D rotation ends (exclusive)

# Raw dimensions for known sub-feature keys, used as fallback when the
# features dict doesn't contain granular sub-feature entries (e.g. when
# using robocandywrapper's combined .meta.features for multi-datasets).
KNOWN_RAW_DIMS: dict[str, int] = {
    "observation.state.pos": 7,
    "observation.state.eef_pose": 7,
    "action.pos": 7,
    "action.eef_pose": 7,
}


def detect_sub_features(features: dict) -> tuple[list[str], list[str]]:
    """Auto-detect state and action sub-feature keys from dataset metadata.

    Returns (state_keys, action_keys): sorted lists of feature keys to concatenate.
    Empty lists if the dataset already has flat observation.state / action.
    """
    state_keys = []
    action_keys = []

    if OBS_STATE not in features:
        state_keys = sorted(
            k for k in features
            if k.startswith(f"{OBS_STATE}.") and features[k].get("dtype") != "video"
        )

    if ACTION not in features:
        action_keys = sorted(
            k for k in features
            if k.startswith(f"{ACTION}.") and features[k].get("dtype") != "video"
        )

    return state_keys, action_keys


def select_default_keys(
    state_keys: list[str],
    action_keys: list[str],
) -> tuple[list[str], list[str]]:
    """Narrow detected sub-features to pos + eef_pose only (if available)."""
    selected_state = [k for k in DEFAULT_STATE_KEYS if k in state_keys] or state_keys
    selected_action = [k for k in DEFAULT_ACTION_KEYS if k in action_keys] or action_keys
    return selected_state, selected_action


def compute_adapted_features(
    features: dict,
    state_keys: list[str],
    action_keys: list[str],
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """Build input_features and output_features dicts with 17D shapes.

    Accounts for RPY -> 6D expansion: eef_pose contributes 10D (not 7D).
    """
    input_features = {}
    output_features = {}

    for key, ft in features.items():
        if key == "index":
            continue
        shape = tuple(ft["shape"])
        if key.startswith(OBS_IMAGE):
            h, w, c = shape
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))

    if state_keys:
        total_dim = _compute_total_dim(features, state_keys)
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(total_dim,))
    elif OBS_STATE in features:
        shape = tuple(features[OBS_STATE]["shape"])
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=shape)

    if action_keys:
        total_dim = _compute_total_dim(features, action_keys)
        output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(total_dim,))
    elif ACTION in features:
        shape = tuple(features[ACTION]["shape"])
        output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=shape)

    return input_features, output_features


def _compute_total_dim(features: dict | None, keys: list[str]) -> int:
    """Sum dimensions, accounting for RPY->6D expansion on eef_pose keys.

    Falls back to KNOWN_RAW_DIMS when the features dict doesn't contain
    granular sub-feature entries (multi-dataset combined metadata).
    """
    total = 0
    for k in keys:
        if features and k in features:
            raw_dim = features[k]["shape"][0]
        elif k in KNOWN_RAW_DIMS:
            raw_dim = KNOWN_RAW_DIMS[k]
        else:
            raise KeyError(
                f"Unknown sub-feature {k!r}: not in metadata and no known dimension. "
                f"Add it to KNOWN_RAW_DIMS or pass a features dict that contains it."
            )
        if k.endswith(".eef_pose"):
            total += EEF_CONVERTED_DIM  # 7 -> 10 after RPY->6D
        else:
            total += raw_dim
    return total


def adapt_batch(
    batch: dict[str, Tensor],
    state_keys: list[str],
    action_keys: list[str],
    norm_mask: Tensor,
) -> dict[str, Tensor]:
    """Transform a batch: select sub-features, RPY->6D, delta actions, concatenate.

    Args:
        batch: raw batch from DataLoader with sub-feature keys.
        state_keys: e.g. ["observation.state.pos", "observation.state.eef_pose"]
        action_keys: e.g. ["action.pos", "action.eef_pose"]
        norm_mask: (17,) bool tensor — True for dims that get delta'd.

    Returns:
        Batch with observation.state (B, n_obs, 17) and action (B, horizon, 17).
    """
    adapted = dict(batch)

    # --- Build observation.state (17D) ---
    obs_parts = []
    for k in state_keys:
        val = adapted.pop(k, None)
        if val is None:
            continue
        if k.endswith(".eef_pose"):
            val = convert_eef_pose(val)
        obs_parts.append(val)
    if obs_parts:
        adapted[OBS_STATE] = torch.cat(obs_parts, dim=-1)

    # --- Build action (17D) ---
    act_parts = []
    for k in action_keys:
        val = adapted.pop(k, None)
        if val is None:
            continue
        if k.endswith(".eef_pose"):
            val = convert_eef_pose(val)
        act_parts.append(val)
    if act_parts:
        adapted[ACTION] = torch.cat(act_parts, dim=-1)

    # --- Delta actions (all dims except 6D rotation) ---
    if OBS_STATE in adapted and ACTION in adapted:
        current_obs = adapted[OBS_STATE][:, -1:, :]  # (B, 1, 17)
        m = norm_mask.to(adapted[ACTION].device)
        adapted[ACTION][..., m] = adapted[ACTION][..., m] - current_obs[..., m]

    # --- Merge _is_pad flags (take the first, identical temporal structure) ---
    for sub_keys, target in [(state_keys, OBS_STATE), (action_keys, ACTION)]:
        pad_key = f"{target}_is_pad"
        first_pad = None
        for k in sub_keys:
            pk = f"{k}_is_pad"
            if pk in adapted:
                if first_pad is None:
                    first_pad = adapted.pop(pk)
                else:
                    adapted.pop(pk)
        if first_pad is not None:
            adapted[pad_key] = first_pad

    return adapted
