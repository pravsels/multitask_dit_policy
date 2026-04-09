"""Ramen-style per-timestep percentile normalization.

Normalizes state and action features using 2nd/98th percentile statistics
computed per feature dimension and per timestep.  6D rotation dimensions
are passed through unchanged.

Reference: Ramen paper normalization procedure.
"""

import logging
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from multitask_dit_policy.utils.configuration import DatasetSchema
from multitask_dit_policy.utils.dataset_adapter import assemble_vector

# Keys that receive ImageNet MEAN_STD normalization (unchanged from before)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def build_norm_mask(state_dim: int, rot6d_start: int, rot6d_end: int) -> Tensor:
    """Build a boolean mask: True for dims that get delta + normalization.

    Args:
        state_dim: total dimensionality of the state/action vector (e.g. 17).
        rot6d_start: first index of the 6D rotation slice (inclusive).
        rot6d_end: last index of the 6D rotation slice (exclusive).

    Returns:
        (state_dim,) bool tensor.
    """
    mask = torch.ones(state_dim, dtype=torch.bool)
    mask[rot6d_start:rot6d_end] = False
    return mask


def ramen_normalize(x: Tensor, q02: Tensor, q98: Tensor, norm_mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Apply Ramen percentile normalization.

    y = clip(2 * (x - q02) / (q98 - q02) - 1,  -1.5,  1.5)

    Only the dimensions where norm_mask is True are normalized.
    Dimensions where norm_mask is False are passed through unchanged.

    Args:
        x: (..., T, D) tensor to normalize.
        q02: (T, D) or (D,) 2nd percentile stats.
        q98: (T, D) or (D,) 98th percentile stats.
        norm_mask: (D,) bool tensor.
    """
    y = x.clone()
    m = norm_mask
    denom = q98[..., m] - q02[..., m]
    denom = torch.where(denom.abs() < eps, torch.tensor(eps, device=x.device, dtype=x.dtype), denom)
    y[..., m] = (2 * (x[..., m] - q02[..., m]) / denom - 1).clamp(-1.5, 1.5)
    return y


def ramen_unnormalize(y: Tensor, q02: Tensor, q98: Tensor, norm_mask: Tensor) -> Tensor:
    """Inverse of ramen_normalize (without clipping — lossy for values outside [-1.5, 1.5]).

    x = (y + 1) / 2 * (q98 - q02) + q02

    Args:
        y: (..., T, D) normalized tensor.
        q02: (T, D) or (D,) stats.
        q98: (T, D) or (D,) stats.
        norm_mask: (D,) bool tensor.
    """
    x = y.clone()
    m = norm_mask
    x[..., m] = (y[..., m] + 1) / 2 * (q98[..., m] - q02[..., m]) + q02[..., m]
    return x


def _transform_sample(
    sample: dict,
    schema: DatasetSchema,
    norm_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Transform a single dataset sample into (obs, delta_action).

    Assembles state/action from schema entries (with RPY->6D where declared),
    then computes delta actions on the shared dim prefix.
    """
    obs = assemble_vector(sample, schema.state, pop=False)
    act = assemble_vector(sample, schema.action, pop=False)

    if obs is None or act is None:
        raise ValueError("Schema entries produced no tensors from sample")

    shared = min(obs.shape[-1], act.shape[-1])
    current_obs = obs[-1:]
    delta_act = act.clone()
    m = norm_mask[:shared]
    delta_act[..., :shared][..., m] = act[..., :shared][..., m] - current_obs[..., :shared][..., m]

    return obs, delta_act


def compute_ramen_stats(
    dataset,
    schema: DatasetSchema,
    norm_mask: Tensor,
    cache_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Compute per-timestep q02/q98 stats for Ramen normalization.

    Iterates the full dataset, transforms each sample using the schema
    (assemble + RPY->6D + delta), and computes 2nd/98th percentile
    per dimension per timestep.

    Results are cached to ``cache_path`` if provided.

    Returns:
        Dict with keys obs_q02, obs_q98, action_q02, action_q98,
        each of shape (T, D), and norm_mask.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        logging.info(f"Loading cached Ramen stats from {cache_path}")
        return torch.load(cache_path, map_location=device, weights_only=True)

    logging.info(f"Computing Ramen stats over {len(dataset)} samples...")

    all_obs = []
    all_delta_actions = []

    for i in tqdm(range(len(dataset)), desc="Ramen stats"):
        sample = dataset[i]
        obs, delta_act = _transform_sample(sample, schema, norm_mask)
        all_obs.append(obs)
        all_delta_actions.append(delta_act)

    all_obs = torch.stack(all_obs)            # (N, n_obs, 17)
    all_delta_actions = torch.stack(all_delta_actions)  # (N, horizon, 17)

    stats = {
        "obs_q02": torch.quantile(all_obs.float(), 0.02, dim=0),
        "obs_q98": torch.quantile(all_obs.float(), 0.98, dim=0),
        "action_q02": torch.quantile(all_delta_actions.float(), 0.02, dim=0),
        "action_q98": torch.quantile(all_delta_actions.float(), 0.98, dim=0),
        "norm_mask": norm_mask,
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, cache_path)
        logging.info(f"Saved Ramen stats to {cache_path}")

    return {k: v.to(device) for k, v in stats.items()}


def ramen_normalize_batch(
    batch: dict,
    ramen_stats: dict[str, Tensor],
    norm_mask: Tensor,
) -> dict:
    """Normalize a full training batch using Ramen normalization.

    - observation.state and action: Ramen percentile normalization (6D rot exempt)
    - observation.images.*: ImageNet MEAN_STD normalization
    - Other keys (task, metadata): passed through unchanged

    norm_mask is sliced to match state_dim / action_dim automatically
    (they may differ).
    """
    normalized = {}

    for key, val in batch.items():
        if key == "observation.state":
            m = norm_mask[: val.shape[-1]]
            normalized[key] = ramen_normalize(
                val, ramen_stats["obs_q02"], ramen_stats["obs_q98"], m,
            )
        elif key == "action":
            m = norm_mask[: val.shape[-1]]
            normalized[key] = ramen_normalize(
                val, ramen_stats["action_q02"], ramen_stats["action_q98"], m,
            )
        elif key.startswith("observation.image") and isinstance(val, Tensor) and val.is_floating_point():
            mean = _IMAGENET_MEAN.to(val.device, val.dtype)
            std = _IMAGENET_STD.to(val.device, val.dtype)
            shape = [1] * (val.ndim - 3) + [3, 1, 1]
            normalized[key] = (val - mean.view(*shape)) / std.view(*shape)
        else:
            normalized[key] = val

    return normalized
