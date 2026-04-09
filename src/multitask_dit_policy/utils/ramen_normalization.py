"""Ramen-style per-timestep percentile normalization.

Normalizes state and action features using 2nd/98th percentile statistics
computed per feature dimension and per timestep.  6D rotation dimensions
are passed through unchanged.

Reference: Ramen paper normalization procedure.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from multitask_dit_policy.utils.configuration import DatasetSchema, SchemaEntry
from multitask_dit_policy.utils.rotation import convert_eef_pose

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


def _bulk_read_columns(
    dataset,
    keys: list[str],
) -> dict[str, Tensor]:
    """Read numerical columns directly from parquet via hf_dataset, skipping video decoding.

    Supports both single-dataset (LeRobotDataset) and multi-dataset
    (WrappedRobotDataset with _datasets list) wrappers.
    """
    inner_datasets = getattr(dataset, "_datasets", [dataset])

    per_key: dict[str, list[Tensor]] = {k: [] for k in keys}
    for ds in inner_datasets:
        hf = ds.hf_dataset
        available = set(hf.column_names)
        cols_to_read = [k for k in keys if k in available]
        if not cols_to_read:
            continue
        subset = hf.select_columns(cols_to_read)
        for k in cols_to_read:
            per_key[k].append(torch.tensor(np.array(subset[k]), dtype=torch.float32))

    return {k: torch.cat(v, dim=0) for k, v in per_key.items() if v}


def _assemble_bulk(
    data: dict[str, Tensor],
    entries: list[SchemaEntry],
) -> Tensor:
    """Assemble a feature vector from bulk-read columns, applying RPY->rot6d."""
    parts: list[Tensor] = []
    for entry in entries:
        val = data.get(entry.key)
        if val is None:
            continue
        if entry.convert_rotation:
            val = convert_eef_pose(val)
        parts.append(val)
    return torch.cat(parts, dim=-1)


def compute_ramen_stats(
    dataset,
    schema: DatasetSchema,
    norm_mask: Tensor,
    cache_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Compute per-timestep q02/q98 stats for Ramen normalization.

    Reads numerical columns directly from the underlying parquet files
    (via hf_dataset), completely bypassing video decoding.  This is
    orders of magnitude faster than iterating dataset[i] which decodes
    a video frame per sample.

    Results are cached to ``cache_path`` if provided.

    Returns:
        Dict with keys obs_q02, obs_q98, action_q02, action_q98,
        each of shape (T, D), and norm_mask.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        logging.info(f"Loading cached Ramen stats from {cache_path}")
        return torch.load(cache_path, map_location=device, weights_only=True)

    all_keys = list({e.key for e in schema.state + schema.action})
    n_frames = len(dataset)
    logging.info(
        f"Computing Ramen stats — bulk-reading {n_frames} frames, "
        f"columns: {all_keys}"
    )

    data = _bulk_read_columns(dataset, all_keys)
    logging.info("Columns loaded, assembling vectors + RPY→rot6d...")

    all_obs = _assemble_bulk(data, schema.state)          # (N, state_dim)
    all_act = _assemble_bulk(data, schema.action)         # (N, action_dim)

    # Delta actions on shared prefix
    shared = min(all_obs.shape[-1], all_act.shape[-1])
    m = norm_mask[:shared]
    all_delta = all_act.clone()
    all_delta[..., :shared][..., m] = (
        all_act[..., :shared][..., m] - all_obs[..., :shared][..., m]
    )

    # Unsqueeze T=1 dim to match per-timestep convention (N, 1, D)
    all_obs = all_obs.unsqueeze(1)
    all_delta = all_delta.unsqueeze(1)

    logging.info("Computing percentiles...")
    stats = {
        "obs_q02": torch.quantile(all_obs.float(), 0.02, dim=0),
        "obs_q98": torch.quantile(all_obs.float(), 0.98, dim=0),
        "action_q02": torch.quantile(all_delta.float(), 0.02, dim=0),
        "action_q98": torch.quantile(all_delta.float(), 0.98, dim=0),
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
