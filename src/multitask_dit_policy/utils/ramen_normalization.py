"""Ramen-style per-timestep percentile normalization.

Normalizes state and action features using 2nd/98th percentile statistics
computed per feature dimension and per timestep.  6D rotation dimensions
are passed through unchanged.

Reference: Ramen paper normalization procedure.
"""

import json
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

_RAMEN_STATS_FORMAT = "ramen_norm_stats"
_RAMEN_STATS_FORMAT_VERSION = 1


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


_DEFAULT_CLIP_VALUE = 1.5
"""Historical default for the Ramen normalize clamp.

Kept as the function-level default so callers that don't yet thread a
config-driven `clip_value` through still match the original behaviour.
The single source of truth for production code paths is
`MultiTaskDiTConfig.ramen_clip_value`.
"""


def _to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _ramen_stats_to_json_payload(ramen_stats: dict[str, Tensor]) -> dict:
    obs_q02 = ramen_stats["obs_q02"].detach().cpu()
    obs_q98 = ramen_stats["obs_q98"].detach().cpu()
    action_q02 = ramen_stats["action_q02"].detach().cpu()
    action_q98 = ramen_stats["action_q98"].detach().cpu()
    norm_mask = ramen_stats["norm_mask"].detach().cpu()

    return {
        "format": _RAMEN_STATS_FORMAT,
        "format_version": _RAMEN_STATS_FORMAT_VERSION,
        "metadata": {
            "horizon": _to_int(ramen_stats["horizon"]),
            "n_obs_steps": _to_int(ramen_stats["n_obs_steps"]),
            "state_dim": int(obs_q02.shape[-1]),
            "action_dim": int(action_q02.shape[-1]),
        },
        "norm_mask": norm_mask.tolist(),
        "state": {
            "q02": obs_q02.tolist(),
            "q98": obs_q98.tolist(),
        },
        "action": {
            "q02": action_q02.tolist(),
            "q98": action_q98.tolist(),
        },
    }


def _json_payload_to_ramen_stats(payload: dict, *, device: str = "cpu") -> dict[str, Tensor]:
    if payload.get("format") != _RAMEN_STATS_FORMAT:
        raise ValueError(f"Unsupported Ramen stats format: {payload.get('format')!r}")
    if payload.get("format_version") != _RAMEN_STATS_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Ramen stats format_version: {payload.get('format_version')!r}"
        )

    metadata = payload["metadata"]
    stats = {
        "obs_q02": torch.tensor(payload["state"]["q02"], dtype=torch.float32, device=device),
        "obs_q98": torch.tensor(payload["state"]["q98"], dtype=torch.float32, device=device),
        "action_q02": torch.tensor(payload["action"]["q02"], dtype=torch.float32, device=device),
        "action_q98": torch.tensor(payload["action"]["q98"], dtype=torch.float32, device=device),
        "norm_mask": torch.tensor(payload["norm_mask"], dtype=torch.bool, device=device),
        "horizon": torch.tensor(metadata["horizon"], dtype=torch.int64, device=device),
        "n_obs_steps": torch.tensor(metadata["n_obs_steps"], dtype=torch.int64, device=device),
    }
    return stats


def save_ramen_stats(ramen_stats: dict[str, Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".pt":
        torch.save(ramen_stats, path)
        return

    payload = _ramen_stats_to_json_payload(ramen_stats)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_ramen_stats(path: str | Path, *, device: str = "cpu") -> dict[str, Tensor]:
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return _json_payload_to_ramen_stats(payload, device=device)

    loaded = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in loaded.items()}


def ramen_normalize(
    x: Tensor,
    q02: Tensor,
    q98: Tensor,
    norm_mask: Tensor,
    eps: float = 1e-8,
    *,
    clip_value: float = _DEFAULT_CLIP_VALUE,
) -> Tensor:
    """Apply Ramen percentile normalization.

    y = clip((2 * (x - q02) / (q98 - q02)) - 1,  -clip_value,  +clip_value)

    Only the dimensions where norm_mask is True are normalized.
    Dimensions where norm_mask is False are passed through unchanged.

    Args:
        x: (..., T, D) tensor to normalize.
        q02: (T, D) or (D,) 2nd percentile stats.
        q98: (T, D) or (D,) 98th percentile stats.
        norm_mask: (D,) bool tensor.
        clip_value: positive saturation boundary in normalized space.
            Production code should pass ``MultiTaskDiTConfig.ramen_clip_value``
            so that the training-time clamp and the inference-time DDIM
            ``clip_sample_range`` stay in lockstep.
    """
    if clip_value <= 0:
        raise ValueError(f"clip_value must be positive, got {clip_value}")
    y = x.clone()
    m = norm_mask
    denom = q98[..., m] - q02[..., m]
    denom = torch.where(denom.abs() < eps, torch.tensor(eps, device=x.device, dtype=x.dtype), denom)
    y[..., m] = ((2 * (x[..., m] - q02[..., m]) / denom) - 1).clamp(-clip_value, clip_value)
    return y


def ramen_unnormalize(y: Tensor, q02: Tensor, q98: Tensor, norm_mask: Tensor) -> Tensor:
    """Inverse of ramen_normalize (no clip; lossy if y was clamped during normalize).

    x = (((y + 1) / 2) * (q98 - q02)) + q02

    Args:
        y: (..., T, D) normalized tensor.
        q02: (T, D) or (D,) stats.
        q98: (T, D) or (D,) stats.
        norm_mask: (D,) bool tensor.
    """
    x = y.clone()
    m = norm_mask
    x[..., m] = (((y[..., m] + 1) / 2) * (q98[..., m] - q02[..., m])) + q02[..., m]
    return x


def _bulk_read_columns(
    dataset,
    keys: list[str],
) -> dict[str, Tensor]:
    """Read numerical columns directly from parquet via hf_dataset, skipping video decoding.

    Supports both single-dataset (LeRobotDataset) and multi-dataset
    (WrappedRobotDataset with _datasets list) wrappers.
    """
    from multitask_dit_policy.utils.valid_indices import _unwrap_dataset

    dataset = _unwrap_dataset(dataset)
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


def _get_episode_lengths(dataset) -> list[int]:
    """Per-episode frame counts in the same row order as _bulk_read_columns."""
    from multitask_dit_policy.utils.valid_indices import _episode_bounds, _unwrap_dataset

    root = _unwrap_dataset(dataset)
    inner = getattr(root, "_datasets", [root])
    lengths: list[int] = []
    for ds in inner:
        ep_from, ep_to = _episode_bounds(ds)
        lengths.extend(int(ep_to[i] - ep_from[i]) for i in range(len(ep_from)))
    return lengths


def compute_ramen_stats(
    dataset,
    schema: DatasetSchema,
    norm_mask: Tensor,
    cache_path: str | Path | None = None,
    *,
    horizon: int,
    n_obs_steps: int,
    drop_n_last_frames: int = 0,
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
        cached = load_ramen_stats(cache_path, device=device)
        aq02 = cached.get("action_q02")
        cached_h = cached.get("horizon")
        cached_n = cached.get("n_obs_steps")
        if hasattr(cached_h, "item"):
            cached_h = int(cached_h.item())
        if hasattr(cached_n, "item"):
            cached_n = int(cached_n.item())
        ok = (
            aq02 is not None
            and aq02.ndim == 2
            and aq02.shape[0] == horizon
            and (cached_h is None or cached_h == horizon)
            and (cached_n is None or cached_n == n_obs_steps)
        )
        if ok:
            logging.info(f"Loading cached Ramen stats from {cache_path}")
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in cached.items()}
        logging.warning(
            "Cached stats at %s are incompatible (action_q02 shape %s, horizon=%s, n_obs_steps=%s); "
            "recomputing for horizon=%s, n_obs_steps=%s.",
            cache_path,
            getattr(aq02, "shape", None),
            cached_h,
            cached_n,
            horizon,
            n_obs_steps,
        )

    all_keys = list({e.key for e in schema.state + schema.action})
    n_frames = len(dataset)
    logging.info(
        f"Computing Ramen stats — bulk-reading {n_frames} frames, "
        f"columns: {all_keys}"
    )

    data = _bulk_read_columns(dataset, all_keys)
    logging.info("Columns loaded, assembling vectors + RPY→rot6d...")

    all_obs = _assemble_bulk(data, schema.state)
    all_act = _assemble_bulk(data, schema.action)

    episode_lengths = _get_episode_lengths(dataset)
    delta_indices = list(range(0, horizon))
    k_min = delta_indices[0]
    k_max = delta_indices[-1]
    offset_tensor = torch.tensor(delta_indices, dtype=torch.long)

    chunks: list[Tensor] = []
    start = 0
    for ep_len in episode_lengths:
        usable_len = max(0, ep_len - drop_n_last_frames)
        t_lo = max(0, -k_min)
        t_hi = min(ep_len, usable_len) - k_max
        if t_hi > t_lo:
            obs_ep = all_obs[start : start + ep_len]
            act_ep = all_act[start : start + ep_len]
            anchors = torch.arange(t_lo, t_hi, dtype=torch.long)
            rows = anchors[:, None] + offset_tensor[None, :]
            act_window = act_ep[rows]
            state_t = obs_ep[anchors][:, None, :]

            delta = act_window.clone()
            shared = min(state_t.shape[-1], delta.shape[-1])
            m = norm_mask[:shared]
            delta[..., :shared][..., m] = (
                act_window[..., :shared][..., m] - state_t[..., :shared][..., m]
            )
            chunks.append(delta)
        start += ep_len

    if not chunks:
        raise ValueError(
            "No valid action chunks available for Ramen stats "
            f"(horizon={horizon}, n_obs_steps={n_obs_steps}, drop_n_last_frames={drop_n_last_frames})."
        )

    chunk_deltas = torch.cat(chunks, dim=0)
    all_obs = all_obs.unsqueeze(1)

    logging.info("Computing percentiles...")
    stats = {
        "obs_q02": torch.quantile(all_obs.float(), 0.02, dim=0),
        "obs_q98": torch.quantile(all_obs.float(), 0.98, dim=0),
        "action_q02": torch.quantile(chunk_deltas.float(), 0.02, dim=0),
        "action_q98": torch.quantile(chunk_deltas.float(), 0.98, dim=0),
        "horizon": torch.tensor(horizon, dtype=torch.int64),
        "n_obs_steps": torch.tensor(n_obs_steps, dtype=torch.int64),
        "norm_mask": norm_mask,
    }

    if cache_path:
        save_ramen_stats(stats, cache_path)
        logging.info(f"Saved Ramen stats to {cache_path}")

    return {k: v.to(device) for k, v in stats.items()}


def ramen_normalize_batch(
    batch: dict,
    ramen_stats: dict[str, Tensor],
    norm_mask: Tensor,
    *,
    clip_value: float = _DEFAULT_CLIP_VALUE,
) -> dict:
    """Normalize a full training batch using Ramen normalization.

    - observation.state and action: Ramen percentile normalization (6D rot exempt)
    - observation.images.*: ImageNet MEAN_STD normalization
    - Other keys (task, metadata): passed through unchanged

    norm_mask is sliced to match state_dim / action_dim automatically
    (they may differ). ``clip_value`` is forwarded to `ramen_normalize`
    and should come from ``MultiTaskDiTConfig.ramen_clip_value`` so it
    matches the inference-time DDIM ``clip_sample_range``.
    """
    normalized = {}

    for key, val in batch.items():
        if key == "observation.state":
            m = norm_mask[: val.shape[-1]]
            normalized[key] = ramen_normalize(
                val, ramen_stats["obs_q02"], ramen_stats["obs_q98"], m,
                clip_value=clip_value,
            )
        elif key == "action":
            m = norm_mask[: val.shape[-1]]
            normalized[key] = ramen_normalize(
                val, ramen_stats["action_q02"], ramen_stats["action_q98"], m,
                clip_value=clip_value,
            )
        elif key.startswith("observation.image") and isinstance(val, Tensor) and val.is_floating_point():
            mean = _IMAGENET_MEAN.to(val.device, val.dtype)
            std = _IMAGENET_STD.to(val.device, val.dtype)
            shape = [1] * (val.ndim - 3) + [3, 1, 1]
            normalized[key] = (val - mean.view(*shape)) / std.view(*shape)
        else:
            normalized[key] = val

    return normalized
