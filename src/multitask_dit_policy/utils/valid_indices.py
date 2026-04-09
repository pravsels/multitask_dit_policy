"""Filter DAgger datasets to exclude autonomous policy frames.

DAgger datasets contain segments where the policy ran autonomously
(control_mode == "policy") followed by human corrections.  Only
human-controlled frames should be used for training.

The filtering uses robocandywrapper's ``ControlModePlugin`` which
exposes per-episode segment metadata via ``episode_modes``.  When
no plugin metadata is present (e.g. non-DAgger datasets), all frames
are treated as human and no filtering occurs.

A ``valid_indices.json`` report is written per-dataset with excluded
ranges so that filtering can be verified independently of dataset
ordering.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

VALID_INDICES_FILENAME = "valid_indices.json"


def _unwrap_dataset(dataset):
    """Walk through TransformedDataset wrappers to the underlying multi-dataset."""
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset
    return dataset


def _get_plugin_instance(instances: list[object], attr_name: str) -> object | None:
    for instance in instances:
        if hasattr(instance, attr_name):
            return instance
    return None


def _episode_bounds(ds) -> tuple[list[int], list[int]]:
    episode_data_index = getattr(ds, "episode_data_index", None)
    if episode_data_index is not None:
        return list(episode_data_index["from"]), list(episode_data_index["to"])

    hf_dataset = getattr(ds, "hf_dataset", None)
    if hf_dataset is not None:
        episode_index = hf_dataset["episode_index"]
        ep_from: list[int] = []
        ep_to: list[int] = []
        current_episode: object = object()
        for idx, episode_idx in enumerate(episode_index):
            if episode_idx != current_episode:
                ep_from.append(idx)
                if len(ep_from) > 1:
                    ep_to.append(idx)
                current_episode = episode_idx
        if ep_from:
            ep_to.append(len(episode_index))
            return ep_from, ep_to

    raise ValueError(f"Dataset {getattr(ds, 'repo_id', '<unknown>')} is missing episode_data_index")


def _local_to_global_indices(index_map) -> dict[int, int] | None:
    if index_map is None:
        return None
    return {int(local_idx): virtual_idx for virtual_idx, local_idx in enumerate(index_map)}


def _policy_segments(segments) -> list[tuple[int, int]]:
    """Return (start, end) pairs for policy-mode segments within an episode."""
    if segments is None:
        return []
    ranges = []
    for segment in segments:
        if getattr(segment, "mode", None) != "policy":
            continue
        ranges.append((int(segment.start_index), int(segment.end_index)))
    return ranges


def _human_frame_indices(segments, episode_length: int) -> set[int]:
    """Return frame offsets within an episode that are human-controlled."""
    human_frames = set(range(episode_length))
    for start, end in _policy_segments(segments):
        human_frames -= set(range(start, end + 1))
    return human_frames


def compute_valid_indices(dataset) -> tuple[list[int], dict[str, Any]]:
    """Compute valid global indices and a per-dataset filtering report.

    Returns:
        A tuple of (valid_global_indices, report) where report is a dict
        keyed by repo_id with total_frames, valid_frames, and
        excluded_ranges (dataset-local frame indices).
    """
    dataset = _unwrap_dataset(dataset)
    valid: list[int] = []
    report: dict[str, Any] = {}

    for ds_idx, ds in enumerate(dataset._datasets):
        repo_id = getattr(ds, "repo_id", f"<dataset_{ds_idx}>")
        plugin_instances = dataset._plugin_instances[ds_idx]
        control_instance = _get_plugin_instance(plugin_instances, "episode_modes")
        episode_modes = {} if control_instance is None else dict(control_instance.episode_modes)

        ep_from, ep_to = _episode_bounds(ds)
        num_episodes = len(ep_from)

        local_to_virtual = _local_to_global_indices(dataset._index_maps[ds_idx])
        global_offset = int(dataset._cumulative_lengths[ds_idx])

        ds_total = 0
        ds_excluded: list[list[int]] = []

        for ep_idx in range(num_episodes):
            ep_start = int(ep_from[ep_idx])
            episode_length = int(ep_to[ep_idx]) - ep_start
            ds_total += episode_length

            segments = episode_modes.get(ep_idx)
            for seg_start, seg_end in _policy_segments(segments):
                ds_excluded.append([ep_start + seg_start, ep_start + seg_end])

            human_frames = _human_frame_indices(segments, episode_length)
            for frame_in_episode in sorted(human_frames):
                local_idx = ep_start + frame_in_episode
                if local_to_virtual is None:
                    global_idx = global_offset + local_idx
                else:
                    virtual_idx = local_to_virtual.get(local_idx)
                    if virtual_idx is None:
                        continue
                    global_idx = global_offset + virtual_idx
                valid.append(global_idx)

        ds_valid = ds_total - sum(e[1] - e[0] + 1 for e in ds_excluded)
        report[repo_id] = {
            "total_frames": ds_total,
            "valid_frames": ds_valid,
            "excluded_ranges": ds_excluded,
        }

    return valid, report


def write_filtering_report(report: dict[str, Any], path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote filtering report to %s", path)
    return path
