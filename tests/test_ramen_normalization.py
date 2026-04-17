import json

import torch

from multitask_dit_policy.utils.configuration import DatasetSchema, SchemaEntry
from multitask_dit_policy.utils.dataset_adapter import adapt_batch
from multitask_dit_policy.utils.ramen_normalization import (
    build_norm_mask,
    compute_ramen_stats,
    load_ramen_stats,
    ramen_normalize,
    save_ramen_stats,
    ramen_unnormalize,
)


class _FakeHFColumns:
    def __init__(self, columns: dict[str, torch.Tensor]):
        self._columns = {k: v.clone() for k, v in columns.items()}
        self.column_names = list(columns.keys())

    def select_columns(self, keys: list[str]):
        return {k: self._columns[k].cpu().numpy() for k in keys}

    def __getitem__(self, key: str):
        return self._columns[key].cpu().numpy()


class _FakeSingleDataset:
    def __init__(self, columns: dict[str, torch.Tensor]):
        self.hf_dataset = _FakeHFColumns(columns)
        self.repo_id = "fake_repo"


class _FakeWrappedDataset:
    def __init__(self, datasets: list[_FakeSingleDataset]):
        self._datasets = datasets

    def __len__(self) -> int:
        return int(sum(len(ds.hf_dataset["episode_index"]) for ds in self._datasets))


def _fake_schema(state_dim: int, action_dim: int) -> DatasetSchema:
    return DatasetSchema(
        state=[SchemaEntry(key="observation.state", dim=state_dim)],
        action=[SchemaEntry(key="action", dim=action_dim)],
        rot6d_slice=(state_dim, state_dim),
    )


def _make_fake_dataset(
    *,
    n_episodes: int,
    ep_len: int,
    state_dim: int,
    action_dim: int,
) -> _FakeWrappedDataset:
    datasets = []
    for episode_idx in range(n_episodes):
        state = torch.arange(ep_len, dtype=torch.float32).unsqueeze(-1).repeat(1, state_dim)
        state = state + episode_idx * 100.0
        action = state[:, :action_dim].clone()
        episode_index = torch.full((ep_len,), episode_idx, dtype=torch.int64)
        datasets.append(
            _FakeSingleDataset(
                {
                    "observation.state": state,
                    "action": action,
                    "episode_index": episode_index,
                }
            )
        )
    return _FakeWrappedDataset(datasets)


def _make_ramp_dataset(
    *,
    n_episodes: int,
    ep_len: int,
    ramp_per_step: float,
    state_dim: int,
    action_dim: int,
) -> _FakeWrappedDataset:
    datasets = []
    for episode_idx in range(n_episodes):
        t = torch.arange(ep_len, dtype=torch.float32)
        state = torch.zeros(ep_len, state_dim, dtype=torch.float32)
        state[:, 0] = ramp_per_step * t
        if state_dim > 1:
            state[:, 1:] = state[:, :1]

        action = state[:, :action_dim].clone()
        if action_dim > 1:
            action[:, 1:] = action[:, :1]

        episode_index = torch.full((ep_len,), episode_idx, dtype=torch.int64)
        datasets.append(
            _FakeSingleDataset(
                {
                    "observation.state": state,
                    "action": action,
                    "episode_index": episode_index,
                }
            )
        )
    return _FakeWrappedDataset(datasets)


def _get_one_adapted_batch(
    dataset: _FakeWrappedDataset,
    *,
    schema: DatasetSchema,
    norm_mask: torch.Tensor,
    anchor_t: int = 2,
    horizon: int = 8,
    n_obs_steps: int = 2,
):
    ds = dataset._datasets[0]
    obs = torch.stack(
        [
            ds.hf_dataset._columns["observation.state"][anchor_t - 1],
            ds.hf_dataset._columns["observation.state"][anchor_t],
        ],
        dim=0,
    ).unsqueeze(0)
    rows = torch.arange(anchor_t, anchor_t + horizon, dtype=torch.int64)
    action = ds.hf_dataset._columns["action"][rows].unsqueeze(0)
    batch = {
        "observation.state": obs,
        "action": action,
    }
    return adapt_batch(batch, schema, norm_mask)


def _get_one_adapted_batch_next_action(
    dataset: _FakeWrappedDataset,
    *,
    schema: DatasetSchema,
    norm_mask: torch.Tensor,
    anchor_t: int = 2,
    horizon: int = 8,
):
    ds = dataset._datasets[0]
    obs = torch.stack(
        [
            ds.hf_dataset._columns["observation.state"][anchor_t - 1],
            ds.hf_dataset._columns["observation.state"][anchor_t],
        ],
        dim=0,
    ).unsqueeze(0)
    rows = torch.arange(anchor_t, anchor_t + horizon, dtype=torch.int64)
    action = ds.hf_dataset._columns["action"][rows].unsqueeze(0)
    batch = {
        "observation.state": obs,
        "action": action,
    }
    return adapt_batch(batch, schema, norm_mask)


def _sample_stats() -> dict[str, torch.Tensor]:
    return {
        "obs_q02": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "obs_q98": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        "action_q02": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        "action_q98": torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float32),
        "norm_mask": torch.tensor([True, False]),
        "horizon": torch.tensor(2, dtype=torch.int64),
        "n_obs_steps": torch.tensor(2, dtype=torch.int64),
    }


def test_compute_ramen_stats_action_shape_is_horizon_by_dim(tmp_path):
    H, N = 8, 2
    dataset = _make_fake_dataset(n_episodes=4, ep_len=20, state_dim=4, action_dim=4)
    schema = _fake_schema(state_dim=4, action_dim=4)
    norm_mask = torch.ones(4, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=norm_mask,
        cache_path=tmp_path / "stats.pt",
        horizon=H,
        n_obs_steps=N,
    )

    assert stats["action_q02"].shape == (H, 4)
    assert stats["action_q98"].shape == (H, 4)
    assert stats["obs_q02"].shape == (1, 4)
    assert stats["obs_q98"].shape == (1, 4)


def test_compute_ramen_stats_q98_grows_with_position(tmp_path):
    H, N = 16, 2
    dataset = _make_ramp_dataset(
        n_episodes=8,
        ep_len=200,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    norm_mask = torch.ones(2, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=norm_mask,
        cache_path=tmp_path / "stats.pt",
        horizon=H,
        n_obs_steps=N,
    )

    q98 = stats["action_q98"][:, 0]
    assert (q98[1:] >= q98[:-1] - 1e-4).all()
    assert q98[0].item() >= -1e-6
    expected = (H - 1) * 0.01
    assert abs(q98[H - 1].item() - expected) < 0.05 * expected


def test_compute_ramen_stats_episode_boundary_handling(tmp_path):
    H, N = 8, 2
    short = _make_ramp_dataset(
        n_episodes=20,
        ep_len=20,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    long = _make_ramp_dataset(
        n_episodes=2,
        ep_len=200,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    s_short = compute_ramen_stats(
        short,
        schema=schema,
        norm_mask=mask,
        cache_path=tmp_path / "s.pt",
        horizon=H,
        n_obs_steps=N,
    )
    s_long = compute_ramen_stats(
        long,
        schema=schema,
        norm_mask=mask,
        cache_path=tmp_path / "l.pt",
        horizon=H,
        n_obs_steps=N,
    )

    assert torch.isfinite(s_short["action_q98"]).all()
    assert torch.isfinite(s_long["action_q98"]).all()
    assert torch.allclose(s_short["action_q98"], s_long["action_q98"], atol=0.05)


def test_compute_ramen_stats_position_zero_is_current_action_when_n_obs_steps_gt_1(tmp_path):
    H, N = 4, 2
    dataset = _make_ramp_dataset(
        n_episodes=4,
        ep_len=40,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=mask,
        cache_path=tmp_path / "stats.pt",
        horizon=H,
        n_obs_steps=N,
    )

    actual = stats["action_q98"][:, 0]
    assert (actual[1:] >= actual[:-1] - 1e-4).all()
    assert actual[0].item() >= -1e-6
    assert actual[-1].item() > actual[0].item()


def test_compute_ramen_stats_recomputes_on_shape_mismatch(tmp_path):
    H, N = 8, 2
    dataset = _make_ramp_dataset(
        n_episodes=4,
        ep_len=40,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)
    cache = tmp_path / "stats.pt"

    stale = {
        "action_q02": torch.zeros(1, 2),
        "action_q98": torch.zeros(1, 2),
        "obs_q02": torch.zeros(1, 2),
        "obs_q98": torch.zeros(1, 2),
        "norm_mask": mask,
    }
    torch.save(stale, cache)

    stats = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=mask,
        cache_path=cache,
        horizon=H,
        n_obs_steps=N,
    )
    assert stats["action_q98"].shape == (H, 2)


def test_compute_ramen_stats_recomputes_on_horizon_mismatch(tmp_path):
    N = 2
    dataset = _make_ramp_dataset(
        n_episodes=4,
        ep_len=80,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)
    cache = tmp_path / "stats.pt"

    s1 = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=mask,
        cache_path=cache,
        horizon=4,
        n_obs_steps=N,
    )
    assert s1["action_q98"].shape == (4, 2)

    s2 = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=mask,
        cache_path=cache,
        horizon=8,
        n_obs_steps=N,
    )
    assert s2["action_q98"].shape == (8, 2)


def test_ramen_action_round_trip_matches_dataset_chunk(tmp_path):
    H, N = 8, 2
    dataset = _make_ramp_dataset(
        n_episodes=4,
        ep_len=80,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=mask,
        cache_path=tmp_path / "stats.pt",
        horizon=H,
        n_obs_steps=N,
    )
    batch = _get_one_adapted_batch(
        dataset,
        schema=schema,
        norm_mask=mask,
        horizon=H,
        n_obs_steps=N,
    )
    action = batch["action"]

    action_norm = ramen_normalize(
        action,
        stats["action_q02"],
        stats["action_q98"],
        mask,
    )
    action_rt = ramen_unnormalize(
        action_norm,
        stats["action_q02"],
        stats["action_q98"],
        mask,
    )

    assert torch.allclose(action_rt, action, atol=1e-6, rtol=1e-6)


def test_build_norm_mask_excludes_rot6d_dims():
    mask = build_norm_mask(state_dim=17, rot6d_start=10, rot6d_end=16)
    assert mask.shape == (17,)
    assert mask[:10].all()
    assert not mask[10:16].any()
    assert mask[16]


def test_adapted_chunk_slot_zero_is_current_action_under_next_action_contract():
    dataset = _make_ramp_dataset(
        n_episodes=1,
        ep_len=20,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    batch = _get_one_adapted_batch_next_action(
        dataset,
        schema=schema,
        norm_mask=mask,
        anchor_t=5,
        horizon=4,
    )
    actual = batch["action"][0, :, 0]
    expected = torch.tensor([0.00, 0.01, 0.02, 0.03])

    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual[0].item() == 0.0


def test_compute_ramen_stats_position_zero_is_current_action_when_chunk_starts_at_current(tmp_path):
    H, N = 4, 2
    dataset = _make_ramp_dataset(
        n_episodes=4,
        ep_len=40,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=mask,
        cache_path=tmp_path / "stats.pt",
        horizon=H,
        n_obs_steps=N,
    )

    q98 = stats["action_q98"][:, 0]
    assert q98[0].item() >= -1e-6
    assert torch.allclose(q98, torch.tensor([0.00, 0.01, 0.02, 0.03]), atol=0.02)


def test_save_ramen_stats_writes_canonical_json(tmp_path):
    path = tmp_path / "ramen_stats.json"

    save_ramen_stats(_sample_stats(), path)

    payload = json.loads(path.read_text())
    assert payload["format"] == "ramen_norm_stats"
    assert payload["format_version"] == 1
    assert payload["metadata"] == {
        "horizon": 2,
        "n_obs_steps": 2,
        "state_dim": 2,
        "action_dim": 2,
    }
    assert payload["norm_mask"] == [True, False]
    assert payload["state"]["q02"] == [[1.0, 2.0]]
    assert payload["action"]["q98"] == [[9.0, 10.0], [11.0, 12.0]]


def test_load_ramen_stats_reads_canonical_json(tmp_path):
    path = tmp_path / "ramen_stats.json"
    save_ramen_stats(_sample_stats(), path)

    loaded = load_ramen_stats(path)

    assert torch.equal(loaded["obs_q02"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(loaded["obs_q98"], torch.tensor([[3.0, 4.0]]))
    assert torch.equal(loaded["action_q02"], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    assert torch.equal(loaded["action_q98"], torch.tensor([[9.0, 10.0], [11.0, 12.0]]))
    assert torch.equal(loaded["norm_mask"], torch.tensor([True, False]))
    assert int(loaded["horizon"].item()) == 2
    assert int(loaded["n_obs_steps"].item()) == 2


def test_load_ramen_stats_supports_legacy_pt(tmp_path):
    path = tmp_path / "ramen_stats.pt"
    torch.save(_sample_stats(), path)

    loaded = load_ramen_stats(path)

    assert torch.equal(loaded["obs_q02"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(loaded["action_q98"], torch.tensor([[9.0, 10.0], [11.0, 12.0]]))
    assert torch.equal(loaded["norm_mask"], torch.tensor([True, False]))


def test_compute_ramen_stats_json_cache_path_writes_canonical_json(tmp_path):
    H, N = 4, 2
    dataset = _make_ramp_dataset(
        n_episodes=2,
        ep_len=40,
        ramp_per_step=0.01,
        state_dim=2,
        action_dim=2,
    )
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)
    cache = tmp_path / "ramen_stats.json"

    stats = compute_ramen_stats(
        dataset,
        schema=schema,
        norm_mask=mask,
        cache_path=cache,
        horizon=H,
        n_obs_steps=N,
    )

    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert payload["format"] == "ramen_norm_stats"
    assert payload["metadata"]["horizon"] == H
    assert payload["metadata"]["n_obs_steps"] == N
    assert len(payload["action"]["q02"]) == H
    assert stats["action_q02"].shape == (H, 2)
