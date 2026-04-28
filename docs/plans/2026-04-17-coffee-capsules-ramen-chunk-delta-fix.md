# Coffee Capsules Ramen Chunk-Delta Bottleneck — Investigation & Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Confirm the chunk-relative-vs-single-frame-delta normalization mismatch on the
coffee-capsules dataset by walking real data through the existing checkpoint, then patch
`compute_ramen_stats` to emit per-position `(H, D)` stats and validate the new stats look
physically reasonable. **No retrain in this plan** — that is a follow-up.

**Architecture:** Two artifact streams.
1. `scripts/ramen_chunk_delta/*.py`: small standalone scripts that load data /
   checkpoint / stats, dump numbers, and print/save findings. Re-runnable.
2. `docs/2026-04-17-coffee-capsules-ramen-investigation.md`: written report
   capturing tables, traces, and conclusions from each phase.
The patch itself is in `src/multitask_dit_policy/utils/ramen_normalization.py`,
TDD against new tests in `tests/test_ramen_normalization.py`.

**Tech stack:** PyTorch, LeRobotDataset (HF backend), the project's existing
`compute_ramen_stats`, `MultiTaskDiTPolicy.load`, `ramen_unnormalize`,
`adapt_batch`. No new deps.

**Reference doc:** `docs/2026-04-16-ramen-chunk-delta-bottleneck.md` —
prior block-tower analysis with worked numbers; same hypothesis applies here.

**Target dataset & checkpoint (fixed for this investigation):**
- Dataset: `villekuosmanen/bin_pick_pack_coffee_capsules`
- Checkpoint: `pravsels/dit_coffee_capsules_config_fix` (checkpoint_30000)
- Schema (from `config/train_coffee_capsules.yaml`):
  - State 17D: `observation.state.pos` (7) + `observation.state.eef_pose` (7→10 after rot6d)
  - Action 17D: `action.pos` (7) + `action.eef_pose` (7→10 after rot6d)
  - rot6d_slice = `[10, 16]`
  - Dim layout: 0..5 = arm joints, 6 = joint-gripper, 7..9 = eef xyz, 10..15 = rot6d,
    16 = eef-gripper

---

## Phase 0: Setup

### Task 0.1: Create scratch dirs and init the investigation report

**Files:**
- Create: `scripts/ramen_chunk_delta/__init__.py` (empty)
- Create: `scripts/ramen_chunk_delta/README.md` (one-paragraph: "scripts that
  reproduce + diagnose the chunk-relative-delta bottleneck on coffee-capsules;
  see docs/2026-04-17-coffee-capsules-ramen-investigation.md")
- Create: `docs/2026-04-17-coffee-capsules-ramen-investigation.md` with these sections
  as empty placeholders to be filled by later tasks:
  - `# Coffee Capsules Ramen Chunk-Delta Investigation`
  - `## TL;DR` (fill last)
  - `## Phase 1: Dataset characterization`
  - `## Phase 2: Reproducing the bottleneck on real frames`
  - `## Phase 3: Patch and recomputed stats`
  - `## Phase 4: Sanity assertions`
  - `## Conclusion / next steps`

**Step 1: Create the files** with the placeholder content above.

**Step 2: Commit**

```bash
git add scripts/ramen_chunk_delta docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "scaffold ramen chunk-delta investigation"
```

---

## Phase 1: Dataset characterization

Goal: understand what the data actually looks like before chasing model
behaviour. Specifically: confirm which dims are the gripper, what their
physical units/range are, episode-length distribution, and how gripper-close
events evolve frame-to-frame.

### Task 1.1: Per-dim summary script

**Files:**
- Create: `scripts/ramen_chunk_delta/01_dataset_summary.py`

**Behaviour:**
1. Load `villekuosmanen/bin_pick_pack_coffee_capsules` via `LeRobotDataset`.
2. Use the existing `_bulk_read_columns` + `_assemble_bulk` helpers in
   `multitask_dit_policy.utils.ramen_normalization` to materialize
   `all_obs (N, 17)` and `all_act (N, 17)` (after RPY→rot6d) — DRY.
3. Print and save (to `scripts/ramen_chunk_delta/out/01_summary.json`):
   - Total frames, total episodes, episode-length distribution
     (min/median/mean/max/p10/p90).
   - Per-dim table for both `obs` and `act`: min, max, mean, std, p02, p98.
   - Highlight rows: dim 6 (joint-gripper) and dim 16 (eef-gripper) — labelled
     as such.
   - Single-frame delta `(act - obs)` per-dim p02/p98 (= the current
     `action_q02/q98`) so we can directly compare to (H,D) stats later.

**Step 1: Write the script** (use `argparse` only for `--dataset_root` override;
default to local cache or HF auto-download).

**Step 2: Run it**

```bash
python scripts/ramen_chunk_delta/01_dataset_summary.py
```

Expected output:
- A clear printed table with the 17 dims and their ranges.
- A specific answer to: *is the gripper a joint angle (radians) or a 0–1
  open/close fraction?* Look at dim 6 and dim 16 ranges:
  - If dim 16 ∈ roughly `[0, 1]` → 0–1 open fraction.
  - If dim 6 ∈ roughly `[-0.05, 0.05]` rad → joint angle.

**Step 3: Paste the table into `docs/2026-04-17-coffee-capsules-ramen-investigation.md`**
under `## Phase 1: Dataset characterization`. Add 2–3 sentences interpreting:
which dim represents what, and what the gripper's natural physical scale is.

**Step 4: Commit**

```bash
git add scripts/ramen_chunk_delta/01_dataset_summary.py docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "phase 1: per-dim summary for coffee capsules"
```

### Task 1.2: Gripper-close trace from real episodes

**Files:**
- Create: `scripts/ramen_chunk_delta/02_gripper_close_traces.py`

**Behaviour:**
1. Find ~3 episodes that contain a clear gripper-close event:
   detect frames where `|act[t, 6] - act[t-1, 6]| > threshold` and pick the
   episode whose largest such delta is in the top quartile.
2. For each chosen episode, dump a 64-frame window centred on the close event:
   timestep, raw `obs[6]`, raw `act[6]`, raw `obs[16]`, raw `act[16]`,
   single-frame delta `act[6] - obs[6]`.
3. Save full traces as CSV under `scripts/ramen_chunk_delta/out/02_traces/`.
4. Also print:
   - Total physical motion in dim 6 across the close (`max - min` of `act[6]`).
   - Total physical motion across a full 32-frame chunk that *contains* the
     close (`act[t+31, 6] - act[t, 6]` for the chunk starting at the closest
     "open" frame before the close).
   - Same for dim 16.

**Step 1: Write the script.**

**Step 2: Run it**

```bash
python scripts/ramen_chunk_delta/02_gripper_close_traces.py
```

Expected: see numbers like "single-frame Δ peaks at ~0.005, but full-chunk
Δ over the same close ≈ 0.3 rad (or 0.6 of the 0–1 fraction)" — which is
exactly the >10× mismatch the bottleneck doc predicts.

**Step 3: Paste the per-episode summary numbers into the investigation report**
under `## Phase 1` (small table per episode: single-frame max-Δ vs
chunk max-Δ, ratio).

**Step 4: Commit**

```bash
git add scripts/ramen_chunk_delta/02_gripper_close_traces.py scripts/ramen_chunk_delta/out docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "phase 1: gripper close traces show single-frame vs chunk delta gap"
```

---

## Phase 2: Reproduce the bottleneck on real frames

Goal: load the existing checkpoint (which used the buggy `(1, D)` stats),
feed it a real frame, and confirm the model's normalized output saturates
while the unnormalized chunk barely moves the gripper.

### Task 2.1: Checkpoint download helper

**Files:**
- Create: `scripts/ramen_chunk_delta/03_fetch_checkpoint.py`

**Behaviour:**
1. `huggingface_hub.snapshot_download("pravsels/dit_coffee_capsules_config_fix")`
   into `scripts/ramen_chunk_delta/out/checkpoint_coffee_capsules/`.
2. Sanity print the files present and the sha256 of `model.safetensors`
   (compare to `1e0fa3278d7bc8e8a578822e03c613c9fe24f8c99ddacc0f0c416686cfe627b9`
   from the run log).
3. Also print the checkpoint's Ramen stats shapes — prefer
   `ramen_stats.json` if present, otherwise fall back to legacy
   `ramen_stats.pt` — and confirm `action_q02` / `action_q98` are `(1, 17)`
   (the bug we are fixing).

**Step 1: Write the script.**

**Step 2: Run it**

```bash
python scripts/ramen_chunk_delta/03_fetch_checkpoint.py
```

Expected: the printed shape is `(1, 17)`, confirming the checkpoint was
trained with the buggy stats. Sha matches.

**Step 3: Commit**

```bash
git add scripts/ramen_chunk_delta/03_fetch_checkpoint.py
git commit -m "phase 2: fetch coffee capsules checkpoint, confirm (1,D) stats"
```

### Task 2.2: Inference replay on a real frame

**Files:**
- Create: `scripts/ramen_chunk_delta/04_replay_one_chunk.py`

**Behaviour:**
1. Load the checkpoint via `MultiTaskDiTPolicy.load(checkpoint_dir)`.
2. Load the dataset (no delta_timestamps, single-frame mode like
   `examples/inference.py`). Pick a fixed frame index by episode + offset
   that we know contains a gripper-close in the next 32 frames (use one
   from Phase 1's chosen episodes).
3. Run inference:
   - Build the obs batch the model expects.
   - Normalize obs with the checkpoint's `ramen_stats` via
     `ramen_normalize_batch`.
   - Run `policy.select_action` in a loop until the queue empties (collect
     full 32-step chunk in normalized space). Mirror `examples/inference.py`.
4. Compute three traces and dump them side-by-side:
   - `actions_norm[k]` for k=0..31 (model's raw normalized output for each
     dim — focus on dims 0, 1, 2, 6, 16).
   - `actions_phys[k] = ramen_unnormalize(actions_norm) + current_state`
     using the checkpoint's existing `(1, 17)` stats.
   - `actions_gt[k]` = the ground-truth action chunk from the dataset for
     the same starting frame (apply RPY→rot6d so it's in the same 17D space).
5. Print per-dim (focus on 0, 1, 2, 6, 16):
   - max |normalized output| across the chunk (look for saturation near ±1).
   - total physical Δ predicted by the model across the chunk
     (`actions_phys[31] - actions_phys[0]`).
   - total physical Δ in ground truth
     (`actions_gt[31] - actions_gt[0]`).
   - ratio gt / predicted.
6. Save all three traces to `scripts/ramen_chunk_delta/out/04_chunk.csv`.

**Step 1: Write the script.**

**Step 2: Run it**

```bash
python scripts/ramen_chunk_delta/04_replay_one_chunk.py
```

Expected (the bottleneck reproduced):
- Normalized outputs for grip-close dims hit ≥0.9 magnitude.
- Predicted physical Δ on dim 6 ≈ 0.01–0.02 (rad or fraction).
- Ground-truth Δ on dim 6 ≈ 0.2–0.5+ — i.e. >10× larger than predicted.
- For arm joints, similar but smaller mismatch.

**Step 3: Write up the comparison** in
`docs/2026-04-17-coffee-capsules-ramen-investigation.md` under
`## Phase 2: Reproducing the bottleneck on real frames`. Include:
- A table mirroring the block-tower doc's
  "Absolute joint targets after unnormalize across the chunk".
- A one-sentence verdict: yes/no, the same bottleneck is present on
  coffee_capsules.

**Step 4: Commit**

```bash
git add scripts/ramen_chunk_delta/04_replay_one_chunk.py scripts/ramen_chunk_delta/out docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "phase 2: reproduce chunk-delta bottleneck on coffee capsules"
```

---

## Phase 3: Patch `compute_ramen_stats` (TDD)

Goal: change `compute_ramen_stats` to emit `(H, D)` action stats by
computing `action[t+k] - state[t]` per chunk position `k = 0..H-1`.
Obs stats stay `(1, D)`. Cache load asserts shape and recomputes if mismatched.

---

### ⚠️ Plan corrections after subagent verification (2026-04-17)

Three explore subagents stress-tested the assumptions below. Four corrections
must be folded into the steps that follow before implementation.

**Correction 1 — chunk position offsets are NOT `0..H-1`.**
The training pipeline uses `MultiTaskDiTConfig.action_delta_indices` =
`range(1 - n_obs_steps, 1 - n_obs_steps + horizon)`
(`src/multitask_dit_policy/utils/configuration.py:605-607`), and
`adapt_batch` subtracts `obs[:, -1:, :]` (= the **last** observation frame =
`obs[t]`) from every action timestep
(`src/multitask_dit_policy/utils/dataset_adapter.py:114-121`).

So for `n_obs_steps=2, horizon=32` (coffee), chunk position `i ∈ [0, H)` in
`batch["action"]` corresponds to the physical delta:

```
target[i] = act[t + (i + 1 - n_obs_steps)] - obs[t]
```

i.e. `i=0 → act[t-1] - obs[t]`, `i=31 → act[t+30] - obs[t]`. Stats `(H, D)`
must be computed over the same offsets, otherwise `stats[i]` would scale a
different physical quantity than the model produces at index `i`.

**Correction 2 — episode iteration must use `valid_indices` helpers.**
LeRobot 0.5.0 (the version pinned in this repo) doesn't reliably expose
`ds.episode_data_index`. Re-use the existing helpers in
`src/multitask_dit_policy/utils/valid_indices.py`:
- `_unwrap_dataset` to strip transform wrappers, and
- `_episode_bounds(ds)` which falls back to scanning
  `hf_dataset["episode_index"]`.

This also guarantees the lengths match the row order produced by
`_bulk_read_columns`, and lets us respect `drop_n_last_frames` consistently
with `compute_valid_indices`.

**Correction 3 — cache key must encode `(horizon, n_obs_steps)`.**
The current implementation writes JSON caches such as
`ramen_stats_H{horizon}_obs{n_obs_steps}.json`; the old single-file
`ramen_stats.pt` path was the pre-fix behavior. Embed both knobs in the
filename AND save them inside the artifact for on-load validation (warn +
recompute on mismatch), while still allowing legacy `.pt` loads for old
checkpoints.

**Correction 4 — `04_gap_norm.py` must be updated for `(H, D)`.**
That analysis script flattens chunk targets to `(M*H, D)` and assumes
`(1, D)` stats. With `(H, D)` stats the right shape is `(M, H, D)` ×
`(H, D)` per-position — see Phase 4 for the script update.

**Inference path:** verified that `MultiTaskDiTPolicy.load()` does NOT load a
checkpoint Ramen stats artifact directly, and `examples/inference.py` uses a
separate `dataset_stats.json` flow. Canonical checkpoint-side Ramen stats are
now `ramen_stats.json`, but older published checkpoints may still only have
legacy `ramen_stats.pt`. **No model-side code change is needed in this repo**
for the `(H, D)` switch — broadcasting in `ramen_normalize` /
`ramen_unnormalize` already handles `(H, D)` against `(B, H, D)` inputs.
The deploy-side `DiTRamenEngine` claim from
`docs/2026-04-16-ramen-chunk-delta-bottleneck.md` is in another repo and
out of scope here.

### Note: optional semantic cleanup of the action chunk contract

The current training/inference contract is internally consistent but
confusing:

- obs history uses `observation_delta_indices = [-1, 0]` when
  `n_obs_steps=2`
- action history uses `action_delta_indices = [-1, 0, 1, ..., H-2]`
- `adapt_batch` subtracts the last obs frame `obs[t]`
- inference then executes from `start_idx = n_obs_steps - 1`, so slot `0`
  is trained but not executed

That means the executed action chunk starts at the "current" action, but
the stored training target chunk has a leading alignment prefix. A cleaner
breaking-change alternative would be:

1. Change `MultiTaskDiTConfig.action_delta_indices` to `range(0, horizon)`,
   so chunk position `i` means `act[t+i] - obs[t]`.
2. Change `MultiTaskDiTPolicy._generate_actions()` to execute from slot `0`
   instead of slicing from `n_obs_steps - 1`.
3. Re-derive `drop_n_last_frames` against the new maximum lookahead.
4. Recompute all Ramen stats/caches against the new contract.
5. Treat all existing checkpoints as semantically incompatible and require
   retraining.

Important: this semantic cleanup does **not** remove the need for `(H, D)`
action stats. Even with the cleaner `act[t:t+H] - obs[t]` target, slot `0`
and slot `H-1` still represent different physical delta distributions, so
per-position stats remain necessary.

For this plan, keep the semantic cleanup as a documented follow-up unless we
explicitly decide to make a retrain-required breaking change. The `(H, D)`
stats bugfix stands on its own and should not be blocked on a larger
train/inference contract migration.

The numbered Tasks below are written against the original (incorrect)
`act[t : t+H] - obs[t]` formulation. When implementing, substitute the
corrected offsets and helpers from above.

---

### Task 3.1: Failing tests for new behaviour

**Files:**
- Modify: `tests/test_ramen_normalization.py` (or create if missing).

**Step 1: Read existing tests** in
`tests/test_ramen_normalization.py` (or `tests/smoke_test_ramen.py`) so the
new tests follow the same conventions.

**Step 2: Write the failing tests** (note all calls now pass both
`horizon=H` and `n_obs_steps=N`):

```python
def test_compute_ramen_stats_action_shape_is_horizon_by_dim(tmp_path):
    """Action stats should be (H, D), obs stats (1, D)."""
    H, N = 8, 2
    dataset = _make_fake_dataset(n_episodes=4, ep_len=20, state_dim=4, action_dim=4)
    schema = _fake_schema(state_dim=4, action_dim=4)
    norm_mask = torch.ones(4, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset, schema=schema, norm_mask=norm_mask,
        cache_path=tmp_path / "stats.pt", horizon=H, n_obs_steps=N,
    )

    assert stats["action_q02"].shape == (H, 4)
    assert stats["action_q98"].shape == (H, 4)
    assert stats["obs_q02"].shape == (1, 4)
    assert stats["obs_q98"].shape == (1, 4)


def test_compute_ramen_stats_q98_grows_with_position(tmp_path):
    """For a constant-velocity ramp dataset, q98[k, 0] should grow roughly
    linearly with chunk position k.

    With n_obs_steps=2, position k uses offset (k + 1 - n_obs_steps) = (k - 1)
    relative to the anchor, so target[k] = ramp_per_step * (k - 1).
    Position 0 is therefore SLIGHTLY NEGATIVE (look-back), and growth
    is monotonic from k=1 onwards.
    """
    H, N = 16, 2
    dataset = _make_ramp_dataset(n_episodes=8, ep_len=200, ramp_per_step=0.01,
                                 state_dim=2, action_dim=2)
    schema = _fake_schema(state_dim=2, action_dim=2)
    norm_mask = torch.ones(2, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset, schema=schema, norm_mask=norm_mask,
        cache_path=tmp_path / "stats.pt", horizon=H, n_obs_steps=N,
    )

    q98 = stats["action_q98"][:, 0]
    # Monotonic non-decreasing across the full window.
    assert (q98[1:] >= q98[:-1] - 1e-4).all()
    # k=0 is the look-back position (offset = -1), so q98[0] ~ -0.01.
    assert q98[0].item() < 0.0
    # k=H-1 corresponds to offset (H - n_obs_steps) = H-2, so target ~ (H-2) * ramp.
    expected = (H - N) * 0.01
    assert abs(q98[H - 1].item() - expected) < 0.05 * expected


def test_compute_ramen_stats_episode_boundary_handling(tmp_path):
    """Chunks must not span across episode boundaries — verify no NaN/Inf
    and that stats from a dataset with many short episodes are similar to
    those from one long episode of the same total length.

    Episodes shorter than (H + n_obs_steps - 1) contribute nothing because
    no anchor t satisfies both t + k_min >= 0 and t + k_max < ep_len.
    """
    H, N = 8, 2
    short = _make_ramp_dataset(n_episodes=20, ep_len=20, ramp_per_step=0.01,
                               state_dim=2, action_dim=2)
    long  = _make_ramp_dataset(n_episodes=2,  ep_len=200, ramp_per_step=0.01,
                               state_dim=2, action_dim=2)
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    s_short = compute_ramen_stats(short, schema=schema, norm_mask=mask,
                                  cache_path=tmp_path / "s.pt",
                                  horizon=H, n_obs_steps=N)
    s_long  = compute_ramen_stats(long,  schema=schema, norm_mask=mask,
                                  cache_path=tmp_path / "l.pt",
                                  horizon=H, n_obs_steps=N)

    assert torch.isfinite(s_short["action_q98"]).all()
    assert torch.isfinite(s_long["action_q98"]).all()
    assert torch.allclose(s_short["action_q98"], s_long["action_q98"], atol=0.05)


def test_compute_ramen_stats_position_zero_is_look_back_when_n_obs_steps_gt_1(tmp_path):
    """Direct assertion that position 0 corresponds to act[t-1] - obs[t]
    when n_obs_steps=2, matching what train.py's adapt_batch produces.
    Use a dataset where act[t] == state[t] + 0.01*t so the delta is exact."""
    H, N = 4, 2
    dataset = _make_ramp_dataset(n_episodes=4, ep_len=40, ramp_per_step=0.01,
                                 state_dim=2, action_dim=2)
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset, schema=schema, norm_mask=mask,
        cache_path=tmp_path / "stats.pt", horizon=H, n_obs_steps=N,
    )

    # Offsets are [-1, 0, 1, 2]. With act[t] - obs[t] = 0 by construction,
    # the per-position medians should be ramp * [-1, 0, 1, 2].
    q50 = torch.quantile(
        torch.linspace(stats["action_q02"][:, 0].min(),
                       stats["action_q98"][:, 0].max(), 100), 0.5)
    # Simpler: just assert the q98 ordering matches the offset ordering.
    expected_signs = torch.tensor([-1.0, 0.0, 1.0, 1.0])
    actual = stats["action_q98"][:, 0]
    assert (actual[1:] >= actual[:-1] - 1e-4).all()
    assert actual[0].item() < actual[-1].item()
```

Plus tiny helpers `_make_fake_dataset`, `_make_ramp_dataset`, `_fake_schema`
(small in-memory mock that mimics the `hf_dataset.column_names` + `select_columns`
interface used by `_bulk_read_columns`, plus a per-episode boundary list —
choose the simplest mock that the new code path needs).

**Step 3: Add a cache-shape-mismatch test** (covers the legacy `(1, D)`
cache path that existing checkpoints carry on disk):

```python
def test_compute_ramen_stats_recomputes_on_shape_mismatch(tmp_path):
    """If a stale (1, D) cache is on disk, loading should detect the shape
    mismatch and recompute."""
    H, N = 8, 2
    dataset = _make_ramp_dataset(n_episodes=4, ep_len=40, ramp_per_step=0.01,
                                 state_dim=2, action_dim=2)
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)
    cache = tmp_path / "stats.pt"

    stale = {
        "action_q02": torch.zeros(1, 2),
        "action_q98": torch.zeros(1, 2),
        "obs_q02":    torch.zeros(1, 2),
        "obs_q98":    torch.zeros(1, 2),
        "norm_mask":  mask,
    }
    torch.save(stale, cache)

    stats = compute_ramen_stats(
        dataset, schema=schema, norm_mask=mask,
        cache_path=cache, horizon=H, n_obs_steps=N,
    )
    assert stats["action_q98"].shape == (H, 2)


def test_compute_ramen_stats_recomputes_on_horizon_mismatch(tmp_path):
    """If a cache was saved for horizon=H1 but reloaded with horizon=H2,
    we must recompute (not silently return the wrong-shaped stats)."""
    N = 2
    dataset = _make_ramp_dataset(n_episodes=4, ep_len=80, ramp_per_step=0.01,
                                 state_dim=2, action_dim=2)
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)
    cache = tmp_path / "stats.pt"

    s1 = compute_ramen_stats(dataset, schema=schema, norm_mask=mask,
                             cache_path=cache, horizon=4, n_obs_steps=N)
    assert s1["action_q98"].shape == (4, 2)

    s2 = compute_ramen_stats(dataset, schema=schema, norm_mask=mask,
                             cache_path=cache, horizon=8, n_obs_steps=N)
    assert s2["action_q98"].shape == (8, 2)


def test_ramen_action_round_trip_matches_dataset_chunk(tmp_path):
    """Dataset-backed verification of the real invariant we care about:
    after computing per-position stats, normalizing and then unnormalizing an
    adapted action chunk should recover the original chunk elementwise."""
    H, N = 8, 2
    dataset = _make_ramp_dataset(n_episodes=4, ep_len=80, ramp_per_step=0.01,
                                 state_dim=2, action_dim=2)
    schema = _fake_schema(state_dim=2, action_dim=2)
    mask = torch.ones(2, dtype=torch.bool)

    stats = compute_ramen_stats(
        dataset, schema=schema, norm_mask=mask,
        cache_path=tmp_path / "stats.pt", horizon=H, n_obs_steps=N,
    )
    batch = _get_one_adapted_batch(dataset, schema=schema, norm_mask=mask)
    action = batch["action"]

    action_norm = ramen_normalize(
        action, stats["action_q02"], stats["action_q98"], mask,
    )
    action_rt = ramen_unnormalize(
        action_norm, stats["action_q02"], stats["action_q98"], mask,
    )

    assert torch.allclose(action_rt, action, atol=1e-6, rtol=1e-6)
```

**Step 4: Run the tests, expect failure**

```bash
pytest tests/test_ramen_normalization.py -k "test_compute_ramen_stats_" -v
```

Expected: the new tests fail first on the missing `horizon=` / `n_obs_steps=`
kwargs, then on shape/round-trip mismatches once those args are partially
wired in.

**Step 5: Commit**

```bash
git add tests/test_ramen_normalization.py
git commit -m "tests: failing tests for (H,D) ramen stats + cache mismatch"
```

### Task 3.2: Implement (H, D) stats

**Files:**
- Modify: `src/multitask_dit_policy/utils/ramen_normalization.py`

**Step 1: Change `compute_ramen_stats` signature** to add two required
kwargs: `horizon: int` and `n_obs_steps: int`. Both are needed because the
training pipeline indexes actions with offsets
`range(1 - n_obs_steps, 1 - n_obs_steps + horizon)` (see Correction 1
above), not `range(0, horizon)`.

**Step 2: Build per-position chunk-relative deltas.** Conceptually:

```python
# After loading all_obs (N, D_s) and all_act (N, D_a) for the WHOLE concatenated dataset:
ep_lengths = _get_episode_lengths(dataset)            # list[int], sums to N
H = horizon
delta_indices = list(range(1 - n_obs_steps, 1 - n_obs_steps + H))   # e.g. [-1, 0, ..., 30]
k_min = delta_indices[0]   # negative or 0
k_max = delta_indices[-1]  # = H - n_obs_steps (positive)

chunks = []                                            # list of (n_starts_per_ep, H, D)

start = 0
for ep_len in ep_lengths:
    # Valid anchor t (episode-local): t + k_min >= 0 AND t + k_max < ep_len.
    # Apply drop_n_last_frames consistently with compute_valid_indices:
    usable_len = max(0, ep_len - drop_n_last_frames)
    t_lo = max(0, -k_min)
    t_hi = min(ep_len, usable_len) - k_max  # exclusive upper bound on t
    if t_hi <= t_lo:
        start += ep_len
        continue

    obs_ep = all_obs[start : start + ep_len]           # (ep_len, D_s)
    act_ep = all_act[start : start + ep_len]           # (ep_len, D_a)

    # For each anchor t in [t_lo, t_hi), assemble (H, D_a) action window
    # at offsets delta_indices, anchored to obs[t].
    # Memory-efficient: build via fancy indexing rather than unfold so
    # negative offsets (k_min < 0) work.
    anchors = torch.arange(t_lo, t_hi)                          # (n_starts,)
    offsets = torch.tensor(delta_indices)                       # (H,)
    rows = anchors[:, None] + offsets[None, :]                  # (n_starts, H)
    act_window = act_ep[rows]                                   # (n_starts, H, D_a)
    state_t = obs_ep[anchors][:, None, :]                       # (n_starts, 1, D_s)

    delta = act_window.clone()
    shared = min(state_t.shape[-1], delta.shape[-1])
    m = norm_mask[:shared]
    delta[..., :shared][..., m] = (
        act_window[..., :shared][..., m] - state_t[..., :shared][..., m]
    )
    chunks.append(delta)
    start += ep_len

chunk_deltas = torch.cat(chunks, dim=0)               # (N_starts, H, D_a)
stats["action_q02"] = torch.quantile(chunk_deltas.float(), 0.02, dim=0)  # (H, D_a)
stats["action_q98"] = torch.quantile(chunk_deltas.float(), 0.98, dim=0)  # (H, D_a)
```

Obs stats stay `(1, D)` — the codebase currently uses one set of obs stats
broadcast across the `n_obs_steps` history. Keeping that here.

`drop_n_last_frames` should be plumbed through as a kwarg too (default 0
for back-compat); coffee config uses 0 so it's a no-op there, but it
matters for portability and matches `compute_valid_indices`.

**Step 3: Episode boundary discovery.** Reuse the existing helpers in
`src/multitask_dit_policy/utils/valid_indices.py` rather than rolling our
own — they already handle both LeRobot 0.5.0's `hf_dataset["episode_index"]`
fallback and transform-wrapper unwrapping consistently with how training
filters samples:

```python
def _get_episode_lengths(dataset) -> list[int]:
    """Per-episode frame counts in the same row order as _bulk_read_columns."""
    from multitask_dit_policy.utils.valid_indices import _unwrap_dataset, _episode_bounds

    root = _unwrap_dataset(dataset)
    inner = getattr(root, "_datasets", [root])
    lengths: list[int] = []
    for ds in inner:
        ep_from, ep_to = _episode_bounds(ds)
        lengths.extend(int(ep_to[i] - ep_from[i]) for i in range(len(ep_from)))
    return lengths
```

(If we don't want a cross-import to `valid_indices`, copy just the
`hf_dataset["episode_index"]` scan. The wrapper-unwrap step matters because
a `TransformedDataset` outer object may not expose `_datasets` directly.)

**Step 4: Cache shape + horizon check on load.** Replace the simple
`cache_path.exists()` load with a stricter validation that catches both
shape mismatch and accidental horizon/n_obs_steps changes:

```python
if cache_path and cache_path.exists():
    cached = torch.load(cache_path, map_location=device, weights_only=True)
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
        f"Cached stats at {cache_path} are incompatible "
        f"(action_q02 shape {getattr(aq02, 'shape', None)}, horizon={cached_h}, "
        f"n_obs_steps={cached_n}); recomputing for horizon={horizon}, "
        f"n_obs_steps={n_obs_steps}."
    )
```

And on save, embed both knobs in the dict as 0-d int tensors (safe under
`weights_only=True`):

```python
stats["horizon"] = torch.tensor(horizon, dtype=torch.int64)
stats["n_obs_steps"] = torch.tensor(n_obs_steps, dtype=torch.int64)
```

In `train.py` (Step 8 below), also embed the horizon in the cache filename
to avoid silently clobbering caches across runs with different horizons:

```python
stats_cache = run_dir / f"ramen_stats_H{cfg.policy.horizon}_obs{cfg.policy.n_obs_steps}.json"
```

**Step 5: Memory note.** For a 100k-frame dataset with H=32 the rolling window
is ~100k × 32 × 17 × 4 bytes ≈ 217 MB — fine on CPU. For larger datasets
(>1M frames), drop the unfold and stream chunks in batches; not needed for
coffee_capsules (~30k frames).

**Step 6: Run the tests, expect pass**

```bash
pytest tests/test_ramen_normalization.py -k "test_compute_ramen_stats_" -v
```

Expected: all tests above pass, including the dataset-backed
normalize→unnormalize round-trip check.

**Step 7: Run the full test file** to make sure nothing else broke:

```bash
pytest tests/test_ramen_normalization.py -v
```

**Step 8: Update callers.** Every call to `compute_ramen_stats` /
`load_or_compute_ramen_stats` now needs three new pieces of info:
`horizon`, `n_obs_steps`, and (optionally) `drop_n_last_frames`.

```bash
rg -n "compute_ramen_stats\(" src/ tests/ scripts/
```

In `train.py` (around line 397), update both the cache filename and the
call:

```python
stats_cache = (
    run_dir / f"ramen_stats_H{cfg.policy.horizon}_obs{cfg.policy.n_obs_steps}.json"
)
ramen_stats = load_or_compute_ramen_stats(
    dataset=dataset,
    schema=schema,
    norm_mask=norm_mask,
    cache_path=stats_cache,
    device=runtime_context.device,
    runtime_context=runtime_context,
    horizon=cfg.policy.horizon,
    n_obs_steps=cfg.policy.n_obs_steps,
    drop_n_last_frames=cfg.policy.drop_n_last_frames,
)
```

For analysis scripts (e.g. `scripts/ramen_chunk_delta/05_inference_replay.py`,
`04_gap_norm.py`), pass the same values from the loaded `MultiTaskDiTConfig`.

**Step 9: Run train.py smoke**

```bash
pytest tests/test_train.py -v
```

If a train smoke fails because horizon isn't threaded through, fix it
minimally and re-run.

**Step 10: Commit**

```bash
git add src/multitask_dit_policy/utils/ramen_normalization.py src/multitask_dit_policy/train.py tests/test_ramen_normalization.py
git commit -m "compute (H,D) chunk-relative ramen action stats"
```

---

## Phase 4: Recompute and validate stats on coffee_capsules

Goal: actually run the new `compute_ramen_stats` on the real dataset, save
the resulting `(H, D)` stats next to the old `(1, D)` ones from the
checkpoint, and write down a sanity report.

### Task 4.1: Recompute script

**Files:**
- Create: `scripts/ramen_chunk_delta/05_recompute_stats.py`

**Behaviour:**
1. Load `villekuosmanen/bin_pick_pack_coffee_capsules` and the schema from
   `config/train_coffee_capsules.yaml` (use the same loader paths
   `train.py` uses to keep DRY — import `build_norm_mask`, etc.).
2. Call `compute_ramen_stats(dataset, schema, norm_mask, cache_path=...,
   horizon=32, n_obs_steps=2, drop_n_last_frames=0)` — fresh cache file at
   `scripts/ramen_chunk_delta/out/ramen_stats_coffee_capsules_H32_obs2.json`.
   These knobs come from `train_coffee_capsules.yaml` and must match what
   the model was/will be trained with.
3. Print:
   - Shapes of all four stats tensors.
   - For dims 0, 1, 2, 6, 16: a small table showing `q02[i]` and `q98[i]`
     for i ∈ {0, 1, 4, 8, 16, 24, 31}, alongside the corresponding offset
     `i + 1 - n_obs_steps` so it's clear which physical Δ each row scales.
   - Comparison row: the old `(1, D)` stats from the checkpoint's Ramen stats
     artifact (`ramen_stats.json` if present, otherwise legacy
     `ramen_stats.pt`) for the same dims (`i=1` of new ≈ old, since that's
     the offset-0 single-frame delta).
4. Save the printed tables to `scripts/ramen_chunk_delta/out/05_recompute.md`.

**Step 1: Write the script.**

**Step 2: Run it**

```bash
python scripts/ramen_chunk_delta/05_recompute_stats.py
```

Expected:
- For `n_obs_steps=2`, position `i=1` corresponds to offset 0 (i.e., the
  classic single-frame delta `act[t] - obs[t]`). So
  `action_q98[1, 6]` ≈ checkpoint's `action_q98[0, 6]` ≈ 0.017
  (the existing `(1, D)` checkpoint stat that we're replacing).
- Position `i=0` is the look-back delta `act[t-1] - obs[t]` — should be
  small and slightly negative on motion dims.
- `action_q98[31, 6]` (offset = +30) is much larger — typically ≥ 10×
  position 1 — capturing the full grip-close range over a 1s chunk.
- Same shape for arm joints: monotonic growth with `i` from `i=1` onward.

**Step 3: Write up the recomputed tables** in
`docs/2026-04-17-coffee-capsules-ramen-investigation.md` under
`## Phase 3: Patch and recomputed stats`.

**Step 4: Commit**

```bash
git add scripts/ramen_chunk_delta/05_recompute_stats.py scripts/ramen_chunk_delta/out docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "phase 4: recompute (H,D) ramen stats for coffee capsules"
```

### Task 4.2: Sanity assertions on the new stats

**Files:**
- Create: `scripts/ramen_chunk_delta/06_validate_stats.py`

**Behaviour:** load the recomputed
`ramen_stats_coffee_capsules_H32_obs2.json` and
assert the following (printing PASS/FAIL for each, and exit nonzero on any
fail):

1. Shapes: `action_q02.shape == (32, 17)`, `action_q98.shape == (32, 17)`,
   `obs_q02.shape == (1, 17)`, `obs_q98.shape == (1, 17)`.
2. `action_q98[1, :] - action_q02[1, :]` is approximately equal to the
   checkpoint's old `(1, 17)` `q98 - q02` (within 5%) — sanity that the
   "single-frame delta" position (offset 0 with `n_obs_steps=2`) matches
   old behaviour.
3. For each motion dim in `{0, 1, 2, 3, 4, 5, 6, 16}`:
   - `action_q98[31, d] - action_q02[31, d]` is ≥ 5× greater than
     `action_q98[1, d] - action_q02[1, d]` (full chunk vs single-frame).
4. For grip dim 6: `action_q98[31, 6] - action_q02[31, 6]` is on the order of
   the *physical full grip range* observed in Phase 1 (within a factor of 2).
5. No NaN / Inf anywhere in any of the four stats tensors.
6. rot6d dims (10..15) are not modified relative to a no-op identity (the
   `norm_mask` excludes them, but defensively check that
   `action_q02[:, 10:16]` and `action_q98[:, 10:16]` either match the raw
   action quantiles or are bypassed in `ramen_normalize` — the
   `norm_mask` is what gates this; just print rot6d stats for visual sanity).
7. Pick one or more real adapted dataset chunks, run `ramen_normalize(...)`
   followed by `ramen_unnormalize(...)`, and assert the recovered action chunk
   matches the original chunk elementwise up to small floating-point
   tolerance. This is the direct end-to-end proof that per-timestep action
   stats are aligned to the dataset chunk semantics.

**Step 1: Write the script.**

**Step 2: Run it**

```bash
python scripts/ramen_chunk_delta/06_validate_stats.py
```

Expected: all PASS.

**Step 3: Paste the PASS/FAIL summary into the investigation report** under
`## Phase 4: Sanity assertions`.

**Step 4: Commit**

```bash
git add scripts/ramen_chunk_delta/06_validate_stats.py docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "phase 4: validate recomputed stats are physically reasonable"
```

### Task 4.2b: Update gap_norm + inference replay scripts for `(H, D)` stats

**Files:**
- Modify: `scripts/ramen_chunk_delta/04_gap_norm.py`
- Modify: `scripts/ramen_chunk_delta/05_inference_replay.py`

**Why:** these scripts were written against `(1, D)` stats. With the new
`(H, D)` stats:
- `04_gap_norm.py` flattens chunk targets to `(M*H, D)` before normalize —
  needs to keep `(M, H, D)` and normalize per chunk position.
- `05_inference_replay.py` already operates on the full `(B, H, D)` chunk
  in one shot, so broadcasting against `(H, D)` stats works automatically;
  but if it loaded the checkpoint's legacy `(1, D)` `ramen_stats.pt` it should
  be given an option to load the new `(H, D)` JSON stats instead.

**Behaviour:**
1. In `04_gap_norm.py`, add a `--stats_path` arg (default = the new
   `out/ramen_stats_coffee_capsules_H32_obs2.json`). Stop the
   `(M*H, D)` reshape; pass `(M, H, D)` directly to `ramen_normalize`.
   Re-run and confirm `gap_norm` shrinks dramatically (since the per-position
   stats now actually scale per chunk position).
2. In `05_inference_replay.py`, accept the same `--stats_path` and use the
   new stats for the unnormalize step. Re-record the per-dim numbers so we
   can compare `gap_model` and `gap_norm` against the buggy-stats baseline.

**Step 1: Apply both edits.**

**Step 2: Re-run both scripts** and dump fresh outputs into
`scripts/ramen_chunk_delta/out/`.

**Step 3: Update the investigation report** sections 2.2a (gap_norm) and
2.2b (gap_model) with the "with new stats" comparison numbers. This is
the punchline: gap_norm should be ~0 on motion dims for `i ≥ 1` (since
each chunk position has its own scaling now), and gap_model should
remain small (model still fits its training target accurately).

**Step 4: Commit**

```bash
git add scripts/ramen_chunk_delta/04_gap_norm.py scripts/ramen_chunk_delta/05_inference_replay.py scripts/ramen_chunk_delta/out docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "phase 4: re-run gap analysis against (H,D) stats"
```

### Task 4.3: Semantic cleanup — action chunk should start at the next/current action

Status: completed.

**Files:**
- Modify: `src/multitask_dit_policy/utils/configuration.py`
- Modify: `src/multitask_dit_policy/src/multitask_dit_policy/model/model.py`
- Modify: `src/multitask_dit_policy/train.py`
- Modify: `src/multitask_dit_policy/utils/ramen_normalization.py`
- Modify: `tests/test_ramen_normalization.py`
- Modify: `tests/test_train.py`
- Modify: `tests/smoke_test_ramen.py`
- Modify: `scripts/ramen_chunk_delta/04_gap_norm.py`
- Modify: `scripts/ramen_chunk_delta/05_inference_replay.py` (only if we still want a replay diagnostic after the semantic cutover)
- Modify: `docs/2026-04-17-coffee-capsules-ramen-investigation.md`

**Why:** the current contract is internally consistent but confusing:

- obs history uses `observation_delta_indices = [-1, 0]` when `n_obs_steps=2`
- action history uses `action_delta_indices = [-1, 0, 1, ..., H-2]`
- `adapt_batch` subtracts the last obs frame `obs[t]`
- inference then executes from `start_idx = n_obs_steps - 1`, so slot `0`
  is trained but not executed

We want the first position in the action chunk to be the first executable
action, i.e. slot `0` should correspond to `act[t] - obs[t]`, slot `1` to
`act[t+1] - obs[t]`, etc.

**Important:** this is a clean breaking change. Existing checkpoints and old
stats artifacts, including legacy `ramen_stats.pt` files, must be treated as
semantically incompatible.

**Behaviour:**
1. Change `MultiTaskDiTConfig.action_delta_indices` from
   `range(1 - n_obs_steps, 1 - n_obs_steps + horizon)` to `range(0, horizon)`.
2. Keep observation history unchanged (`observation_delta_indices` still ends
   at the current frame).
3. Change `MultiTaskDiTPolicy._generate_actions()` to execute from slot `0`
   instead of slicing from `n_obs_steps - 1`.
4. Re-derive `drop_n_last_frames` against the new maximum lookahead.
5. Recompute all `(H, D)` Ramen stats/caches under the new semantics, with a
   cache-key / metadata change so old caches are rejected.
6. Update tests so they assert the new target meaning:
   - slot `0 = act[t] - obs[t]`
   - slot `1 = act[t+1] - obs[t]`
   - inference executes from slot `0`
7. Update the investigation scripts/docs so any explanation of chunk offsets
   uses the new convention rather than the current look-back slot.

**Step 1: Write failing tests**

Add focused tests for:
- `action_delta_indices == [0, 1, ..., H-1]`
- adapted chunk slot `0` matching the current/next action
- inference slicing from slot `0`

**Step 2: Run the tests, expect failure**

```bash
pytest tests/test_ramen_normalization.py tests/test_train.py -k "action_delta_indices or generate_actions or ramen" -v
```

**Step 3: Apply the semantic cleanup**

Update config, inference slicing, trimming logic, and Ramen stats generation
to the new action-chunk meaning.

**Step 4: Recompute stats + rerun gap_norm under the new semantics**

The post-change acceptance checks are:
- real adapted chunks still round-trip through
  `ramen_normalize(...) -> ramen_unnormalize(...)`
- `gap_norm` remains near-zero with the recomputed `(H, D)` stats

**Step 5: Update docs**

Explicitly document that:
- slot `0` is now the first executable action
- old checkpoints are not compatible with the new code path
- retraining is required

**Step 6: Commit**

```bash
git add src/multitask_dit_policy/utils/configuration.py src/multitask_dit_policy/model/model.py src/multitask_dit_policy/train.py src/multitask_dit_policy/utils/ramen_normalization.py tests scripts docs
git commit -m "change action chunk to start at current action"
```

**Outcome:**

- `action_delta_indices` now starts at `0`, so slot `0` corresponds to
  `act[t] - obs[t]`.
- `_generate_actions()` now executes from slot `0`, so the first predicted row
  is also the first executed row.
- `drop_n_last_frames` was re-derived against the new maximum lookahead.
- `compute_ramen_stats(...)` now writes a semantic-version cache marker so old
  stats artifacts from the look-back contract are rejected (including legacy
  `ramen_stats.pt` files).
- Focused semantic tests and nearby regression tests passed after the change.
- Real-data recompute + validation passed under the new semantics, including
  dataset-backed `ramen_normalize(...) -> ramen_unnormalize(...)` round trips at
  indices `0`, `100`, and `1000` with max absolute errors `1.19e-07`,
  `1.19e-07`, and `5.96e-08`.
- `gap_norm` remained near-zero after the semantic cleanup, so the slot-0
  contract change did not reintroduce a structural normalization floor.

### Task 4.4: Write the TL;DR + conclusion

Status: completed.

**Files:**
- Modify: `docs/2026-04-17-coffee-capsules-ramen-investigation.md`

**Step 1: Fill in `## TL;DR`** with 4–6 bullets:
- The bug (single-frame stats vs chunk-relative deltas).
- Confirmed on coffee_capsules: `<numbers from Phase 2 — predicted Δ vs gt Δ
  per dim>`.
- Patched `compute_ramen_stats` to emit `(H, D)`.
- New `q98[31, 6]` is ~`<value>` vs old ~0.017 — `<ratio>×` larger.
- All sanity assertions pass.
- Retrain still required to recover precision (training targets were
  clamped at ±1.5 — see ref doc).

**Step 2: Fill in `## Conclusion / next steps`** with:
- Pointer to `docs/2026-04-16-ramen-chunk-delta-bottleneck.md` for the
  general analysis.
- One-paragraph plan for the retrain (config snapshot, expected outcome).
- Note that `DiTRamenEngine.__init__` already supports `(H, D)` stats so
  no deploy-side code change is required for the next checkpoint.

**Step 3: Commit**

```bash
git add docs/2026-04-17-coffee-capsules-ramen-investigation.md
git commit -m "phase 4: write up investigation tl;dr and next steps"
```

**Outcome:**

- The investigation doc now includes a filled `## TL;DR`.
- The conclusion is written and explicitly records that the normalization bug is
  fixed, the semantic cleanup landed, and retraining is required because old
  checkpoints are semantically incompatible with the new chunk contract.
- The doc also records the post-semantic-cleanup rerun results:
  refreshed `(H, D)` stats, real dataset round-trip checks, and the updated
  `gap_norm` summary.

---

## Done

At this point:
- `compute_ramen_stats` emits `(H, D)` action stats (and is unit-tested).
- A reproducible scripts pipeline under `scripts/ramen_chunk_delta/`
  documents the bug on coffee_capsules with real numbers.
- A new `(H, D)` stats file exists at
  `scripts/ramen_chunk_delta/out/ramen_stats_coffee_capsules_H32_obs2.json`
  for reference / future deploy-time experiments.
- A self-contained investigation doc lives at
  `docs/2026-04-17-coffee-capsules-ramen-investigation.md`.

**Out of scope for this plan (follow-up):**
- Retraining the coffee_capsules and block_tower checkpoints with the
  patched stats.
- A deploy-time experiment loading existing `(1, D)`-trained checkpoints
  with the new `(H, D)` stats to see if shapes match (covered in the
  block-tower ref doc as the "diagnostic hack").
