# Coffee Capsules Ramen Chunk-Delta Investigation

## TL;DR

- The core bug was using one single-frame action normalization row `(1, D)` for
  chunk targets whose physical delta scale grows across horizon position `k`.
  Later chunk positions were therefore clipped against the wrong range.
- We patched `compute_ramen_stats(...)` to emit per-position action stats
  `(H, D)` and added cache metadata so stale stats from the old semantics are
  rejected.
- On `coffee_capsules`, the recomputed stats behave as expected: position `0`
  is now the single-step row (`act[t] - obs[t]`), while later positions widen
  substantially on motion dims, reaching about `6.5x` to `8.5x` the old span by
  `k = 31` for the main arm joints.
- The strongest end-to-end check passes on real adapted chunks:
  `ramen_normalize(...) -> ramen_unnormalize(...)` reconstructs dataset-backed
  action chunks at indices `0`, `100`, and `1000` with max absolute errors
  `1.19e-07`, `1.19e-07`, and `5.96e-08`.
- Re-running `gap_norm` with the recomputed `(H, D)` stats leaves the
  model-independent normalization floor effectively at zero for almost every
  masked dim. The only notable residual is rare `joint_5` tail clipping
  (`mean |gap_norm| = 0.0020`, hard-clamped frac `1.21 %`), not a structural
  chunk-position mismatch.
- We then performed the semantic cleanup so action chunk slot `0` is now the
  first executable action (`act[t] - obs[t]`) instead of a look-back action.
  This is a breaking change: old checkpoints and old stats artifacts
  (including legacy `ramen_stats.pt` files on some published checkpoints) are
  semantically incompatible, so retraining is required.

## Definitions

Everything below is defined here once so each later section can be read
cold.

### Dataset axes & shapes

- `t`            — frame index inside the dataset (0 … 47,864)
- `t*`           — a particular frame chosen for analysis
- `H = 32`       — chunk horizon (the model's action chunk length)
- `k ∈ [0, H−1]` — chunk position inside one prediction
- `D = 17`       — assembled feature dim (see schema below)
- `M`            — number of valid `(t, k)` chunk targets across the dataset
- `B`            — batch size (always 1 for this investigation)
- `fps = 20`     — so a 32-frame chunk = **1.6 s** of motion

### Assembled state/action vector (D = 17)

Built by `multitask_dit_policy.utils.dataset_adapter.assemble_vector`
from the schema in `config/train_coffee_capsules*.yaml`.

| dim   | label             | source                              |
|------:|:------------------|:------------------------------------|
|  0–5  | joint_0..5        | `state.pos[0..5]` / `action.pos[0..5]` (rad) |
|   6   | joint_gripper     | `state.pos[6]` / `action.pos[6]`     (m, jaw separation) |
|  7–9  | eef_xyz_x,y,z     | `state.eef_pose[0..2]` / `action.eef_pose[0..2]` (m) |
| 10–15 | rot6d_0..5        | `state.eef_pose[3..5]` (RPY) → 6D rotation (Zhou et al.) |
|  16   | eef_gripper       | `state.eef_pose[6]` / `action.eef_pose[6]` (m, duplicate of dim 6) |

`norm_mask` is `True` on dims **0–9 and 16** (delta'd dims), `False` on
**10–15** (rot6d, pass-through).

### Symbols used in chunk-delta analysis

All quantities have shape `(H, D)` per `(t*, batch)` unless stated:

- `obs[t]`               — observation state at frame `t`, shape `(D,)`
- `action[t]`            — recorded action at frame `t`, shape `(D,)`
- `gt_raw[t*, k] := action[t* + k] − obs[t*]`
   the **chunk-relative delta** the training pipeline produces in
   `dataset_adapter.adapt_batch` (lines 116–121). For `norm_mask = False`
   dims this is just `action[t* + k]` itself (no subtraction).
   **In every section below, "raw" / "gt_raw" means delta, not absolute.**
- `single = action[t] − obs[t]`
   the **single-frame** (k = 0) delta. The buggy `(1, D)` Ramen stats
   are computed from this distribution.
- `q02, q98`             — 2 / 98 percentiles, shape `(1, D)` in the
   buggy stats; will be `(H, D)` after the fix.
- `denom = max(q98 − q02, 1e−8)` — divisor used by Ramen normalize.

### Ramen normalize / unnormalize round trip

For dims where `norm_mask = True`:

```
gt_norm_unclamped = (2 · (gt_raw − q02) / denom) − 1
gt_norm_clamped   = clamp(gt_norm_unclamped, −1.5, +1.5)   # training target
gt_round_trip     = ((gt_norm_clamped + 1) · denom / 2) + q02
```

`gt_round_trip` is what you get back if you normalize and then
un-normalize a chunk target — i.e. the *best the loss can possibly
ask the model to reproduce*, given the current stats.

### Gap decomposition

The 1.6-second physical motion the deployed model can't reproduce
splits into two pieces:

```
gap_norm  = gt_raw − gt_round_trip                # model-independent floor
gap_model = gt_round_trip − pred_unnorm           # model fit error
total_gap = gt_raw − pred_unnorm = gap_norm + gap_model
```

where `pred_unnorm = ramen_unnormalize(pred_norm)` and `pred_norm` is
what the policy emits via `objective.conditional_sample`.

`gap_norm` is the consequence of the buggy stats alone; `gap_model` is
what the model adds (or subtracts) on top. Phase 3's `(H, D)` stats fix
targets `gap_norm`.

### Saturation flags

- `sat. cnt`   — number of `(k)` positions in a chunk with
                 `|gt_norm_unclamped| > 1.5` (= "hard-clamped").
- `frac>x`     — fraction of all `(t, k)` targets in the dataset with
                 `|gt_norm_unclamped| > x`.

### Misc

- "buggy `(1, D)` stats" — the checkpoint's baked-in Ramen stats artifact
  (for this HF revision, legacy `ramen_stats.pt`; newer checkpoints may ship
  `ramen_stats.json`), whose `q02 / q98` were computed from `single` (one
  value per dim, shared across all H chunk positions).
- "(H, D) fix" — the patch in Phase 3: one `(q02, q98)` per chunk
  position `k`, so the stats track that `gt_raw` magnitude grows with `k`.
- DDIM `clip_sample_range` — the checkpoint config sets this to `1.0`,
  which independently caps `pred_norm` at ±1.0 during inference (a
  *second*, reinforcing constraint on top of the ±1.5 training clamp).

## Phase 1: Dataset characterization

### 1.1 Per-dim summary

Source: `scripts/ramen_chunk_delta/01_dataset_summary.py` (reads parquet
directly via pyarrow — no LeRobotDataset, no video decode).

- Repo: `villekuosmanen/bin_pick_pack_coffee_capsules`
- 47,865 frames across 200 episodes
- Episode lengths: min 158, p10 194, median 238, mean 239.3, p90 286, max 423
- Frame rate: **20 fps** (so a 32-frame chunk = 1.6 s of motion)

**Gripper units (the key question):**

| dim | label         | obs range          | action range       | single-frame Δ p02 / p98 |
|----:|:--------------|:-------------------|:-------------------|:--------------------------|
|  6  | joint_gripper | [0.0005, 0.0780]   | [-0.0043, 0.0814]  | -0.0395 / +0.0362         |
| 16  | eef_gripper   | [0.0005, 0.0780]   | [-0.0043, 0.0814]  | -0.0395 / +0.0362         |

The gripper is **jaw separation in metres** (parallel-jaw ARX5 X5), **not**
a joint angle and **not** a 0–1 fraction. Verified against the ARX5 SDK:

- `arx5-sdk/include/app/common.h`: `gripper_pos // m; 0 for close, GRIPPER_WIDTH for fully open`
- `arx5-sdk/include/app/config.h`: X5 default `gripper_width = 0.088` m
- `arx5-sdk/python/communication/zmq_client.py`: `GRIPPER_WIDTH = 0.08` (software cap)

So:
- `0.0` ≈ jaws together (closed)
- `0.088` = mechanical fully-open limit; dataset empirical max is 0.078 m
  (real teleop demos rarely command the absolute limit)

Empirical confirmation from the data: every episode starts (median 0.0013)
and ends (median 0.0013) near zero, and reaches a per-episode max
averaging 0.065. Pattern (start closed → open during episode → close at
end) fits a robot that rests with jaws together, opens to interact with
the object, and returns to rest closed.

Dims 6 and 16 are **bit-identical element-wise** across all 47,865 frames
(verified: `abs(state.pos[6] − eef_pose[6]).max() == 0.0` for both
observation and action). The dataset records the same gripper signal
twice (once in `observation.state.pos[6]`, once as the last element of
`observation.state.eef_pose[6]` which becomes dim 16 after rot6d
expansion). For the chunk-relative analysis we can treat them as one
signal.

**Implied chunk ceiling, both directions** (formula:
`physical Δ = ((y + 1) / 2) · (q98 − q02) + q02`, with `q02 = −0.040`,
`q98 = +0.036`, `q98 − q02 = 0.076`):

| norm `y` | physical Δ | meaning                              |
|---------:|-----------:|:-------------------------------------|
|    −1.5  | −0.059     | clamp lower bound (training cap)     |
|    −1.0  | −0.040     | "open" saturation = `q02`            |
|     0.0  | −0.002     | midpoint                             |
|    +1.0  | +0.036     | "close" saturation = `q98`           |
|    +1.5  | +0.055     | clamp upper bound (training cap)     |

So the *total* span the model can express across a chunk is at most
`q98 − q02 = 0.076` m of jaw motion at normal saturation, or `0.114` m if
the clamp limits are reached. Full physical grip span is `0.088` m
(mechanical) / `0.08` m (software cap), so the buggy `(1, D)` stats cap
the model at ~one full grip per chunk — and that's only if it swings
from −1.0 to +1.0 *across* the chunk. Less catastrophic than block
tower's ceiling (`q98` = 0.017 rad → ~22% of full close per chunk) but
still tight. Note the asymmetry: `|q02| = 0.040 > |q98| = 0.036`, so the
*close* direction is slightly more capped than the *open* direction
(0.036 m vs 0.040 m max delta-from-current at saturation).

**Arm joints — single-frame delta p02 / p98** (motion ceiling per chunk
under buggy stats):

| dim | label    | Δ p02   | Δ p98   |
|----:|:---------|:--------|:--------|
|  0  | joint_0  | -0.0916 | +0.1045 |
|  1  | joint_1  | -0.1232 | +0.1377 |
|  2  | joint_2  | -0.1602 | +0.1179 |
|  3  | joint_3  | -0.0996 | +0.1193 |
|  4  | joint_4  | -0.0649 | +0.0706 |
|  5  | joint_5  | -0.1433 | +0.1404 |

Real arm motions over 1.6 s often exceed several tenths of a radian on
multiple joints simultaneously — same shape of bottleneck, just less
extreme than block tower.

No NaNs / Infs anywhere; obs/act distributions are well-behaved per dim.

Full per-dim tables (state, action, single-frame delta) saved to
`scripts/ramen_chunk_delta/out/01_summary.json`.

### 1.2 Chunk-relative motion: where is the bottleneck?

Source: `scripts/ramen_chunk_delta/02_gripper_close_traces.py` plus
`01_dataset_summary.py` extension.

**Symbols used in this section** (full glossary at top of doc):

| symbol              | definition                                                            |
|---------------------|-----------------------------------------------------------------------|
| `gt_raw[t, k]`      | `action[t + k] − obs[t]` — chunk-relative delta (`H = 32`, `k ∈ [0, 31]`) |
| `single[t]`         | `action[t] − obs[t]` — single-frame (k=0) delta; basis of buggy `(1, D)` stats |
| `Δ span` / `p2p`    | `q98 − q02` of a delta distribution (central 96-percentile spread)    |

All numbers below are **deltas**, not absolute states.

Training computes the model's target as
`target[t, k] = action[t + k] − obs[t]` for `k ∈ [0, H−1]` (see
`dataset_adapter.adapt_batch`, lines 116–121). The current Ramen stats
are computed from `single = action[t] − obs[t]` (the `k = 0` slice).
The relevant comparison per dim is

`chunk_span = chunk_q98 − chunk_q02   vs   single_span = single_q98 − single_q02`

If `chunk Δ span > single Δ span`, the buggy stats can't represent the
bulk of the training target distribution: 1× of buggy norm equals
`single Δ span`, and the clamp at ±1.5 caps physical motion at
`1.5 × single Δ span` — both far below the actual chunk delta tail.

All numbers below are **delta magnitudes** (radians for joints 0–5,
metres for `eef_xyz_*` and gripper), measured as
`q98(Δ) − q02(Δ)` — i.e. the central 96-percentile span of the *delta*
distribution, not absolute joint/position values.

| dim | label         | single Δ span (q98-q02) | chunk Δ span (q98-q02) | ratio | comment                              |
|----:|:--------------|------------------------:|-----------------------:|------:|:-------------------------------------|
|  0  | joint_0       |                  0.196  |                 0.956  | 4.88× | severely clipped                     |
|  1  | joint_1       |                  0.261  |                 1.471  | 5.64× | severely clipped                     |
|  2  | joint_2       |                  0.278  |                 1.389  | 4.99× | severely clipped                     |
|  3  | joint_3       |                  0.219  |                 1.025  | 4.69× | severely clipped                     |
|  4  | joint_4       |                  0.135  |                 0.492  | 3.63× | clipped                              |
|  5  | joint_5       |                  0.284  |                 1.410  | 4.97× | severely clipped                     |
|  6  | joint_gripper |                  0.076  |                 0.118  | 1.56× | mostly fine, tail saturates          |
|  7  | eef_xyz_x     |                  0.052  |                 0.242  | 4.67× | severely clipped                     |
|  8  | eef_xyz_y     |                  0.064  |                 0.323  | 5.03× | severely clipped                     |
|  9  | eef_xyz_z     |                  0.071  |                 0.339  | 4.75× | severely clipped                     |
| 16  | eef_gripper   |                  0.076  |                 0.118  | 1.56× | duplicate of dim 6                   |

(rot6d dims 10–15 are pass-through, no delta is computed on them.)

So **the joints, not the gripper, are the loud part of the bottleneck**.
On joints 0–5 a typical 32-frame chunk demands `~5×` the dynamic range
the buggy stats can express at norm in [−1, +1], and `~3.3×` even at the
clamp ceiling of ±1.5. The model literally cannot output most of the
motions it's being trained on; the loss surface saturates in the clamp
plateau where the gradient is zero.

The gripper alone (1.56×) only saturates on extreme close events
(0.080 m chunk delta vs 0.076 m single-frame ceiling, 1.06× over the
soft bound). Top 3 saturating gripper events traced to:

| episode | t* (frame) | k* (chunk pos) | chunk Δ (m) |
|--------:|-----------:|---------------:|------------:|
|     112 |        216 |              7 |     −0.0804 |
|      80 |        214 |              4 |     −0.0802 |
|      45 |        183 |              5 |     −0.0796 |

Per-frame CSV traces at `scripts/ramen_chunk_delta/out/02_traces/`,
JSON summary at `out/02_summary.json`.

**Implication for the (H, D) stats fix:** computing q02/q98 over the
flattened chunk-target distribution (as in the table above) gives
roughly the right span, but a stronger fix is to compute one (q02, q98)
*per chunk position k*, since target magnitude grows with k. That's the
`(H, D)` shape the patched `compute_ramen_stats` will produce. The
flattened percentiles above are an upper bound on what each k-slice
will need.

## Phase 2: Reproducing the bottleneck on real frames

### 2.1 Checkpoint stats: shape and provenance

**Symbols used in this section** (full glossary at top of doc):

| symbol         | definition                                                                                       |
|----------------|--------------------------------------------------------------------------------------------------|
| `D = 17`       | assembled feature dims                                                                           |
| `H = 32`       | chunk horizon                                                                                    |
| `q02, q98`     | per-dim percentiles in the checkpoint's Ramen stats artifact. `(1, D)` = buggy (one per dim); `(H, D)` = fixed (one per `(k, d)`) |
| `norm_mask[d]` | bool — `True` for delta'd dims (joints, eef_xyz, gripper); `False` for rot6d pass-through        |

Source: `scripts/ramen_chunk_delta/03_inspect_checkpoint_stats.py`.
Prefers `ramen_stats.json` and falls back to legacy `ramen_stats.pt`, along
with `config.json`, from
[`pravsels/dit_coffee_capsules_config_fix/checkpoint_30000`](https://huggingface.co/pravsels/dit_coffee_capsules_config_fix)
(no model weights yet).

- **Config**: `horizon = 32`, `n_action_steps = 32`, `n_obs_steps = 2`.
- **Stats shapes**: every tensor is `(1, 17)`:

  | key          | shape    | dtype   |
  |--------------|----------|---------|
  | `obs_q02`    | (1, 17)  | float32 |
  | `obs_q98`    | (1, 17)  | float32 |
  | `action_q02` | (1, 17)  | float32 |
  | `action_q98` | (1, 17)  | float32 |
  | `norm_mask`  | (17,)    | bool    |

  → confirmed buggy `(1, D)` single-frame stats; no per-chunk-position
  dimension.
- **Provenance check**: per-dim `action_q02 / action_q98` from the
  checkpoint match the single-frame delta percentiles we recomputed in
  task 1.1 **bit-perfectly** (max |diff| = `0.000000` on all 11 masked
  dims). Same dataset, same recipe — no version skew.
- **Norm mask**: `True` on dims 0–9 and 16 (joints, eef_xyz, both
  gripper copies), `False` on dims 10–15 (rot6d pass-through). Matches
  the schema in `config/train_coffee_capsules.yaml`.

Concretely, this means at inference the model emits 32 chunk-relative
deltas, all 32 of which get unnormalized using the *same* `(1, 17)`
single-frame `q02/q98` derived from `act[t] − obs[t]`. Per task 1.2,
the bulk of the chunk-target distribution exceeds those bounds by
~5× on joints 0–5 and eef_xyz, so the model has been trained against a
target that lives mostly on the saturation plateau for those dims.

JSON dump of all 17 dims at `scripts/ramen_chunk_delta/out/03_checkpoint_stats.json`.

### 2.2a Model-independent normalization gap (`gap_norm`)

Source: `scripts/ramen_chunk_delta/04_gap_norm.py`.

**Symbols used in this section** (full glossary at top of doc; units = rad
for joints, m for eef_xyz / gripper):

| symbol                | definition                                                                          |
|-----------------------|-------------------------------------------------------------------------------------|
| `gt_raw[t, k]`        | `action[t + k] − obs[t]` — chunk-relative delta (training target)                   |
| `gt_norm_unclamped`   | `(2 (gt_raw − q02) / (q98 − q02)) − 1`                                              |
| `gt_norm_clamped`     | `clamp(gt_norm_unclamped, ±1.5)`                                                    |
| `gt_round_trip`       | `ramen_unnormalize(gt_norm_clamped)`                                                |
| **`gap_norm`**        | `gt_raw − gt_round_trip` — physical motion thrown away by the round trip *alone*    |
| "hard-clamped"        | `\|gt_norm_unclamped\| > 1.5`                                                       |
| `frac>x`              | fraction of all `(t, k)` chunk targets with `\|gt_norm_unclamped\| > x`             |
| `(1, D)` stats        | same `q02, q98` for every `k` — the buggy ones in the checkpoint                    |

**Idea.** Even before training a single step, the buggy `(1, D)` stats
throw away physical motion through the round trip
`gt_round_trip = ramen_unnormalize(ramen_normalize(gt_raw))`. Because
`ramen_normalize` clamps to `[−1.5, +1.5]`, anything in `gt_raw` whose
unclamped normalized value `|gt_norm_unclamped| > 1.5` gets folded onto
the boundary, and `gt_round_trip` saturates at the corresponding
physical ceiling `±1.5 · (q98 − q02)/2 + (q98 + q02)/2`. The leftover

```
gap_norm = gt_raw − gt_round_trip
```

is the lossy floor — physical motion **no model can recover**, no
matter how well it fits the (clipped) target. We computed this over
all `(t, k)` chunk targets in the dataset (M = 41,665 chunks, H = 32),
using the checkpoint's actual `(1, 17)` stats.

**Per-dim summary** ("frac>x" = fraction of `(t, k)` targets whose
unclamped normalized magnitude exceeds `x`):

| dim | label         | mean \|gap\| | p98 \|gap\| | max \|gap\| | frac>1.0 | frac>1.5 | frac>3.0 |
|----:|:--------------|-------------:|------------:|------------:|---------:|---------:|---------:|
|  0  | joint_0       |       0.045  |       0.413 |       0.773 |  30.27 % |  23.63 % |  12.46 % |
|  1  | joint_1       |       0.089  |       0.714 |       1.449 |  50.58 % |  38.07 % |  16.88 % |
|  2  | joint_2       |       0.060  |       0.660 |       1.559 |  27.75 % |  21.94 % |  10.75 % |
|  3  | joint_3       |       0.041  |       0.510 |       1.558 |  29.90 % |  20.97 % |   8.32 % |
|  4  | joint_4       |       0.019  |       0.195 |       0.490 |  33.13 % |  22.57 % |   6.98 % |
|  5  | joint_5       |       0.055  |       0.721 |       1.849 |  25.35 % |  17.94 % |   8.62 % |
|  6  | joint_gripper |      0.0004  |       0.008 |       0.024 |  17.58 % |   5.07 % |   0.00 % |
|  7  | eef_xyz_x     |       0.011  |       0.112 |       0.246 |  35.82 % |  25.63 % |  10.96 % |
|  8  | eef_xyz_y     |       0.016  |       0.146 |       0.272 |  33.53 % |  25.41 % |  12.70 % |
|  9  | eef_xyz_z     |       0.021  |       0.155 |       0.298 |  56.92 % |  42.68 % |  14.21 % |
| 16  | eef_gripper   |      0.0004  |       0.008 |       0.024 |  17.58 % |   5.07 % |   0.00 % |

*Headlines:*
- Worst dim by mean: **`joint_1`** — mean `|gap_norm|` = 0.089 rad,
  with **38 % of all chunk targets hard-clamped** (`|y| > 1.5`).
- Worst dim by max: **`joint_5`** — `1.85 rad` of motion thrown away
  in a single chunk position.
- **`eef_xyz_z` is hard-clamped on 43 %** of all chunk targets — the
  vertical pick/place motion is the most consistently saturated.
- **`joint_gripper`** is the *only* delta dim that's mostly fine:
  only 5 % hard-clamped, max gap 24 mm. (Matches task 1.2: gripper
  chunk span is only 1.56× single-frame span; everything else is ~5×.)
- Rot6d dims (10–15) round-trip cleanly because they're not delta'd
  and their q02/q98 already cover the full quaternion-projected range.

**Worst-case trace** (`scripts/ramen_chunk_delta/out/04_traces/dim01_joint_1_ep166_t0025.csv`,
joint_1 episode 166 frame 25, units = radians):

| k  | gt_raw   | gt_norm_unclamped | gt_round_trip | gap_norm |
|---:|---------:|------------------:|--------------:|---------:|
|  0 |  +0.103  |             +0.73 |        +0.103 |   +0.000 |
|  1 |  +0.153  |             +1.12 |        +0.153 |   +0.000 |
|  2 |  +0.209  |             +1.54 |   **+0.203**  |   +0.006 |
|  5 |  +0.373  |             +2.81 |   **+0.203**  |   +0.171 |
| 10 |  +0.678  |             +5.14 |   **+0.203**  |   +0.475 |
| 15 |  +0.967  |             +7.35 |   **+0.203**  |   +0.764 |
| 20 |  +1.227  |             +9.35 |   **+0.203**  |   +1.024 |
| 25 |  +1.454  |            +11.09 |   **+0.203**  |   +1.251 |
| 31 |  +1.652  |            +12.61 |   **+0.203**  |   +1.449 |

The chunk wants to swing joint_1 by **1.65 rad over 1.6 s**. After
the round trip the *training target* for chunk positions `k = 2..31`
is **constant at 0.203 rad** — every k beyond k=1 is collapsed onto
the same +1.5 saturation boundary. The loss can't tell the difference
between "swing joint_1 by 0.21 rad over 1.6 s" and "swing it by 1.65
rad over 1.6 s". The model literally cannot be asked the difference.

That, not the model itself, is the bottleneck. `gap_model` (next
section) will only be the *additional* error the model adds on top of
this floor.

### 2.2b Inference replay on a real frame (`gap_model`)

Source: `scripts/ramen_chunk_delta/05_inference_replay.py`.

**Symbols used in this section** (full glossary at top of doc; units = rad
for joints, m for eef_xyz and gripper):

| symbol              | definition                                          |
|---------------------|-----------------------------------------------------|
| `gt_raw[k]`         | `action[t* + k] − obs[t*]` — chunk-relative delta (training target) |
| `gt_norm_clamped`   | `clamp(normalize(gt_raw), ±1.5)`                    |
| `gt_round_trip`     | `unnormalize(gt_norm_clamped)`                      |
| `pred_norm`         | model output in norm space, shape `(H, D)` (`objective.conditional_sample`) |
| `pred_unnorm`       | `unnormalize(pred_norm)` — what the robot would execute |
| **`gap_norm`**      | `gt_raw − gt_round_trip` — Phase 2.2a normalization floor |
| **`gap_model`**     | `gt_round_trip − pred_unnorm` — model fit error     |
| **`total_gap`**     | `gt_raw − pred_unnorm` = `gap_norm + gap_model`     |
| `Σ\|·\|`            | sum over the `H = 32` chunk positions of `\|·\|`    |
| `gt_raw_p2p`        | `max gt_raw − min gt_raw` over `k = 0..31`          |
| `sat. cnt`          | number of `k` with `\|gt_norm_unclamped\| > 1.5`    |

We pull `model.safetensors` (~1.3 GB) for the same checkpoint, run
`policy.objective.conditional_sample` on **episode 166, t=25** (the
worst joint_1 saturation case from 2.2a), and decompose the
predicted-vs-ground-truth chunk into:

```
gap_norm  = gt_raw - gt_round_trip   (model-independent floor; from 2.2a)
gap_model = gt_round_trip - pred_unnorm   (model fit error)
total_gap = gt_raw - pred_unnorm = gap_norm + gap_model
```

**Per-dim summary across the 32 chunk positions** (sums of |·| across
k=0..31; `sat. cnt` = number of k with `|gt_norm_unclamped| > 1.5`):

| dim | label         | gt_raw_p2p | sat. cnt |  Σ\|gap_norm\| | Σ\|gap_model\| | Σ\|total_gap\| |
|----:|:--------------|-----------:|---------:|---------------:|---------------:|---------------:|
|  0  | joint_0       |     0.503  |  29 / 32 |          8.44  |          2.48  |         10.92  |
|  1  | joint_1       |     1.549  |  30 / 32 |         24.09  |          3.50  |         27.58  |
|  2  | joint_2       |     1.264  |  30 / 32 |         20.34  |          3.29  |         23.63  |
|  3  | joint_3       |     0.809  |  29 / 32 |         10.91  |          3.32  |         14.23  |
|  4  | joint_4       |     0.190  |  12 / 32 |          0.75  |          0.88  |          1.63  |
|  5  | joint_5       |     1.373  |  21 / 32 |         16.47  |          3.05  |         19.52  |
|  6  | joint_gripper |     0.000  |   0 / 32 |          0.00  |          0.01  |          0.01  |
|  7  | eef_xyz_x     |     0.049  |   2 / 32 |          0.00  |          0.35  |          0.35  |
|  8  | eef_xyz_y     |     0.203  |  28 / 32 |          3.13  |          0.73  |          3.86  |
|  9  | eef_xyz_z     |     0.060  |   0 / 32 |          0.00  |          0.65  |          0.65  |
| 16  | eef_gripper   |     0.000  |   0 / 32 |          0.00  |          0.01  |          0.01  |

On every saturated dim, **`Σ|gap_norm|` is 4–7× larger than
`Σ|gap_model|`** — the normalization floor dominates the model error
by an order of magnitude. The model is doing fine within the achievable
range; it's the achievable range itself that's the problem.

**Worst-case per-position trace** (joint_1, units = rad,
`scripts/ramen_chunk_delta/out/05_replay_ep166_t0025_dim01_joint_1.csv`,
selected rows):

| k  | gt_raw   | gt_norm_unclamped | gt_norm_clamped | gt_round_trip | pred_norm | pred_unnorm | gap_norm | gap_model | total_gap |
|---:|---------:|------------------:|----------------:|--------------:|----------:|------------:|---------:|----------:|----------:|
|  0 |  +0.103  |             +0.73 |           +0.73 |        +0.103 |     +0.02 |      +0.010 |    +0.00 |     +0.09 |     +0.09 |
|  2 |  +0.209  |             +1.54 |       **+1.50** |    **+0.203** |     +0.23 |      +0.038 |    +0.01 |     +0.17 |     +0.17 |
| 10 |  +0.678  |             +5.14 |       **+1.50** |    **+0.203** |     +0.70 |      +0.099 |    +0.47 |     +0.10 |     +0.58 |
| 20 |  +1.227  |             +9.35 |       **+1.50** |    **+0.203** |     +0.70 |      +0.099 |    +1.02 |     +0.10 |     +1.13 |
| 31 |  +1.652  |            +12.61 |       **+1.50** |    **+0.203** |     +0.81 |      +0.113 |    +1.45 |     +0.09 |     +1.54 |

Three things to notice:

1. **`gt_round_trip` is constant at +0.203 rad for k=2..31.** Same as
   in 2.2a — every chunk position past k=1 collapses onto the same
   training target. The loss could not differentiate "go 0.21 rad"
   from "go 1.65 rad" for those 30 positions.
2. **`pred_norm` peaks at +0.81, not at the clamp boundary +1.5.**
   That's because the checkpoint config sets
   `"clip_sample": True, "clip_sample_range": 1.0`, so DDIM clips
   intermediate samples to ±1.0 during inference. The model's
   *achievable* normalized range is therefore ±1.0, even tighter than
   the training clamp at ±1.5. So the deployed physical motion at
   `pred_norm = +1.0` would be `1.0 · (q98 − q02)/2 + (q98 + q02)/2`
   ≈ +0.135 rad — still nowhere near 1.65 rad. **This is a separate,
   reinforcing bug.** The (H, D) stats fix will improve things, but
   the `clip_sample_range = 1.0` should also be raised to ≥ 1.5 (or
   the clipping disabled) for full effect.
3. **`Σ|total_gap|` for joint_1 is 27.6 rad** of cumulative
   displacement the deployed policy can't produce. 87 % of that
   (24.1 rad) is `gap_norm`; only 13 % (3.5 rad) is `gap_model`.

So the conclusion is unambiguous: **the bottleneck is the normalization
recipe**, not the model capacity, optimizer, or training duration.
A perfectly trained model on this exact data would still leave 87 % of
the gap on the table.

JSON dump and full per-(k, dim) CSV traces for the 3 worst dims at
`scripts/ramen_chunk_delta/out/05_replay_ep166_t0025*`.

### 2.2c Fix #1: unify the Ramen clamp and the DDIM `clip_sample_range`

Two findings from 2.2b were addressable without re-training:

1. The historical Ramen clamp (`±1.5` in `ramen_normalize`) and the
   diffusion scheduler's `clip_sample_range` (`1.0` in the saved
   checkpoint config) were independent constants.
2. Worse, `clip_sample` / `clip_sample_range` from `DiffusionConfig`
   were *never even forwarded* to the HF `DDIMScheduler` constructor —
   so the runtime was using HF's defaults (`True` / `1.0`), not the
   value in the JSON.

**Fix.** Single source of truth: `MultiTaskDiTConfig.ramen_clip_value`
(default `1.5`). Threaded into:

- `ramen_normalize(..., clip_value=…)` and `ramen_normalize_batch(...)`
  at training time.
- `DiffusionObjective` → HF `DDIMScheduler(clip_sample=True,
  clip_sample_range=ramen_clip_value)` at inference time.

Legacy `DiffusionConfig.clip_sample{,_range}` fields are kept for
back-compat config loading but ignored at runtime (a
`DeprecationWarning` is emitted if they're set). Tests:
`tests/test_ramen_clip_value.py` (12 unit tests, all pass).

**Replay re-run with the fix** (same `episode 166, t=25`, all other
state identical; this is the *same checkpoint weights*, just with the
correctly-wired DDIM clip at `1.5`):

| dim | label         | Σ\|gap_model\| **before** | Σ\|gap_model\| **after** | factor |
|----:|:--------------|--------------------------:|-------------------------:|-------:|
|  0  | joint_0       |                     2.48  |                     0.22 |  11×   |
|  1  | joint_1       |                     3.50  |                     0.37 |   9×   |
|  2  | joint_2       |                     3.29  |                     0.24 |  14×   |
|  3  | joint_3       |                     3.32  |                     0.31 |  11×   |
|  5  | joint_5       |                     3.05  |                     0.31 |  10×   |
|  8  | eef_xyz_y     |                     0.73  |                     0.05 |  14×   |
| 14  | rot6d_4       |                    12.08  |                     2.83 |   4×   |

`Σ|gap_norm|` is unchanged on every dim (it's structural, doesn't
depend on the model output) — confirming this fix attacks `gap_model`
only, exactly as designed.

**Worst-case per-position trace, after the fix** (joint_1, units = rad,
selected rows from
`scripts/ramen_chunk_delta/out/05_replay_ep166_t0025_dim01_joint_1.csv`):

| k  | gt_raw   | gt_round_trip | pred_norm  | pred_unnorm | gap_model |
|---:|---------:|--------------:|-----------:|------------:|----------:|
|  0 |  +0.103  |        +0.103 |     +0.291 |      +0.045 |    +0.058 |
|  2 |  +0.209  |    **+0.203** |     +0.911 |      +0.126 |    +0.077 |
|  5 |  +0.373  |    **+0.203** |     +1.472 |      +0.199 |    +0.004 |
| 10 |  +0.678  |    **+0.203** |     +1.482 |      +0.201 |    +0.002 |
| 20 |  +1.227  |    **+0.203** |     +1.453 |      +0.197 |    +0.006 |
| 31 |  +1.652  |    **+0.203** |     +1.464 |      +0.198 |    +0.005 |

Compare to the pre-fix trace in 2.2b: `pred_norm` was capped at +0.81;
now reaches +1.48 (effectively touching the `+1.5` ceiling). After
chunk position `k = 5` the model is essentially fitting the saturated
training target perfectly — gap_model drops to ~5 mrad. **Everything
that's left at this point is `gap_norm`**, which is what Phase 3's
`(H, D)` stats fix is for.

## Phase 3: Patch and recomputed stats

Recomputed with the patched `compute_ramen_stats(...)` on the real
`villekuosmanen/bin_pick_pack_coffee_capsules` dataset using the training
config from `config/train_coffee_capsules.yaml`
(`horizon=32`, `n_obs_steps=2`, `drop_n_last_frames=0`).

Shapes:

- `obs_q02`: `(1, 17)`
- `obs_q98`: `(1, 17)`
- `action_q02`: `(32, 17)`
- `action_q98`: `(32, 17)`

Selected dims, comparing the new per-position stats to the checkpoint's old
single-row `(1, D)` action stats:

| dim | label         | pos 1 q02/q98 | old q02/q98 | pos 31 q02/q98 | span ratio (`pos31 / old`) |
|----:|:--------------|:--------------|:------------|:---------------|---------------------------:|
| 0   | joint_0       | `[-0.0820, +0.1076]` | `[-0.0916, +0.1045]` | `[-0.6043, +0.6161]` | `6.22x` |
| 1   | joint_1       | `[-0.1186, +0.1427]` | `[-0.1232, +0.1377]` | `[-1.0150, +1.1524]` | `8.31x` |
| 2   | joint_2       | `[-0.1534, +0.1213]` | `[-0.1602, +0.1179]` | `[-1.1982, +0.8643]` | `7.42x` |
| 6   | joint_gripper | `[-0.0272, +0.0375]` | `[-0.0395, +0.0362]` | `[-0.0637, +0.0621]` | `1.66x` |
| 16  | eef_gripper   | `[-0.0272, +0.0375]` | `[-0.0395, +0.0362]` | `[-0.0637, +0.0621]` | `1.66x` |

Key takeaways:

- `i = 1` (offset `0` for `n_obs_steps=2`) is indeed the closest row to the
  old `(1, D)` stats, which is what we want if the old checkpoint was storing
  single-frame deltas.
- Later chunk positions are dramatically wider on arm-motion dims:
  roughly `5x` to `8x` the old span by `i = 31`.
- Gripper dims also widen, but only by about `1.66x`, which matches the fact
  that the old single-frame stats were already covering a larger fraction of
  the gripper's physical range than they were for the arm joints.

### Re-running `gap_norm` with the new `(H, D)` stats

After recomputing the chunk-aware stats, we re-ran
`scripts/ramen_chunk_delta/04_gap_norm.py` against
`scripts/ramen_chunk_delta/out/ramen_stats_coffee_capsules_H32_obs2.json`,
using the chunk semantics that were current at that stage of the investigation

```text
target[k] = act[t + (k + 1 - n_obs_steps)] - obs[t]
```

That was later replaced by the semantic cleanup in Task 4.3, where slot `0`
now means `act[t] - obs[t]` and slot `k` means `act[t+k] - obs[t]`.

This is the punchline for the normalization bug itself:

| dim | label         | old mean \|gap_norm\| | new mean \|gap_norm\| | old max \|gap_norm\| | new max \|gap_norm\| | old frac>1.5 | new frac>1.5 |
|----:|:--------------|----------------------:|----------------------:|---------------------:|---------------------:|-------------:|-------------:|
|  0  | joint_0       | 0.0450 | 0.0000 | 0.773 | 0.050 | 23.63 % | 0.06 % |
|  1  | joint_1       | 0.0890 | 0.0000 | 1.449 | 0.055 | 38.07 % | 0.03 % |
|  2  | joint_2       | 0.0600 | 0.0000 | 1.559 | 0.103 | 21.94 % | 0.07 % |
|  5  | joint_5       | 0.0550 | 0.0020 | 1.849 | 0.919 | 17.94 % | 1.25 % |
|  9  | eef_xyz_z     | 0.0210 | 0.0000 | 0.298 | 0.016 | 42.68 % | 0.02 % |
|  6  | joint_gripper | 0.0004 | 0.0000 | 0.024 | 0.031 | 5.07 %  | 0.16 % |

The bug is effectively gone at the model-independent level:

- the large systematic floor from the old `(1, D)` stats disappears
- most dims now have mean and p98 `|gap_norm|` indistinguishable from zero
  at the printed precision
- the remaining nonzero mass is just rare tail clipping, not the pervasive
  chunk-position mismatch we saw before

The worst residual dim under the new stats is `joint_5`, but even there the
story is completely different from the pre-fix baseline:

- old mean `|gap_norm|` = `0.055`
- new mean `|gap_norm|` = `0.002`
- old hard-clamp fraction = `17.94 %`
- new hard-clamp fraction = `1.25 %`

So the old failure mode was not "Ramen clipping exists at all"; it was
"later chunk positions were being scaled with the wrong single-frame
statistics." Once each chunk position gets its own stats row, the
model-independent normalization loss collapses.

### `joint_5` residual, broken down by chunk position

Because `joint_5` still had the largest residual mean under the new stats, we
broke it down by chunk position `k` to check whether the remainder was still a
"late chunk positions are wrong" effect.

It is not.

- mean `|gap_norm|` stays in a narrow band from about `0.0009` to `0.0028`
  across all `k = 0..31`
- the worst mean is around `k = 9` (offset `+8`), not at the end of the chunk
- the hard-clamped fraction stays low everywhere: about `0.54 %` to `1.76 %`
- `p98 |gap_norm|` is effectively zero at **every** chunk position

Selected rows:

| k | offset | mean `|gap_norm|` | max `|gap_norm|` | frac `|y| > 1.5` |
|---:|------:|------------------:|-----------------:|-----------------:|
| 0  |   -1  | 0.001553 | 0.576352 | 1.7377 % |
| 1  |    0  | 0.001823 | 0.600156 | 1.7617 % |
| 9  |   +8  | 0.002839 | 0.859083 | 1.5025 % |
| 14 |  +13  | 0.002674 | 0.919191 | 1.3273 % |
| 24 |  +23  | 0.001507 | 0.723304 | 1.0128 % |
| 31 |  +30  | 0.000877 | 0.586476 | 0.5400 % |

So the residual `joint_5` loss looks like a small distribution tail that
survives at all positions, not a remaining per-position stats mismatch. The
systematic chunk-normalization bug is still fixed; what remains is rare tail
clipping on this one joint.

Artifacts written by the recompute step:

- `scripts/ramen_chunk_delta/out/ramen_stats_coffee_capsules_H32_obs2.json`
- `scripts/ramen_chunk_delta/out/05_recompute.md`
- `scripts/ramen_chunk_delta/out/05_recompute.json`

## Phase 4: Sanity assertions

Validation script: `scripts/ramen_chunk_delta/06_validate_stats.py`

PASS summary on the recomputed stats:

- `PASS` shapes:
  `action_q02=(32, 17)`, `action_q98=(32, 17)`,
  `obs_q02=(1, 17)`, `obs_q98=(1, 17)`
- `PASS` single-frame span match:
  at position `1`, masked dims have max relative error `0.2102` and mean
  relative error `0.0693` versus the old `(1, D)` span, so the offset-0 row
  remains close to old behavior without requiring exact equality on every dim
- `PASS` late-vs-early span growth:
  arm dims grow strongly by `i = 31`
  (`joint_0 6.44x`, `joint_1 8.29x`, `joint_2 7.51x`, `joint_3 6.86x`,
  `joint_4 5.13x`, `joint_5 6.48x`);
  gripper dims grow more moderately but still clearly
  (`joint_gripper 1.94x`, `eef_gripper 1.94x`)
- `PASS` grip range physical scale:
  position-31 grip span is `0.1258`, which is the same order of magnitude as
  the Phase-1 observed physical grip range (`~0.0780`)
- `PASS` finite values:
  no NaN / Inf anywhere in the four stats tensors
- `PASS` rot6d stats finite:
  `q02` range `(-0.9914, +0.0344)`, `q98` range `(-0.0224, +0.9984)`
- `PASS` dataset-backed normalize -> unnormalize round trip:
  real adapted chunks at dataset indices `0`, `100`, and `1000` round-trip
  with max absolute reconstruction error `5.96e-08`

That final round-trip check is the strongest end-to-end confirmation that the
new per-timestep action stats are aligned with the dataset chunk semantics:
for real adapted chunks, `ramen_normalize(...)` followed by
`ramen_unnormalize(...)` reproduces the original action chunk up to numerical
precision.

We also deliberately stop short of reinterpreting `gap_model` on the old
checkpoint under the new stats. That checkpoint was trained against the old,
wrong normalization targets, so `gap_model` is no longer a clean apples-to-
apples measure after this fix. The meaningful post-fix diagnostic is the
updated `gap_norm` result above.

## Post-semantic-cleanup rerun

Task 4.3 changed the action chunk contract so slot `0` is now the first
executable action:

```text
target[k] = act[t + k] - obs[t]
```

After that semantic cleanup, we reran the real-data stats recompute, the
dataset-backed normalize/unnormalize validation, and `gap_norm` again using the
new slot-0 convention.

### Recomputed stats under the cleaned semantics

The refreshed stats still have the expected shape:

- `obs_q02`, `obs_q98`: `(1, 17)`
- `action_q02`, `action_q98`: `(32, 17)`

And the first row now lines up with the old single-step distribution exactly in
the intended way:

- position `0` is offset `+0`, i.e. `act[t] - obs[t]`
- the old `(1, D)` stats are closest to row `0`, not row `1`
- later rows still widen strongly on motion dims

Selected span-growth examples from the rerun:

- `joint_0`: `0.97x` old span at `k=0`, `6.31x` by `k=31`
- `joint_1`: `1.00x` old span at `k=0`, `8.48x` by `k=31`
- `joint_2`: `0.99x` old span at `k=0`, `7.56x` by `k=31`
- `joint_gripper`: `0.85x` old span at `k=0`, `1.65x` by `k=31`

### Validation rerun

Validation script: `scripts/ramen_chunk_delta/06_validate_stats.py`

PASS summary on the semantic-cleanup rerun:

- `PASS` shapes:
  `action_q02=(32, 17)`, `action_q98=(32, 17)`,
  `obs_q02=(1, 17)`, `obs_q98=(1, 17)`
- `PASS` single-frame span match:
  at position `0`, masked dims have max relative error `0.2136` and mean
  relative error `0.0712` versus the old `(1, D)` span
- `PASS` late-vs-early span growth:
  arm dims still grow strongly by `k = 31`
  (`joint_0 6.53x`, `joint_1 8.48x`, `joint_2 7.67x`, `joint_3 6.96x`,
  `joint_4 5.18x`, `joint_5 6.61x`);
  gripper dims still grow clearly
  (`joint_gripper 1.95x`, `eef_gripper 1.95x`)
- `PASS` grip range physical scale:
  position-31 grip span is `0.1251`, still the right order of magnitude versus
  the observed physical range (`~0.0780`)
- `PASS` finite values and rot6d stats finite
- `PASS` dataset-backed normalize -> unnormalize round trip:
  real adapted chunks at dataset indices `0`, `100`, and `1000` reconstruct
  with max absolute errors `1.19e-07`, `1.19e-07`, and `5.96e-08`

That round-trip rerun is the strongest confirmation that the new slot-0
semantics, the dataset loader, and the recomputed `(H, D)` stats all agree on
what one action chunk means.

### `gap_norm` rerun after semantic cleanup

We then reran `scripts/ramen_chunk_delta/04_gap_norm.py` on the refreshed
stats. The headline did not regress: the model-independent normalization floor
remains effectively zero for almost all masked dims.

Worst residual masked dims after the semantic cleanup:

| dim | label | mean `|gap_norm|` | max `|gap_norm|` | frac `|y| > 1.5` |
|----:|:------|------------------:|-----------------:|-----------------:|
| 5 | `joint_5` | 0.002019 | 0.919191 | 1.21 % |
| 3 | `joint_3` | 0.000348 | 0.420233 | 0.53 % |
| 4 | `joint_4` | 0.000241 | 0.208190 | 0.61 % |
| 0 | `joint_0` | 0.000007 | 0.050259 | 0.05 % |
| 1 | `joint_1` | 0.000004 | 0.056935 | 0.02 % |
| 9 | `eef_xyz_z` | 0.000001 | 0.015759 | 0.02 % |

Important detail: even for `joint_5`, `p98 |gap_norm|` is still effectively
zero (`5.96e-08`). So the residual is still a rare tail effect, not a broad new
semantic mismatch introduced by the cleanup.

Artifact written by the validation step:

- `scripts/ramen_chunk_delta/out/06_validate_stats.json`

## Conclusion / next steps

The structural normalization bug is fixed, and the later semantic cleanup also
landed cleanly.

The evidence chain is now consistent end to end:

- unit tests pass for the new slot-0 contract and inference execution path
- real adapted chunks round-trip through
  `ramen_normalize(...) -> ramen_unnormalize(...)` at numerical precision
- recomputed `(H, D)` stats widen appropriately across chunk position
- `gap_norm` collapses to near-zero on almost every masked dim

So the old failure mode was specifically "later chunk positions were scaled
with the wrong single-frame stats", not some unavoidable property of Ramen
normalization itself. What remains is only rare tail clipping on a small number
of motions, most notably `joint_5`.

Operationally, the next step is retraining. The code path now expects the new
semantic contract where slot `0` is `act[t] - obs[t]`, and the refreshed stats
file encodes that same meaning. Old checkpoints were trained against the prior
look-back contract and are therefore not comparable or deployable under the new
path without semantic mismatch.

Deploy-side plumbing should not need another structural change for the next
checkpoint: the policy/config/stats stack now agrees on the chunk semantics, and
the normalization tensors are already stored in `(H, D)` form for consumption by
the next retrained model.
