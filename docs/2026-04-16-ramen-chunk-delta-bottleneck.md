# Ramen Stats Bottleneck on Chunk-Relative Deltas

## TL;DR

Ramen action stats are computed from **single-frame** deltas
(`action[t] - state[t]`), giving `q02 / q98` shape `(1, D)`. At inference,
the model produces a 32-step **chunk-relative** action sequence
(each `action[k]` is a delta from the *current* state for `k = 0..31`),
unnormalized with the same `(1, D)` per-frame stats. This caps the maximum
physical motion the model can express across an entire chunk to roughly one
*single-frame* `q98` per dim — far too small to actually execute a pick
or close a gripper.

Empirically, on `dit_block_tower_config_fix/checkpoint_40000`, the model
fully saturates the normalized output (gripper hits +0.99) but the
unnormalized command moves the gripper joint by only ~0.017 rad (~1°)
across the entire 32-step chunk.

## Where the stats come from

`src/multitask_dit_policy/utils/ramen_normalization.py::compute_ramen_stats`:

```python
data = _bulk_read_columns(dataset, all_keys)        # single-frame columns
all_obs = _assemble_bulk(data, schema.state)        # (N, state_dim)
all_act = _assemble_bulk(data, schema.action)       # (N, action_dim)

# Single-frame delta on shared prefix
all_delta[..., :shared][..., m] = (
    all_act[..., :shared][..., m] - all_obs[..., :shared][..., m]
)

# Unsqueeze T=1 dim → stats shape (1, D)
all_obs = all_obs.unsqueeze(1)
all_delta = all_delta.unsqueeze(1)

stats = {
    "action_q02": torch.quantile(all_delta.float(), 0.02, dim=0),  # (1, D)
    "action_q98": torch.quantile(all_delta.float(), 0.98, dim=0),  # (1, D)
    ...
}
```

So each entry of `all_delta[i]` is "how much the controller is being asked to
move beyond the current state, **for one frame**". At 30 fps, that magnitude
is naturally tiny (e.g. a gripper opening/closing over ~0.5 s contributes
single-frame deltas around 1/30 of the total motion).

For the block-tower checkpoint these come out to:

| dim | meaning | q02 | q98 | unnorm half-range |
|---|---|---|---|---|
| 0 | base joint | -0.0375 | 0.0327 | ~0.035 rad |
| 1 | shoulder | -0.0754 | 0.0617 | ~0.069 rad |
| 2 | elbow | -0.0782 | 0.0515 | ~0.065 rad |
| 6 | gripper joint | -0.0192 | 0.0172 | ~0.018 rad |
| 16 | eef gripper | -0.0036 | 0.0659 | ~0.035 |

These are the magnitudes of motion the unnormalize operation can ever
produce when the model saturates at `±1`.

## Where the chunk lives

At inference, `MultiTaskDiTPolicy._generate_actions` emits an
`(B, n_action_steps, D)` chunk in normalized space. Deploy code then does:

```python
actions_abs = ramen_unnormalize(actions, q02, q98, mask)   # broadcast (1, D) → (1, 32, D)
current = self._current_obs.unsqueeze(0).unsqueeze(0)
shared = min(state_dim, action_dim)
m = norm_mask[:shared]
actions_abs[..., :shared][..., m] += current[..., m]       # delta-from-current
```

So `actions_abs[k]` is interpreted as `current_state + delta_k`, where
`delta_k` = "physical delta from current state to the target k frames in the
future". For `k = 31` at 30 fps, `delta_k` represents a full second of
motion.

**The mismatch:** `delta_k` is calibrated against single-frame stats,
which represent ~33 ms of motion. The unnormalize range can express motion
on the order of one frame; we ask the model to express motion on the order
of 32 frames.

## What the model actually does

Walked through `actions` (normalized) and `actions_abs` (after unnorm +
current) on a real chunk from `checkpoint_40000`:

**Normalized joint-gripper trajectory across the 31-step chunk:**

```
t= 0..9 : -0.13, -0.14, -0.15, -0.16, -0.16, -0.17, -0.17, -0.17, -0.18, -0.18
t=10..19: -0.18, -0.16, -0.14, -0.07, +0.02, +0.14, +0.40, +0.71, +0.88, +0.99
t=20..30: +1.00, +0.99, +0.99, +0.99, +0.99, +0.96, +0.95, +0.96, +0.93, +0.90, +0.88
```

The model **wants** the gripper to close hard — it ramps the normalized
output from -0.18 to a saturated +0.99 and holds. Same trend on dim 16
(eef gripper) and on the major joints (joint 0 swings from +0.12 to -0.94
in normalized space).

**Absolute joint targets after unnormalize across the same chunk:**

| dim | t=0 | t=30 | Δ across full chunk |
|---|---|---|---|
| 0 | -0.1345 | -0.1717 | -0.037 rad (~2°) |
| 1 |  0.3602 |  0.4239 | +0.064 rad (~3.7°) |
| 2 |  0.1589 |  0.1929 | +0.034 rad (~1.9°) |
| 3 | -0.3715 | -0.4034 | -0.032 rad (~1.8°) |
| 6 (gripper) |  0.0201 |  0.0383 | +0.018 rad (~1°) |

The model's "fully close gripper" command becomes a 1° wiggle; "fully swing
base" becomes 2°. The robot will never grasp a block.

## Why this is fundamentally a normalization mismatch (not a model issue)

`ramen_unnormalize` is `x = (y + 1)/2 * (q98 - q02) + q02`. Even with the
training-time `clamp(-1.5, 1.5)` allowing slightly more headroom, the
maximum physical delta the model can express is bounded by `q98` per dim:

```
max_delta ≈ 1.0 * q98     (typical saturation)
         ≈ 1.5 * q98     (training-time clamp upper bound)
```

For the gripper dim that's 0.017 rad (or 0.026 with full clamp). A typical
ARX5 grasp closure is several tenths of a rad. The model is mathematically
unable to express the required motion regardless of how well it has
learned. Same story for any joint whose realistic motion over a 1 s chunk
exceeds its single-frame `q98`, which is most of them.

### Worked examples (block-tower checkpoint, real values from one chunk)

Formula: `physical_delta = (norm + 1) / 2 * (q98 - q02) + q02`

**Joint gripper (dim 6)** — `q02 = -0.0192`, `q98 = +0.0172`,
range `= 0.0364`:

| chunk position | model output (norm) | physical delta computed | absolute target | what the robot does |
|---|---|---|---|---|
| t=0  | -0.126 | `(-0.126+1)/2 * 0.0364 + (-0.0192)` = -0.003 rad | 0.020 | barely change |
| t=10 | -0.176 | `(-0.176+1)/2 * 0.0364 + (-0.0192)` = -0.004 rad | 0.019 | barely change |
| t=19 | +0.992 | `(0.992+1)/2 * 0.0364 + (-0.0192)`  = +0.017 rad | 0.038 | "fully close" → 1° move |
| t=30 | +0.877 | `(0.877+1)/2 * 0.0364 + (-0.0192)`  = +0.015 rad | 0.038 | held closed → 1° move |

The model goes from "open all the way" to *saturating* at "close all the
way". The unnormalize step turns that 1.17-unit normalized swing into a
0.020 rad (≈ 1.1°) physical change. The gripper jaws don't actually close
on a block.

**Joint 0 / base (dim 0)** — `q02 = -0.0375`, `q98 = +0.0327`,
range `= 0.0702`:

| chunk position | model output (norm) | physical delta computed | absolute target |
|---|---|---|---|
| t=0  | +0.122 | `(0.122+1)/2 * 0.0702 + (-0.0375)`  = +0.002 rad | -0.1345 |
| t=15 | -0.062 | `(-0.062+1)/2 * 0.0702 + (-0.0375)` = -0.005 rad | -0.1410 |
| t=20 | +0.012 | `(0.012+1)/2 * 0.0702 + (-0.0375)`  = -0.002 rad | -0.1389 |
| t=30 | -0.937 | `(-0.937+1)/2 * 0.0702 + (-0.0375)` = -0.035 rad | -0.1717 |

The model wants the base to swing hard (saturating at -0.94 by end of
chunk), but the entire 1-second motion adds up to -0.037 rad ≈ 2°.

**Saturation ceiling per dim** (what `norm = +1.0` and `norm = +1.5`
unnormalize to):

| dim | q98 | max_delta @ norm=+1 | max_delta @ norm=+1.5 (clamp) |
|---|---|---|---|
| 0 (base) | 0.0327 | 0.033 rad (1.9°) | 0.050 rad (2.9°) |
| 1 (shoulder) | 0.0617 | 0.062 rad (3.5°) | 0.093 rad (5.3°) |
| 2 (elbow) | 0.0515 | 0.052 rad (3.0°) | 0.078 rad (4.5°) |
| 6 (gripper) | 0.0172 | 0.017 rad (1.0°) | 0.026 rad (1.5°) |

These are the physical motion ceilings *per chunk*, no matter how
confidently the model commands motion.

## Deploy-side confirmation hack

For diagnostic purposes only — not for production — multiplying the
unnormalize range by `n_action_steps` (or another large factor) lets the
model's saturated outputs translate to physically meaningful motion:

```python
# In DiTRamenEngine.__init__, after computing aq02/aq98:
self.action_q02 = (aq02 * unnorm_scale).to(device)
self.action_q98 = (aq98 * unnorm_scale).to(device)
```

If a small sweep of `unnorm_scale ∈ {8, 16, 32}` causes the gripper to
actually close on hardware, the diagnosis is locked in. This is uniform
across dims and not what a properly trained model should need — it's a
yes/no test for the bottleneck.

## Two correct fixes

### Option A: Per-position chunk-relative stats (recommended)

Compute and save stats with shape `(H, D)` instead of `(1, D)`, where
position `k` is calibrated on `action[t+k] - state[t]`:

```python
# Sketch (training side, in compute_ramen_stats)
H = horizon                                 # e.g. 32
chunks = []                                  # list of (H, D) per starting frame
for t in valid_starts:
    chunks.append(all_act[t:t+H] - all_obs[t])  # (H, D)
chunk_deltas = torch.stack(chunks)              # (N_starts, H, D)

stats["action_q02"] = torch.quantile(chunk_deltas, 0.02, dim=0)  # (H, D)
stats["action_q98"] = torch.quantile(chunk_deltas, 0.98, dim=0)  # (H, D)
```

This makes `q98[k]` grow naturally with `k`: position 0 has tiny range,
position 31 has full per-second range. Each chunk position gets normalized
into its own `[-1, 1]` envelope. The model learns to output values whose
magnitudes are roughly uniform across chunk positions, even though the
underlying physical deltas are not.

The deploy side **already supports** `(H, D)` stats — see
`hw_control/new_lerobot_integrations/deploy_policy_dit.py::DiTRamenEngine.__init__`,
which slices `[n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps]` if
`q02.shape[0] > 1`. No deploy changes needed.

Boundary handling for episodes shorter than `H` frames: skip those
starting frames or pad with the last available action (consistent with
how chunks are built during training).

### Option B: Per-step incremental deltas (architectural change)

Change the convention so `action[k]` is the delta from `action[k-1]` (or
from a velocity command), making single-frame `q02 / q98` correctly sized.
At deploy, integrate the chunk step-by-step before sending. This requires
both training-time target conversion and deploy-time integration logic.
More invasive than Option A, and removes some of the "absolute target
prediction" property that diffusion benefits from.

## Action items for the next training run

1. Modify `compute_ramen_stats` to return `(H, D)` action stats per
   Option A. Keep obs stats at `(1, D)` (state is single-step at inference).
2. Bump the cache path / file naming so old `(1, D)` caches don't get
   silently reused.
3. Verify by loading the new `ramen_stats.pt` and asserting:
   - `action_q98[31] - action_q98[0]` is much greater than zero on motion-heavy dims
   - `action_q98[31, gripper_idx]` is on the order of the full grip range, not 0.017
4. Retrain. No deploy-side changes required — the existing
   `aq02.shape[0] > 1` slicing branch already does the right thing.

## References

- `external/multitask_dit_policy/src/multitask_dit_policy/utils/ramen_normalization.py`
- `hw_control/new_lerobot_integrations/deploy_policy_dit.py::DiTRamenEngine`
- Affected checkpoints (Apr 2026): `dit_block_tower_config_fix`,
  `dit_coffee_capsules_config_fix` (and any other checkpoint trained with
  the current `compute_ramen_stats`)
