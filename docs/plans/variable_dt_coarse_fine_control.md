# Variable dt: Coarse-Fine Control with a Single Action Expert

## Motivation

Current policy predicts dense action chunks at fixed frequency (~30 Hz). For pick-and-place tasks like block tower, most of the trajectory is free-space transit where dense prediction is wasteful. Contact-rich phases (grasp, placement) need fine temporal resolution but are short.

By predicting `(pose, dt)` pairs instead of just `pose`, a single action expert can implicitly operate in two regimes:
- **Transit**: coarse waypoints ~1s apart, interpolated at inference time
- **Contact**: dense actions at native control frequency

This extends the effective temporal horizon of a fixed-size chunk (e.g. 50 steps covering 50 seconds of transit vs 1.7 seconds of contact) without any architecture changes.

## Key Idea

Append `dt` as an extra action dimension. The model learns the appropriate temporal density from data — no MoE, no routing, no explicit regime switching.

## Data Pipeline

### Step 1: Auto-Segment Demos

For each episode, classify every frame into one of three phases:

1. **Contact** — gripper state changing (opening/closing). Keep at native frequency.
2. **Approach** — EEF velocity local minima near contact events (hover-before-grasp, pre-placement pause). Keep as anchor waypoints.
3. **Transit** — everything else. Subsample to ~1 Hz.

Segmentation signals (all available in current data):
- `gripper_state` (continuous 0-1): threshold rate of change for contact detection
- `eef_velocity` (derived from consecutive EEF poses): local minima → inflection/direction-change points
- Velocity minima also naturally capture approach phases for free

**Critical constraint**: never subsample through a direction change. Local velocity minima are always retained as anchor points regardless of dt label. Otherwise interpolation at inference cuts corners the real trajectory avoided.

### Step 2: Assign dt Labels

For each retained frame, compute `dt` = time to next retained frame:
- Contact frames: `dt ≈ 0.033s` (native 30 Hz)
- Approach anchors: variable, whatever the actual gap is
- Transit waypoints: `dt ≈ 1.0s` (after subsampling)

### Step 3: Augment Action Vector

Current action: `(joint_angles[7], eef_pose[7])` → 17D after rot6d conversion.
New action: `(joint_angles[7], eef_pose[7], dt[1])` → 18D after rot6d conversion.

Schema change is minimal — add one `SchemaEntry` with `dim: 1` to the action list.

## Inference

1. Model outputs 50-step chunk of `(pose, dt)` pairs
2. For steps with large `dt` (transit): interpolate between waypoints using robot motion planner
3. For steps with small `dt` (contact): execute directly at native control frequency
4. Read `dt` from the model's own prediction — no external mode switching needed

## Implementation Order

1. **Segmentation script** (~50-100 lines numpy)
   - Input: episode parquet with EEF pose + gripper state columns
   - Compute EEF velocity magnitude from consecutive poses
   - Find local velocity minima (approach anchors)
   - Detect gripper state transitions (contact phases)
   - Output: frame mask + dt labels per frame

2. **Visualization / validation**
   - Plot a few episodes with color-coded phases (transit/approach/contact)
   - Verify anchor points capture direction changes and hover moments
   - Check dt distribution looks reasonable

3. **Dataset construction**
   - Apply segmentation to all episodes
   - Subsample transit frames to ~1 Hz, keep contact + anchors at native rate
   - Write new parquet with dt column appended
   - Or: construct dt on-the-fly in a dataset transform (avoids rewriting data)

4. **Training config update**
   - Add `dt` dimension to action schema (store as `log(dt)`)
   - Adjust `rot6d_slice` if needed (dt is not a rotation dim)
   - `norm_mask` should include dt — Ramen normalization applies on top of log-transform
   - Add per-step loss weighting: 5x multiplier for steps where `dt < 0.1s`

5. **Inference loop update**
   - Read predicted `log(dt)` from action chunk, exponentiate to recover dt in seconds
   - Threshold at 0.1s: below → execute directly, above → interpolate via motion planner

## Design Decisions

### dt normalization: log-scale

Log-transform dt before Ramen normalization. The bimodal distribution (0.033 vs 1.0) would map to [-1, +1] under linear Ramen with nothing in between — the diffusion model sees a bimodal target that's hard to denoise. Log-scale collapses the gap (log(0.033) ≈ -3.5, log(1.0) = 0) and captures the multiplicative structure of dt. Approach-phase dt values (0.1–0.5s) spread more naturally in log space instead of clustering near zero.

Concretely: store `log(dt)` as the 18th action dimension, apply Ramen normalization on top of that. Inverse at inference: `dt = exp(unnormalized_log_dt)`.

Alternative considered: two learned embeddings (coarse/fine) — cleaner for a pure bimodal case but more invasive and loses the ability to predict intermediate dt.

### Loss weighting: upweight contact steps

A typical chunk has ~45 transit steps and ~5 contact steps. Unweighted, the diffusion loss is dominated by transit — but contact is where success/failure lives. Upweight contact-phase steps 5–10x in the diffusion loss, analogous to how some RL work upweights terminal states.

Implementation: per-step weight vector in the loss, keyed off the dt label. Steps with `dt < threshold` (contact/approach) get higher weight. Exact multiplier to be tuned empirically, starting at 5x.

### Inference dispatch: threshold on predicted dt

The model may predict intermediate dt values (e.g. 0.3s) that don't cleanly map to "interpolate" or "execute directly." Use a hard threshold at inference:
- `dt < 0.1s` → execute at native control frequency (direct)
- `dt >= 0.1s` → interpolate between waypoints via motion planner

0.1s is ~3x the native 30 Hz period — anything below that is effectively dense control. The threshold is a deploy-time knob, not a training parameter. If intermediate dt predictions are common in practice, that's a signal the segmentation labels are noisy or the model needs more data.

## Open Questions

- **Chunk boundary handling**: if the model predicts a coarse waypoint at the end of a chunk, the next chunk needs to pick up from there. Current re-planning should handle this, but worth verifying.
- **Evaluation**: how to fairly compare against fixed-frequency baseline? Wall-clock task completion time? Success rate?

## Prerequisites

- Block tower baseline (job 3707697) converges and can be evaluated
- Failure analysis on eval set to confirm contact-phase failures motivate this work
