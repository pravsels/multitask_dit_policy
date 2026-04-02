# Stage 1 Multimodal Abstraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal multimodal encoder abstraction to `ObservationEncoder` so legacy CLIP/DINO configs still work unchanged while a unified encoder path becomes available for future Qwen integration.

**Architecture:** Keep the downstream DiT contract unchanged by requiring every encoder path to return a flat conditioning tensor. Extend config with an optional multimodal encoder section and route `ObservationEncoder` through either the legacy separate text/vision path or a new unified multimodal path.

**Tech Stack:** Python dataclasses, `draccus` choice registries, PyTorch modules, pytest, Docker

---

### Task 1: Add failing tests for multimodal config and routing

**Files:**
- Modify: `tests/test_observation_encoder.py`

**Step 1: Write the failing test for config selection**

```python
def test_observation_encoder_uses_multimodal_encoder_when_configured():
    ...
```

Cover:
- a fake multimodal encoder is constructed when `cfg.observation_encoder.multimodal` is set
- legacy text encoder is skipped in multimodal mode
- `conditioning_dim` stays consistent with the returned multimodal feature size

**Step 2: Run test to verify it fails**

Run:
`MTDP_SHARED_DIR=/home/user/Desktop/code/multitask_dit_policy ./docker/run_local.sh python3 -m pytest tests/test_observation_encoder.py -k multimodal -v`

Expected:
- FAIL because there is no `multimodal` config or routing path yet

**Step 3: Add a second failing test for legacy compatibility**

```python
def test_observation_encoder_keeps_legacy_text_and_vision_path_by_default():
    ...
```

Cover:
- CLIP text encoder path still initializes when no multimodal config is set
- existing separate-camera logic remains intact

**Step 4: Run focused test selection again**

Run:
`MTDP_SHARED_DIR=/home/user/Desktop/code/multitask_dit_policy ./docker/run_local.sh python3 -m pytest tests/test_observation_encoder.py -k "multimodal or legacy" -v`

Expected:
- at least the new multimodal test fails for the missing feature

### Task 2: Extend configuration for an optional multimodal encoder

**Files:**
- Modify: `src/multitask_dit_policy/utils/configuration.py`
- Test: `tests/test_observation_encoder.py`

**Step 1: Add a multimodal config registry**

Create:
- `MultimodalEncoderConfig`
- one minimal concrete config, e.g. `PooledMultimodalEncoderConfig`

Include:
- output/projection dimension controls if needed
- optional `freeze_backbone` flag for future Qwen use
- a `type`-selected config shape that fits current `draccus` usage

**Step 2: Add the new field to `ObservationEncoderConfig`**

Add:
- `multimodal: MultimodalEncoderConfig | None = None`

Keep:
- `vision`
- `text`

untouched so old configs and checkpoints still load.

**Step 3: Add tiny compatibility helpers if useful**

Example:
- `uses_multimodal_encoder` property on `ObservationEncoderConfig`

**Step 4: Run focused tests**

Run:
`MTDP_SHARED_DIR=/home/user/Desktop/code/multitask_dit_policy ./docker/run_local.sh python3 -m pytest tests/test_observation_encoder.py -k multimodal -v`

Expected:
- still FAIL until routing is implemented

### Task 3: Implement multimodal routing in `ObservationEncoder`

**Files:**
- Modify: `src/multitask_dit_policy/model/observation_encoder.py`
- Test: `tests/test_observation_encoder.py`

**Step 1: Add encoder factories/interfaces**

Create minimal abstractions:
- text encoder factory for legacy path
- multimodal encoder factory for unified path

Do not replace the existing vision factory yet unless necessary.

**Step 2: Route initialization by mode**

Behavior:
- if `config.observation_encoder.multimodal` is set:
  - skip legacy text/vision encoder construction
  - construct exactly one multimodal encoder
- otherwise:
  - preserve current vision + CLIP text behavior

**Step 3: Route `conditioning_dim` calculation**

Behavior:
- multimodal mode uses the multimodal encoder output dimension plus state/env dimensions
- legacy mode keeps current calculation

**Step 4: Route `encode()`**

Behavior:
- multimodal mode:
  - preprocess images using existing preprocessing utilities
  - pass batched observations plus optional task text into the multimodal encoder
  - append the returned features in timestep-aligned form before flattening
- legacy mode:
  - unchanged behavior

**Step 5: Run focused tests**

Run:
`MTDP_SHARED_DIR=/home/user/Desktop/code/multitask_dit_policy ./docker/run_local.sh python3 -m pytest tests/test_observation_encoder.py -v`

Expected:
- PASS on both legacy and new multimodal routing tests

### Task 4: Verify no collateral regressions in the prep branch

**Files:**
- Modify: none unless fixes are needed

**Step 1: Run focused regression checks**

Run:
- `MTDP_SHARED_DIR=/home/user/Desktop/code/multitask_dit_policy ./docker/run_local.sh python3 -m pytest tests/test_observation_encoder.py tests/test_docker_run_local.py tests/test_category1_submit_script.py -v`

Expected:
- PASS

**Step 2: Read lints for touched files**

Check:
- `src/multitask_dit_policy/model/observation_encoder.py`
- `src/multitask_dit_policy/utils/configuration.py`
- `tests/test_observation_encoder.py`

Expected:
- no new diagnostics, or fix any obvious ones immediately
