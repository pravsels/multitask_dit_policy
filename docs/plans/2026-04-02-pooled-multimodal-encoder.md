# Pooled Multimodal Encoder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the placeholder multimodal seam with a real pooled Hugging Face image-text encoder path that preserves the current flat conditioning contract.

**Architecture:** Add a `PooledHuggingFaceMultimodalEncoder` that uses `AutoProcessor` plus `AutoModelForImageTextToText`, freezes the pretrained backbone by default, pools the final hidden states into one feature vector per timestep, then projects to a fixed output dimension. `ObservationEncoder` will continue to own state concatenation and temporal flattening.

**Tech Stack:** PyTorch, Hugging Face `transformers`, torchvision image conversion, pytest, Docker

---

### Task 1: Add failing tests for the real pooled encoder path

**Files:**
- Modify: `tests/test_observation_encoder.py`

**Step 1: Write a failing factory test**

Add a test that:
- configures `cfg.observation_encoder.multimodal` with the real pooled config class
- verifies `create_multimodal_encoder()` returns the pooled Hugging Face encoder
- monkeypatches Hugging Face auto classes so no model download is needed

**Step 2: Run the focused test to verify it fails**

Run:
`docker run ... python3 -m pytest tests/test_observation_encoder.py -k pooled -v`

Expected:
- FAIL because the pooled encoder class/factory does not exist yet

**Step 3: Write a failing forward-path test**

Add a test that:
- monkeypatches the processor and model to return deterministic hidden states
- builds a small fake batch with images plus task strings
- verifies the pooled encoder returns `(B, T, output_dim)`
- verifies the processor receives one text prompt per timestep

**Step 4: Run the focused test selection again**

Run:
`docker run ... python3 -m pytest tests/test_observation_encoder.py -k pooled -v`

Expected:
- FAIL for the missing implementation, not because of the test harness

### Task 2: Implement the pooled encoder and wire the factory

**Files:**
- Modify: `src/multitask_dit_policy/model/observation_encoder.py`
- Modify: `src/multitask_dit_policy/utils/configuration.py`
- Test: `tests/test_observation_encoder.py`

**Step 1: Extend multimodal config**

Update `PooledMultimodalEncoderConfig` with only the fields needed now:
- `model`
- `output_dim`
- `freeze_backbone`
- `max_text_length`

Keep validation narrow and YAGNI.

**Step 2: Implement `PooledHuggingFaceMultimodalEncoder`**

Behavior:
- instantiate `AutoProcessor.from_pretrained(model)`
- instantiate `AutoModelForImageTextToText.from_pretrained(model)` when pretrained
- optionally freeze backbone params
- create a projection layer from model hidden size to `output_dim`

**Step 3: Implement pooling**

Behavior:
- flatten `(B, T, N, C, H, W)` into per-timestep samples
- repeat task text across timesteps
- convert camera images into the processor input shape expected for multi-image samples
- call processor with padding/truncation and `return_tensors="pt"`
- call the model with `output_hidden_states=True`
- mean-pool the last hidden states using `attention_mask` when available
- project and reshape back to `(B, T, output_dim)`

**Step 4: Wire `create_multimodal_encoder()`**

Behavior:
- return the pooled encoder for the pooled config
- keep the explicit `NotImplementedError` for future multimodal encoder types

**Step 5: Run focused tests**

Run:
`docker run ... python3 -m pytest tests/test_observation_encoder.py -k pooled -v`

Expected:
- PASS

### Task 3: Run Stage 1 regression checks

**Files:**
- Modify: none unless regressions appear

**Step 1: Run the existing focused suite**

Run:
- `docker run ... python3 -m pytest tests/test_observation_encoder.py tests/test_docker_run_local.py tests/test_category1_submit_script.py -v`

Expected:
- PASS

**Step 2: Read lints for touched files**

Check:
- `src/multitask_dit_policy/model/observation_encoder.py`
- `src/multitask_dit_policy/utils/configuration.py`
- `tests/test_observation_encoder.py`

Expected:
- no new actionable diagnostics
