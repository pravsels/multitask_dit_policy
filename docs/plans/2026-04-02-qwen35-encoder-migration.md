# Qwen3.5 Encoder Migration Design

## Why this branch exists

The current policy stack is tightly coupled to CLIP-style conditioning:

- `src/multitask_dit_policy/model/observation_encoder.py` builds vision features from `timm` and text features from a hardcoded `CLIPTextEncoder`
- `src/multitask_dit_policy/utils/configuration.py` defaults both text and vision encoders to CLIP
- existing train configs, such as `config/train_coffee_capsules.yaml`, explicitly select CLIP for both modalities

That design is serviceable for the baseline, but it makes it awkward to adopt a modern multimodal model such as [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), which is positioned as a unified vision-language model rather than two independent CLIP towers.

## Current constraints

The current implementation bakes in several assumptions that will block a clean Qwen migration:

1. Vision and text are instantiated through different patterns.
   Vision uses a registry-like factory (`create_vision_encoder`), while text directly constructs `CLIPTextEncoder`.

2. Text conditioning is always a single pooled vector.
   `ObservationEncoder.encode()` expands one text embedding across all observation timesteps. That is fine for CLIP, but a VLM path may want token-level outputs or fused visual-text features.

3. The conditioning contract is "flat vector only".
   Everything is concatenated into one dense vector before entering the DiT. A Qwen-backed path may still end there, but we should make the projection boundary explicit instead of assuming CLIP-shaped pooled embeddings.

4. Backward compatibility matters.
   Existing checkpoints and YAML configs should keep working unchanged while the new path is introduced.

## Recommended migration shape

Do this in stages rather than swapping CLIP for Qwen in one shot.

### Stage 1: Generalize encoder interfaces

Refactor `ObservationEncoder` around encoder roles instead of model names:

- `VisionFeatureEncoder`: image-only encoder returning `(B, image_feature_dim)`
- `TextFeatureEncoder`: text-only encoder returning `(B, text_feature_dim)` or token features plus pooling
- `MultimodalConditionEncoder`: optional encoder that can consume both images and task text and return the final conditioning feature directly

The important prep step is to make `ObservationEncoder` choose one of two modes:

- legacy mode: separate vision + text encoders
- multimodal mode: one encoder owns both modalities

That preserves CLIP and Dino behavior while opening a slot for Qwen.

### Stage 2: Extend config without breaking old YAMLs

Keep the current fields valid:

- `observation_encoder.vision`
- `observation_encoder.text`

Add a new optional section:

- `observation_encoder.multimodal`

Rules:

- if `multimodal` is unset, run the existing separate-encoder path
- if `multimodal` is set, skip separate text/vision encoders and let the multimodal encoder provide the conditioning features
- old JSON checkpoints should still deserialize because the existing config fields remain unchanged

This is the safest compatibility point for experimentation.

### Stage 3: Add a Qwen-backed implementation

The first Qwen implementation should be intentionally narrow:

- frozen pretrained backbone
- trainable projection head into `config.transformer.hidden_dim`
- support batched task strings plus one or more camera images
- produce a fixed-size conditioning vector so the downstream DiT does not need to change immediately

Avoid token-level conditioning in the first pass. It would likely require a larger transformer interface change and is not necessary to validate whether the newer encoder materially helps policy quality.

### Stage 4: Decide fusion strategy empirically

There are two plausible Qwen integration styles:

1. Use Qwen as a multimodal feature extractor and pool to one vector.
   Lowest risk and easiest to slot into the current architecture.

2. Feed token-level multimodal features into a redesigned policy conditioner.
   Higher upside, but this is a bigger architecture change and should be a separate experiment.

Recommendation: start with pooled multimodal features, then only revisit token-level conditioning if the pooled path saturates.

## Validation plan

Before any full Qwen integration:

- add unit tests covering config selection between legacy and multimodal paths
- add encoder construction tests that do not require downloading weights
- preserve current CLIP/Dino behavior in existing tests
- add one smoke path for a fake multimodal encoder that proves `ObservationEncoder.encode()` can route through the unified interface

## Practical branch workflow

This branch already includes a worktree-friendly Docker runner update so experiments can happen from the worktree while still reusing the shared `data`, `weights`, and `outputs` directories from the main checkout.

Recommended invocation pattern from this branch:

```bash
MTDP_SHARED_DIR=/home/user/Desktop/code/multitask_dit_policy ./docker/run_local.sh
```

That keeps `src` and `tests` sourced from the worktree while avoiding duplicate caches and datasets.
