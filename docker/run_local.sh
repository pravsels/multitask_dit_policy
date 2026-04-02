#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SOURCE_PROJECT_DIR="${MTDP_PROJECT_DIR:-$PROJECT_DIR}"
SHARED_PROJECT_DIR="${MTDP_SHARED_DIR:-$SOURCE_PROJECT_DIR}"
IMAGE_REF="${MTDP_IMAGE_REF:-multitask-dit-policy:amd64}"

docker run --rm -it --network host --gpus all \
  -v "$SOURCE_PROJECT_DIR/src:/workspace/src" \
  -v "$SOURCE_PROJECT_DIR/tests:/workspace/tests" \
  -v "$SHARED_PROJECT_DIR/weights:/workspace/weights" \
  -v "$SHARED_PROJECT_DIR/data:/workspace/data" \
  -v "$SHARED_PROJECT_DIR/outputs:/workspace/outputs" \
  -e HF_HUB_CACHE=/workspace/weights \
  -e HF_DATASETS_CACHE=/workspace/data \
  -e HF_LEROBOT_HOME=/workspace/data/lerobot \
  "$IMAGE_REF" \
  "${@:-bash}"
