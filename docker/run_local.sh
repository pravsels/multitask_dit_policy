#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

docker run --rm -it --network host --gpus all \
  -v "$PROJECT_DIR/src:/workspace/src" \
  -v "$PROJECT_DIR/tests:/workspace/tests" \
  -v "$PROJECT_DIR/weights:/workspace/weights" \
  -v "$PROJECT_DIR/data:/workspace/data" \
  -v "$PROJECT_DIR/outputs:/workspace/outputs" \
  -e HF_HUB_CACHE=/workspace/weights \
  -e HF_DATASETS_CACHE=/workspace/data \
  -e HF_LEROBOT_HOME=/workspace/data/lerobot \
  multitask-dit-policy:amd64 \
  "${@:-bash}"
