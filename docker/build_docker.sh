#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64)  HOST_ARCH="amd64" ;;
    aarch64) HOST_ARCH="arm64" ;;
esac

PLATFORM="${1:-$HOST_ARCH}"
IMAGE_NAME="multitask-dit-policy"
IMAGE_TAG="${PLATFORM}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} for linux/${PLATFORM}"

if [ "$PLATFORM" = "$HOST_ARCH" ]; then
    docker build \
        --network "$DOCKER_BUILD_NETWORK" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "$REPO_DIR"
else
    docker buildx build \
        --platform "linux/${PLATFORM}" \
        --network "$DOCKER_BUILD_NETWORK" \
        --load \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "$REPO_DIR"
fi

echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
