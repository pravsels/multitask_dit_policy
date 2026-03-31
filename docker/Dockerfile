FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV HF_HUB_CACHE=/workspace/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    build-essential \
    git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./

# Install deps with a stub package (cached unless pyproject.toml changes)
RUN mkdir -p src/multitask_dit_policy && \
    touch src/multitask_dit_policy/__init__.py && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        .

RUN pip install --no-cache-dir --no-dependencies robocandywrapper

# Copy actual source — only this and the next layer re-run on code changes
COPY src ./src
RUN pip install --no-cache-dir --no-dependencies .
