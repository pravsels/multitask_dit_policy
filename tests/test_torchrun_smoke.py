from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import pytest
import torch.distributed as dist

sys.modules.setdefault(
    "robocandywrapper",
    types.SimpleNamespace(make_dataset_without_config=lambda **kwargs: None),
)

from multitask_dit_policy.train import cleanup_distributed, get_runtime_context, setup_distributed


def test_torchrun_cpu_ddp_smoke():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        pytest.skip("torchrun smoke test requires multiple ranks")

    shared_dir = Path(os.environ["TORCHRUN_SMOKE_DIR"])
    marker = shared_dir / "rank0.txt"
    runtime_context = get_runtime_context("cpu")

    setup_distributed(runtime_context)
    try:
        assert runtime_context.use_ddp is True
        if runtime_context.is_main_process:
            marker.write_text("ready")
        dist.barrier()
        assert marker.read_text() == "ready"
    finally:
        cleanup_distributed()
