from __future__ import annotations

import importlib
import sys
import types

import torch
import torch.nn as nn


_rcw_plugins = types.SimpleNamespace(ControlModePlugin=type("ControlModePlugin", (), {}))
sys.modules.setdefault(
    "robocandywrapper",
    types.SimpleNamespace(make_dataset_without_config=lambda **kwargs: None),
)
sys.modules.setdefault("robocandywrapper.plugins", _rcw_plugins)

train_module = importlib.import_module("multitask_dit_policy.train")


class BFloat16Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4, dtype=torch.bfloat16)
        self.head = nn.Linear(4, 4, dtype=torch.float32)


class Float32Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4, dtype=torch.float32)


def test_get_amp_settings_disables_grad_scaler_for_bfloat16_parameters():
    amp_enabled, amp_dtype, use_grad_scaler = train_module.get_amp_settings(
        BFloat16Policy(),
        device="cuda",
        use_amp=True,
    )

    assert amp_enabled is True
    assert amp_dtype == torch.bfloat16
    assert use_grad_scaler is False


def test_get_amp_settings_keeps_float16_scaler_for_float32_parameters():
    amp_enabled, amp_dtype, use_grad_scaler = train_module.get_amp_settings(
        Float32Policy(),
        device="cuda",
        use_amp=True,
    )

    assert amp_enabled is True
    assert amp_dtype == torch.float16
    assert use_grad_scaler is True


def test_get_runtime_context_defaults_to_single_process(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: False)

    assert hasattr(train_module, "get_runtime_context")
    ctx = train_module.get_runtime_context("cpu")

    assert ctx.use_ddp is False
    assert ctx.world_size == 1
    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert ctx.device == "cpu"
    assert ctx.autocast_device_type == "cpu"
    assert ctx.is_main_process is True


def test_get_runtime_context_uses_local_rank_cuda_device(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(train_module.torch.cuda, "is_available", lambda: True)

    assert hasattr(train_module, "get_runtime_context")
    ctx = train_module.get_runtime_context("cuda")

    assert ctx.use_ddp is True
    assert ctx.world_size == 4
    assert ctx.rank == 2
    assert ctx.local_rank == 2
    assert ctx.device == "cuda:2"
    assert ctx.autocast_device_type == "cuda"
    assert ctx.is_main_process is False



def test_cleanup_distributed_destroys_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_module.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(train_module.dist, "destroy_process_group", lambda: calls.append("destroy"))

    assert hasattr(train_module, "cleanup_distributed")
    train_module.cleanup_distributed()

    assert calls == ["barrier", "destroy"]


def test_unwrap_model_returns_inner_module_when_present():
    assert hasattr(train_module, "unwrap_model")
    wrapped = types.SimpleNamespace(module="policy")

    assert train_module.unwrap_model(wrapped) == "policy"
    assert train_module.unwrap_model("policy") == "policy"


def test_load_or_compute_ramen_stats_main_rank_computes_before_barrier(monkeypatch, tmp_path):
    calls = []
    runtime_context = train_module.RuntimeContext(
        use_ddp=True,
        world_size=2,
        rank=0,
        local_rank=0,
        device="cuda:0",
        autocast_device_type="cuda",
        is_main_process=True,
    )

    monkeypatch.setattr(
        train_module,
        "compute_ramen_stats",
        lambda *args, **kwargs: calls.append("compute") or {"norm_mask": torch.ones(1)},
    )
    monkeypatch.setattr(train_module.dist, "barrier", lambda: calls.append("barrier"))

    from multitask_dit_policy.utils.configuration import DatasetSchema

    assert hasattr(train_module, "load_or_compute_ramen_stats")
    train_module.load_or_compute_ramen_stats(
        dataset=[],
        schema=DatasetSchema(),
        norm_mask=torch.ones(1, dtype=torch.bool),
        cache_path=tmp_path / "ramen_stats.pt",
        device="cuda:0",
        runtime_context=runtime_context,
    )

    assert calls == ["compute", "barrier"]


def test_load_or_compute_ramen_stats_non_main_rank_waits_for_barrier(monkeypatch, tmp_path):
    calls = []
    runtime_context = train_module.RuntimeContext(
        use_ddp=True,
        world_size=2,
        rank=1,
        local_rank=1,
        device="cuda:1",
        autocast_device_type="cuda",
        is_main_process=False,
    )

    monkeypatch.setattr(train_module.dist, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(
        train_module,
        "compute_ramen_stats",
        lambda *args, **kwargs: calls.append("compute") or {"norm_mask": torch.ones(1)},
    )

    from multitask_dit_policy.utils.configuration import DatasetSchema

    assert hasattr(train_module, "load_or_compute_ramen_stats")
    train_module.load_or_compute_ramen_stats(
        dataset=[],
        schema=DatasetSchema(),
        norm_mask=torch.ones(1, dtype=torch.bool),
        cache_path=tmp_path / "ramen_stats.pt",
        device="cuda:1",
        runtime_context=runtime_context,
    )

    assert calls == ["barrier", "compute"]
