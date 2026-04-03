from __future__ import annotations

import importlib
import sys
import types

import torch
import torch.nn as nn


sys.modules.setdefault(
    "robocandywrapper",
    types.SimpleNamespace(make_dataset_without_config=lambda **kwargs: None),
)

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
