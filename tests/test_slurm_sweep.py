from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pytest
import torch

from multitask_dit_policy.sweeps.slurm_sweep import (
    build_sbatch_command,
    build_study_index_markdown,
    build_train_override_args,
    build_variant_run_name,
    build_variant_run_record,
    load_study_manifest,
    submit_study,
    write_study_logs,
)
from multitask_dit_policy.train import TrainConfig, build_lr_scheduler


MANIFEST_TEXT = """
study_name: coffee_capsules_baseline_v1_lr_sweep
description: Sweep nearby learning rates around the cleaned baseline.
base_config: config/train_coffee_capsules.yaml
output_dir: /scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs
slurm_script: slurm/train.sh
shared_overrides:
  - --device=cuda
variants:
  - name: baseline
    description: Clean baseline config as-is.
    overrides: []
  - name: lr_1e5
    description: Lower learning rate.
    overrides:
      - --policy.optimizer_lr=1e-5
"""


def test_load_study_manifest_and_build_run_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "study.yaml"
        manifest_path.write_text(MANIFEST_TEXT)

        manifest = load_study_manifest(manifest_path)

    assert manifest.study_name == "coffee_capsules_baseline_v1_lr_sweep"
    assert manifest.base_config == "config/train_coffee_capsules.yaml"
    assert build_variant_run_name(manifest.study_name, "lr_1e5") == "coffee_capsules_baseline_v1_lr_sweep-lr_1e5"


def test_build_train_override_args_includes_run_name_and_shared_overrides():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "study.yaml"
        manifest_path.write_text(MANIFEST_TEXT)
        manifest = load_study_manifest(manifest_path)

    variant = manifest.variants[1]
    args = build_train_override_args(manifest, variant)

    assert args[0] == "--run_name=coffee_capsules_baseline_v1_lr_sweep-lr_1e5"
    assert "--device=cuda" in args
    assert "--policy.optimizer_lr=1e-5" in args


def test_build_sbatch_command_encodes_overrides_for_slurm():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "study.yaml"
        manifest_path.write_text(MANIFEST_TEXT)
        manifest = load_study_manifest(manifest_path)

    variant = manifest.variants[1]
    command = build_sbatch_command(manifest, variant)

    assert command[:2] == ["sbatch", "--parsable"]
    export_arg = next(part for part in command if part.startswith("--export="))
    assert export_arg.startswith("--export=NONE,")
    encoded = export_arg.split("EXTRA_TRAIN_ARGS_B64=", 1)[1]
    decoded = base64.b64decode(encoded).decode()
    assert "--run_name=coffee_capsules_baseline_v1_lr_sweep-lr_1e5" in decoded
    assert "--policy.optimizer_lr=1e-5" in decoded


def test_write_study_logs_creates_index_and_per_run_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        manifest_path = repo_root / "config" / "sweeps" / "study.yaml"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(MANIFEST_TEXT)
        manifest = load_study_manifest(manifest_path)

        run_records = [build_variant_run_record(manifest, variant, job_id=None) for variant in manifest.variants]
        study_dir = repo_root / "run_logs" / manifest.study_name
        write_study_logs(study_dir, manifest, run_records)

        index_path = study_dir / "index.md"
        baseline_log = study_dir / "coffee_capsules_baseline_v1_lr_sweep-baseline.md"

        assert index_path.exists()
        assert baseline_log.exists()
        assert "coffee_capsules_baseline_v1_lr_sweep-baseline" in index_path.read_text()
        assert "Clean baseline config as-is." in baseline_log.read_text()


def test_build_study_index_markdown_lists_variants():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "study.yaml"
        manifest_path.write_text(MANIFEST_TEXT)
        manifest = load_study_manifest(manifest_path)

    records = [build_variant_run_record(manifest, variant, job_id=None) for variant in manifest.variants]
    markdown = build_study_index_markdown(manifest, records)

    assert "# coffee_capsules_baseline_v1_lr_sweep" in markdown
    assert "lr_1e5" in markdown
    assert "Lower learning rate." in markdown
    assert "| `baseline` | `coffee_capsules_baseline_v1_lr_sweep-baseline` | `-` | `planned` |" in markdown


def test_submit_study_resolves_relative_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        manifest_path = repo_root / "config" / "sweeps" / "study.yaml"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(MANIFEST_TEXT.replace("/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs", "outputs"))

        records = submit_study(manifest_path, repo_root=repo_root, submit=False)

    assert records[0].output_dir == str(repo_root / "outputs")


def test_build_lr_scheduler_progresses_every_step():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    cfg = TrainConfig(
        dataset_path="dummy_repo",
        train_steps=10,
        lr_scheduler="cosine",
        lr_warmup_steps=2,
        lr_scheduler_min_lr_scale=0.1,
    )

    scheduler = build_lr_scheduler(optimizer, cfg)
    lrs_used = [optimizer.param_groups[0]["lr"]]

    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs_used.append(optimizer.param_groups[0]["lr"])

    assert lrs_used[0] == pytest.approx(0.5)
    assert lrs_used[1] == pytest.approx(1.0)
    assert lrs_used[2] == pytest.approx(1.0)
    assert lrs_used[3] == pytest.approx(0.9554359905560885)
    assert lrs_used[4] < lrs_used[3]
