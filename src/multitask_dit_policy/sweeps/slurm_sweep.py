from __future__ import annotations

import argparse
import base64
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SweepVariant:
    name: str
    description: str
    overrides: list[str] = field(default_factory=list)


@dataclass
class SweepStudy:
    study_name: str
    description: str
    base_config: str
    output_dir: str
    slurm_script: str
    shared_overrides: list[str] = field(default_factory=list)
    variants: list[SweepVariant] = field(default_factory=list)


@dataclass
class VariantRunRecord:
    variant_name: str
    run_name: str
    description: str
    overrides: list[str]
    config_path: str
    output_dir: str
    slurm_script: str
    job_id: str | None
    status: str


def load_study_manifest(path: Path) -> SweepStudy:
    data = yaml.safe_load(path.read_text())
    variants = [SweepVariant(**variant) for variant in data.get("variants", [])]
    return SweepStudy(
        study_name=data["study_name"],
        description=data["description"],
        base_config=data["base_config"],
        output_dir=data["output_dir"],
        slurm_script=data["slurm_script"],
        shared_overrides=data.get("shared_overrides", []),
        variants=variants,
    )


def build_variant_run_name(study_name: str, variant_name: str) -> str:
    return f"{study_name}-{variant_name}"


def build_train_override_args(study: SweepStudy, variant: SweepVariant) -> list[str]:
    return [f"--run_name={build_variant_run_name(study.study_name, variant.name)}", *study.shared_overrides, *variant.overrides]


def build_sbatch_command(study: SweepStudy, variant: SweepVariant) -> list[str]:
    encoded_overrides = base64.b64encode(shlex.join(build_train_override_args(study, variant)).encode()).decode()
    export_fields = [
        "NONE",
        f"CONFIG_FILE={study.base_config}",
        f"OUTPUT_DIR={study.output_dir}",
        f"EXTRA_TRAIN_ARGS_B64={encoded_overrides}",
    ]
    return ["sbatch", "--parsable", f"--export={','.join(export_fields)}", study.slurm_script]


def build_variant_run_record(study: SweepStudy, variant: SweepVariant, job_id: str | None) -> VariantRunRecord:
    return VariantRunRecord(
        variant_name=variant.name,
        run_name=build_variant_run_name(study.study_name, variant.name),
        description=variant.description,
        overrides=build_train_override_args(study, variant),
        config_path=study.base_config,
        output_dir=study.output_dir,
        slurm_script=study.slurm_script,
        job_id=job_id,
        status="submitted" if job_id else "planned",
    )


def build_study_index_markdown(study: SweepStudy, run_records: list[VariantRunRecord]) -> str:
    lines = [
        f"# {study.study_name}",
        "",
        f"## Goal",
        f"- {study.description}",
        "",
        "## Baseline",
        f"- base_config: `{study.base_config}`",
        f"- output_dir: `{study.output_dir}`",
        f"- slurm_script: `{study.slurm_script}`",
        "",
        "## Shared Overrides",
    ]
    if study.shared_overrides:
        lines.extend(f"- `{override}`" for override in study.shared_overrides)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Variants",
            "| variant | run_name | job_id | status | description |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for record in run_records:
        lines.append(
            f"| `{record.variant_name}` | `{record.run_name}` | `{record.job_id or '-'}` | `{record.status}` | {record.description} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def build_variant_log_markdown(record: VariantRunRecord) -> str:
    lines = [
        f"# {record.run_name}",
        "",
        "## Variant",
        f"- variant: `{record.variant_name}`",
        f"- description: {record.description}",
        f"- status: `{record.status}`",
        f"- job_id: `{record.job_id or 'pending'}`",
        "",
        "## Config",
        f"- base_config: `{record.config_path}`",
        f"- output_dir: `{record.output_dir}`",
        f"- slurm_script: `{record.slurm_script}`",
        "",
        "## Overrides",
    ]
    lines.extend(f"- `{override}`" for override in record.overrides)
    lines.extend(["", "## Status", "- created"])
    return "\n".join(lines).rstrip() + "\n"


def write_study_logs(study_dir: Path, study: SweepStudy, run_records: list[VariantRunRecord]) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "index.md").write_text(build_study_index_markdown(study, run_records))
    for record in run_records:
        (study_dir / f"{record.run_name}.md").write_text(build_variant_log_markdown(record))


def _resolve_path(repo_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else repo_root / path)


def submit_study(
    manifest_path: Path,
    *,
    repo_root: Path,
    submit: bool,
) -> list[VariantRunRecord]:
    study = load_study_manifest(manifest_path)
    resolved_study = SweepStudy(
        study_name=study.study_name,
        description=study.description,
        base_config=_resolve_path(repo_root, study.base_config),
        output_dir=_resolve_path(repo_root, study.output_dir),
        slurm_script=_resolve_path(repo_root, study.slurm_script),
        shared_overrides=study.shared_overrides,
        variants=study.variants,
    )

    run_records: list[VariantRunRecord] = []
    for variant in resolved_study.variants:
        job_id = None
        if submit:
            result = subprocess.run(build_sbatch_command(resolved_study, variant), check=True, capture_output=True, text=True)
            job_id = result.stdout.strip()
        run_records.append(build_variant_run_record(resolved_study, variant, job_id=job_id))

    write_study_logs(repo_root / "run_logs" / resolved_study.study_name, resolved_study, run_records)
    return run_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a Slurm sweep study and create run logs.")
    parser.add_argument("--manifest", required=True, help="Path to sweep manifest YAML.")
    parser.add_argument("--submit", action="store_true", help="Actually submit sbatch jobs. Default is dry run.")
    parser.add_argument("--repo_root", default=".", help="Repository root for resolving relative paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    records = submit_study(manifest_path, repo_root=repo_root, submit=args.submit)
    for record in records:
        print(f"{record.run_name}\t{record.job_id or 'dry-run'}\t{record.status}")


if __name__ == "__main__":
    main()
