#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

study_name="coffee_capsules_category1_core_diffusion"
config_file="${CONFIG_FILE:-${REPO_DIR}/config/train_coffee_capsules.yaml}"
output_dir="${OUTPUT_DIR:-/scratch/u6cr/pravsels.u6cr/multitask_dit_policy/outputs}"
train_script="${REPO_DIR}/slurm/train.sh"
log_dir="${REPO_DIR}/run_logs/${study_name}"
submitted_at="$(date -Is --utc)"

variants=(
  baseline
  ddpm_20
  ddpm_50
  ddpm_rope
  ddim_20
  ddim_50
  ddim_rope_20
  ddim_rope_50
)

mkdir -p "${log_dir}"

variant_description() {
  case "$1" in
    baseline) echo "Clean baseline config as-is." ;;
    ddpm_20) echo "Keep DDPM and reduce inference steps to 20." ;;
    ddpm_50) echo "Keep DDPM and reduce inference steps to 50." ;;
    ddpm_rope) echo "Keep DDPM baseline sampler and enable RoPE." ;;
    ddim_20) echo "Switch sampler to DDIM with 20 inference steps." ;;
    ddim_50) echo "Switch sampler to DDIM with 50 inference steps." ;;
    ddim_rope_20) echo "Enable RoPE and use DDIM with 20 inference steps." ;;
    ddim_rope_50) echo "Enable RoPE and use DDIM with 50 inference steps." ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

variant_args() {
  case "$1" in
    baseline) ;;
    ddpm_20)
      printf '%s\n' \
        --policy.objective.noise_scheduler_type=DDPM \
        --policy.objective.num_inference_steps=20
      ;;
    ddpm_50)
      printf '%s\n' \
        --policy.objective.noise_scheduler_type=DDPM \
        --policy.objective.num_inference_steps=50
      ;;
    ddpm_rope)
      printf '%s\n' \
        --policy.transformer.use_rope=true
      ;;
    ddim_20)
      printf '%s\n' \
        --policy.objective.noise_scheduler_type=DDIM \
        --policy.objective.num_inference_steps=20
      ;;
    ddim_50)
      printf '%s\n' \
        --policy.objective.noise_scheduler_type=DDIM \
        --policy.objective.num_inference_steps=50
      ;;
    ddim_rope_20)
      printf '%s\n' \
        --policy.objective.noise_scheduler_type=DDIM \
        --policy.objective.num_inference_steps=20 \
        --policy.transformer.use_rope=true
      ;;
    ddim_rope_50)
      printf '%s\n' \
        --policy.objective.noise_scheduler_type=DDIM \
        --policy.objective.num_inference_steps=50 \
        --policy.transformer.use_rope=true
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

build_encoded_args() {
  local run_name="$1"
  shift

  local escaped_args=""
  printf -v escaped_args '%q ' "--run_name=${run_name}" "$@"
  escaped_args="${escaped_args% }"
  printf '%s' "${escaped_args}" | base64 | tr -d '\n'
}

write_index_header() {
  cat > "${log_dir}/index.md" <<EOF
# ${study_name}

## Goal
- Category 1 core diffusion ablations around the cleaned coffee-capsules baseline.

## Baseline
- base_config: \`${config_file}\`
- output_dir: \`${output_dir}\`
- slurm_script: \`${train_script}\`
- submitted_at: \`${submitted_at}\`

## Variants
| variant | run_name | job_id | status | description |
| --- | --- | --- | --- | --- |
EOF
}

append_index_row() {
  local variant="$1"
  local run_name="$2"
  local job_id="$3"
  local description="$4"

  cat >> "${log_dir}/index.md" <<EOF
| \`${variant}\` | \`${run_name}\` | \`${job_id}\` | \`submitted\` | ${description} |
EOF
}

write_run_log() {
  local variant="$1"
  local run_name="$2"
  local job_id="$3"
  local description="$4"
  shift 4
  local overrides=("$@")

  {
    echo "# ${run_name}"
    echo
    echo "## Variant"
    echo "- variant: \`${variant}\`"
    echo "- description: ${description}"
    echo "- status: \`submitted\`"
    echo "- job_id: \`${job_id}\`"
    echo "- submitted_at: \`${submitted_at}\`"
    echo
    echo "## Config"
    echo "- base_config: \`${config_file}\`"
    echo "- output_dir: \`${output_dir}\`"
    echo "- slurm_script: \`${train_script}\`"
    echo
    echo "## Overrides"
    echo "- \`--run_name=${run_name}\`"
    if ((${#overrides[@]} > 0)); then
      for override in "${overrides[@]}"; do
        echo "- \`${override}\`"
      done
    else
      echo "- none"
    fi
    echo
    echo "## Status"
    echo "- submitted"
  } > "${log_dir}/${run_name}.md"
}

write_index_header

for variant in "${variants[@]}"; do
  description="$(variant_description "${variant}")"
  run_name="${study_name}-${variant}"
  mapfile -t overrides < <(variant_args "${variant}" || true)
  encoded_args="$(build_encoded_args "${run_name}" "${overrides[@]}")"
  job_id="$(
    sbatch --parsable \
      --export="NONE,CONFIG_FILE=${config_file},OUTPUT_DIR=${output_dir},EXTRA_TRAIN_ARGS_B64=${encoded_args}" \
      "${train_script}"
  )"

  append_index_row "${variant}" "${run_name}" "${job_id}" "${description}"
  write_run_log "${variant}" "${run_name}" "${job_id}" "${description}" "${overrides[@]}"
  echo "${run_name} ${job_id}"
done
