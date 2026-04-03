#!/bin/bash
#SBATCH --job-name=mtdit-qwen
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0G
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --partition=workq
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -e

module purge
module load brics/apptainer-multi-node

# Paths — repo on home, heavy assets on scratch. Override any of these at submit time.
HOME_DIR="${HOME_DIR:-/home/u6cr/pravsels.u6cr}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/u6cr/pravsels.u6cr}"
REPO_DIR="${REPO_DIR:-${HOME_DIR}/multitask_dit_policy_stage1_multimodal_abstraction}"
DATA_DIR="${DATA_DIR:-${SCRATCH_DIR}/multitask_dit_policy}"
CONTAINER="${CONTAINER:-${DATA_DIR}/container/multitask-dit-policy_arm64.sif}"

# Shared caches.
HF_CACHE="${HF_CACHE:-${SCRATCH_DIR}/huggingface_cache}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_CACHE}/lerobot}"

# Project-specific paths.
WANDB_DIR="${WANDB_DIR:-${DATA_DIR}}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_DIR}/.cache/wandb}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${SCRATCH_DIR}/.config/wandb}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/outputs}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_DIR}/config/train_coffee_capsules_qwen_pooled.yaml}"
EXTRA_TRAIN_ARGS_B64="${EXTRA_TRAIN_ARGS_B64:-}"
EXTRA_TRAIN_ARGS=""
if [ -n "${EXTRA_TRAIN_ARGS_B64}" ]; then
    EXTRA_TRAIN_ARGS="$(printf '%s' "${EXTRA_TRAIN_ARGS_B64}" | base64 --decode)"
fi

mkdir -p "${HF_CACHE}" "${HF_LEROBOT_HOME}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${OUTPUT_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "Repo dir: ${REPO_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "===================================="
if [ -n "${EXTRA_TRAIN_ARGS}" ]; then
    echo "Extra train args: ${EXTRA_TRAIN_ARGS}"
fi

printf -v TRAIN_CMD 'python3 -m multitask_dit_policy.train --config_path %q --output_dir %q' "${CONFIG_FILE}" "${OUTPUT_DIR}"
if [ -n "${EXTRA_TRAIN_ARGS}" ]; then
    TRAIN_CMD="${TRAIN_CMD} ${EXTRA_TRAIN_ARGS}"
fi

WANDB_TOKEN_FILE="${SCRATCH_DIR}/.wandb_token"
if [ -f "${WANDB_TOKEN_FILE}" ]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_TOKEN_FILE}")"
    echo "W&B API key loaded from ${WANDB_TOKEN_FILE}"
else
    echo "WARNING: No W&B token found at ${WANDB_TOKEN_FILE} — logging disabled"
fi

EXPORT_VARS="export PYTHONPATH=${REPO_DIR}/src:\${PYTHONPATH:-}"
EXPORT_VARS="${EXPORT_VARS} && export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_MODE=offline"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_API_KEY=${WANDB_API_KEY:-}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_DIR=${WANDB_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR}"

set +e
apptainer exec --nv \
    --pwd "${REPO_DIR}" \
    --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" \
    --env "HF_HOME=${HF_CACHE}" \
    --env "HF_HUB_CACHE=${HF_CACHE}/hub" \
    --env "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}" \
    "${CONTAINER}" \
    bash -c "${EXPORT_VARS} && ${TRAIN_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"
echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for details"
    exit ${EXIT_CODE}
fi
