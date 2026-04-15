#!/bin/bash
#SBATCH --job-name=mtdit-bt-bs320
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
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

# Paths — repo on home, everything heavy on scratch.
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/multitask_dit_policy_stage1_multimodal_abstraction"
data_dir="${scratch_dir}/multitask_dit_policy"
container="${data_dir}/container/multitask-dit-policy_arm64.sif"

# Shared caches (reused across projects on this account).
HF_CACHE="${scratch_dir}/huggingface_cache"
HF_LEROBOT_HOME="${HF_CACHE}/lerobot"

# Project-specific paths.
WANDB_DIR="${data_dir}"
WANDB_CACHE_DIR="${scratch_dir}/.cache/wandb"
WANDB_CONFIG_DIR="${scratch_dir}/.config/wandb"
OUTPUT_DIR="${data_dir}/outputs"

mkdir -p "${HF_CACHE}" "${HF_LEROBOT_HOME}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${OUTPUT_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

CONFIG_FILE="${repo_dir}/config/train_block_tower_bs320_lr3e4.yaml"

printf -v TRAIN_CMD 'torchrun --standalone --nnodes=1 --nproc_per_node=4 -m multitask_dit_policy.train --config_path %q --output_dir %q' "${CONFIG_FILE}" "${OUTPUT_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Output dir: ${OUTPUT_DIR}"

HF_TOKEN_FILE="${home_dir}/.hf_token"
if [ -f "${HF_TOKEN_FILE}" ]; then
    HF_TOKEN="$(cat "${HF_TOKEN_FILE}" | tr -d '[:space:]')"
    echo "HF token loaded from ${HF_TOKEN_FILE}"
else
    echo "WARNING: No HF token found at ${HF_TOKEN_FILE} — unauthenticated requests may be rate-limited"
fi

WANDB_TOKEN_FILE="${scratch_dir}/.wandb_token"
if [ -f "${WANDB_TOKEN_FILE}" ]; then
    WANDB_API_KEY="$(cat "${WANDB_TOKEN_FILE}" | tr -d '[:space:]')"
    echo "W&B API key loaded from ${WANDB_TOKEN_FILE}"
else
    echo "WARNING: No W&B token found at ${WANDB_TOKEN_FILE} — logging disabled"
fi

EXPORT_VARS="export PYTHONPATH=${repo_dir}/src:\${PYTHONPATH:-}"
EXPORT_VARS="${EXPORT_VARS} && export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export OMP_NUM_THREADS=1"
EXPORT_VARS="${EXPORT_VARS} && export TORCH_NCCL_TRACE_BUFFER_SIZE=1000"
EXPORT_VARS="${EXPORT_VARS} && export NCCL_ASYNC_ERROR_HANDLING=1"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_MODE=offline"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_API_KEY=${WANDB_API_KEY:-}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_DIR=${WANDB_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR}"

set +e
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --env "HF_HOME=${HF_CACHE}" \
    --env "HF_HUB_CACHE=${HF_CACHE}/hub" \
    --env "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    "${container}" \
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
