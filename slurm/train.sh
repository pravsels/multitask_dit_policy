#!/bin/bash
#SBATCH --job-name=mtdit-train
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

# Paths — repo on home, everything heavy on scratch.
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/multitask_dit_policy"
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

# Experiment config — use REPO_DIR, not SCRIPT_DIR (Slurm copies scripts to spool).
CONFIG_FILE="${repo_dir}/config/train_coffee_capsules.yaml"

# Resume: set path to resume from a checkpoint.
LOAD_CKPT_PATH=""

TRAIN_CMD="python3 -m multitask_dit_policy.train \
    --config_path ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR}"

if [ -n "${LOAD_CKPT_PATH}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --checkpoint_path ${LOAD_CKPT_PATH}"
    echo "Resuming from: ${LOAD_CKPT_PATH}"
fi

WANDB_TOKEN_FILE="${scratch_dir}/.wandb_token"
if [ -f "${WANDB_TOKEN_FILE}" ]; then
    WANDB_API_KEY="$(cat "${WANDB_TOKEN_FILE}" | tr -d '[:space:]')"
    echo "W&B API key loaded from ${WANDB_TOKEN_FILE}"
else
    echo "WARNING: No W&B token found at ${WANDB_TOKEN_FILE} — logging disabled"
fi

EXPORT_VARS="export PYTHONUNBUFFERED=1"
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
