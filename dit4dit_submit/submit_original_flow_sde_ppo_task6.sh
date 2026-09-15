#!/bin/bash
#SBATCH --job-name=dit4dit_fppo_t6
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_flow_sde_ppo_task6.out
#SBATCH -e logs/%j_flow_sde_ppo_task6.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Keep the run script's defaults, while allowing overrides via yhbatch --export
export SAMPLING_MODE="${SAMPLING_MODE:-weighted}"
export TARGET_TASK_WEIGHT="${TARGET_TASK_WEIGHT:-0.5}"
export TRAIN_TASK_ID="${TRAIN_TASK_ID:-6}"
export LIBERO_TASK_ID="${LIBERO_TASK_ID:-${TRAIN_TASK_ID}}"

exec bash "${SCRIPT_DIR}/run_original_flow_sde_ppo_task6.sh"
