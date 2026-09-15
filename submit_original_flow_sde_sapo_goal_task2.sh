#!/bin/bash
# Submit Flow-SDE SAPO post-training for LIBERO-Goal task 2.
# Example:
#   yhbatch submit_original_flow_sde_sapo_goal_task2.sh
#SBATCH --job-name=dit4dit_fsapo_goal_t2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_flow_sde_sapo_goal_task2.out
#SBATCH -e logs/%j_flow_sde_sapo_goal_task2.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LIBERO_SUITE="${LIBERO_SUITE:-libero_goal}"
export TRAIN_TASK_ID="${TRAIN_TASK_ID:-2}"
export LIBERO_TASK_ID="${LIBERO_TASK_ID:-${TRAIN_TASK_ID}}"
export LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-all}"
export SAMPLING_MODE="${SAMPLING_MODE:-weighted}"
export TARGET_TASK_WEIGHT="${TARGET_TASK_WEIGHT:-0.5}"
exec bash "${SCRIPT_DIR}/run_original_flow_sde_sapo_goal_task2.sh"
