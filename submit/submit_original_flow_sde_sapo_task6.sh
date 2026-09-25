#!/bin/bash
# Submit Flow-SDE SAPO post-training for LIBERO-10 task 6.
# For example:
#   TRAIN_TASK_ID=6 SAMPLING_MODE=weighted TARGET_TASK_WEIGHT=0.5 ROLLOUT_EPISODES=32 \
#   TRAINABLE_SCOPE=lora LR=5e-5 PPO_EPOCHS=1 SFT_ANCHOR_WEIGHT=0.25 \
#   K_STEPS=16 EXECUTION_HORIZON=4 NOISE_LEVEL=0.1 NOISE_LEVEL_END=0.05 \
#   MIN_SUCCESSFUL_ROLLOUTS=4 CRITIC_WARMUP_STEPS=25 UPDATES=20 \
#   EVAL_FREQ_UPDATES=2 EVAL_EPISODES=20 yhbatch submit_original_flow_sde_sapo_task6.sh
#SBATCH --job-name=dit4dit_fsapo_t6_l10
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_flow_sde_sapo_task6.out
#SBATCH -e logs/%j_flow_sde_sapo_task6.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LIBERO_SUITE="${LIBERO_SUITE:-libero_10}"
export TRAIN_TASK_ID="${TRAIN_TASK_ID:-6}"
export LIBERO_TASK_ID="${LIBERO_TASK_ID:-${TRAIN_TASK_ID}}"
export SAMPLING_MODE="${SAMPLING_MODE:-weighted}"
export TARGET_TASK_WEIGHT="${TARGET_TASK_WEIGHT:-0.5}"
exec bash "${SCRIPT_DIR}/run_original_flow_sde_sapo_task6.sh"
