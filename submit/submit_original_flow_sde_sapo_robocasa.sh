#!/usr/bin/env bash
# Submit the RoboCasa SAPO pilot with yhbatch.
# Example:
# SFT_CKPT=/server/path/best_SFT/pytorch_model.pt N_EPISODES=2 yhbatch submit_original_flow_sde_sapo_robocasa.sh
#SBATCH --job-name=dit4dit_robocasa_sapo
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -Eeuo pipefail
PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PROJECT_DIR}}"
cd "${PROJECT_DIR}"
RUN_SCRIPT="${PROJECT_DIR}/run_original_flow_sde_sapo_robocasa.sh"
[[ -f "${RUN_SCRIPT}" ]] || RUN_SCRIPT="${SCRIPT_DIR}/run_original_flow_sde_sapo_robocasa.sh"
[[ -f "${RUN_SCRIPT}" ]] || { echo "Missing ${RUN_SCRIPT}" >&2; exit 1; }
exec bash "${RUN_SCRIPT}"
