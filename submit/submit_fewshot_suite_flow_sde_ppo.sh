#!/bin/bash
#SBATCH --job-name=dit4dit_fs_suite_ppo
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_fewshot_suite_flow_sde_ppo.out
#SBATCH -e logs/%j_fewshot_suite_flow_sde_ppo.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${SCRIPT_DIR}/run_fewshot_suite_flow_sde_ppo.sh"
