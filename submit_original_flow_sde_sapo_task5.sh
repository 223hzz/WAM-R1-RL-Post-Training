#!/bin/bash
#SBATCH --job-name=dit4dit_fsapo_t5
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_flow_sde_sapo_task5.out
#SBATCH -e logs/%j_flow_sde_sapo_task5.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${SCRIPT_DIR}/run_original_flow_sde_sapo_task5.sh"
