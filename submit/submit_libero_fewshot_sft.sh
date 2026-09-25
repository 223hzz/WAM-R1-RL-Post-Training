#!/bin/bash
#SBATCH --job-name=dit4dit_fewshot_sft
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_fewshot_sft.out
#SBATCH -e logs/%j_fewshot_sft.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${SCRIPT_DIR}/run_libero_fewshot_sft.sh"
