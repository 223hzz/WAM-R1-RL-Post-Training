#!/bin/bash
#SBATCH --job-name=dit4dit_awr_object
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --time=24:00:00
#SBATCH -o logs/%j_awr_object.out
#SBATCH -e logs/%j_awr_object.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${SCRIPT_DIR}/run_original_awr_object.sh"
