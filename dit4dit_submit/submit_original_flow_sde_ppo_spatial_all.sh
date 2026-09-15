#!/bin/bash
# Full LIBERO-Spatial suite Flow-SDE PPO job.
# Override runtime settings with environment variables before submission.
#SBATCH --job-name=dit4dit_spatial_all_fppo
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_spatial_all_flow_sde_ppo.out
#SBATCH -e logs/%j_spatial_all_flow_sde_ppo.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
exec bash "${SCRIPT_DIR}/run_original_flow_sde_ppo_spatial_all.sh"
