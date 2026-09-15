#!/bin/bash
#SBATCH --job-name=dit4dit_rl_grpo
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_rl_grpo.out
#SBATCH -e logs/%j_rl_grpo.err
# ============================================================
# Causal-Action-GRPO post-training script (sbatch submission version)
#
# Two stages: few-shot SFT cold start -> online Causal-Action-GRPO evolution
#
# Usage:
#   1. Mock smoke test (validates the training loop, no checkpoint needed):
#      bash submit_dit4dit_rl_train.sh mock
#
#   2. Real training (requires a checkpoint produced by the SFT stage):
#      bash submit_dit4dit_rl_train.sh real
#
# Prerequisites (real training):
#   - dit4dit_multistream SFT training has produced a checkpoint
#   - the LIBERO simulation environment is installed
# ============================================================

set -euo pipefail

# ---- environment activation ----
source /path/to/cluster/toolshs/set-cluster-env.sh
module load anaconda3/2023.09
module load CUDA/12.2
source /path/to/cluster/anaconda3/etc/profile.d/conda.sh

CONDA_ENV=/path/to/conda_envs/dit4dit
conda activate "${CONDA_ENV}"

PROJECT_DIR=/path/to/DiT4DiT-main
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dit4dit_multistream:${PROJECT_DIR}/LIBERO:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "===== Causal-Action-GRPO Post-Training ====="
date
hostname
echo "CONDA_ENV=${CONDA_ENV}"
echo "PROJECT_DIR=${PROJECT_DIR}"
which python
nvidia-smi

MODE=${1:-mock}

# ============================================================
# Mode 1: mock smoke test
#   Runs the full SFT->GRPO flow with lightweight mock components
#   Needs no SFT checkpoint, no simulation environment, no real VAE
#   Validates that the training-loop logic is bug-free
# ============================================================
if [ "${MODE}" = "mock" ]; then
    echo "===== Running Mock Smoke Test ====="
    CUDA_VISIBLE_DEVICES=0 \
    "${CONDA_ENV}/bin/python" -m dit4dit_rl_trainer.trainer --mock
    echo "===== Mock Smoke Test Complete ====="
    exit 0
fi

# ============================================================
# Mode 2: real training
#   Loads the SFT checkpoint, freezes the backbone, trains only the Action DiT
#   Improves the policy by interacting with the simulation environment via online GRPO
#
#   Note: the following variables must be set first:
#     SFT_CKPT  - path to the checkpoint produced by the SFT stage
#     (optional) SFT_DATA - SFT data path (unset means skip SFT and go straight to GRPO)
# ============================================================
if [ "${MODE}" = "real" ]; then
    # ---- configuration (adjust as needed) ----
    SFT_CKPT="${SFT_CKPT:-./playground/Checkpoints_multistream_libero/dit4dit_multistream_libero_80000_train/checkpoints/latest_checkpoint.pt}"

    echo "===== Running Real GRPO Training ====="
    echo "SFT_CKPT=${SFT_CKPT}"

    CUDA_VISIBLE_DEVICES=0 \
    "${CONDA_ENV}/bin/python" -m dit4dit_rl_trainer.trainer --real \
        --config_yaml ./dit4dit_multistream/dit4dit_multistream_libero.yaml \
        --checkpoint "${SFT_CKPT}" \
        --env libero \
        --device cuda \
        --dtype bfloat16 \
        --output_dir ./playground/Checkpoints_rl_grpo \
        --grpo_iters 10000 \
        --group_size 8 \
        --k_steps 4 \
        --noise_level 0.7 \
        --lr 1e-5 \
        --clip_range 1e-3 \
        --beta_kl 0.04 \
        --save_freq 500 \
        --eval_freq 200 \
        --raft_checkpoint "${PROJECT_DIR}/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

    echo "===== Real GRPO Training Complete ====="
    exit 0
fi

echo "Usage: bash submit_dit4dit_rl_train.sh [mock|real]"
echo "  mock - Mock smoke test (no checkpoint/environment needed)"
echo "  real - Real training (needs SFT checkpoint + LIBERO environment)"
exit 1
