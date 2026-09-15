#!/bin/bash
#SBATCH --job-name=dit4dit_multistream_2gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 2
#SBATCH --time=24:00:00
#SBATCH -o logs/%j_multistream_train.out
#SBATCH -e logs/%j_multistream_train.err
#
# Submit from PROJECT_DIR with: yhbatch submit_dit4dit_multistream_train.sh
# This script must not be launched through srun or placed in the background.

set -euo pipefail

PROJECT_DIR=/path/to/DiT4DiT-main
CONDA_ENV=/path/to/conda_envs/dit4dit

# This cache was completed by the single-rank RAFT preprocessing job. Two ranks
# may safely read it, but must never attempt to create missing episodes together.
CACHE_DIR="${PROJECT_DIR}/cache/multistream_cache_ed7da336f9c7"
EXPECTED_CACHE_EPISODES=1693

source /path/to/cluster/toolshs/set-cluster-env.sh
module load anaconda3/2023.09
module load CUDA/12.2
source /path/to/cluster/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
mkdir -p logs

if [[ ! -d "${CACHE_DIR}" ]]; then
    echo "ERROR: expected completed multistream cache is missing: ${CACHE_DIR}" >&2
    exit 2
fi

CACHE_FILE_COUNT=$(find "${CACHE_DIR}" -maxdepth 1 -type f -name 'ep_*.pkl' -size +0c | wc -l)
if [[ "${CACHE_FILE_COUNT}" -ne "${EXPECTED_CACHE_EPISODES}" ]]; then
    echo "ERROR: cache is incomplete (${CACHE_FILE_COUNT}/${EXPECTED_CACHE_EPISODES})." >&2
    echo "Run the one-GPU cache build first; do not let two ranks write this cache." >&2
    exit 2
fi

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dit4dit_multistream:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export NCCL_ASYNC_ERROR_HANDLING=1
export DIT4DIT_CACHE_ONLY=0
export DIT4DIT_USE_DEEPSPEED=0
export MASTER_PORT="${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-37} % 1000)))}"

echo "===== DiT4DiT-Multistream two-GPU batch training ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "cache=${CACHE_DIR} (${CACHE_FILE_COUNT}/${EXPECTED_CACHE_EPISODES})"
which python
which torchrun
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1 \
"${CONDA_ENV}/bin/torchrun" \
  --nproc_per_node=2 \
  --master_port "${MASTER_PORT}" \
  dit4dit_multistream/train_multistream.py \
  --config_yaml ./dit4dit_multistream/dit4dit_multistream_libero.yaml \
  --framework.name DiT4DiT-Multistream \
  --framework.cosmos25.base_model /path/to/pretrained/DiT4DiT/Cosmos-Predict2.5-2B \
  --framework.action_model.action_model_type DiT-B \
  --datasets.vla_data.data_root_dir /path/to/data/libero \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.dataset_py multistream \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.num_workers 1 \
  --datasets.vla_data.action_video_freq_ratio 2 \
  --datasets.vla_data.action_horizon 8 \
  --datasets.vla_data.raft_checkpoint "${PROJECT_DIR}/raft_large_C_T_SKHT_V2-ff5fadd5.pth" \
  --trainer.freeze_modules backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae \
  --trainer.max_train_steps 80000 \
  --trainer.is_resume true \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.backbone_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.num_warmup_steps 5000 \
  --framework.cosmos25.extract_layer 17 \
  --framework.cosmos25.flow_matching.time_distribution uniform \
  --framework.cosmos25.flow_matching.high_sigma_ratio null \
  --framework.cosmos25.flow_matching.high_sigma_min null \
  --framework.cosmos25.conditional_frame_timestep 0.0001 \
  --framework.cosmos25.flow_motion_boost 2.0 \
  --framework.cosmos25.use_causal_mask true \
  --run_root_dir ./playground/Checkpoints_multistream_libero \
  --run_id dit4dit_multistream_libero_80000_train

echo "===== DiT4DiT-Multistream batch training finished ====="
date
