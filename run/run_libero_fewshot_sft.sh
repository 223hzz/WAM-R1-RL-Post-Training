#!/bin/bash
# Reproducible SFT cold start for a LIBERO suite. The manifest is the experiment's
# data split artifact; preserve it together with every resulting checkpoint.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
DATA_ROOT="${DATA_ROOT:-/path/to/data/libero}"
DATA_MIX="${DATA_MIX:-libero_spatial}"
DEMOS_PER_TASK="${DEMOS_PER_TASK:-10}"
SUBSET_SEED="${SUBSET_SEED:-20260820}"
SFT_SEED="${SFT_SEED:-42}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-25000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/pretrained/DiT4DiT/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt}"
RUN_ID="${RUN_ID:-dit4dit_${DATA_MIX}_transfer_gr1_k${DEMOS_PER_TASK}_b${PER_DEVICE_BATCH_SIZE}_s${MAX_TRAIN_STEPS}_seed${SUBSET_SEED}}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${PROJECT_DIR}/playground/Checkpoints_fewshot}"
MANIFEST="${MANIFEST:-${RUN_ROOT_DIR}/${RUN_ID}/fewshot_manifest.json}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
COSMOS_MODEL="${COSMOS_MODEL:-/path/to/pretrained/DiT4DiT/Cosmos-Predict2.5-2B}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

[[ -f "${CONFIG_YAML}" ]] || { echo "Missing config: ${CONFIG_YAML}" >&2; exit 1; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing data root: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${PRETRAINED_CKPT}" ]] || { echo "Missing GR1 transfer checkpoint: ${PRETRAINED_CKPT}" >&2; exit 1; }
mkdir -p "$(dirname "${MANIFEST}")" "${PROJECT_DIR}/logs"

source /path/to/cluster/toolshs/set-cluster-env.sh 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || true
module load CUDA/12.2 2>/dev/null || true
source /path/to/cluster/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${CONDA_ENV}/bin/python" utils/create_libero_fewshot_manifest.py \
  --data_root "${DATA_ROOT}" --data_mix "${DATA_MIX}" --demos_per_task "${DEMOS_PER_TASK}" \
  --seed "${SUBSET_SEED}" --output "${MANIFEST}"

TRAIN_ARGS=(
  DiT4DiT/training/train.py
  --config_yaml "${CONFIG_YAML}"
  --seed "${SFT_SEED}"
  --framework.name DiT4DiT
  --framework.cosmos25.base_model "${COSMOS_MODEL}"
  --framework.action_model.action_model_type DiT-B
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.episode_subset_manifest "${MANIFEST}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}"
  --trainer.pretrained_shape_filter true
  --trainer.freeze_modules backbone_interface
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --trainer.use_deepspeed false
  --trainer.logging_frequency 10
  --trainer.learning_rate.action_model 1e-4
  --trainer.num_warmup_steps "${WARMUP_STEPS}"
  --framework.cosmos25.extract_layer 17
  --framework.cosmos25.flow_matching.time_distribution uniform
  --framework.cosmos25.flow_matching.high_sigma_ratio null
  --framework.cosmos25.flow_matching.high_sigma_min null
  --framework.cosmos25.conditional_frame_timestep 0.0001
  --datasets.vla_data.action_video_freq_ratio 2
  --run_root_dir "${RUN_ROOT_DIR}" --run_id "${RUN_ID}"
)

# A single A800 should use one ordinary Python process.  torchrun is only
# needed when the user explicitly requests multi-process training.
if (( NPROC_PER_NODE > 1 )); then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CONDA_ENV}/bin/torchrun" \
    --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${CONDA_ENV}/bin/python" \
    "${TRAIN_ARGS[@]}"
fi
