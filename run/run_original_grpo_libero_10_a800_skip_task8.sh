#!/usr/bin/env bash
# LIBERO-10 Action-GRPO on A800, excluding task 8.
# This is standalone and does not modify or call the original LIBERO-10 launcher.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/playground/dit4dit_libero_all_80000_2a800/checkpoints/steps_80000_pytorch_model.pt}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-/path/to/pretrained/DiT4DiT/Cosmos-Predict2.5-2B}"
RSFT_ANCHOR_DATA="${RSFT_ANCHOR_DATA:-/path/to/data/libero}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LIBERO_TASK_IDS="0-7,9"
RUN_NAME="${RUN_NAME:-libero_10_grpo_a800_skip_task8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/Checkpoints_rl_original/${RUN_NAME}}"
PREFLIGHT_EVAL_EPISODES="${PREFLIGHT_EVAL_EPISODES:-35}"
EVAL_EPISODES="${EVAL_EPISODES:-35}"
REQUIRE_PREFLIGHT_SUCCESS="${REQUIRE_PREFLIGHT_SUCCESS:-1}"
MAX_NO_SIGNAL_UPDATES="${MAX_NO_SIGNAL_UPDATES:-5}"

require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }
require_dir "${PROJECT_DIR}"; require_file "${PYTHON_BIN}"; require_file "${SFT_CKPT}"
require_file "${CONFIG_YAML}"; require_dir "${COSMOS_BASE_MODEL}"; require_dir "${RSFT_ANCHOR_DATA}"

SFT_RUN_DIR="$(dirname "${SFT_CKPT}")"
if [[ ! -f "${SFT_RUN_DIR}/config.yaml" || ! -f "${SFT_RUN_DIR}/dataset_statistics.json" ]]; then
  SFT_RUN_DIR="$(dirname "${SFT_RUN_DIR}")"
fi
require_file "${SFT_RUN_DIR}/config.yaml"; require_file "${SFT_RUN_DIR}/dataset_statistics.json"

source /path/to/cluster/toolshs/set-cluster-env.sh 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || true
module load CUDA/12.2 2>/dev/null || true
source /path/to/cluster/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/LIBERO:${PYTHONPATH:-}"
export DIT4DIT_COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL}" CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_RENDER_GPU_DEVICE_ID="${CUDA_DEVICE}" MUJOCO_EGL_DEVICE_ID="${CUDA_DEVICE}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_grpo_libero_10_a800_skip_task8.out"
STDERR_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_grpo_libero_10_a800_skip_task8.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"

EXTRA_ARGS=(--rsft_anchor_data "${RSFT_ANCHOR_DATA}" --rsft_anchor_weight 0.25 --rsft_anchor_batch_size 8
  --rsft_online_fraction 0.3 --collect_all_chunks --no-global_std)
if [[ "${REQUIRE_PREFLIGHT_SUCCESS}" == "1" ]]; then EXTRA_ARGS+=(--require_preflight_success); else EXTRA_ARGS+=(--no-require_preflight_success); fi

{
  echo "===== LIBERO-10 Action-GRPO on A800 (task 8 excluded) ====="
  echo "suite=libero_10 tasks=${LIBERO_TASK_IDS} gpu=${CUDA_DEVICE}"
  echo "preflight=${PREFLIGHT_EVAL_EPISODES} eval=${EVAL_EPISODES} max_no_signal=${MAX_NO_SIGNAL_UPDATES}"
  echo "checkpoint=${SFT_CKPT} output=${OUTPUT_DIR}"
  "${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true
  "${PYTHON_BIN}" -m dit4dit_rl_trainer.trainer --real --model_variant original \
    --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" --env libero --libero_suite libero_10 \
    --libero_task_id 0 --libero_task_ids "${LIBERO_TASK_IDS}" --libero_task_schedule round_robin \
    --libero_task_block_size 1 --action_unnorm_mode min_max --device cuda --dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}" --grpo_iters 10 --group_size 8 --grpo_inner_epochs 1 \
    --rl_chunks_per_episode 4 --rl_chunk_stride 4 --k_steps 4 --noise_level 0.1 --noise_level_end 0.05 \
    --trainable_scope lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.0 --lr 5e-6 \
    --clip_range 2e-6 --beta_kl 0.08 --save_freq 1 --eval_freq 1 --eval_episodes "${EVAL_EPISODES}" \
    --eval_seed 10000 --max_episode_steps 520 --progress_steps 64 \
    --preflight_eval_episodes "${PREFLIGHT_EVAL_EPISODES}" --max_no_signal_updates "${MAX_NO_SIGNAL_UPDATES}" \
    --rollout_stall_timeout 600 --rollout_heartbeat_seconds 30 --cpu_threads 4 "${EXTRA_ARGS[@]}"
} >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"

echo "GRPO finished successfully. Output: ${OUTPUT_DIR}"
