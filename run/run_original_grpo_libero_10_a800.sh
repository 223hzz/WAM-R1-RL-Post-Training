#!/usr/bin/env bash
# Single-mode Action-GRPO launcher for LIBERO-10 on an A800 node.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/playground/dit4dit_libero_all_80000_2a800/checkpoints/steps_80000_pytorch_model.pt}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-/path/to/pretrained/DiT4DiT/Cosmos-Predict2.5-2B}"
ANCHOR_DATA="${ANCHOR_DATA:-/path/to/data/libero}"
RSFT_ANCHOR_DATA="${RSFT_ANCHOR_DATA:-${ANCHOR_DATA}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

LIBERO_SUITE="${LIBERO_SUITE:-libero_10}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-all}"
RUN_NAME="${RUN_NAME:-libero_10_grpo_a800_full_trajectory}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/Checkpoints_rl_original/${RUN_NAME}}"
RESUME_RL="${RESUME_RL:-}"

# One fixed GRPO configuration; override only paths/device/run name as needed.
GRPO_ITERS=10
GROUP_SIZE=8
GRPO_INNER_EPOCHS=1
POLICY_STEPS_PER_ITERATION=1
RL_CHUNKS_PER_EPISODE=4
COLLECT_ALL_CHUNKS=1
RL_CHUNK_STRIDE=4
K_STEPS=4
MAX_EPISODE_STEPS=520
PREFLIGHT_EVAL_EPISODES=35
EVAL_EPISODES=35
EVAL_FREQ=1
SAVE_FREQ=1
NOISE_LEVEL=0.1
NOISE_LEVEL_END=0.05
SFT_ANCHOR_WEIGHT=0.25
SFT_ANCHOR_BATCH_SIZE=8
SFT_ONLINE_FRACTION=0.3
TRAINABLE_SCOPE=lora
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.0
RL_LR=5e-6
CLIP_RANGE=2e-6
BETA_KL=0.08
GLOBAL_STD=0
ACTION_UNNORM_MODE=min_max
MAX_NO_SIGNAL_UPDATES=5
ROLLOUT_STALL_TIMEOUT=600
ROLLOUT_HEARTBEAT_SECONDS=30
CPU_THREADS=4
REQUIRE_PREFLIGHT_SUCCESS=1

require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }
[[ "${LIBERO_SUITE}" == "libero_10" ]] || { echo "This launcher requires LIBERO_SUITE=libero_10" >&2; exit 2; }
[[ "${LIBERO_TASK_IDS}" == "all" || "${LIBERO_TASK_IDS}" == "0-9" ]] || { echo "LIBERO_TASK_IDS must be all or 0-9" >&2; exit 2; }
require_dir "${PROJECT_DIR}"
require_file "${PYTHON_BIN}"
require_file "${SFT_CKPT}"
require_file "${CONFIG_YAML}"
require_dir "${COSMOS_BASE_MODEL}"
require_dir "${RSFT_ANCHOR_DATA}"

SFT_RUN_DIR="$(dirname "${SFT_CKPT}")"
if [[ ! -f "${SFT_RUN_DIR}/config.yaml" || ! -f "${SFT_RUN_DIR}/dataset_statistics.json" ]]; then
  SFT_RUN_DIR="$(dirname "${SFT_RUN_DIR}")"
fi
require_file "${SFT_RUN_DIR}/config.yaml"
require_file "${SFT_RUN_DIR}/dataset_statistics.json"

source /path/to/cluster/toolshs/set-cluster-env.sh 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || true
module load CUDA/12.2 2>/dev/null || true
source /path/to/cluster/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/LIBERO:${PYTHONPATH:-}"
export DIT4DIT_COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_RENDER_GPU_DEVICE_ID="${MUJOCO_RENDER_GPU_DEVICE_ID:-${CUDA_DEVICE}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_DEVICE}}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_grpo_libero_10_a800.out"
STDERR_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_grpo_libero_10_a800.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"

EXTRA_ARGS=(
  --rsft_anchor_data "${RSFT_ANCHOR_DATA}"
  --rsft_anchor_weight "${SFT_ANCHOR_WEIGHT}"
  --rsft_anchor_batch_size "${SFT_ANCHOR_BATCH_SIZE}"
  --rsft_online_fraction "${SFT_ONLINE_FRACTION}"
  --collect_all_chunks
  --no-global_std
  --require_preflight_success
)
[[ -n "${RESUME_RL}" ]] && EXTRA_ARGS+=(--resume_rl "${RESUME_RL}")

{
  echo "===== LIBERO-10 Action-GRPO on A800 ====="
  echo "suite=${LIBERO_SUITE} tasks=${LIBERO_TASK_IDS} gpu=${CUDA_DEVICE}"
  echo "iters=${GRPO_ITERS} group=${GROUP_SIZE} inner_epochs=${GRPO_INNER_EPOCHS}"
  echo "full_trajectory_chunks=true legacy_chunks_per_episode=${RL_CHUNKS_PER_EPISODE}"
  echo "preflight=${PREFLIGHT_EVAL_EPISODES} eval=${EVAL_EPISODES} anchor_batch=${SFT_ANCHOR_BATCH_SIZE}"
  echo "checkpoint=${SFT_CKPT} output=${OUTPUT_DIR}"
  "${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true
  "${PYTHON_BIN}" -m dit4dit_rl_trainer.trainer --real \
    --model_variant original --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" \
    --env libero --libero_suite "${LIBERO_SUITE}" --libero_task_id 0 \
    --libero_task_ids "${LIBERO_TASK_IDS}" --libero_task_schedule round_robin --libero_task_block_size 1 \
    --action_unnorm_mode "${ACTION_UNNORM_MODE}" --device cuda --dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}" --grpo_iters "${GRPO_ITERS}" --group_size "${GROUP_SIZE}" \
    --grpo_inner_epochs "${GRPO_INNER_EPOCHS}" --rl_chunks_per_episode "${RL_CHUNKS_PER_EPISODE}" \
    --rl_chunk_stride "${RL_CHUNK_STRIDE}" --k_steps "${K_STEPS}" --noise_level "${NOISE_LEVEL}" \
    --noise_level_end "${NOISE_LEVEL_END}" --trainable_scope "${TRAINABLE_SCOPE}" \
    --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
    --lr "${RL_LR}" --clip_range "${CLIP_RANGE}" --beta_kl "${BETA_KL}" \
    --save_freq "${SAVE_FREQ}" --eval_freq "${EVAL_FREQ}" --eval_episodes "${EVAL_EPISODES}" \
    --eval_seed 10000 --max_episode_steps "${MAX_EPISODE_STEPS}" \
    --progress_steps 64 --preflight_eval_episodes "${PREFLIGHT_EVAL_EPISODES}" \
    --max_no_signal_updates "${MAX_NO_SIGNAL_UPDATES}" --rollout_stall_timeout "${ROLLOUT_STALL_TIMEOUT}" \
    --rollout_heartbeat_seconds "${ROLLOUT_HEARTBEAT_SECONDS}" --cpu_threads "${CPU_THREADS}" \
    "${EXTRA_ARGS[@]}"
} >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"

echo "GRPO finished successfully. Output: ${OUTPUT_DIR}"
