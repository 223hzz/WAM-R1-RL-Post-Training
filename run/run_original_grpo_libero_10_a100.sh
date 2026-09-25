#!/usr/bin/env bash
# Direct single-A100 Action-GRPO launcher for a LIBERO task suite.
#
# Example:
#   CUDA_DEVICE=7 bash run_original_grpo_libero_10_a100.sh
#   RESUME_RL=/path/to/grpo_step_10_resume.pt bash run_original_grpo_libero_10_a100.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-/path/to/pretrained/Cosmos-Predict2.5-2B}"
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/playground/final model/pytorch_model.pt}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
NVIDIA_EGL_LIB_DIR="${NVIDIA_EGL_LIB_DIR:-/path/to/nvidia_egl_lib}"
ANCHOR_DATA="${ANCHOR_DATA:-/path/to/data/LIBERO}"
RSFT_ANCHOR_DATA="${RSFT_ANCHOR_DATA:-${ANCHOR_DATA}}"

LIBERO_SUITE="${LIBERO_SUITE:-libero_goal}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-all}"
LIBERO_TASK_SCHEDULE="${LIBERO_TASK_SCHEDULE:-round_robin}"
LIBERO_TASK_BLOCK_SIZE="${LIBERO_TASK_BLOCK_SIZE:-1}"
RUN_NAME="${RUN_NAME:-${LIBERO_SUITE}_grpo_a100_round_robin}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/Checkpoints_rl_original/${RUN_NAME}}"
RESUME_RL="${RESUME_RL:-}"

if [[ -n "${CUDA_DEVICE:-}" ]]; then
  CUDA_DEVICE="${CUDA_DEVICE}"
else
  CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
  CUDA_DEVICE="${CUDA_DEVICE%%,*}"
fi
if ! [[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_DEVICE must be one physical GPU index; got '${CUDA_DEVICE}'" >&2
  exit 2
fi
if [[ "${LIBERO_SUITE}" != "libero_goal" && "${LIBERO_SUITE}" != "libero_10" ]]; then
  echo "This launcher supports LIBERO_SUITE=libero_goal or libero_10; got ${LIBERO_SUITE}." >&2
  exit 2
fi
if [[ "${LIBERO_TASK_IDS}" != "all" && "${LIBERO_TASK_IDS}" != "0-9" ]]; then
  echo "This launcher requires LIBERO_TASK_IDS=all or 0-9." >&2
  exit 2
fi
if [[ "${LIBERO_TASK_SCHEDULE}" != "round_robin" || "${LIBERO_TASK_BLOCK_SIZE}" != "1" ]]; then
  echo "LIBERO-10 GRPO requires round_robin scheduling with block size 1." >&2
  exit 2
fi

GRPO_ITERS=10
GROUP_SIZE=8
K_STEPS=4
if [[ "${LIBERO_SUITE}" == "libero_goal" ]]; then
  MAX_EPISODE_STEPS=300
else
  MAX_EPISODE_STEPS=520
fi
PREFLIGHT_EVAL_EPISODES=30
SAVE_FREQ=1
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
RL_CHUNKS_PER_EPISODE=4
# GRPO uses every replanning chunk from the complete episode.  The legacy
# RL_CHUNKS_PER_EPISODE value remains only for compatibility when this switch
# is disabled; this launcher always enables full-trajectory training.
COLLECT_ALL_CHUNKS=1
RL_CHUNK_STRIDE=4
POLICY_STEPS_PER_ITERATION=1
GRPO_INNER_EPOCHS=1
NOISE_LEVEL=0.1
NOISE_LEVEL_END=0.05
RL_LR="${RL_LR:-5e-6}"
CLIP_RANGE="${CLIP_RANGE:-2e-6}"
BETA_KL="${BETA_KL:-0.08}"
EVAL_FREQ=1
EVAL_EPISODES=30
EVAL_SEED="${EVAL_SEED:-10000}"
PROGRESS_STEPS="${PROGRESS_STEPS:-64}"
MAX_NO_SIGNAL_UPDATES="${MAX_NO_SIGNAL_UPDATES:-5}"
ROLLOUT_STALL_TIMEOUT="${ROLLOUT_STALL_TIMEOUT:-600}"
ROLLOUT_HEARTBEAT_SECONDS="${ROLLOUT_HEARTBEAT_SECONDS:-30}"
CPU_THREADS="${CPU_THREADS:-4}"
ACTION_UNNORM_MODE="${ACTION_UNNORM_MODE:-min_max}"
SFT_ANCHOR_WEIGHT=0.25
SFT_ANCHOR_BATCH_SIZE=4
SFT_ONLINE_FRACTION=0.3
GLOBAL_STD="${GLOBAL_STD:-0}"
REQUIRE_PREFLIGHT_SUCCESS="${REQUIRE_PREFLIGHT_SUCCESS:-1}"
MAX_AUTO_RESUME_RETRIES="${MAX_AUTO_RESUME_RETRIES:-2}"

require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }
require_dir "${PROJECT_DIR}"
require_file "${PYTHON_BIN}"
require_file "${SFT_CKPT}"
require_file "${CONFIG_YAML}"
require_dir "${COSMOS_BASE_MODEL}"
if [[ "${SFT_ANCHOR_WEIGHT}" != "0" ]]; then
  require_dir "${RSFT_ANCHOR_DATA}"
fi

# baseframework accepts metadata next to the checkpoint or one directory up.
SFT_RUN_DIR="$(dirname "${SFT_CKPT}")"
if [[ ! -f "${SFT_RUN_DIR}/config.yaml" || ! -f "${SFT_RUN_DIR}/dataset_statistics.json" ]]; then
  SFT_RUN_DIR="$(dirname "${SFT_RUN_DIR}")"
fi
require_file "${SFT_RUN_DIR}/config.yaml"
require_file "${SFT_RUN_DIR}/dataset_statistics.json"

if [[ -f /path/to/cluster/toolshs/set-cluster-env.sh ]]; then
  # shellcheck disable=SC1091
  source /path/to/cluster/toolshs/set-cluster-env.sh 2>/dev/null || true
fi
if type module >/dev/null 2>&1; then
  module load CUDA/12.2 2>/dev/null || true
fi

cd "${PROJECT_DIR}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/LIBERO:${PYTHONPATH:-}"
export DIT4DIT_COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __GLX_VENDOR_LIBRARY_NAME=nvidia
export MUJOCO_RENDER_GPU_DEVICE_ID="${MUJOCO_RENDER_GPU_DEVICE_ID:-${CUDA_DEVICE}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_DEVICE}}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:${LD_LIBRARY_PATH:-}"
if [[ -d "${NVIDIA_EGL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_EGL_LIB_DIR}:${LD_LIBRARY_PATH}"
fi
if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
  for manifest in /usr/share/glvnd/egl_vendor.d/10_nvidia.json /usr/share/egl/egl_vendor.d/10_nvidia.json; do
    if [[ -f "${manifest}" ]]; then
      export __EGL_VENDOR_LIBRARY_FILENAMES="${manifest}"
      break
    fi
  done
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers" TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export WANDB_MODE="${WANDB_MODE:-disabled}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${OUTPUT_DIR}/${TIMESTAMP}_grpo_${LIBERO_SUITE}.out"
STDERR_LOG="${OUTPUT_DIR}/${TIMESTAMP}_grpo_${LIBERO_SUITE}.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"

build_extra_args() {
  EXTRA_ARGS=()
  [[ -n "${RESUME_RL}" ]] && EXTRA_ARGS+=(--resume_rl "${RESUME_RL}")
  [[ -n "${RSFT_ANCHOR_DATA}" && "${SFT_ANCHOR_WEIGHT}" != "0" ]] && EXTRA_ARGS+=(--rsft_anchor_data "${RSFT_ANCHOR_DATA}")
  EXTRA_ARGS+=(--rsft_anchor_weight "${SFT_ANCHOR_WEIGHT}" --rsft_anchor_batch_size "${SFT_ANCHOR_BATCH_SIZE}" --rsft_online_fraction "${SFT_ONLINE_FRACTION}")
  [[ "${COLLECT_ALL_CHUNKS}" == "1" ]] && EXTRA_ARGS+=(--collect_all_chunks) || EXTRA_ARGS+=(--no-collect_all_chunks)
  [[ "${GLOBAL_STD}" == "1" ]] && EXTRA_ARGS+=(--global_std) || EXTRA_ARGS+=(--no-global_std)
  [[ "${REQUIRE_PREFLIGHT_SUCCESS}" == "1" ]] && EXTRA_ARGS+=(--require_preflight_success) || EXTRA_ARGS+=(--no-require_preflight_success)
}

find_latest_resume() {
  find "${OUTPUT_DIR}/checkpoints" -maxdepth 1 -type f -name 'grpo_step_*_resume.pt' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

{
  echo "===== Original DiT4DiT LIBERO-10 Action-GRPO on A100 ====="
  echo "mode=single gpu=${CUDA_DEVICE} tasks=${LIBERO_TASK_IDS} output=${OUTPUT_DIR}"
  echo "iters=${GRPO_ITERS} group=${GROUP_SIZE} chunks_per_episode=all (legacy_limit=${RL_CHUNKS_PER_EPISODE})"
  echo "k=${K_STEPS} horizon=${MAX_EPISODE_STEPS} scope=${TRAINABLE_SCOPE} policy_steps=${POLICY_STEPS_PER_ITERATION} inner_epochs=${GRPO_INNER_EPOCHS}"
  echo "preflight_episodes=${PREFLIGHT_EVAL_EPISODES} eval_episodes=${EVAL_EPISODES} anchor_weight=${SFT_ANCHOR_WEIGHT} noise=${NOISE_LEVEL}->${NOISE_LEVEL_END}"
  "${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

  RESTART_ATTEMPT=0
  while true; do
    build_extra_args
    set +e
    "${PYTHON_BIN}" -m dit4dit_rl_trainer.trainer --real \
      --model_variant original --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" \
      --env libero --libero_suite "${LIBERO_SUITE}" --libero_task_id 0 \
      --libero_task_ids "${LIBERO_TASK_IDS}" --libero_task_schedule "${LIBERO_TASK_SCHEDULE}" \
      --libero_task_block_size "${LIBERO_TASK_BLOCK_SIZE}" --action_unnorm_mode "${ACTION_UNNORM_MODE}" \
      --device cuda --dtype bfloat16 --output_dir "${OUTPUT_DIR}" \
      --grpo_iters "${GRPO_ITERS}" --group_size "${GROUP_SIZE}" --k_steps "${K_STEPS}" \
      --trainable_scope "${TRAINABLE_SCOPE}" --lora_rank "${LORA_RANK}" \
      --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
      --rl_chunks_per_episode "${RL_CHUNKS_PER_EPISODE}" --rl_chunk_stride "${RL_CHUNK_STRIDE}" \
      --grpo_inner_epochs "${GRPO_INNER_EPOCHS}" --noise_level "${NOISE_LEVEL}" \
      --noise_level_end "${NOISE_LEVEL_END}" \
      --lr "${RL_LR}" --clip_range "${CLIP_RANGE}" --beta_kl "${BETA_KL}" \
      --save_freq "${SAVE_FREQ}" --eval_freq "${EVAL_FREQ}" --eval_episodes "${EVAL_EPISODES}" \
      --eval_seed "${EVAL_SEED}" --max_episode_steps "${MAX_EPISODE_STEPS}" \
      --progress_steps "${PROGRESS_STEPS}" --preflight_eval_episodes "${PREFLIGHT_EVAL_EPISODES}" \
      --max_no_signal_updates "${MAX_NO_SIGNAL_UPDATES}" --rollout_stall_timeout "${ROLLOUT_STALL_TIMEOUT}" \
      --rollout_heartbeat_seconds "${ROLLOUT_HEARTBEAT_SECONDS}" --cpu_threads "${CPU_THREADS}" \
      "${EXTRA_ARGS[@]}"
    EXIT_CODE=$?
    set -e
    [[ "${EXIT_CODE}" -eq 0 ]] && break
    if [[ "${EXIT_CODE}" -ne 124 || "${RESTART_ATTEMPT}" -ge "${MAX_AUTO_RESUME_RETRIES}" ]]; then
      exit "${EXIT_CODE}"
    fi
    RESUME_RL="$(find_latest_resume)"
    [[ -f "${RESUME_RL}" ]] || { echo "Watchdog fired but no resume checkpoint exists" >&2; exit 124; }
    RESTART_ATTEMPT=$((RESTART_ATTEMPT + 1))
    echo "Restarting from ${RESUME_RL} (${RESTART_ATTEMPT}/${MAX_AUTO_RESUME_RETRIES})"
  done
} >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"

echo "GRPO finished successfully. Output: ${OUTPUT_DIR}"
