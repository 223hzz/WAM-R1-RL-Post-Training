#!/usr/bin/env bash
# Direct A100 launcher for original DiT4DiT object-suite AWR.
#
# This script intentionally does not use yhbatch, sbatch, or cluster modules.
# Run it from any directory with:
#   bash run_original_awr_object_a100.sh
#
# Every setting below can be overridden with an environment variable, e.g.
#   CUDA_DEVICE=1 ONLINE_CHUNK_BATCH_SIZE=4 bash run_original_awr_object_a100.sh
# CUDA_DEVICE is the physical index from nvidia-smi; no MUJOCO_*_DEVICE_ID
# variables need to be supplied by the caller.
set -Eeuo pipefail
echo "run_original_awr_object_a100.sh: NVIDIA-EGL physical-GPU mode (2026-08-29)" >&2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-/path/to/pretrained/Cosmos-Predict2.5-2B}"
# User-local NVIDIA EGL libraries used on the amax node.  Override this path
# when the cluster provides the driver libraries elsewhere.
NVIDIA_EGL_LIB_DIR="${NVIDIA_EGL_LIB_DIR:-/path/to/nvidia_egl_lib}"

# The checkpoint directory must also contain config.yaml and
# dataset_statistics.json.  This is required by baseframework.from_pretrained.
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/playground/final model/pytorch_model.pt}"
# This is the runtime/trainer config.  The checkpoint's own config.yaml is
# still loaded by baseframework for model construction and receives the Cosmos
# runtime override below.  The repository LIBERO config contains image_size
# and the other environment/dataloader fields required by post_trainer.
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
SFT_RUN_DIR="${SFT_RUN_DIR:-$(dirname "${SFT_CKPT}")}"

# post_trainer expects the parent of the named LeRobot dataset.
ANCHOR_DATA="${ANCHOR_DATA:-/path/to/data/libero}"
ANCHOR_DATA_MIX="${ANCHOR_DATA_MIX:-libero_object_a100}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_object}"
OBJECT_DATASET_DIR="${OBJECT_DATASET_DIR:-${ANCHOR_DATA}/libero_object_no_noops_lerobot}"

RUN_NAME="${RUN_NAME:-object_awr_full_u10_traj50_retry}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/Checkpoints_post_original/${RUN_NAME}}"
RESUME="${RESUME:-}"
# CUDA_DEVICE is always a physical GPU index (the value shown by nvidia-smi).
# MuJoCo's EGL device enumeration is independent of CUDA_VISIBLE_DEVICES, so
# do not remap this value to logical device 0.
if [[ -n "${CUDA_DEVICE:-}" ]]; then
  CUDA_DEVICE="${CUDA_DEVICE}"
else
  CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-}"
  CUDA_DEVICE="${CUDA_DEVICE%%,*}"
  CUDA_DEVICE="${CUDA_DEVICE:-0}"
fi
if ! [[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_DEVICE must be a physical GPU index (for example 0, 1, 2, or 7); got '${CUDA_DEVICE}'" >&2
  exit 2
fi
# Keep the process single-GPU. CUDA maps this card to torch/cuda:0, while
# MuJoCo below receives the physical EGL index explicitly.
CUDA_VISIBLE_DEVICES_LIST="${CUDA_DEVICE}"

# AWR settings requested for the object-suite run.
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-full}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
ITERATIONS="${ITERATIONS:-10}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-50}"
MIN_SUCCESSFUL_ROLLOUTS="${MIN_SUCCESSFUL_ROLLOUTS:-20}"
MAX_COLLECTION_EPISODES="${MAX_COLLECTION_EPISODES:-100}"
ONLINE_TRAJECTORIES_PER_UPDATE="${ONLINE_TRAJECTORIES_PER_UPDATE:-25}"
ONLINE_CHUNK_BATCH_SIZE="${ONLINE_CHUNK_BATCH_SIZE:-16}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
POLICY_STEPS_PER_ITERATION="${POLICY_STEPS_PER_ITERATION:-1}"
ONLINE_FRACTION="${ONLINE_FRACTION:-0.5}"
PREFLIGHT_EVAL_EPISODES="${PREFLIGHT_EVAL_EPISODES:-20}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_FREQ="${EVAL_FREQ:-2}"
BEST_EVAL_FREQ="${BEST_EVAL_FREQ:-1}"
LR="${LR:-5e-6}"
K_STEPS="${K_STEPS:-16}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-4}"
NOISE_LEVEL="${NOISE_LEVEL:-0.1}"
VALUE_STEPS_PER_ITERATION="${VALUE_STEPS_PER_ITERATION:-25}"

# Preserve the sampler used by the previous object AWR launcher.  Set
# COLLECTION_SAMPLER=ode to match the deterministic evaluation policy.
COLLECTION_SAMPLER="${COLLECTION_SAMPLER:-sde}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
VALUE_LR="${VALUE_LR:-1e-4}"
GAMMA="${GAMMA:-0.99}"
AWR_BETA="${AWR_BETA:-0.5}"
AWR_MAX_WEIGHT="${AWR_MAX_WEIGHT:-10.0}"
SAVE_FREQ="${SAVE_FREQ:-1}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-8192}"
EVAL_SEED="${EVAL_SEED:-10000}"
ROLLOUT_STALL_TIMEOUT="${ROLLOUT_STALL_TIMEOUT:-600}"
ROLLOUT_HEARTBEAT_SECONDS="${ROLLOUT_HEARTBEAT_SECONDS:-30}"
CPU_THREADS="${CPU_THREADS:-4}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }
}

require_dir "${PROJECT_DIR}"
require_file "${PYTHON_BIN}"
require_file "${SFT_CKPT}"
require_file "${CONFIG_YAML}"
require_dir "${COSMOS_BASE_MODEL}"
# Accept either checkpoint-local metadata or the legacy parent layout.
if [[ ! -f "${SFT_RUN_DIR}/config.yaml" || ! -f "${SFT_RUN_DIR}/dataset_statistics.json" ]]; then
  SFT_RUN_DIR="$(dirname "${SFT_RUN_DIR}")"
fi
require_file "${SFT_RUN_DIR}/config.yaml"
require_file "${SFT_RUN_DIR}/dataset_statistics.json"
require_dir "${OBJECT_DATASET_DIR}"

if [[ "${ANCHOR_DATA_MIX}" != "libero_object_a100" && "${ANCHOR_DATA_MIX}" != "libero_10_a100" && "${ANCHOR_DATA_MIX}" != "libero_object" && "${ANCHOR_DATA_MIX}" != "libero_10" ]]; then
  echo "Unsupported ANCHOR_DATA_MIX=${ANCHOR_DATA_MIX}; expected an object or libero_10 mixture." >&2
  exit 2
fi

if ! [[ "${ONLINE_FRACTION}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
   ! awk -v value="${ONLINE_FRACTION}" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
  echo "ONLINE_FRACTION must be in (0, 1]" >&2
  exit 2
fi

for positive_value in \
  ITERATIONS ROLLOUT_EPISODES MAX_COLLECTION_EPISODES ONLINE_TRAJECTORIES_PER_UPDATE \
  ONLINE_CHUNK_BATCH_SIZE REPLAY_CAPACITY BATCH_SIZE POLICY_STEPS_PER_ITERATION \
  PREFLIGHT_EVAL_EPISODES EVAL_EPISODES EVAL_FREQ BEST_EVAL_FREQ K_STEPS \
  EXECUTION_HORIZON VALUE_STEPS_PER_ITERATION; do
  if ! [[ "${!positive_value}" =~ ^[0-9]+$ ]] || [[ "${!positive_value}" -lt 1 ]]; then
    echo "${positive_value} must be a positive integer" >&2
    exit 2
  fi
done

# Directly use the environment's interpreter.  Activation is optional and is
# only attempted when this conda installation provides the shell hook.
if [[ -f "${CONDA_ENV}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ENV}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true
fi

# Load the cluster CUDA module when this is a module-based node.  Without the
# driver module, libEGL_nvidia is absent from the dynamic linker search path
# even though the host has an A100 allocation.
if [[ -f /path/to/cluster/toolshs/set-cluster-env.sh ]]; then
  # shellcheck disable=SC1091
  source /path/to/cluster/toolshs/set-cluster-env.sh 2>/dev/null || true
fi
if type module >/dev/null 2>&1; then
  module load anaconda3/2023.09 2>/dev/null || true
  module load CUDA/12.2 2>/dev/null || true
fi

cd "${PROJECT_DIR}"
export PATH="${CONDA_ENV}/bin:${PATH}"
# Prefer the system GLVND/NVIDIA libraries over any conda Mesa OpenGL build.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/LIBERO:${PYTHONPATH:-}"
export DIT4DIT_COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL}"
# Keep all caches/logs in this project and prevent optional telemetry or
# dependency version checks from attempting network access on the cluster.
export DIT4DIT_ROBOSUITE_LOG="${PROJECT_DIR}/logs/robosuite.log"
# This A100 launcher uses NVIDIA EGL rendering exclusively.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia
if [[ -d "${NVIDIA_EGL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_EGL_LIB_DIR}:/usr/lib/x86_64-linux-gnu:/usr/lib64:${LD_LIBRARY_PATH:-}"
else
  echo "NVIDIA EGL library directory not found: ${NVIDIA_EGL_LIB_DIR}" >&2
  echo "Set NVIDIA_EGL_LIB_DIR to the directory containing libEGL_nvidia.so.*" >&2
fi
if [[ "${MUJOCO_GL}" == "egl" ]]; then
  # These are physical EGL indices (the same IDs shown by nvidia-smi). This
  # keeps a request such as CUDA_DEVICE=2 on physical GPU 2; it is not mapped
  # to GPU 0 or silently changed to logical index 0.
  # CUDA_VISIBLE_DEVICES masks the selected card to torch/cuda:0.  Keep the
  # physical-GPU default for existing nodes, but allow EGL-specific overrides
  # because some robosuite builds enumerate only logical device 0 (or none).
  export MUJOCO_RENDER_GPU_DEVICE_ID="${MUJOCO_RENDER_GPU_DEVICE_ID:-${CUDA_DEVICE}}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_DEVICE}}"
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    for egl_manifest in \
      /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
      /usr/share/egl/egl_vendor.d/10_nvidia.json \
      /usr/lib/x86_64-linux-gnu/egl_vendor.d/10_nvidia.json \
      /usr/lib64/egl_vendor.d/10_nvidia.json \
      "${HOME:-}/.local/share/glvnd/egl_vendor.d/10-nvidia-custom.json" \
      "${HOME:-}/.local/share/glvnd/egl_vendor.d/10_nvidia.json"; do
      if [[ -f "${egl_manifest}" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_manifest}"
        break
      fi
    done
  fi
  if [[ -n "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && ! -f "${__EGL_VENDOR_LIBRARY_FILENAMES}" ]]; then
    echo "Configured NVIDIA EGL manifest does not exist: ${__EGL_VENDOR_LIBRARY_FILENAMES}" >&2
    unset __EGL_VENDOR_LIBRARY_FILENAMES
  fi
  # Cluster images sometimes install the manifest in a versioned directory.
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    egl_manifest="$(find /usr /etc "${HOME:-/nonexistent}/.local/share/glvnd" \
      -type f -path '*/egl_vendor.d/*' -iname '*nvidia*.json' 2>/dev/null | head -n 1 || true)"
    if [[ -n "${egl_manifest}" ]]; then
      export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_manifest}"
    fi
  fi
  # Some container images ship libEGL_nvidia but omit the system manifest.
  # Build a private manifest in the project cache so GLVND can load NVIDIA
  # directly without scanning Mesa or touching /dev/dri.
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    nvidia_egl_lib="$(
      ldconfig -p 2>/dev/null | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}' || true
    )"
    if [[ -z "${nvidia_egl_lib}" ]]; then
      nvidia_egl_lib="$(find /usr /opt /usr/local -type f \( -name 'libEGL_nvidia.so.0' -o -name 'libEGL_nvidia.so.*' \) 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -z "${nvidia_egl_lib}" ]]; then
      nvidia_egl_lib="$(find /usr /opt /usr/local -type l \( -name 'libEGL_nvidia.so.0' -o -name 'libEGL_nvidia.so.*' \) 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -n "${nvidia_egl_lib}" ]]; then
      export LD_LIBRARY_PATH="$(dirname "${nvidia_egl_lib}"):${LD_LIBRARY_PATH:-}"
      nvidia_egl_basename="$(basename "${nvidia_egl_lib}")"
      private_egl_dir="${PROJECT_DIR}/.cache/egl"
      private_egl_manifest="${private_egl_dir}/10_nvidia.json"
      mkdir -p "${private_egl_dir}"
      printf '%s\n' \
        "{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"${nvidia_egl_basename}\"}}" \
        >"${private_egl_manifest}"
      export __EGL_VENDOR_LIBRARY_FILENAMES="${private_egl_manifest}"
      echo "Generated private NVIDIA EGL manifest: ${private_egl_manifest}"
    fi
  fi
  if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    echo "NVIDIA EGL manifest and libEGL_nvidia driver library were not found; refusing to start because GLVND would fall back to Mesa and probe /dev/dri." >&2
    echo "Install/load the NVIDIA driver (libEGL_nvidia.so.*), or set __EGL_VENDOR_LIBRARY_FILENAMES to its manifest." >&2
    echo "--- NVIDIA EGL diagnostics ---" >&2
    echo "hostname=$(hostname 2>/dev/null || true)" >&2
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}" >&2
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L >&2 || true
    else
      echo "nvidia-smi: not found" >&2
    fi
    if command -v ldconfig >/dev/null 2>&1; then
      ldconfig -p 2>/dev/null | grep -i 'EGL_nvidia' >&2 || echo 'ldconfig: no libEGL_nvidia entry' >&2
    fi
    if type module >/dev/null 2>&1; then
      module list 2>&1 | tail -n 20 >&2 || true
    fi
    echo "--- end diagnostics ---" >&2
    exit 3
  fi
  if ! grep -qi nvidia "${__EGL_VENDOR_LIBRARY_FILENAMES}"; then
    echo "EGL vendor manifest is not NVIDIA: ${__EGL_VENDOR_LIBRARY_FILENAMES}" >&2
    echo "Refusing to start so Mesa cannot probe /dev/dri." >&2
    exit 3
  fi
  echo "Using NVIDIA EGL vendor: ${__EGL_VENDOR_LIBRARY_FILENAMES} (physical GPU ${CUDA_DEVICE})"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${OUTPUT_DIR}/${TIMESTAMP}_awr_object.out"
STDERR_LOG="${OUTPUT_DIR}/${TIMESTAMP}_awr_object.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"

{
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "CONDA_ENV=${CONDA_ENV}"
  echo "COSMOS_BASE_MODEL=${COSMOS_BASE_MODEL}"
  echo "NVIDIA_EGL_LIB_DIR=${NVIDIA_EGL_LIB_DIR}"
  echo "SFT_CKPT=${SFT_CKPT}"
  echo "CONFIG_YAML=${CONFIG_YAML}"
  echo "ANCHOR_DATA=${ANCHOR_DATA}"
  echo "ANCHOR_DATA_MIX=${ANCHOR_DATA_MIX}"
  echo "OBJECT_DATASET_DIR=${OBJECT_DATASET_DIR}"
  echo "LIBERO_SUITE=${LIBERO_SUITE}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "CUDA_DEVICE=${CUDA_DEVICE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST}"
  echo "MUJOCO_GL=${MUJOCO_GL}"
  echo "MUJOCO_RENDER_GPU_DEVICE_ID=${MUJOCO_RENDER_GPU_DEVICE_ID:-}"
  echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-}"
  echo "__EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "TRAINABLE_SCOPE=${TRAINABLE_SCOPE}"
  echo "ITERATIONS=${ITERATIONS}"
  echo "ROLLOUT_EPISODES=${ROLLOUT_EPISODES}"
  echo "MIN_SUCCESSFUL_ROLLOUTS=${MIN_SUCCESSFUL_ROLLOUTS}"
  echo "MAX_COLLECTION_EPISODES=${MAX_COLLECTION_EPISODES}"
  echo "ONLINE_TRAJECTORIES_PER_UPDATE=${ONLINE_TRAJECTORIES_PER_UPDATE}"
  echo "ONLINE_CHUNK_BATCH_SIZE=${ONLINE_CHUNK_BATCH_SIZE}"
  echo "REPLAY_CAPACITY=${REPLAY_CAPACITY}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "POLICY_STEPS_PER_ITERATION=${POLICY_STEPS_PER_ITERATION}"
  echo "ONLINE_FRACTION=${ONLINE_FRACTION}"
  echo "LR=${LR}"
  echo "K_STEPS=${K_STEPS}"
  echo "EXECUTION_HORIZON=${EXECUTION_HORIZON}"
  echo "NOISE_LEVEL=${NOISE_LEVEL}"
  echo "VALUE_STEPS_PER_ITERATION=${VALUE_STEPS_PER_ITERATION}"
  echo "COLLECTION_SAMPLER=${COLLECTION_SAMPLER}"
} >"${OUTPUT_DIR}/post_awr_run_config.txt"

EXTRA_ARGS=()
[[ -n "${RESUME}" ]] && EXTRA_ARGS+=(--resume "${RESUME}")

echo "Starting AWR on CUDA device ${CUDA_DEVICE}. Logs: ${STDOUT_LOG} and ${STDERR_LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}" "${PYTHON_BIN}" -m dit4dit_rl_trainer.post_trainer \
  --method awr \
  --config_yaml "${CONFIG_YAML}" \
  --checkpoint "${SFT_CKPT}" \
  --anchor_data "${ANCHOR_DATA}" \
  --anchor_data_mix "${ANCHOR_DATA_MIX}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda --dtype bfloat16 \
  --train_task_id 0 --train_task_ids all \
  --libero_suite "${LIBERO_SUITE}" --libero_task_id 0 --libero_task_ids all \
  --action_unnorm_mode min_max \
  --trainable_scope "${TRAINABLE_SCOPE}" \
  --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
  --lr "${LR}" --iterations "${ITERATIONS}" \
  --episodes_per_iteration "${ROLLOUT_EPISODES}" \
  --min_successful_rollouts "${MIN_SUCCESSFUL_ROLLOUTS}" \
  --max_collection_episodes "${MAX_COLLECTION_EPISODES}" \
  --online_trajectories_per_update "${ONLINE_TRAJECTORIES_PER_UPDATE}" \
  --online_chunk_batch_size "${ONLINE_CHUNK_BATCH_SIZE}" \
  --policy_steps_per_iteration "${POLICY_STEPS_PER_ITERATION}" \
  --batch_size "${BATCH_SIZE}" --online_fraction "${ONLINE_FRACTION}" \
  --replay_capacity "${REPLAY_CAPACITY}" --k_steps "${K_STEPS}" \
  --collection_sampler "${COLLECTION_SAMPLER}" --noise_level "${NOISE_LEVEL}" \
  --execution_horizon "${EXECUTION_HORIZON}" --max_episode_steps 280 \
  --gamma "${GAMMA}" --awr_beta "${AWR_BETA}" --awr_max_weight "${AWR_MAX_WEIGHT}" \
  --value_lr "${VALUE_LR}" --value_steps_per_iteration "${VALUE_STEPS_PER_ITERATION}" \
  --save_freq "${SAVE_FREQ}" --eval_freq "${EVAL_FREQ}" --best_eval_freq "${BEST_EVAL_FREQ}" \
  --preflight_eval_episodes "${PREFLIGHT_EVAL_EPISODES}" \
  --eval_episodes "${EVAL_EPISODES}" --eval_seed "${EVAL_SEED}" \
  --rollout_stall_timeout "${ROLLOUT_STALL_TIMEOUT}" \
  --rollout_heartbeat_seconds "${ROLLOUT_HEARTBEAT_SECONDS}" \
  --cpu_threads "${CPU_THREADS}" "${EXTRA_ARGS[@]}" \
  >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"

echo "AWR finished successfully. Output: ${OUTPUT_DIR}"
