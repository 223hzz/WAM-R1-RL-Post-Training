#!/bin/bash
# Standalone task-5 AWR launcher. This intentionally does not call the shared
# RFT/AWR launcher, so AWR sweeps do not alter any other training entry point.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/playground/dit4dit_libero_all_80000_2a800/pytorch_model.pt}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
ANCHOR_DATA="${ANCHOR_DATA:-/path/to/data/libero}"
RUN_NAME="${RUN_NAME:-spatial_task5_awr_sweep}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/Checkpoints_post_original/${RUN_NAME}}"
RESUME="${RESUME:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# AWR-native sweep knobs. ROLLOUT_EPISODES is a launcher alias for
# post_trainer's episodes_per_iteration because it is the repository's
# established name for collected episodes.
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-16}"
MIN_SUCCESSFUL_ROLLOUTS="${MIN_SUCCESSFUL_ROLLOUTS:-8}"
MAX_COLLECTION_EPISODES="${MAX_COLLECTION_EPISODES:-64}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${LR:-5e-5}"
ONLINE_FRACTION="${ONLINE_FRACTION:-0.75}"
K_STEPS="${K_STEPS:-16}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-4}"
NOISE_LEVEL="${NOISE_LEVEL:-0.1}"
ITERATIONS="${ITERATIONS:-20}"
POLICY_STEPS_PER_ITERATION="${POLICY_STEPS_PER_ITERATION:-1}"
VALUE_STEPS_PER_ITERATION="${VALUE_STEPS_PER_ITERATION:-25}"
EVAL_FREQ="${EVAL_FREQ:-2}"
BEST_EVAL_FREQ="${BEST_EVAL_FREQ:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"

BATCH_SIZE="${BATCH_SIZE:-8}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-512}"
COLLECTION_SAMPLER="${COLLECTION_SAMPLER:-sde}"
GAMMA="${GAMMA:-0.99}"
AWR_BETA="${AWR_BETA:-0.5}"
AWR_MAX_WEIGHT="${AWR_MAX_WEIGHT:-10.0}"
VALUE_LR="${VALUE_LR:-1e-4}"
SAVE_FREQ="${SAVE_FREQ:-1}"
# Keep the baseline preflight count aligned with EVAL_EPISODES by default;
# override independently when a cheaper smoke check is desired.
PREFLIGHT_EVAL_EPISODES="${PREFLIGHT_EVAL_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-10000}"
ROLLOUT_STALL_TIMEOUT="${ROLLOUT_STALL_TIMEOUT:-600}"

if [[ "${ROLLOUT_EPISODES}" -lt 1 || "${POLICY_STEPS_PER_ITERATION}" -lt 1 ||
      "${VALUE_STEPS_PER_ITERATION}" -lt 1 || "${ITERATIONS}" -lt 1 ]]; then
  echo "ROLLOUT_EPISODES, POLICY_STEPS_PER_ITERATION, VALUE_STEPS_PER_ITERATION, and ITERATIONS must be positive" >&2
  exit 1
fi
if [[ "${MIN_SUCCESSFUL_ROLLOUTS}" -lt 0 || "${MAX_COLLECTION_EPISODES}" -lt 1 ]]; then
  echo "MIN_SUCCESSFUL_ROLLOUTS must be non-negative and MAX_COLLECTION_EPISODES must be positive" >&2
  exit 1
fi

for required_path in "${SFT_CKPT}" "${CONFIG_YAML}" "${ANCHOR_DATA}"; do
  [[ -e "${required_path}" ]] || { echo "Required path not found: ${required_path}" >&2; exit 1; }
done
SFT_RUN_DIR="$(dirname "${SFT_CKPT}")"
[[ -f "${SFT_RUN_DIR}/config.yaml" ]] || SFT_RUN_DIR="$(cd "$(dirname "${SFT_CKPT}")/.." && pwd)"
for metadata in config.yaml dataset_statistics.json; do
  [[ -f "${SFT_RUN_DIR}/${metadata}" ]] || { echo "Missing SFT metadata: ${SFT_RUN_DIR}/${metadata}" >&2; exit 1; }
done

source /path/to/cluster/toolshs/set-cluster-env.sh 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || true
module load CUDA/12.2 2>/dev/null || true
source /path/to/cluster/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/LIBERO:${PYTHONPATH:-}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"

if ! awk -v fraction="${ONLINE_FRACTION}" 'BEGIN { exit !(fraction > 0 && fraction <= 1) }'; then
  echo "ONLINE_FRACTION must be in (0, 1]" >&2
  exit 1
fi
ANCHOR_FRACTION="$(awk -v fraction="${ONLINE_FRACTION}" 'BEGIN { printf "%.8f", 1.0-fraction }')"
RUN_CONFIG="${OUTPUT_DIR}/post_awr_run_config.txt"
{
  echo "METHOD=awr"
  echo "ROLLOUT_EPISODES=${ROLLOUT_EPISODES}"
  echo "MIN_SUCCESSFUL_ROLLOUTS=${MIN_SUCCESSFUL_ROLLOUTS}"
  echo "MAX_COLLECTION_EPISODES=${MAX_COLLECTION_EPISODES}"
  echo "TRAINABLE_SCOPE=${TRAINABLE_SCOPE}"
  echo "LR=${LR}"
  echo "POLICY_STEPS_PER_ITERATION=${POLICY_STEPS_PER_ITERATION}"
  echo "ONLINE_FRACTION=${ONLINE_FRACTION} (anchor fraction=${ANCHOR_FRACTION})"
  echo "K_STEPS=${K_STEPS}"
  echo "EXECUTION_HORIZON=${EXECUTION_HORIZON}"
  echo "NOISE_LEVEL=${NOISE_LEVEL}"
  echo "VALUE_STEPS_PER_ITERATION=${VALUE_STEPS_PER_ITERATION}"
  echo "ITERATIONS=${ITERATIONS}"
  echo "EVAL_FREQ=${EVAL_FREQ}"
  echo "BEST_EVAL_FREQ=${BEST_EVAL_FREQ}"
  echo "EVAL_EPISODES=${EVAL_EPISODES}"
  echo "PREFLIGHT_EVAL_EPISODES=${PREFLIGHT_EVAL_EPISODES}"
} >"${RUN_CONFIG}"

EXTRA_ARGS=()
[[ -n "${RESUME}" ]] && EXTRA_ARGS+=(--resume "${RESUME}")
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_awr_task5.out"
STDERR_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_awr_task5.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CONDA_ENV}/bin/python" -m dit4dit_rl_trainer.post_trainer \
  --method awr --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" \
  --anchor_data "${ANCHOR_DATA}" --output_dir "${OUTPUT_DIR}" --device cuda --dtype bfloat16 \
  --train_task_id 5 --libero_suite libero_spatial --libero_task_id 5 --libero_task_ids all \
  --action_unnorm_mode min_max --trainable_scope "${TRAINABLE_SCOPE}" \
  --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" --lr "${LR}" \
  --iterations "${ITERATIONS}" --episodes_per_iteration "${ROLLOUT_EPISODES}" \
  --min_successful_rollouts "${MIN_SUCCESSFUL_ROLLOUTS}" \
  --max_collection_episodes "${MAX_COLLECTION_EPISODES}" \
  --policy_steps_per_iteration "${POLICY_STEPS_PER_ITERATION}" --batch_size "${BATCH_SIZE}" \
  --online_fraction "${ONLINE_FRACTION}" --replay_capacity "${REPLAY_CAPACITY}" \
  --k_steps "${K_STEPS}" --collection_sampler "${COLLECTION_SAMPLER}" \
  --noise_level "${NOISE_LEVEL}" --execution_horizon "${EXECUTION_HORIZON}" \
  --max_episode_steps 220 --gamma "${GAMMA}" --awr_beta "${AWR_BETA}" \
  --awr_max_weight "${AWR_MAX_WEIGHT}" --value_lr "${VALUE_LR}" \
  --value_steps_per_iteration "${VALUE_STEPS_PER_ITERATION}" --save_freq "${SAVE_FREQ}" \
  --eval_freq "${EVAL_FREQ}" --best_eval_freq "${BEST_EVAL_FREQ}" \
  --preflight_eval_episodes "${PREFLIGHT_EVAL_EPISODES}" \
  --eval_episodes "${EVAL_EPISODES}" --eval_seed "${EVAL_SEED}" \
  --rollout_stall_timeout "${ROLLOUT_STALL_TIMEOUT}" "${EXTRA_ARGS[@]}" \
  >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"
