#!/bin/bash
# Flow-SDE SAPO post-training for LIBERO-Goal task 2.
# Online rollout collection uses weighted suite sampling: target task 2 has
# TARGET_TASK_WEIGHT probability and the remaining probability samples other
# tasks from LIBERO-Goal.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/playground/dit4dit_libero_all_80000_2a800/pytorch_model.pt}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_goal}"
TRAIN_TASK_ID="${TRAIN_TASK_ID:-2}"
LIBERO_TASK_ID="${LIBERO_TASK_ID:-${TRAIN_TASK_ID}}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-all}"
RUN_NAME="${RUN_NAME:-goal_task2_flow_sde_sapo}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/FlowSDESAPO/${RUN_NAME}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

SAMPLING_MODE="${SAMPLING_MODE:-weighted}"
TARGET_TASK_WEIGHT="${TARGET_TASK_WEIGHT:-0.5}"
K_STEPS="${K_STEPS:-16}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-4}"
NOISE_LEVEL="${NOISE_LEVEL:-0.1}"
NOISE_LEVEL_END="${NOISE_LEVEL_END:-0.05}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-32}"
MIN_ROLLOUT_CHUNKS="${MIN_ROLLOUT_CHUNKS:-256}"
UPDATES="${UPDATES:-20}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-8}"
LR="${LR:-5e-5}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-25}"
SFT_ANCHOR_DATA="${SFT_ANCHOR_DATA:-/path/to/data/libero}"
SFT_ANCHOR_WEIGHT="${SFT_ANCHOR_WEIGHT:-0.25}"
SFT_ANCHOR_BATCH_SIZE="${SFT_ANCHOR_BATCH_SIZE:-4}"
BETA_KL="${BETA_KL:-0.02}"
TAU_POS="${TAU_POS:-1.0}"
TAU_NEG="${TAU_NEG:-1.0}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_FREQ_UPDATES="${EVAL_FREQ_UPDATES:-2}"
SAVE_FREQ_UPDATES="${SAVE_FREQ_UPDATES:-1}"
EVAL_SEED="${EVAL_SEED:-10000}"
MIN_SUCCESSFUL_ROLLOUTS="${MIN_SUCCESSFUL_ROLLOUTS:-4}"
MAX_NO_SIGNAL_UPDATES="${MAX_NO_SIGNAL_UPDATES:-3}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

case "${LIBERO_SUITE}" in
  libero_goal) ;;
  *) echo "This launcher targets libero_goal; got LIBERO_SUITE=${LIBERO_SUITE}" >&2; exit 2 ;;
esac
if [[ "${TRAIN_TASK_ID}" != "2" || "${LIBERO_TASK_ID}" != "2" ]]; then
  echo "This launcher targets LIBERO-Goal task 2; set TRAIN_TASK_ID=2 and LIBERO_TASK_ID=2" >&2
  exit 2
fi

for required_path in "${SFT_CKPT}" "${CONFIG_YAML}"; do
  [[ -f "${required_path}" ]] || { echo "Required file not found: ${required_path}" >&2; exit 1; }
done
SFT_RUN_DIR="$(dirname "${SFT_CKPT}")"
if [[ ! -f "${SFT_RUN_DIR}/config.yaml" ]]; then
  SFT_RUN_DIR="$(cd "$(dirname "${SFT_CKPT}")/.." && pwd)"
fi
for metadata in config.yaml dataset_statistics.json; do
  [[ -f "${SFT_RUN_DIR}/${metadata}" ]] || {
    echo "Missing SFT metadata: ${SFT_RUN_DIR}/${metadata}" >&2
    exit 1
  }
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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_flow_sde_sapo_goal_task2.out"
STDERR_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_flow_sde_sapo_goal_task2.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"
{
  echo "POST_TRAINING_METHOD=SAPO"; echo "LOSS_TYPE=sapo"; echo "LIBERO_SUITE=${LIBERO_SUITE}"
  echo "TRAIN_TASK_ID=${TRAIN_TASK_ID}"; echo "LIBERO_TASK_ID=${LIBERO_TASK_ID}"; echo "LIBERO_TASK_IDS=${LIBERO_TASK_IDS}"
  echo "SAMPLING_MODE=${SAMPLING_MODE}"; echo "TARGET_TASK_WEIGHT=${TARGET_TASK_WEIGHT}"; echo "ROLLOUT_EPISODES=${ROLLOUT_EPISODES}"
  echo "TRAINABLE_SCOPE=${TRAINABLE_SCOPE}"; echo "LR=${LR}"; echo "PPO_EPOCHS=${PPO_EPOCHS}"; echo "SFT_ANCHOR_WEIGHT=${SFT_ANCHOR_WEIGHT}"
  echo "BETA_KL=${BETA_KL}"; echo "TAU_POS=${TAU_POS}"; echo "TAU_NEG=${TAU_NEG}"
  echo "K_STEPS=${K_STEPS}"; echo "EXECUTION_HORIZON=${EXECUTION_HORIZON}"; echo "NOISE_LEVEL=${NOISE_LEVEL}"; echo "NOISE_LEVEL_END=${NOISE_LEVEL_END}"
  echo "MIN_SUCCESSFUL_ROLLOUTS=${MIN_SUCCESSFUL_ROLLOUTS}"; echo "CRITIC_WARMUP_STEPS=${CRITIC_WARMUP_STEPS}"; echo "UPDATES=${UPDATES}"
  echo "EVAL_FREQ_UPDATES=${EVAL_FREQ_UPDATES}"; echo "EVAL_EPISODES=${EVAL_EPISODES}"; echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
} >"${OUTPUT_DIR}/flow_sde_sapo_goal_task2_run_config.txt"

EXTRA_ARGS=()
[[ "${PREFLIGHT_ONLY}" == "1" ]] && EXTRA_ARGS+=(--preflight_only)
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CONDA_ENV}/bin/python" -m dit4dit_rl_trainer.flow_sde_sapo_trainer \
  --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" --output_dir "${OUTPUT_DIR}" \
  --device cuda --dtype bfloat16 --train_task_id "${TRAIN_TASK_ID}" --libero_suite "${LIBERO_SUITE}" \
  --libero_task_id "${LIBERO_TASK_ID}" --libero_task_ids "${LIBERO_TASK_IDS}" --action_unnorm_mode min_max \
  --sampling_mode "${SAMPLING_MODE}" --target_task_weight "${TARGET_TASK_WEIGHT}" \
  --k_steps "${K_STEPS}" --execution_horizon "${EXECUTION_HORIZON}" --noise_level "${NOISE_LEVEL}" --noise_level_end "${NOISE_LEVEL_END}" \
  --rollout_episodes "${ROLLOUT_EPISODES}" --min_rollout_chunks "${MIN_ROLLOUT_CHUNKS}" --updates "${UPDATES}" --ppo_epochs "${PPO_EPOCHS}" \
  --minibatch_size "${MINIBATCH_SIZE}" --loss_type sapo --tau_pos "${TAU_POS}" --tau_neg "${TAU_NEG}" --beta_kl "${BETA_KL}" \
  --lr "${LR}" --trainable_scope "${TRAINABLE_SCOPE}" --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
  --critic_lr "${CRITIC_LR}" --max_episode_steps "${MAX_EPISODE_STEPS}" --critic_warmup_steps "${CRITIC_WARMUP_STEPS}" \
  --sft_anchor_data "${SFT_ANCHOR_DATA}" --sft_anchor_weight "${SFT_ANCHOR_WEIGHT}" --sft_anchor_batch_size "${SFT_ANCHOR_BATCH_SIZE}" \
  --min_successful_rollouts "${MIN_SUCCESSFUL_ROLLOUTS}" --max_no_signal_updates "${MAX_NO_SIGNAL_UPDATES}" \
  --eval_episodes "${EVAL_EPISODES}" --eval_freq_updates "${EVAL_FREQ_UPDATES}" --save_freq_updates "${SAVE_FREQ_UPDATES}" \
  --eval_seed "${EVAL_SEED}" "${EXTRA_ARGS[@]}" >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"
