#!/bin/bash
# Independent suite-level few-shot Flow-SDE PPO launcher. It does not invoke
# run_original_flow_sde_ppo_task5.sh or alter its task-5 checkpoints.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
SFT_RUN_ID="${SFT_RUN_ID:-dit4dit_libero_spatial_transfer_gr1_k10_b4_s25000_seed20260820}"
SFT_RUN_DIR="${SFT_RUN_DIR:-${PROJECT_DIR}/playground/Checkpoints_fewshot/${SFT_RUN_ID}}"
SFT_CKPT="${SFT_CKPT:-${SFT_RUN_DIR}/checkpoints/best_pytorch_model.pt}"
SFT_ANCHOR_DATA="${SFT_ANCHOR_DATA:-/path/to/data/libero}"
EPISODE_SUBSET_MANIFEST="${EPISODE_SUBSET_MANIFEST:-${SFT_RUN_DIR}/fewshot_manifest.json}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_spatial}"
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-all}"
TRAIN_TASK_SCHEDULE="${TRAIN_TASK_SCHEDULE:-round_robin}"
RUN_NAME="${RUN_NAME:-${LIBERO_SUITE}_transfer_gr1_k10_flow_sde_ppo}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/FewShotSuiteFlowSDEPPO/${RUN_NAME}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

K_STEPS="${K_STEPS:-3}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-5}"
NOISE_LEVEL="${NOISE_LEVEL:-0.30}"
NOISE_LEVEL_END="${NOISE_LEVEL_END:-0.03}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-8}"
MIN_ROLLOUT_CHUNKS="${MIN_ROLLOUT_CHUNKS:-128}"
MAX_COLLECTION_EPISODES="${MAX_COLLECTION_EPISODES:-16}"
UPDATES="${UPDATES:-500}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-8}"
LR="${LR:-5e-6}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-action_head}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-0}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-25}"
SFT_ANCHOR_WEIGHT="${SFT_ANCHOR_WEIGHT:-0.25}"
SFT_ANCHOR_BATCH_SIZE="${SFT_ANCHOR_BATCH_SIZE:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_FREQ_UPDATES="${EVAL_FREQ_UPDATES:-25}"
SAVE_FREQ_UPDATES="${SAVE_FREQ_UPDATES:-25}"
EVAL_SEED="${EVAL_SEED:-10000}"
MIN_SUCCESSFUL_ROLLOUTS="${MIN_SUCCESSFUL_ROLLOUTS:-1}"
MAX_NO_SIGNAL_UPDATES="${MAX_NO_SIGNAL_UPDATES:-10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-240}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_RANGE="${CLIP_RANGE:-0.2}"
BETA_KL="${BETA_KL:-0.0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

for required_path in "${SFT_CKPT}" "${CONFIG_YAML}" "${EPISODE_SUBSET_MANIFEST}"; do
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
STDOUT_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_fewshot_suite_flow_sde_ppo.out"
STDERR_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_fewshot_suite_flow_sde_ppo.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"
{
  echo "SFT_RUN_ID=${SFT_RUN_ID}"
  echo "SFT_RUN_DIR=${SFT_RUN_DIR}"
  echo "SFT_CKPT=${SFT_CKPT}"
  echo "SFT_ANCHOR_DATA=${SFT_ANCHOR_DATA}"
  echo "EPISODE_SUBSET_MANIFEST=${EPISODE_SUBSET_MANIFEST}"
  echo "LIBERO_SUITE=${LIBERO_SUITE}"
  echo "TRAIN_TASK_IDS=${TRAIN_TASK_IDS}"
  echo "TRAIN_TASK_SCHEDULE=${TRAIN_TASK_SCHEDULE}"
  echo "K_STEPS=${K_STEPS}"
  echo "EXECUTION_HORIZON=${EXECUTION_HORIZON}"
  echo "NOISE_LEVEL=${NOISE_LEVEL}"
  echo "NOISE_LEVEL_END=${NOISE_LEVEL_END}"
  echo "UPDATES=${UPDATES}"
  echo "EVAL_EPISODES=${EVAL_EPISODES}"
  echo "EVAL_SEED=${EVAL_SEED}"
  echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
  echo "PPO_EPOCHS=${PPO_EPOCHS}"
  echo "MINIBATCH_SIZE=${MINIBATCH_SIZE}"
  echo "LR=${LR}"
  echo "TRAINABLE_SCOPE=${TRAINABLE_SCOPE}"
  echo "LORA_RANK=${LORA_RANK}"
  echo "LORA_ALPHA=${LORA_ALPHA}"
  echo "CRITIC_LR=${CRITIC_LR}"
  echo "CRITIC_WARMUP_STEPS=${CRITIC_WARMUP_STEPS}"
  echo "SFT_ANCHOR_WEIGHT=${SFT_ANCHOR_WEIGHT}"
  echo "GAMMA=${GAMMA}"
  echo "GAE_LAMBDA=${GAE_LAMBDA}"
  echo "CLIP_RANGE=${CLIP_RANGE}"
  echo "BETA_KL=${BETA_KL}"
  echo "EVAL_FREQ_UPDATES=${EVAL_FREQ_UPDATES}"
  echo "SAVE_FREQ_UPDATES=${SAVE_FREQ_UPDATES}"
} >"${OUTPUT_DIR}/fewshot_suite_flow_sde_ppo_run_config.txt"

EXTRA_ARGS=()
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--preflight_only)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CONDA_ENV}/bin/python" \
  -m dit4dit_rl_trainer.fewshot_suite_flow_sde_ppo_trainer \
  --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" --output_dir "${OUTPUT_DIR}" \
  --device cuda --dtype bfloat16 --libero_suite "${LIBERO_SUITE}" \
  --libero_task_id 0 --libero_task_ids all --action_unnorm_mode min_max \
  --train_task_ids "${TRAIN_TASK_IDS}" --train_task_schedule "${TRAIN_TASK_SCHEDULE}" \
  --k_steps "${K_STEPS}" --execution_horizon "${EXECUTION_HORIZON}" \
  --noise_level "${NOISE_LEVEL}" --noise_level_end "${NOISE_LEVEL_END}" \
  --rollout_episodes "${ROLLOUT_EPISODES}" --min_rollout_chunks "${MIN_ROLLOUT_CHUNKS}" \
  --max_collection_episodes "${MAX_COLLECTION_EPISODES}" --updates "${UPDATES}" \
  --ppo_epochs "${PPO_EPOCHS}" --minibatch_size "${MINIBATCH_SIZE}" --lr "${LR}" \
  --gamma "${GAMMA}" --gae_lambda "${GAE_LAMBDA}" --clip_range "${CLIP_RANGE}" --beta_kl "${BETA_KL}" \
  --trainable_scope "${TRAINABLE_SCOPE}" --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
  --critic_lr "${CRITIC_LR}" --critic_warmup_steps "${CRITIC_WARMUP_STEPS}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" --sft_anchor_data "${SFT_ANCHOR_DATA}" \
  --episode_subset_manifest "${EPISODE_SUBSET_MANIFEST}" \
  --sft_anchor_weight "${SFT_ANCHOR_WEIGHT}" --sft_anchor_batch_size "${SFT_ANCHOR_BATCH_SIZE}" \
  --min_successful_rollouts "${MIN_SUCCESSFUL_ROLLOUTS}" \
  --max_no_signal_updates "${MAX_NO_SIGNAL_UPDATES}" \
  --eval_episodes "${EVAL_EPISODES}" --eval_freq_updates "${EVAL_FREQ_UPDATES}" \
  --save_freq_updates "${SAVE_FREQ_UPDATES}" \
  --eval_seed "${EVAL_SEED}" "${EXTRA_ARGS[@]}" >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"
