#!/bin/bash
# Object-suite AWR launcher. Kept separate from task-5 AWR and RFT launchers.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
SFT_CKPT="${SFT_CKPT:-${PROJECT_DIR}/playground/dit4dit_libero_all_80000_2a800/pytorch_model.pt}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
ANCHOR_DATA="${ANCHOR_DATA:-/path/to/data/libero}"
ANCHOR_DATA_MIX="${ANCHOR_DATA_MIX:-libero_object}"
RUN_NAME="${RUN_NAME:-object_awr_r20_p4_b32}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/Checkpoints_post_original/${RUN_NAME}}"
RESUME="${RESUME:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-200}"
MIN_SUCCESSFUL_ROLLOUTS="${MIN_SUCCESSFUL_ROLLOUTS:-50}"
MAX_COLLECTION_EPISODES="${MAX_COLLECTION_EPISODES:-400}"
ONLINE_TRAJECTORIES_PER_UPDATE="${ONLINE_TRAJECTORIES_PER_UPDATE:-50}"
ONLINE_CHUNK_BATCH_SIZE="${ONLINE_CHUNK_BATCH_SIZE:-2}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-lora}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${LR:-5e-5}"
ONLINE_FRACTION="${ONLINE_FRACTION:-0.5}"
K_STEPS="${K_STEPS:-16}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-4}"
NOISE_LEVEL="${NOISE_LEVEL:-0.1}"
ITERATIONS="${ITERATIONS:-10}"
POLICY_STEPS_PER_ITERATION="${POLICY_STEPS_PER_ITERATION:-4}"
VALUE_STEPS_PER_ITERATION="${VALUE_STEPS_PER_ITERATION:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-8192}"
COLLECTION_SAMPLER="${COLLECTION_SAMPLER:-sde}"
GAMMA="${GAMMA:-0.99}"
AWR_BETA="${AWR_BETA:-0.5}"
AWR_MAX_WEIGHT="${AWR_MAX_WEIGHT:-10.0}"
VALUE_LR="${VALUE_LR:-1e-4}"
SAVE_FREQ="${SAVE_FREQ:-1}"
EVAL_FREQ="${EVAL_FREQ:-2}"
BEST_EVAL_FREQ="${BEST_EVAL_FREQ:-1}"
PREFLIGHT_EVAL_EPISODES="${PREFLIGHT_EVAL_EPISODES:-20}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-10000}"
ROLLOUT_STALL_TIMEOUT="${ROLLOUT_STALL_TIMEOUT:-600}"

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

EXTRA_ARGS=()
[[ -n "${RESUME}" ]] && EXTRA_ARGS+=(--resume "${RESUME}")
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_awr_object.out"
STDERR_LOG="${PROJECT_DIR}/logs/${TIMESTAMP}_awr_object.err"
printf '%s\n' "${STDOUT_LOG}" >"${OUTPUT_DIR}/stdout_log_path.txt"
printf '%s\n' "${STDERR_LOG}" >"${OUTPUT_DIR}/stderr_log_path.txt"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CONDA_ENV}/bin/python" -m dit4dit_rl_trainer.post_trainer \
  --method awr --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" \
  --anchor_data "${ANCHOR_DATA}" --anchor_data_mix "${ANCHOR_DATA_MIX}" \
  --output_dir "${OUTPUT_DIR}" --device cuda --dtype bfloat16 \
  --train_task_id 0 --train_task_ids all --libero_suite libero_object \
  --libero_task_id 0 --libero_task_ids all --action_unnorm_mode min_max \
  --trainable_scope "${TRAINABLE_SCOPE}" --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
  --lr "${LR}" --iterations "${ITERATIONS}" --episodes_per_iteration "${ROLLOUT_EPISODES}" \
  --min_successful_rollouts "${MIN_SUCCESSFUL_ROLLOUTS}" \
  --max_collection_episodes "${MAX_COLLECTION_EPISODES}" \
  --online_trajectories_per_update "${ONLINE_TRAJECTORIES_PER_UPDATE}" \
  --online_chunk_batch_size "${ONLINE_CHUNK_BATCH_SIZE}" \
  --policy_steps_per_iteration "${POLICY_STEPS_PER_ITERATION}" --batch_size "${BATCH_SIZE}" \
  --online_fraction "${ONLINE_FRACTION}" --replay_capacity "${REPLAY_CAPACITY}" \
  --k_steps "${K_STEPS}" --collection_sampler "${COLLECTION_SAMPLER}" \
  --noise_level "${NOISE_LEVEL}" --execution_horizon "${EXECUTION_HORIZON}" \
  --max_episode_steps 280 --gamma "${GAMMA}" --awr_beta "${AWR_BETA}" \
  --awr_max_weight "${AWR_MAX_WEIGHT}" --value_lr "${VALUE_LR}" \
  --value_steps_per_iteration "${VALUE_STEPS_PER_ITERATION}" --save_freq "${SAVE_FREQ}" \
  --eval_freq "${EVAL_FREQ}" --best_eval_freq "${BEST_EVAL_FREQ}" \
  --preflight_eval_episodes "${PREFLIGHT_EVAL_EPISODES}" --eval_episodes "${EVAL_EPISODES}" \
  --eval_seed "${EVAL_SEED}" --rollout_stall_timeout "${ROLLOUT_STALL_TIMEOUT}" \
  "${EXTRA_ARGS[@]}" >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"
