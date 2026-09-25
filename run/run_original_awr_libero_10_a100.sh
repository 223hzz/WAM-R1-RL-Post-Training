#!/usr/bin/env bash
# A100 AWR continuation launcher for all 10 tasks in the LIBERO_10 suite.
# Collection is deterministic and balanced: task ids 0..9 are visited in
# round-robin order (one episode per task before returning to task 0).
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Match the A100 command/data layout supplied for the object run. Every value
# remains overridable from the environment, including RESUME for continuation.
export PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
export CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
export COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-/path/to/pretrained/Cosmos-Predict2.5-2B}"
export SFT_CKPT="/path/to/DiT4DiT-main/playground/final model/pytorch_model.pt"
export CUDA_DEVICE="${CUDA_DEVICE:-7}"
export ANCHOR_DATA="${ANCHOR_DATA:-/path/to/data/LIBERO}"
export ANCHOR_DATA_MIX="${ANCHOR_DATA_MIX:-libero_10}"
export LIBERO_SUITE="${LIBERO_SUITE:-libero_10}"
export OBJECT_DATASET_DIR="${OBJECT_DATASET_DIR:-${ANCHOR_DATA}/libero_10_no_noops_1.0.0_lerobot}"
export CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/libero/dit4dit_libero.yaml}"
export RUN_NAME="${RUN_NAME:-libero_10_awr_round_robin}"

# A full 10-task cycle is five episodes with the A100 defaults. The trainer
# itself enforces task-balanced replay in addition to this collection order.
export TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-full}"
export ITERATIONS="${ITERATIONS:-10}"
export ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-50}"
export MIN_SUCCESSFUL_ROLLOUTS="${MIN_SUCCESSFUL_ROLLOUTS:-20}"
export MAX_COLLECTION_EPISODES="${MAX_COLLECTION_EPISODES:-100}"
export ONLINE_TRAJECTORIES_PER_UPDATE="${ONLINE_TRAJECTORIES_PER_UPDATE:-25}"
export ONLINE_CHUNK_BATCH_SIZE="${ONLINE_CHUNK_BATCH_SIZE:-64}"
export REPLAY_CAPACITY="${REPLAY_CAPACITY:-8192}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export POLICY_STEPS_PER_ITERATION="${POLICY_STEPS_PER_ITERATION:-1}"
export ONLINE_FRACTION="${ONLINE_FRACTION:-0.5}"
export PREFLIGHT_EVAL_EPISODES="${PREFLIGHT_EVAL_EPISODES:-20}"
export EVAL_EPISODES="${EVAL_EPISODES:-20}"
export EVAL_FREQ="${EVAL_FREQ:-2}"
export BEST_EVAL_FREQ="${BEST_EVAL_FREQ:-1}"
export LIBERO_TASK_SCHEDULE="${LIBERO_TASK_SCHEDULE:-round_robin}"
export LIBERO_TASK_BLOCK_SIZE="${LIBERO_TASK_BLOCK_SIZE:-1}"

if [[ "${LIBERO_TASK_SCHEDULE}" != "round_robin" || "${LIBERO_TASK_BLOCK_SIZE}" != "1" ]]; then
  echo "This launcher requires LIBERO_TASK_SCHEDULE=round_robin and LIBERO_TASK_BLOCK_SIZE=1 for uniform 10-task rotation." >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/run_original_awr_object_a100.sh" "$@"
