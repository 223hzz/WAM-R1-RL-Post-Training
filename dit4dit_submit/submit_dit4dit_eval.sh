#!/bin/bash
#SBATCH --job-name=dit4dit_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -o logs/%j_eval.out
#SBATCH -e logs/%j_eval.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
CONDA_ENV="${CONDA_ENV:-/path/to/conda_envs/dit4dit}"
LIBERO_HOME="${LIBERO_HOME:-/path/to/LIBERO}"

source /path/to/cluster/toolshs/set-cluster-env.sh 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || true
module load CUDA/12.2 2>/dev/null || true
source /path/to/cluster/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
PY="${CONDA_ENV}/bin/python"

cd "${PROJECT_DIR}"

export PYTHONPATH="${LIBERO_HOME}:${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
if [[ ! -d "${LIBERO_HOME}/libero" ]]; then
    echo "LIBERO Python package not found at ${LIBERO_HOME}/libero" >&2
    exit 1
fi
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

mkdir -p logs

echo "===== DiT4DiT Evaluation ====="
date
hostname
echo "Python executable: ${PY}"
"${PY}" -c "import sys; print('Python runtime:', sys.executable)"
"${PY}" -c "import tyro; import libero; print('tyro/libero imports: OK')"
nvidia-smi

CKPT_PATH="${CKPT_PATH:-/path/to/pretrained/DiT4DiT/steps_20000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-$((5700 + (${SLURM_JOB_ID:-40} % 200)))}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_SERVER_WAIT_SECONDS="${MAX_SERVER_WAIT_SECONDS:-1800}"
TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"

echo "Checkpoint Path      : ${CKPT_PATH}"
echo "GPU ID               : ${GPU_ID}"
echo "Port                 : ${PORT}"
echo "Num Trials Per Task  : ${NUM_TRIALS_PER_TASK}"
echo "LIBERO_HOME          : ${LIBERO_HOME}"

[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}" >&2; exit 1; }
METADATA_DIR=""
CKPT_DIR="$(dirname "${CKPT_PATH}")"
for candidate in "${CKPT_DIR}" "$(dirname "${CKPT_DIR}")" "$(dirname "$(dirname "${CKPT_DIR}")")"; do
    if [[ -f "${candidate}/config.yaml" && -f "${candidate}/dataset_statistics.json" ]]; then
        METADATA_DIR="${candidate}"
        break
    fi
done
[[ -n "${METADATA_DIR}" ]] || {
    echo "Missing config.yaml and/or dataset_statistics.json near checkpoint: ${CKPT_PATH}" >&2
    exit 1
}
echo "Metadata directory   : ${METADATA_DIR}"

LOG_DIR="logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PY}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    --idle_timeout -1 \
    > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

SERVER_PID=$!
echo "Policy server PID: ${SERVER_PID}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && ps -p "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

SERVER_READY=0
for ((i=1; i<=MAX_SERVER_WAIT_SECONDS; i++)); do
    if (echo > "/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
        echo "Policy server ready after ${i}s" | tee -a "${LOG_DIR}/overview.log"
        SERVER_READY=1
        break
    fi
    if ! ps -p "${SERVER_PID}" >/dev/null 2>&1; then
        echo "Policy server exited early; see ${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" >&2
        tail -200 "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" >&2 || true
        exit 1
    fi
    if (( i % 30 == 0 )); then
        echo "Waiting for policy server: ${i}/${MAX_SERVER_WAIT_SECONDS}s"
    fi
    sleep 1
done

if [[ "${SERVER_READY}" -ne 1 ]]; then
    echo "Policy server timeout; see ${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" >&2
    tail -200 "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" >&2 || true
    exit 1
fi

echo "suite,success,total,success_rate" > "${LOG_DIR}/suite_summary.csv"

for TASK_SUITE in ${TASK_SUITES}; do
    VIDEO_OUT="${LOG_DIR}/${TASK_SUITE}/videos"
    mkdir -p "${VIDEO_OUT}"

    echo ""
    echo "=== Evaluating: ${TASK_SUITE} ==="

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PY}" ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "${CKPT_PATH}" \
        --args.host "127.0.0.1" \
        --args.port "${PORT}" \
        --args.task-suite-name "${TASK_SUITE}" \
        --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
        --args.video-out-path "${VIDEO_OUT}" \
        2>&1 | tee "${LOG_DIR}/eval_${TASK_SUITE}_gpu${GPU_ID}.log"

    SUCCESS_COUNT=$(find "${VIDEO_OUT}" -type f -name '*success*.mp4' | wc -l | tr -d ' ')
    TOTAL_COUNT=$(find "${VIDEO_OUT}" -type f -name '*.mp4' | wc -l | tr -d ' ')
    if [[ "${TOTAL_COUNT}" -gt 0 ]]; then
        RATE=$(awk -v s="${SUCCESS_COUNT}" -v t="${TOTAL_COUNT}" 'BEGIN { printf "%.4f", s/t }')
    else
        RATE="0.0000"
    fi
    echo "${TASK_SUITE},${SUCCESS_COUNT},${TOTAL_COUNT},${RATE}" | tee -a "${LOG_DIR}/suite_summary.csv"
    echo "=== Finished: ${TASK_SUITE} ==="
done

echo "===== All evaluations finished ====="
cat "${LOG_DIR}/suite_summary.csv"
echo "Results: ${LOG_DIR}"
