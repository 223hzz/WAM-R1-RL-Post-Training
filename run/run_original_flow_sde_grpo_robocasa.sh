#!/usr/bin/env bash
# Online Flow-SDE GRPO post-training for one RoboCasa-GR1 task suite.
#
# GRPO replaces the PPO/SAPO value critic with a group-relative advantage:
#   group = one task, G rollouts from G different initial states
#   A_i   = (R_i - mean(R)) / (std(R) + eps)      R_i = undiscounted episode return
# so each update needs GROUP_SIZE * GROUPS_PER_UPDATE full rollouts.
#
# This is intentionally a single-GPU pilot: RL rollouts and backprop share one A800.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
MODEL_CONDA_ENV="${MODEL_CONDA_ENV:-/path/to/conda_envs/dit4dit}"
ROBOCASA_CONDA_ENV="${ROBOCASA_CONDA_ENV:-/path/to/conda_envs/dit4dit}"
MODEL_PYTHON="${MODEL_PYTHON:-${MODEL_CONDA_ENV}/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml}"
SFT_CKPT="${SFT_CKPT:-}"
ROBOCASA_TASK="${ROBOCASA_TASK:-gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env}"
ROBOCASA_TASKS="${ROBOCASA_TASKS:-gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env}"
GR1_SRC="${GR1_SRC:-/path/to/robocasa-gr1-tabletop-tasks}"
ROBOSUITE_SRC="${ROBOSUITE_SRC:-/path/to/robosuite-arise}"
COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-/path/to/pretrained/DiT4DiT/Cosmos-Predict2.5-2B}"
ROBOCASA_DATASET_DIR="${ROBOCASA_DATASET_DIR:-/path/to/data/robocasa/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_NAME="${RUN_NAME:-robocasa_grpo_$(echo "${ROBOCASA_TASK}" | tr '/' '_')_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/FlowSDEGRPO_RoboCasa/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run.log}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# --- GRPO group sampling -----------------------------------------------------
GROUP_SIZE="${GROUP_SIZE:-8}"                # G: rollouts per group, one init state each
GROUPS_PER_UPDATE="${GROUPS_PER_UPDATE:-4}"  # groups per optimizer update
# Rollouts per update = GROUP_SIZE * GROUPS_PER_UPDATE. Each rollout is a full
# episode, so this is the dominant wall-clock cost.
UPDATES="${UPDATES:-10}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"                # inner epochs over the collected groups
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32}"       # chunk-level minibatch, as in the PPO/SAPO paths
GRPO_ADV_CLIP_MAX="${GRPO_ADV_CLIP_MAX:-5.0}"
GRPO_GLOBAL_STD="${GRPO_GLOBAL_STD:-0}"      # 1 = normalize with cross-group reward std
GRPO_SUCCESS_BONUS="${GRPO_SUCCESS_BONUS:-0.0}"

K_STEPS="${K_STEPS:-4}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-12}" # match official N_ACTION_STEPS=12 evaluation cadence
NOISE_LEVEL="${NOISE_LEVEL:-0.10}"
NOISE_LEVEL_END="${NOISE_LEVEL_END:-0.05}"
LR="${LR:-1e-5}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-action_head}" # full Action DiT only; no VLM/Video DiT, no LoRA
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
SFT_ANCHOR_WEIGHT="${SFT_ANCHOR_WEIGHT:-0.25}"
SFT_ANCHOR_BATCH_SIZE="${SFT_ANCHOR_BATCH_SIZE:-32}"
SFT_ANCHOR_DATA="${SFT_ANCHOR_DATA:-${ROBOCASA_DATASET_DIR}}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
BASELINE_SUCCESSES="${BASELINE_SUCCESSES:-0}" # Set when --skip_preflight is enabled; measured successes in BASELINE_EPISODES
TASK_COUNT="${TASK_COUNT:-$(awk -F, '{print NF}' <<< "${ROBOCASA_TASKS}")}"
BASELINE_EPISODES="${BASELINE_EPISODES:-$((TASK_COUNT * EVAL_EPISODES))}"
# GRPO has no critic and no GAE, so the PPO/SAPO knobs ROLLOUT_EPISODES,
# MIN_ROLLOUT_CHUNKS and MIN_SUCCESSFUL_ROLLOUTS are deliberately not set here:
# rollouts per update come from GROUP_SIZE * GROUPS_PER_UPDATE, and a group is
# kept or dropped by whether its rewards differ, not by a success count.
MAX_NO_SIGNAL_UPDATES="${MAX_NO_SIGNAL_UPDATES:-3}" # consecutive degenerate-reward updates before abort
BATCH_SIZE="${BATCH_SIZE:-32}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-${BATCH_SIZE}}"
SFT_ANCHOR_BATCH_SIZE="${SFT_ANCHOR_BATCH_SIZE:-${BATCH_SIZE}}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1
trap 'rc=$?; echo "[$(date -Is)] run finished with exit_code=${rc}"; exit ${rc}' EXIT
echo "[$(date -Is)] RoboCasa Flow-SDE GRPO run starting"
echo "[config] PROJECT_DIR=${PROJECT_DIR}"
echo "[config] SFT_CKPT=${SFT_CKPT}"
echo "[config] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[config] EXECUTION_HORIZON=${EXECUTION_HORIZON} (official N_ACTION_STEPS equivalent)"
echo "[config] TRAINABLE_SCOPE=${TRAINABLE_SCOPE} LORA_RANK=${LORA_RANK} LORA_ALPHA=${LORA_ALPHA}"
echo "[config] GROUP_SIZE=${GROUP_SIZE} GROUPS_PER_UPDATE=${GROUPS_PER_UPDATE} rollouts_per_update=$((GROUP_SIZE * GROUPS_PER_UPDATE))"
echo "[config] ADV_CLIP_MAX=${GRPO_ADV_CLIP_MAX} GLOBAL_STD=${GRPO_GLOBAL_STD} SUCCESS_BONUS=${GRPO_SUCCESS_BONUS}"
echo "[config] EVAL_EPISODES=${EVAL_EPISODES} BASELINE_SUCCESSES=${BASELINE_SUCCESSES} BASELINE_EPISODES=${BASELINE_EPISODES} UPDATES=${UPDATES}"
echo "[config] MAX_NO_SIGNAL_UPDATES=${MAX_NO_SIGNAL_UPDATES}"
echo "[config] RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-none}"
RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi
GRPO_FLAGS=()
if [[ "${GRPO_GLOBAL_STD}" == "1" ]]; then
  GRPO_FLAGS+=(--grpo_global_std)
fi
for path in "${PROJECT_DIR}" "${MODEL_PYTHON}" "${ROBOCASA_CONDA_ENV}/bin/python" "${CONFIG_YAML}" "${SFT_CKPT}" "${GR1_SRC}" "${ROBOSUITE_SRC}" "${COSMOS_BASE_MODEL}" "${ROBOCASA_DATASET_DIR}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] Missing required path: ${path}" >&2; exit 1; }
done
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "[ERROR] Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${GR1_SRC}:${ROBOSUITE_SRC}:${PYTHONPATH:-}"
export DIT4DIT_COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL}"
export ROBOCASA_DATASET_DIR MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_RENDER_GPU_DEVICE_ID="${CUDA_DEVICE}" MUJOCO_EGL_DEVICE_ID="${CUDA_DEVICE}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${ROBOCASA_CONDA_ENV}/bin/python" -c 'import gymnasium as gym, robocasa, robosuite; import robocasa.utils.gym_utils.gymnasium_groot; [gym.spec(x) for x in "'"${ROBOCASA_TASKS}"'".split(",")]; print("RoboCasa tasks OK")'
# RL is in-process: the training interpreter must provide both model and
# simulator imports. Two separate interpreters can be used for evaluation,
# but cannot exchange gradients during post-training.
"${MODEL_PYTHON}" -c 'import torch, diffusers, gymnasium, robocasa, robosuite; import robocasa.utils.gym_utils.gymnasium_groot; print("Combined RL dependencies OK")'
"${MODEL_PYTHON}" -m dit4dit_rl_trainer.robocasa_flow_sde_grpo_trainer \
  --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" \
  --output_dir "${OUTPUT_DIR}" --device cuda --dtype bfloat16 \
  --robocasa_tasks "${ROBOCASA_TASKS}" --robocasa_max_episode_steps "${MAX_EPISODE_STEPS}" \
  --robocasa_render_gpu_device_id "${CUDA_DEVICE}" --train_task_id 0 --k_steps "${K_STEPS}" \
  --execution_horizon "${EXECUTION_HORIZON}" --noise_level "${NOISE_LEVEL}" \
  --noise_level_end "${NOISE_LEVEL_END}" \
  --grpo_group_size "${GROUP_SIZE}" --grpo_groups_per_update "${GROUPS_PER_UPDATE}" \
  --grpo_adv_clip_max "${GRPO_ADV_CLIP_MAX}" --grpo_success_bonus "${GRPO_SUCCESS_BONUS}" \
  "${GRPO_FLAGS[@]}" \
  --sampling_mode uniform \
  --updates "${UPDATES}" \
  --ppo_epochs "${PPO_EPOCHS}" --minibatch_size "${MINIBATCH_SIZE}" --loss_type ppo \
  --lr "${LR}" --trainable_scope "${TRAINABLE_SCOPE}" --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" --sft_anchor_data "${SFT_ANCHOR_DATA}" \
  --sft_anchor_weight "${SFT_ANCHOR_WEIGHT}" --sft_anchor_batch_size "${SFT_ANCHOR_BATCH_SIZE}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" --eval_episodes "${EVAL_EPISODES}" \
  --skip_preflight --baseline_successes "${BASELINE_SUCCESSES}" --baseline_episodes "${BASELINE_EPISODES}" \
  --max_no_signal_updates "${MAX_NO_SIGNAL_UPDATES}" "${RESUME_ARGS[@]}"
status=$?
if [[ ${status} -ne 0 ]]; then
  echo "[ERROR] GRPO trainer exited with status ${status}; inspect ${RUN_LOG} and ${OUTPUT_DIR}/flow_sde_grpo_metrics.jsonl"
else
  echo "[OK] GRPO training completed; inspect checkpoints under ${OUTPUT_DIR}/checkpoints"
fi
exit "${status}"
