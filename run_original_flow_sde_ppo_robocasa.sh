#!/usr/bin/env bash
# ============================================================
# Online Flow-SDE PPO post-training for RoboCasa-GR1.
#
# The only difference from run_original_flow_sde_sapo_robocasa.sh is the actor objective:
#   PPO  = clipped-ratio objective (--clip_range) + KL penalty (--beta_kl)
#   SAPO = asymmetric sigmoid gate (--tau_pos / --tau_neg)
# rollout, critic, GAE, reference KL, SFT anchor and checkpoint gating are fully shared.
# The entry module dit4dit_rl_trainer.robocasa_flow_sde_ppo_trainer enforces
# --loss_type ppo, so selecting the wrong algorithm fails at startup instead of silently swapping the loss.
#
# Defaults come from a SAPO configuration that ran to completion (LR=5e-6, TRAINABLE_SCOPE=action_head,
# MINIBATCH_SIZE=SFT_ANCHOR_BATCH_SIZE=64, MIN_SUCCESSFUL_ROLLOUTS=18,
# EVAL_EPISODES=5, BASELINE_SUCCESSES=28); only the actor objective is swapped to PPO.
#
# ------------------------------------------------------------
# READ BEFORE TRAINING: under sparse rewards PPO aborts outright
#
#   Each update collects rollouts until both rollout_episodes(96) and
#   min_successful_rollouts(18) are satisfied, capped at max_collection_episodes(192);
#   too few successes skips that update, and max_no_signal_updates(3) consecutive skips raise
#     RuntimeError: Flow-SDE PPO received no successful rollout signal.
#
#   The default baseline of 28/120 ~= 23% implies 22-45 successes out of 96-192 episodes,
#   usually enough to clear the gate of 18; if a different checkpoint or a larger NOISE_LEVEL starts producing
#   "skipped: no successful rollout", first lower MIN_SUCCESSFUL_ROLLOUTS or
#   raise MAX_NO_SIGNAL_UPDATES rather than increasing UPDATES.
#
#   The baseline must be measured for your own checkpoint, never inherited:
#     PREFLIGHT_ONLY=1 bash <this script>
#   prints preflight suite success=X/Y; after changing checkpoints feed it back into
#   BASELINE_SUCCESSES / BASELINE_EPISODES before training.
#   A baseline of 0 leaves PPO without a usable gradient (all rewards within a group identical); go back and improve SFT,
#   or lower NOISE_LEVEL / EXECUTION_HORIZON to make the task easier to solve.
#
#   This baseline also anchors checkpoint gating: _accept_checkpoint requires a candidate success rate
#   strictly above max(best-so-far, baseline rate) to accept, so a wrong baseline disables gating.
#
# Usage:
#   # train directly (default checkpoint and default baseline)
#   bash run_original_flow_sde_ppo_robocasa.sh
#
#   # after swapping checkpoints, re-measure the baseline first
#   SFT_CKPT=/path/to/other.pt PREFLIGHT_ONLY=1 bash run_original_flow_sde_ppo_robocasa.sh
#
#   # resume from a PPO resume checkpoint
#   RESUME_CHECKPOINT=<...>/checkpoints/flow_sde_ppo_best_resume.pt \
#     bash run_original_flow_sde_ppo_robocasa.sh
#
# This script deliberately never calls conda activate: it always uses the absolute path ${MODEL_PYTHON}, so it is unaffected by
# whichever environment the login shell happens to activate (for example remote_sensing).
# Requirement: MODEL_CONDA_ENV must provide both the model dependencies (torch/diffusers) and the simulation dependencies
# (robocasa/robosuite) -- RL is in-process, so rollout and backprop share one interpreter.
# ============================================================
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/DiT4DiT-main}"
MODEL_CONDA_ENV="${MODEL_CONDA_ENV:-/path/to/conda_envs/dit4dit}"
ROBOCASA_CONDA_ENV="${ROBOCASA_CONDA_ENV:-/path/to/conda_envs/dit4dit}"
MODEL_PYTHON="${MODEL_PYTHON:-${MODEL_CONDA_ENV}/bin/python}"
CONFIG_YAML="${CONFIG_YAML:-${PROJECT_DIR}/DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml}"
# The SFT checkpoint must be a .pt file, not a directory. The default is RoboCasa-GR1's final_model;
# override it with the environment variable when swapping checkpoints.
SFT_CKPT="${SFT_CKPT:-/path/to/pretrained/DiT4DiT/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt}"
ROBOCASA_TASKS="${ROBOCASA_TASKS:-gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env}"
GR1_SRC="${GR1_SRC:-/path/to/robocasa-gr1-tabletop-tasks}"
ROBOSUITE_SRC="${ROBOSUITE_SRC:-/path/to/robosuite-arise}"
COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL:-/path/to/pretrained/DiT4DiT/Cosmos-Predict2.5-2B}"
ROBOCASA_DATASET_DIR="${ROBOCASA_DATASET_DIR:-/path/to/data/robocasa/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_NAME="${RUN_NAME:-robocasa_ppo_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/playground/FlowSDEPPO_RoboCasa/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run.log}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ---- rollout / eval parameters: same values as the sapo script (independent of the loss) ----
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-96}"       # initial online episode budget
MIN_ROLLOUT_CHUNKS="${MIN_ROLLOUT_CHUNKS:-1}"    # gate only; every collected episode keeps all chunks
MAX_COLLECTION_EPISODES="${MAX_COLLECTION_EPISODES:-192}"
UPDATES="${UPDATES:-10}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
# BATCH_SIZE is the common source for both minibatch and SFT anchor batch; either can be overridden separately.
BATCH_SIZE="${BATCH_SIZE:-64}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-${BATCH_SIZE}}"
K_STEPS="${K_STEPS:-4}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-12}"     # match official N_ACTION_STEPS=12 evaluation cadence
NOISE_LEVEL="${NOISE_LEVEL:-0.10}"
NOISE_LEVEL_END="${NOISE_LEVEL_END:-0.05}"
LR="${LR:-5e-6}"
# Trains the Action DiT (action_head) only, not VLM/Video DiT, and does not use LoRA.
# LORA_RANK / LORA_ALPHA only matter when TRAINABLE_SCOPE=lora; kept here purely for switchability.
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-action_head}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
SFT_ANCHOR_WEIGHT="${SFT_ANCHOR_WEIGHT:-0.25}"
SFT_ANCHOR_BATCH_SIZE="${SFT_ANCHOR_BATCH_SIZE:-${BATCH_SIZE}}"
SFT_ANCHOR_DATA="${SFT_ANCHOR_DATA:-${ROBOCASA_DATASET_DIR}}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MIN_SUCCESSFUL_ROLLOUTS="${MIN_SUCCESSFUL_ROLLOUTS:-18}"
MAX_NO_SIGNAL_UPDATES="${MAX_NO_SIGNAL_UPDATES:-3}"

# ---- PPO-specific parameters ----
# CLIP_RANGE / BETA_KL / GAMMA / GAE_LAMBDA / CRITIC_* all equal the trainer's own
# defaults (flow_sde_ppo_trainer.py:788-803); written out explicitly so they are tunable and
# so this script is self-documenting. Lower CLIP_RANGE to make PPO more conservative (the LIBERO side used ~1e-3).
CLIP_RANGE="${CLIP_RANGE:-0.2}"
BETA_KL="${BETA_KL:-0.01}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-25}"

# ---- gating modes ----
# PREFLIGHT_ONLY=1 : run the real suite evaluation only to obtain a baseline, no training. --skip_preflight is not passed,
#                    otherwise RobocasaFlowSDEPPOTrainer._evaluate_preflight would skip the evaluation entirely.
# SKIP_PREFLIGHT=1 : train directly, using BASELINE_* as the gating reference (same default as the sapo script).
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
# 28/120 is the measured baseline for this RoboCasa-GR1 final_model checkpoint (24 tasks x EVAL_EPISODES=5).
# After swapping checkpoints you must re-measure with PREFLIGHT_ONLY=1, otherwise the gating reference is wrong.
BASELINE_SUCCESSES="${BASELINE_SUCCESSES:-28}"
TASK_COUNT="${TASK_COUNT:-$(awk -F, '{print NF}' <<< "${ROBOCASA_TASKS}")}"
BASELINE_EPISODES="${BASELINE_EPISODES:-$((TASK_COUNT * EVAL_EPISODES))}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1
trap 'rc=$?; echo "[$(date -Is)] run finished with exit_code=${rc}"; exit ${rc}' EXIT

echo "[$(date -Is)] RoboCasa Flow-SDE PPO run starting"
echo "[config] PROJECT_DIR=${PROJECT_DIR}"
echo "[config] MODEL_PYTHON=${MODEL_PYTHON}"
echo "[config] SFT_CKPT=${SFT_CKPT}"
echo "[config] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[config] RUN_LOG=${RUN_LOG}"
echo "[config] CUDA_DEVICE=${CUDA_DEVICE} TASK_COUNT=${TASK_COUNT}"
echo "[config] PREFLIGHT_ONLY=${PREFLIGHT_ONLY} SKIP_PREFLIGHT=${SKIP_PREFLIGHT}"
echo "[config] BASELINE_SUCCESSES=${BASELINE_SUCCESSES} BASELINE_EPISODES=${BASELINE_EPISODES}"
echo "[config] UPDATES=${UPDATES} PPO_EPOCHS=${PPO_EPOCHS} BATCH_SIZE=${BATCH_SIZE} MINIBATCH_SIZE=${MINIBATCH_SIZE} SFT_ANCHOR_BATCH_SIZE=${SFT_ANCHOR_BATCH_SIZE}"
echo "[config] CLIP_RANGE=${CLIP_RANGE} BETA_KL=${BETA_KL} GAMMA=${GAMMA} GAE_LAMBDA=${GAE_LAMBDA}"
echo "[config] LR=${LR} CRITIC_LR=${CRITIC_LR} CRITIC_WARMUP_STEPS=${CRITIC_WARMUP_STEPS}"
echo "[config] ROLLOUT_EPISODES=${ROLLOUT_EPISODES} MIN_SUCCESSFUL_ROLLOUTS=${MIN_SUCCESSFUL_ROLLOUTS} MAX_NO_SIGNAL_UPDATES=${MAX_NO_SIGNAL_UPDATES}"
echo "[config] K_STEPS=${K_STEPS} EXECUTION_HORIZON=${EXECUTION_HORIZON} NOISE_LEVEL=${NOISE_LEVEL}->${NOISE_LEVEL_END}"
echo "[config] TRAINABLE_SCOPE=${TRAINABLE_SCOPE} LORA_RANK=${LORA_RANK} LORA_ALPHA=${LORA_ALPHA}"
echo "[config] SFT_ANCHOR_WEIGHT=${SFT_ANCHOR_WEIGHT} SFT_ANCHOR_DATA=${SFT_ANCHOR_DATA}"
echo "[config] RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-none}"

# ------------------------------------------------------------
# Pre-flight validation: required paths must exist
# ------------------------------------------------------------
for path in "${PROJECT_DIR}" "${MODEL_PYTHON}" "${ROBOCASA_CONDA_ENV}/bin/python" \
            "${CONFIG_YAML}" "${SFT_CKPT}" "${GR1_SRC}" "${ROBOSUITE_SRC}" \
            "${COSMOS_BASE_MODEL}" "${ROBOCASA_DATASET_DIR}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] Missing required path: ${path}" >&2; exit 1; }
done
# --checkpoint expects a .pt file; passing a directory fails inside engine._load_pretrained_model
if [[ -d "${SFT_CKPT}" ]]; then
  echo "[ERROR] SFT_CKPT must be a .pt file, not a directory: ${SFT_CKPT}" >&2
  echo "        Expected form: .../best_SFT/checkpoints/steps_80000_pytorch_model.pt" >&2
  exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "[ERROR] Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${GR1_SRC}:${ROBOSUITE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

# PYTHONPATH sanitisation: if the login shell has another conda environment active, its
# site-packages may linger in PYTHONPATH and be imported first, making results non-deterministic.
if [[ "${ALLOW_FOREIGN_PYTHONPATH:-0}" != "1" ]]; then
  FOREIGN_ENTRIES=()
  IFS=':' read -ra _pp_entries <<< "${PYTHONPATH}"
  for _entry in "${_pp_entries[@]}"; do
    [[ -z "${_entry}" ]] && continue
    case "${_entry}" in
      "${PROJECT_DIR}"|"${PROJECT_DIR}"/*|"${GR1_SRC}"|"${GR1_SRC}"/*|"${ROBOSUITE_SRC}"|"${ROBOSUITE_SRC}"/*) ;;
      */conda_envs/*) FOREIGN_ENTRIES+=("${_entry}") ;;
    esac
  done
  if (( ${#FOREIGN_ENTRIES[@]} > 0 )); then
    echo "[ERROR] PYTHONPATH contains paths from another conda environment:" >&2
    for _entry in "${FOREIGN_ENTRIES[@]}"; do echo "        ${_entry}" >&2; done
    echo "        Run in a fresh shell, or unset PYTHONPATH and retry;" >&2
    echo "        set ALLOW_FOREIGN_PYTHONPATH=1 to proceed if you are sure it is harmless." >&2
    exit 1
  fi
fi
echo "[env] PYTHONPATH=${PYTHONPATH}"

export DIT4DIT_COSMOS_BASE_MODEL="${COSMOS_BASE_MODEL}"
export ROBOCASA_DATASET_DIR MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_RENDER_GPU_DEVICE_ID="${CUDA_DEVICE}" MUJOCO_EGL_DEVICE_ID="${CUDA_DEVICE}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------------------------------------
# Dependency self-check
# ------------------------------------------------------------
"${ROBOCASA_CONDA_ENV}/bin/python" -c 'import gymnasium as gym, robocasa, robosuite; import robocasa.utils.gym_utils.gymnasium_groot; [gym.spec(x) for x in "'"${ROBOCASA_TASKS}"'".split(",")]; print("RoboCasa tasks OK")'
# RL is in-process: the training interpreter must supply both model and simulation dependencies.
# Two separate interpreters can run evaluation, but cannot exchange gradients during post-training.
"${MODEL_PYTHON}" -c 'import torch, diffusers, gymnasium, robocasa, robosuite; import robocasa.utils.gym_utils.gymnasium_groot; print("Combined RL dependencies OK")'

# ------------------------------------------------------------
# Argument assembly
# ------------------------------------------------------------
GATE_ARGS=()
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  # Note: --skip_preflight is deliberately omitted here, otherwise the evaluation would be short-circuited.
  # --allow_zero_preflight is required, otherwise require_preflight_success raises when the baseline is 0,
  # and "baseline is 0" is exactly the fact we want to measure at this step.
  GATE_ARGS+=(--preflight_only --allow_zero_preflight)
elif [[ "${SKIP_PREFLIGHT}" == "1" ]]; then
  GATE_ARGS+=(--skip_preflight --baseline_successes "${BASELINE_SUCCESSES}" --baseline_episodes "${BASELINE_EPISODES}")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "============================================================"
echo "Launch | loss_type=ppo | mode: $([[ "${PREFLIGHT_ONLY}" == "1" ]] && echo 'PREFLIGHT_ONLY' || { [[ "${SKIP_PREFLIGHT}" == "1" ]] && echo 'train(skip_preflight)' || echo 'train(real preflight)'; })"
echo "============================================================"

# ------------------------------------------------------------
# Launch
# ------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${MODEL_PYTHON}" \
  -m dit4dit_rl_trainer.robocasa_flow_sde_ppo_trainer \
  --config_yaml "${CONFIG_YAML}" --checkpoint "${SFT_CKPT}" \
  --output_dir "${OUTPUT_DIR}" --device cuda --dtype bfloat16 \
  --robocasa_tasks "${ROBOCASA_TASKS}" \
  --robocasa_max_episode_steps "${MAX_EPISODE_STEPS}" \
  --robocasa_render_gpu_device_id "${CUDA_DEVICE}" \
  --train_task_id 0 --sampling_mode uniform \
  --k_steps "${K_STEPS}" --execution_horizon "${EXECUTION_HORIZON}" \
  --noise_level "${NOISE_LEVEL}" --noise_level_end "${NOISE_LEVEL_END}" \
  --rollout_episodes "${ROLLOUT_EPISODES}" \
  --min_rollout_chunks "${MIN_ROLLOUT_CHUNKS}" \
  --max_collection_episodes "${MAX_COLLECTION_EPISODES}" \
  --updates "${UPDATES}" --ppo_epochs "${PPO_EPOCHS}" \
  --minibatch_size "${MINIBATCH_SIZE}" \
  --loss_type ppo --clip_range "${CLIP_RANGE}" --beta_kl "${BETA_KL}" \
  --gamma "${GAMMA}" --gae_lambda "${GAE_LAMBDA}" \
  --critic_lr "${CRITIC_LR}" --critic_warmup_steps "${CRITIC_WARMUP_STEPS}" \
  --lr "${LR}" --trainable_scope "${TRAINABLE_SCOPE}" \
  --lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" \
  --sft_anchor_data "${SFT_ANCHOR_DATA}" \
  --sft_anchor_weight "${SFT_ANCHOR_WEIGHT}" \
  --sft_anchor_batch_size "${SFT_ANCHOR_BATCH_SIZE}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" --eval_episodes "${EVAL_EPISODES}" \
  --min_successful_rollouts "${MIN_SUCCESSFUL_ROLLOUTS}" \
  --max_no_signal_updates "${MAX_NO_SIGNAL_UPDATES}" \
  ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

status=$?
if [[ ${status} -ne 0 ]]; then
  echo "[ERROR] PPO trainer exited with status ${status}; inspect ${RUN_LOG} and ${OUTPUT_DIR}/flow_sde_ppo_metrics.jsonl"
else
  echo "[OK] PPO training completed; inspect checkpoints under ${OUTPUT_DIR}/checkpoints"
fi
exit "${status}"
