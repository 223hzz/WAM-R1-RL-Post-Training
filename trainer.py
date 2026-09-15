# -*- coding: utf-8 -*-
"""
Step 4: 两阶段训练主循环 (Training Loop)
==========================================

Action-RL 后训练的最终模块，包含：

阶段 1 — Few-shot SFT 冷启动::
    用 ~40 条专家演示，Flow Matching MSE 损失训练 Action DiT。
    目标：初始成功率 0% → 40%~50%。

阶段 2 — 在线 Action-RL 演化::
    1. 环境观测 o_t → RAFT 光流 + Affordance 热力图 → 三流 VAE 编码 → H_video
    2. sample_hybrid_sde_action → G 条并行动作轨迹 (SDE 探索)
    3. 环境执行动作块 → 获取 reward
    4. compute_grpo_loss → backward → optimizer.step
    5. 下一步观测：同步驱动 RAFT + Affordance 更新 Z_f, Z_v

严格约束：
  - 不修改 dit4dit_multistream/ 内的任何代码
  - H_video 通过 backbone_interface.forward() + Hook 拦截第 18 层获取
  - RAFT / Affordance 作为冻结的推理模块，在 rollout 时实时驱动
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
import os
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from dit4dit_rl_trainer.engine import RLEngine, RLEngineConfig
from dit4dit_rl_trainer.sde_rollout import (
    RolloutResult,
    sample_hybrid_sde_action,
    sample_group_hybrid_sde,
)
from dit4dit_rl_trainer.math_utils import (
    compute_group_advantages,
    compute_grpo_loss,
    compute_sft_loss,
)

logger = logging.getLogger(__name__)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


# ============================================================================
# 环境抽象接口
# ============================================================================

@runtime_checkable
class EnvProtocol(Protocol):
    """仿真环境抽象接口 (兼容 LIBERO / ManiSkill / RoboCasa 等)。

    实现者需提供以下方法。具体环境适配见 EnvWrapper 子类。
    """

    def reset(self) -> Dict[str, Any]:
        """重置环境，返回初始观测。

        Returns:
            obs: 观测字典，至少包含:
                - "rgb": np.ndarray (H, W, 3) uint8 当前帧 RGB
                - "state": np.ndarray (D_state,) 机器人状态
                - "lang": str 语言指令
                - "contact_points": np.ndarray (K, 3) 接触点坐标 (可选)
        """
        ...

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """执行单步动作。

        Args:
            action: np.ndarray (D_action,) 单步动作

        Returns:
            obs: 下一时刻观测 (同 reset 格式)
            reward: float 奖励
            done: bool 是否结束
            info: dict 额外信息
        """
        ...


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class SFTConfig:
    """阶段 1: Few-shot SFT 配置。"""
    num_demos: int = 40                # 专家演示数量 (~40 条)
    num_epochs: int = 50               # SFT 训练轮数
    learning_rate: float = 1e-4        # SFT 学习率 (比 RL 阶段大)
    batch_size: int = 8                # SFT batch size
    save_path: str = "checkpoints/sft_checkpoint.pt"


@dataclass
class GRPOConfig:
    """阶段 2: 在线 GRPO 配置。"""
    chunks_per_episode: int = 1
    collect_all_chunks: bool = False  # 记录整条 episode 的所有重规划 chunk
    chunk_stride: int = 4
    inner_epochs: int = 1
    num_iterations: int = 10000        # GRPO 迭代次数
    group_size: int = 8                # GRPO 组大小 G
    action_horizon: int = 8            # 动作块长度 T
    execution_horizon: int = 0         # 每次重规划前实际执行的动作数；0 表示完整 chunk
    num_inference_steps: int = 4       # SDE 采样步数 K (建议 K=4)
    noise_level: float = 0.7           # SDE 噪声系数 a_noise
    noise_level_end: float = 0.7       # 线性退火终点；默认保持恒定
    sde_window_size: int = 1           # SDE 步数 (随机选 1 步)
    sde_window_range: Tuple[int, int] = (0, 2)  # SDE 窗口范围
    clip_range: float = 2e-6           # Continuous SDE log-probs need a tight PPO clip
    beta_kl: float = 0.08              # KL penalty beta
    adv_clip_max: float = 5.0          # 优势裁剪
    global_std: bool = True            # 全局标准差
    learning_rate: float = 5e-6        # Conservative LoRA RL learning rate
    max_grad_norm: float = 1.0         # 梯度裁剪
    save_freq: int = 500               # 保存频率
    eval_freq: int = 200               # 评估频率
    log_freq: int = 10                 # 日志频率
    progress_steps: int = 64           # rollout 内进度日志间隔；0 表示关闭
    warmup_steps: int = 100            # 学习率预热
    reward_success: float = 1.0        # 成功奖励
    reward_penalty: float = 0.0        # 失败惩罚
    max_episode_steps: int = 520       # LIBERO episode 上限
    preflight_eval_episodes: int = 0   # GRPO 前验证无噪声 SFT 基线；0 表示关闭
    eval_episodes: int = 10
    eval_seed: int = 10000
    require_preflight_success: bool = True
    stop_on_negative_paired_delta: bool = False
    max_no_signal_updates: int = 5    # 连续同奖励 group 的终止阈值；0 表示关闭
    temporal_credit_decay: float = 1.0  # Later chunks get more credit when < 1
    spread_chunks_across_episode: bool = False
    common_continuation_seed: bool = False  # Common random numbers for untrained chunks
    rsft_anchor_weight: float = 0.0  # Auxiliary successful-trajectory Flow Matching loss
    rsft_online_fraction: float = 0.3  # Target fraction of online successes in RSFT batches
    rsft_anchor_batch_size: int = 4
    rsft_replay_capacity: int = 128
    rollout_stall_timeout: float = 0.0  # 0 disables the watchdog
    rollout_heartbeat_seconds: float = 30.0


class RolloutStallWatchdog:
    """Dump Python stacks and terminate when a rollout stops making progress."""

    def __init__(
        self,
        timeout_seconds: float,
        heartbeat_seconds: float,
        output_dir: str,
    ) -> None:
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.heartbeat_seconds = max(0.0, float(heartbeat_seconds))
        self.output_dir = Path(output_dir)
        self._lock = threading.Lock()
        self._active = False
        self._phase = "idle"
        self._last_beat = time.monotonic()
        self._last_log = self._last_beat
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self.timeout_seconds > 0.0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="rollout-stall-watchdog",
            daemon=True,
        )
        self._thread.start()
        print(
            "[GRPO][Watchdog] enabled: rollout timeout="
            f"{self.timeout_seconds:.0f}s; send SIGUSR1 to dump Python stacks.",
            flush=True,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def arm(self, phase: str) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            self._active = True
            self._phase = phase
            self._last_beat = now
            self._last_log = now

    def beat(self, phase: Optional[str] = None) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            if not self._active:
                return
            if phase is not None:
                self._phase = phase
            current_phase = self._phase
            should_log = (
                self.heartbeat_seconds > 0.0
                and now - self._last_log >= self.heartbeat_seconds
            )
            self._last_beat = now
            if should_log:
                self._last_log = now
        if should_log:
            print(f"[GRPO][Heartbeat] {current_phase}", flush=True)

    def disarm(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active = False
            self._phase = "idle"

    def _run(self) -> None:
        poll_seconds = min(5.0, max(0.5, self.timeout_seconds / 10.0))
        while not self._stop_event.wait(poll_seconds):
            with self._lock:
                if not self._active:
                    continue
                elapsed = time.monotonic() - self._last_beat
                phase = self._phase
            if elapsed < self.timeout_seconds:
                continue

            diagnostics_dir = self.output_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dump_path = diagnostics_dir / f"rollout_stall_{timestamp}.txt"
            with dump_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    "Rollout watchdog timeout\n"
                    f"phase={phase}\n"
                    f"stalled_seconds={elapsed:.1f}\n\n"
                )
                faulthandler.dump_traceback(file=handle, all_threads=True)
            print(
                "[GRPO][FATAL] Rollout made no progress for "
                f"{elapsed:.0f}s in {phase}. Python stacks: {dump_path}. "
                "Exiting with code 124 so the job can be resumed from its last checkpoint.",
                flush=True,
            )
            os._exit(124)


@dataclass
class GRPOChunkSample:
    """One recorded action decision and the observation that conditioned it."""

    rollout: RolloutResult
    vl_embs: torch.Tensor
    state: Optional[torch.Tensor]
    chunk_index: int
    ref_velocities: Optional[List[torch.Tensor]] = None


@dataclass
class RSFTReplaySample:
    """A detached successful observation/action pair for online Flow Matching."""

    vl_embs: torch.Tensor
    target_actions: torch.Tensor
    action_mask: torch.Tensor
    state: Optional[torch.Tensor]


def build_training_chunk_indices(
    chunks_per_episode: int,
    chunk_stride: int,
    max_decision_index: Optional[int] = None,
) -> List[int]:
    """Return deterministic training decisions, optionally spanning the episode."""
    if chunks_per_episode < 1:
        raise ValueError("chunks_per_episode must be >= 1")
    if chunk_stride < 1:
        raise ValueError("chunk_stride must be >= 1")
    if max_decision_index is None:
        return [index * chunk_stride for index in range(chunks_per_episode)]
    if max_decision_index < 0:
        raise ValueError("max_decision_index must be >= 0")
    if chunks_per_episode == 1 or max_decision_index == 0:
        return [0]
    indices = [
        round(position * max_decision_index / (chunks_per_episode - 1))
        for position in range(chunks_per_episode)
    ]
    return sorted(set(indices))


def build_temporal_chunk_weights(
    chunk_indices: List[int], decay: float = 1.0
) -> List[float]:
    """Return normalized credit weights, applying decay once per recorded chunk."""
    if not chunk_indices:
        raise ValueError("chunk_indices must not be empty")
    if not 0.0 < decay <= 1.0:
        raise ValueError("temporal credit decay must be in (0, 1]")
    raw = [
        decay ** (len(chunk_indices) - 1 - position)
        for position in range(len(chunk_indices))
    ]
    total = sum(raw)
    return [weight / total for weight in raw]


@dataclass
class TrainerConfig:
    """总训练配置。"""
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    engine: RLEngineConfig = field(default_factory=RLEngineConfig)
    device: str = "cuda"
    output_dir: str = "checkpoints"
    wandb_project: str = "action_rl"
    seed: int = 42
    model_variant: str = "auto"
    normalize_state: bool = True
    unnorm_key: Optional[str] = None
    action_unnorm_mode: str = "auto"
    preflight_only: bool = False


# ============================================================================
# 主训练器
# ============================================================================

class RLTrainer:
    """两阶段训练器：Few-shot SFT → 在线 Action-RL。

    核心数据流::

        环境观测 o_t
          ├→ RGB ──────────────────→ VAE.encode → Z_x (语义)
          ├→ RAFT(prev_rgb, rgb) → HSV → VAE.encode → Z_f (物理)
          └→ Affordance(rgb, contact) → VAE.encode → Z_v (意图)
                                                        │
                                          [Z_x; Z_f; Z_v] → Video DiT → H_video (B, 3N, D)
                                                        │
                                          sample_hybrid_sde_action(H_video) → 动作块
                                                        │
                                          环境执行 → reward → GRPO loss → backward
    """

    def __init__(
        self,
        engine: RLEngine,
        env: EnvProtocol,
        raft_extractor: Optional[Any],
        flow_codec: Optional[Any],
        affordance_generator: Optional[Any],
        cfg: TrainerConfig,
        sft_dataset: Optional[Any] = None,
        rsft_anchor_dataset: Optional[Any] = None,
    ):
        """
        Args:
            engine: Step 1 初始化的 RL 引擎 (含冻结 backbone + 可训练 Action DiT)
            env: 仿真环境 (实现 EnvProtocol)
            raft_extractor: RAFT 光流提取器 (冻结, rollout 时实时驱动)
            flow_codec: 光流 HSV 编码器 (冻结)
            affordance_generator: 意图热力图生成器 (冻结)
            cfg: 训练配置
            sft_dataset: 阶段 1 SFT 数据集 (None 时跳过 SFT 直接进入 GRPO)
        """
        self.engine = engine
        self.env = env
        self.raft = raft_extractor
        self.flow_codec = flow_codec
        self.affordance = affordance_generator
        self.cfg = cfg
        self.sft_dataset = sft_dataset
        self.rsft_anchor_dataset = rsft_anchor_dataset
        self._rsft_anchor_iterator = (
            iter(rsft_anchor_dataset) if rsft_anchor_dataset is not None else None
        )
        self.rsft_replay: List[RSFTReplaySample] = []

        self.device = torch.device(getattr(engine, "device", cfg.device))
        self.model = engine.model  # 原版或 Multistream DiT4DiT (backbone 冻结)
        self.action_model = engine.action_model
        self.optimizer = engine.optimizer
        self.model_variant = getattr(engine, "model_variant", "multistream")
        if cfg.model_variant != "auto":
            self.model_variant = cfg.model_variant

        self.action_norm_stats = None
        norm_stats = getattr(self.model, "norm_stats", None)
        if norm_stats:
            from DiT4DiT.model.framework.base_framework import baseframework

            unnorm_key = baseframework._check_unnorm_key(norm_stats, cfg.unnorm_key)
            self.action_norm_stats = norm_stats[unnorm_key]["action"]
            self.action_unnorm_mode = cfg.action_unnorm_mode
            if self.action_unnorm_mode == "auto":
                self.action_unnorm_mode = (
                    "min_max" if self.model_variant == "original" else "q99"
                )
            low_key, high_key = (
                ("min", "max")
                if self.action_unnorm_mode == "min_max"
                else ("q01", "q99")
            )
            if low_key not in self.action_norm_stats or high_key not in self.action_norm_stats:
                raise KeyError(
                    f"Action statistics do not contain {low_key}/{high_key} "
                    f"required by action_unnorm_mode={self.action_unnorm_mode!r}"
                )
            logger.info(f"[Trainer] Action unnormalization key: {unnorm_key}")
            print(
                f"[Preprocess] Action unnormalization key={unnorm_key}; "
                f"mode={self.action_unnorm_mode}; "
                f"low={np.asarray(self.action_norm_stats[low_key]).tolist()}; "
                f"high={np.asarray(self.action_norm_stats[high_key]).tolist()}",
                flush=True,
            )
        else:
            print(
                "[Preprocess][WARN] No dataset action statistics were loaded; "
                "sampled normalized actions will be sent to the environment "
                "without unnormalization.",
                flush=True,
            )

        self._image_preprocessing_logged = False
        self._state_preprocessing_logged = False
        self._action_preprocessing_logged = False

        # 感知模块设为 eval (冻结)
        self._freeze_perception_modules()

        # 奖励历史 (用于 global_std)
        self.reward_history: List[torch.Tensor] = []

        # 全局步数
        self.global_step = 0
        self.consecutive_no_signal_updates = 0
        self.no_signal_updates_by_task: Dict[str, int] = {}
        self.best_eval_successes = -1
        self.best_eval_episodes = 0
        self.best_eval_step = 0
        # A tied first paired evaluation is still a valid best checkpoint.  Start
        # below every feasible delta so it is retained for later comparison.
        self.best_eval_paired_delta = float("-inf")
        self.preflight_records: Dict[Tuple[int, int, int], bool] = {}
        self.rollout_watchdog = RolloutStallWatchdog(
            timeout_seconds=self.cfg.grpo.rollout_stall_timeout,
            heartbeat_seconds=self.cfg.grpo.rollout_heartbeat_seconds,
            output_dir=self.cfg.output_dir,
        )
        self._enable_signal_stack_dump()

        if self.cfg.grpo.rsft_anchor_weight > 0.0:
            print(
                "[RSFT] online_success_replay enabled: "
                f"weight={self.cfg.grpo.rsft_anchor_weight}, "
                f"online_fraction={self.cfg.grpo.rsft_online_fraction}, "
                f"capacity={self.cfg.grpo.rsft_replay_capacity}, "
                f"offline_dataset={'yes' if rsft_anchor_dataset is not None else 'no'}",
                flush=True,
            )

        logger.info("[Trainer] Initialized. "
                    f"SFT dataset: {'✓' if sft_dataset else '✗ (skip SFT)'}, "
                    f"Env: {type(env).__name__}")

    # ========================================================================
    # 三流特征提取 —— 环境观测 → H_video
    # ========================================================================

    @staticmethod
    def _enable_signal_stack_dump() -> None:
        """Make ``kill -USR1 <pid>`` write all Python thread stacks to stderr."""
        if not hasattr(signal, "SIGUSR1"):
            return
        try:
            faulthandler.enable(all_threads=True)
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not enable SIGUSR1 stack dumps: %s", exc)

    def _observe_to_images(
        self,
        obs: Dict[str, Any],
        prev_rgb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """将环境观测转换为三路图像 (实时协同更新)。

        Args:
            obs: 环境观测字典
            prev_rgb: 上一帧 RGB (用于光流计算)。第一帧时为 None。

        Returns:
            rgb_img: (H, W, 3) uint8
            flow_img: (H, W, 3) uint8 光流 HSV 伪彩色
            value_img: (H, W, 3) uint8 意图热力图
        """
        rgb = obs["rgb"]  # (H, W, 3) uint8

        if self.model_variant == "original":
            wrist_rgb = obs.get("wrist_rgb")
            if wrist_rgb is None:
                raise RuntimeError(
                    "Original DiT4DiT LIBERO inference requires both agentview "
                    "and wrist images, matching the official evaluation path."
                )
            if rgb.shape[0] != wrist_rgb.shape[0]:
                raise ValueError(
                    "Primary and wrist images must have the same height before "
                    f"concatenation, got {rgb.shape} and {wrist_rgb.shape}."
                )
            # Official model2libero_interface.py concatenates both 224x224
            # views width-wise before build_cosmos_inputs.
            concat_rgb = np.concatenate([rgb, wrist_rgb], axis=1)
            if not self._image_preprocessing_logged:
                print(
                    f"[Preprocess] Original image input: primary={rgb.shape}, "
                    f"wrist={wrist_rgb.shape}, concatenated={concat_rgb.shape}",
                    flush=True,
                )
                self._image_preprocessing_logged = True
            return concat_rgb, None, None

        if self.raft is None or self.flow_codec is None or self.affordance is None:
            raise RuntimeError("Multistream training requires RAFT, FlowCodec and Affordance")

        # ---- 物理光流: RAFT(prev_rgb → rgb) → HSV 编码 ----
        if prev_rgb is not None:
            # 注意: RAFTFlowExtractor 的接口是 __call__(frame1, frame2), 不是 extract()
            flow = self.raft(prev_rgb, rgb)  # (H, W, 2) 光流位移
            # 注意: FlowCodec.encode() 返回 (rgb_image, max_magnitude) 元组
            flow_img, _ = self.flow_codec.encode(flow)  # (H, W, 3) HSV 伪彩色
        else:
            # 第一帧无光流，用零光流
            flow_img = np.zeros_like(rgb)

        # ---- 意图价值: Affordance 热力图 ----
        # 注意: AffordanceMapGenerator.generate() 不接收 rgb,
        # 签名为 generate(contact_points=None, camera_intrinsics=None, ...)
        contact_points = obs.get("contact_points", None)
        value_img = self.affordance.generate(contact_points=contact_points)  # (H, W, 3)

        return rgb, flow_img, value_img

    def extract_h_video(
        self,
        rgb_frames: List[np.ndarray],
        flow_frames: Optional[List[np.ndarray]],
        value_frames: Optional[List[np.ndarray]],
        lang: str,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """原版 RGB 或三流图像序列 → Video DiT 条件特征。

        通过 backbone_interface 前向传播 + Hook 拦截获取。
        backbone 已冻结，在 no_grad 下执行。

        Args:
            rgb_frames: List of (H, W, 3) uint8
            flow_frames: List of (H, W, 3) uint8
            value_frames: List of (H, W, 3) uint8
            lang: 语言指令
            batch_size: B (通常为 1 或 G)

        Returns:
            H_video: 原版 (B,S,D) 或三流版 (B,3N,D)
        """
        if self.model_variant == "multistream":
            backbone_inputs = self.model.backbone_interface.build_multistream_inputs(
                rgb_images=[rgb_frames],
                flow_images=[flow_frames],
                value_images=[value_frames],
                instructions=[lang],
            )
        else:
            backbone_inputs = self.model.backbone_interface.build_cosmos_inputs(
                images=[rgb_frames],
                instructions=[lang],
            )

        # 前向传播 (backbone 冻结, no_grad)
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                backbone_outputs = self.model.backbone_interface(
                    **backbone_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            # Hook 拦截的第 18 层特征 = hidden_states[-1]
            H_video = backbone_outputs.hidden_states[-1]  # (1, 3N, D)

        # 扩展到 batch_size (GRPO 需要 G 份)
        if batch_size > 1 and H_video.shape[0] == 1:
            H_video = H_video.expand(batch_size, -1, -1)  # (G, 3N, D)

        return H_video

    def _prepare_state(
        self, state_value: Any, batch_size: int = 1
    ) -> Optional[torch.Tensor]:
        if state_value is None or getattr(self.action_model, "state_encoder", None) is None:
            return None
        state = np.asarray(state_value, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if self.cfg.normalize_state:
            # Match the training-time StateActionSinCosTransform exactly.
            state = np.concatenate((np.sin(state), np.cos(state)), axis=-1)

        target_dim = int(getattr(self.action_model.config, "state_dim", state.shape[-1]))
        if state.shape[-1] < target_dim:
            state = np.pad(state, ((0, 0), (0, target_dim - state.shape[-1])))
        elif state.shape[-1] > target_dim:
            state = state[:, :target_dim]
        if not self._state_preprocessing_logged:
            print(
                f"[Preprocess] State input: encoded_dim={state.shape[-1]}, "
                f"model_state_dim={target_dim}, sin_then_cos="
                f"{self.cfg.normalize_state}",
                flush=True,
            )
            self._state_preprocessing_logged = True
        tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        return tensor

    def _prepare_env_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(normalized_actions, dtype=np.float32)
        if self.action_norm_stats is None:
            return actions

        # The original DiT4DiT LIBERO checkpoint predicts 8 channels, while
        # the LIBERO action statistics describe the executable 7-D action.
        # Slice before unnormalizing so NumPy broadcasts per-dimension stats
        # against the correct action shape.  The environment performs its own
        # final 7-D validation as well.
        low_key, high_key = (
            ("min", "max")
            if self.action_unnorm_mode == "min_max"
            else ("q01", "q99")
        )
        action_low = np.asarray(self.action_norm_stats[low_key], dtype=np.float32)
        action_high = np.asarray(self.action_norm_stats[high_key], dtype=np.float32)
        stats_dim = int(action_low.shape[-1])
        if actions.shape[-1] < stats_dim:
            raise ValueError(
                f"Sampled action has {actions.shape[-1]} dimensions, but "
                f"normalization statistics require {stats_dim}."
            )
        actions = np.clip(actions[..., :stats_dim].copy(), -1.0, 1.0)
        normalized_snapshot = actions.copy()
        # Only LIBERO's 7/8-D action layout has a binary gripper at index 6.
        # RoboCasa-GR1 has 29 continuous joint/action dimensions; index 6 is
        # the last left-arm joint and must remain continuous.
        if stats_dim >= 7:
            actions[..., 6] = np.where(actions[..., 6] < 0.5, 0.0, 1.0)
        mask = np.asarray(
            self.action_norm_stats.get("mask", np.ones(stats_dim, dtype=bool)),
            dtype=bool,
        )
        env_actions = np.where(
            mask,
            0.5 * (actions + 1.0) * (action_high - action_low) + action_low,
            actions,
        )
        if not self._action_preprocessing_logged:
            print(
                f"[Preprocess] First action chunk: normalized_first="
                f"{normalized_snapshot.reshape(-1, stats_dim)[0].tolist()}; "
                f"env_first={env_actions.reshape(-1, stats_dim)[0].tolist()}; "
                f"mask={mask.tolist()}",
                flush=True,
            )
            self._action_preprocessing_logged = True
        return env_actions

    def _actions_for_execution(self, actions: np.ndarray) -> np.ndarray:
        """Limit a predicted chunk without changing its Flow Matching target."""
        execution_horizon = int(getattr(self.cfg.grpo, "execution_horizon", 0))
        if execution_horizon <= 0:
            return actions
        return actions[:execution_horizon]

    # ========================================================================
    # 阶段 1: Few-shot SFT 冷启动
    # ========================================================================

    def train_sft(self) -> str:
        """阶段 1: Few-shot SFT。

        用 ~40 条专家演示，Flow Matching MSE 损失训练 Action DiT。
        目标：初始成功率 0% → 40%~50%。

        Returns:
            sft_checkpoint_path: SFT checkpoint 保存路径
        """
        if self.sft_dataset is None:
            logger.info("[SFT] Skipped (no SFT dataset provided)")
            return self.cfg.engine.pretrained_checkpoint

        cfg = self.cfg.sft
        logger.info(f"[SFT] ═══ Stage 1: Few-shot SFT ═══")
        logger.info(f"  Demos: {cfg.num_demos}, Epochs: {cfg.num_epochs}, "
                    f"LR: {cfg.learning_rate}, Batch: {cfg.batch_size}")

        # SFT 用单独的优化器 (学习率不同)
        sft_optimizer = torch.optim.AdamW(
            [p for p in self.action_model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=0.0,
        )

        self.action_model.train()
        os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

        for epoch in range(cfg.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for demo_index, batch in enumerate(self.sft_dataset):
                if demo_index >= cfg.num_demos:
                    break
                # batch 包含: rgb_frames, flow_frames, value_frames, lang, action, action_mask, state
                # Step 1: 提取 H_video (backbone 冻结)
                rgb_frames = batch.get("rgb", batch.get("image"))
                if not isinstance(rgb_frames, (list, tuple)):
                    rgb_frames = [rgb_frames]
                H_video = self.extract_h_video(
                    rgb_frames=rgb_frames,
                    flow_frames=batch.get("flow"),
                    value_frames=batch.get("value"),
                    lang=batch["lang"],
                )  # (1, 3N, D)

                # Step 2: 准备 action 数据
                actions = torch.tensor(
                    np.array(batch["action"]),
                    device=self.device,
                    dtype=H_video.dtype,
                )  # (1, T, D_action)
                if actions.dim() == 2:
                    actions = actions.unsqueeze(0)
                action_mask_value = batch.get("action_mask")
                action_mask = (
                    torch.as_tensor(action_mask_value, device=self.device, dtype=torch.float32)
                    if action_mask_value is not None
                    else torch.ones_like(actions)
                )
                if action_mask.dim() == 2:
                    action_mask = action_mask.unsqueeze(0)
                state_value = batch.get("state")
                if state_value is not None and np.asarray(state_value).ndim > 1:
                    state_value = np.asarray(state_value)[0]
                state = self._prepare_state(state_value, batch_size=actions.shape[0])

                # Step 3: SFT loss (Flow Matching MSE)
                loss = compute_sft_loss(
                    action_model=self.action_model,
                    vl_embs=H_video,
                    target_actions=actions,
                    action_mask=action_mask,
                    state=state,
                    N_per_stream=getattr(self.action_model, "N_per_stream", None),
                )

                # Step 4: 反向传播
                sft_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.action_model.parameters(), 1.0
                )
                sft_optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"  [SFT] Epoch {epoch+1}/{cfg.num_epochs} | "
                        f"Loss: {avg_loss:.6f}")

        # 保存 SFT checkpoint
        self.engine.save_checkpoint(cfg.save_path)
        logger.info(f"[SFT] ═══ SFT complete. Checkpoint: {cfg.save_path} ═══")

        # 同步 ref model 到 SFT 后的权重 (作为 GRPO 的参考)
        self.engine.sync_ref_to_current()

        return cfg.save_path

    # ========================================================================
    # 阶段 2: 在线 Action-RL
    # ========================================================================

    def _grpo_noise_level(self) -> float:
        """Linearly anneal rollout exploration over the configured updates."""
        cfg = self.cfg.grpo
        if cfg.noise_level <= 0.0 or cfg.noise_level_end <= 0.0:
            raise ValueError("GRPO noise levels must be positive")
        if cfg.noise_level_end > cfg.noise_level:
            raise ValueError("noise_level_end must be <= noise_level")
        denominator = max(1, cfg.num_iterations - 1)
        progress = min(1.0, max(0.0, float(self.global_step) / denominator))
        return (1.0 - progress) * cfg.noise_level + progress * cfg.noise_level_end

    def rollout_one_trajectory(
        self,
        init_obs: Dict[str, Any],
        H_video: torch.Tensor,
        action_shape: Tuple[int, int],
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, float, bool]:
        """执行一条动作轨迹 (单条, 用于 GRPO 组采样)。

        流程:
          1. sample_hybrid_sde_action → 动作块 (T, D_action)
          2. 逐步执行到环境
          3. 返回动作块、总 reward、是否成功

        Args:
            init_obs: 初始观测
            H_video: (1, 3N, D) Video DiT 特征
            action_shape: (T, D_action)
            state: 机器人状态 (1, D_state)

        Returns:
            actions: (T, D_action) 执行的动作块
            total_reward: float 总奖励
            success: bool 是否成功
        """
        cfg = self.cfg.grpo

        # 1. SDE 采样动作块
        rollout = sample_hybrid_sde_action(
            action_model=self.action_model,
            vl_embs=H_video,
            action_shape=action_shape,
            state=state,
            num_inference_steps=cfg.num_inference_steps,
            noise_level=self._grpo_noise_level(),
            sde_window_size=cfg.sde_window_size,
            sde_window_range=cfg.sde_window_range,
            N_per_stream=getattr(self.action_model, "N_per_stream", None),
            batch_size=1,
            device=self.device,
            dtype=h_video.dtype,
        )
        actions = self._prepare_env_actions(
            rollout.final_action.squeeze(0).cpu().numpy()
        )

        # 2. 逐步执行到环境
        obs = init_obs
        prev_rgb = obs["rgb"]
        total_reward = 0.0
        success = False

        for t in range(action_shape[0]):
            obs, reward, done, info = self.env.step(actions[t])
            total_reward += reward
            if info.get("success", False):
                success = True
            # 实时更新光流/热力图 (不使用旧帧)
            prev_rgb = obs["rgb"]
            if done:
                break

        return rollout, total_reward, success

    def _train_grpo_step_single_chunk_legacy(self) -> Dict[str, float]:
        """单步 GRPO 训练迭代。

        Returns:
            metrics: 训练指标
        """
        step_started_at = time.monotonic()
        cfg = self.cfg.grpo
        G = cfg.group_size
        T = cfg.action_horizon
        D_action = self.action_model.action_dim
        action_shape = (T, D_action)

        # ---- 1. 环境重置 → 初始观测 ----
        logger.info("[GRPO] Resetting environment for next candidate group")
        init_obs = self.env.reset()

        # ---- 2. 三流特征提取 (实时 RAFT + Affordance) ----
        rgb_img, flow_img, value_img = self._observe_to_images(
            init_obs, prev_rgb=None
        )
        lang = init_obs.get("lang", "")

        # 提取 H_video (G 份, 用于组采样)
        H_video = self.extract_h_video(
            rgb_frames=[rgb_img],
            flow_frames=[flow_img],
            value_frames=[value_img],
            lang=lang,
            batch_size=G,
        )  # (G, 3N, D)

        state = self._prepare_state(init_obs.get("state"), batch_size=G)

        # ---- 3. 并行 SDE 组采样 (G 条轨迹) ----
        logger.info("[GRPO] Sampling initial action chunk for %d candidates", G)
        rollout = sample_hybrid_sde_action(
            action_model=self.action_model,
            vl_embs=H_video,
            action_shape=action_shape,
            state=state,
            num_inference_steps=cfg.num_inference_steps,
            noise_level=self._grpo_noise_level(),
            sde_window_size=cfg.sde_window_size,
            sde_window_range=cfg.sde_window_range,
            N_per_stream=getattr(self.action_model, "N_per_stream", None),
            batch_size=G,
            device=self.device,
            dtype=h_video.dtype,
            grad_context=torch.no_grad(),  # rollout 不需要梯度
        )
        # rollout.final_action: (G, T, D_action)

        # ---- 4. 环境执行 G 条轨迹 (串行) → 获取 reward ----
        rewards = torch.zeros(G, device=self.device)
        actions_np = np.stack(
            [self._prepare_env_actions(x) for x in rollout.final_action.cpu().numpy()]
        )

        for g in range(G):
            candidate_started_at = time.monotonic()
            logger.info("[GRPO] Executing candidate %d/%d", g + 1, G)
            print(f"[GRPO] Executing candidate {g + 1}/{G}", flush=True)
            if hasattr(self.env, "reset_to_group_start"):
                obs = self.env.reset_to_group_start()
            else:
                obs = self.env.reset()
            _, total_reward, success = self._execute_episode(
                obs,
                actions_np[g],
                max_steps=cfg.max_episode_steps,
                continuation_seed_base=(
                    self.cfg.seed + (self.global_step + 1) * 1_000_000
                ),
            )
            # 奖励: 成功 → reward_success, 失败 → 0
            rewards[g] = cfg.reward_success if success else (
                cfg.reward_penalty + float(total_reward)
            )
            logger.info(
                "[GRPO] Candidate %d/%d finished: success=%s reward=%.3f time=%.1fs",
                g + 1,
                G,
                success,
                rewards[g].item(),
                time.monotonic() - candidate_started_at,
            )
            print(
                f"[GRPO] Candidate {g + 1}/{G} finished: "
                f"success={success} reward={rewards[g].item():.3f} "
                f"time={time.monotonic() - candidate_started_at:.1f}s",
                flush=True,
            )

        # ---- 5. reference policy KL + GRPO loss ----
        ref_velocities = None
        if hasattr(self.engine, "compute_ref_velocities"):
            ref_velocities = self.engine.compute_ref_velocities(
                rollout_result=rollout,
                vl_embs=H_video,
                state=state,
                N_per_stream=getattr(self.action_model, "N_per_stream", None),
            )
        loss, metrics = compute_grpo_loss(
            action_model=self.action_model,
            rollout_result=rollout,
            vl_embs=H_video,
            rewards=rewards,
            state=state,
            N_per_stream=getattr(self.action_model, "N_per_stream", None),
            ref_velocities=ref_velocities,
            clip_range=cfg.clip_range,
            beta_kl=cfg.beta_kl,
            adv_clip_max=cfg.adv_clip_max,
            global_std=cfg.global_std,
            reward_history=self.reward_history,
        )

        # ---- 6. 反向传播 ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.action_model.parameters(), cfg.max_grad_norm
        )
        self.optimizer.step()

        # ---- 7. 记录奖励历史 ----
        self.reward_history.append(rewards.detach().cpu())
        if len(self.reward_history) > 100:  # 保留最近 100 个 batch
            self.reward_history.pop(0)

        self.global_step += 1
        metrics["global_step"] = self.global_step
        metrics["success_rate"] = (rewards > 0).float().mean().item()
        metrics["step_seconds"] = time.monotonic() - step_started_at
        if hasattr(self.env, "task_id"):
            metrics["task_id"] = int(self.env.task_id)

        return metrics

    @staticmethod
    def _slice_rollout_batch(rollout: RolloutResult, index: int) -> RolloutResult:
        """Detach one candidate from a batched rollout while preserving batch dim."""
        batch_size = rollout.final_action.shape[0]

        def select(value):
            if not isinstance(value, torch.Tensor):
                return value
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value[index:index + 1].detach()
            return value.detach()

        records = [
            type(record)(**{name: select(value) for name, value in vars(record).items()})
            for record in rollout.sde_records
        ]
        return RolloutResult(
            final_action=select(rollout.final_action),
            sde_records=records,
            all_velocities=[select(value) for value in rollout.all_velocities],
            timesteps_schedule=rollout.timesteps_schedule.detach(),
            sde_step_indices=list(rollout.sde_step_indices),
        )

    def _execute_training_episode(
        self,
        initial_obs: Dict[str, Any],
        initial_chunk: GRPOChunkSample,
        max_steps: int,
        continuation_seed_base: int,
        watchdog_label: str,
    ) -> Tuple[Dict[str, Any], float, bool, List[GRPOChunkSample]]:
        """Execute an episode and retain temporally distributed decisions."""
        cfg = self.cfg.grpo
        obs = initial_obs
        previous_condition_rgb = np.asarray(obs["rgb"])
        pending_actions = self._prepare_env_actions(
            initial_chunk.rollout.final_action[0].cpu().numpy()
        )
        chunks = [initial_chunk]
        total_reward = 0.0
        success = False
        steps = 0
        continuation_index = 0
        decision_index = 0
        max_decision_index = max(
            0, math.ceil(max_steps / cfg.action_horizon) - 1
        )
        if cfg.collect_all_chunks:
            training_chunk_indices = set(range(1, max_decision_index + 1))
        else:
            training_chunk_indices = set(
                build_training_chunk_indices(
                    cfg.chunks_per_episode,
                    cfg.chunk_stride,
                    max_decision_index=(
                        max_decision_index if cfg.spread_chunks_across_episode else None
                    ),
                )
            )

        while steps < max_steps:
            for action in self._actions_for_execution(pending_actions):
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: env.step {steps + 1}/{max_steps}"
                )
                obs, reward, done, info = self.env.step(action)
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: completed env.step {steps + 1}/{max_steps}"
                )
                total_reward += float(reward)
                steps += 1
                success = success or bool(info.get("success", False))
                if done or success or steps >= max_steps:
                    return obs, total_reward, success, chunks

            if cfg.progress_steps > 0 and steps % cfg.progress_steps == 0:
                print(
                    f"[GRPO] Episode progress: {steps}/{max_steps} env steps; "
                    f"training_chunks={len(chunks)}/"
                    f"{'all' if cfg.collect_all_chunks else cfg.chunks_per_episode}",
                    flush=True,
                )

            current_rgb = np.asarray(obs["rgb"])
            decision_index += 1
            if decision_index in training_chunk_indices:
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: sample training chunk {decision_index}"
                )
                rgb, flow, value = self._observe_to_images(
                    obs, prev_rgb=previous_condition_rgb
                )
                h_video = self.extract_h_video(
                    rgb_frames=[rgb],
                    flow_frames=[flow] if flow is not None else None,
                    value_frames=[value] if value is not None else None,
                    lang=obs.get("lang", ""),
                    batch_size=1,
                )
                state = self._prepare_state(obs.get("state"), batch_size=1)
                rollout = sample_hybrid_sde_action(
                    action_model=self.action_model,
                    vl_embs=h_video,
                    action_shape=(cfg.action_horizon, self.action_model.action_dim),
                    state=state,
                    num_inference_steps=cfg.num_inference_steps,
                    noise_level=self._grpo_noise_level(),
                    sde_window_size=cfg.sde_window_size,
                    sde_window_range=cfg.sde_window_range,
                    N_per_stream=getattr(self.action_model, "N_per_stream", None),
                    batch_size=1,
                    device=self.device,
                    dtype=torch.float32,
                    grad_context=torch.no_grad(),
                )
                chunks.append(
                    GRPOChunkSample(
                        rollout=rollout,
                        vl_embs=h_video.detach(),
                        state=state.detach() if state is not None else None,
                        chunk_index=decision_index,
                    )
                )
                pending_actions = self._prepare_env_actions(
                    rollout.final_action[0].cpu().numpy()
                )
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: sampled training chunk {decision_index}"
                )
            else:
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: sample continuation {continuation_index}"
                )
                pending_actions = self._sample_ode_chunk_with_seed(
                    obs,
                    prev_rgb=previous_condition_rgb,
                    seed=continuation_seed_base + continuation_index,
                )
                continuation_index += 1
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: sampled continuation {continuation_index}"
                )
            previous_condition_rgb = current_rgb

        return obs, total_reward, success, chunks

    def _add_successful_chunks_to_rsft_replay(
        self, chunks: List[GRPOChunkSample]
    ) -> None:
        """Keep successful on-policy chunks as detached Flow Matching targets."""
        cfg = self.cfg.grpo
        if cfg.rsft_anchor_weight <= 0.0:
            return
        for chunk in chunks:
            self.rsft_replay.append(
                RSFTReplaySample(
                    vl_embs=chunk.vl_embs.detach().cpu(),
                    target_actions=chunk.rollout.final_action.detach().cpu(),
                    action_mask=torch.ones_like(chunk.rollout.final_action).cpu(),
                    state=(chunk.state.detach().cpu() if chunk.state is not None else None),
                )
            )
        if len(self.rsft_replay) > cfg.rsft_replay_capacity:
            del self.rsft_replay[:-cfg.rsft_replay_capacity]

    def _next_rsft_anchor_sample(self) -> Optional[RSFTReplaySample]:
        """Read one original SFT item and convert it to the online replay format."""
        if self._rsft_anchor_iterator is None:
            return None
        try:
            batch = next(self._rsft_anchor_iterator)
        except StopIteration:
            self._rsft_anchor_iterator = iter(self.rsft_anchor_dataset)
            batch = next(self._rsft_anchor_iterator)

        return self._batch_to_rsft_sample(batch)

    def _batch_to_rsft_sample(self, batch: Dict[str, Any]) -> RSFTReplaySample:
        """Encode one repository dataset item as a Flow Matching target."""

        rgb_frames = batch.get("rgb", batch.get("image"))
        if not isinstance(rgb_frames, (list, tuple)):
            rgb_frames = [rgb_frames]
        h_video = self.extract_h_video(
            rgb_frames=rgb_frames,
            flow_frames=batch.get("flow"),
            value_frames=batch.get("value"),
            lang=batch.get("lang", ""),
        )
        target_actions = torch.as_tensor(
            np.asarray(batch["action"]), device=self.device, dtype=h_video.dtype
        )
        if target_actions.ndim == 2:
            target_actions = target_actions.unsqueeze(0)
        action_mask_value = batch.get("action_mask")
        action_mask = (
            torch.as_tensor(action_mask_value, device=self.device, dtype=h_video.dtype)
            if action_mask_value is not None
            else torch.ones_like(target_actions)
        )
        if action_mask.ndim == 2:
            action_mask = action_mask.unsqueeze(0)
        state_value = batch.get("state")
        if state_value is not None and np.asarray(state_value).ndim > 1:
            state_value = np.asarray(state_value)[0]
        state = self._prepare_state(state_value, batch_size=target_actions.shape[0])
        return RSFTReplaySample(
            vl_embs=h_video.detach(),
            target_actions=target_actions.detach(),
            action_mask=action_mask.detach(),
            state=state.detach() if state is not None else None,
        )

    def _compute_rsft_anchor_loss(self) -> Tuple[Optional[torch.Tensor], int, int]:
        """Compute mixed 70/30 original-SFT and successful-rollout Flow Matching loss."""
        cfg = self.cfg.grpo
        if cfg.rsft_anchor_weight <= 0.0:
            return None, 0, 0
        batch_size = max(1, cfg.rsft_anchor_batch_size)
        online_target = (
            int(round(batch_size * cfg.rsft_online_fraction))
            if self._rsft_anchor_iterator is not None
            else batch_size
        )
        online_count = min(len(self.rsft_replay), online_target)
        offline_count = batch_size - online_count if self._rsft_anchor_iterator else 0
        if online_count == 0 and offline_count == 0:
            return None, 0, 0

        samples: List[RSFTReplaySample] = []
        if online_count:
            indices = np.random.choice(len(self.rsft_replay), online_count, replace=False)
            samples.extend(self.rsft_replay[int(index)] for index in indices)
        for _ in range(offline_count):
            sample = self._next_rsft_anchor_sample()
            if sample is not None:
                samples.append(sample)

        compatible_groups: Dict[Tuple[Any, ...], List[RSFTReplaySample]] = {}
        for sample in samples:
            key = (
                sample.state is not None,
                tuple(sample.vl_embs.shape[1:]),
                tuple(sample.target_actions.shape[1:]),
                tuple(sample.state.shape[1:]) if sample.state is not None else (),
            )
            compatible_groups.setdefault(key, []).append(sample)

        weighted_losses = []
        for key, group in compatible_groups.items():
            has_state = bool(key[0])
            vl_embs = torch.cat([sample.vl_embs.to(self.device) for sample in group])
            target_actions = torch.cat(
                [sample.target_actions.to(self.device) for sample in group]
            )
            action_mask = torch.cat(
                [sample.action_mask.to(self.device) for sample in group]
            )
            state = (
                torch.cat([sample.state.to(self.device) for sample in group])
                if has_state else None
            )
            group_loss = compute_sft_loss(
                action_model=self.action_model,
                vl_embs=vl_embs,
                target_actions=target_actions,
                action_mask=action_mask,
                state=state,
                N_per_stream=getattr(self.action_model, "N_per_stream", None),
            )
            weighted_losses.append(group_loss * len(group))
        loss = torch.stack(weighted_losses).sum() / len(samples)
        return loss, online_count, len(samples) - online_count

    def train_grpo_step(self) -> Dict[str, float]:
        """Collect multi-chunk trajectories and run trajectory-balanced GRPO."""
        step_started_at = time.monotonic()
        cfg = self.cfg.grpo
        if cfg.chunks_per_episode < 1:
            raise ValueError("chunks_per_episode must be >= 1")
        if cfg.chunk_stride < 1:
            raise ValueError("chunk_stride must be >= 1")
        if cfg.inner_epochs < 1:
            raise ValueError("inner_epochs must be >= 1")
        if not 0.0 < cfg.temporal_credit_decay <= 1.0:
            raise ValueError("temporal_credit_decay must be in (0, 1]")
        if not 0.0 <= cfg.rsft_online_fraction <= 1.0:
            raise ValueError("rsft_online_fraction must be in [0, 1]")
        if cfg.rsft_anchor_weight < 0.0:
            raise ValueError("rsft_anchor_weight must be >= 0")
        if cfg.rsft_anchor_batch_size < 1 or cfg.rsft_replay_capacity < 1:
            raise ValueError("RSFT batch size and replay capacity must be >= 1")

        group_size = cfg.group_size
        action_shape = (cfg.action_horizon, self.action_model.action_dim)
        self.rollout_watchdog.arm(
            f"update {self.global_step + 1}: reset and sample rollout group"
        )
        init_obs = self.env.reset()
        self.rollout_watchdog.beat(
            f"update {self.global_step + 1}: encode rollout group observation"
        )
        rgb, flow, value = self._observe_to_images(init_obs, prev_rgb=None)
        h_video = self.extract_h_video(
            rgb_frames=[rgb],
            flow_frames=[flow],
            value_frames=[value],
            lang=init_obs.get("lang", ""),
            batch_size=group_size,
        )
        state = self._prepare_state(init_obs.get("state"), batch_size=group_size)
        initial_rollout = sample_hybrid_sde_action(
            action_model=self.action_model,
            vl_embs=h_video,
            action_shape=action_shape,
            state=state,
            num_inference_steps=cfg.num_inference_steps,
            noise_level=self._grpo_noise_level(),
            sde_window_size=cfg.sde_window_size,
            sde_window_range=cfg.sde_window_range,
            N_per_stream=getattr(self.action_model, "N_per_stream", None),
            batch_size=group_size,
            device=self.device,
            dtype=torch.float32,
            grad_context=torch.no_grad(),
        )
        self.rollout_watchdog.disarm()

        rewards = torch.zeros(group_size, device=self.device)
        trajectories: List[List[GRPOChunkSample]] = []
        for candidate_index in range(group_size):
            candidate_started_at = time.monotonic()
            candidate_label = (
                f"update {self.global_step + 1} candidate "
                f"{candidate_index + 1}/{group_size}"
            )
            self.rollout_watchdog.arm(f"{candidate_label}: reset")
            print(
                f"[GRPO] Executing candidate {candidate_index + 1}/{group_size}",
                flush=True,
            )
            if hasattr(self.env, "reset_to_group_start"):
                obs = self.env.reset_to_group_start()
            else:
                obs = self.env.reset()
            self.rollout_watchdog.beat(f"{candidate_label}: execute episode")
            initial_chunk = GRPOChunkSample(
                rollout=self._slice_rollout_batch(initial_rollout, candidate_index),
                vl_embs=h_video[candidate_index:candidate_index + 1].detach(),
                state=(
                    state[candidate_index:candidate_index + 1].detach()
                    if state is not None else None
                ),
                chunk_index=0,
            )
            _, total_reward, success, chunks = self._execute_training_episode(
                obs,
                initial_chunk,
                max_steps=cfg.max_episode_steps,
                continuation_seed_base=(
                    self.cfg.seed
                    + (self.global_step + 1) * 1_000_000
                    + (
                        0
                        if cfg.common_continuation_seed
                        else candidate_index * 10_000
                    )
                ),
                watchdog_label=candidate_label,
            )
            trajectories.append(chunks)
            if success:
                self._add_successful_chunks_to_rsft_replay(chunks)
            rewards[candidate_index] = (
                cfg.reward_success
                if success
                else cfg.reward_penalty + float(total_reward)
            )
            print(
                f"[GRPO] Candidate {candidate_index + 1}/{group_size} finished: "
                f"success={success} reward={rewards[candidate_index].item():.3f} "
                f"training_chunks={len(chunks)} "
                f"chunk_indices={[chunk.chunk_index for chunk in chunks]} "
                f"time={time.monotonic() - candidate_started_at:.1f}s",
                flush=True,
            )
            self.rollout_watchdog.disarm()

        reward_std = rewards.std(unbiased=False).item()
        if reward_std <= 1e-8:
            # A group with identical rewards contains no relative GRPO signal.
            self.optimizer.zero_grad(set_to_none=True)
            rsft_loss, rsft_online_count, rsft_offline_count = (
                self._compute_rsft_anchor_loss()
            )
            grad_norm = 0.0
            update_norm = 0.0
            if rsft_loss is not None:
                tracked_parameters = [
                    (name, parameter)
                    for name, parameter in self.action_model.named_parameters()
                    if parameter.requires_grad
                ]
                (cfg.rsft_anchor_weight * rsft_loss).backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.action_model.parameters(), cfg.max_grad_norm
                    )
                )
                before_update = {
                    name: parameter.detach().clone()
                    for name, parameter in tracked_parameters
                }
                self.optimizer.step()
                update_norm = math.sqrt(
                    sum(
                        (
                            parameter.detach().float()
                            - before_update[name].float()
                        ).square().sum().item()
                        for name, parameter in tracked_parameters
                    )
                )
            self.reward_history.append(rewards.detach().cpu())
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            self.global_step += 1
            chunk_counts = [len(chunks) for chunks in trajectories]
            metrics = {
                "loss": (
                    cfg.rsft_anchor_weight * rsft_loss.item()
                    if rsft_loss is not None else 0.0
                ),
                "policy_loss": 0.0,
                "kl_loss": 0.0,
                "ratio_mean": 1.0,
                "ratio_sq_mean": 1.0,
                "ratio_std": 0.0,
                "ratio_min": 1.0,
                "ratio_max": 1.0,
                "clip_fraction": 0.0,
                "log_ratio_mean": 0.0,
                "log_ratio_sq_mean": 0.0,
                "log_ratio_std": 0.0,
                "log_ratio_abs_max": 0.0,
                "grad_norm": grad_norm,
                "trainable_update_norm": update_norm,
                "lora_update_norm": update_norm,
                "rsft_anchor_loss": rsft_loss.item() if rsft_loss is not None else 0.0,
                "rsft_online_samples": float(rsft_online_count),
                "rsft_offline_samples": float(rsft_offline_count),
                "rsft_replay_size": float(len(self.rsft_replay)),
                "inner_epoch": 0.0,
                "optimizer_step_skipped": float(rsft_loss is None),
                "global_step": self.global_step,
                "success_rate": (rewards > 0).float().mean().item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": reward_std,
                "valid_reward_group": 0.0,
                "training_chunks_mean": float(np.mean(chunk_counts)),
                "training_chunks_min": float(min(chunk_counts)),
                "training_chunks_max": float(max(chunk_counts)),
                "step_seconds": time.monotonic() - step_started_at,
            }
            if hasattr(self.env, "task_id"):
                metrics["task_id"] = int(self.env.task_id)
            if rsft_loss is None:
                message = "Skipping optimizer.step"
            else:
                message = "Applied RSFT-only optimizer.step"
            print(
                f"[GRPO] {message}; this group has no relative GRPO signal.",
                flush=True,
            )
            return metrics

        advantages = compute_group_advantages(
            rewards,
            eps=1e-8,
            global_std=cfg.global_std,
            reward_history=self.reward_history,
        ).clamp(-cfg.adv_clip_max, cfg.adv_clip_max)

        for chunks in trajectories:
            for chunk in chunks:
                if hasattr(self.engine, "compute_ref_velocities"):
                    chunk.ref_velocities = self.engine.compute_ref_velocities(
                        rollout_result=chunk.rollout,
                        vl_embs=chunk.vl_embs,
                        state=chunk.state,
                        N_per_stream=getattr(self.action_model, "N_per_stream", None),
                    )

        epoch_metrics: Dict[str, float] = {}
        tracked_parameters = [
            (name, parameter)
            for name, parameter in self.action_model.named_parameters()
            if parameter.requires_grad
        ]
        for inner_epoch in range(cfg.inner_epochs):
            self.optimizer.zero_grad()
            accumulated: Dict[str, float] = {}
            extrema: Dict[str, float] = {}
            rsft_loss, rsft_online_count, rsft_offline_count = (
                self._compute_rsft_anchor_loss()
            )
            if rsft_loss is not None:
                (cfg.rsft_anchor_weight * rsft_loss).backward()
            for candidate_index, chunks in enumerate(trajectories):
                chunk_weights = dict(
                    zip(
                        (chunk.chunk_index for chunk in chunks),
                        build_temporal_chunk_weights(
                            [chunk.chunk_index for chunk in chunks],
                            cfg.temporal_credit_decay,
                        ),
                    )
                )
                for chunk in chunks:
                    trajectory_weight = chunk_weights[chunk.chunk_index] / group_size
                    loss, chunk_metrics = compute_grpo_loss(
                        action_model=self.action_model,
                        rollout_result=chunk.rollout,
                        vl_embs=chunk.vl_embs,
                        rewards=rewards[candidate_index:candidate_index + 1],
                        state=chunk.state,
                        N_per_stream=getattr(self.action_model, "N_per_stream", None),
                        ref_velocities=chunk.ref_velocities,
                        clip_range=cfg.clip_range,
                        beta_kl=cfg.beta_kl,
                        adv_clip_max=cfg.adv_clip_max,
                        precomputed_advantages=(
                            advantages[candidate_index:candidate_index + 1]
                        ),
                    )
                    (loss * trajectory_weight).backward()
                    for key, value in chunk_metrics.items():
                        numeric_value = float(value)
                        if key == "ratio_min":
                            extrema[key] = min(
                                extrema.get(key, numeric_value), numeric_value
                            )
                        elif key in {"ratio_max", "log_ratio_abs_max"}:
                            extrema[key] = max(
                                extrema.get(key, numeric_value), numeric_value
                            )
                        else:
                            accumulated[key] = (
                                accumulated.get(key, 0.0)
                                + numeric_value * trajectory_weight
                            )

            accumulated.update(extrema)
            accumulated["ratio_std"] = math.sqrt(
                max(
                    accumulated["ratio_sq_mean"]
                    - accumulated["ratio_mean"] ** 2,
                    0.0,
                )
            )
            accumulated["log_ratio_std"] = math.sqrt(
                max(
                    accumulated["log_ratio_sq_mean"]
                    - accumulated["log_ratio_mean"] ** 2,
                    0.0,
                )
            )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.action_model.parameters(), cfg.max_grad_norm
            )
            before_update = {
                name: parameter.detach().clone()
                for name, parameter in tracked_parameters
            }
            self.optimizer.step()
            update_sq = 0.0
            for name, parameter in tracked_parameters:
                update_sq += (
                    parameter.detach().float() - before_update[name].float()
                ).square().sum().item()

            epoch_metrics = accumulated
            if rsft_loss is not None:
                epoch_metrics["loss"] += cfg.rsft_anchor_weight * rsft_loss.item()
                epoch_metrics["rsft_anchor_loss"] = rsft_loss.item()
            else:
                epoch_metrics["rsft_anchor_loss"] = 0.0
            epoch_metrics["rsft_online_samples"] = float(rsft_online_count)
            epoch_metrics["rsft_offline_samples"] = float(rsft_offline_count)
            epoch_metrics["rsft_replay_size"] = float(len(self.rsft_replay))
            epoch_metrics["grad_norm"] = float(grad_norm)
            epoch_metrics["trainable_update_norm"] = math.sqrt(update_sq)
            # Compatibility for existing metric consumers. New logs use the
            # scope-agnostic name above, including for full Action DiT tuning.
            epoch_metrics["lora_update_norm"] = epoch_metrics["trainable_update_norm"]
            epoch_metrics["inner_epoch"] = float(inner_epoch + 1)
            epoch_metrics["optimizer_step_skipped"] = 0.0
            print(
                f"[GRPO][Epoch {inner_epoch + 1}/{cfg.inner_epochs}] "
                f"loss={epoch_metrics['loss']:.12e} "
                f"policy={epoch_metrics['policy_loss']:.12e} "
                f"kl={epoch_metrics['kl_loss']:.12e} "
                f"ratio={epoch_metrics['ratio_mean']:.12f}+/-"
                f"{epoch_metrics['ratio_std']:.12e} "
                f"log_ratio={epoch_metrics['log_ratio_mean']:.12e}+/-"
                f"{epoch_metrics['log_ratio_std']:.12e} "
                f"log_ratio_abs_max={epoch_metrics['log_ratio_abs_max']:.12e} "
                f"clip_fraction={epoch_metrics['clip_fraction']:.2%} "
                f"rsft={epoch_metrics['rsft_anchor_loss']:.6f} "
                f"rsft_mix={int(epoch_metrics['rsft_offline_samples'])}/"
                f"{int(epoch_metrics['rsft_online_samples'])} "
                f"grad_norm={epoch_metrics['grad_norm']:.6f} "
                f"trainable_update_norm={epoch_metrics['trainable_update_norm']:.6e}",
                flush=True,
            )

        self.reward_history.append(rewards.detach().cpu())
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        self.global_step += 1
        chunk_counts = [len(chunks) for chunks in trajectories]
        epoch_metrics.update(
            {
                "global_step": self.global_step,
                "success_rate": (rewards > 0).float().mean().item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std(unbiased=False).item(),
                "valid_reward_group": float(
                    rewards.std(unbiased=False).item() > 1e-8
                ),
                "training_chunks_mean": float(np.mean(chunk_counts)),
                "training_chunks_min": float(min(chunk_counts)),
                "training_chunks_max": float(max(chunk_counts)),
                "step_seconds": time.monotonic() - step_started_at,
            }
        )
        if hasattr(self.env, "task_id"):
            epoch_metrics["task_id"] = int(self.env.task_id)
        return epoch_metrics

    @torch.no_grad()
    def _sample_ode_chunk(
        self, obs: Dict[str, Any], prev_rgb: Optional[np.ndarray]
    ) -> np.ndarray:
        cfg = self.cfg.grpo
        rgb, flow, value = self._observe_to_images(obs, prev_rgb=prev_rgb)
        if self.model_variant == "original":
            state = self._prepare_state(obs.get("state"), batch_size=1)
            example = {
                "image": [rgb],
                "lang": obs.get("lang", ""),
            }
            if state is not None:
                example["state"] = state.detach().cpu().numpy()
            response = self.model.predict_action(examples=[example])
            normalized_actions = np.asarray(
                response["normalized_actions"][0], dtype=np.float32
            )
            return self._prepare_env_actions(normalized_actions)

        h_video = self.extract_h_video(
            rgb_frames=[rgb],
            flow_frames=[flow] if flow is not None else None,
            value_frames=[value] if value is not None else None,
            lang=obs.get("lang", ""),
            batch_size=1,
        )
        state = self._prepare_state(obs.get("state"), batch_size=1)
        rollout = sample_hybrid_sde_action(
            action_model=self.action_model,
            vl_embs=h_video,
            action_shape=(cfg.action_horizon, self.action_model.action_dim),
            state=state,
            num_inference_steps=cfg.num_inference_steps,
            noise_level=0.0,
            sde_window_size=0,
            N_per_stream=getattr(self.action_model, "N_per_stream", None),
            batch_size=1,
            device=self.device,
            dtype=torch.float32,
        )
        return self._prepare_env_actions(rollout.final_action[0].cpu().numpy())

    def _sample_ode_chunk_with_seed(
        self,
        obs: Dict[str, Any],
        prev_rgb: Optional[np.ndarray],
        seed: int,
    ) -> np.ndarray:
        """Sample a continuation chunk without advancing the training RNG."""
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return self._sample_ode_chunk(obs, prev_rgb)
        finally:
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _execute_episode(
        self,
        initial_obs: Dict[str, Any],
        initial_actions: np.ndarray,
        max_steps: int,
        continuation_seed_base: Optional[int] = None,
        watchdog_label: str = "evaluation rollout",
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Execute one exploratory chunk, then ODE continuation from sampled noise."""
        obs = initial_obs
        previous_condition_rgb = np.asarray(obs["rgb"])
        pending_actions = initial_actions
        total_reward = 0.0
        success = False
        steps = 0
        continuation_index = 0
        progress_steps = self.cfg.grpo.progress_steps

        while steps < max_steps:
            for action in self._actions_for_execution(pending_actions):
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: env.step {steps + 1}/{max_steps}"
                )
                obs, reward, done, info = self.env.step(action)
                self.rollout_watchdog.beat(
                    f"{watchdog_label}: completed env.step {steps + 1}/{max_steps}"
                )
                total_reward += float(reward)
                steps += 1
                success = success or bool(info.get("success", False))
                if done or success or steps >= max_steps:
                    return obs, total_reward, success

            if progress_steps > 0 and steps % progress_steps == 0:
                print(
                    f"[GRPO] Episode progress: {steps}/{max_steps} env steps",
                    flush=True,
                )

            current_rgb = np.asarray(obs["rgb"])
            self.rollout_watchdog.beat(
                f"{watchdog_label}: sample continuation {continuation_index}"
            )
            if continuation_seed_base is None:
                pending_actions = self._sample_ode_chunk(
                    obs, prev_rgb=previous_condition_rgb
                )
            else:
                pending_actions = self._sample_ode_chunk_with_seed(
                    obs,
                    prev_rgb=previous_condition_rgb,
                    seed=continuation_seed_base + continuation_index,
                )
            continuation_index += 1
            self.rollout_watchdog.beat(
                f"{watchdog_label}: sampled continuation {continuation_index}"
            )
            previous_condition_rgb = current_rgb

        return obs, total_reward, success

    def train_grpo(self):
        """阶段 2: 在线 Action-RL 演化。"""
        cfg = self.cfg.grpo
        logger.info(f"[GRPO] ═══ Stage 2: Online Action-RL ═══")
        logger.info(f"  Iterations: {cfg.num_iterations}, Group: {cfg.group_size}, "
                    f"K: {cfg.num_inference_steps}, β: {cfg.beta_kl}")

        self.action_model.train()
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        for iteration in range(self.global_step, cfg.num_iterations):
            logger.info(
                "[GRPO] Starting update %d/%d", iteration + 1, cfg.num_iterations
            )
            print(
                f"[GRPO] Starting update {iteration + 1}/{cfg.num_iterations}",
                flush=True,
            )
            metrics = self.train_grpo_step()

            # 日志
            remaining_updates = cfg.num_iterations - metrics["global_step"]
            eta_hours = remaining_updates * metrics["step_seconds"] / 3600.0
            if iteration % cfg.log_freq == 0:
                logger.info(
                    f"  [GRPO] Step {metrics['global_step']} | "
                    f"Loss: {metrics['loss']:.6f} | "
                    f"Policy: {metrics['policy_loss']:.6f} | "
                    f"KL: {metrics['kl_loss']:.6f} | "
                    f"Success: {metrics['success_rate']:.2%} | "
                    f"Reward: {metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f}"
                )

            # 保存
            logger.info(
                "[GRPO] Update timing: %.1fs; estimated remaining time: %.1fh",
                metrics["step_seconds"],
                eta_hours,
            )
            print(
                f"[GRPO] Update {metrics['global_step']} timing: "
                f"{metrics['step_seconds']:.1f}s; "
                f"task={metrics.get('task_id', 'n/a')}; "
                f"loss={metrics['loss']:.6f}; "
                f"kl={metrics['kl_loss']:.6f}; "
                f"success={metrics['success_rate']:.2%}; "
                f"reward={metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f}; "
                f"ETA={eta_hours:.1f}h",
                flush=True,
            )
            print(
            "[GRPO] Diagnostics: "
                f"ratio_mean={metrics.get('ratio_mean', float('nan')):.12f}; "
                f"ratio_std={metrics.get('ratio_std', float('nan')):.12e}; "
                f"log_ratio_mean={metrics.get('log_ratio_mean', float('nan')):.12e}; "
                f"log_ratio_std={metrics.get('log_ratio_std', float('nan')):.12e}; "
                f"log_ratio_abs_max={metrics.get('log_ratio_abs_max', float('nan')):.12e}; "
                f"clip_fraction={metrics.get('clip_fraction', float('nan')):.2%}; "
                f"grad_norm={metrics.get('grad_norm', float('nan')):.6f}; "
                f"trainable_update_norm={metrics.get('trainable_update_norm', 0.0):.6e}; "
                f"rsft_anchor_loss={metrics.get('rsft_anchor_loss', 0.0):.6f}; "
                f"rsft_replay_size={metrics.get('rsft_replay_size', len(self.rsft_replay)):.0f}; "
                f"valid_reward_group={int(metrics.get('valid_reward_group', 0))}; "
                f"training_chunks={metrics.get('training_chunks_mean', 0.0):.2f} "
                f"[{metrics.get('training_chunks_min', 0.0):.0f},"
                f"{metrics.get('training_chunks_max', 0.0):.0f}]",
                flush=True,
            )
            task_key = str(metrics.get("task_id", "single"))
            if metrics["reward_std"] <= 1e-8:
                task_no_signal = self.no_signal_updates_by_task.get(task_key, 0) + 1
                self.no_signal_updates_by_task[task_key] = task_no_signal
                self.consecutive_no_signal_updates = task_no_signal
                print(
                    "[GRPO][WARN] reward_std=0: this group has no relative "
                    "GRPO signal; RSFT-only anchoring may still run. "
                    f"task={task_key} count={task_no_signal}/"
                    f"{cfg.max_no_signal_updates}",
                    flush=True,
                )
            else:
                task_no_signal = 0
                self.no_signal_updates_by_task[task_key] = 0
                self.consecutive_no_signal_updates = 0

            if (
                cfg.max_no_signal_updates > 0
                and task_no_signal >= cfg.max_no_signal_updates
            ):
                if metrics["reward_mean"] <= 0.0:
                    advice = (
                        "All candidates failed. Verify official SFT evaluation "
                        "and, if the preflight baseline succeeds, lower NOISE_LEVEL."
                    )
                else:
                    advice = (
                        "All candidates received the same positive reward; increase "
                        "NOISE_LEVEL or GROUP_SIZE to obtain relative variation."
                    )
                raise RuntimeError(
                    f"GRPO stopped after task {task_key} produced {task_no_signal} "
                    f"groups without relative reward signal. {advice}"
                )

            should_save = metrics["global_step"] == 1 or (
                cfg.save_freq > 0
                and metrics["global_step"] % cfg.save_freq == 0
            )
            if should_save:
                ckpt_path = os.path.join(
                    self.cfg.output_dir,
                    "checkpoints",
                    f"grpo_step_{metrics['global_step']}_resume.pt",
                )
                self.engine.save_checkpoint(
                    ckpt_path, trainer_state=self._checkpoint_trainer_state()
                )
                logger.info(f"  [GRPO] Checkpoint saved: {ckpt_path}")

            # 评估
            if cfg.eval_freq > 0 and metrics["global_step"] % cfg.eval_freq == 0:
                eval_metrics = self.evaluate(
                    num_episodes=cfg.eval_episodes,
                    label=f"grpo_step_{metrics['global_step']}",
                )
                logger.info(
                    f"  [GRPO] Eval | Success: {eval_metrics['success_rate']:.2%}"
                )
                current_score = (
                    eval_metrics["successes"] / eval_metrics["episodes"]
                )
                paired_delta = None
                if self.preflight_records:
                    candidate_records = {
                        (
                            int(record["task_id"]),
                            int(record["init_state_id"]),
                            int(record["episode_seed"]),
                        ): bool(record["success"])
                        for record in eval_metrics["records"]
                    }
                    paired_keys = set(candidate_records) & set(self.preflight_records)
                    if not paired_keys:
                        raise RuntimeError(
                            "Periodic evaluation has no episode keys paired with preflight"
                        )
                    baseline_successes = sum(
                        int(self.preflight_records[key]) for key in paired_keys
                    )
                    candidate_successes = sum(
                        int(candidate_records[key]) for key in paired_keys
                    )
                    paired_delta = (
                        candidate_successes - baseline_successes
                    ) / len(paired_keys)
                    print(
                        "[GRPO] Paired checkpoint selection: "
                        f"baseline={baseline_successes}/{len(paired_keys)}; "
                        f"candidate={candidate_successes}/{len(paired_keys)}; "
                        f"delta={paired_delta:+.2%}",
                        flush=True,
                    )

                best_score = (
                    self.best_eval_successes / self.best_eval_episodes
                    if self.best_eval_episodes > 0 else -1.0
                )
                is_better = (
                    paired_delta > self.best_eval_paired_delta
                    if paired_delta is not None
                    else current_score > best_score
                )
                if is_better:
                    self.best_eval_successes = int(eval_metrics["successes"])
                    self.best_eval_episodes = int(eval_metrics["episodes"])
                    self.best_eval_step = int(metrics["global_step"])
                    if paired_delta is not None:
                        self.best_eval_paired_delta = paired_delta
                    best_path = os.path.join(
                        self.cfg.output_dir, "checkpoints", "grpo_best_resume.pt"
                    )
                    self.engine.save_checkpoint(
                        best_path, trainer_state=self._checkpoint_trainer_state()
                    )
                    logger.info(
                        "  [GRPO] New best checkpoint at step %d: %.2f%%",
                        self.best_eval_step,
                        100.0 * current_score,
                    )

                if (
                    cfg.stop_on_negative_paired_delta
                    and paired_delta is not None
                    and paired_delta < 0.0
                ):
                    rejected_path = os.path.join(
                        self.cfg.output_dir,
                        "checkpoints",
                        f"grpo_rejected_step_{metrics['global_step']}_resume.pt",
                    )
                    self.engine.save_checkpoint(
                        rejected_path,
                        trainer_state=self._checkpoint_trainer_state(),
                    )
                    print(
                        "[GRPO] Early stop: paired evaluation is below the "
                        f"preflight baseline (delta={paired_delta:+.2%}). "
                        f"Rejected checkpoint saved to {rejected_path}",
                        flush=True,
                    )
                    return

        # 最终保存
        checkpoint_dir = os.path.join(self.cfg.output_dir, "checkpoints")
        resume_path = os.path.join(checkpoint_dir, "grpo_final_resume.pt")
        self.engine.save_checkpoint(
            resume_path, trainer_state=self._checkpoint_trainer_state()
        )
        final_path = os.path.join(checkpoint_dir, "grpo_final_pytorch_model.pt")
        if hasattr(self.engine, "export_pretrained_checkpoint"):
            self.engine.export_pretrained_checkpoint(final_path)
        best_resume_path = os.path.join(checkpoint_dir, "grpo_best_resume.pt")
        if self.best_eval_step > 0 and os.path.exists(best_resume_path):
            self.engine.load_action_checkpoint(best_resume_path, load_optimizer=False)
            best_path = os.path.join(checkpoint_dir, "grpo_best_pytorch_model.pt")
            self.engine.export_pretrained_checkpoint(best_path)
            self.engine.load_action_checkpoint(resume_path, load_optimizer=False)
            logger.info(
                "[GRPO] Best evaluated checkpoint exported: step=%d path=%s",
                self.best_eval_step,
                best_path,
            )
        logger.info(f"[GRPO] ═══ Training complete. Final: {final_path} ═══")

    def _checkpoint_trainer_state(self) -> Dict[str, Any]:
        state = {
            "global_step": self.global_step,
            "reward_history": self.reward_history,
            "consecutive_no_signal_updates": self.consecutive_no_signal_updates,
            "no_signal_updates_by_task": self.no_signal_updates_by_task,
            "best_eval_successes": self.best_eval_successes,
            "best_eval_episodes": self.best_eval_episodes,
            "best_eval_step": self.best_eval_step,
            "best_eval_paired_delta": self.best_eval_paired_delta,
        }
        if hasattr(self.env, "training_state_dict"):
            state["env_state"] = self.env.training_state_dict()
        return state

    def restore_training_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.global_step = int(state.get("global_step", 0))
        self.reward_history = list(state.get("reward_history", []))
        self.consecutive_no_signal_updates = int(
            state.get("consecutive_no_signal_updates", 0)
        )
        self.no_signal_updates_by_task = {
            str(key): int(value)
            for key, value in state.get("no_signal_updates_by_task", {}).items()
        }
        self.best_eval_successes = int(state.get("best_eval_successes", -1))
        self.best_eval_episodes = int(state.get("best_eval_episodes", 0))
        self.best_eval_step = int(state.get("best_eval_step", 0))
        restored_paired_delta = float(
            state.get("best_eval_paired_delta", float("-inf"))
        )
        # Checkpoints written before paired selection used 0.0 as the initial
        # sentinel.  Migrate that no-best state so a tied first evaluation wins.
        self.best_eval_paired_delta = (
            float("-inf")
            if self.best_eval_step <= 0 and restored_paired_delta <= 0.0
            else restored_paired_delta
        )
        if hasattr(self.env, "load_training_state_dict"):
            self.env.load_training_state_dict(state.get("env_state", {}))
        print(
            f"[Real] Restored trainer state: global_step={self.global_step}, "
            f"reward_history={len(self.reward_history)}",
            flush=True,
        )

    # ========================================================================
    # 评估
    # ========================================================================

    @torch.no_grad()
    def _evaluate_legacy(self, num_episodes: int = 20) -> Dict[str, float]:
        """评估当前策略。

        Args:
            num_episodes: 评估回合数

        Returns:
            metrics: {"success_rate": float, "avg_reward": float}
        """
        self.action_model.eval()
        cfg = self.cfg.grpo
        successes = 0
        total_rewards = 0.0

        for ep in range(num_episodes):
            print(
                f"[Preflight/Eval] Episode {ep + 1}/{num_episodes} starting",
                flush=True,
            )
            obs = self.env.reset()
            actions = self._sample_ode_chunk(obs, prev_rgb=None)

            _, ep_reward, ep_success = self._execute_episode(
                obs, actions, max_steps=cfg.max_episode_steps
            )

            if ep_success:
                successes += 1
            total_rewards += ep_reward
            print(
                f"[Preflight/Eval] Episode {ep + 1}/{num_episodes} finished: "
                f"success={ep_success} reward={ep_reward:.3f}",
                flush=True,
            )

        self.action_model.train()
        return {
            "success_rate": successes / num_episodes,
            "avg_reward": total_rewards / num_episodes,
        }

    # ========================================================================
    # 两阶段训练入口
    # ========================================================================

    @torch.no_grad()
    def evaluate(
        self, num_episodes: int = 20, label: str = "eval"
    ) -> Dict[str, Any]:
        """Evaluate with paired task/init-state/policy-noise seeds.

        ``num_episodes`` is interpreted per task so suite-level aggregates are
        balanced instead of being dominated by whichever task was sampled most.
        """
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        cfg = self.cfg.grpo
        task_ids = list(
            getattr(self.env, "task_ids", [getattr(self.env, "task_id", 0)])
        )
        was_training = self.action_model.training
        self.action_model.eval()

        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        numpy_state = np.random.get_state()
        records: List[Dict[str, Any]] = []
        try:
            for task_id in task_ids:
                for episode_index in range(num_episodes):
                    episode_seed = int(cfg.eval_seed + task_id * 100000 + episode_index)
                    np.random.seed(episode_seed % (2**32 - 1))
                    torch.manual_seed(episode_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(episode_seed)

                    print(
                        f"[Preflight/Eval] task={task_id} episode="
                        f"{episode_index + 1}/{num_episodes} seed={episode_seed} starting",
                        flush=True,
                    )
                    episode_label = (
                        f"{label} task {task_id} episode "
                        f"{episode_index + 1}/{num_episodes}"
                    )
                    self.rollout_watchdog.arm(f"{episode_label}: reset")
                    if hasattr(self.env, "reset_for_task"):
                        obs = self.env.reset_for_task(task_id, episode_index)
                    else:
                        obs = self.env.reset()
                    # Give every action chunk its own stable policy-noise seed.
                    # This prevents simulator/RNG consumption from changing all
                    # later actions in an otherwise paired checkpoint evaluation.
                    action_seed_base = episode_seed * 1000
                    self.rollout_watchdog.beat(f"{episode_label}: sample initial chunk")
                    actions = self._sample_ode_chunk_with_seed(
                        obs, prev_rgb=None, seed=action_seed_base
                    )
                    self.rollout_watchdog.beat(f"{episode_label}: execute episode")
                    _, ep_reward, ep_success = self._execute_episode(
                        obs,
                        actions,
                        max_steps=cfg.max_episode_steps,
                        continuation_seed_base=action_seed_base + 1,
                        watchdog_label=episode_label,
                    )
                    init_state_id = episode_index
                    if hasattr(self.env, "current_episode_key"):
                        _, init_state_id = self.env.current_episode_key
                    record = {
                        "task_id": int(task_id),
                        "init_state_id": int(init_state_id),
                        "episode_seed": episode_seed,
                        "success": bool(ep_success),
                        "reward": float(ep_reward),
                    }
                    records.append(record)
                    print(
                        f"[Preflight/Eval] task={task_id} episode="
                        f"{episode_index + 1}/{num_episodes} finished: "
                        f"success={ep_success} reward={ep_reward:.3f}",
                        flush=True,
                    )
                    self.rollout_watchdog.disarm()
        finally:
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            self.action_model.train(was_training)

        successes = sum(int(record["success"]) for record in records)
        total = len(records)
        ci_low, ci_high = _wilson_interval(successes, total)
        per_task: Dict[str, Dict[str, Any]] = {}
        for task_id in task_ids:
            task_records = [record for record in records if record["task_id"] == task_id]
            task_successes = sum(int(record["success"]) for record in task_records)
            task_low, task_high = _wilson_interval(task_successes, len(task_records))
            per_task[str(task_id)] = {
                "successes": task_successes,
                "episodes": len(task_records),
                "success_rate": task_successes / len(task_records),
                "ci95": [task_low, task_high],
            }

        loaded_checkpoint = (
            getattr(self.engine, "loaded_action_checkpoint", None)
            or self.cfg.engine.pretrained_checkpoint
        )
        evaluated_checkpoint = loaded_checkpoint
        post_train_prefix = next(
            (
                prefix
                for prefix in ("grpo", "rft", "awr", "recovery")
                if label.startswith(f"{prefix}_step_")
            ),
            None,
        )
        flow_sde_update = label.startswith("flow_sde_ppo_update_")
        if flow_sde_update and self.global_step > 0:
            update_checkpoint = Path(self.cfg.output_dir) / "checkpoints" / (
                f"flow_sde_ppo_update_{self.global_step}_resume.pt"
            )
            evaluated_checkpoint = (
                str(update_checkpoint)
                if update_checkpoint.exists()
                else f"in_memory:flow_sde_ppo_update_{self.global_step}"
            )
        elif post_train_prefix is not None and self.global_step > 0:
            step_checkpoint = Path(self.cfg.output_dir) / "checkpoints" / (
                f"{post_train_prefix}_step_{self.global_step}_resume.pt"
            )
            evaluated_checkpoint = (
                str(step_checkpoint)
                if step_checkpoint.exists()
                else f"in_memory:{post_train_prefix}_step_{self.global_step}"
            )

        metrics: Dict[str, Any] = {
            "label": label,
            "policy_checkpoint": evaluated_checkpoint,
            "base_checkpoint": loaded_checkpoint,
            "global_step": self.global_step,
            "successes": successes,
            "episodes": total,
            "success_rate": successes / total,
            "ci95": [ci_low, ci_high],
            "avg_reward": sum(record["reward"] for record in records) / total,
            "episodes_per_task": num_episodes,
            "per_task": per_task,
            "records": records,
        }
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        result_path = Path(self.cfg.output_dir) / "evaluation_results.jsonl"
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")
        print(
            f"[EvaluationSummary] label={label} successes={successes}/{total} "
            f"success_rate={metrics['success_rate']:.2%} "
            f"ci95=[{ci_low:.2%}, {ci_high:.2%}] results={result_path}",
            flush=True,
        )
        per_task_summary = ", ".join(
            f"task={task_id} {task_metrics['successes']}/{task_metrics['episodes']} "
            f"({task_metrics['success_rate']:.2%})"
            for task_id, task_metrics in per_task.items()
        )
        print(
            f"[EvaluationPerTask] label={label} {per_task_summary}",
            flush=True,
        )
        return metrics

    def train(self):
        """两阶段训练入口：SFT → GRPO。"""
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║  Action-RL 两阶段后训练                          ║")
        logger.info("╚══════════════════════════════════════════════════╝")

        # 阶段 1: SFT 冷启动
        sft_path = self.train_sft()

        if self.cfg.preflight_only and self.cfg.grpo.preflight_eval_episodes <= 0:
            raise ValueError(
                "preflight_only requires preflight_eval_episodes greater than 0"
            )
        if self.cfg.grpo.preflight_eval_episodes > 0:
            episodes = self.cfg.grpo.preflight_eval_episodes
            task_count = len(getattr(self.env, "task_ids", [0]))
            print(
                f"[Preflight] Evaluating the loaded policy with official inference: "
                f"{episodes} episodes/task x {task_count} task(s) "
                f"= {episodes * task_count} episodes before GRPO",
                flush=True,
            )
            baseline = self.evaluate(num_episodes=episodes, label="preflight")
            self.preflight_records = {
                (
                    int(record["task_id"]),
                    int(record["init_state_id"]),
                    int(record["episode_seed"]),
                ): bool(record["success"])
                for record in baseline["records"]
            }
            print(
                f"[Preflight] Loaded policy: success_rate="
                f"{baseline['success_rate']:.2%}, avg_reward="
                f"{baseline['avg_reward']:.3f}",
                flush=True,
            )
            zero_success_tasks = [
                task_id
                for task_id, task_metrics in baseline["per_task"].items()
                if task_metrics["successes"] == 0
            ]
            if (
                not self.cfg.preflight_only
                and self.cfg.grpo.require_preflight_success
                and zero_success_tasks
            ):
                raise RuntimeError(
                    f"Loaded-policy preflight had zero successes on tasks "
                    f"{zero_success_tasks}. Sparse LIBERO rewards cannot "
                    "bootstrap GRPO on those tasks, so training was "
                    "stopped before producing no-op checkpoints. First verify "
                    "this exact checkpoint/task with official evaluation and "
                    "check dataset_statistics.json / unnorm_key."
                )
            if self.cfg.preflight_only:
                print(
                    "[Preflight] Preflight-only mode complete; GRPO was not started.",
                    flush=True,
                )
                return

        # 阶段 2: 在线 GRPO 演化
        self.train_grpo()

        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║  全部训练完成                                    ║")
        logger.info("╚══════════════════════════════════════════════════╝")

    # ========================================================================
    # 辅助
    # ========================================================================

    def _freeze_perception_modules(self):
        """冻结感知模块 (RAFT, Affordance)。"""
        for module in [self.raft, self.flow_codec, self.affordance]:
            if module is None:
                continue
            if hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad_(False)
                module.eval() if hasattr(module, "eval") else None
        logger.info("[Trainer] Perception modules (RAFT, FlowCodec, Affordance) frozen")


# ============================================================================
# 主入口 (Main Entry Point)
# ============================================================================
# 两种运行模式:
#   1. --mock   : 轻量 smoke test, 用 Mock 组件跑通 SFT→GRPO 完整流程 (无需 SFT checkpoint / 仿真环境)
#   2. --real   : 真实训练, 加载 SFT checkpoint + 真实感知模块 + 仿真环境
#
# 用法:
#   python -m dit4dit_rl_trainer.trainer --mock
#   python -m dit4dit_rl_trainer.trainer --real --config_yaml xxx.yaml --checkpoint /path/to/sft.pt
# ============================================================================


# ----------------------------------------------------------------------------
# Mock 组件 (仅用于 smoke test, 不依赖 DiT4DiT 真实模型)
# ----------------------------------------------------------------------------

class _MockBackboneInterface:
    """Mock 多模态主干: 跳过真实 VAE/Video DiT, 直接返回随机隐特征。"""

    def __init__(self, hidden_dim: int = 128, N_per_stream: int = 16):
        self.hidden_dim = hidden_dim
        self.N_per_stream = N_per_stream

    def build_multistream_inputs(self, rgb_images, flow_images, value_images, instructions):
        """与 MultiStreamInterface.build_multistream_inputs 接口一致。"""
        return {
            "rgb_images": rgb_images,
            "flow_images": flow_images,
            "value_images": value_images,
            "instructions": instructions,
        }

    def __call__(self, **kwargs):
        """模拟 backbone 前向: 返回含 hidden_states 的对象。"""
        rgb_images = kwargs.get("rgb_images", kwargs.get("rgb", [None]))
        B = len(rgb_images) if isinstance(rgb_images, list) else 1
        out = SimpleNamespace()
        # hidden_states[-1] = (B, 3*N, D), 与真实 Video DiT 第 18 层输出一致
        out.hidden_states = [
            torch.randn(B, 3 * self.N_per_stream, self.hidden_dim)
        ]
        out.future_video_loss = None
        return out


class _MockModel:
    """Mock DiT4DiT_Multistream: 仅含 backbone_interface + action_model。"""

    def __init__(self, action_model: nn.Module, hidden_dim: int = 128, N_per_stream: int = 16):
        self.action_model = action_model
        self.backbone_interface = _MockBackboneInterface(hidden_dim, N_per_stream)


class _MockEngine:
    """Mock RLEngine: 提供 trainer 所需的 .model/.action_model/.optimizer。"""

    def __init__(self, action_model: nn.Module, hidden_dim: int = 128, N_per_stream: int = 16, lr: float = 1e-4):
        self.model = _MockModel(action_model, hidden_dim, N_per_stream)
        self.action_model = action_model
        self.optimizer = torch.optim.AdamW(
            [p for p in action_model.parameters() if p.requires_grad], lr=lr
        )

    def save_checkpoint(
        self, path: str, trainer_state: Optional[Dict[str, Any]] = None
    ):
        logger.info(f"[MockEngine] save_checkpoint → {path} (no-op in mock mode)")

    def sync_ref_to_current(self):
        logger.info("[MockEngine] sync_ref_to_current (no-op in mock mode)")


class _MockEnv:
    """Mock 仿真环境: 实现 EnvProtocol, 返回随机观测。"""

    def __init__(self, action_dim: int = 7, img_size: Tuple[int, int] = (224, 224), horizon: int = 8):
        self.action_dim = action_dim
        self.img_size = img_size
        self.horizon = horizon
        self._step_count = 0

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._group_start = {
            "rgb": np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8),
            "state": np.random.randn(10).astype(np.float32),
            "lang": "pick up the red block",
            "contact_points": np.array([[0.5, 0.5, 0.3, 0.9]]),
        }
        return self.reset_to_group_start()

    def reset_to_group_start(self) -> Dict[str, Any]:
        self._step_count = 0
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._group_start.items()
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self._step_count += 1
        done = self._step_count >= self.horizon
        # 50% 概率成功 (用于测试 GRPO 组优势计算)
        success = done and (np.random.rand() > 0.5)
        obs = {
            "rgb": np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8),
            "state": np.random.randn(10).astype(np.float32),
            "lang": "pick up the red block",
            "contact_points": np.array([[0.5, 0.5, 0.3, 0.9]]),
        }
        reward = 1.0 if success else 0.0
        return obs, reward, done, {"success": success}


class _MockRAFT:
    """Mock RAFT 光流提取器: 不依赖 torchvision, 返回随机光流。"""

    def __call__(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        H, W = frame1.shape[:2]
        return np.random.randn(H, W, 2).astype(np.float32) * 2.0


class _MockSFTDataset:
    """Mock SFT 数据集: 可迭代, 每次产生一个 batch。"""

    def __init__(self, n_batches: int = 2, T: int = 8, action_dim: int = 7,
                 img_size: Tuple[int, int] = (224, 224)):
        self.n_batches = n_batches
        self.T = T
        self.action_dim = action_dim
        self.img_size = img_size

    def __iter__(self):
        for _ in range(self.n_batches):
            yield {
                "rgb": [np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8)
                        for _ in range(self.T)],
                "flow": [np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8)
                         for _ in range(self.T)],
                "value": [np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8)
                          for _ in range(self.T)],
                "lang": "pick up the red block",
                # action shape (1, T, D_action) — compute_sft_loss 期望 3D (B, T, D)
                "action": np.random.randn(1, self.T, self.action_dim).astype(np.float32),
                "action_mask": np.ones((1, self.T, self.action_dim), dtype=np.float32),
                "state": np.random.randn(1, 10).astype(np.float32),
            }


# ----------------------------------------------------------------------------
# Mock 模式: smoke test
# ----------------------------------------------------------------------------

def _run_mock_smoke_test():
    """Mock 模式: 用轻量组件跑通 SFT→GRPO 完整流程, 验证训练循环无 bug。

    不需要 SFT checkpoint、仿真环境、真实 VAE/Video DiT。
    复用 verify_pipeline.py 中已验证的 MockActionModel。
    """
    np.random.seed(42)
    torch.manual_seed(42)
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Mock Smoke Test: 验证 RLTrainer 完整流程            ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ---- 复用已验证的 MockActionModel ----
    from dit4dit_rl_trainer.verify_pipeline import MockActionModel

    HIDDEN_DIM = 128
    N_PER_STREAM = 16
    ACTION_DIM = 7

    action_model = MockActionModel(
        hidden_size=HIDDEN_DIM, action_dim=ACTION_DIM, state_dim=10,
        N_per_stream=N_PER_STREAM,
    )

    # ---- Mock 引擎 ----
    engine = _MockEngine(action_model, hidden_dim=HIDDEN_DIM, N_per_stream=N_PER_STREAM)

    # ---- Mock 环境 ----
    env = _MockEnv(action_dim=ACTION_DIM, img_size=(224, 224), horizon=8)

    # ---- 感知模块: Mock RAFT + 真实 FlowCodec/AffordanceMapGenerator ----
    # FlowCodec 和 AffordanceMapGenerator 仅依赖 numpy/cv2, 可直接用真实实现
    raft = _MockRAFT()
    try:
        from dit4dit_multistream.dataset_multistream import FlowCodec, AffordanceMapGenerator
        flow_codec = FlowCodec()
        affordance = AffordanceMapGenerator(image_size=(224, 224))
        print("  感知模块: 真实 FlowCodec + AffordanceMapGenerator ✓")
    except Exception as e:
        # cv2 不可用时回退到 mock
        print(f"  感知模块回退到 mock (原因: {e})")
        flow_codec = _MockFlowCodec()
        affordance = _MockAffordance()

    # ---- 配置: 极小规模, 仅验证流程 ----
    _tmp_dir = tempfile.gettempdir()
    cfg = TrainerConfig(
        sft=SFTConfig(num_demos=2, num_epochs=2, learning_rate=1e-3, batch_size=1,
                      save_path=os.path.join(_tmp_dir, "mock_sft.pt")),
        grpo=GRPOConfig(
            num_iterations=3, group_size=4, action_horizon=8,
            num_inference_steps=4, noise_level=0.7,
            sde_window_size=1, sde_window_range=(0, 2),
            clip_range=0.2, beta_kl=0.04,
            learning_rate=1e-4, save_freq=100, eval_freq=100, log_freq=1,
            global_std=False,  # mock 模式用组内 std (历史少)
        ),
        engine=RLEngineConfig(pretrained_checkpoint="", device="cpu"),
        device="cpu",
        output_dir=os.path.join(_tmp_dir, "mock_rl_output"),
        seed=42,
        model_variant="multistream",
        normalize_state=False,
    )

    # ---- Mock SFT 数据集 ----
    sft_dataset = _MockSFTDataset(n_batches=2, T=8, action_dim=ACTION_DIM, img_size=(224, 224))

    # ---- 创建 trainer ----
    trainer = RLTrainer(
        engine=engine,
        env=env,
        raft_extractor=raft,
        flow_codec=flow_codec,
        affordance_generator=affordance,
        cfg=cfg,
        sft_dataset=sft_dataset,
    )

    # ---- 运行两阶段训练 ----
    print("\n--- 开始 SFT → GRPO 两阶段训练 (mock) ---")
    trainer.train()
    print("\n--- Mock smoke test 完成 ---")

    # ---- 验证关键指标 ----
    assert trainer.global_step == cfg.grpo.num_iterations, \
        f"GRPO 步数不匹配: {trainer.global_step} != {cfg.grpo.num_iterations}"
    print(f"  ✓ GRPO 完成步数: {trainer.global_step}")
    print(f"  ✓ 奖励历史长度: {len(trainer.reward_history)}")
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  🎉 Mock smoke test 通过! 训练循环可正常运行          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("\n下一步 (真实训练):")
    print("  python -m dit4dit_rl_trainer.trainer --real \\")
    print("    --config_yaml dit4dit_multistream/dit4dit_multistream_libero.yaml \\")
    print("    --checkpoint /path/to/sft_checkpoint.pt")


class _MockFlowCodec:
    """FlowCodec 回退 mock (cv2 不可用时)。"""
    def encode(self, flow, max_magnitude=None, sigma=0.15):
        H, W = flow.shape[:2]
        return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8), 1.0


class _MockAffordance:
    """AffordanceMapGenerator 回退 mock。"""
    def generate(self, contact_points=None, camera_intrinsics=None,
                 camera_extrinsics=None, default_value=0.0):
        return np.zeros((224, 224, 3), dtype=np.uint8)


# ----------------------------------------------------------------------------
# 真实模式: 加载 SFT checkpoint + 真实环境训练
# ----------------------------------------------------------------------------

def _run_real_training(args):
    """真实模式: 加载 SFT checkpoint, 创建真实感知模块 + 环境, 运行两阶段训练。

    需要提供:
      - --config_yaml: 原版或 Multistream DiT4DiT 配置文件
      - --checkpoint: SFT 阶段产出的 checkpoint 路径
      - --env: 仿真环境名称 (libero/maniskill/robocasa), 需有对应 EnvWrapper
    """
    from omegaconf import OmegaConf

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        print(f"[Real] PyTorch CPU threads limited to {args.cpu_threads}", flush=True)

    if args.resume_rl:
        # A resume checkpoint stores only Action DiT/LoRA weights. Reusing it
        # with a different frozen SFT backbone silently creates a different
        # policy, so always restore the base checkpoint recorded at save time.
        resume_metadata = torch.load(args.resume_rl, map_location="cpu")
        saved_engine_cfg = resume_metadata.get("engine_cfg", {})
        saved_base_checkpoint = saved_engine_cfg.get("pretrained_checkpoint", "")
        if saved_base_checkpoint:
            saved_base_path = Path(saved_base_checkpoint).expanduser()
            if not saved_base_path.is_file():
                raise FileNotFoundError(
                    "The RL resume checkpoint requires its original SFT checkpoint, "
                    f"but it is unavailable: {saved_base_path}"
                )
            requested_base_path = Path(args.checkpoint).expanduser()
            if requested_base_path.resolve() != saved_base_path.resolve():
                print(
                    "[Real] Resume base checkpoint override: "
                    f"{requested_base_path} -> {saved_base_path}",
                    flush=True,
                )
            args.checkpoint = str(saved_base_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- 1. 加载配置 ----
    cfg_yaml = OmegaConf.load(args.config_yaml)
    print(f"[Real] Config loaded from {args.config_yaml}")
    print(f"[Real] Framework: {cfg_yaml.framework.name}")

    configured_name = str(cfg_yaml.framework.name).lower()
    inferred_variant = "multistream" if "multistream" in configured_name else "original"
    model_variant = inferred_variant if args.model_variant == "auto" else args.model_variant

    # ---- 3. 创建 RL 引擎 (加载 SFT checkpoint, 冻结 backbone) ----
    engine_cfg = RLEngineConfig(
        pretrained_checkpoint=args.checkpoint,
        learning_rate=args.lr,
        device=args.device,
        dtype=args.dtype,
        ref_model_offload=args.ref_offload,
        model_variant=model_variant,
        trainable_scope=args.trainable_scope,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    engine = RLEngine(engine_cfg)
    resume_trainer_state: Dict[str, Any] = {}
    if args.resume_rl:
        resume_trainer_state = engine.load_action_checkpoint(
            args.resume_rl, load_optimizer=True
        )
        print(f"[Real] Resumed Action DiT/optimizer from {args.resume_rl}")
    print(f"[Real] Engine initialized. Action DiT trainable params: "
          f"{sum(p.numel() for p in engine.action_model.parameters() if p.requires_grad):,}")

    # ---- 4. Multistream-only perception modules ----
    raft = flow_codec = affordance = None
    if model_variant == "multistream":
        from dit4dit_multistream.dataset_multistream import (
            RAFTFlowExtractor, FlowCodec, AffordanceMapGenerator,
        )
        image_size = tuple(cfg_yaml.datasets.vla_data.image_size)
        raft = RAFTFlowExtractor(
            device=args.device,
            checkpoint=getattr(args, "raft_checkpoint", None),
        )
        flow_codec = FlowCodec()
        affordance = AffordanceMapGenerator(image_size=image_size)
        print("[Real] Perception modules: RAFT + FlowCodec + AffordanceMapGenerator")
    else:
        print("[Real] Original DiT4DiT: RGB backbone only")

    # ---- 5. 创建仿真环境 ----
    env = _build_real_env(args.env, cfg_yaml, args)
    print(f"[Real] Environment: {type(env).__name__}")

    # ---- 6. (可选) SFT 数据集 ----
    sft_dataset = None
    if args.sft_data:
        sft_dataset = _build_sft_dataset(
            args.sft_data, cfg_yaml, model_variant, args.raft_checkpoint
        )
        print(f"[Real] SFT dataset loaded from {args.sft_data}")
    rsft_anchor_dataset = None
    if args.rsft_anchor_data:
        rsft_anchor_dataset = _build_sft_dataset(
            args.rsft_anchor_data, cfg_yaml, model_variant, args.raft_checkpoint
        )
        print(f"[Real] RSFT anchor dataset loaded from {args.rsft_anchor_data}")

    # ---- 7. 训练配置 ----
    trainer_cfg = TrainerConfig(
        sft=SFTConfig(
            num_demos=args.sft_demos,
            num_epochs=args.sft_epochs,
            learning_rate=args.sft_lr,
            batch_size=args.sft_batch,
            save_path=os.path.join(args.output_dir, "sft_checkpoint.pt"),
        ),
        grpo=GRPOConfig(
            num_iterations=args.grpo_iters,
            group_size=args.group_size,
            action_horizon=cfg_yaml.framework.action_model.future_action_window_size + 1,
            num_inference_steps=args.k_steps,
            noise_level=args.noise_level,
            noise_level_end=args.noise_level_end,
            learning_rate=args.lr,
            clip_range=args.clip_range,
            beta_kl=args.beta_kl,
            global_std=args.global_std,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            max_episode_steps=args.max_episode_steps,
            progress_steps=args.progress_steps,
            preflight_eval_episodes=args.preflight_eval_episodes,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            require_preflight_success=args.require_preflight_success,
            stop_on_negative_paired_delta=args.stop_on_negative_paired_delta,
            max_no_signal_updates=args.max_no_signal_updates,
            chunks_per_episode=args.rl_chunks_per_episode,
            collect_all_chunks=args.collect_all_chunks,
            chunk_stride=args.rl_chunk_stride,
            inner_epochs=args.grpo_inner_epochs,
            temporal_credit_decay=args.temporal_credit_decay,
            spread_chunks_across_episode=args.spread_chunks_across_episode,
            common_continuation_seed=args.common_continuation_seed,
            rsft_anchor_weight=args.rsft_anchor_weight,
            rsft_online_fraction=args.rsft_online_fraction,
            rsft_anchor_batch_size=args.rsft_anchor_batch_size,
            rsft_replay_capacity=args.rsft_replay_capacity,
            rollout_stall_timeout=args.rollout_stall_timeout,
            rollout_heartbeat_seconds=args.rollout_heartbeat_seconds,
        ),
        engine=engine_cfg,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        model_variant=model_variant,
        normalize_state=not args.state_already_normalized,
        unnorm_key=args.unnorm_key,
        action_unnorm_mode=args.action_unnorm_mode,
        preflight_only=args.preflight_only,
    )

    # ---- 8. 创建 trainer 并启动训练 ----
    trainer = RLTrainer(
        engine=engine,
        env=env,
        raft_extractor=raft,
        flow_codec=flow_codec,
        affordance_generator=affordance,
        cfg=trainer_cfg,
        sft_dataset=sft_dataset,
        rsft_anchor_dataset=rsft_anchor_dataset,
    )
    trainer.restore_training_state(resume_trainer_state)
    try:
        trainer.rollout_watchdog.start()
        trainer.train()
    finally:
        trainer.rollout_watchdog.stop()
        if hasattr(env, "close"):
            env.close()


def _build_real_env(env_name: str, cfg, args) -> EnvProtocol:
    """构建真实仿真环境 (用户根据实际环境适配)。

    支持: libero / maniskill / robocasa
    若无对应 wrapper, 抛出 NotImplementedError 并提示用户实现。
    """
    env_name = env_name.lower()
    if env_name == "libero":
        from dit4dit_rl_trainer.envs import LiberoEnvWrapper

        resolution = int(cfg.datasets.vla_data.image_size[0])
        return LiberoEnvWrapper(
            benchmark_name=args.libero_suite,
            task_id=args.libero_task_id,
            task_ids=args.libero_task_ids,
            task_schedule=args.libero_task_schedule,
            task_block_size=args.libero_task_block_size,
            resolution=resolution,
            seed=args.seed,
            init_state_id=args.libero_init_state_id,
            num_steps_wait=args.libero_wait_steps,
            camera_name=args.libero_camera,
            wrist_camera_name=args.libero_wrist_camera,
            env_action_dim=args.env_action_dim,
            render_gpu_device_id=int(os.environ.get("MUJOCO_RENDER_GPU_DEVICE_ID", "-1")),
        )
    else:
        raise NotImplementedError(
            f"环境 '{env_name}' 的 wrapper 未实现。\n"
            f"请创建 EnvWrapper(EnvProtocol) 子类并在此处注册。\n"
            f"EnvProtocol 接口: reset() -> obs_dict, step(action) -> (obs, reward, done, info)\n"
            f"obs_dict 需包含: rgb (H,W,3), state (D,), lang (str), contact_points (N,3+)"
        )


def _build_sft_dataset(
    sft_data_path: str,
    cfg,
    model_variant: str,
    raft_checkpoint: Optional[str],
    data_mix: Optional[str] = None,
):
    """Build the optional cold-start dataset using the repository loaders."""
    data_cfg = cfg.datasets.vla_data.copy()
    data_cfg.data_root_dir = sft_data_path
    if data_mix:
        data_cfg.data_mix = data_mix
    if model_variant == "original":
        from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset

        return get_vla_dataset(data_cfg=data_cfg)

    from dit4dit_multistream.dataset_multistream import MultiStreamDataset

    return MultiStreamDataset(
        root_dir=sft_data_path,
        data_mix=data_cfg.data_mix,
        image_size=tuple(data_cfg.image_size),
        device=str(getattr(cfg.framework.cosmos25, "device", "cuda")),
        raft_checkpoint=raft_checkpoint,
        action_horizon=int(data_cfg.action_horizon),
        action_video_freq_ratio=int(data_cfg.action_video_freq_ratio),
        max_action_dim=int(data_cfg.max_action_dim),
        video_horizon=len(data_cfg.video_delta_indices),
    )


# ----------------------------------------------------------------------------
# argparse 主入口
# ----------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Action-RL 两阶段后训练 (SFT → 在线 GRPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行示例:
  # Mock smoke test (无需 checkpoint/环境, 验证训练循环)
  python -m dit4dit_rl_trainer.trainer --mock

  # 真实训练
  python -m dit4dit_rl_trainer.trainer --real \\
    --config_yaml dit4dit_multistream/dit4dit_multistream_libero.yaml \\
    --checkpoint /path/to/sft_checkpoint.pt \\
    --env libero
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true",
                      help="Mock smoke test: 用轻量组件跑通完整流程 (无需 checkpoint/环境)")
    mode.add_argument("--real", action="store_true",
                      help="真实训练: 加载 SFT checkpoint + 真实环境")

    # 真实模式参数
    real = parser.add_argument_group("真实模式参数 (--real)")
    real.add_argument("--config_yaml", type=str, default=None,
                      help="原版或 Multistream DiT4DiT YAML 配置文件路径")
    real.add_argument("--checkpoint", type=str, default="",
                      help="SFT 阶段产出的 checkpoint 路径")
    real.add_argument("--resume_rl", type=str, default=None,
                      help="可选：grpo_*_resume.pt，用于恢复 Action DiT 与优化器")
    real.add_argument("--env", type=str, default="libero",
                      choices=["libero", "maniskill", "robocasa"],
                      help="仿真环境名称")
    real.add_argument("--model_variant", choices=["auto", "original", "multistream"],
                      default="auto", help="模型适配模式；auto 从 YAML 推断")
    real.add_argument("--unnorm_key", type=str, default=None,
                      help="dataset_statistics.json 中的动作反归一化 key")
    real.add_argument("--action_unnorm_mode", choices=["auto", "min_max", "q99"],
                      default="auto", help="动作反归一化范围；原版 LIBERO 使用 min_max")
    real.add_argument("--preflight_only", action="store_true",
                      help="仅评测 SFT 基线并退出，不启动 GRPO")
    real.add_argument("--state_already_normalized", action="store_true",
                      help="输入 state 已完成 sin/cos 与 padding 时使用")
    real.add_argument("--sft_data", type=str, default=None,
                      help="SFT 数据路径 (None 跳过 SFT 直接 GRPO)")
    real.add_argument(
        "--rsft_anchor_data",
        type=str,
        default=None,
        help="Original SFT data used only by the GRPO auxiliary anchor loss",
    )
    real.add_argument("--device", type=str, default="cuda")
    real.add_argument("--dtype", type=str, default="bfloat16")
    real.add_argument("--output_dir", type=str, default="checkpoints/rl")
    real.add_argument("--seed", type=int, default=42)

    libero = parser.add_argument_group("LIBERO 环境参数")
    libero.add_argument("--libero_suite", type=str, default="libero_10")
    libero.add_argument("--libero_task_id", type=int, default=0)
    libero.add_argument(
        "--libero_task_ids",
        type=str,
        default="",
        help="Suite task ids: 'all', '0-9', or a comma-separated list",
    )
    libero.add_argument(
        "--libero_task_schedule",
        choices=("round_robin", "random"),
        default="round_robin",
    )
    libero.add_argument("--libero_task_block_size", type=int, default=1)
    libero.add_argument("--libero_init_state_id", type=int, default=None)
    libero.add_argument("--libero_wait_steps", type=int, default=10)
    libero.add_argument("--libero_camera", type=str, default="agentview_image")
    libero.add_argument("--libero_wrist_camera", type=str,
                        default="robot0_eye_in_hand_image")
    libero.add_argument("--env_action_dim", type=int, default=7)

    # 训练超参
    hp = parser.add_argument_group("训练超参")
    hp.add_argument(
        "--trainable_scope", choices=("full", "output", "lora"), default="lora"
    )
    hp.add_argument("--lora_rank", type=int, default=8)
    hp.add_argument("--lora_alpha", type=float, default=16.0)
    hp.add_argument("--lora_dropout", type=float, default=0.0)
    hp.add_argument(
        "--rl_chunks_per_episode",
        type=int,
        default=1,
        help="Number of replanning action chunks retained per trajectory",
    )
    hp.add_argument(
        "--collect_all_chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="保留整条 episode 的所有重规划 chunk，而不是只保留前 N 个",
    )
    hp.add_argument(
        "--rl_chunk_stride",
        type=int,
        default=4,
        help="Spacing between trainable action chunks (4 selects 0,4,8,...)",
    )
    hp.add_argument(
        "--grpo_inner_epochs",
        type=int,
        default=1,
        help="Optimizer passes over each collected rollout group",
    )
    hp.add_argument(
        "--temporal_credit_decay",
        type=float,
        default=1.0,
        help="Credit multiplier per chunk distance; values below 1 favor later chunks",
    )
    hp.add_argument(
        "--spread_chunks_across_episode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Spread recorded decisions from episode start through the final action chunk",
    )
    hp.add_argument(
        "--common_continuation_seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use common random numbers for untrained continuation chunks in a group",
    )
    hp.add_argument("--rsft_anchor_weight", type=float, default=0.0)
    hp.add_argument("--rsft_online_fraction", type=float, default=0.3)
    hp.add_argument("--rsft_anchor_batch_size", type=int, default=4)
    hp.add_argument("--rsft_replay_capacity", type=int, default=128)
    hp.add_argument("--lr", type=float, default=5e-6, help="GRPO learning rate")
    hp.add_argument("--sft_lr", type=float, default=1e-4, help="SFT 学习率")
    hp.add_argument("--sft_demos", type=int, default=40, help="SFT 演示数量")
    hp.add_argument("--sft_epochs", type=int, default=50, help="SFT 训练轮数")
    hp.add_argument("--sft_batch", type=int, default=8, help="SFT batch size")
    hp.add_argument("--grpo_iters", type=int, default=10000, help="GRPO 迭代次数")
    hp.add_argument("--group_size", type=int, default=8, help="GRPO 组大小 G")
    hp.add_argument("--k_steps", type=int, default=4, help="SDE 采样步数 K")
    hp.add_argument("--noise_level", type=float, default=0.7, help="SDE 噪声系数起点")
    hp.add_argument("--noise_level_end", type=float, default=0.7,
                    help="GRPO rollout 噪声线性退火终点")
    hp.add_argument("--clip_range", type=float, default=2e-6, help="PPO clip epsilon")
    hp.add_argument("--beta_kl", type=float, default=0.08, help="KL penalty beta")
    hp.add_argument("--global_std", action=argparse.BooleanOptionalAction, default=True,
                    help="是否用历史全局标准差计算组优势")
    hp.add_argument("--save_freq", type=int, default=500)
    hp.add_argument("--eval_freq", type=int, default=200)
    hp.add_argument("--eval_episodes", type=int, default=10,
                    help="Periodic evaluation episodes per configured task")
    hp.add_argument("--eval_seed", type=int, default=10000,
                    help="Seed base for paired SFT/RL evaluation episodes")
    hp.add_argument("--max_episode_steps", type=int, default=520,
                    help="每条 LIBERO rollout 的最大环境步数")
    hp.add_argument("--progress_steps", type=int, default=64,
                    help="每隔多少环境步输出 rollout 进度；0 表示关闭")
    hp.add_argument("--preflight_eval_episodes", type=int, default=5,
                    help="GRPO 前运行原版 SFT 推理完整评测回合数；0 表示关闭")
    hp.add_argument(
        "--require_preflight_success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SFT 预检成功率为 0 时终止，避免稀疏奖励下空跑",
    )
    hp.add_argument(
        "--stop_on_negative_paired_delta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop after a paired evaluation falls below the preflight baseline",
    )
    hp.add_argument("--max_no_signal_updates", type=int, default=5,
                    help="连续多少个同奖励 group 后终止；0 表示关闭")
    hp.add_argument(
        "--rollout_stall_timeout",
        type=float,
        default=0.0,
        help="Seconds without rollout progress before stack dump and exit; 0 disables",
    )
    hp.add_argument(
        "--rollout_heartbeat_seconds",
        type=float,
        default=30.0,
        help="Emit a rollout heartbeat at this interval while work is progressing",
    )
    hp.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="PyTorch CPU worker threads; 0 keeps PyTorch's default",
    )
    hp.add_argument("--ref_offload", action="store_true",
                    help="reference model CPU offload (省显存)")
    hp.add_argument("--raft_checkpoint", type=str, default=None,
                    help="RAFT 权重本地路径 (无网络时使用)")

    return parser


if __name__ == "__main__":
    # 确保项目根目录在 path 中
    import sys as _sys
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_argparser()
    args = parser.parse_args()

    if args.mock:
        _run_mock_smoke_test()
    elif args.real:
        if not args.config_yaml:
            parser.error("--real 模式需要 --config_yaml")
        if not args.checkpoint:
            parser.error("--real 模式需要 --checkpoint (SFT checkpoint 路径)")
        _run_real_training(args)
