# -*- coding: utf-8 -*-
"""
Step 2: SDE 采样模块 (SDE Rollout Worker)
=================================================================

实现单步 SDE 混合采样，在 Action DiT 的连续速度场上进行 RL 探索。

核心数学 (Flow-SDE 采样)：

1. 速度场参数化 (因果掩码约束)::
     v_φ^action(a_τ, τ | H_video, M)
     其中 M 是因果掩码：前 N 列 = -1e9 (屏蔽语义 Z_x)，后 2N 列 = 0 (开放物理+意图)

2. SDE 漂移项::
     μ_φ(a_τ, τ) = v_φ + (σ_τ² / 2τ) · (a_τ + (1-τ)·v_φ)

3. SDE 更新步::
     a_{τ+Δτ} = a_τ + μ_φ·Δτ + σ_τ·√|Δτ|·ε_τ
     其中 σ_τ = a_noise · √(τ / (1 - τ + ε_s))

4. 混合 ODE-SDE 采样::
     K 步推理中，在 sde_window_range 范围内随机选 sde_window_size 步注入 SDE 噪声，
     其余步骤保持确定性 ODE。好钢用在刀刃上，既保留探索能力又加速训练。

参考实现：
  - 公开 Flow-SDE 实现
    (sde_step_with_logprob 的漂移项与 log_prob 计算)
  - 公开 GRPO 配置
    (noise_level, sde_window_size, sde_window_range 参数)

兼容范围：
  - 原版 DiT4DiT: 使用标准 Action DiT cross-attention
  - DiT4DiT-Multistream: 自动检测并注入动态 causal_key_mask
  两种模式都不修改模型目录内的源码，仅通过 Action Head 的公开子模块计算速度场。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构：SDE 步记录 (供 Step 3 高斯转移 log_prob + GRPO loss 使用)
# ============================================================================

@dataclass
class SDEStepRecord:
    """记录 SDE 步的关键张量，用于 Step 3 计算离散高斯转移概率。

    在混合采样中，只有 SDE 步需要记录（ODE 步是确定性的，log_prob 不贡献梯度）。

    Attributes:
        v_phi: 速度场 v_φ(a_τ, τ | H_video, M)，shape (B, T, D_action)
        a_tau: SDE 步前的动作 a_τ，shape (B, T, D_action)
        a_next: SDE 步后的动作 a_{τ+Δτ}，shape (B, T, D_action)
        sigma_tau: 噪声系数 σ_τ，shape (B,) 或标量
        epsilon: 注入的高斯噪声 ε_τ，shape (B, T, D_action)
        prev_sample_mean: 漂移均值项 μ_φ·Δτ (即 prev_sample_mean)，
            shape (B, T, D_action)。用于离散高斯 log_prob。
        timestep: 当前时间步 τ，shape (B,) 或标量
        dt: 步长 Δτ = τ_{k+1} - τ_k < 0，shape (B,) 或标量
        sqrt_dt: √|Δτ|，用于 log_prob
    """
    v_phi: torch.Tensor
    a_tau: torch.Tensor
    a_next: torch.Tensor
    sigma_tau: torch.Tensor
    epsilon: torch.Tensor
    prev_sample_mean: torch.Tensor
    timestep: torch.Tensor
    dt: torch.Tensor
    sqrt_dt: torch.Tensor


@dataclass
class RolloutResult:
    """混合 SDE 采样的完整输出。

    Attributes:
        final_action: 最终生成的动作块 a_0，shape (B, T, D_action)
        sde_records: SDE 步的记录列表（用于 Step 3 的 log_prob 计算）
        all_velocities: 所有 K 步的速度场列表 (用于 KL 散度 ||v_φ - v_ref||²)
        timesteps_schedule: 时间步调度 τ_0, τ_1, ..., τ_K
        sde_step_indices: 哪些步骤使用了 SDE（索引列表）
    """
    final_action: torch.Tensor
    sde_records: List[SDEStepRecord]
    all_velocities: List[torch.Tensor]
    timesteps_schedule: torch.Tensor
    sde_step_indices: List[int]


# ============================================================================
# 核心函数 1: 动作速度场计算
# ============================================================================

def compute_action_velocity(
    action_model: nn.Module,
    a_tau: torch.Tensor,
    tau: torch.Tensor,
    vl_embs: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    N_per_stream: Optional[int] = None,
) -> torch.Tensor:
    """计算 Action DiT 的动作速度场 v_φ(a_τ, τ | H_video, M)。

    复用 action_model 的子模块（不修改原代码）：
      action_encoder → state_encoder → position_embedding → MaskedDiT → action_decoder

    Multistream Action Head 会通过 _build_causal_mask(vl_embs) 构建因果掩码；
    原版 Action Head 没有该接口，直接使用其标准 cross-attention。

    Args:
        action_model: 原版或 Multistream FlowmatchingActionHead (可训练或 reference)
        a_tau: 当前动作块 a_τ，shape (B, T, D_action)
        tau: 当前时间步 τ，shape (B,) 或标量。τ∈(0,1]，τ=1 为纯噪声。
        vl_embs: Video DiT 条件特征。原版为 (B, S, D)，Multistream 为 (B, 3*N, D)
        state: 机器人状态，shape (B, D_state) 或 None
        N_per_stream: 每路模态的 token 数 N。None 时从 vl_embs 推断 (S//3)。

    Returns:
        v_phi: 预测速度场 v_φ，shape (B, T, D_action)
            在 Rectified Flow 中 v = noise - data，指向去噪方向。
    """
    B, T, D_action = a_tau.shape
    device = a_tau.device
    model_dtype = next(action_model.parameters()).dtype
    model_a_tau = a_tau.to(dtype=model_dtype)
    model_vl_embs = vl_embs.to(device=device, dtype=model_dtype)
    model_state = state.to(device=device, dtype=model_dtype) if state is not None else None

    # ---- 1. 时间步离散化 ----
    # action_model.num_timestep_buckets 将连续 τ 映射到离散 bucket
    num_buckets = action_model.num_timestep_buckets
    if tau.dim() == 0:
        tau = tau.expand(B)
    # Match ``ActionDiT.predict_action`` exactly.  In particular, tau=1 is
    # represented by ``num_buckets`` (1000 in the LIBERO checkpoint), not
    # clamped to 999.  The time embedding is sinusoidal, so this is a valid
    # value rather than an embedding-table index.
    t_discretized = (tau * num_buckets).long().clamp_min(0)  # (B,)

    # ---- 2. 编码 noised action ----
    # action_encoder: (B, T, D_action) + (B,) → (B, T, w)
    action_features = action_model.action_encoder(model_a_tau, t_discretized)

    # ---- 3. 编码 state + position embedding ----
    state_features = None
    if model_state is not None and action_model.state_encoder is not None:
        state_features = action_model.state_encoder(model_state)  # (B, w) or (B, 1, w)
        if state_features.dim() == 2:
            state_features = state_features.unsqueeze(1)  # (B, 1, w)

    # position embedding
    if action_model.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = action_model.position_embedding(pos_ids).unsqueeze(0)  # (1, T, w)
        action_features = action_features + pos_embs

    # 拼接 state + action
    if state_features is not None:
        sa_embs = torch.cat((state_features, action_features), dim=1)  # (B, 1+T, w)
    else:
        sa_embs = action_features  # (B, T, w)

    # ---- 4. Action DiT 前向 ----
    model_kwargs = dict(
        hidden_states=sa_embs,
        encoder_hidden_states=model_vl_embs,
        encoder_attention_mask=None,
        timestep=t_discretized,
        return_all_hidden_states=False,
    )
    build_causal_mask = getattr(action_model, "_build_causal_mask", None)
    if callable(build_causal_mask):
        if N_per_stream is not None:
            action_model.N_per_stream = N_per_stream
        model_kwargs["causal_key_mask"] = build_causal_mask(model_vl_embs)

    model_output = action_model.model(**model_kwargs)

    # ---- 6. 解码为速度场 ----
    pred = action_model.action_decoder(model_output)  # (B, 1+T, D_action) or (B, T, D_action)
    # 取最后 T 个 token（对应 action 部分，跳过 state token）
    v_phi = pred[:, -T:].float()  # fp32 keeps SDE/log-prob math stable

    return v_phi


# ============================================================================
# 核心函数 2: SDE 单步更新 + log_prob 记录
# ============================================================================

def compute_sde_transition_mean(
    v_phi: torch.Tensor,
    a_tau: torch.Tensor,
    tau: torch.Tensor,
    dt: torch.Tensor,
    sigma_tau: torch.Tensor,
    eps_s: float = 1e-6,
) -> torch.Tensor:
    """Recompute the Gaussian transition mean for a recorded SDE step."""
    batch_size = a_tau.shape[0]
    tau = _as_batch_broadcast(tau, batch_size, a_tau.ndim, a_tau.device, a_tau.dtype)
    dt = _as_batch_broadcast(dt, batch_size, a_tau.ndim, a_tau.device, a_tau.dtype)
    sigma_tau = _as_batch_broadcast(
        sigma_tau, batch_size, a_tau.ndim, a_tau.device, a_tau.dtype
    )
    tau_safe = tau.clamp(min=eps_s)
    correction = sigma_tau.square() / (2.0 * tau_safe)
    drift = v_phi + correction * (a_tau + (1.0 - tau) * v_phi)
    return a_tau + drift * dt


def _as_batch_broadcast(
    value: torch.Tensor,
    batch_size: int,
    target_ndim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.numel() == 1:
        value = value.reshape(1).expand(batch_size)
    else:
        value = value.reshape(batch_size)
    return value.view(batch_size, *([1] * (target_ndim - 1)))


def sde_step_with_logprob(
    v_phi: torch.Tensor,
    a_tau: torch.Tensor,
    tau: torch.Tensor,
    dt: torch.Tensor,
    noise_level: float = 0.7,
    eps_s: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> SDEStepRecord:
    """SDE 单步更新，记录用于高斯转移 log_prob 的所有张量。

    数学公式::

        σ_τ = a_noise · √(τ / (1 - τ + ε_s))           # 噪声系数
        μ_φ = v_φ + (σ_τ² / 2τ) · (a_τ + (1-τ)·v_φ)    # 漂移项
        a_next = a_τ + μ_φ·Δτ + σ_τ·√|Δτ|·ε_τ           # SDE 更新

    其中 Δτ < 0 (从 τ=1 噪声端向 τ=0 数据端推进)。

    等价实现::

        std_dev_t = sqrt(σ/(1-σ)) * noise_level          # = σ_τ
        prev_sample_mean = a + [v + σ_τ²/(2τ)·(a + (1-τ)v)] · dt
                         = a + μ_φ · dt
        a_next = prev_sample_mean + σ_τ · √|dt| · ε

    Args:
        v_phi: 速度场 v_φ(a_τ, τ)，shape (B, T, D_action)
        a_tau: 当前动作 a_τ，shape (B, T, D_action)
        tau: 当前时间步 τ，shape (B, 1, 1) 或 (B,) 或标量
        dt: 步长 Δτ = τ_{k+1} - τ_k < 0，shape 同 tau
        noise_level: 噪声系数 a_noise
        eps_s: 防除零小常数 ε_s
        generator: 随机数生成器 (可复现性)

    Returns:
        SDEStepRecord: 包含 a_next, v_phi, sigma_tau, epsilon, prev_sample_mean 等
    """
    # 统一 shape 便于广播
    # tau, dt: → (B, 1, 1) 以便与 (B, T, D) 广播
    if tau.dim() == 0:
        tau = tau.view(1, 1, 1)
    elif tau.dim() == 1:
        tau = tau.view(-1, 1, 1)
    if dt.dim() == 0:
        dt = dt.view(1, 1, 1)
    elif dt.dim() == 1:
        dt = dt.view(-1, 1, 1)

    # ---- 1. 计算噪声系数 σ_τ ----
    # σ_τ = a_noise · √(τ / (1 - τ + ε_s))
    # std_dev_t = sqrt(σ/(1-σ)) * noise_level
    # 当 ε_s→0 时两者等价：√(τ/(1-τ)) = √(τ/(1-τ))
    # Flow-GRPO replaces the singular denominator at tau=1 with the next
    # schedule value. Without this, the first SDE step explodes as 1/sqrt(eps).
    denominator_tau = torch.where(tau >= 1.0 - eps_s, tau + dt, tau)
    denominator_tau = denominator_tau.clamp(min=0.0, max=1.0 - eps_s)
    sigma_tau = noise_level * torch.sqrt(
        tau.clamp(min=0.0) / (1.0 - denominator_tau)
    )  # (B, 1, 1)

    # ---- 2. 高斯转移均值 ----
    prev_sample_mean = compute_sde_transition_mean(
        v_phi=v_phi,
        a_tau=a_tau,
        tau=tau,
        dt=dt,
        sigma_tau=sigma_tau,
        eps_s=eps_s,
    )

    # ---- 4. 采样高斯噪声 ε_τ ----
    sqrt_dt = torch.sqrt(-dt)  # √|Δτ|，因为 dt<0
    epsilon = torch.randn(
        a_tau.shape,
        generator=generator,
        device=a_tau.device,
        dtype=a_tau.dtype,
    )  # (B, T, D)

    # ---- 5. SDE 更新: a_next = prev_sample_mean + σ_τ · √|Δτ| · ε ----
    a_next = prev_sample_mean + sigma_tau * sqrt_dt * epsilon  # (B, T, D)

    return SDEStepRecord(
        v_phi=v_phi,
        a_tau=a_tau,
        a_next=a_next,
        sigma_tau=sigma_tau.squeeze(),   # (B,) 或标量
        epsilon=epsilon,
        prev_sample_mean=prev_sample_mean,
        timestep=tau.squeeze(),          # (B,) 或标量
        dt=dt.squeeze(),
        sqrt_dt=sqrt_dt.squeeze(),
    )


# ============================================================================
# 核心函数 3: ODE 单步更新 (确定性)
# ============================================================================

def ode_step(
    v_phi: torch.Tensor,
    a_tau: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """ODE 单步更新 (确定性 Rectified Flow)。

    a_{τ+Δτ} = a_τ + v_φ · Δτ

    Args:
        v_phi: 速度场，shape (B, T, D)
        a_tau: 当前动作，shape (B, T, D)
        dt: 步长 < 0

    Returns:
        a_next: 下一步动作，shape (B, T, D)
    """
    if dt.dim() == 0:
        dt = dt.view(1, 1, 1)
    elif dt.dim() == 1:
        dt = dt.view(-1, 1, 1)
    return a_tau + v_phi * dt


# ============================================================================
# 核心函数 4: 混合 ODE-SDE 采样主函数
# ============================================================================

def sample_hybrid_sde_action(
    action_model: nn.Module,
    vl_embs: torch.Tensor,
    action_shape: Tuple[int, int],  # (T, D_action)
    state: Optional[torch.Tensor] = None,
    num_inference_steps: int = 4,
    noise_level: float = 0.7,
    sde_window_size: int = 1,
    sde_window_range: Optional[Tuple[int, int]] = None,
    N_per_stream: Optional[int] = None,
    eps_s: float = 1e-6,
    generator: Optional[torch.Generator] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    grad_context: Optional[Any] = None,
) -> RolloutResult:
    """混合 ODE-SDE 采样生成动作块。

    流程：
      1. 从纯噪声 a_1 (τ=1) 出发
      2. K 步推理 (τ: 1→0)，时间步调度 τ_k = 1 - k/K
      3. 在 sde_window_range 范围内随机选 sde_window_size 步用 SDE（注入噪声）
         其余步骤用 ODE（确定性）
      4. SDE 步记录所有张量 (v_φ, a_τ, σ_τ, ε, prev_sample_mean) 供 Step 3 使用

    因果掩码在 compute_action_velocity 中自动注入，确保探索梯度
    100% 聚焦于物理 Z_f + 意图 Z_v，不被语义 Z_x 污染。

    Args:
        action_model: MaskedFlowmatchingActionHead (可训练或 reference)
        vl_embs: Video DiT 第 18 层联合特征 H_video，shape (B, 3N, D)
        action_shape: (T, D_action) 动作块的时序长度和维度
        state: 机器人状态 (B, D_state) 或 None
        num_inference_steps: 推理步数 K (建议 K=4)
        noise_level: SDE 噪声系数 a_noise
        sde_window_size: SDE 步数 (默认 1：只选 1 步注入噪声)
        sde_window_range: SDE 步的选取范围 (start, end)，索引范围 [0, K)。
            None 时默认 (0, K//2)（前半段注入噪声）
        N_per_stream: 每路模态 token 数
        eps_s: 防除零常数
        generator: 随机数生成器
        batch_size: B (若 None 从 vl_embs 推断)
        device / dtype: 张量设备与精度
        grad_context: 梯度上下文管理器 (如 torch.enable_grad())。
            None 时默认 torch.no_grad()（rollout 不需要梯度，梯度在 Step 3 重计算）。

    Returns:
        RolloutResult: 含 final_action, sde_records, all_velocities
    """
    T, D_action = action_shape
    if batch_size is None:
        batch_size = vl_embs.shape[0]
    if device is None:
        device = vl_embs.device

    K = num_inference_steps

    # ---- 1. 时间步调度: τ_0=1, τ_1=1-1/K, ..., τ_K=0 ----
    # sigmas: [1.0, 1-1/K, 1-2/K, ..., 0.0]，共 K+1 个点，K 步
    timesteps = torch.linspace(1.0, 0.0, K + 1, device=device, dtype=dtype)  # (K+1,)

    # ---- 2. 确定 SDE 步索引 ----
    if sde_window_range is None:
        sde_window_range = (0, max(1, K // 2))  # 默认前半段
    window_start, window_end = sde_window_range
    window_end = min(window_end, K)  # 不超过总步数

    # 在窗口范围内随机选 sde_window_size 步。
    # A zero-SDE rollout is also used as an exact ODE parity path.  Do not
    # call randperm in that case: even ``perm[:0]`` advances the RNG and makes
    # its initial noise differ from ActionDiT.predict_action.
    if sde_window_size < 0:
        raise ValueError("sde_window_size must be >= 0")
    window_indices = list(range(window_start, window_end))
    if sde_window_size == 0:
        sde_step_indices = set()
    elif len(window_indices) <= sde_window_size:
        sde_step_indices = set(window_indices)
    else:
        # 用 generator 确保可复现
        if generator is not None:
            perm = torch.randperm(len(window_indices), generator=generator)
        else:
            perm = torch.randperm(len(window_indices))
        sde_step_indices = set(
            window_indices[i] for i in perm[:sde_window_size].tolist()
        )

    logger.debug(f"[Rollout] K={K}, SDE steps={sorted(sde_step_indices)}, "
                 f"window={sde_window_range}")

    # ---- 3. 初始化: a_1 ~ N(0, I) (纯噪声) ----
    a_tau = torch.randn(
        (batch_size, T, D_action),
        generator=generator,
        device=device,
        dtype=dtype,
    )  # (B, T, D)

    sde_records: List[SDEStepRecord] = []
    all_velocities: List[torch.Tensor] = []

    # ---- 4. 梯度上下文 ----
    # Rollout 阶段通常不需要梯度（log_prob 在 Step 3 重新计算时才需要梯度）
    if grad_context is None:
        grad_context = torch.no_grad()

    # ---- 5. K 步推理 ----
    with grad_context:
        for k in range(K):
            tau_k = timesteps[k]       # 当前时间步 τ_k
            tau_k1 = timesteps[k + 1]  # 下一时间步 τ_{k+1}
            dt = tau_k1 - tau_k        # Δτ < 0

            # 5a. 计算动作速度场 v_φ(a_τ, τ_k | H_video, M)
            tau_tensor = tau_k.expand(batch_size) if tau_k.dim() == 0 else tau_k
            dt_tensor = dt.expand(batch_size) if dt.dim() == 0 else dt

            v_phi = compute_action_velocity(
                action_model=action_model,
                a_tau=a_tau,
                tau=tau_tensor,
                vl_embs=vl_embs,
                state=state,
                N_per_stream=N_per_stream,
            )  # (B, T, D)
            all_velocities.append(v_phi)

            # 5b. SDE 步 or ODE 步
            if k in sde_step_indices:
                # ---- SDE 步: 注入噪声 + 记录 ----
                record = sde_step_with_logprob(
                    v_phi=v_phi,
                    a_tau=a_tau,
                    tau=tau_tensor,
                    dt=dt_tensor,
                    noise_level=noise_level,
                    eps_s=eps_s,
                    generator=generator,
                )
                a_tau = record.a_next
                sde_records.append(record)
            else:
                # ---- ODE 步: 确定性更新 ----
                a_tau = ode_step(v_phi, a_tau, dt_tensor)

    # ---- 6. 返回结果 ----
    result = RolloutResult(
        final_action=a_tau,           # a_0 (去噪后的动作块)
        sde_records=sde_records,      # SDE 步记录 (用于 log_prob)
        all_velocities=all_velocities,  # 所有步速度场 (用于 KL)
        timesteps_schedule=timesteps,
        sde_step_indices=sorted(sde_step_indices),
    )

    logger.debug(f"[Rollout] Done. SDE records: {len(sde_records)}, "
                 f"final_action shape: {tuple(a_tau.shape)}")

    return result


# ============================================================================
# 辅助: 组采样 (GRPO 需要同一 prompt 的 G 条并行轨迹)
# ============================================================================

def sample_group_hybrid_sde(
    action_model: nn.Module,
    vl_embs_single: torch.Tensor,
    state_single: Optional[torch.Tensor],
    action_shape: Tuple[int, int],
    group_size: int = 8,
    **kwargs,
) -> RolloutResult:
    """为 GRPO 采样 G 条并行动作轨迹 (组采样)。

    将同一观测条件 (vl_embs, state) 复制 G 份，并行采样 G 条不同的动作轨迹。
    不同的轨迹通过不同的随机噪声初始化和 SDE 噪声注入产生差异。

    Args:
        action_model: Action DiT 模型
        vl_embs_single: 单条观测的 Video DiT 特征，shape (3N, D) 或 (1, 3N, D)
        state_single: 单条机器人状态，shape (D_state,) 或 (1, D_state)
        action_shape: (T, D_action)
        group_size: GRPO 组大小 G
        **kwargs: 传给 sample_hybrid_sde_action 的其他参数

    Returns:
        RolloutResult: batch 维=G 的采样结果 (GRPO 直接用 batch 形式处理)
    """
    # 扩展到 G 份
    if vl_embs_single.dim() == 2:
        vl_embs_single = vl_embs_single.unsqueeze(0)  # (1, 3N, D)
    vl_embs_group = vl_embs_single.expand(group_size, -1, -1)  # (G, 3N, D)

    state_group = None
    if state_single is not None:
        if state_single.dim() == 1:
            state_single = state_single.unsqueeze(0)
        state_group = state_single.expand(group_size, -1)  # (G, D_state)

    # 一次性并行采样 G 条轨迹
    result = sample_hybrid_sde_action(
        action_model=action_model,
        vl_embs=vl_embs_group,
        action_shape=action_shape,
        state=state_group,
        batch_size=group_size,
        **kwargs,
    )

    # 将 batch 维度拆分为 G 条独立结果
    # 但通常 GRPO 直接用 batch 形式处理，不需要拆分
    # 这里返回单条 RolloutResult（batch 维=G），供 Step 3 直接使用
    return result
