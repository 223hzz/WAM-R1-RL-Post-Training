# -*- coding: utf-8 -*-
"""
Step 3: 数学计算模块 (Math Utils)
====================================

实现离散 SDE 高斯转移概率与 Action-RL 损失。

核心数学 (GRPO 组优势与策略损失)：

1. Euler-Maruyama 高斯转移概率::
     log π_φ = log N(a_next; mean_φ, σ_k·√|Δτ|)

2. GRPO 组优势 (无 Critic)::
     Â_i = (R_i - mean({R_j})) / (std({R_j}) + 1e-8)

     不需要价值网络，仅用组内相对排名。

3. 策略比值与 Clipping Loss::
     r_k = exp(log π_new - log π_old)
     L_policy = -mean(min(r·Â, clip(r, 1-ε, 1+ε)·Â))

4. Reference 转移均值约束::
     L_kl = ||mean_new - mean_ref||² / (2·σ²)

参考实现：
  - 公开 GRPO 实现 (GRPO loss)
  - 公开 Flow-SDE 实现 (log_prob)
  - 公开 GRPO 实现 (组优势计算)

梯度流设计：
  - Rollout 阶段 (no_grad): 记录 a_τ, a_next, mean_old, σ, τ
  - Loss 阶段 (enable_grad):
      v_new = compute_action_velocity(new_model, a_τ, τ, ...)  ← 有梯度
      mean_new = compute_sde_transition_mean(v_new, ...)       ← 有梯度
      log_prob_new = log N(a_next; mean_new, std)              ← 有梯度
      log_prob_old = log N(a_next; mean_old, std)              ← 无梯度
      ratio = exp(log_prob_new - log_prob_old)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dit4dit_rl_trainer.sde_rollout import (
    SDEStepRecord,
    RolloutResult,
    compute_action_velocity,
    compute_sde_transition_mean,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 1. 概率计算
# ============================================================================

def compute_girsanov_log_prob(
    v_phi: torch.Tensor,
    epsilon: torch.Tensor,
    sigma_tau: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Legacy continuous-time Girsanov diagnostic.

    The trainer uses :func:`compute_gaussian_transition_log_prob`, which
    matches the actual discrete Euler-Maruyama transition sampled in rollout.

    公式::

        log π_φ = (v_φ^T · ε / σ) · √|Δτ| - (1/2) · (||v_φ||² / σ²) · |Δτ|

    数学推导：
        SDE: da = μ·dτ + σ·dW
        Girsanov 对数似然比 (策略 vs 纯扩散参考测度):
        log(dP/dQ) = ∫(μ/σ)dW - (1/2)∫(μ/σ)²dτ

        在 flow matching SDE 中，有效漂移 μ 的 Girsanov 项中
        修正项 (σ²/2τ)(a+(1-τ)v) 与参考测度相同而消去，
        只剩 v_φ 的贡献 → 上述闭式解。

    Args:
        v_phi: 速度场 v_φ，shape (B, T, D_action)。
            可以是有梯度的张量 (new model) 或 detach 的张量 (old model)。
        epsilon: SDE 注入的高斯噪声 ε，shape (B, T, D_action)。
        sigma_tau: 噪声系数 σ_τ，shape (B,) 或标量。
        dt: 步长 Δτ < 0，shape (B,) 或标量。

    Returns:
        log_prob: 对数概率，shape (B,)。每个 batch 元素一个标量。
    """
    # ---- 确保 sigma_tau, dt 可广播 ----
    # v_phi, epsilon: (B, T, D)，在 T, D 维度上求和
    reduce_dims = tuple(range(1, v_phi.dim()))  # (1, 2) → 对 T, D 求和

    # sigma_tau, dt: → (B,) 或标量
    sigma = sigma_tau
    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)  # (1,)

    dt_val = dt
    if dt_val.dim() == 0:
        dt_val = dt_val.unsqueeze(0)

    # |Δτ| 和 √|Δτ|（Δτ < 0，取绝对值）
    abs_dt = (-dt_val).abs()  # (B,) 或 (1,)
    sqrt_abs_dt = torch.sqrt(abs_dt + 1e-12)  # √|Δτ|，加 1e-12 防止梯度爆炸

    # σ²
    sigma_sq = sigma ** 2  # (B,) 或 (1,)

    # ---- v_φ^T · ε: 内积，在 T, D 维度求和 ----
    v_dot_eps = (v_phi * epsilon).sum(dim=reduce_dims)  # (B,)

    # ---- ||v_φ||²: L2 范数平方 ----
    v_norm_sq = (v_phi ** 2).sum(dim=reduce_dims)  # (B,)

    # ---- log π = (v·ε/σ)·√|Δτ| - (1/2)·(||v||²/σ²)·|Δτ| ----
    term1 = (v_dot_eps / sigma) * sqrt_abs_dt        # (B,)
    term2 = 0.5 * (v_norm_sq / sigma_sq) * abs_dt     # (B,)

    log_prob = term1 - term2  # (B,)

    return log_prob


def compute_log_prob_from_record(
    record: SDEStepRecord,
    v_phi_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """从 SDEStepRecord 计算 log_prob。

    便捷封装：直接从 rollout 记录中提取张量计算 Girsanov log_prob。

    Args:
        record: SDE 步记录 (含 v_phi, epsilon, sigma_tau, dt)
        v_phi_override: 可选的速度场覆盖。
            - None: 用 record 中记录的 v_phi (old model，无梯度)
            - 传入新 v_phi: 用新速度场计算 (new model，有梯度)

    Returns:
        log_prob: (B,)
    """
    v = v_phi_override if v_phi_override is not None else record.v_phi
    return compute_girsanov_log_prob(
        v_phi=v,
        epsilon=record.epsilon,
        sigma_tau=record.sigma_tau,
        dt=record.dt,
    )


def compute_gaussian_transition_log_prob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Log probability of an SDE transition, reduced per trajectory.

    This matches Flow-GRPO's ``sde_step_with_logprob`` implementation. The
    calculation is forced to fp32 because bf16 can overflow in the squared
    residual and variance terms.
    """
    sample_f = sample.detach().float()
    mean_f = mean.float()
    std_f = torch.as_tensor(std, device=mean.device, dtype=torch.float32)
    if std_f.numel() == 1:
        std_f = std_f.reshape(1).expand(sample.shape[0])
    else:
        std_f = std_f.reshape(sample.shape[0])
    std_f = std_f.view(sample.shape[0], *([1] * (sample.ndim - 1))).clamp_min(eps)
    log_prob = (
        -((sample_f - mean_f).square()) / (2.0 * std_f.square())
        - torch.log(std_f)
        - 0.5 * math.log(2.0 * math.pi)
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


# ============================================================================
# 2. GRPO 组优势
# ============================================================================

def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
    global_std: bool = False,
    reward_history: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """计算 GRPO 组优势 Â_i (无 Critic)。

    公式::

        Â_i = (R_i - mean({R_j}_{j=1}^G)) / (std({R_j}_{j=1}^G) + ε)

    Args:
        rewards: 组内奖励 R_i，shape (G,)。
        eps: 防除零小常数。
        global_std: 是否使用全局标准差 (跨 batch 的历史 std) 而非当前组的 std。
            全局奖励标准差选项。
        reward_history: 全局奖励历史列表 (global_std=True 时使用)。

    Returns:
        advantages: 组优势 Â_i，shape (G,)。
    """
    if rewards.dim() > 1:
        rewards = rewards.squeeze()

    group_mean = rewards.mean()
    if global_std and reward_history is not None and len(reward_history) > 1:
        # 全局标准差：用所有历史 batch 的奖励计算 std
        all_rewards = torch.cat([r.squeeze() for r in reward_history])
        group_std = all_rewards.std(unbiased=False)
    else:
        # 当前组标准差
        # Use torch.std's default
        # sample standard deviation within each prompt/task group.
        group_std = rewards.std(unbiased=True) if rewards.numel() > 1 else rewards.new_zeros(())

    advantages = (rewards - group_mean) / (group_std + eps)
    return advantages


# ============================================================================
# 3. KL 散度 (速度场 L2 距离)
# ============================================================================

def compute_kl_loss(
    mean_new_list: List[torch.Tensor],
    mean_ref_list: List[torch.Tensor],
    noise_scale_list: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute Flow-GRPO's reference transition-mean penalty.

    公式::

        L_kl = Σ_k ||mean_new - mean_ref||² / (2·sigma_k²)

    等价实现::
        kl_loss = ((prev_sample_mean - prev_sample_mean_ref)²).mean() / (2·std_dev_t²)

    其中 prev_sample_mean ≈ a + v·dt，所以
    (prev_sample_mean - prev_sample_mean_ref) ≈ (v_new - v_ref)·dt
    → kl_loss ≈ (v_new - v_ref)²·dt² / (2σ²)

    简化后等价于 ||v_new - v_ref||² / (2σ²) 的形式。

    Args:
        mean_new_list: 新策略的 SDE 转移均值 (有梯度)
        mean_ref_list: reference 策略的 SDE 转移均值 (无梯度)
        noise_scale_list: 各 SDE 步 transition standard deviation
            (sigma_t * sqrt(|dt|))。None 时用 1.0。

    Returns:
        kl_loss: 标量 KL 损失。
    """
    assert len(mean_new_list) == len(mean_ref_list), \
        f"new ({len(mean_new_list)}) and ref ({len(mean_ref_list)}) length mismatch"

    kl_total = torch.tensor(
        0.0, device=mean_new_list[0].device, dtype=mean_new_list[0].dtype
    )
    n_steps = len(mean_new_list)

    for k, (mean_new, mean_ref) in enumerate(zip(mean_new_list, mean_ref_list)):
        diff_sq = (mean_new.float() - mean_ref.float()).square()
        if noise_scale_list is not None and k < len(noise_scale_list):
            std = noise_scale_list[k].float()
            if std.numel() == 1:
                std = std.reshape(1).expand(diff_sq.shape[0])
            std = std.reshape(diff_sq.shape[0], *([1] * (diff_sq.dim() - 1)))
            kl_step = diff_sq / (2.0 * std.square().clamp_min(1e-12))
        else:
            kl_step = diff_sq / 2.0
        kl_total = kl_total + kl_step.mean()

    return kl_total / n_steps


# ============================================================================
# 4. Action-RL 损失 (主函数)
# ============================================================================

def compute_grpo_loss(
    action_model: nn.Module,
    rollout_result: RolloutResult,
    vl_embs: torch.Tensor,
    rewards: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    N_per_stream: Optional[int] = None,
    ref_velocities: Optional[List[torch.Tensor]] = None,
    clip_range: float = 2e-6,
    beta_kl: float = 0.08,
    adv_clip_max: float = 5.0,
    global_std: bool = False,
    reward_history: Optional[List[torch.Tensor]] = None,
    precomputed_advantages: Optional[torch.Tensor] = None,
    log_ratio_clip: float = 20.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算 Action-RL 损失。

    完整流程:
      1. 用 new model 重新计算各 SDE 步的速度场 v_new (有梯度)
      2. 用新策略转移均值计算离散高斯 log_prob_new
      3. 用 rollout 行为策略转移均值计算 log_prob_old (无梯度)
      4. ratio = exp(log_prob_new - log_prob_old)
      5. 优势 Â = (R - mean) / (std + ε)
      6. L_policy = -mean(min(r·Â, clip(r)·Â))
      7. L_kl = ||mean_new - mean_ref||² / (2σ²)
      8. L_total = L_policy + β·L_kl

    Args:
        action_model: 可训练的 Action DiT (new model)
        rollout_result: Rollout 输出 (含 SDE 步记录)
        vl_embs: Video DiT 条件特征；原版 (G,S,D)，三流版 (G,3N,D)
        rewards: 组内奖励, shape (G,)
        state: 机器人状态
        N_per_stream: 每路模态 token 数
        ref_velocities: reference 模型各步速度场 (None 时用 record.v_phi)
        clip_range: PPO clip 范围 ε
        beta_kl: KL 惩罚系数 β
        adv_clip_max: 优势裁剪最大值 (防异常值)
        global_std: 是否用全局标准差
        reward_history: 全局奖励历史

    Returns:
        loss: 标量总损失 (有梯度)
        metrics: 训练指标字典
    """
    G = rewards.shape[0]
    sde_records = rollout_result.sde_records
    K_sde = len(sde_records)  # SDE 步数

    if K_sde == 0:
        logger.warning("[GRPO] No SDE steps in rollout, returning zero loss")
        zero = torch.tensor(0.0, device=vl_embs.device, requires_grad=True)
        return zero, {
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "ratio_mean": 1.0,
            "ratio_sq_mean": 1.0,
            "ratio_std": 0.0,
            "ratio_min": 1.0,
            "ratio_max": 1.0,
            "clip_fraction": 0.0,
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
            "log_prob_new_mean": 0.0,
            "log_prob_old_mean": 0.0,
            "log_ratio_mean": 0.0,
            "log_ratio_sq_mean": 0.0,
            "log_ratio_std": 0.0,
            "log_ratio_abs_max": 0.0,
        }

    # ---- 1. 计算组优势 ----
    if precomputed_advantages is None:
        advantages = compute_group_advantages(
            rewards, eps=1e-8, global_std=global_std, reward_history=reward_history
        )  # (G,)
    else:
        advantages = precomputed_advantages.to(
            device=rewards.device, dtype=rewards.dtype
        ).reshape(-1)
        if advantages.shape != rewards.reshape(-1).shape:
            raise ValueError(
                "precomputed_advantages must match rewards; "
                f"got {tuple(advantages.shape)} and {tuple(rewards.shape)}"
            )
    # 优势裁剪 (防异常值导致训练崩溃)
    advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

    # ---- 2. 用 new model 重新计算 SDE 步的速度场 (有梯度) ----
    log_probs_new = []
    log_probs_old = []
    mean_new_list = []
    mean_ref_list = []
    noise_scale_list = []

    # ref_velocities 迭代器 (不修改原列表)
    ref_vel_iter = iter(ref_velocities) if ref_velocities is not None else None

    for record in sde_records:
        # 2a. new model 速度场 (有梯度)
        v_new = compute_action_velocity(
            action_model=action_model,
            a_tau=record.a_tau,        # rollout 时记录的 a_τ (固定)
            tau=record.timestep,        # rollout 时记录的 τ (固定)
            vl_embs=vl_embs,
            state=state,
            N_per_stream=N_per_stream,
        )  # (G, T, D)
        # 2b. reference 速度场 (无梯度)
        if ref_vel_iter is not None:
            v_ref = next(ref_vel_iter)  # 从迭代器取出 (不修改原列表)
        else:
            v_ref = record.v_phi.detach()  # 用 rollout 时记录的 old v_phi
        # 2c. Recompute Gaussian transition means on the recorded trajectory.
        mean_new = compute_sde_transition_mean(
            v_new, record.a_tau, record.timestep, record.dt, record.sigma_tau
        )
        mean_ref = compute_sde_transition_mean(
            v_ref, record.a_tau, record.timestep, record.dt, record.sigma_tau
        )
        transition_std = record.sigma_tau * record.sqrt_dt

        log_prob_new = compute_gaussian_transition_log_prob(
            sample=record.a_next,
            mean=mean_new,
            std=transition_std,
        )
        log_probs_new.append(log_prob_new)

        # PPO/GRPO denominator must be the rollout behavior policy, not the
        # frozen reference policy used by the KL regularizer.
        log_prob_old = compute_gaussian_transition_log_prob(
            sample=record.a_next,
            mean=record.prev_sample_mean.detach(),
            std=transition_std,
        )
        log_probs_old.append(log_prob_old)

        mean_new_list.append(mean_new)
        mean_ref_list.append(mean_ref)
        # KL must use the same Gaussian transition scale as the transition
        # log-probability above, i.e. sigma_tau * sqrt(|dt|).
        noise_scale_list.append(record.sigma_tau * record.sqrt_dt)

    # 堆叠: (G, K_sde)
    log_probs_new = torch.stack(log_probs_new, dim=1)  # (G, K_sde)
    log_probs_old = torch.stack(log_probs_old, dim=1)  # (G, K_sde)

    # ---- 3. 策略比值 ----
    log_ratio = (log_probs_new - log_probs_old).clamp(
        min=-log_ratio_clip, max=log_ratio_clip
    )
    ratio = torch.exp(log_ratio)  # (G, K_sde)

    # ---- 4. Clipped policy loss ----
    # 优势扩展到 K_sde 步 (每步共享同一组优势)
    adv_expanded = advantages.unsqueeze(1).expand(-1, K_sde)  # (G, K_sde)

    unclipped_loss = -adv_expanded * ratio
    clipped_loss = -adv_expanded * torch.clamp(
        ratio, 1.0 - clip_range, 1.0 + clip_range
    )
    policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()

    # ---- 5. KL 散度损失 ----
    kl_loss = compute_kl_loss(
        mean_new_list, mean_ref_list, noise_scale_list
    )

    # ---- 6. 总损失 ----
    loss = policy_loss + beta_kl * kl_loss

    # ---- 7. 指标 ----
    with torch.no_grad():
        # Variance is around 1e-12 in these runs. Use FP64 diagnostics so
        # E[r^2] - E[r]^2 remains measurable when ratios are close to one.
        ratio_metrics = ratio.double()
        log_ratio_metrics = log_ratio.double()
        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "ratio_mean": ratio_metrics.mean().item(),
            "ratio_sq_mean": ratio_metrics.square().mean().item(),
            "ratio_std": ratio_metrics.std(unbiased=False).item(),
            "ratio_min": ratio_metrics.min().item(),
            "ratio_max": ratio_metrics.max().item(),
            "clip_fraction": (
                (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
            ).float().mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
            "log_prob_new_mean": log_probs_new.mean().item(),
            "log_prob_old_mean": log_probs_old.mean().item(),
            "log_ratio_mean": log_ratio_metrics.mean().item(),
            "log_ratio_sq_mean": log_ratio_metrics.square().mean().item(),
            "log_ratio_std": log_ratio_metrics.std(unbiased=False).item(),
            "log_ratio_abs_max": log_ratio_metrics.abs().max().item(),
        }

    return loss, metrics


# ============================================================================
# 5. SFT 损失 (阶段1 冷启动)
# ============================================================================

def compute_sft_loss(
    action_model: nn.Module,
    vl_embs: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    N_per_stream: Optional[int] = None,
) -> torch.Tensor:
    """Few-shot SFT 损失 (阶段1 冷启动)。

    用标准的 Flow Matching 监督损失 (MSE on velocity field) 冷启动 Action DiT。
    目标：把初始成功率从 0% 拉到 40%~50%。

    公式 (Rectified Flow 监督损失)::

        v_target = noise - action          (GT 速度场)
        v_pred = action_model(a_τ, τ)      (预测速度场)
        L_sft = MSE(v_pred, v_target)

    与 action_model.forward() 的训练逻辑一致，但返回标量 loss 供外部优化器使用。

    Args:
        action_model: 可训练 Action DiT
        vl_embs: (B, 3N, D) Video DiT 特征
        target_actions: (B, T, D_action) 专家动作
        action_mask: (B, T, D_action) 动作掩码
        state: 机器人状态
        N_per_stream: 每路模态 token 数

    Returns:
        loss: 标量 SFT 损失 (有梯度)
    """
    B, T, D_action = target_actions.shape
    device = target_actions.device
    dtype = target_actions.dtype

    # 采样时间步 τ ~ Beta(α, β) (与 action_model.sample_time 一致)
    tau = action_model.sample_time(B, device=device, dtype=dtype)  # (B,)
    tau_expanded = tau[:, None, None]  # (B, 1, 1)

    # 采样噪声
    noise = torch.randn_like(target_actions)

    # 构建加噪动作: a_τ = (1-τ)·action + τ·noise
    noisy_trajectory = (1.0 - tau_expanded) * target_actions + tau_expanded * noise

    # GT 速度场: v = noise - action
    velocity_target = noise - target_actions

    # 预测速度场 (带因果掩码)
    v_pred = compute_action_velocity(
        action_model=action_model,
        a_tau=noisy_trajectory,
        tau=tau,
        vl_embs=vl_embs,
        state=state,
        N_per_stream=N_per_stream,
    )  # (B, T, D_action)

    # MSE 损失 (带 mask)
    diff = (v_pred - velocity_target) ** 2  # (B, T, D)
    if action_mask is not None:
        diff = diff * action_mask
        loss = diff.sum() / (action_mask.sum() + 1e-8)
    else:
        loss = diff.mean()

    return loss


def compute_weighted_sft_loss(
    action_model: nn.Module,
    vl_embs: torch.Tensor,
    target_actions: torch.Tensor,
    sample_weights: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    N_per_stream: Optional[int] = None,
) -> torch.Tensor:
    """Flow Matching loss with one non-negative weight per action chunk."""
    batch_size = target_actions.shape[0]
    if sample_weights.numel() != batch_size:
        raise ValueError(
            "sample_weights must contain one value per batch item, got "
            f"{sample_weights.numel()} for batch size {batch_size}"
        )
    weights = sample_weights.to(
        device=target_actions.device, dtype=target_actions.dtype
    ).reshape(batch_size)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weights must be finite and non-negative")

    tau = action_model.sample_time(
        batch_size, device=target_actions.device, dtype=target_actions.dtype
    )
    tau_expanded = tau[:, None, None]
    noise = torch.randn_like(target_actions)
    noisy_trajectory = (1.0 - tau_expanded) * target_actions + tau_expanded * noise
    velocity_target = noise - target_actions
    velocity_prediction = compute_action_velocity(
        action_model=action_model,
        a_tau=noisy_trajectory,
        tau=tau,
        vl_embs=vl_embs,
        state=state,
        N_per_stream=N_per_stream,
    )

    squared_error = (velocity_prediction - velocity_target).square()
    if action_mask is None:
        per_sample_loss = squared_error.flatten(1).mean(dim=1)
    else:
        mask = action_mask.to(squared_error)
        per_sample_loss = (
            (squared_error * mask).flatten(1).sum(dim=1)
            / mask.flatten(1).sum(dim=1).clamp_min(1e-8)
        )
    return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)
