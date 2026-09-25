# -*- coding: utf-8 -*-
"""
Step 3: Math Utils
====================================

Implements the discrete-SDE Gaussian transition probability and the Action-RL loss.

Core math (GRPO group advantage and policy loss):

1. Euler-Maruyama Gaussian transition probability::
     log π_φ = log N(a_next; mean_φ, σ_k·√|Δτ|)

2. GRPO group advantage (critic-free)::
     Â_i = (R_i - mean({R_j})) / (std({R_j}) + 1e-8)

     No value network needed; only intra-group relative ranking.

3. Policy ratio and clipping loss::
     r_k = exp(log π_new - log π_old)
     L_policy = -mean(min(r·Â, clip(r, 1-ε, 1+ε)·Â))

4. Reference transition-mean constraint::
     L_kl = ||mean_new - mean_ref||² / (2·σ²)

References:
  - public GRPO implementations (GRPO loss)
  - public Flow-SDE implementations (log_prob)
  - public GRPO implementations (group advantage computation)

Gradient-flow design:
  - rollout phase (no_grad): records a_tau, a_next, mean_old, sigma, tau
  - loss phase (enable_grad):
      v_new = compute_action_velocity(new_model, a_tau, tau, ...)  <- has grad
      mean_new = compute_sde_transition_mean(v_new, ...)           <- has grad
      log_prob_new = log N(a_next; mean_new, std)                  <- has grad
      log_prob_old = log N(a_next; mean_old, std)                  <- no grad
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
# 1. Probability computation
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

    Formula::

        log π_φ = (v_φ^T · ε / σ) · √|Δτ| - (1/2) · (||v_φ||² / σ²) · |Δτ|

    Derivation:
        SDE: da = μ·dτ + σ·dW
        Girsanov log-likelihood ratio (policy vs. pure-diffusion reference measure):
        log(dP/dQ) = ∫(μ/σ)dW - (1/2)∫(μ/σ)²dτ

        In the flow-matching SDE the Girsanov term of the effective drift mu carries
        a correction (sigma^2/2tau)(a+(1-tau)v) that is identical under the reference
        measure and therefore cancels, leaving only v_phi -> the closed form above.

    Args:
        v_phi: velocity field v_phi, shape (B, T, D_action).
            May carry grad (new model) or be detached (old model).
        epsilon: Gaussian noise eps injected by the SDE, shape (B, T, D_action).
        sigma_tau: noise coefficient sigma_tau, shape (B,) or scalar.
        dt: step size delta_tau < 0, shape (B,) or scalar.

    Returns:
        log_prob: log probability, shape (B,). One scalar per batch element.
    """
    # ---- make sigma_tau and dt broadcastable ----
    # v_phi, epsilon: (B, T, D); reduce over the T and D dims
    reduce_dims = tuple(range(1, v_phi.dim()))  # (1, 2) -> reduce over T and D

    # sigma_tau, dt -> (B,) or scalar
    sigma = sigma_tau
    if sigma.dim() == 0:
        sigma = sigma.unsqueeze(0)  # (1,)

    dt_val = dt
    if dt_val.dim() == 0:
        dt_val = dt_val.unsqueeze(0)

    # |delta_tau| and sqrt(|delta_tau|); delta_tau < 0, so take the absolute value
    abs_dt = (-dt_val).abs()  # (B,) or (1,)
    sqrt_abs_dt = torch.sqrt(abs_dt + 1e-12)  # sqrt(|delta_tau|); 1e-12 prevents gradient blow-up

    # sigma^2
    sigma_sq = sigma ** 2  # (B,) or (1,)

    # ---- v_phi^T . eps: inner product, reduced over T and D ----
    v_dot_eps = (v_phi * epsilon).sum(dim=reduce_dims)  # (B,)

    # ---- ||v_phi||^2: squared L2 norm ----
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
    """Compute log_prob from an SDEStepRecord.

    Convenience wrapper: pull the tensors straight out of a rollout record and compute the Girsanov log_prob.

    Args:
        record: SDE step record (holds v_phi, epsilon, sigma_tau, dt)
        v_phi_override: optional velocity-field override.
            - None: use the v_phi stored in the record (old model, no grad)
            - a new v_phi: compute with the new velocity field (new model, has grad)

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
# 2. GRPO group advantage
# ============================================================================

def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
    global_std: bool = False,
    reward_history: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute the GRPO group advantage A_i (critic-free).

    Formula::

        Â_i = (R_i - mean({R_j}_{j=1}^G)) / (std({R_j}_{j=1}^G) + ε)

    Args:
        rewards: intra-group rewards R_i, shape (G,).
        eps: small constant guarding against division by zero.
        global_std: use a global standard deviation (historical, across batches) instead of the current group's.
        reward_history: global reward history, used when global_std=True.

    Returns:
        advantages: group advantage A_i, shape (G,).
    """
    if rewards.dim() > 1:
        rewards = rewards.squeeze()

    group_mean = rewards.mean()
    if global_std and reward_history is not None and len(reward_history) > 1:
        # Global std: compute std over the rewards of all historical batches
        all_rewards = torch.cat([r.squeeze() for r in reward_history])
        group_std = all_rewards.std(unbiased=False)
    else:
        # Current-group std
        # Use torch.std's default
        # sample standard deviation within each prompt/task group.
        group_std = rewards.std(unbiased=True) if rewards.numel() > 1 else rewards.new_zeros(())

    advantages = (rewards - group_mean) / (group_std + eps)
    return advantages


# ============================================================================
# 3. KL divergence (L2 distance between velocity fields)
# ============================================================================

def compute_kl_loss(
    mean_new_list: List[torch.Tensor],
    mean_ref_list: List[torch.Tensor],
    noise_scale_list: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute Flow-GRPO's reference transition-mean penalty.

    Formula::

        L_kl = Σ_k ||mean_new - mean_ref||² / (2·sigma_k²)

    Equivalent implementation::
        kl_loss = ((prev_sample_mean - prev_sample_mean_ref)²).mean() / (2·std_dev_t²)

    Since prev_sample_mean ~= a + v*dt,
    (prev_sample_mean - prev_sample_mean_ref) ≈ (v_new - v_ref)·dt
    → kl_loss ≈ (v_new - v_ref)²·dt² / (2σ²)

    this reduces to the form ||v_new - v_ref||^2 / (2*sigma^2).

    Args:
        mean_new_list: SDE transition means of the new policy (has grad)
        mean_ref_list: SDE transition means of the reference policy (no grad)
        noise_scale_list: per-SDE-step transition standard deviation
            (sigma_t * sqrt(|dt|)). Falls back to 1.0 when None.

    Returns:
        kl_loss: scalar KL loss.
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
# 4. Action-RL loss (main function)
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
    """Compute the Action-RL loss.

    Full pipeline:
      1. recompute the per-step velocity field v_new with the new model (has grad)
      2. compute the discrete-Gaussian log_prob_new from the new policy's transition means
      3. compute log_prob_old from the rollout behaviour policy's transition means (no grad)
      4. ratio = exp(log_prob_new - log_prob_old)
      5. advantage A = (R - mean) / (std + eps)
      6. L_policy = -mean(min(r·Â, clip(r)·Â))
      7. L_kl = ||mean_new - mean_ref||² / (2σ²)
      8. L_total = L_policy + β·L_kl

    Args:
        action_model: trainable Action DiT (new model)
        rollout_result: rollout output (holds the SDE step records)
        vl_embs: Video DiT conditioning features; original (G,S,D), three-stream variant (G,3N,D)
        rewards: intra-group rewards, shape (G,)
        state: robot state
        N_per_stream: number of tokens per modality stream
        ref_velocities: per-step velocity fields of the reference model (falls back to record.v_phi when None)
        clip_range: PPO clip range eps
        beta_kl: KL penalty coefficient beta
        adv_clip_max: maximum advantage magnitude, guarding against outliers
        global_std: whether to use the global standard deviation
        reward_history: global reward history

    Returns:
        loss: scalar total loss (has grad)
        metrics: dictionary of training metrics
    """
    G = rewards.shape[0]
    sde_records = rollout_result.sde_records
    K_sde = len(sde_records)  # number of SDE steps

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

    # ---- 1. group advantage ----
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
    # Advantage clipping (prevents outlier-driven training collapse)
    advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

    # ---- 2. recompute the per-step velocity fields with the new model (has grad) ----
    log_probs_new = []
    log_probs_old = []
    mean_new_list = []
    mean_ref_list = []
    noise_scale_list = []

    # Iterator over ref_velocities (leaves the original list untouched)
    ref_vel_iter = iter(ref_velocities) if ref_velocities is not None else None

    for record in sde_records:
        # 2a. new-model velocity field (has grad)
        v_new = compute_action_velocity(
            action_model=action_model,
            a_tau=record.a_tau,        # a_tau recorded during rollout (frozen)
            tau=record.timestep,       # tau recorded during rollout (frozen)
            vl_embs=vl_embs,
            state=state,
            N_per_stream=N_per_stream,
        )  # (G, T, D)
        # 2b. reference velocity field (no grad)
        if ref_vel_iter is not None:
            v_ref = next(ref_vel_iter)  # take from the iterator, leaving the list untouched
        else:
            v_ref = record.v_phi.detach()  # fall back to the old v_phi recorded during rollout
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

    # Stack: (G, K_sde)
    log_probs_new = torch.stack(log_probs_new, dim=1)  # (G, K_sde)
    log_probs_old = torch.stack(log_probs_old, dim=1)  # (G, K_sde)

    # ---- 3. policy ratio ----
    log_ratio = (log_probs_new - log_probs_old).clamp(
        min=-log_ratio_clip, max=log_ratio_clip
    )
    ratio = torch.exp(log_ratio)  # (G, K_sde)

    # ---- 4. Clipped policy loss ----
    # Broadcast the advantage over the K_sde steps (all steps share one group advantage)
    adv_expanded = advantages.unsqueeze(1).expand(-1, K_sde)  # (G, K_sde)

    unclipped_loss = -adv_expanded * ratio
    clipped_loss = -adv_expanded * torch.clamp(
        ratio, 1.0 - clip_range, 1.0 + clip_range
    )
    policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()

    # ---- 5. KL divergence loss ----
    kl_loss = compute_kl_loss(
        mean_new_list, mean_ref_list, noise_scale_list
    )

    # ---- 6. total loss ----
    loss = policy_loss + beta_kl * kl_loss

    # ---- 7. metrics ----
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
# 5. SFT loss (Stage 1 cold start)
# ============================================================================

def compute_sft_loss(
    action_model: nn.Module,
    vl_embs: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    N_per_stream: Optional[int] = None,
) -> torch.Tensor:
    """Few-shot SFT loss (Stage 1 cold start).

    Cold-starts the Action DiT with the standard flow-matching supervised loss (MSE on the velocity field).
    Goal: lift the initial success rate from 0% to roughly 40-50%.

    Formula (rectified-flow supervised loss)::

        v_target = noise - action          (ground-truth velocity field)
        v_pred = action_model(a_tau, tau)  (predicted velocity field)
        L_sft = MSE(v_pred, v_target)

    Mirrors the training logic in action_model.forward() but returns a scalar loss for an external optimizer.

    Args:
        action_model: trainable Action DiT
        vl_embs: (B, 3N, D) Video DiT features
        target_actions: (B, T, D_action) expert actions
        action_mask: (B, T, D_action) action mask
        state: robot state
        N_per_stream: number of tokens per modality stream

    Returns:
        loss: scalar SFT loss (has grad)
    """
    B, T, D_action = target_actions.shape
    device = target_actions.device
    dtype = target_actions.dtype

    # Sample the timestep tau ~ Beta(alpha, beta), matching action_model.sample_time
    tau = action_model.sample_time(B, device=device, dtype=dtype)  # (B,)
    tau_expanded = tau[:, None, None]  # (B, 1, 1)

    # Sample noise
    noise = torch.randn_like(target_actions)

    # Build the noised action: a_tau = (1-tau)*action + tau*noise
    noisy_trajectory = (1.0 - tau_expanded) * target_actions + tau_expanded * noise

    # Ground-truth velocity field: v = noise - action
    velocity_target = noise - target_actions

    # Predicted velocity field (with the causal mask)
    v_pred = compute_action_velocity(
        action_model=action_model,
        a_tau=noisy_trajectory,
        tau=tau,
        vl_embs=vl_embs,
        state=state,
        N_per_stream=N_per_stream,
    )  # (B, T, D_action)

    # MSE loss (masked)
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
