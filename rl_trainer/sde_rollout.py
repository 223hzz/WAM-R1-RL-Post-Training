# -*- coding: utf-8 -*-
"""
Step 2: SDE Rollout Worker
=================================================================

Implements single-step hybrid SDE sampling for RL exploration on the Action DiT's continuous velocity field.

Core math (Flow-SDE sampling):

1. Velocity-field parameterisation (causal-mask constraint)::
     v_φ^action(a_τ, τ | H_video, M)
     where M is the causal mask: first N cols = -1e9 (masks semantic Z_x), last 2N cols = 0 (opens physical+intent)

2. SDE drift term::
     μ_φ(a_τ, τ) = v_φ + (σ_τ² / 2τ) · (a_τ + (1-τ)·v_φ)

3. SDE update step::
     a_{τ+Δτ} = a_τ + μ_φ·Δτ + σ_τ·√|Δτ|·ε_τ
     where sigma_tau = a_noise * sqrt(tau / (1 - tau + eps_s))

4. Hybrid ODE-SDE sampling::
     Over K inference steps, sde_window_size steps are drawn at random within sde_window_range
     to inject SDE noise; the rest stay deterministic ODE. This keeps exploration capacity while speeding up training.

References:
  - public Flow-SDE implementations
    (the drift term and log_prob computation in sde_step_with_logprob)
  - public GRPO configurations
    (the noise_level, sde_window_size and sde_window_range parameters)

Compatibility:
  - original DiT4DiT: uses standard Action DiT cross-attention
  - multistream variant: auto-detects and injects a dynamic causal_key_mask
  Neither mode modifies sources inside the model directory; the velocity field is computed
  only through the Action Head's public submodules.
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
# Data structure: SDE step record (used by Step 3 for the Gaussian transition log_prob + GRPO loss)
# ============================================================================

@dataclass
class SDEStepRecord:
    """Records the key tensors of an SDE step; Step 3 uses them for the discrete Gaussian transition probability.

    Only SDE steps are recorded: ODE steps are deterministic and contribute no gradient to log_prob.

    Attributes:
        v_phi: velocity field v_phi(a_tau, tau | H_video, M), shape (B, T, D_action)
        a_tau: action before the SDE step, shape (B, T, D_action)
        a_next: action after the SDE step, shape (B, T, D_action)
        sigma_tau: noise coefficient sigma_tau, shape (B,) or scalar
        epsilon: injected Gaussian noise eps_tau, shape (B, T, D_action)
        prev_sample_mean: drift-mean term mu_phi*delta_tau (the quantity named prev_sample_mean),
            shape (B, T, D_action). Feeds the discrete Gaussian log_prob.
        timestep: current timestep tau, shape (B,) or scalar
        dt: step size delta_tau = tau_{k+1} - tau_k < 0, shape (B,) or scalar
        sqrt_dt: sqrt(|delta_tau|), used by log_prob
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
    """Full output of hybrid SDE sampling.

    Attributes:
        final_action: final generated action chunk a_0, shape (B, T, D_action)
        sde_records: list of SDE step records, used for the Step 3 log_prob computation
        all_velocities: velocity fields of all K steps, used for the KL divergence ||v_phi - v_ref||^2
        timesteps_schedule: timestep schedule tau_0, tau_1, ..., tau_K
        sde_step_indices: indices of the steps that ran with SDE
    """
    final_action: torch.Tensor
    sde_records: List[SDEStepRecord]
    all_velocities: List[torch.Tensor]
    timesteps_schedule: torch.Tensor
    sde_step_indices: List[int]


# ============================================================================
# Core function 1: action velocity-field computation
# ============================================================================

def compute_action_velocity(
    action_model: nn.Module,
    a_tau: torch.Tensor,
    tau: torch.Tensor,
    vl_embs: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    N_per_stream: Optional[int] = None,
) -> torch.Tensor:
    """Compute the Action DiT velocity field v_phi(a_tau, tau | H_video, M).

    Reuses action_model's submodules without modifying the original code:
      action_encoder → state_encoder → position_embedding → MaskedDiT → action_decoder

    The multistream Action Head builds the causal mask via _build_causal_mask(vl_embs);
    the original Action Head has no such interface and uses its standard cross-attention directly.

    Args:
        action_model: original or multistream FlowmatchingActionHead (trainable or reference)
        a_tau: current action chunk a_tau, shape (B, T, D_action)
        tau: current timestep tau, shape (B,) or scalar. tau in (0,1]; tau=1 is pure noise.
        vl_embs: Video DiT conditioning features. (B, S, D) for the original, (B, 3*N, D) for the multistream variant
        state: robot state, shape (B, D_state) or None
        N_per_stream: tokens per modality stream N. Inferred from vl_embs (S//3) when None.

    Returns:
        v_phi: predicted velocity field, shape (B, T, D_action)
            Under rectified flow v = noise - data, so it points in the denoising direction.
    """
    B, T, D_action = a_tau.shape
    device = a_tau.device
    model_dtype = next(action_model.parameters()).dtype
    model_a_tau = a_tau.to(dtype=model_dtype)
    model_vl_embs = vl_embs.to(device=device, dtype=model_dtype)
    model_state = state.to(device=device, dtype=model_dtype) if state is not None else None

    # ---- 1. timestep discretisation ----
    # action_model.num_timestep_buckets maps the continuous tau onto discrete buckets
    num_buckets = action_model.num_timestep_buckets
    if tau.dim() == 0:
        tau = tau.expand(B)
    # Match ``ActionDiT.predict_action`` exactly.  In particular, tau=1 is
    # represented by ``num_buckets`` (1000 in the LIBERO checkpoint), not
    # clamped to 999.  The time embedding is sinusoidal, so this is a valid
    # value rather than an embedding-table index.
    t_discretized = (tau * num_buckets).long().clamp_min(0)  # (B,)

    # ---- 2. encode the noised action ----
    # action_encoder: (B, T, D_action) + (B,) → (B, T, w)
    action_features = action_model.action_encoder(model_a_tau, t_discretized)

    # ---- 3. encode state + position embedding ----
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

    # Concatenate state + action
    if state_features is not None:
        sa_embs = torch.cat((state_features, action_features), dim=1)  # (B, 1+T, w)
    else:
        sa_embs = action_features  # (B, T, w)

    # ---- 4. Action DiT forward pass ----
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

    # ---- 6. decode into the velocity field ----
    pred = action_model.action_decoder(model_output)  # (B, 1+T, D_action) or (B, T, D_action)
    # Take the last T tokens (the action part; state tokens are skipped)
    v_phi = pred[:, -T:].float()  # fp32 keeps SDE/log-prob math stable

    return v_phi


# ============================================================================
# Core function 2: single SDE step update + log_prob recording
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
    """Single SDE step update, recording every tensor the Gaussian transition log_prob needs.

    Math::

        sigma_tau = a_noise * sqrt(tau / (1 - tau + eps_s))                  # noise coefficient
        mu_phi = v_phi + (sigma_tau^2 / 2tau) * (a_tau + (1-tau)*v_phi)      # drift term
        a_next = a_tau + mu_phi*delta_tau + sigma_tau*sqrt(|delta_tau|)*eps_tau   # SDE update

    Here delta_tau < 0, advancing from the noise end (tau=1) towards the data end (tau=0).

    Equivalent implementation::

        std_dev_t = sqrt(σ/(1-σ)) * noise_level          # = σ_τ
        prev_sample_mean = a + [v + σ_τ²/(2τ)·(a + (1-τ)v)] · dt
                         = a + μ_φ · dt
        a_next = prev_sample_mean + σ_τ · √|dt| · ε

    Args:
        v_phi: velocity field v_phi(a_tau, tau), shape (B, T, D_action)
        a_tau: current action a_tau, shape (B, T, D_action)
        tau: current timestep tau, shape (B, 1, 1), (B,) or scalar
        dt: step size delta_tau = tau_{k+1} - tau_k < 0, same shape as tau
        noise_level: noise coefficient a_noise
        eps_s: small constant eps_s guarding against division by zero
        generator: random-number generator, for reproducibility

    Returns:
        SDEStepRecord: holds a_next, v_phi, sigma_tau, epsilon, prev_sample_mean, etc.
    """
    # Unify shapes so they broadcast cleanly
    # tau, dt -> (B, 1, 1) so they broadcast against (B, T, D)
    if tau.dim() == 0:
        tau = tau.view(1, 1, 1)
    elif tau.dim() == 1:
        tau = tau.view(-1, 1, 1)
    if dt.dim() == 0:
        dt = dt.view(1, 1, 1)
    elif dt.dim() == 1:
        dt = dt.view(-1, 1, 1)

    # ---- 1. noise coefficient sigma_tau ----
    # σ_τ = a_noise · √(τ / (1 - τ + ε_s))
    # std_dev_t = sqrt(σ/(1-σ)) * noise_level
    # The two agree as eps_s -> 0: sqrt(tau/(1-tau)) = sqrt(tau/(1-tau))
    # Flow-GRPO replaces the singular denominator at tau=1 with the next
    # schedule value. Without this, the first SDE step explodes as 1/sqrt(eps).
    denominator_tau = torch.where(tau >= 1.0 - eps_s, tau + dt, tau)
    denominator_tau = denominator_tau.clamp(min=0.0, max=1.0 - eps_s)
    sigma_tau = noise_level * torch.sqrt(
        tau.clamp(min=0.0) / (1.0 - denominator_tau)
    )  # (B, 1, 1)

    # ---- 2. Gaussian transition mean ----
    prev_sample_mean = compute_sde_transition_mean(
        v_phi=v_phi,
        a_tau=a_tau,
        tau=tau,
        dt=dt,
        sigma_tau=sigma_tau,
        eps_s=eps_s,
    )

    # ---- 4. sample the Gaussian noise eps_tau ----
    sqrt_dt = torch.sqrt(-dt)  # sqrt(|delta_tau|), since dt < 0
    epsilon = torch.randn(
        a_tau.shape,
        generator=generator,
        device=a_tau.device,
        dtype=a_tau.dtype,
    )  # (B, T, D)

    # ---- 5. SDE update: a_next = prev_sample_mean + sigma_tau * sqrt(|delta_tau|) * eps ----
    a_next = prev_sample_mean + sigma_tau * sqrt_dt * epsilon  # (B, T, D)

    return SDEStepRecord(
        v_phi=v_phi,
        a_tau=a_tau,
        a_next=a_next,
        sigma_tau=sigma_tau.squeeze(),   # (B,) or scalar
        epsilon=epsilon,
        prev_sample_mean=prev_sample_mean,
        timestep=tau.squeeze(),          # (B,) or scalar
        dt=dt.squeeze(),
        sqrt_dt=sqrt_dt.squeeze(),
    )


# ============================================================================
# Core function 3: single ODE step update (deterministic)
# ============================================================================

def ode_step(
    v_phi: torch.Tensor,
    a_tau: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Single ODE step update (deterministic rectified flow).

    a_{τ+Δτ} = a_τ + v_φ · Δτ

    Args:
        v_phi: velocity field, shape (B, T, D)
        a_tau: current action, shape (B, T, D)
        dt: step size < 0

    Returns:
        a_next: next action, shape (B, T, D)
    """
    if dt.dim() == 0:
        dt = dt.view(1, 1, 1)
    elif dt.dim() == 1:
        dt = dt.view(-1, 1, 1)
    return a_tau + v_phi * dt


# ============================================================================
# Core function 4: hybrid ODE-SDE sampling (main entry point)
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
    """Generate an action chunk with hybrid ODE-SDE sampling.

    Pipeline:
      1. start from pure noise a_1 (tau=1)
      2. K inference steps (tau: 1 -> 0) with the schedule tau_k = 1 - k/K
      3. draw sde_window_size steps at random within sde_window_range and run them with SDE (noise injected)
         the remaining steps run with ODE (deterministic)
      4. SDE steps record all tensors (v_phi, a_tau, sigma_tau, eps, prev_sample_mean) for Step 3

    The causal mask is injected automatically inside compute_action_velocity, so the exploration
    gradient stays focused entirely on the physical Z_f + intent Z_v and is never polluted by the semantic Z_x.

    Args:
        action_model: MaskedFlowmatchingActionHead (trainable or reference)
        vl_embs: Video DiT layer-18 joint features H_video, shape (B, 3N, D)
        action_shape: (T, D_action), temporal length and dimension of the action chunk
        state: robot state (B, D_state) or None
        num_inference_steps: number of inference steps K (K=4 recommended)
        noise_level: SDE noise coefficient a_noise
        sde_window_size: number of SDE steps (default 1: a single step receives noise)
        sde_window_range: (start, end) window used to select SDE steps, indices in [0, K).
            Defaults to (0, K//2) when None, i.e. noise goes into the first half.
        N_per_stream: number of tokens per modality stream
        eps_s: constant guarding against division by zero
        generator: random-number generator
        batch_size: B (inferred from vl_embs when None)
        device / dtype: tensor device and precision
        grad_context: gradient context manager, e.g. torch.enable_grad().
            Defaults to torch.no_grad() when None: rollout needs no gradient, Step 3 recomputes them.

    Returns:
        RolloutResult: holds final_action, sde_records, all_velocities
    """
    T, D_action = action_shape
    if batch_size is None:
        batch_size = vl_embs.shape[0]
    if device is None:
        device = vl_embs.device

    K = num_inference_steps

    # ---- 1. timestep schedule: tau_0=1, tau_1=1-1/K, ..., tau_K=0 ----
    # sigmas: [1.0, 1-1/K, 1-2/K, ..., 0.0], K+1 points spanning K steps
    timesteps = torch.linspace(1.0, 0.0, K + 1, device=device, dtype=dtype)  # (K+1,)

    # ---- 2. choose the SDE step indices ----
    if sde_window_range is None:
        sde_window_range = (0, max(1, K // 2))  # default: first half
    window_start, window_end = sde_window_range
    window_end = min(window_end, K)  # never exceed the total step count

    # Draw sde_window_size steps at random inside the window.
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
        # Use the generator so results stay reproducible
        if generator is not None:
            perm = torch.randperm(len(window_indices), generator=generator)
        else:
            perm = torch.randperm(len(window_indices))
        sde_step_indices = set(
            window_indices[i] for i in perm[:sde_window_size].tolist()
        )

    logger.debug(f"[Rollout] K={K}, SDE steps={sorted(sde_step_indices)}, "
                 f"window={sde_window_range}")

    # ---- 3. initialise: a_1 ~ N(0, I) (pure noise) ----
    a_tau = torch.randn(
        (batch_size, T, D_action),
        generator=generator,
        device=device,
        dtype=dtype,
    )  # (B, T, D)

    sde_records: List[SDEStepRecord] = []
    all_velocities: List[torch.Tensor] = []

    # ---- 4. gradient context ----
    # Rollout normally needs no gradient; log_prob gains them only when Step 3 recomputes it
    if grad_context is None:
        grad_context = torch.no_grad()

    # ---- 5. K inference steps ----
    with grad_context:
        for k in range(K):
            tau_k = timesteps[k]       # current timestep tau_k
            tau_k1 = timesteps[k + 1]  # next timestep tau_{k+1}
            dt = tau_k1 - tau_k        # Δτ < 0

            # 5a. compute the velocity field v_phi(a_tau, tau_k | H_video, M)
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

            # 5b. SDE step or ODE step
            if k in sde_step_indices:
                # ---- SDE step: inject noise + record ----
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
                # ---- ODE step: deterministic update ----
                a_tau = ode_step(v_phi, a_tau, dt_tensor)

    # ---- 6. return the result ----
    result = RolloutResult(
        final_action=a_tau,           # a_0 (the denoised action chunk)
        sde_records=sde_records,      # SDE step records (used for log_prob)
        all_velocities=all_velocities,  # velocity fields of every step (used for KL)
        timesteps_schedule=timesteps,
        sde_step_indices=sorted(sde_step_indices),
    )

    logger.debug(f"[Rollout] Done. SDE records: {len(sde_records)}, "
                 f"final_action shape: {tuple(a_tau.shape)}")

    return result


# ============================================================================
# Helper: group sampling (GRPO needs G parallel trajectories for one prompt)
# ============================================================================

def sample_group_hybrid_sde(
    action_model: nn.Module,
    vl_embs_single: torch.Tensor,
    state_single: Optional[torch.Tensor],
    action_shape: Tuple[int, int],
    group_size: int = 8,
    **kwargs,
) -> RolloutResult:
    """Sample G parallel action trajectories for GRPO (group sampling).

    Replicates one observation (vl_embs, state) G times and samples G distinct action trajectories in parallel.
    Trajectories diverge through different initial noise and different injected SDE noise.

    Args:
        action_model: the Action DiT model
        vl_embs_single: Video DiT features of one observation, shape (3N, D) or (1, 3N, D)
        state_single: robot state of one observation, shape (D_state,) or (1, D_state)
        action_shape: (T, D_action)
        group_size: GRPO group size G
        **kwargs: extra arguments forwarded to sample_hybrid_sde_action

    Returns:
        RolloutResult: sampling result with batch dim = G (GRPO consumes it directly in batch form)
    """
    # Expand to G copies
    if vl_embs_single.dim() == 2:
        vl_embs_single = vl_embs_single.unsqueeze(0)  # (1, 3N, D)
    vl_embs_group = vl_embs_single.expand(group_size, -1, -1)  # (G, 3N, D)

    state_group = None
    if state_single is not None:
        if state_single.dim() == 1:
            state_single = state_single.unsqueeze(0)
        state_group = state_single.expand(group_size, -1)  # (G, D_state)

    # Sample all G trajectories in one parallel call
    result = sample_hybrid_sde_action(
        action_model=action_model,
        vl_embs=vl_embs_group,
        action_shape=action_shape,
        state=state_group,
        batch_size=group_size,
        **kwargs,
    )

    # Split the batch dimension into G independent results
    # GRPO normally consumes the batch form directly, so no split is needed
    # A single RolloutResult (batch dim = G) is returned for Step 3 to use as-is
    return result
