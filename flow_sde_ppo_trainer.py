"""PiRL-style Flow-SDE PPO/SAPO post-training for the original DiT4DiT Action Head.

This is deliberately separate from the repository's GRPO/AWR experiments.
Rollouts use the tractable Gaussian transitions of the Flow-SDE sampler; PPO or
SAPO updates the Action Head with a four-layer value critic and chunk-level GAE.
The frozen Video DiT/backbone is never optimized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dit4dit_rl_trainer.engine import CausalActionGRPOEngine, RLEngineConfig
from dit4dit_rl_trainer.math_utils import (
    compute_gaussian_transition_log_prob,
    compute_kl_loss,
)
from dit4dit_rl_trainer.sde_rollout import (
    RolloutResult,
    SDEStepRecord,
    compute_action_velocity,
    compute_sde_transition_mean,
    sample_hybrid_sde_action,
)
from dit4dit_rl_trainer.trainer import (
    CausalActionGRPOTrainer,
    GRPOConfig,
    TrainerConfig,
    _build_real_env,
    _build_sft_dataset,
)


@dataclass
class FlowSDEPPOConfig:
    train_task_id: int = 5
    sampling_mode: str = "single"
    target_task_weight: float = 0.5
    action_horizon: int = 8
    execution_horizon: int = 4
    k_steps: int = 4
    noise_level: float = 0.1
    noise_level_end: float = 0.02
    sde_window_size: int = 1
    sde_window_range: Tuple[int, int] = (0, 2)
    rollout_episodes: int = 16
    min_rollout_chunks: int = 256
    max_collection_episodes: int = 32
    min_successful_rollouts: int = 1
    max_no_signal_updates: int = 3
    updates: int = 20
    ppo_epochs: int = 4
    minibatch_size: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    loss_type: str = "ppo"
    clip_range: float = 0.2
    tau_pos: float = 1.0
    tau_neg: float = 1.0
    beta_kl: float = 0.01
    sft_anchor_weight: float = 0.25
    sft_anchor_batch_size: int = 4
    critic_lr: float = 1e-4
    critic_warmup_steps: int = 25
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    max_episode_steps: int = 220
    eval_episodes: int = 20
    preflight_eval_episodes: int = 20
    eval_freq_updates: int = 1
    eval_seed: int = 10000
    save_freq_updates: int = 1
    require_preflight_success: bool = True
    preflight_only: bool = False


class FourLayerValueCritic(nn.Module):
    """State value V(o) conditioned on frozen Video-DiT and robot state."""

    def __init__(self, video_dim: int, state_dim: int, width: int = 512) -> None:
        super().__init__()
        input_dim = int(video_dim) + int(state_dim)
        self.state_dim = int(state_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width // 2), nn.SiLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, vl_embs: torch.Tensor, state: Optional[torch.Tensor]) -> torch.Tensor:
        video = vl_embs.float().mean(dim=1)
        if self.state_dim:
            if state is None:
                state = torch.zeros(video.shape[0], self.state_dim, device=video.device)
            state = state.float()
            if state.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Critic expected state_dim={self.state_dim}, got {state.shape[-1]}"
                )
            video = torch.cat((video, state), dim=-1)
        expected = self.network[0].normalized_shape[0]
        if video.shape[-1] != expected:
            raise ValueError(
                f"PPO critic expected pooled feature width {expected}, "
                f"got {video.shape[-1]} (Video-DiT/action-head dimensions are different)"
            )
        return self.network(video).squeeze(-1)


@dataclass
class PPOChunk:
    rollout: RolloutResult
    vl_embs: torch.Tensor
    state: Optional[torch.Tensor]
    next_vl_embs: Optional[torch.Tensor]
    next_state: Optional[torch.Tensor]
    reward: float
    done: bool
    episode_id: int
    decision_index: int
    value: float = 0.0
    advantage: float = 0.0
    return_target: float = 0.0


def _cpu_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if value is None else value.detach().cpu()


def _rollout_to_cpu(rollout: RolloutResult) -> RolloutResult:
    records = [
        SDEStepRecord(**{name: _cpu_tensor(value) for name, value in vars(record).items()})
        for record in rollout.sde_records
    ]
    return RolloutResult(
        final_action=_cpu_tensor(rollout.final_action),
        sde_records=records,
        all_velocities=[_cpu_tensor(value) for value in rollout.all_velocities],
        timesteps_schedule=_cpu_tensor(rollout.timesteps_schedule),
        sde_step_indices=list(rollout.sde_step_indices),
    )


def _rollout_to_device(rollout: RolloutResult, device: torch.device) -> RolloutResult:
    def move(value: Any) -> Any:
        return value.to(device) if isinstance(value, torch.Tensor) else value

    return RolloutResult(
        final_action=move(rollout.final_action),
        sde_records=[SDEStepRecord(**{name: move(value) for name, value in vars(record).items()})
                     for record in rollout.sde_records],
        all_velocities=[move(value) for value in rollout.all_velocities],
        timesteps_schedule=move(rollout.timesteps_schedule),
        sde_step_indices=list(rollout.sde_step_indices),
    )


class FlowSDEPPOTrainer(CausalActionGRPOTrainer):
    """On-policy Flow-SDE PPO with chunk rewards and generalized advantage estimation."""

    def __init__(self, *args: Any, ppo_cfg: FlowSDEPPOConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ppo_cfg = ppo_cfg
        # ``ActionHead.hidden_size`` is the decoder width (2560 in the
        # released LIBERO checkpoint), not the width of the Video-DiT
        # conditioning tokens (2048).  Using it here makes the critic fail on
        # the first rollout with a matrix-shape mismatch.  Prefer the actual
        # backbone/attention width and keep the action width only as a legacy
        # fallback for small mock models.
        video_dim = self._infer_video_feature_dim()
        if video_dim <= 0:
            raise ValueError("Could not infer Video-DiT feature width for PPO critic")
        state_dim = int(getattr(self.action_model.config, "state_dim", 0))
        print(
            f"[FlowSDEPPO] critic input: video_dim={video_dim} state_dim={state_dim}",
            flush=True,
        )
        self.critic = FourLayerValueCritic(video_dim, state_dim).to(self.device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=ppo_cfg.critic_lr)
        self.collection_episode = 0
        self.best_target_gain = 0
        self.preflight_records: Dict[Tuple[int, int, int], bool] = {}

    def _infer_video_feature_dim(self) -> int:
        """Infer the last dimension of ``extract_h_video`` without a rollout."""
        candidates = [
            getattr(getattr(self.model, "backbone_interface", None), "hidden_size", None),
            getattr(
                getattr(getattr(self.model, "backbone_interface", None), "extractor", None),
                "hidden_size",
                None,
            ),
            getattr(getattr(getattr(self.action_model, "model", None), "config", None),
                    "cross_attention_dim", None),
            getattr(getattr(self.action_model, "full_config", None), "vl_hidden_dim", None),
        ]
        for value in candidates:
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return int(getattr(self.action_model, "hidden_size", 0))

    @torch.no_grad()
    def _value(self, vl_embs: torch.Tensor, state: Optional[torch.Tensor]) -> float:
        self.critic.eval()
        value = self.critic(vl_embs.to(self.device), None if state is None else state.to(self.device))
        return float(value.item())

    @torch.no_grad()
    def _next_features(
        self, obs: Dict[str, Any], previous_rgb: np.ndarray
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rgb, flow, value = self._observe_to_images(obs, prev_rgb=previous_rgb)
        h_video = self.extract_h_video(
            rgb_frames=[rgb],
            flow_frames=[flow] if flow is not None else None,
            value_frames=[value] if value is not None else None,
            lang=obs.get("lang", ""),
            batch_size=1,
        )
        return h_video.detach(), self._prepare_state(obs.get("state"), batch_size=1)

    def collect_episode(
        self, episode_id: int, noise_level: float
    ) -> Tuple[List[PPOChunk], Dict[str, Any]]:
        cfg = self.ppo_cfg
        current_task_id = self._select_rollout_task_id(episode_id)
        if hasattr(self.env, "reset_for_task"):
            obs = self.env.reset_for_task(current_task_id, self.collection_episode)
        else:
            obs = self.env.reset()
        self.collection_episode += 1
        chunks: List[PPOChunk] = []
        previous_rgb: Optional[np.ndarray] = None
        total_reward = 0.0
        success = False
        env_steps = 0
        decision_index = 0

        while env_steps < cfg.max_episode_steps and not success:
            current_rgb = np.asarray(obs["rgb"])
            h_video, state = self._next_features(obs, previous_rgb) if previous_rgb is not None else self._next_features(obs, current_rgb)
            # The SDE sampler is the behavior policy.  Its detached Gaussian means
            # are the PPO denominator, while the current Action Head is replayed in update.
            rollout = sample_hybrid_sde_action(
                action_model=self.action_model,
                vl_embs=h_video,
                action_shape=(cfg.action_horizon, self.action_model.action_dim),
                state=state,
                num_inference_steps=cfg.k_steps,
                noise_level=noise_level,
                sde_window_size=cfg.sde_window_size,
                sde_window_range=cfg.sde_window_range,
                N_per_stream=getattr(self.action_model, "N_per_stream", None),
                batch_size=1,
                device=self.device,
                dtype=torch.float32,
                grad_context=torch.no_grad(),
            )
            env_actions = self._prepare_env_actions(rollout.final_action[0].cpu().numpy())
            chunk_reward = 0.0
            done = False
            for action in env_actions[:cfg.execution_horizon]:
                obs, reward, env_done, info = self.env.step(action)
                became_success = bool(info.get("success", False)) and not success
                # LIBERO's reward differs between versions; terminal success is always
                # represented as one sparse reward exactly once.
                chunk_reward += max(float(reward), float(became_success))
                total_reward += max(float(reward), float(became_success))
                env_steps += 1
                success = success or bool(info.get("success", False))
                done = bool(env_done) or success or env_steps >= cfg.max_episode_steps
                if done:
                    break
            if done:
                next_h_video = None
                next_state = None
            else:
                next_h_video, next_state = self._next_features(obs, current_rgb)
            chunks.append(PPOChunk(
                rollout=_rollout_to_cpu(rollout), vl_embs=_cpu_tensor(h_video),
                state=_cpu_tensor(state), next_vl_embs=_cpu_tensor(next_h_video),
                next_state=_cpu_tensor(next_state), reward=chunk_reward, done=done,
                episode_id=episode_id, decision_index=decision_index,
            ))
            previous_rgb = current_rgb
            decision_index += 1
            if done:
                break
        return chunks, {
            "success": success, "reward": total_reward, "chunks": len(chunks),
            "env_steps": env_steps, "task_id": current_task_id,
            "noise_level": noise_level,
        }

    def _select_rollout_task_id(self, episode_id: int) -> int:
        """Select the task used by this rollout without changing single-task mode."""
        cfg = self.ppo_cfg
        all_tasks = list(getattr(self.env, "task_ids", range(10)))
        if cfg.sampling_mode == "uniform":
            return int(all_tasks[episode_id % len(all_tasks)])
        if cfg.sampling_mode == "weighted":
            if not 0.0 <= cfg.target_task_weight <= 1.0:
                raise ValueError("target_task_weight must be in [0, 1]")
            if random.random() < cfg.target_task_weight:
                return cfg.train_task_id
            other_tasks = [task_id for task_id in all_tasks if task_id != cfg.train_task_id]
            return int(random.choice(other_tasks)) if other_tasks else cfg.train_task_id
        if cfg.sampling_mode == "single":
            return cfg.train_task_id
        raise ValueError(f"Unknown sampling_mode: {cfg.sampling_mode!r}")

    def _set_monte_carlo_returns(self, chunks: Sequence[PPOChunk]) -> None:
        """Build critic targets before the critic is used for GAE."""
        by_episode: Dict[int, List[PPOChunk]] = {}
        for chunk in chunks:
            by_episode.setdefault(chunk.episode_id, []).append(chunk)
        discount = self.ppo_cfg.gamma ** self.ppo_cfg.execution_horizon
        for episode_chunks in by_episode.values():
            running_return = 0.0
            for chunk in reversed(episode_chunks):
                running_return = chunk.reward + discount * running_return * float(not chunk.done)
                chunk.return_target = running_return

    def _warmup_critic(self, chunks: Sequence[PPOChunk]) -> float:
        """Fit V(o) to rollout returns before using it for credit assignment."""
        if self.ppo_cfg.critic_warmup_steps <= 0:
            return 0.0
        indices = list(range(len(chunks)))
        losses: List[float] = []
        self.critic.train()
        for _ in range(self.ppo_cfg.critic_warmup_steps):
            random.shuffle(indices)
            for start in range(0, len(indices), self.ppo_cfg.minibatch_size):
                batch = [chunks[index] for index in indices[start:start + self.ppo_cfg.minibatch_size]]
                vl_embs = torch.cat([chunk.vl_embs for chunk in batch]).to(self.device)
                states = None if batch[0].state is None else torch.cat(
                    [chunk.state for chunk in batch]
                ).to(self.device)
                returns = torch.tensor(
                    [chunk.return_target for chunk in batch], device=self.device
                )
                value_loss = F.mse_loss(self.critic(vl_embs, states), returns)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.ppo_cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.ppo_cfg.max_grad_norm
                )
                self.critic_optimizer.step()
                losses.append(float(value_loss.detach()))
        return float(np.mean(losses)) if losses else 0.0

    def _compute_gae(self, chunks: Sequence[PPOChunk]) -> None:
        by_episode: Dict[int, List[PPOChunk]] = {}
        for chunk in chunks:
            chunk.value = self._value(chunk.vl_embs, chunk.state)
            by_episode.setdefault(chunk.episode_id, []).append(chunk)
        discount = self.ppo_cfg.gamma ** self.ppo_cfg.execution_horizon
        gae_discount = discount * self.ppo_cfg.gae_lambda
        for episode_chunks in by_episode.values():
            running_advantage = 0.0
            for chunk in reversed(episode_chunks):
                next_value = 0.0 if chunk.done else self._value(chunk.next_vl_embs, chunk.next_state)
                delta = chunk.reward + discount * next_value - chunk.value
                running_advantage = delta + gae_discount * running_advantage * float(not chunk.done)
                chunk.advantage = running_advantage
                chunk.return_target = chunk.value + running_advantage

        advantages = torch.tensor([chunk.advantage for chunk in chunks], dtype=torch.float32)
        normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        for chunk, advantage in zip(chunks, normalized.tolist()):
            chunk.advantage = float(advantage)

    def _policy_loss(self, chunk: PPOChunk) -> Tuple[torch.Tensor, Dict[str, float]]:
        rollout = _rollout_to_device(chunk.rollout, self.device)
        vl_embs = chunk.vl_embs.to(self.device)
        state = None if chunk.state is None else chunk.state.to(self.device)
        advantage = torch.tensor(chunk.advantage, device=self.device)
        transition_losses: List[torch.Tensor] = []
        means_new: List[torch.Tensor] = []
        means_ref: List[torch.Tensor] = []
        scales: List[torch.Tensor] = []
        ratios: List[torch.Tensor] = []
        sapo_gates: List[torch.Tensor] = []
        for record in rollout.sde_records:
            v_new = compute_action_velocity(
                self.action_model, record.a_tau, record.timestep, vl_embs, state,
                getattr(self.action_model, "N_per_stream", None),
            )
            with torch.no_grad():
                v_ref = compute_action_velocity(
                    self.engine.ref_action_model, record.a_tau, record.timestep, vl_embs, state,
                    getattr(self.action_model, "N_per_stream", None),
                )
            mean_new = compute_sde_transition_mean(
                v_new, record.a_tau, record.timestep, record.dt, record.sigma_tau
            )
            scale = record.sigma_tau * record.sqrt_dt
            log_new = compute_gaussian_transition_log_prob(record.a_next, mean_new, scale)
            log_old = compute_gaussian_transition_log_prob(
                record.a_next, record.prev_sample_mean.detach(), scale
            )
            if not all(torch.isfinite(value).all() for value in (mean_new, scale, log_new, log_old)):
                raise FloatingPointError(
                    "Non-finite Flow-SDE PPO transition statistics; lower rollout noise "
                    "or inspect the sampler schedule before updating the policy."
                )
            ratio = torch.exp((log_new - log_old).clamp(-20.0, 20.0))
            if self.ppo_cfg.loss_type == "sapo":
                # SAPO: asymmetric, differentiable soft gating of the policy ratio.
                # The gate is selected by the sign of the (GAE) advantage and is
                # intentionally not detached, so it remains a policy gradient term.
                gate_pos = torch.sigmoid(self.ppo_cfg.tau_pos * (ratio - 1.0)) * (4.0 / self.ppo_cfg.tau_pos)
                gate_neg = torch.sigmoid(self.ppo_cfg.tau_neg * (ratio - 1.0)) * (4.0 / self.ppo_cfg.tau_neg)
                soft_gate = torch.where(advantage > 0.0, gate_pos, gate_neg)
                transition_losses.append(-soft_gate * advantage)
                sapo_gates.append(soft_gate.detach())
            else:
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - self.ppo_cfg.clip_range, 1.0 + self.ppo_cfg.clip_range) * advantage
                transition_losses.append(-torch.minimum(unclipped, clipped))
            means_new.append(mean_new)
            means_ref.append(compute_sde_transition_mean(
                v_ref, record.a_tau, record.timestep, record.dt, record.sigma_tau
            ))
            scales.append(scale)
            ratios.append(ratio.detach())
        if not transition_losses:
            raise RuntimeError("Flow-SDE PPO requires at least one stochastic SDE transition")
        policy_loss = torch.stack(transition_losses).mean()
        kl_loss = compute_kl_loss(means_new, means_ref, scales)
        ratio_values = torch.stack(ratios)
        clip_fraction = (
            float(((ratio_values < 1.0 - self.ppo_cfg.clip_range) |
                   (ratio_values > 1.0 + self.ppo_cfg.clip_range)).float().mean())
            if self.ppo_cfg.loss_type == "ppo" else 0.0
        )
        return policy_loss + self.ppo_cfg.beta_kl * kl_loss, {
            "policy_loss": float(policy_loss.detach()), "kl_loss": float(kl_loss.detach()),
            "ratio_mean": float(ratio_values.mean()),
            "clip_fraction": clip_fraction,
            "sapo_gate_mean": float(torch.stack(sapo_gates).mean()) if sapo_gates else 0.0,
        }

    def update_ppo(self, chunks: List[PPOChunk]) -> Dict[str, float]:
        self._set_monte_carlo_returns(chunks)
        critic_warmup_loss = self._warmup_critic(chunks)
        self._compute_gae(chunks)
        self.action_model.train()
        self.critic.train()
        policy_losses: List[float] = []
        value_losses: List[float] = []
        kl_losses: List[float] = []
        ratios: List[float] = []
        clip_fractions: List[float] = []
        sapo_gate_means: List[float] = []
        anchor_losses: List[float] = []
        indices = list(range(len(chunks)))
        for _ in range(self.ppo_cfg.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), self.ppo_cfg.minibatch_size):
                batch = [chunks[index] for index in indices[start:start + self.ppo_cfg.minibatch_size]]
                policy_terms: List[torch.Tensor] = []
                diagnostics: List[Dict[str, float]] = []
                for chunk in batch:
                    policy_term, info = self._policy_loss(chunk)
                    policy_terms.append(policy_term)
                    diagnostics.append(info)
                actor_loss = torch.stack(policy_terms).mean()
                anchor_loss, _, _ = self._compute_rsft_anchor_loss()
                if anchor_loss is not None:
                    actor_loss = actor_loss + self.ppo_cfg.sft_anchor_weight * anchor_loss
                    anchor_losses.append(float(anchor_loss.detach()))
                self.optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), self.ppo_cfg.max_grad_norm)
                self.optimizer.step()

                vl_embs = torch.cat([chunk.vl_embs for chunk in batch]).to(self.device)
                states = None if batch[0].state is None else torch.cat([chunk.state for chunk in batch]).to(self.device)
                returns = torch.tensor([chunk.return_target for chunk in batch], device=self.device)
                values = self.critic(vl_embs, states)
                value_loss = F.mse_loss(values, returns)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.ppo_cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo_cfg.max_grad_norm)
                self.critic_optimizer.step()
                policy_losses.append(float(actor_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                kl_losses.append(float(np.mean([item["kl_loss"] for item in diagnostics])))
                ratios.append(float(np.mean([item["ratio_mean"] for item in diagnostics])))
                clip_fractions.append(float(np.mean([item["clip_fraction"] for item in diagnostics])))
                sapo_gate_means.append(float(np.mean([item["sapo_gate_mean"] for item in diagnostics])))
        value_tensor = torch.tensor([chunk.value for chunk in chunks])
        return_tensor = torch.tensor([chunk.return_target for chunk in chunks])
        explained_variance = 1.0 - float((return_tensor - value_tensor).var(unbiased=False) /
                                         return_tensor.var(unbiased=False).clamp_min(1e-6))
        return {
            "policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses)),
            "kl_loss": float(np.mean(kl_losses)), "ratio_mean": float(np.mean(ratios)),
            "clip_fraction": float(np.mean(clip_fractions)), "explained_variance": explained_variance,
            "sapo_gate_mean": float(np.mean(sapo_gate_means)) if sapo_gate_means else 0.0,
            "critic_warmup_loss": critic_warmup_loss,
            "sft_anchor_loss": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
            "advantage_mean": float(np.mean([chunk.advantage for chunk in chunks])),
            "reward_mean": float(np.mean([chunk.reward for chunk in chunks])),
        }

    @staticmethod
    def _record_map(metrics: Dict[str, Any]) -> Dict[Tuple[int, int, int], bool]:
        return {
            (int(row["task_id"]), int(row["init_state_id"]), int(row["episode_seed"])): bool(row["success"])
            for row in metrics["records"]
        }

    def _accept_checkpoint(self, candidate: Dict[str, Any]) -> Tuple[bool, Dict[str, int]]:
        baseline = self.preflight_records
        current = self._record_map(candidate)
        keys = baseline.keys() & current.keys()
        if not keys:
            raise RuntimeError("PPO evaluation did not preserve preflight episode keys")
        target = self.ppo_cfg.train_task_id
        if self.ppo_cfg.sampling_mode == "uniform":
            gains = sum(not baseline[key] and current[key] for key in keys)
            losses = sum(baseline[key] and not current[key] for key in keys)
            protected_losses = 0
        else:
            gains = sum(not baseline[key] and current[key] for key in keys if key[0] == target)
            losses = sum(baseline[key] and not current[key] for key in keys if key[0] == target)
            protected_losses = sum(
                baseline[key] and not current[key] for key in keys if key[0] != target
            )
        net_gain = int(gains) - int(losses)
        accepted = net_gain > self.best_target_gain and protected_losses == 0
        details = {"gains": int(gains), "losses": int(losses), "net_gain": net_gain,
                   "protected_losses": int(protected_losses),
                   "sampling_mode": self.ppo_cfg.sampling_mode}
        print(f"[FlowSDEPPO Gate] {details} accepted={accepted}", flush=True)
        if accepted:
            self.best_target_gain = net_gain
        return accepted, details

    def save_resume(self, name: str) -> Path:
        algorithm = "sapo" if self.ppo_cfg.loss_type == "sapo" else "ppo"
        path = Path(self.cfg.output_dir) / "checkpoints" / f"flow_sde_{algorithm}_{name}_resume.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "action_model": self.action_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "critic": self.critic.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
            "flow_sde_ppo_config": asdict(self.ppo_cfg),
            "flow_sde_algorithm": algorithm,
            "engine_cfg": self.cfg.engine.__dict__,
            "collection_episode": self.collection_episode,
            "global_step": self.global_step,
            "best_target_gain": self.best_target_gain,
        }, path)
        return path

    def resume_from_checkpoint(self, path: str) -> None:
        """Restore a Flow-SDE resume checkpoint, including trainer state."""
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        required = ("action_model", "optimizer", "critic", "critic_optimizer")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(
                f"Invalid Flow-SDE resume checkpoint {checkpoint}; missing {missing}"
            )
        self.action_model.load_state_dict(payload["action_model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.collection_episode = int(payload.get("collection_episode", 0))
        saved_step = payload.get("global_step")
        if saved_step is None:
            # Older checkpoints predate the explicit global_step field.
            match = re.search(r"_update_(\d+)_resume\.pt$", checkpoint.name)
            saved_step = int(match.group(1)) if match else 0
        self.global_step = int(saved_step)
        self.best_target_gain = float(payload.get("best_target_gain", 0.0))
        print(
            f"[FlowSDEPPO] resumed checkpoint={checkpoint} "
            f"global_step={self.global_step} collection_episode={self.collection_episode}",
            flush=True,
        )

    def _evaluate_preflight(self) -> Dict[str, Any]:
        baseline = self.evaluate(self.ppo_cfg.eval_episodes, label="flow_sde_ppo_preflight")
        self.preflight_records = self._record_map(baseline)
        target = baseline["per_task"].get(str(self.ppo_cfg.train_task_id), {})
        # RoboCasa runs a 24-task suite; task 0 is not a privileged target.
        print(f"[FlowSDEPPO] SFT preflight task={self.ppo_cfg.train_task_id}: "
              f"{target.get('successes', 0)}/{target.get('episodes', 0)}; all={baseline['successes']}/{baseline['episodes']}", flush=True)
        if self.ppo_cfg.require_preflight_success and target.get("successes", 0) == 0:
            raise RuntimeError(
                "Task-5 SFT preflight has zero successes. Sparse online RL cannot be "
                "validated from this checkpoint; first fix official inference alignment."
            )
        return baseline

    def train_flow_sde_ppo(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self._evaluate_preflight()
        if self.ppo_cfg.preflight_only:
            print("[FlowSDEPPO] Preflight-only mode complete; PPO was not started.", flush=True)
            return
        metrics_path = Path(self.cfg.output_dir) / "flow_sde_ppo_metrics.jsonl"
        # Keep a recoverable policy before the first online update. If an
        # update fails evaluation, the next rollout must start from the last
        # accepted policy instead of compounding the rejected change.
        best_resume_path = self.save_resume("initial")
        no_signal_updates = 0
        for update in range(self.global_step + 1, self.ppo_cfg.updates + 1):
            progress = (update - 1) / max(1, self.ppo_cfg.updates - 1)
            rollout_noise = (
                (1.0 - progress) * self.ppo_cfg.noise_level
                + progress * self.ppo_cfg.noise_level_end
            )
            chunks: List[PPOChunk] = []
            episodes: List[Dict[str, Any]] = []
            while (
                len(episodes) < self.ppo_cfg.rollout_episodes
                or len(chunks) < self.ppo_cfg.min_rollout_chunks
                or sum(int(row["success"]) for row in episodes)
                < self.ppo_cfg.min_successful_rollouts
            ) and len(episodes) < self.ppo_cfg.max_collection_episodes:
                episode_chunks, summary = self.collect_episode(
                    len(episodes), rollout_noise
                )
                chunks.extend(episode_chunks)
                episodes.append(summary)
                print(f"[FlowSDEPPO] update={update} rollout={len(episodes)} "
                      f"chunks={len(chunks)} success={summary['success']} steps={summary['env_steps']}", flush=True)
            if not chunks:
                raise RuntimeError("No rollout chunks were collected")
            successful_rollouts = sum(int(row["success"]) for row in episodes)
            if successful_rollouts < self.ppo_cfg.min_successful_rollouts:
                no_signal_updates += 1
                metrics = {
                    "update": update,
                    "chunks": len(chunks),
                    "episodes": episodes,
                    "rollout_success_rate": float(np.mean([row["success"] for row in episodes])),
                    "successful_rollouts": successful_rollouts,
                    "skipped": 1.0,
                    "reason": "no_successful_rollout",
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")
                print(
                    f"[FlowSDEPPO] update={update} skipped: no successful rollout "
                    f"after {len(episodes)} episodes ({no_signal_updates}/"
                    f"{self.ppo_cfg.max_no_signal_updates})",
                    flush=True,
                )
                if no_signal_updates >= self.ppo_cfg.max_no_signal_updates:
                    raise RuntimeError(
                        "Flow-SDE PPO received no successful rollout signal. "
                        "Lower noise_level/execution_horizon or verify the ODE "
                        "preflight before spending updates on sparse reward."
                    )
                continue
            no_signal_updates = 0
            metrics = self.update_ppo(chunks)
            # ``global_step`` is the number of completed optimizer updates.
            # Increment it before evaluation so result metadata and exported
            # checkpoint names identify the weights that were just evaluated.
            self.global_step += 1
            metrics.update({"update": update, "chunks": len(chunks), "episodes": episodes,
                            "successful_rollouts": successful_rollouts,
                            "rollout_success_rate": float(np.mean([row['success'] for row in episodes])),
                            "global_step": self.global_step})
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")
            print(f"[FlowSDEPPO] update={update}/{self.ppo_cfg.updates} chunks={len(chunks)} "
                  f"noise={rollout_noise:.4f} rollout_success={metrics['rollout_success_rate']:.2%} "
                  f"policy={metrics['policy_loss']:.6f} value={metrics['value_loss']:.6f} "
                  f"warmup={metrics['critic_warmup_loss']:.6f} anchor={metrics['sft_anchor_loss']:.6f} "
                  f"kl={metrics['kl_loss']:.6f} ratio={metrics['ratio_mean']:.4f} "
                  f"clip={metrics['clip_fraction']:.2%} gate={metrics.get('sapo_gate_mean', 0.0):.4f} "
                  f"ev={metrics['explained_variance']:.3f}", flush=True)
            if self.ppo_cfg.save_freq_updates > 0 and update % self.ppo_cfg.save_freq_updates == 0:
                print(f"[FlowSDEPPO] resume={self.save_resume(f'update_{update}')}", flush=True)
            # The gate must see every update.  ``eval_freq_updates`` is kept as
            # a backwards-compatible reporting knob, but skipping evaluation
            # here would make an intermediate best checkpoint unknowable.
            evaluation = self.evaluate(
                self.ppo_cfg.eval_episodes, label=f"flow_sde_ppo_update_{update}"
            )
            accepted, gate = self._accept_checkpoint(evaluation)
            if accepted:
                # Persist accepted weights immediately so a later update cannot
                # overwrite the only copy of an intermediate best model.
                resume_path = self.save_resume("best")
                best_resume_path = resume_path
                export_path = Path(self.cfg.output_dir) / "checkpoints" / "flow_sde_ppo_best_pytorch_model.pt"
                self.engine.export_pretrained_checkpoint(str(export_path))
                print(f"[FlowSDEPPO] accepted best: resume={resume_path}; deployable={export_path}", flush=True)
            else:
                print(f"[FlowSDEPPO] rejected checkpoint: {gate}", flush=True)
                if best_resume_path.exists():
                    self.engine.load_action_checkpoint(str(best_resume_path), load_optimizer=True)
                    print(
                        f"[FlowSDEPPO] rolled back rejected update to {best_resume_path}",
                        flush=True,
                    )
                else:
                    # A manually removed best file must not abort the run.
                    # Keep the evaluated candidate and create a replacement
                    # recovery point for the next update.
                    best_resume_path = self.save_resume("recovery")
                    print(
                        f"[FlowSDEPPO][WARN] best checkpoint missing; "
                        f"created recovery point {best_resume_path}",
                        flush=True,
                    )
            if self.ppo_cfg.eval_freq_updates > 0 and update % self.ppo_cfg.eval_freq_updates == 0:
                print(
                    f"[FlowSDEPPO] scheduled evaluation boundary reached at update={update}; "
                    "the per-update gate has already run",
                    flush=True,
                )
        print(f"[FlowSDEPPO] final_resume={self.save_resume('final')}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flow-SDE PPO/SAPO post-training for DiT4DiT")
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trainable_scope", choices=("action_head", "full", "output", "lora"), default="action_head")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_task_id", type=int, default=5)
    parser.add_argument("--sampling_mode", choices=("single", "uniform", "weighted"), default="single")
    parser.add_argument("--target_task_weight", type=float, default=0.5)
    parser.add_argument("--libero_suite", default="libero_spatial")
    parser.add_argument("--libero_task_id", type=int, default=5)
    parser.add_argument("--libero_task_ids", default="all")
    parser.add_argument("--libero_task_schedule", choices=("round_robin", "random"), default="round_robin")
    parser.add_argument("--libero_task_block_size", type=int, default=1)
    parser.add_argument("--libero_init_state_id", type=int, default=None)
    parser.add_argument("--libero_wait_steps", type=int, default=10)
    parser.add_argument("--libero_camera", default="agentview_image")
    parser.add_argument("--libero_wrist_camera", default="robot0_eye_in_hand_image")
    parser.add_argument("--env_action_dim", type=int, default=7)
    parser.add_argument("--unnorm_key", default=None)
    parser.add_argument("--action_unnorm_mode", choices=("auto", "min_max", "q99"), default="min_max")
    parser.add_argument("--state_already_normalized", action="store_true")
    parser.add_argument("--k_steps", type=int, default=4)
    parser.add_argument("--execution_horizon", type=int, default=4)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--noise_level_end", type=float, default=0.02)
    parser.add_argument("--sde_window_size", type=int, default=1)
    parser.add_argument("--sde_window_range", default="0,2")
    parser.add_argument("--rollout_episodes", type=int, default=16)
    parser.add_argument("--min_rollout_chunks", type=int, default=256)
    parser.add_argument("--max_collection_episodes", type=int, default=32)
    parser.add_argument("--min_successful_rollouts", type=int, default=1)
    parser.add_argument("--max_no_signal_updates", type=int, default=3)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--loss_type", choices=("ppo", "sapo"), default="ppo",
                        help="Actor objective; SAPO uses asymmetric sigmoid ratio gates")
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--tau_pos", type=float, default=1.0,
                        help="SAPO temperature for positive advantages")
    parser.add_argument("--tau_neg", type=float, default=1.0,
                        help="SAPO temperature for non-positive advantages")
    parser.add_argument("--beta_kl", type=float, default=0.01)
    parser.add_argument("--sft_anchor_data", default="")
    parser.add_argument("--sft_anchor_weight", type=float, default=0.25)
    parser.add_argument("--sft_anchor_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--critic_warmup_steps", type=int, default=25)
    parser.add_argument("--max_episode_steps", type=int, default=220)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--preflight_eval_episodes", type=int, default=20)
    parser.add_argument(
        "--eval_freq_updates", type=int, default=1,
        help="Reporting boundary; checkpoint gating is evaluated every update",
    )
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--save_freq_updates", type=int, default=1)
    parser.add_argument("--allow_zero_preflight", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--resume_checkpoint", default="",
                        help="Flow-SDE *_resume.pt checkpoint to continue from")
    return parser


def main(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    if args.k_steps <= 0 or args.execution_horizon <= 0:
        raise ValueError("k_steps and execution_horizon must be positive")
    if args.min_successful_rollouts < 1 or args.max_no_signal_updates < 1:
        raise ValueError("min_successful_rollouts and max_no_signal_updates must be positive")
    if args.loss_type == "sapo" and (args.tau_pos <= 0.0 or args.tau_neg <= 0.0):
        raise ValueError("SAPO tau_pos and tau_neg must be positive")
    if not 0.0 <= args.target_task_weight <= 1.0:
        raise ValueError("target_task_weight must be in [0, 1]")
    if not 0.0 < args.noise_level_end <= args.noise_level:
        raise ValueError("noise_level_end must be positive and no larger than noise_level")
    if args.sft_anchor_weight < 0.0 or args.sft_anchor_batch_size < 1:
        raise ValueError("SFT anchor weight must be non-negative and batch size positive")
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        # Resume files store the parameterization used by the optimizer. A
        # LoRA resume cannot be loaded into a plain Action DiT module because
        # its keys are lora_A/lora_B/base.* rather than raw linear weights.
        resume_payload = torch.load(resume_path, map_location="cpu")
        resume_state = resume_payload.get("action_model", {})
        has_lora = any(".lora_A" in str(key) for key in resume_state)
        requested_scope = str(args.trainable_scope).lower()
        if has_lora and requested_scope != "lora":
            print(
                "[FlowSDEPPO][WARN] Resume checkpoint contains LoRA weights; "
                f"overriding trainable_scope={args.trainable_scope!r} to 'lora'.",
                flush=True,
            )
            args.trainable_scope = "lora"
        elif not has_lora and requested_scope == "lora":
            raise ValueError(
                "TRAINABLE_SCOPE=lora was requested, but resume checkpoint "
                "does not contain LoRA parameters. Use the scope used to create it."
            )
    if (
        args.sft_anchor_weight > 0.0
        and not args.sft_anchor_data
        and not args.preflight_only
    ):
        raise ValueError("--sft_anchor_data is required when --sft_anchor_weight is positive")
    window = tuple(int(item) for item in str(args.sde_window_range).split(","))
    if len(window) != 2 or not 0 <= window[0] < window[1] <= args.k_steps:
        raise ValueError("sde_window_range must be START,END inside [0,k_steps]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    yaml_cfg = OmegaConf.load(args.config_yaml)
    action_horizon = int(yaml_cfg.framework.action_model.future_action_window_size) + 1
    if args.execution_horizon > action_horizon:
        raise ValueError(f"execution_horizon must be <= Action Head horizon {action_horizon}")
    engine_cfg = RLEngineConfig(
        pretrained_checkpoint=args.checkpoint, learning_rate=args.lr, device=args.device,
        dtype=args.dtype, model_variant="original", trainable_scope=args.trainable_scope,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        create_reference_model=True,
    )
    engine = CausalActionGRPOEngine(engine_cfg)
    engine.action_model.num_inference_timesteps = args.k_steps
    engine.action_model.config.num_inference_timesteps = args.k_steps
    env = _build_real_env("libero", yaml_cfg, args)
    if args.train_task_id not in env.task_ids:
        raise ValueError("train_task_id must be included in libero_task_ids for protected paired evaluation")
    trainer_cfg = TrainerConfig(
        grpo=GRPOConfig(action_horizon=action_horizon, execution_horizon=args.execution_horizon,
                        num_inference_steps=args.k_steps, max_episode_steps=args.max_episode_steps,
                        eval_seed=args.eval_seed, rsft_anchor_weight=args.sft_anchor_weight,
                        rsft_anchor_batch_size=args.sft_anchor_batch_size,
                        rsft_online_fraction=0.0),
        engine=engine_cfg, device=args.device, output_dir=args.output_dir, seed=args.seed,
        model_variant="original", normalize_state=not args.state_already_normalized,
        unnorm_key=args.unnorm_key, action_unnorm_mode=args.action_unnorm_mode,
    )
    ppo_cfg = FlowSDEPPOConfig(
        train_task_id=args.train_task_id, sampling_mode=args.sampling_mode,
        target_task_weight=args.target_task_weight, action_horizon=action_horizon,
        execution_horizon=args.execution_horizon, k_steps=args.k_steps, noise_level=args.noise_level,
        noise_level_end=args.noise_level_end,
        sde_window_size=args.sde_window_size, sde_window_range=window,
        rollout_episodes=args.rollout_episodes, min_rollout_chunks=args.min_rollout_chunks,
        max_collection_episodes=args.max_collection_episodes, updates=args.updates,
        min_successful_rollouts=args.min_successful_rollouts,
        max_no_signal_updates=args.max_no_signal_updates,
        ppo_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size, gamma=args.gamma,
        gae_lambda=args.gae_lambda, loss_type=args.loss_type, clip_range=args.clip_range,
        tau_pos=args.tau_pos, tau_neg=args.tau_neg, beta_kl=args.beta_kl,
        sft_anchor_weight=args.sft_anchor_weight, sft_anchor_batch_size=args.sft_anchor_batch_size,
        critic_lr=args.critic_lr, max_episode_steps=args.max_episode_steps,
        critic_warmup_steps=args.critic_warmup_steps,
        eval_episodes=args.eval_episodes, eval_freq_updates=args.eval_freq_updates,
        preflight_eval_episodes=args.preflight_eval_episodes,
        eval_seed=args.eval_seed, save_freq_updates=args.save_freq_updates,
        require_preflight_success=not args.allow_zero_preflight,
        preflight_only=args.preflight_only,
    )
    anchor_dataset = (
        _build_sft_dataset(args.sft_anchor_data, yaml_cfg, "original", None)
        if args.sft_anchor_weight > 0.0 and not args.preflight_only else None
    )
    trainer = FlowSDEPPOTrainer(engine=engine, env=env, raft_extractor=None, flow_codec=None,
                                affordance_generator=None, cfg=trainer_cfg, sft_dataset=None,
                                rsft_anchor_dataset=anchor_dataset,
                                ppo_cfg=ppo_cfg)
    if args.resume_checkpoint:
        trainer.resume_from_checkpoint(args.resume_checkpoint)
    try:
        trainer.train_flow_sde_ppo()
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main(build_parser().parse_args())
