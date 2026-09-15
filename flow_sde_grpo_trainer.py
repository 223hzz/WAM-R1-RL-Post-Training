"""Flow-SDE GRPO: critic-free, group-relative post-training for the Action Head.

This is a third actor objective next to ``flow_sde_ppo_trainer``.  It keeps that
module's rollout machinery unchanged -- Flow-SDE stochastic sampling, the
Gaussian transition ratio, the KL anchor to the frozen SFT reference and the
supervised SFT anchor -- and replaces only *how the advantage is produced*:

===========================  ==========================================
``--loss_type ppo``          clipped surrogate over ``V(o)`` + chunk GAE
``--loss_type sapo``         asymmetric sigmoid gate over the same GAE
**this module**              clipped surrogate over a group-relative return
===========================  ==========================================

Why dropping the critic matters here: the four-layer value critic is the least
constrained part of the PPO/SAPO path (it must regress returns from a frozen
Video-DiT feature and a robot state, and it is warm-started on every update).
GRPO removes that estimator entirely and normalizes the sampled returns inside
a group of rollouts instead:

    A_i = (R_i - mean({R_j})) / (std({R_j}) + eps)

``R_i`` is the *undiscounted* episode return of rollout ``i`` (sparse success
plus any environment shaping), and ``A_i`` is broadcast to every SDE transition
of that trajectory.  This is the standard GRPO formulation: an outcome reward,
one scalar advantage per sample, no value function.

Group definition
----------------
Following RLinf-style GRPO for manipulation, a group is **one task** with ``G``
rollouts taken from **different initial states**:

    group(task t) = [ rollout(t, init 0), ..., rollout(t, init G-1) ]

Grouping by task rather than by a single shared initial state is what the
existing environment adapters support without modification: ``reset_for_task``
selects a distinct init state per candidate (LIBERO from its init-state buffer,
RoboCasa via ``reset(seed + index)``), so the relative comparison is across
initial conditions rather than across sampler noise.  The GRPO signal is
therefore about *which initial conditions the policy handles*, and a group of
rollouts that all score identically carries no relative information at all --
see ``has_signal`` below.

Trajectory-level advantage and length bias
------------------------------------------
Sparse reward means the trajectories inside a group differ in length (success
terminates an episode early).  A plain mean over chunks would therefore
*under*-weight exactly the trajectories that succeeded.  Each chunk is
weighted by ``1 / len(trajectory)`` and the minibatch loss is a weighted mean,
so every rollout contributes one unit of weight regardless of its length.

Reused from the PPO path
------------------------
``_policy_loss`` (ratio, clip/SAPO gate, KL to the SFT reference),
``_compute_rsft_anchor_loss``, ``evaluate``, ``_accept_checkpoint`` and the
checkpoint gate/rollback structure are inherited unchanged.  Only collection
(group instead of episode), the advantage estimator and the update are new.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dit4dit_rl_trainer import flow_sde_ppo_trainer as base
from dit4dit_rl_trainer.engine import CausalActionGRPOEngine, RLEngineConfig
from dit4dit_rl_trainer.flow_sde_ppo_trainer import (
    FlowSDEPPOConfig,
    FlowSDEPPOTrainer,
    PPOChunk,
)
from dit4dit_rl_trainer.math_utils import compute_group_advantages
from dit4dit_rl_trainer.trainer import (
    CausalActionGRPOTrainer,
    GRPOConfig,
    TrainerConfig,
    _build_real_env,
    _build_sft_dataset,
)

__all__ = [
    "FlowSDEGRPOConfig",
    "GRPOTrajectory",
    "GRPOGroup",
    "FlowSDEGRPOMixin",
    "FlowSDEGRPOTrainer",
    "build_parser",
    "main",
]


@dataclass
class FlowSDEGRPOConfig(FlowSDEPPOConfig):
    """Flow-SDE GRPO settings.

    Inherits the field set of :class:`FlowSDEPPOConfig` so the shared rollout,
    KL and gate code keeps working.  The critic/GAE fields inherited from there
    (``critic_lr``, ``critic_warmup_steps``, ``value_coef``, ``gamma``,
    ``gae_lambda``) are deliberately unused: GRPO has no value function, and the
    undiscounted group return needs no discount factor.  They are kept rather
    than removed so a resume file written by either path stays loadable.
    """

    # --- group sampling -----------------------------------------------------
    group_size: int = 8
    groups_per_update: int = 4
    # --- group-relative advantage ------------------------------------------
    adv_clip_max: float = 5.0
    global_std: bool = False
    success_bonus: float = 0.0
    reward_history_capacity: int = 100


@dataclass
class GRPOTrajectory:
    """One rollout inside a group: its chunks plus the outcome reward."""

    chunks: List[PPOChunk]
    reward: float
    success: bool
    init_state_id: int
    env_steps: int
    advantage: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "success": bool(self.success),
            "init_state_id": int(self.init_state_id),
            "env_steps": int(self.env_steps),
            "chunks": len(self.chunks),
        }


@dataclass
class GRPOGroup:
    """``G`` rollouts of one task, and the relative advantage they induce."""

    task_id: int
    trajectories: List[GRPOTrajectory]
    reward_mean: float = 0.0
    reward_std: float = 0.0
    has_signal: bool = False
    advantages: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "has_signal": bool(self.has_signal),
            "trajectories": [trajectory.summary() for trajectory in self.trajectories],
        }


class FlowSDEGRPOMixin:
    """Critic-free group-relative policy optimization over Flow-SDE transitions.

    Mixin rather than a plain subclass so the RoboCasa entry point can combine
    it with the RoboCasa environment adapter
    (``robocasa_flow_sde_sapo_trainer.RobocasaFlowSDEPPOTrainer``) without
    inheriting the PPO/SAPO collection loop or the value critic.
    """

    def __init__(self, *args: Any, ppo_cfg: FlowSDEGRPOConfig, **kwargs: Any) -> None:
        # Deliberately the grandparent constructor: ``FlowSDEPPOTrainer.__init__``
        # allocates the four-layer value critic, which GRPO does not use.  Every
        # other attribute it sets (critic bookkeeping aside) is reproduced here.
        CausalActionGRPOTrainer.__init__(self, *args, **kwargs)
        self.ppo_cfg = ppo_cfg
        self.collection_episode = 0
        self.best_target_gain = 0
        self.group_index = 0
        self._pinned_task_id: Optional[int] = None
        print(
            f"[FlowSDEGRPO] critic-free: group_size={ppo_cfg.group_size} "
            f"groups_per_update={ppo_cfg.groups_per_update} "
            f"adv_clip_max={ppo_cfg.adv_clip_max} global_std={ppo_cfg.global_std}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Group construction
    # ------------------------------------------------------------------
    def _select_group_task(self, group_index: int) -> int:
        """Pick the task for a group; mirrors ``_select_rollout_task_id``.

        The same ``--sampling_mode`` therefore means the same task schedule as
        in the PPO/SAPO paths, only advanced per group instead of per episode.
        """
        cfg = self.ppo_cfg
        all_tasks = list(getattr(self.env, "task_ids", range(10)))
        if cfg.sampling_mode == "uniform":
            return int(all_tasks[group_index % len(all_tasks)])
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

    def _select_rollout_task_id(self, episode_id: int) -> int:
        """Pin every candidate of a group to the group's task."""
        if self._pinned_task_id is not None:
            return int(self._pinned_task_id)
        return super()._select_rollout_task_id(episode_id)

    def collect_group(self, group_index: int, noise_level: float) -> GRPOGroup:
        """Collect ``group_size`` rollouts of one task from distinct init states.

        ``collect_episode`` resets through
        ``env.reset_for_task(task, self.collection_episode)``, so pinning the
        task and the collection counter is enough to place each candidate on its
        own initial state without touching the rollout implementation.  The
        counter advances by ``group_size`` per group, so init states are not
        reused between groups until the environment's state list wraps around.
        """
        cfg = self.ppo_cfg
        task_id = self._select_group_task(group_index)
        first_init_index = group_index * cfg.group_size
        trajectories: List[GRPOTrajectory] = []
        self._pinned_task_id = task_id
        try:
            for candidate in range(cfg.group_size):
                self.collection_episode = first_init_index + candidate
                chunks, summary = super().collect_episode(
                    group_index * cfg.group_size + candidate, noise_level
                )
                init_state_id = -1
                try:
                    init_state_id = int(self.env.current_episode_key[1])
                except (AttributeError, RuntimeError, TypeError, IndexError):
                    pass
                # ``summary["reward"]`` already carries the sparse success
                # reward; success_bonus widens the margin between a success and
                # a failure when the environment's own shaping is weak.
                reward = float(summary["reward"]) + cfg.success_bonus * float(
                    bool(summary["success"])
                )
                trajectories.append(
                    GRPOTrajectory(
                        chunks=list(chunks),
                        reward=reward,
                        success=bool(summary["success"]),
                        init_state_id=init_state_id,
                        env_steps=int(summary["env_steps"]),
                    )
                )
                print(
                    f"[FlowSDEGRPO] group={group_index} candidate={candidate + 1}/"
                    f"{cfg.group_size} task={task_id} init_state={init_state_id} "
                    f"chunks={len(chunks)} success={bool(summary['success'])} "
                    f"reward={reward:.3f} steps={int(summary['env_steps'])}",
                    flush=True,
                )
        finally:
            self._pinned_task_id = None

        group = GRPOGroup(task_id=task_id, trajectories=trajectories)
        init_state_ids = [trajectory.init_state_id for trajectory in trajectories]
        if all(state_id >= 0 for state_id in init_state_ids):
            distinct_count = len(set(init_state_ids))
            if distinct_count < len(init_state_ids):
                # Not fatal -- a short init-state list wraps around legitimately
                # -- but the group is then comparing partly identical initial
                # conditions, so the relative advantage is weaker than it looks.
                # Say so rather than let it pass as a full-diversity group.
                print(
                    f"[FlowSDEGRPO][WARN] group={group_index} task={task_id} covered only "
                    f"{distinct_count} distinct initial state(s) across "
                    f"{len(init_state_ids)} candidates; the rest of the group varies only "
                    "in Flow-SDE noise. Check the env's init-state coverage.",
                    flush=True,
                )
        return group

    def _assign_group_advantages(self, group: GRPOGroup) -> GRPOGroup:
        """Set ``A_i`` on every trajectory and broadcast it to their chunks."""
        cfg = self.ppo_cfg
        rewards = torch.tensor(
            [trajectory.reward for trajectory in group.trajectories], dtype=torch.float32
        )
        group.reward_mean = float(rewards.mean())
        group.reward_std = float(rewards.std(unbiased=False))
        advantages = compute_group_advantages(
            rewards,
            eps=1e-8,
            global_std=cfg.global_std,
            reward_history=self.reward_history,
        ).clamp(-cfg.adv_clip_max, cfg.adv_clip_max)
        # Mirror trainer.train_grpo_step: the group's rewards enter the history
        # *after* normalization, so a group never normalizes against itself.
        self.reward_history.append(rewards.detach().cpu())
        if len(self.reward_history) > cfg.reward_history_capacity:
            self.reward_history.pop(0)

        group.advantages = [float(value) for value in advantages.tolist()]
        # GRPO's signal is purely *relative*: if every rollout in the group gets
        # the same advantage then no ordering survives, whatever the absolute
        # reward level is.  This is the real degeneracy test, and it stays
        # correct in both std modes.
        group.has_signal = float(advantages.std(unbiased=False)) > 1e-8
        for trajectory, advantage in zip(group.trajectories, group.advantages):
            trajectory.advantage = advantage if group.has_signal else 0.0
            for chunk in trajectory.chunks:
                chunk.advantage = trajectory.advantage
        return group

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update_grpo(self, groups: List[GRPOGroup]) -> Dict[str, float]:
        """Clipped-surrogate update over group-relative advantages, no critic."""
        cfg = self.ppo_cfg
        # ``(chunk, weight)`` pairs: weight is ``1 / len(trajectory)`` so a short
        # successful rollout counts as much as a long failed one.
        items: List[Tuple[PPOChunk, float]] = []
        for group in groups:
            if not group.has_signal:
                continue
            for trajectory in group.trajectories:
                if not trajectory.chunks:
                    continue
                weight = 1.0 / len(trajectory.chunks)
                items.extend((chunk, weight) for chunk in trajectory.chunks)

        signaled = [group for group in groups if group.has_signal]
        metrics: Dict[str, float] = {
            "groups_collected": float(len(groups)),
            "signaled_groups": float(len(signaled)),
            "degenerate_groups": float(len(groups) - len(signaled)),
            "trajectories_collected": float(
                sum(len(group.trajectories) for group in groups)
            ),
            "trajectories_used": float(
                sum(len(group.trajectories) for group in signaled)
            ),
            "chunks_collected": float(
                sum(len(t.chunks) for group in groups for t in group.trajectories)
            ),
            "chunks_used": float(len(items)),
            "group_reward_mean": float(np.mean([g.reward_mean for g in groups])),
            "group_reward_std": float(np.mean([g.reward_std for g in groups])),
            "rollout_success_rate": float(
                np.mean(
                    [
                        float(trajectory.success)
                        for group in groups
                        for trajectory in group.trajectories
                    ]
                )
            ),
            "advantage_abs_mean": float(
                np.mean(
                    [
                        abs(trajectory.advantage)
                        for group in signaled
                        for trajectory in group.trajectories
                    ]
                )
            )
            if signaled
            else 0.0,
            "advantage_max": float(
                max(
                    (
                        abs(trajectory.advantage)
                        for group in signaled
                        for trajectory in group.trajectories
                    ),
                    default=0.0,
                )
            ),
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "ratio_mean": 1.0,
            "clip_fraction": 0.0,
            "sapo_gate_mean": 0.0,
            "sft_anchor_loss": 0.0,
            "grad_norm": 0.0,
            "optimizer_step_taken": 0.0,
            "anchor_only_update": 0.0,
        }
        self.action_model.train()

        if not items:
            # No group carried relative signal.  Mirroring
            # ``train_grpo_step``, fall back to the supervised anchor alone so
            # the policy keeps its SFT prior instead of drifting on noise; with
            # no anchor configured there is genuinely nothing to learn from.
            anchor_loss, _, _ = self._compute_rsft_anchor_loss()
            if anchor_loss is None:
                print(
                    "[FlowSDEGRPO] every group was degenerate and no SFT anchor is "
                    "configured; optimizer.step skipped",
                    flush=True,
                )
                return metrics
            self.optimizer.zero_grad(set_to_none=True)
            (cfg.sft_anchor_weight * anchor_loss).backward()
            metrics["grad_norm"] = float(
                torch.nn.utils.clip_grad_norm_(
                    self.action_model.parameters(), cfg.max_grad_norm
                )
            )
            self.optimizer.step()
            metrics["optimizer_step_taken"] = 1.0
            metrics["anchor_only_update"] = 1.0
            metrics["sft_anchor_loss"] = float(anchor_loss.detach())
            print(
                "[FlowSDEGRPO] every group was degenerate; applied an SFT-anchor-only "
                "optimizer.step",
                flush=True,
            )
            return metrics

        policy_losses: List[float] = []
        kl_losses: List[float] = []
        ratios: List[float] = []
        clip_fractions: List[float] = []
        sapo_gate_means: List[float] = []
        anchor_losses: List[float] = []
        grad_norms: List[float] = []
        order = list(range(len(items)))
        for _ in range(cfg.ppo_epochs):
            random.shuffle(order)
            for start in range(0, len(order), cfg.minibatch_size):
                batch = [items[index] for index in order[start:start + cfg.minibatch_size]]
                terms: List[torch.Tensor] = []
                weights: List[float] = []
                diagnostics: List[Dict[str, float]] = []
                for chunk, weight in batch:
                    term, info = self._policy_loss(chunk)
                    terms.append(term)
                    weights.append(weight)
                    diagnostics.append(info)
                # Weighted mean rather than a plain mean: trajectory-balanced,
                # and scale-stable because the weights are normalized.
                weight_tensor = torch.tensor(
                    weights, dtype=torch.float32, device=self.device
                )
                actor_loss = (
                    torch.stack(terms) * (weight_tensor / weight_tensor.sum())
                ).sum()
                anchor_loss, _, _ = self._compute_rsft_anchor_loss()
                if anchor_loss is not None:
                    actor_loss = actor_loss + cfg.sft_anchor_weight * anchor_loss
                    anchor_losses.append(float(anchor_loss.detach()))
                self.optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                grad_norms.append(
                    float(
                        torch.nn.utils.clip_grad_norm_(
                            self.action_model.parameters(), cfg.max_grad_norm
                        )
                    )
                )
                self.optimizer.step()
                policy_losses.append(float(actor_loss.detach()))
                kl_losses.append(float(np.mean([item["kl_loss"] for item in diagnostics])))
                ratios.append(float(np.mean([item["ratio_mean"] for item in diagnostics])))
                clip_fractions.append(
                    float(np.mean([item["clip_fraction"] for item in diagnostics]))
                )
                sapo_gate_means.append(
                    float(np.mean([item["sapo_gate_mean"] for item in diagnostics]))
                )

        metrics.update(
            {
                "policy_loss": float(np.mean(policy_losses)),
                "kl_loss": float(np.mean(kl_losses)),
                "ratio_mean": float(np.mean(ratios)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "sapo_gate_mean": float(np.mean(sapo_gate_means)),
                "sft_anchor_loss": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
                "grad_norm": float(np.mean(grad_norms)),
                "optimizer_step_taken": 1.0,
            }
        )
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing: no critic to persist
    # ------------------------------------------------------------------
    def save_resume(self, name: str) -> Path:
        path = Path(self.cfg.output_dir) / "checkpoints" / f"flow_sde_grpo_{name}_resume.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_model": self.action_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "flow_sde_grpo_config": asdict(self.ppo_cfg),
                "flow_sde_algorithm": "grpo",
                "engine_cfg": self.cfg.engine.__dict__,
                "group_index": self.group_index,
                "collection_episode": self.collection_episode,
                "global_step": self.global_step,
                "best_target_gain": self.best_target_gain,
            },
            path,
        )
        return path

    def resume_from_checkpoint(self, path: str) -> None:
        """Restore a GRPO resume checkpoint (critic state is optional)."""
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        required = ("action_model", "optimizer")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Invalid Flow-SDE GRPO resume checkpoint {checkpoint}; missing {missing}")
        self.action_model.load_state_dict(payload["action_model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if "critic" in payload:
            # A PPO/SAPO resume file also carries critic weights. They are
            # ignored rather than loaded: GRPO has no value function, and the
            # action-model/optimizer state is what carries over.
            print(
                "[FlowSDEGRPO] resume checkpoint contains critic weights; ignored "
                "(GRPO is critic-free)",
                flush=True,
            )
        self.group_index = int(payload.get("group_index", 0))
        self.collection_episode = int(payload.get("collection_episode", 0))
        saved_step = payload.get("global_step")
        if saved_step is None:
            # Older checkpoints predate the explicit global_step field.
            match = re.search(r"_update_(\d+)_resume\.pt$", checkpoint.name)
            saved_step = int(match.group(1)) if match else 0
        self.global_step = int(saved_step)
        self.best_target_gain = float(payload.get("best_target_gain", 0.0))
        print(
            f"[FlowSDEGRPO] resumed checkpoint={checkpoint} global_step={self.global_step} "
            f"group_index={self.group_index} collection_episode={self.collection_episode}",
            flush=True,
        )

    def _evaluate_preflight(self) -> Dict[str, Any]:
        """Announce the GRPO context around the inherited preflight gate.

        The preflight itself -- including the zero-success guard that stops a run
        whose SFT policy cannot recover at all -- is deliberately *not* copied
        here: it is a safety check, and a second copy would silently diverge from
        the tested one.  Its own log lines carry a ``[FlowSDEPPO]`` prefix, so
        the GRPO framing is stated explicitly instead.
        """
        print(
            "[FlowSDEGRPO] running the inherited SFT preflight gate before any "
            "online update",
            flush=True,
        )
        return super()._evaluate_preflight()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train_flow_sde_grpo(self) -> None:
        cfg = self.ppo_cfg
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self._evaluate_preflight()
        if cfg.preflight_only:
            print("[FlowSDEGRPO] Preflight-only mode complete; GRPO was not started.", flush=True)
            return
        metrics_path = Path(self.cfg.output_dir) / "flow_sde_grpo_metrics.jsonl"
        best_resume_path = self.save_resume("initial")
        no_signal_updates = 0
        for update in range(self.global_step + 1, cfg.updates + 1):
            progress = (update - 1) / max(1, cfg.updates - 1)
            rollout_noise = (
                (1.0 - progress) * cfg.noise_level + progress * cfg.noise_level_end
            )
            groups: List[GRPOGroup] = []
            for _ in range(cfg.groups_per_update):
                group = self.collect_group(self.group_index, rollout_noise)
                self._assign_group_advantages(group)
                groups.append(group)
                print(
                    f"[FlowSDEGRPO] update={update} group={self.group_index} "
                    f"task={group.task_id} reward_mean={group.reward_mean:.3f} "
                    f"reward_std={group.reward_std:.3f} signal={group.has_signal}",
                    flush=True,
                )
                self.group_index += 1
            metrics = self.update_grpo(groups)
            # ``global_step`` counts completed optimizer updates, so it only
            # moves when the weights actually changed.
            if metrics["optimizer_step_taken"] > 0.0:
                self.global_step += 1
            metrics.update(
                {
                    "update": update,
                    "rollout_noise": rollout_noise,
                    "global_step": self.global_step,
                    "group_summaries": [group.summary() for group in groups],
                }
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")
            print(
                f"[FlowSDEGRPO] update={update}/{cfg.updates} "
                f"groups={int(metrics['groups_collected'])} "
                f"signal={int(metrics['signaled_groups'])}/{int(metrics['groups_collected'])} "
                f"noise={rollout_noise:.4f} "
                f"rollout_success={metrics['rollout_success_rate']:.2%} "
                f"policy={metrics['policy_loss']:.6f} kl={metrics['kl_loss']:.6f} "
                f"ratio={metrics['ratio_mean']:.4f} clip={metrics['clip_fraction']:.2%} "
                f"anchor={metrics['sft_anchor_loss']:.6f} grad={metrics['grad_norm']:.4f} "
                f"adv_abs={metrics['advantage_abs_mean']:.4f}",
                flush=True,
            )
            if cfg.save_freq_updates > 0 and update % cfg.save_freq_updates == 0:
                print(f"[FlowSDEGRPO] resume={self.save_resume(f'update_{update}')}", flush=True)

            if metrics["signaled_groups"] <= 0.0:
                no_signal_updates += 1
            else:
                no_signal_updates = 0

            if metrics["optimizer_step_taken"] <= 0.0:
                # No optimizer step ran, so the weights are bit-identical to the
                # last evaluated ones and the paired gate provably cannot change.
                # Skipping the evaluation here is what keeps a long stretch of
                # degenerate groups from costing a full suite evaluation each.
                print(
                    f"[FlowSDEGRPO] update={update} took no optimizer step; weights "
                    f"unchanged, evaluation and gate skipped "
                    f"(no-signal updates {no_signal_updates}/{cfg.max_no_signal_updates})",
                    flush=True,
                )
            else:
                evaluation = self.evaluate(
                    cfg.eval_episodes, label=f"flow_sde_grpo_update_{update}"
                )
                accepted, gate = self._accept_checkpoint(evaluation)
                if accepted:
                    # Persist accepted weights immediately so a later update
                    # cannot overwrite the only copy of an intermediate best model.
                    resume_path = self.save_resume("best")
                    best_resume_path = resume_path
                    export_path = (
                        Path(self.cfg.output_dir)
                        / "checkpoints"
                        / "flow_sde_grpo_best_pytorch_model.pt"
                    )
                    self.engine.export_pretrained_checkpoint(str(export_path))
                    print(
                        f"[FlowSDEGRPO] accepted best: resume={resume_path}; "
                        f"deployable={export_path}",
                        flush=True,
                    )
                else:
                    print(f"[FlowSDEGRPO] rejected checkpoint: {gate}", flush=True)
                    if best_resume_path.exists():
                        self.engine.load_action_checkpoint(
                            str(best_resume_path), load_optimizer=True
                        )
                        print(
                            f"[FlowSDEGRPO] rolled back rejected update to {best_resume_path}",
                            flush=True,
                        )
                    else:
                        # A manually removed best file must not abort the run.
                        best_resume_path = self.save_resume("recovery")
                        print(
                            f"[FlowSDEGRPO][WARN] best checkpoint missing; created "
                            f"recovery point {best_resume_path}",
                            flush=True,
                        )

            if no_signal_updates >= cfg.max_no_signal_updates:
                raise RuntimeError(
                    f"Flow-SDE GRPO saw no relative signal for {no_signal_updates} "
                    "consecutive updates: every group scored identically, so "
                    "A_i = (R_i - mean) / std is zero and there is nothing to "
                    "optimize. On sparse-reward tasks this usually means the SFT "
                    "policy never succeeds: try a larger --grpo_group_size or "
                    "--grpo_groups_per_update for a wider reward spread, a lower "
                    "--noise_level or --execution_horizon for easier rollouts, or "
                    "let --grpo_success_bonus / --sft_anchor_weight carry the early "
                    "signal. Raise --max_no_signal_updates to keep waiting."
                )
        print(f"[FlowSDEGRPO] final_resume={self.save_resume('final')}", flush=True)


class FlowSDEGRPOTrainer(FlowSDEGRPOMixin, FlowSDEPPOTrainer):
    """LIBERO Flow-SDE GRPO entry point (see :class:`FlowSDEGRPOMixin`)."""


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = "Flow-SDE GRPO (critic-free group-relative) post-training for DiT4DiT"
    parser.add_argument("--grpo_group_size", type=int, default=8,
                        help="Rollouts per group (GRPO's G)")
    parser.add_argument("--grpo_groups_per_update", type=int, default=4,
                        help="Groups collected per optimizer update")
    parser.add_argument("--grpo_adv_clip_max", type=float, default=5.0,
                        help="Clip |A_i| after group normalization")
    parser.add_argument("--grpo_global_std", action="store_true",
                        help="Normalize with the cross-group reward std instead of the "
                             "current group's (off = standard GRPO)")
    parser.add_argument("--grpo_success_bonus", type=float, default=0.0,
                        help="Extra return added to a successful rollout")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.grpo_group_size < 1 or args.grpo_groups_per_update < 1:
        raise ValueError("grpo_group_size and grpo_groups_per_update must be positive")
    if args.k_steps <= 0 or args.execution_horizon <= 0:
        raise ValueError("k_steps and execution_horizon must be positive")
    if args.max_no_signal_updates < 1:
        raise ValueError("max_no_signal_updates must be positive")
    if args.ppo_epochs < 1 or args.minibatch_size < 1:
        raise ValueError("ppo_epochs and minibatch_size must be positive")
    if args.eval_episodes < 1:
        raise ValueError("eval_episodes must be positive")
    if args.loss_type == "sapo" and (args.tau_pos <= 0.0 or args.tau_neg <= 0.0):
        raise ValueError("SAPO tau_pos and tau_neg must be positive")
    if not 0.0 <= args.target_task_weight <= 1.0:
        raise ValueError("target_task_weight must be in [0, 1]")
    if not 0.0 < args.noise_level_end <= args.noise_level:
        raise ValueError("noise_level_end must be positive and no larger than noise_level")
    if args.sft_anchor_weight < 0.0 or args.sft_anchor_batch_size < 1:
        raise ValueError("SFT anchor weight must be non-negative and batch size positive")
    if (
        args.sft_anchor_weight > 0.0
        and not args.sft_anchor_data
        and not args.preflight_only
    ):
        raise ValueError("--sft_anchor_data is required when --sft_anchor_weight is positive")


def _resolve_trainable_scope(args: argparse.Namespace) -> None:
    """Match a resume checkpoint's parameterization (plain Action DiT vs LoRA)."""
    if not args.resume_checkpoint:
        return
    resume_path = Path(args.resume_checkpoint)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    resume_state = torch.load(resume_path, map_location="cpu").get("action_model", {})
    has_lora = any(".lora_A" in str(key) for key in resume_state)
    if has_lora and args.trainable_scope != "lora":
        print(
            "[FlowSDEGRPO][WARN] Resume checkpoint contains LoRA weights; overriding "
            f"trainable_scope={args.trainable_scope!r} to 'lora'.",
            flush=True,
        )
        args.trainable_scope = "lora"
    elif not has_lora and args.trainable_scope == "lora":
        raise ValueError(
            "TRAINABLE_SCOPE=lora was requested, but the resume checkpoint does not "
            "contain LoRA parameters. Use the scope used to create it."
        )


def main(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    _validate(args)
    _resolve_trainable_scope(args)
    window = tuple(int(item) for item in str(args.sde_window_range).split(","))
    if len(window) != 2 or not 0 <= window[0] < window[1] <= args.k_steps:
        raise ValueError("sde_window_range must be START,END inside [0,k_steps]")
    # The inherited PPO/SAPO knobs that GRPO does not read. Named here so a
    # launcher that still sets them fails loudly in the log instead of silently
    # having no effect.
    print(
        "[FlowSDEGRPO] ignored inherited knobs: rollout_episodes, min_rollout_chunks, "
        "min_successful_rollouts, critic_lr, critic_warmup_steps, value_coef, gamma, "
        "gae_lambda (GRPO has no critic and no GAE)",
        flush=True,
    )
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
        raise ValueError(
            "train_task_id must be included in libero_task_ids for protected paired evaluation"
        )
    trainer_cfg = TrainerConfig(
        grpo=GRPOConfig(
            action_horizon=action_horizon, execution_horizon=args.execution_horizon,
            num_inference_steps=args.k_steps, max_episode_steps=args.max_episode_steps,
            eval_seed=args.eval_seed, rsft_anchor_weight=args.sft_anchor_weight,
            rsft_anchor_batch_size=args.sft_anchor_batch_size, rsft_online_fraction=0.0,
        ),
        engine=engine_cfg, device=args.device, output_dir=args.output_dir, seed=args.seed,
        model_variant="original", normalize_state=not args.state_already_normalized,
        unnorm_key=args.unnorm_key, action_unnorm_mode=args.action_unnorm_mode,
    )
    grpo_cfg = FlowSDEGRPOConfig(
        train_task_id=args.train_task_id, sampling_mode=args.sampling_mode,
        target_task_weight=args.target_task_weight, action_horizon=action_horizon,
        execution_horizon=args.execution_horizon, k_steps=args.k_steps,
        noise_level=args.noise_level, noise_level_end=args.noise_level_end,
        sde_window_size=args.sde_window_size, sde_window_range=window,
        updates=args.updates, max_no_signal_updates=args.max_no_signal_updates,
        ppo_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
        loss_type=args.loss_type, clip_range=args.clip_range,
        tau_pos=args.tau_pos, tau_neg=args.tau_neg, beta_kl=args.beta_kl,
        sft_anchor_weight=args.sft_anchor_weight,
        sft_anchor_batch_size=args.sft_anchor_batch_size,
        max_episode_steps=args.max_episode_steps, max_grad_norm=1.0,
        eval_episodes=args.eval_episodes, eval_freq_updates=args.eval_freq_updates,
        preflight_eval_episodes=args.preflight_eval_episodes,
        eval_seed=args.eval_seed, save_freq_updates=args.save_freq_updates,
        require_preflight_success=not args.allow_zero_preflight,
        preflight_only=args.preflight_only,
        group_size=args.grpo_group_size,
        groups_per_update=args.grpo_groups_per_update,
        adv_clip_max=args.grpo_adv_clip_max, global_std=args.grpo_global_std,
        success_bonus=args.grpo_success_bonus,
    )
    anchor_dataset = (
        _build_sft_dataset(args.sft_anchor_data, yaml_cfg, "original", None)
        if args.sft_anchor_weight > 0.0 and not args.preflight_only else None
    )
    # Looked up as a module global so the RoboCasa entry point can rebind it,
    # exactly as the PPO/SAPO entry points do.
    trainer = FlowSDEGRPOTrainer(
        engine=engine, env=env, raft_extractor=None, flow_codec=None,
        affordance_generator=None, cfg=trainer_cfg, sft_dataset=None,
        rsft_anchor_dataset=anchor_dataset, ppo_cfg=grpo_cfg,
    )
    if args.resume_checkpoint:
        trainer.resume_from_checkpoint(args.resume_checkpoint)
    try:
        trainer.train_flow_sde_grpo()
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main(build_parser().parse_args())
