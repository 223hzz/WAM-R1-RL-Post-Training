"""Dedicated RoboCasa Flow-SDE SAPO entry point.

The original LIBERO SAPO entry point remains untouched.  This module reuses
its Critic/GAE/SAPO implementation and injects only the RoboCasa adapter and
RoboCasa-specific RGB/state/action preprocessing.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import flow_sde_ppo_trainer as base
from .robocasa_env import RobocasaEnvWrapper


class RobocasaFlowSDEPPOTrainer(base.FlowSDEPPOTrainer):
    skip_preflight = False
    baseline_episodes = 0

    def collect_episode(self, episode_id: int, noise_level: float):
        """Collect one episode and retain every chunk for SAPO/PPO.

        ``min_successful_rollouts`` is a signal gate in the base trainer: it
        controls whether an update is safe to run. It must not turn the
        additional collection budget into a success-only replay buffer. Failed
        trajectories provide the negative advantage signal that distinguishes
        online policy optimization from supervised fine-tuning.
        """
        chunks, summary = super().collect_episode(episode_id, noise_level)
        summary["chunks_discarded"] = 0
        summary["chunks_retained"] = len(chunks)
        return chunks, summary
    baseline_successes = 0
    def _observe_to_images(self, obs: Dict[str, Any], prev_rgb: Optional[np.ndarray] = None):
        return np.asarray(obs["rgb"], dtype=np.uint8), None, None

    def _prepare_env_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(normalized_actions, dtype=np.float32)
        if self.action_norm_stats is None:
            return actions[..., :29]
        low = np.asarray(self.action_norm_stats["min"], dtype=np.float32)
        high = np.asarray(self.action_norm_stats["max"], dtype=np.float32)
        dim = min(29, int(low.shape[-1]))
        actions = np.clip(actions[..., :dim].copy(), -1.0, 1.0)
        mask = np.asarray(self.action_norm_stats.get("mask", np.ones(dim, dtype=bool)), dtype=bool)[:dim]
        return np.where(mask, 0.5 * (actions + 1.0) * (high[:dim] - low[:dim]) + low[:dim], actions)

    def _evaluate_preflight(self) -> Dict[str, Any]:
        if self.skip_preflight:
            self.preflight_records = {}
            print("[RoboCasa SAPO] preflight skipped; starting online SAPO updates", flush=True)
            return {"successes": 0, "episodes": 0, "records": [], "per_task": {}}
        baseline = self.evaluate(self.ppo_cfg.eval_episodes, label="robocasa_sapo_preflight")
        self.preflight_records = self._record_map(baseline)
        print(
            f"[RoboCasa SAPO] preflight suite success="
            f"{baseline['successes']}/{baseline['episodes']}", flush=True,
        )
        if self.ppo_cfg.require_preflight_success and baseline["successes"] == 0:
            raise RuntimeError("RoboCasa SFT preflight produced zero successful episodes")
        return baseline

    def _accept_checkpoint(self, candidate: Dict[str, Any]):
        if not self.skip_preflight:
            return super()._accept_checkpoint(candidate)
        successes = int(candidate.get("successes", 0))
        episodes = int(candidate.get("episodes", 0))
        candidate_rate = successes / max(1, episodes)
        baseline_rate = (
            float(self.baseline_successes) / max(1, int(self.baseline_episodes))
            if int(self.baseline_episodes) > 0
            else float(self.baseline_successes)
        )
        reference_rate = max(float(self.best_target_gain), baseline_rate)
        accepted = candidate_rate > reference_rate
        if accepted:
            self.best_target_gain = candidate_rate
        details = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": candidate_rate,
            "baseline_successes": int(self.baseline_successes),
            "baseline_episodes": int(self.baseline_episodes),
            "baseline_rate": baseline_rate,
            "best_rate": max(float(self.best_target_gain), baseline_rate),
            "sampling_mode": "robocasa_suite_eval",
        }
        print(f"[RoboCasa SAPO Gate] {details} accepted={accepted}", flush=True)
        return accepted, details


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.add_argument("--robocasa_tasks", required=True)
    parser.add_argument("--robocasa_max_episode_steps", type=int, default=720)
    parser.add_argument("--robocasa_render_gpu_device_id", type=int, default=0)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--baseline_successes", type=int, default=0)
    parser.add_argument("--baseline_episodes", type=int, default=0)
    return parser


def main(args: argparse.Namespace) -> None:
    # Keep the well-tested base main wiring (checkpoint, Critic, SAPO, saves)
    # while substituting only the environment constructor and trainer class.
    base.FlowSDEPPOTrainer = RobocasaFlowSDEPPOTrainer
    RobocasaFlowSDEPPOTrainer.skip_preflight = bool(args.skip_preflight)
    RobocasaFlowSDEPPOTrainer.baseline_successes = int(args.baseline_successes)
    RobocasaFlowSDEPPOTrainer.baseline_episodes = int(args.baseline_episodes)
    base._build_real_env = lambda _name, _cfg, parsed: RobocasaEnvWrapper(
        parsed.robocasa_tasks,
        seed=parsed.seed,
        max_episode_steps=parsed.robocasa_max_episode_steps,
        render_gpu_device_id=parsed.robocasa_render_gpu_device_id,
    )
    args.state_already_normalized = True
    base.main(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
