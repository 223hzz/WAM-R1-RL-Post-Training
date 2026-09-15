"""Dedicated RoboCasa Flow-SDE GRPO entry point.

The asymmetry with the PPO/SAPO pair is small but worth stating: those two
share one algorithm-agnostic RoboCasa class
(``robocasa_flow_sde_sapo_trainer.RobocasaFlowSDEPPOTrainer``) because
``flow_sde_ppo_trainer`` implements both objectives behind ``--loss_type``.  GRPO
adds a *critic-free* update that the base class does not have, so it lives in
``flow_sde_grpo_trainer.FlowSDEGRPOMixin`` and this module is the RoboCasa
combination of that mixin with the RoboCasa environment adapter.

Everything RoboCasa-specific is inherited rather than restated:

* ``_observe_to_images`` / ``_prepare_env_actions`` -- the RGB/state/action
  preprocessing, taken as-is from the SAPO adapter.
* ``collect_episode`` -- likewise; for GRPO its ``chunks_discarded=0`` metadata
  is informational only, since ``collect_group`` retains every chunk by
  construction.

``_evaluate_preflight`` and ``_accept_checkpoint`` are re-implemented here
instead of inherited.  They are the same logic as the SAPO adapter's, but the
SAPO versions hardcode a ``[RoboCasa SAPO]`` prefix in every log line the gate
writes, and mislabelled gate output is exactly the kind of thing that makes a
cluster log unreadable after the fact.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from dit4dit_rl_trainer import flow_sde_grpo_trainer as grpo
from dit4dit_rl_trainer.flow_sde_ppo_trainer import FlowSDEPPOTrainer
from dit4dit_rl_trainer.robocasa_env import RobocasaEnvWrapper
from dit4dit_rl_trainer.robocasa_flow_sde_sapo_trainer import RobocasaFlowSDEPPOTrainer

__all__ = ["RobocasaFlowSDEGRPOTrainer", "build_parser", "main"]


class RobocasaFlowSDEGRPOTrainer(grpo.FlowSDEGRPOMixin, RobocasaFlowSDEPPOTrainer):
    """RoboCasa-GR1 Flow-SDE GRPO: GRPO update on the RoboCasa adapter."""

    skip_preflight = False
    baseline_episodes = 0
    baseline_successes = 0

    def _evaluate_preflight(self) -> Dict[str, Any]:
        if self.skip_preflight:
            self.preflight_records = {}
            print(
                "[RoboCasa GRPO] preflight skipped; starting online GRPO updates",
                flush=True,
            )
            return {"successes": 0, "episodes": 0, "records": [], "per_task": {}}
        baseline = self.evaluate(
            self.ppo_cfg.eval_episodes, label="robocasa_grpo_preflight"
        )
        self.preflight_records = self._record_map(baseline)
        print(
            f"[RoboCasa GRPO] preflight suite success="
            f"{baseline['successes']}/{baseline['episodes']}",
            flush=True,
        )
        if self.ppo_cfg.require_preflight_success and baseline["successes"] == 0:
            raise RuntimeError("RoboCasa SFT preflight produced zero successful episodes")
        return baseline

    def _accept_checkpoint(self, candidate: Dict[str, Any]):
        if not self.skip_preflight:
            # Explicit rather than ``super()``: this class is not a subclass of
            # the RoboCasa PPO class by design (see the module docstring), so the
            # paired-episode gate is taken straight from the shared PPO trainer.
            return FlowSDEPPOTrainer._accept_checkpoint(self, candidate)
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
        print(f"[RoboCasa GRPO Gate] {details} accepted={accepted}", flush=True)
        return accepted, details


def build_parser() -> argparse.ArgumentParser:
    parser = grpo.build_parser()
    # Same RoboCasa argument set as the SAPO adapter, restated here because the
    # SAPO module builds it inline and this path must not modify that file.
    parser.add_argument("--robocasa_tasks", required=True)
    parser.add_argument("--robocasa_max_episode_steps", type=int, default=720)
    parser.add_argument("--robocasa_render_gpu_device_id", type=int, default=0)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--baseline_successes", type=int, default=0)
    parser.add_argument("--baseline_episodes", type=int, default=0)
    return parser


def main(args: argparse.Namespace) -> None:
    RobocasaFlowSDEGRPOTrainer.skip_preflight = bool(args.skip_preflight)
    RobocasaFlowSDEGRPOTrainer.baseline_successes = int(args.baseline_successes)
    RobocasaFlowSDEGRPOTrainer.baseline_episodes = int(args.baseline_episodes)

    # ``grpo.main`` builds its trainer through the module-global
    # ``FlowSDEGRPOTrainer`` name, so this rebinding is required; importing the
    # class into this module alone would have no effect on that entry point.
    grpo.FlowSDEGRPOTrainer = RobocasaFlowSDEGRPOTrainer
    # Substitute only the environment constructor; rollout, group advantage, the
    # SFT anchor and the checkpoint gate stay on the shared GRPO wiring.
    grpo._build_real_env = lambda _name, _cfg, parsed: RobocasaEnvWrapper(
        parsed.robocasa_tasks,
        seed=parsed.seed,
        max_episode_steps=parsed.robocasa_max_episode_steps,
        render_gpu_device_id=parsed.robocasa_render_gpu_device_id,
    )
    # RoboCasa observations are already normalized by the adapter, matching the
    # PPO/SAPO paths.
    args.state_already_normalized = True

    grpo.main(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
