"""Dedicated RoboCasa Flow-SDE PPO entry point.

``flow_sde_ppo_trainer`` already implements both actor objectives behind
``--loss_type`` (:436-439 is the clipped PPO objective, :427-435 is the SAPO
gate).  The RoboCasa adapter in ``robocasa_flow_sde_sapo_trainer`` only
overrides environment plumbing -- ``collect_episode``, ``_observe_to_images``,
``_prepare_env_actions``, ``_evaluate_preflight`` and ``_accept_checkpoint`` --
none of which read ``loss_type``.  So the RoboCasa path needs no new loss code;
it only needs an entry point that cannot silently fall back to the other
objective.

This module is the PPO counterpart of ``robocasa_flow_sde_sapo_trainer`` and
enforces ``--loss_type ppo`` the same way ``flow_sde_sapo_trainer`` enforces
``sapo``.  It imports the trainer class and the RoboCasa argument set from the
SAPO module rather than restating them: the class is algorithm-agnostic (its
name is already ``RobocasaFlowSDEPPOTrainer``) and the SAPO module is a working
file that this PPO path deliberately does not modify.
"""

from __future__ import annotations

import argparse

from . import flow_sde_ppo_trainer as base
from .robocasa_env import RobocasaEnvWrapper
from .robocasa_flow_sde_sapo_trainer import RobocasaFlowSDEPPOTrainer, build_parser

__all__ = ["RobocasaFlowSDEPPOTrainer", "build_parser", "main"]


def main(args: argparse.Namespace) -> None:
    if args.loss_type != "ppo":
        raise ValueError(
            "robocasa_flow_sde_ppo_trainer requires --loss_type ppo "
            f"(got {args.loss_type!r}); use robocasa_flow_sde_sapo_trainer for SAPO"
        )

    # ``base.main`` builds its trainer through the module-global
    # ``FlowSDEPPOTrainer`` name, so this rebinding is required; importing the
    # class into this module alone would have no effect on the base entry point.
    base.FlowSDEPPOTrainer = RobocasaFlowSDEPPOTrainer
    RobocasaFlowSDEPPOTrainer.skip_preflight = bool(args.skip_preflight)
    RobocasaFlowSDEPPOTrainer.baseline_successes = int(args.baseline_successes)
    RobocasaFlowSDEPPOTrainer.baseline_episodes = int(args.baseline_episodes)

    # Substitute only the environment constructor; rollout, critic, GAE and the
    # checkpoint gate stay on the well-tested base wiring.
    base._build_real_env = lambda _name, _cfg, parsed: RobocasaEnvWrapper(
        parsed.robocasa_tasks,
        seed=parsed.seed,
        max_episode_steps=parsed.robocasa_max_episode_steps,
        render_gpu_device_id=parsed.robocasa_render_gpu_device_id,
    )
    # RoboCasa observations are already normalized by the adapter, matching the
    # SAPO path.
    args.state_already_normalized = True

    base.main(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
