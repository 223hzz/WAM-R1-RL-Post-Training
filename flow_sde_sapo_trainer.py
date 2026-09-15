"""Task-5 Flow-SDE SAPO entry point.

The rollout and critic implementation live in ``flow_sde_ppo_trainer``. This
entry point gives SAPO its own launch target while preventing an accidental
fallback to the clipped PPO objective.
"""

from __future__ import annotations

from .flow_sde_ppo_trainer import build_parser, main


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    if parsed_args.loss_type != "sapo":
        raise ValueError("flow_sde_sapo_trainer requires --loss_type sapo")
    main(parsed_args)
