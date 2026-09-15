# dit4dit_rl_trainer package
# Action-RL online reinforcement-learning post-training framework
#
# Hard constraint: nothing in this folder modifies any original network code under
# dit4dit_multistream/; the model is brought in by import only.
#
# Modules:
#   - engine.py           : Step 1 network freezing and decoupling engine
#   - sde_rollout.py       : Step 2 hybrid SDE sampling
#   - math_utils.py        : Step 3 Gaussian transition log-prob + GRPO loss
#   - envs.py              : LIBERO fixed-initial-state environment adapter
#   - trainer.py           : Step 4 two-stage training loop (with a __main__ entry point)
#   - verify_pipeline.py   : end-to-end torch verification (10 tests, needs a torch environment)
#   - verify_math_numpy.py : pure-numpy math verification (8 tests, no torch required)
#
# Usage:
#   # Mock smoke test (verifies the training loop; no checkpoint or simulator)
#   python -m dit4dit_rl_trainer.trainer --mock
#
#   # Real training (needs an SFT checkpoint + a simulator)
#   python -m dit4dit_rl_trainer.trainer --real \
#     --config_yaml dit4dit_multistream/dit4dit_multistream_libero.yaml \
#     --checkpoint /path/to/sft_checkpoint.pt --env libero
