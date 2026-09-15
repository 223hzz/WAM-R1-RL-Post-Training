# dit4dit_rl_trainer package
# Action-RL 在线强化学习后训练框架
#
# 严格约束：本文件夹不修改 dit4dit_multistream/ 内的任何原始网络代码，
# 仅通过 import 引入模型。
#
# 模块：
#   - engine.py           : Step 1 网络冻结与解耦引擎
#   - sde_rollout.py       : Step 2 SDE 混合采样
#   - math_utils.py        : Step 3 Gaussian transition log-prob + GRPO loss
#   - envs.py              : LIBERO fixed-initial-state environment adapter
#   - trainer.py           : Step 4 两阶段训练主循环 (含 __main__ 入口)
#   - verify_pipeline.py   : torch 端到端验证 (10 项测试, 需 torch 环境)
#   - verify_math_numpy.py : 纯 numpy 数学验证 (8 项, 无需 torch)
#
# 运行方式:
#   # Mock smoke test (验证训练循环, 无需 checkpoint/环境)
#   python -m dit4dit_rl_trainer.trainer --mock
#
#   # 真实训练 (需 SFT checkpoint + 仿真环境)
#   python -m dit4dit_rl_trainer.trainer --real \
#     --config_yaml dit4dit_multistream/dit4dit_multistream_libero.yaml \
#     --checkpoint /path/to/sft_checkpoint.pt --env libero
