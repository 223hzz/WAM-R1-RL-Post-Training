## 🎬 Qualitative Demos & Rollout Videos

We provide qualitative rollout execution demonstrations comparing our **WAM-R1** post-trained policies against the baseline imitation policies across diverse manipulation benchmarks.

### 1. LIBERO Benchmark Compilation (2m 07s)
This comprehensive video compilation showcases policy rollouts across the majority of tasks in the **LIBERO** suites (Spatial, Object, Goal, and Long-Horizon 10). It demonstrates how WAM-R1's hybrid-window action reasoning eliminates compounding execution drift and actively self-corrects topological trajectory errors under sparse reward feedback without actuator chatter.

* 📹 **Direct Download / Playback**: [`video/libero_demo.mp4`](video/libero_demo.mp4)

2. RoboCasa-GR1 Humanoid Manipulation (Coming Soon)
⏳ The 29-DoF bi-manual tabletop demonstration video for RoboCasa-GR1 is currently being rendered and curated, and will be updated here shortly.

*(If your browser does not render the inline HTML5 player above, please download the raw `.mp4` directly from the [`video/`](video/) directory).*


## 📦 Checkpoints & Pretrained Models

To adhere to the double-blind review policy and hosting size limits, the full pre-trained base checkpoints and fine-tuned RL policy weights are omitted during the review process. All checkpoints will be publicly released on Hugging Face upon paper acceptance.

Before running training or evaluation, place your base SFT checkpoint at the target path specified inside each shell script (or set via `PRETRAINED_CHECKPOINT` / `pretrained_checkpoint` in `engine.py`).

---

## 🚀 Running Training Scripts

We provide modular bash scripts for post-training across different benchmarks and algorithms (Flow-SDE PPO, GRPO, SAPO, and AWR).

### 1. RoboCasa Benchmark
For tabletop manipulation tasks in RoboCasa, execute the respective Flow-SDE trainer:

```bash
# Flow-SDE PPO
bash run_original_flow_sde_ppo_robocasa.sh

# Flow-SDE GRPO
bash run_original_flow_sde_grpo_robocasa.sh

# Flow-SDE SAPO
bash run_original_flow_sde_sapo_robocasa.sh
2. LIBERO Suite
For benchmark evaluation across LIBERO suites (Spatial, Object, Goal, and LIBERO-10):

Bash
# Few-shot SFT baseline
bash run_libero_fewshot_sft.sh

# Flow-SDE PPO across spatial suites
bash run_original_flow_sde_ppo_spatial_all.sh

# General few-shot suite rollout with Flow-SDE PPO
bash run_fewshot_suite_flow_sde_ppo.sh

# Advantage-Weighted Regression (AWR) on specific tasks / suites
bash run_original_awr_object.sh
bash run_original_awr_task2.sh
bash run_original_awr_task5.sh
3. Cluster & Multi-GPU Execution
For high-performance cluster environments (e.g., A100 / Slurm nodes), use the dedicated hardware-configured launch scripts:

Bash
# Cluster-optimized AWR run on LIBERO-10
bash run_original_awr_libero_10_a100.sh

# Cluster-optimized AWR run on LIBERO Object suite
bash run_original_awr_object_a100.sh
### 4. Cluster Job Submission (`submit/`)

For cluster and distributed execution (e.g., Slurm workload manager on A100 / A800 clusters), batch launch scripts are located in `dit4dit_submit/`:

```bash
# 1. Base Model & SFT Training
sbatch submit/submit_train_80000.sh
sbatch submit/submit_libero_fewshot_sft.sh

# 2. Flow-SDE RL Post-Training (RoboCasa & LIBERO)
sbatch submit/submit_original_flow_sde_sapo_robocasa.sh
sbatch submit/submit_original_flow_sde_ppo_spatial_all.sh
sbatch submit/submit_original_flow_sde_ppo_task5.sh
sbatch submit/submit_original_flow_sde_ppo_task6.sh
sbatch submit/submit_original_flow_sde_sapo_task5.sh
sbatch submit/submit_original_flow_sde_sapo_task6.sh

# 3. Hardware-Optimized GRPO & AWR (A100 / A800 Clusters)
sbatch submit/submit_original_grpo_libero_10_a100.sh
sbatch submit/submit_original_grpo_libero_10_a800.sh
sbatch submit/submit_original_awr_libero_10_a100.sh

# 4. Checkpoint Evaluation
sbatch submit/submit_eval.sh
