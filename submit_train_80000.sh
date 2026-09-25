#!/bin/bash
#SBATCH --job-name=dit4dit_train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -o logs/%j_train.out
#SBATCH -e logs/%j_train.err

source ~/.bashrc
conda activate /path/to/conda_envs/dit4dit

cd /path/to/DiT4DiT-main

export PYTHONPATH=$(pwd)
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=29688
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "===== DiT4DiT training ====="
date
hostname
which python
which torchrun
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1 \
/path/to/conda_envs/dit4dit/bin/torchrun \
  --nproc_per_node=2 \
  DiT4DiT/training/train.py \
  --config_yaml ./DiT4DiT/config/libero/dit4dit_libero.yaml \
  --framework.name DiT4DiT \
  --framework.cosmos25.base_model /path/to/pretrained/DiT4DiT/Cosmos-Predict2.5-2B \
  --framework.action_model.action_model_type DiT-B \
  --datasets.vla_data.data_root_dir /path/to/data/libero \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.freeze_modules backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.backbone_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.num_warmup_steps 5000 \
  --framework.cosmos25.extract_layer 17 \
  --framework.cosmos25.flow_matching.time_distribution uniform \
  --framework.cosmos25.flow_matching.high_sigma_ratio null \
  --framework.cosmos25.flow_matching.high_sigma_min null \
  --framework.cosmos25.conditional_frame_timestep 0.0001 \
  --datasets.vla_data.action_video_freq_ratio 2 \
  --run_root_dir ./playground/Checkpoints_libero \
  --run_id dit4dit_libero_all_80000_train
