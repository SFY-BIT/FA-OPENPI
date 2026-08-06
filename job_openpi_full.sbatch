#!/bin/bash
#SBATCH --job-name=openpi_full
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group1/junjie008/OPENPI.6dof/openpi_full_%j.out
#SBATCH --error=/data/group1/junjie008/OPENPI.6dof/openpi_full_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/OPENPI.6dof

export WANDB_API_KEY='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'
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/train.py pi05_piper_finetune \
  --exp-name=piper_full_siyuan_1gpu_v3 \
  --overwrite \
  --batch_size=8 \
  --num_workers=0 \
  --num_train_steps=100000 \
  --log_interval=50 \
  --save_interval=2000