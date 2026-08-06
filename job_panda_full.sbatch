#!/bin/bash
#SBATCH --job-name=panda_fl
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/panda_full_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/panda_full_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/openpi-force
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# Panda Full: LIMoE + force input, no force output head
# All params trainable (no LoRA, no freeze)
# Dataset: /data/group1/junjie008/datasets/panda-force-full
# ============================================================
echo "=== Panda Full Fine-tune ==="
python scripts/train.py pi05_panda_full \
    --exp-name=panda_full_v1 \
    --overwrite \
    --batch_size=8 \
    --num_workers=0 \
    --num_train_steps=30000 \
    --log_interval=50 \
    --save_interval=2000

echo "=== Training finished ==="
