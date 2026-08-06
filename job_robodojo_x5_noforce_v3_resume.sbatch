#!/bin/bash
#SBATCH --job-name=x5_v3r
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_v3_resume_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_v3_resume_%j.err

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
# RoboDojo ARX X5 V3 RESUME: 从最新 checkpoint 恢复训练
#
# 用途: V3 训练被中断后，从最近保存的 checkpoint 恢复。
#       --resume 会自动找到 checkpoint_dir 下最新的 step 目录。
#
# 前提:
#   - checkpoint 目录已存在（之前训练过）:
#     /data/group1/junjie008/openpi-force/checkpoints/pi05_robodojo_x5_noforce/robodojo_x5_noforce_v3/
#   - 目录下有按 save_interval 保存的 step 子目录（如 20000/）
#
# 注意:
#   - --resume 和 --overwrite 不能同时使用
#   - --resume 会自动加载最新 step 的 params + train_state（含 optimizer 状态）
#   - wandb 会用同一个 run_id 继续记录
#   - num_train_steps 保持 150000（总目标步数，不是剩余步数）
# ============================================================

echo "=== Resume training from latest checkpoint ==="
python scripts/train.py pi05_robodojo_x5_noforce \
    --exp-name=robodojo_x5_noforce_v3 \
    --resume \
    --batch_size=32 \
    --num_workers=0 \
    --num_train_steps=150000 \
    --log_interval=50 \
    --save_interval=2000

echo "=== Training finished ==="
