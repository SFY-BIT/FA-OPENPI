#!/bin/bash
#SBATCH --job-name=eb_eef
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/erase_board_eef_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/erase_board_eef_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# EEF 双空间位姿 loss 全参训练（远端）
#   config: pi05_force_erase_board_eef_remote
#   repo_id: /data/group1/junjie008/datasets/erase_board_flexiv_ft60
#   hot-start: 本地 3w (29999) checkpoint（需先 put 上传到远端对应路径）
#   EEF loss: 0.7 关节 + 0.3 EEF（FK 末端 0.211m, pos0.3+rot1.0）
#   前置：FA-openpi 代码已更新（含 remote config）；
#         erase_board_flexiv_ft60 数据集 + 29999 checkpoint 已上传。
# ============================================================

echo "=== EEF dual-space pose loss training (hot-start 29999) ==="
python scripts/train.py pi05_force_erase_board_eef_remote \
    --exp-name=erase_board_eef \
    --overwrite \
    --batch_size=32 \
    --num_workers=0 \
    --num_train_steps=50000 \
    --log_interval=10 \
    --save_interval=2000 \
    --keep_period=10000

echo "=== Training finished ==="
