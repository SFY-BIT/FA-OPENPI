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
# EEF 双空间位姿 loss 全参训练（远端）- 折中权重版
#   config: pi05_force_erase_board_eef_remote
#   repo_id: /data/group1/junjie008/datasets/erase_board_flexiv_ft60
#   hot-start: 3w (29999) checkpoint（weight_loader 自动载入, 不 resume!）
#   EEF loss: 0.6 关节 + 0.4 EEF（FK 末端 0.211m, pos0.3+rot2.0）
#   ⚠️ 不带 --resume: 从 3w 重开, 干净 optimizer (resume 切权重会 Adam 状态不匹配)
#   前置：FA-openpi 代码已更新（含 remote config 0.6/2.0）;
#         远端需有 29999 checkpoint。
# ============================================================

echo "=== EEF training (0.6 joint + 0.4 EEF, rot2.0, fresh from 29999) ==="
python scripts/train.py pi05_force_erase_board_eef_remote \
    --exp-name=erase_board_eef_v3 \
    --batch_size=64 \
    --num_workers=0 \
    --num_train_steps=50000 \
    --log_interval=10 \
    --save_interval=500 \
    --keep_period=2000

echo "=== Training finished ==="
