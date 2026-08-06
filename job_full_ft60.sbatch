#!/bin/bash
#SBATCH --job-name=ft60_full_k16
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/ft60_full_k16_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/ft60_full_k16_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

# 与主干 sbatch 相同的 WANDB key
export WANDB_API_KEY='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'
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# FA-openpi 全参 K=16（ft_history 分段编码）远端训练
#   config: pi05_force_stamp_seal_ft60_k16_remote
#   repo_id: /data/group1/junjie008/datasets/stamp_seal_v2_flexiv_ft60
#   LR 布局: base=5e-5 (PaliGemma/action/time_mlp/RouterWeights)
#            new=5×  (limoe 专家/force/state_proj)
#   冷启动 pi05_base（不 resume）
# 前置：FA-openpi 全部内容（除 checkpoint）已上传到 /data/group1/junjie008/FA-openpi；
#       ft60 数据集（含 norm_stats.json）已上传到远端。
# ============================================================

echo "=== Fresh full-params K=16 training from pi05_base ==="
python scripts/train.py pi05_force_stamp_seal_ft60_k16_remote \
    --exp-name=ft60_full_k16 \
    --overwrite \
    --batch_size=32 \
    --num_workers=0 \
    --num_train_steps=50000 \
    --log_interval=10 \
    --save_interval=2000 \
    --keep_period=10000

echo "=== Training finished ==="
