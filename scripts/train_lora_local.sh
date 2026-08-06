#!/bin/bash
# ============================================================
# 本地 LoRA K=16 训练（ft_history 分段编码）
#   VLM: LoRA (gemma_2b_lora) 冻结主干 | 视觉全参 | LIMoE/新模块可训练
#   hot-start 自 openpi-force 12000 checkpoint
#   新模块 (ft_encoder/ft_proj) 10× LR
# 运行前：确认 GPU 空闲；建议在 tmux 中运行
# ============================================================
set -euo pipefail

source /home/sfy/miniconda3/etc/profile.d/conda.sh
conda activate rlinf
cd /mnt/hdd/sfy/FA-openpi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

echo "=== LoRA K=16 (hot-start openpi-force 12000) ==="
# bs=4: GPU0 被 liuxu 占 10.2G，可用 ~34.8G；bs=8(~32G) 会 OOM，减半稳妥
python -u scripts/train.py pi05_force_stamp_seal_ft60_forcevla_lora_k16 \
    --exp-name=ft60_k16 \
    --overwrite \
    --batch_size=4 \
    --num_workers=2 \
    --num_train_steps=30000 \
    --log_interval=10 \
    --save_interval=2000 \
    --keep_period=10000 \
    2>&1 | tee logs/ft60_k16.log

echo "=== Training finished ==="
