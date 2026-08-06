#!/bin/bash
#SBATCH --job-name=force_2st
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/force2st_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/force2st_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/openpi-force
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS=ignore
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# ============================================================
# Stage 1: Pure vision pi05 pretraining (no MoE, batch=8)
# ============================================================
echo "=== Stage 1: Pure vision pretraining ==="
python scripts/train.py pi05_usb_pretrain \
    --exp-name=usb_stage1_pure_30k \
    --overwrite \
    --batch_size=8 \
    --num_workers=0 \
    --num_train_steps=30000 \
    --log_interval=50 \
    --save_interval=2000

STAGE1_CKPT="/data/group1/junjie008/openpi-force/checkpoints/pi05_usb_pretrain/usb_stage1_pure_30k/30000/params"

# ============================================================
# Stage 2: LoRA + LIMoE + Force (batch=32, 10k steps)
# ============================================================
echo "=== Stage 2: LoRA + LIMoE + Force ==="
python scripts/train.py pi05_force_lora_stage2 \
    --exp-name=usb_stage2_force_lora \
    --overwrite \
    --batch_size=32 \
    --num_workers=0 \
    --num_train_steps=10000 \
    --log_interval=50 \
    --save_interval=2000 \
    --weight_loader.params_path="$STAGE1_CKPT"

echo "=== Training finished ==="
