#!/bin/bash
#SBATCH --job-name=x5_nf
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_%j.err

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
# RoboDojo ARX X5 Dual-Arm NoForce: pure Pi0.5 baseline
# All params trainable, no LoRA, no freeze, no force
# 14-dim state/action (6+1+6+1 joints+gripper), 3 assembly tasks
# Dataset: /data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21
#
# V2: Resume from V1 checkpoint (step 29999), increase batch_size 8→16
#     V1 only trained 1.33 epochs (bs=8, 30K steps = 240K samples)
#     V2 resumes from step 30000, trains to 150000 (extra 120K steps)
#     Total: 240K + 1.92M = 2.16M samples ≈ 12 epochs
#     LR stays 5e-5 (config has decay_steps=1M, effectively constant)
# ============================================================
echo "=== RoboDojo X5 NoForce Full Fine-tune (V2 Resume) ==="
python scripts/train.py pi05_robodojo_x5_noforce \
    --exp-name=robodojo_x5_noforce_v1 \
    --resume \
    --batch_size=16 \
    --num_workers=0 \
    --num_train_steps=150000 \
    --log_interval=50 \
    --save_interval=2000

echo "=== Training finished ==="
