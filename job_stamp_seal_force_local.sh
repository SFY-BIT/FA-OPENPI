#!/bin/bash
# Local training script for pi05_force_stamp_seal (local dataset path).
# Run on the workstation with the rlinf conda env.
#
# Usage:
#   cd /mnt/hdd/sfy/openpi-force
#   bash job_stamp_seal_force_local.sh
#
# PREREQUISITE: norm_stats must be computed first.
#   PYTHONPATH=src conda run -n rlinf python scripts/compute_norm_stats.py \
#       --config-name=pi05_force_stamp_seal

set -euo pipefail

cd /mnt/hdd/sfy/openpi-force
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS=ignore

# ============================================================
# Pi0Force Dual-Head: stamp_seal_v2_flexiv (LOCAL)
# Config: pi05_force_stamp_seal (local dataset path)
#   /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv
#
# Action: 6 joints delta + 1 gripper absolute
# Force: 6-dim delta (future - current), improves precision
#
# Gradient routing (scheme B+):
#   VLM <- 1.0*action, LIMoE+expert <- 1.0*action+0.1*force,
#   action_out_proj <- 1.0*action, force_out_proj <- 1.0*force
# ============================================================

echo "=== Local training: pi05_force_stamp_seal ==="
conda run -n rlinf python scripts/train.py pi05_force_stamp_seal \
    --exp-name=stamp_seal_force_local_v1 \
    --overwrite \
    --batch_size=64 \
    --num_workers=0 \
    --num_train_steps=30000 \
    --log_interval=50 \
    --save_interval=2000

echo "=== Training finished ==="
