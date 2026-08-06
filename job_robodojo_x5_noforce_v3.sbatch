#!/bin/bash
#SBATCH --job-name=x5_v3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_v3_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_v3_%j.err

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
# RoboDojo ARX X5 Dual-Arm NoForce V3: 3-Camera + State Fix
#
# V1/V2 root causes (all fixed in V3):
#   1. right_wrist_0_rgb hard-coded to zeros + masked → right arm blind
#   2. discrete_state_input=False → proprioceptive state completely dropped
#      (model was pure-vision, no joint angle feedback)
#   3. delta mask make_bool_mask(14) → gripper (dim 6,13) wrongly converted
#      to delta; should stay absolute like Aloha reference
#   4. action_horizon=30 → too short for assembly tasks; Aloha ref uses 50
#
# V3 fixes:
#   - observation_right_wrist_image_key added → 3 real cameras
#   - discrete_state_input removed → __post_init__ auto-sets True (pi05 std)
#   - delta mask → make_bool_mask(6, -1, 6, -1) (gripper absolute)
#   - action_horizon=50 (matches pi05_base_aloha_full_sim_arx-x5_seed_0)
#
# IMPORTANT: norm_stats MUST be recomputed because delta mask changed.
#   Old stats had gripper as delta (mean≈0); new stats need gripper as
#   absolute (mean≈0.8).  Step 1 below regenerates them.
#
# Training: FRESH start from pi05_base (NOT resume from V1/V2).
# V1/V2 had wrong attention patterns (no state tokens, 2 cameras).
#
# Config: pi05_robodojo_x5_noforce (remote dataset path)
# 14-dim state/action, 3 cameras (head + left_wrist + right_wrist)
# bs=16, lr=5e-5, 150K steps (~35 epochs @ 300eps × ~230frm/16)
# ============================================================

# norm_stats 已在本地用 pi05_robodojo_x5_noforce_local 预先计算并上传
echo "=== Fresh training from pi05_base ==="
python scripts/train.py pi05_robodojo_x5_noforce \
    --exp-name=robodojo_x5_noforce_v3 \
    --overwrite \
    --batch_size=16 \
    --num_workers=0 \
    --num_train_steps=150000 \
    --log_interval=50 \
    --save_interval=2000

echo "=== Training finished ==="
