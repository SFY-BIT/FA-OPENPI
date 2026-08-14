#!/bin/bash
# ============================================================
# total_2task 3 config 串行训练 (每个 2000 步, 只保留最后 ckpt)
#   顺序: eef_only → joint_only → eef_joint (GPU 0 串行)
#   bs=8 | 每步 ~18s → 每 config ~1-2h, 总共 ~3-6h
#   save_interval=100000 > 2000 → 只在最后一步(1999)保存
#   → 每个 config 只保留最后一个 checkpoint
# 运行: tmux 中 bash scripts/train_total_task_3config_local.sh
# ============================================================
set -euo pipefail

source /home/sfy/miniconda3/etc/profile.d/conda.sh
conda activate rlinf
cd /mnt/hdd/sfy/FA-openpi

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

# 每 config 2000 步; save_interval 远大于步数 → 仅最后一步保存
COMMON="--num_train_steps=2000 --save_interval=100000 --log_interval=10 --batch_size=8 --num_workers=2"

echo "############################################################"
echo "# [1/3] eef_only_local (total_2task_flexiv_eef, EEF 空间 loss)"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_eef_only_local \
    --exp-name=total_2task_2k \
    --overwrite \
    $COMMON \
    2>&1 | tee logs/total_2task_eef_only_2k.log
echo "=== [1/3] eef_only 完成 $(date) ==="

echo "############################################################"
echo "# [2/3] joint_only_local (total_2task_flexiv_ft60, 纯 joint)"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_joint_only_local \
    --exp-name=total_2task_2k \
    --overwrite \
    $COMMON \
    2>&1 | tee logs/total_2task_joint_only_2k.log
echo "=== [2/3] joint_only 完成 $(date) ==="

echo "############################################################"
echo "# [3/3] eef_joint_local (joint + FK EEF, warmup 2w)"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_eef_joint_local \
    --exp-name=total_2task_2k \
    --overwrite \
    $COMMON \
    2>&1 | tee logs/total_2task_eef_joint_2k.log
echo "=== [3/3] eef_joint 完成 $(date) ==="

echo "############################################################"
echo "# 全部完成! checkpoints:"
echo "#   checkpoints/pi05_force_total_task_eef_only_local/total_2task_2k/1999"
echo "#   checkpoints/pi05_force_total_task_joint_only_local/total_2task_2k/1999"
echo "#   checkpoints/pi05_force_total_task_eef_joint_local/total_2task_2k/1999"
echo "############################################################"
