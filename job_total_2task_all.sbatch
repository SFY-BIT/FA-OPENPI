#!/bin/bash
#SBATCH --job-name=total_2task_all
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/total_2task_all_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/total_2task_all_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# total_2task 完整训练链（单卡串行+并行编排）
#   阶段 1: eef_joint 恢复 (32000 → 40000, bs=16, 独占)
#   阶段 2: 纯 pi05 joint + eef 并行 (各 bs=16, 45% 显存)
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

############################################################
# 阶段 1: eef_joint 恢复训练 (从 32000 resume → 40000)
############################################################
echo "=== [1/2] eef_joint resume (32000 → 40000) ==="
python scripts/train.py pi05_force_total_task_eef_joint_remote \
    --exp-name=eef_joint_remote_exp \
    --resume \
    --batch_size=16 \
    --num_train_steps=40000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=10 \
    --num_workers=4 \
    2>&1 | tee logs/total_2task_all_dual_resume.log
echo "=== [1/2] eef_joint 恢复完成 $(date) ==="

############################################################
# 阶段 2: 纯 pi05 并行 (joint + eef)
############################################################
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COMMON="--num_train_steps=40000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "=== [2/2a] pi05_plain_total_task_joint_remote (纯 pi05, joint) ==="
python -u scripts/train.py pi05_plain_total_task_joint_remote \
    --exp-name=plain_joint \
    --overwrite \
    --batch_size=16 \
    $COMMON \
    2>&1 | tee logs/total_2task_all_plain_joint.log &
P_J=$!

echo "=== [2/2b] pi05_plain_total_task_eef_remote (纯 pi05, eef) ==="
python -u scripts/train.py pi05_plain_total_task_eef_remote \
    --exp-name=plain_eef \
    --overwrite \
    --batch_size=16 \
    $COMMON \
    2>&1 | tee logs/total_2task_all_plain_eef.log &
P_E=$!

echo "=== 纯 pi05 并行进行中: joint PID=$P_J, eef PID=$P_E ==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

wait $P_J $P_E
echo "=== [2/2] 纯 pi05 并行完成 $(date) ==="
echo "=== 全部完成 ==="
