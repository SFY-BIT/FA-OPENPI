#!/bin/bash
#SBATCH --job-name=plain_pi05
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/plain_pi05_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/plain_pi05_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# 纯 pi05 (无 MOE / 无 force) — total_2task 数据集消融
#   并行: joint + eef 各 bs=16 (45% 显存, RTX PRO 6000 96G)
#   与 pi05_force_total_task 唯一区别: 无 LIMoE / 无 force 头
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COMMON="--num_train_steps=40000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "############################################################"
echo "# [1/2] pi05_plain_total_task_joint (纯 pi05, joint 空间)"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_joint \
    --exp-name=plain_joint \
    --overwrite \
    --batch_size=16 \
    $COMMON \
    2>&1 | tee logs/plain_pi05_joint.log &
P_J=$!

echo "############################################################"
echo "# [2/2] pi05_plain_total_task_eef (纯 pi05, EEF 空间)"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_eef \
    --exp-name=plain_eef \
    --overwrite \
    --batch_size=16 \
    $COMMON \
    2>&1 | tee logs/plain_pi05_eef.log &
P_E=$!

echo "=== 并行进行中: joint PID=$P_J, eef PID=$P_E ==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

wait $P_J $P_E
echo "=== 纯 pi05 并行训练完成 $(date) ==="
