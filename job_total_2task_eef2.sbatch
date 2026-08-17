#!/bin/bash
#SBATCH --job-name=total_2task_eef2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/total_2task_eef2_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/total_2task_eef2_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ⚠️ 关键: 关闭 XLA 显存预分配 (必须在所有阶段前)。
#   job 405 失败根因: spawn 数据 worker 重新初始化 JAX 时 CUDA OOM。
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ============================================================
# total_2task EEF 新数据集训练编排（单卡）
#   ⚠️ 前置: job 406 (dual resume) 完成后手动停止，再提交本 job。
#
#   阶段 1（并行, 单卡 45%×2）:
#     pi05_plain_total_task_joint_remote (纯 pi05 joint, 7 维)
#     pi05_plain_total_task_eef_remote   (纯 pi05 EEF rot6d, 10 维 delta)
#     目的: 两个纯 pi05 基础模型并行训练, 用 robotin 数据
#   阶段 2（串行, 独占单卡）:
#     pi05_force_total_task_eef_only_remote (EEF rot6d + force, 16 维)
#     目的: 在前两者完成后, 力传感器数据单独训练
#
# 数据集（已本地生成, 需先 sftp 到远端）:
#   /data/group1/junjie008/datasets/total_2task_flexiv_ft60_noforce   (joint, 已有)
#   /data/group1/junjie008/datasets/total_2task_flexiv_eef_rot6d_noforce (EEF plain, 新)
#   /data/group1/junjie008/datasets/total_2task_flexiv_eef_rot6d        (EEF force, 新)
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

############################################################
# 阶段 1: 并行 plain joint + plain EEF (单卡 45%×2)
############################################################
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COMMON="--num_train_steps=40000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "=== [1a] pi05_plain_total_task_joint_remote (纯 pi05 joint) ==="
python -u scripts/train.py pi05_plain_total_task_joint_remote \
    --exp-name=plain_joint \
    --overwrite \
    --batch_size=16 \
    $COMMON \
    2>&1 | tee logs/total_2task_eef2_plain_joint.log &
P_JOINT=$!

echo "=== [1b] pi05_plain_total_task_eef_remote (纯 pi05 EEF rot6d) ==="
python -u scripts/train.py pi05_plain_total_task_eef_remote \
    --exp-name=plain_eef_rot6d \
    --overwrite \
    --batch_size=16 \
    $COMMON \
    2>&1 | tee logs/total_2task_eef2_plain_eef.log &
P_EEF=$!

echo "=== 并行阶段进行中: joint PID=$P_JOINT, eef PID=$P_EEF ==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

wait $P_JOINT $P_EEF
echo "=== [阶段 1] plain joint + plain EEF 并行完成 $(date) ==="

############################################################
# 阶段 2: EEF force (独占单卡)
############################################################
unset XLA_PYTHON_CLIENT_MEM_FRACTION   # 独占, 不限制显存
echo "=== [2] pi05_force_total_task_eef_only_remote (EEF rot6d + force) ==="
python -u scripts/train.py pi05_force_total_task_eef_only_remote \
    --exp-name=eef_force_rot6d \
    --overwrite \
    --batch_size=16 \
    $COMMON \
    2>&1 | tee logs/total_2task_eef2_force_eef.log
echo "=== [阶段 2] EEF force 完成 $(date) ==="

echo "=== 全部完成! ==="
echo "checkpoints:"
echo "  checkpoints/pi05_plain_total_task_joint_remote/plain_joint/*"
echo "  checkpoints/pi05_plain_total_task_eef_remote/plain_eef_rot6d/*"
echo "  checkpoints/pi05_force_total_task_eef_only_remote/eef_force_rot6d/*"
echo "logs:"
echo "  logs/total_2task_eef2_plain_joint.log"
echo "  logs/total_2task_eef2_plain_eef.log"
echo "  logs/total_2task_eef2_force_eef.log"