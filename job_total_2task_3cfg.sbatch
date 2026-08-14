#!/bin/bash
#SBATCH --job-name=total_2task_3cfg
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/total_2task_3cfg_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/total_2task_3cfg_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

# WANDB key（与主干 sbatch 相同）
export WANDB_API_KEY='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'
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# total_2task 3 config 单卡训练（A100 80G）
#   阶段 1（并行）: eef_only_remote  +  joint_only_remote  各 bs=16
#                   → 每个进程限 45% 显存（80G×0.45≈36G each, 共 ~72G）
#   阶段 2（串行）: eef_joint_remote  bs=32（warmup 2w2）
# 目的: 并行阶段 2 个进程共享单卡, 联合训练（大 bs）独占单卡
# 前置: total_2task_flexiv_{ft60,eef} 数据集（含 norm_stats.json）已上传远端
#       FA-openpi 代码已同步（含 *_remote config）
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

# 并行阶段参数: 每进程 45% 显存
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PARALLEL_COMMON="--num_train_steps=30000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "############################################################"
echo "# [阶段 1a] eef_only_remote  (EEF 空间 loss, bs=16) 并行启动"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_eef_only_remote \
    --exp-name=total_2task_remote \
    --overwrite \
    --batch_size=16 \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_2task_remote_eef_only.log &
P_EEF=$!

echo "############################################################"
echo "# [阶段 1b] joint_only_remote (纯 joint, bs=16) 并行启动"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_joint_only_remote \
    --exp-name=total_2task_remote \
    --overwrite \
    --batch_size=16 \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_2task_remote_joint_only.log &
P_JOINT=$!

echo "=== 并行阶段进行中: eef_only PID=$P_EEF, joint_only PID=$P_JOINT ==="
echo "=== 显存监控（前 60s）==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

# 等两个并行训练都完成
wait $P_EEF $P_JOINT
echo "=== [阶段 1] eef_only + joint_only 并行完成 $(date) ==="

echo "############################################################"
echo "# [阶段 2] eef_joint_remote (joint + FK EEF, warmup 2w2, bs=32)"
echo "############################################################"
unset XLA_PYTHON_CLIENT_MEM_FRACTION   # 独占单卡, 不限制显存
python -u scripts/train.py pi05_force_total_task_eef_joint_remote \
    --exp-name=total_2task_remote \
    --overwrite \
    --batch_size=32 \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_2task_remote_eef_joint.log
echo "=== [阶段 2] eef_joint 完成 $(date) ==="

echo "############################################################"
echo "# 全部完成! checkpoints (exp=total_2task_remote):"
echo "#   checkpoints/pi05_force_total_task_eef_only_remote/total_2task_remote/*"
echo "#   checkpoints/pi05_force_total_task_joint_only_remote/total_2task_remote/*"
echo "#   checkpoints/pi05_force_total_task_eef_joint_remote/total_2task_remote/*"
echo "############################################################"
