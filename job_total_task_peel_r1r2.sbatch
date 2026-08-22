#!/bin/bash
#SBATCH --job-name=total_task_peel_r1r2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/total_task_peel_r1r2_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/total_task_peel_r1r2_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

# WANDB key（与主干 sbatch 相同）
export WANDB_API_KEY='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'
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ⚠️ 关键: 关闭 XLA 显存预分配 (必须在所有阶段前)。
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ============================================================
# total_task_peel 训练 脚本1: 第一轮 + 第二轮（单卡编排, 与 chain 同构）
#   第一轮（并行）: 双 joint 各 bs=8
#     1a) pi05_plain_total_task_peel_joint_remote       (plain joint, ft60_noforce)
#     1b) pi05_force_total_task_peel_joint_only_remote  (force joint, ft60)
#   第二轮（并行）: 双 EEF v2 各 bs=8
#     2a) pi05_plain_total_task_peel_eef_v2_remote      (plain EEF v2, eef_abs_noforce)
#     2b) pi05_force_total_task_peel_eef_v2_remote      (force EEF v2, eef_abs)
#   每进程限 45% 显存（96G ×0.45≈43G each, 共 ~86G）。
#   第三轮 (dual bs=16 串行) 在 脚本2 里 (第二轮 resume 后)。
# 前置: 数据集已上传远端:
#       - total_task_peel_ft60 / total_task_peel_ft60_noforce
#       - total_task_peel_eef_abs / total_task_peel_eef_abs_noforce
#       FA-openpi 代码已同步 (含 peel config)
# exp-name 每阶段独立, 便于区分 wandb run 和 checkpoint 目录
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

# 并行阶段参数: 每进程 45% 显存, bs=8 (与 total_2task 对齐)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
PARALLEL_COMMON="--batch_size=8 --num_train_steps=40000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "############################################################"
echo "# [第一轮 1a] pi05_plain_total_task_peel_joint_remote (plain joint, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_peel_joint_remote \
    --exp-name=plain_joint_peel \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_task_peel_plain_joint.log &
P_J=$!

echo "############################################################"
echo "# [第一轮 1b] pi05_force_total_task_peel_joint_only_remote (force joint, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_peel_joint_only_remote \
    --exp-name=force_joint_peel \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_task_peel_force_joint.log &
P_F=$!

echo "=== 第一轮并行: plain joint PID=$P_J, force joint PID=$P_F ==="
echo "=== 显存监控（前 60s）==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

wait $P_J $P_F
echo "=== [第一轮] 双 joint 并行完成 $(date) ==="

echo "############################################################"
echo "# [第二轮 2a] pi05_plain_total_task_peel_eef_v2_remote (plain EEF v2, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_peel_eef_v2_remote \
    --exp-name=plain_eef_peel \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_task_peel_plain_eef.log &
P_E=$!

echo "############################################################"
echo "# [第二轮 2b] pi05_force_total_task_peel_eef_v2_remote (force EEF v2, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_peel_eef_v2_remote \
    --exp-name=force_eef_peel \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_task_peel_force_eef.log &
P_FE=$!

echo "=== 第二轮并行: plain EEF PID=$P_E, force EEF PID=$P_FE ==="
echo "=== 显存监控（前 60s）==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

wait $P_E $P_FE
echo "=== [第二轮] 双 EEF v2 并行完成 $(date) ==="

echo "############################################################"
echo "# 脚本1 完成! checkpoints:"
echo "#   plain_joint: checkpoints/pi05_plain_total_task_peel_joint_remote/plain_joint_peel/*"
echo "#   force_joint: checkpoints/pi05_force_total_task_peel_joint_only_remote/force_joint_peel/*"
echo "#   plain_eef:   checkpoints/pi05_plain_total_task_peel_eef_v2_remote/plain_eef_peel/*"
echo "#   force_eef:   checkpoints/pi05_force_total_task_peel_eef_v2_remote/force_eef_peel/*"
echo "# 若第二轮因 72h 未跑完 → 提交 脚本2 (resume 第二轮 + dual 第三轮)"
echo "############################################################"
