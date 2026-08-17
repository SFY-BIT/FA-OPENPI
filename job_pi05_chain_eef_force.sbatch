#!/bin/bash
#SBATCH --job-name=total_2task_pi05_chain
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/total_2task_pi05_chain_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/total_2task_pi05_chain_%j.err

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
#   否则 JAX 启动即预分配几乎全部显存, multiprocessing spawn 的数据
#   加载 worker 重新初始化 JAX 时 CUDA context 申请 OOM (job 405 教训)。
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ============================================================
# total_2task pi05 训练链（单卡编排，与 3cfg 同构）
#   阶段 1（并行）: pi05_plain_total_task_joint_remote  +  pi05_plain_total_task_eef_remote  各 bs=8
#                   → 每个进程限 45% 显存（RTX PRO 6000 96G ×0.45≈43G each, 共 ~86G）
#   阶段 2（串行）: pi05_force_total_task_eef_only_remote  bs=8（EEF rot6d force 版）
# 目的: 并行阶段 2 个纯 pi05 进程共享单卡（joint + EEF noforce）,
#       完成后 force EEF 独占单卡（rot6d 双 HEAD+force 显存更大）。
# 前置: 数据集已上传远端:
#       - total_2task_flexiv_ft60_noforce (joint, 已有)
#       - total_2task_flexiv_eef_rot6d_noforce (plain EEF, 新)
#       - total_2task_flexiv_eef_rot6d (force EEF, 新)
#       FA-openpi 代码已同步（含 *_remote config + rot6d 转换）
# exp-name 每阶段独立, 便于区分 wandb run 和 checkpoint 目录
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

# 并行阶段参数: 每进程 45% 显存
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PARALLEL_COMMON="--batch_size=8 --num_train_steps=40000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "############################################################"
echo "# [阶段 1a] pi05_plain_total_task_joint_remote (纯 pi05 joint, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_joint_remote \
    --exp-name=plain_joint_chain \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_2task_pi05_chain_plain_joint.log &
P_J=$!

echo "############################################################"
echo "# [阶段 1b] pi05_plain_total_task_eef_remote (纯 pi05 EEF rot6d, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_eef_remote \
    --exp-name=plain_eef_chain \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_2task_pi05_chain_plain_eef.log &
P_E=$!

echo "=== 并行阶段进行中: joint PID=$P_J, eef PID=$P_E ==="
echo "=== 显存监控（前 60s）==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

# 等两个并行训练都完成
wait $P_J $P_E
echo "=== [阶段 1] plain joint + plain EEF 并行完成 $(date) ==="

echo "############################################################"
echo "# [阶段 2] pi05_force_total_task_eef_only_remote (force EEF rot6d, bs=8)"
echo "############################################################"
unset XLA_PYTHON_CLIENT_MEM_FRACTION   # 独占单卡, 不限制显存
python -u scripts/train.py pi05_force_total_task_eef_only_remote \
    --exp-name=force_eef_chain \
    --overwrite \
    --batch_size=8 \
    --num_train_steps=40000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=10 \
    --num_workers=4 \
    2>&1 | tee logs/total_2task_pi05_chain_force_eef.log
echo "=== [阶段 2] force EEF 完成 $(date) ==="

echo "############################################################"
echo "# 全部完成! checkpoints:"
echo "#   plain_joint:  checkpoints/pi05_plain_total_task_joint_remote/plain_joint_chain/*"
echo "#   plain_eef:    checkpoints/pi05_plain_total_task_eef_remote/plain_eef_chain/*"
echo "#   force_eef:    checkpoints/pi05_force_total_task_eef_only_remote/force_eef_chain/*"
echo "############################################################"