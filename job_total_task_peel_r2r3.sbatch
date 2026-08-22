#!/bin/bash
#SBATCH --job-name=total_task_peel_r2r3
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/total_task_peel_r2r3_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/total_task_peel_r2r3_%j.err

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
# total_task_peel 训练 脚本2: 第二轮 resume + 第三轮 dual（串行）
#   阶段 1（并行 resume）: 双 EEF v2 各 bs=8, --resume 继续 脚本1 未训完的步数
#     1a) pi05_plain_total_task_peel_eef_v2_remote  (plain EEF v2)
#     1b) pi05_force_total_task_peel_eef_v2_remote  (force EEF v2)
#   ★ 若 脚本1 的第二轮已跑完 40k, 这两个 resume 会立即结束 (无剩余步数),
#     随后自动进入第三轮。
#   阶段 2（串行独占）: dual bs=16
#     pi05_force_total_task_peel_eef_joint_remote (force dual, joint 7 维输出 +
#     EEF 辅 loss warmup 3w; 数据集 ft60)
# 前置: 数据集已在远端; FA-openpi 代码已同步。
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

# 阶段 1: 并行 resume, 每进程 45% 显存, bs=8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
PARALLEL_COMMON="--batch_size=8 --num_train_steps=40000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "############################################################"
echo "# [阶段 1a] pi05_plain_total_task_peel_eef_v2_remote (resume, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_peel_eef_v2_remote \
    --exp-name=plain_eef_peel \
    --resume \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_task_peel_plain_eef_resume.log &
P_E=$!

echo "############################################################"
echo "# [阶段 1b] pi05_force_total_task_peel_eef_v2_remote (resume, bs=8) 并行"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_peel_eef_v2_remote \
    --exp-name=force_eef_peel \
    --resume \
    $PARALLEL_COMMON \
    2>&1 | tee logs/total_task_peel_force_eef_resume.log &
P_FE=$!

echo "=== 阶段1 resume 并行: plain EEF PID=$P_E, force EEF PID=$P_FE ==="
echo "=== 显存监控（前 60s）==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

wait $P_E $P_FE
echo "=== [阶段 1] 双 EEF v2 resume 完成 $(date) ==="

echo "############################################################"
echo "# [阶段 2] pi05_force_total_task_peel_eef_joint_remote (dual, bs=16, 独占)"
echo "############################################################"
unset XLA_PYTHON_CLIENT_MEM_FRACTION   # 独占单卡, 不限制显存
python -u scripts/train.py pi05_force_total_task_peel_eef_joint_remote \
    --exp-name=dual_peel \
    --overwrite \
    --batch_size=16 \
    --num_train_steps=40000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=10 \
    --num_workers=4 \
    2>&1 | tee logs/total_task_peel_dual.log
echo "=== [阶段 2] dual 完成 $(date) ==="

echo "############################################################"
echo "# 脚本2 完成! checkpoints:"
echo "#   plain_eef: checkpoints/pi05_plain_total_task_peel_eef_v2_remote/plain_eef_peel/*"
echo "#   force_eef: checkpoints/pi05_force_total_task_peel_eef_v2_remote/force_eef_peel/*"
echo "#   dual:      checkpoints/pi05_force_total_task_peel_eef_joint_remote/dual_peel/*"
echo "############################################################"
