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
# total_task_peel 训练 脚本2: 第三轮 dual（串行独占）
#   脚本1 已完成 (双joint + 双EEF 均 40k 出 ckpt), 故不需要 resume。
#   本脚本直接跑 dual bs=16 独占单卡。
#     pi05_force_total_task_peel_eef_joint_remote (force dual, joint 7 维输出 +
#     EEF 辅 loss warmup 3w; 数据集 ft60)
# 前置: 数据集已在远端; FA-openpi 代码已同步。
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

echo "############################################################"
echo "# [dual] pi05_force_total_task_peel_eef_joint_remote (dual, bs=16, 独占)"
echo "############################################################"
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
echo "=== [dual] 完成 $(date) ==="

echo "############################################################"
echo "# 脚本2 完成! checkpoints:"
echo "#   dual:      checkpoints/pi05_force_total_task_peel_eef_joint_remote/dual_peel/*"
echo "############################################################"
