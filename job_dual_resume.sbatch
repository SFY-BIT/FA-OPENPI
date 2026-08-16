#!/bin/bash
#SBATCH --job-name=dual_resume
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/dual_resume_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/dual_resume_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# eef_joint 恢复训练 (33630 崩溃后, 从 32000 resume → 40000)
#   崩溃原因: CUDA_ERROR_LAUNCH_TIMEOUT (GPU kernel 超时)
#   32000 步 checkpoint 已保存, --resume 从它继续
# ============================================================

echo "=== 恢复 eef_joint (resume from 32000) ==="
python scripts/train.py pi05_force_total_task_eef_joint_remote \
    --exp-name=eef_joint_remote_exp \
    --resume \
    --batch_size=16 \
    --num_train_steps=40000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=10 \
    --num_workers=4

echo "=== 恢复完成 ==="
