#!/bin/bash
#SBATCH --job-name=eefv2_pl
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/eefv2_plain_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/eefv2_plain_%j.err

# ──────────────────────────────────────────────────────────────
# EEF v2 数据集修复训练 ①  — plain pi05 (无 force)
#   config:  pi05_plain_total_task_eef_v2_remote
#   dataset: total_2task_flexiv_eef_abs_noforce (绝对 EEF + 训练时 delta)
#   与 job_eef_v2_force.sbatch 各占一张卡, 两个 sbatch 分别提交即并行
# ──────────────────────────────────────────────────────────────

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ⚠️ 关闭 XLA 显存预分配 (必须在训练前): 否则 JAX 启动即抓 75% 显存,
#   spawn 数据 worker 重新初始化 JAX 时 CUDA context OOM。
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== 远端 GPU 信息 ==="
nvidia-smi

echo "=== EEF v2 plain 训练开始 $(date) ==="
python -u scripts/train.py pi05_plain_total_task_eef_v2_remote \
    --exp-name=eef_v2_plain \
    --batch_size=32 \
    --num_train_steps=40000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=10 \
    --num_workers=4 \
    2>&1 | tee logs/eefv2_plain.log
echo "=== EEF v2 plain 训练完成 $(date) ==="
