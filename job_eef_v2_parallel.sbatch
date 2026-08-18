#!/bin/bash
#SBATCH --job-name=eef_v2_par
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/FA-openpi/logs/eef_v2_par_%j.out
#SBATCH --error=/data/group1/junjie008/FA-openpi/logs/eef_v2_par_%j.err

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
# EEF v2 修复轮训练（单卡并行，与 job_pi05_chain_eef_force 同构）
#   两个 EEF v2 训练并行共享单卡（RTX PRO 6000 96G）:
#     1) pi05_plain_total_task_eef_v2_remote  (plain EEF v2, 纯 pi05, bs=8)
#     2) pi05_force_total_task_eef_v2_remote  (force EEF v2, Pi0Force 全参, bs=8)
#   每进程限 45% 显存 (96G ×0.45≈43G each, 共 ~86G)。
#   ⚠️ force 版是 Pi0Force 全参 + ft_history, 显存比 plain 大;
#     若 force 侧 OOM, 调 plain 0.35 / force 0.55, 或降 force --batch_size=4。
# 前置: 数据集已上传远端:
#       - total_2task_flexiv_eef_abs_noforce (plain EEF v2, 含 norm_stats.json)
#       - total_2task_flexiv_eef_abs (force EEF v2, 含 norm_stats.json)
#       FA-openpi 代码已同步 (8d2b479, 含 v2 config + transforms + convert 脚本)
# exp-name 每阶段独立, 便于区分 wandb run 和 checkpoint 目录
# ============================================================

echo "=== 远端 GPU 信息 ==="
nvidia-smi

# 并行阶段参数: 每进程 45% 显存
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
PARALLEL_COMMON="--batch_size=8 --num_train_steps=40000 --save_interval=2000 --keep_period=10000 --log_interval=10 --num_workers=4"

echo "############################################################"
echo "# [并行 1] pi05_plain_total_task_eef_v2_remote (plain EEF v2, bs=8)"
echo "############################################################"
python -u scripts/train.py pi05_plain_total_task_eef_v2_remote \
    --exp-name=eef_v2_plain \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/eef_v2_plain.log &
P_PL=$!

echo "############################################################"
echo "# [并行 2] pi05_force_total_task_eef_v2_remote (force EEF v2, bs=8)"
echo "############################################################"
python -u scripts/train.py pi05_force_total_task_eef_v2_remote \
    --exp-name=eef_v2_force \
    --overwrite \
    $PARALLEL_COMMON \
    2>&1 | tee logs/eef_v2_force.log &
P_FT=$!

echo "=== 并行训练进行中: plain PID=$P_PL, force PID=$P_FT ==="
echo "=== 显存监控（前 60s）==="
for i in $(seq 1 6); do nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 10; done

# 等两个并行训练都完成
wait $P_PL $P_FT
echo "=== 两个 EEF v2 训练并行完成 $(date) ==="

echo "############################################################"
echo "# 全部完成! checkpoints:"
echo "#   plain: checkpoints/pi05_plain_total_task_eef_v2_remote/eef_v2_plain/*"
echo "#   force: checkpoints/pi05_force_total_task_eef_v2_remote/eef_v2_force/*"
echo "############################################################"
