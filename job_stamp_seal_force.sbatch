#!/bin/bash
#SBATCH --job-name=stamp_seal
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/stamp_seal_force_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/stamp_seal_force_%j.err

set -euo pipefail

source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/openpi-force
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs checkpoints

export WANDB_API_KEY='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'
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# Pi0Force Dual-Head: stamp_seal_v2_flexiv (Piper/Flexiv)
#
# V2 CHANGES (vs V1):
#   1. optimizer.py: RouterWeights → base LR (1× = 5e-5)
#      Prevents router_z_loss explosion by slowing router logit growth.
#   2. optimizer.py: MlpBlock_0 (per-expert FFNs) → 5× LR (2.5e-4)
#      Prevents "rusty expert" death spiral — keeps all 4 experts competitive.
#   3. keep_period=10000 → retains ~6 checkpoints (~300GB), deletes older ones.
#   4. --overwrite → clean training from pi05_base (no resume).
#
# LR layout: base=5e-5, peak_lr=5e-5 (CosineDecay, warmup=5000)
#   Group "base" (1×): PaliGemma, action_proj, time_mlp, RouterWeights
#   Group "new"  (5×): limoe experts (MlpBlock_0), limoe shared, force, state_proj
#
# PREREQUISITE: norm_stats must be computed and uploaded.
#   Local:  compute_norm_stats.py --config-name=pi05_force_stamp_seal
#   Upload: assets/pi05_force_stamp_seal_remote/<remote_repo_id>/norm_stats.json
#           -> /data/group1/junjie008/openpi-force/assets/pi05_force_stamp_seal_remote/
#              <remote_repo_id>/norm_stats.json
#   OR recompute on remote:
#     python scripts/compute_norm_stats.py --config-name=pi05_force_stamp_seal_remote
# ============================================================

echo "=== Fresh training from pi05_base (keep ALL checkpoints) ==="
python scripts/train.py pi05_force_stamp_seal_remote \
    --exp-name=stamp_seal_force_v2 \
    --overwrite \
    --batch_size=32 \
    --num_workers=0 \
    --num_train_steps=50000 \
    --log_interval=10 \
    --save_interval=2000 \
    --keep_period=10000

echo "=== Training finished ==="
