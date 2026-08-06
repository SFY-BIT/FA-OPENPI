# Panda-Force 训练与Rollout完整流程指南

## 目录

1. [环境与路径概览](#1-环境与路径概览)
2. [数据集与归一化](#2-数据集与归一化)
3. [Config说明](#3-config说明)
4. [本地LoRA训练（近端）](#4-本地lora训练近端)
5. [远端全量训练（SLURM）](#5-远端全量训练slurm)
6. [SFTP文件上传](#6-sftp文件上传)
7. [Policy服务部署](#7-policy服务部署)
8. [Rollout评估](#8-rollout评估)
9. [完整工作流速查](#9-完整工作流速查)

---

## 1. 环境与路径概览

### 近端（本地工作站）

| 项目 | 值 |
|---|---|
| GPU | 2× NVIDIA RTX 6000 Ada (46GB each) |
| Conda环境 | `rlinf`（训练/rollout）、`robov2`（数据收集） |
| 代码仓库 | `/mnt/hdd/sfy/openpi-force` |
| 数据集根目录 | `/mnt/hdd/sfy/robosuite/datasets/` |
| Checkpoint目录 | `/mnt/hdd/sfy/openpi-force/checkpoints/` |
| HF缓存 | `/mnt/hdd/sfy/openpi/.cache/huggingface` |
| Python | `/home/sfy/miniconda3/envs/rlinf/bin/python` |

### 远端（SLURM集群）

| 项目 | 值 |
|---|---|
| 用户 | `junjie008` |
| 代码仓库 | `/data/group1/junjie008/openpi-force` |
| 数据集目录 | `/data/group1/junjie008/datasets/` |
| Conda环境 | `openpi` |
| 日志目录 | `/data/group1/junjie008/openpi-force/logs/` |
| SLURM分区 | `debug` |

---

## 2. 数据集与归一化

### 数据集

| 数据集 | 本地路径 | 远端路径 | 规格 |
|---|---|---|---|
| panda-force（含force） | `/mnt/hdd/sfy/robosuite/datasets/panda-force` | `/data/group1/junjie008/datasets/panda-force-full` | 224×224, 136ep, 23551帧 |
| panda-noforce（无force） | `/mnt/hdd/sfy/robosuite/datasets/panda-noforce` | `/data/group1/junjie008/datasets/panda-noforce` | 224×224, 136ep, 23551帧 |
| robodojo_x5_fine_assembly（双臂无force） | `/mnt/hdd/sfy/lerobot_datasets/robodojo_arx_x5_fine_assembly_v21_v21/unified_robot/robodojo_arx_x5_fine_assembly_v21` | `/data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21` | 480×640, 300ep, 180189帧, 14维state/action(7+7双臂无夹爪), 3任务 |

每个数据集目录下包含：
- `meta/info.json` — 数据集元信息
- `data/chunk-000/episode_*.parquet` — 帧数据
- `videos/chunk-000/` — 视频帧（AV1编码）
- `norm_stats.json` — 归一化统计量

### 计算归一化stats

```bash
# 本地LoRA版（force版，state=26维, actions=14维）
cd /mnt/hdd/sfy/openpi-force && PYTHONPATH=src /home/sfy/miniconda3/envs/rlinf/bin/python scripts/compute_norm_stats.py --config-name pi05_force_lora_local

# 本地无force版（state=8维, actions=8维）
cd /mnt/hdd/sfy/openpi-force && PYTHONPATH=src /home/sfy/miniconda3/envs/rlinf/bin/python scripts/compute_norm_stats.py --config-name pi05_panda_noforce_local

# force权重0.1版
cd /mnt/hdd/sfy/openpi-force && PYTHONPATH=src /home/sfy/miniconda3/envs/rlinf/bin/python scripts/compute_norm_stats.py --config-name pi05_force_lora_local_w01

# RoboDojo X5双臂无force版（state=14维, actions=14维）
cd /mnt/hdd/sfy/openpi-force && PYTHONPATH=src /home/sfy/miniconda3/envs/rlinf/bin/python scripts/compute_norm_stats.py --config-name pi05_robodojo_x5_noforce_local
```

> norm_stats.json 会保存到 `assets/<config_name>/<repo_id>/norm_stats.json`。
> RoboDojo X5 的 norm_stats 需传到远端数据集目录下供远端 config 加载。

---

## 3. Config说明

所有config定义在 `src/openpi/training/config.py`。

| Config名 | 用途 | force_loss_weight | LoRA | 数据集repo_id | 训练位置 |
|---|---|---|---|---|---|
| `pi05_force_lora_local` | 本地LoRA+force | 0.3 | ✅ | 本地panda-force | 近端 |
| `pi05_force_lora_local_w01` | 本地LoRA+force低权重 | 0.1 | ✅ | 本地panda-force | 近端 |
| `pi05_panda_noforce_local` | 本地无force推理 | - | ❌ | 本地panda-noforce | 近端(推理用) |
| `pi05_panda_noforce` | 远端全量无force | - | ❌ | 远端panda-noforce | 远端 |
| `pi05_panda_full` | 远端全量+force输入 | - | ❌ | 远端panda-force-full | 远端 |
| `pi05_robodojo_x5_noforce_local` | 本地X5双臂无force推理 | - | ❌ | 本地robodojo_x5 | 近端(推理用) |
| `pi05_robodojo_x5_noforce` | 远端X5双臂全量无force | - | ❌ | 远端robodojo_x5 | 远端 |

### 关键模型参数

```
action_horizon=30, force_start_idx=8 (7关节+1夹爪), force_history_frames=3
num_experts=4, num_top_k=1 (LIMoE)
LoRA: paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
```

---

## 4. 本地LoRA训练（近端）

### 4.1 启动训练（tmux内运行）

先创建/进入tmux：
```bash
tmux new -s lora    # 或 tmux attach -t lora
```

**v7（force权重0.3，GPU 0）：**
```bash
cd /mnt/hdd/sfy/openpi-force && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HOME=/mnt/hdd/sfy/openpi/.cache/huggingface HF_DATASETS_CACHE=/mnt/hdd/sfy/openpi/.cache/datasets PYTHONPATH=src:$PYTHONPATH conda run -n rlinf python scripts/train.py pi05_force_lora_local \
  --exp-name=local_lora_v7 \
  --overwrite \
  --batch_size=8 \
  --num_train_steps=20000 \
  --log_interval=10 \
  --save_interval=1000
```

**v7_w01（force权重0.1，GPU 1，并行训练）：**
```bash
cd /mnt/hdd/sfy/openpi-force && CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HOME=/mnt/hdd/sfy/openpi/.cache/huggingface HF_DATASETS_CACHE=/mnt/hdd/sfy/openpi/.cache/datasets PYTHONPATH=src:$PYTHONPATH conda run -n rlinf python scripts/train.py pi05_force_lora_local_w01 \
  --exp-name=local_lora_v7_w01 \
  --overwrite \
  --batch_size=8 \
  --num_train_steps=20000 \
  --log_interval=10 \
  --save_interval=1000
```

### 4.2 训练参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--exp-name` | `local_lora_v7` | 实验名，checkpoint存到 `checkpoints/pi05_force_lora_local/local_lora_v7/` |
| `--overwrite` | - | 覆盖旧checkpoint目录 |
| `--batch_size=8` | 8 | 单GPU batch size |
| `--num_train_steps=20000` | 20000 | 总训练步数 |
| `--log_interval=10` | 10 | 每10步打印loss |
| `--save_interval=1000` | 1000 | 每1000步保存checkpoint |

### 4.3 恢复训练

```bash
# 去掉 --overwrite，加 --resume
cd /mnt/hdd/sfy/openpi-force && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HOME=/mnt/hdd/sfy/openpi/.cache/huggingface HF_DATASETS_CACHE=/mnt/hdd/sfy/openpi/.cache/datasets PYTHONPATH=src:$PYTHONPATH conda run -n rlinf python scripts/train.py pi05_force_lora_local \
  --exp-name=local_lora_v7 \
  --resume \
  --batch_size=8 \
  --num_train_steps=20000 \
  --log_interval=10 \
  --save_interval=1000
```

### 4.4 监控训练

```bash
# 查看训练日志
tail -f /mnt/hdd/sfy/openpi-force/logs/train_local_lora_v7.log

# 查看checkpoint
ls /mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_lora_local/local_lora_v7/

# GPU使用
nvidia-smi
```

---

## 5. 远端全量训练（SLURM）

### 5.1 SBATCH脚本

脚本位于 `/mnt/hdd/sfy/openpi-force/job_panda_noforce.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=panda_nf
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/panda_noforce_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/panda_noforce_%j.err

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
# Panda NoForce: pure Pi0.5 baseline, no force/LIMoE
# All params trainable, no LoRA, no freeze
# Dataset: /data/group1/junjie008/datasets/panda-noforce
# ============================================================
echo "=== Panda NoForce Baseline ==="
python scripts/train.py pi05_panda_noforce \
    --exp-name=panda_noforce_v1 \
    --overwrite \
    --batch_size=8 \
    --num_workers=0 \
    --num_train_steps=30000 \
    --log_interval=50 \
    --save_interval=2000

echo "=== Training finished ==="
```

### 5.2 SSH到远端提交

```bash
# SSH登录远端
ssh junjie008@<远端地址>

# 提交全量无force训练
cd /data/group1/junjie008/openpi-force
sbatch job_panda_noforce.sbatch

# 查看队列
squeue -u junjie008

# 查看日志
tail -f /data/group1/junjie008/openpi-force/logs/panda_noforce_*.out

# 取消任务
scancel <job_id>
```

### 5.3 远端全量+force训练（如需要）

修改sbatch中的config和exp-name：
```bash
python scripts/train.py pi05_panda_full \
    --exp-name=panda_full_v1 \
    --overwrite \
    --batch_size=8 \
    --num_workers=0 \
    --num_train_steps=30000 \
    --log_interval=50 \
    --save_interval=2000
```

### 5.4 远端RoboDojo X5双臂全量训练

脚本位于 `/mnt/hdd/sfy/openpi-force/job_robodojo_x5_noforce.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=x5_nf
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_%j.out
#SBATCH --error=/data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_%j.err

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
# RoboDojo ARX X5 Dual-Arm NoForce: pure Pi0.5 baseline
# All params trainable, no LoRA, no freeze, no force
# 14-dim state/action (7+7 joints, no gripper), 3 assembly tasks
# Dataset: /data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21
# ============================================================
echo "=== RoboDojo X5 NoForce Full Fine-tune ==="
python scripts/train.py pi05_robodojo_x5_noforce \
    --exp-name=robodojo_x5_noforce_v1 \
    --overwrite \
    --batch_size=8 \
    --num_workers=0 \
    --num_train_steps=30000 \
    --log_interval=50 \
    --save_interval=2000

echo "=== Training finished ==="
```

提交任务：
```bash
ssh junjie008@<远端地址>
cd /data/group1/junjie008/openpi-force
sbatch job_robodojo_x5_noforce.sbatch
squeue -u junjie008
tail -f /data/group1/junjie008/openpi-force/logs/robodojo_x5_noforce_*.out
```
```

---

## 6. SFTP文件上传

在sftp终端中执行，将本地更新同步到远端：

```sftp
# 1. 上传更新后的config.py（包含新config定义）
put /mnt/hdd/sfy/openpi-force/src/openpi/training/config.py /data/group1/junjie008/openpi-force/src/openpi/training/config.py

# 2. 上传sbatch脚本
put /mnt/hdd/sfy/openpi-force/job_panda_noforce.sbatch /data/group1/junjie008/openpi-force/job_panda_noforce.sbatch

# 3. 上传panda-noforce数据集（全量无force训练用，含norm_stats.json）
put -r /mnt/hdd/sfy/robosuite/datasets/panda-noforce /data/group1/junjie008/datasets/panda-noforce

# 4. 上传panda-force数据集（远端pi05_panda_full用，路径名为panda-force-full）
put -r /mnt/hdd/sfy/robosuite/datasets/panda-force /data/group1/junjie008/datasets/panda-force-full

# 5. 上传RoboDojo X5双臂数据集（远端pi05_robodojo_x5_noforce用）
put -r /mnt/hdd/sfy/lerobot_datasets/robodojo_arx_x5_fine_assembly_v21_v21/unified_robot/robodojo_arx_x5_fine_assembly_v21 /data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21

# 6. 上传RoboDojo X5 sbatch脚本
put /mnt/hdd/sfy/openpi-force/job_robodojo_x5_noforce.sbatch /data/group1/junjie008/openpi-force/job_robodojo_x5_noforce.sbatch

# 7. 上传norm_stats到远端数据集目录（本地算好后传）
# norm_stats路径: assets/pi05_robodojo_x5_noforce_local/<repo_id>/norm_stats.json
# 远端config的asset_id=repo_id=/data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21
# 需放到远端: assets/pi05_robodojo_x5_noforce/<远端repo_id>/norm_stats.json
# 或直接放到数据集目录下
put /mnt/hdd/sfy/openpi-force/assets/pi05_robodojo_x5_noforce_local/*/norm_stats.json /data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21/norm_stats.json
```

> `put -r` 会直接覆盖远端同名文件，无需预先删除。
>
> **norm_stats 注意**：远端 config `pi05_robodojo_x5_noforce` 的 `asset_id` 默认等于 `repo_id`（远端路径），
> 所以 norm_stats 需放到 `assets/pi05_robodojo_x5_noforce/<远端repo_id>/norm_stats.json`。
> 如果远端重新跑 `compute_norm_stats.py --config-name pi05_robodojo_x5_noforce` 则会自动生成。

### 从远端下载checkpoint

```sftp
# 下载panda-noforce训练完成的checkpoint到本地
get -r /data/group1/junjie008/openpi-force/checkpoints/pi05_panda_noforce/panda_noforce_v1/29999 /mnt/hdd/sfy/outputs/panda-no-force/

# 下载RoboDojo X5训练完成的checkpoint到本地
get -r /data/group1/junjie008/openpi-force/checkpoints/pi05_robodojo_x5_noforce/robodojo_x5_noforce_v1/29999 /mnt/hdd/sfy/outputs/robodojo-x5-noforce/
```

---

## 7. Policy服务部署

### 7.1 启动force版policy服务（端口8000）

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force
PYTHONPATH=src:packages/openpi-client/src \
python scripts/serve_policy.py --port 8000 \
  --jax-mem-fraction 0.3 --jax-preallocate false \
  policy:checkpoint \
  --policy.config=pi05_force_lora_local \
  --policy.dir=/mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_lora_local/local_lora_v7/19999
```

### 7.2 启动无force版policy服务（端口8001）

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force
PYTHONPATH=src:packages/openpi-client/src \
python scripts/serve_policy.py --port 8001 \
  --jax-mem-fraction 0.3 --jax-preallocate false \
  policy:checkpoint \
  --policy.config=pi05_panda_noforce_local \
  --policy.dir=/mnt/hdd/sfy/outputs/panda-no-force/29999
```

### 7.3 参数说明

| 参数 | 说明 |
|---|---|
| `--port` | WebSocket服务端口 |
| `--jax-mem-fraction 0.3` | JAX显存占比（rollout时用小值，训练时用0.9） |
| `--jax-preallocate false` | 不预分配显存 |
| `policy:checkpoint` | 从checkpoint加载policy |
| `--policy.config` | 训练时用的config名 |
| `--policy.dir` | checkpoint目录路径 |

> 两个policy服务可分别在不同tmux窗口中运行，分别占用不同端口。

---

## 8. Rollout评估

### 8.1 Force版Rollout（端口8000）

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_v6.py \
  --task usb \
  --port 8000 \
  --episodes 5 \
  --max-steps 1200 \
  --num-action-steps 10
```

### 8.2 NoForce版Rollout（端口8001）

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_noforce.py \
  --task usb \
  --port 8001 \
  --episodes 5 \
  --max-steps 1200 \
  --num-action-steps 10
```

### 8.3 Rollout参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task` | `usb` | 任务：`usb`(USBInsert) 或 `whiteboard`(WhiteboardWipe) |
| `--port` | 8000(v6) / 8001(noforce) | policy服务端口 |
| `--episodes` | 1 | 评估episode数 |
| `--max-steps` | 0(用默认: usb=400, whiteboard=1200) | 每episode最大步数 |
| `--num-action-steps` | 1 | 每次推理执行多少步动作（10=较平滑） |
| `--seed-offset` | 0 | 随机种子偏移 |
| `--output-dir` | `/mnt/hdd/sfy/outputs/rollouts_v6` | 输出目录（视频+force日志） |
| `--gripper-open-steps` | -1 | 初始夹爪张开步数（-1=任务默认） |

### 8.4 Rollout输出

```
/mnt/hdd/sfy/outputs/rollouts_v6/
├── episode_000/
│   ├── video.mp4          # rollout视频
│   ├── force_log.json     # force/torque时序数据
│   └── metadata.json      # episode元信息
├── episode_001/
│   └── ...
└── summary.json           # 汇总统计
```

### 8.5 完整评估流程

```bash
# 1. 启动两个policy服务（各一个tmux窗口）
#    窗口1: serve_policy --port 8000 (force版)
#    窗口2: serve_policy --port 8001 (noforce版)

# 2. 等待服务就绪（看到 "Server listening on..." ）

# 3. 在第三个窗口并行跑rollout
# Force版
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_v6.py --task usb --port 8000 --episodes 10 --max-steps 1200 --num-action-steps 10

# NoForce版
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_noforce.py --task usb --port 8001 --episodes 10 --max-steps 1200 --num-action-steps 10

# 4. 查看结果
cat /mnt/hdd/sfy/outputs/rollouts_v6/summary.json
cat /mnt/hdd/sfy/outputs/rollouts_noforce/summary.json
```

---

## 9. 完整工作流速查

### 近端LoRA训练流程

```
1. 数据收集 → /mnt/hdd/sfy/robosuite/datasets/panda-force/
2. 计算norm_stats → compute_norm_stats.py --config-name pi05_force_lora_local
3. tmux启动训练 → train.py pi05_force_lora_local --exp-name=local_lora_v7
4. 启动policy服务 → serve_policy.py --port 8000
5. Rollout评估 → rollout_panda_v6.py --port 8000
```

### 远端全量训练流程

```
1. 本地修改config.py / sbatch脚本
2. SFTP上传 → config.py + sbatch + 数据集
3. SSH远端 → sbatch job_panda_noforce.sbatch
4. 等待训练完成 → squeue / tail日志
5. SFTP下载checkpoint → get -r checkpoints/.../
6. 本地启动policy服务 → serve_policy.py --port 8001
7. Rollout评估 → rollout_panda_noforce.py --port 8001
```

### 对比实验矩阵

| 实验 | Config | force_loss_weight | 训练位置 | Rollout脚本 |
|---|---|---|---|---|
| LoRA + force (w=0.3) | `pi05_force_lora_local` | 0.3 | 近端GPU0 | `rollout_panda_v6.py` |
| LoRA + force (w=0.1) | `pi05_force_lora_local_w01` | 0.1 | 近端GPU1 | `rollout_panda_v6.py` |
| 全量无force | `pi05_panda_noforce` | - | 远端SLURM | `rollout_panda_noforce.py` |
| 全量+force输入 | `pi05_panda_full` | - | 远端SLURM | `rollout_panda_v6.py` |
| X5双臂全量无force | `pi05_robodojo_x5_noforce` | - | 远端SLURM | RoboDojo eval |

### RoboDojo X5双臂完整流程

```
1. 数据集准备 → robodojo_arx_x5_fine_assembly_v21 (v2.1格式, 14维双臂)
2. 修复parquet metadata → 移除huggingface metadata中的List类型
3. 本地计算norm_stats → compute_norm_stats.py --config-name pi05_robodojo_x5_noforce_local
4. SFTP上传 → config.py + piper_policy.py + sbatch + 数据集 + norm_stats
5. SSH远端 → sbatch job_robodojo_x5_noforce.sbatch
6. 等待训练完成 → squeue / tail日志
7. SFTP下载checkpoint → get -r checkpoints/.../
8. 本地启动policy服务 → serve_policy.py --policy.config=pi05_robodojo_x5_noforce_local
9. RoboDojo eval评估
```

### RoboDojo X5 Policy服务部署

```bash
# 本地启动X5 policy服务（下载checkpoint后）
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force
PYTHONPATH=src:packages/openpi-client/src \
python scripts/serve_policy.py --port 8002 \
  --jax-mem-fraction 0.3 --jax-preallocate false \
  policy:checkpoint \
  --policy.config=pi05_robodojo_x5_noforce_local \
  --policy.dir=/mnt/hdd/sfy/outputs/robodojo-x5-noforce/29999
```
