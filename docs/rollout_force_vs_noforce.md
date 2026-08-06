# Panda Force vs No-Force Rollout 使用说明

## 概述

两个 checkpoint 对比：

| 模型 | Config | Checkpoint | Force 输入 | Force 预测 |
|------|--------|------------|-----------|-----------|
| **V6 (force)** | `pi05_force_lora_local` | `checkpoints/pi05_force_lora_local/local_lora_v6/19999` | ✅ 3帧历史 | ✅ |
| **No-force** | `pi05_panda_noforce_local` | `/mnt/hdd/sfy/outputs/panda-no-force/29999` | ❌ | ❌ |

两者都是 **Pi0.5**，Panda 机器人（7 关节 + 1 gripper = 8 维 state），双 task（USB 插入 + 白板擦）。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/serve_policy.py` | Policy server（通用，支持显存控制参数） |
| `scripts/rollout_panda_v6.py` | Force V6 rollout client |
| `scripts/rollout_panda_noforce.py` | No-force rollout client |
| `scripts/compare_rollouts.py` | 关节轨迹对比脚本 |

---

## Server 显存控制

`serve_policy.py` 内置 JAX 显存控制参数，**不需要设环境变量**，直接用命令行参数：

| 参数 | 对应环境变量 | 说明 |
|------|-------------|------|
| `--jax-mem-fraction 0.3` | `XLA_PYTHON_CLIENT_MEM_FRACTION` | 限制 JAX 预分配显存比例（0.3=30%） |
| `--jax-preallocate false` | `XLA_PYTHON_CLIENT_PREALLOCATE` | 禁止预分配，按需申请 |
| `--jax-allocator platform` | `XLA_PYTHON_CLIENT_ALLOCATOR` | 使用 on-demand 分配器 |
| `--jax-cuda-visible-devices 0` | `CUDA_VISIBLE_DEVICES` | 指定 GPU |

### 同时跑两个 server 的显存分配

两个 server 各占 ~30% 显存（RTX 6000 Ada = 48GB，每个 ~14GB）：

```bash
# Server 1: Force V6 (port 8000, GPU 0, 30% 显存)
python scripts/serve_policy.py --port 8000 \
  --jax-mem-fraction 0.3 --jax-preallocate false \
  policy:checkpoint \
  --policy.config=pi05_force_lora_local \
  --policy.dir=/mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_lora_local/local_lora_v6/19999

# Server 2: No-force (port 8001, GPU 0, 30% 显存)
python scripts/serve_policy.py --port 8001 \
  --jax-mem-fraction 0.3 --jax-preallocate false \
  policy:checkpoint \
  --policy.config=pi05_panda_noforce_local \
  --policy.dir=/mnt/hdd/sfy/outputs/panda-no-force/29999
```

> **注意**: `--jax-preallocate false` 配合 `--jax-mem-fraction` 效果最好，不会一开始就占满显存。

---

## 使用方法

### 方案 A: 分别跑两个模型，事后对比

#### 1. 启动 Force V6 Server (Terminal 1)

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force

python scripts/serve_policy.py --port 8000 \
  --jax-mem-fraction 0.3 --jax-preallocate false \
  policy:checkpoint \
  --policy.config=pi05_force_lora_local \
  --policy.dir=/mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_lora_local/local_lora_v6/19999
```

#### 2. 跑 Force V6 Rollout (Terminal 2)

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force

# USB 插入
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_v6.py --task usb --port 8000 \
  --episodes 5 --max-steps 400 --num-action-steps 3

# 白板擦
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_v6.py --task whiteboard --port 8000 \
  --episodes 5 --max-steps 1200 --num-action-steps 3
```

#### 3. 启动 No-force Server (Terminal 1, 关掉 V6 server 后)

```bash
python scripts/serve_policy.py --port 8001 \
  --jax-mem-fraction 0.3 --jax-preallocate false \
  policy:checkpoint \
  --policy.config=pi05_panda_noforce_local \
  --policy.dir=/mnt/hdd/sfy/outputs/panda-no-force/29999
```

#### 4. 跑 No-force Rollout (Terminal 2)

```bash
# USB 插入
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_noforce.py --task usb --port 8001 \
  --episodes 5 --max-steps 400 --num-action-steps 3

# 白板擦
PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_noforce.py --task whiteboard --port 8001 \
  --episodes 5 --max-steps 1200 --num-action-steps 3
```

#### 5. 对比结果

```bash
# USB 插入对比
python scripts/compare_rollouts.py \
  --force-dir /mnt/hdd/sfy/outputs/rollouts_v6 \
  --noforce-dir /mnt/hdd/sfy/outputs/rollouts_noforce \
  --task usb

# 白板擦对比
python scripts/compare_rollouts.py \
  --force-dir /mnt/hdd/sfy/outputs/rollouts_v6 \
  --noforce-dir /mnt/hdd/sfy/outputs/rollouts_noforce \
  --task whiteboard
```

### 方案 B: 同时跑两个 server（需要足够显存）

两个 server 同时启动（各占 30% 显存），用不同端口区分。rollout 时指定 `--port` 即可。

---

## 输出文件

### Force V6 (`/mnt/hdd/sfy/outputs/rollouts_v6/`)

| 文件 | 说明 |
|------|------|
| `rollout_v6_{task}_ep{ep}_seed{seed}.mp4` | 视频 |
| `force_pred_v6_{task}_ep{ep}_seed{seed}.json` | Force 预测 vs 实际 |
| `joint_log_v6_{task}_ep{ep}_seed{seed}.json` | 关节轨迹日志 |

### No-force (`/mnt/hdd/sfy/outputs/rollouts_noforce/`)

| 文件 | 说明 |
|------|------|
| `rollout_noforce_{task}_ep{ep}_seed{seed}.mp4` | 视频 |
| `joint_log_noforce_{task}_ep{ep}_seed{seed}.json` | 关节轨迹日志 |

### Joint Log 格式

```json
[
  {
    "step": 0,
    "state": [j1, j2, j3, j4, j5, j6, j7, gripper],  // 8-dim absolute
    "action_abs": [j1, j2, j3, j4, j5, j6, j7, gripper],  // 8-dim target
    "delta": [dj1, dj2, dj3, dj4, dj5, dj6, dj7, dgripper]  // 8-dim delta sent to controller
  },
  ...
]
```

---

## 对比脚本输出

`compare_rollouts.py` 会输出：

1. **Per-episode 对比表**: 每个 episode 的轨迹 L2 差异、最大差异、最终位置差异
2. **Per-joint 最终位置对比**: 7 个关节 + gripper 的均值/标准差/差异
3. **轨迹发散分析**: 随时间变化的 L2 差异（检测 force 模型是否在接触阶段偏离）
4. **总结**: 整体平均/最大轨迹差异

---

## Norm Stats 说明

两个模型的 norm_stats 来源不同：

| 模型 | norm_stats 位置 | state 维度 |
|------|----------------|-----------|
| Force V6 | `/mnt/hdd/sfy/robosuite/datasets/panda-force/norm_stats.json` | 26 维 (8 + 18 force/torque) |
| No-force | `/mnt/hdd/sfy/robosuite/datasets/panda-noforce/norm_stats.json` | 8 维 (joints + gripper) |

> **No-force config 说明**: 使用 `pi05_panda_noforce_local`（本地路径版本），
> `repo_id` 指向 `/mnt/hdd/sfy/robosuite/datasets/panda-noforce`，
> norm_stats 从本地数据集目录自动加载，无需手动复制。
> 原始 `pi05_panda_noforce` 的 repo_id 指向远端 `/data/group1/...`，本地不可达。
