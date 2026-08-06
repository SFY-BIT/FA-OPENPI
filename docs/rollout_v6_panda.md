# LoRA V6 Panda Force Rollout 使用说明

## 概述

V6 是基于 `pi05_force_lora_local` 配置训练的 LoRA 模型，使用 **Panda** 机器人（7 关节 + 1 gripper）的 force 数据集。

### V6 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| config | `pi05_force_lora_local` | 训练配置名 |
| checkpoint | `checkpoints/pi05_force_lora_local/local_lora_v6/19999` | 最终 checkpoint |
| robot | Panda | 7 arm joints + 1 gripper |
| force_start_idx | 8 | state 前 8 维是关节+gripper |
| force_history_frames | 3 | 使用 3 帧 force/torque 历史 |
| predict_force | True | 模型输出包含 force/torque 预测 |
| use_delta_gripper_actions | True | gripper 也用 delta 训练 |
| force_loss_weight | 0.3 | force loss 权重 |

### State 结构 (26 维)

```
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper,  # 8 dims
 force_t0_Fx, force_t0_Fy, force_t0_Fz,                                    # 3 dims
 force_t1_Fx, force_t1_Fy, force_t1_Fz,                                    # 3 dims
 force_t2_Fx, force_t2_Fy, force_t2_Fz,                                    # 3 dims
 torque_t0_Tx, torque_t0_Ty, torque_t0_Tz,                                 # 3 dims
 torque_t1_Tx, torque_t1_Ty, torque_t1_Tz,                                 # 3 dims
 torque_t2_Tx, torque_t2_Ty, torque_t2_Tz]                                 # 3 dims
```

### Action 结构 (14 维, server 返回绝对值)

```
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper,  # 8 dims (absolute)
 force_Fx, force_Fy, force_Fz,                                             # 3 dims (force prediction)
 torque_Tx, torque_Ty, torque_Tz]                                          # 3 dims (torque prediction)
```

> **注意**: 训练时 action 是 delta，但 server 端 `AbsoluteActions` transform 会自动把 delta 转回绝对值。
> 所以 client 收到的 `actions` 是**绝对关节位置**，需要手动转成 delta 送给 JOINT_POSITION(delta) 控制器。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/serve_policy.py` | Policy server（通用，无需修改） |
| `scripts/rollout_panda_v6.py` | Panda V6 专用 rollout client |

---

## 使用方法

### 1. 启动 Policy Server (Terminal 1)

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
PYTHONPATH=src:packages/openpi-client/src \
python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_force_lora_local \
  --policy.dir=/mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_lora_local/local_lora_v6/19999
```

等待日志出现 `Creating server` 和 `Server listening on ...` 后继续。

### 2. 启动 Rollout Client (Terminal 2)

#### USB 插入任务

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force

PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_v6.py \
  --task usb \
  --host 127.0.0.1 \
  --port 8000 \
  --episodes 5 \
  --max-steps 400 \
  --num-action-steps 1
```

#### 白板擦任务

```bash
conda activate rlinf
cd /mnt/hdd/sfy/openpi-force

PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
python scripts/rollout_panda_v6.py \
  --task whiteboard \
  --host 127.0.0.1 \
  --port 8000 \
  --episodes 5 \
  --max-steps 1200 \
  --num-action-steps 1
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | usb | 任务选择：`usb`（USBInsert）或 `whiteboard`（WhiteboardWipe） |
| `--host` | 127.0.0.1 | Server 地址 |
| `--port` | 8000 | Server 端口 |
| `--episodes` | 1 | 跑多少个 episode |
| `--max-steps` | 0 (自动) | 每个 episode 最大步数（0=按 task 默认：usb=400, whiteboard=1200） |
| `--num-action-steps` | 1 | 每次推理执行几个 action（1=最响应） |
| `--seed-offset` | 0 | 起始 seed |
| `--output-dir` | /mnt/hdd/sfy/outputs/rollouts_v6 | 输出目录 |
| `--gripper-open-steps` | -1 (自动) | 开头强制张开 gripper 的步数（-1=按 task 默认：usb=3, whiteboard=5） |

---

## 控制器配置

V6 rollout 使用 `JOINT_POSITION` 控制器（delta 模式）：

```python
arm_cfg["type"] = "JOINT_POSITION"
arm_cfg["input_type"] = "delta"
arm_cfg["input_max"] = 1
arm_cfg["input_min"] = -1
arm_cfg["output_max"] = 0.1    # 每步最大 0.1 rad ≈ 5.7°
arm_cfg["output_min"] = -0.1
arm_cfg["kp"] = 150
arm_cfg["damping_ratio"] = 1.0  # 临界阻尼
```

### 为什么用 JOINT_POSITION 而不是 OSC？

- 模型输出的是**关节空间 delta**，JOINT_POSITION 直接在关节空间控制，无需空间转换
- OSC_POSE 需要先把关节 delta 转成笛卡尔 delta（正向运动学），引入额外误差
- JOINT_POSITION 不受雅可比奇异点影响，对 USB 插入任务更稳定

### Delta 转换逻辑

```python
# Server 返回绝对 action
action_abs = response["actions"]  # (30, 14), absolute

# 转成 delta 给控制器
current_state = panda_state(env)  # 8-dim
delta = action_abs[:8] - current_state  # joint + gripper delta
delta = np.clip(delta, -1.0, 1.0)       # clip to controller input range
env.step(delta)
```

---

## 输出文件

每个 episode 会生成：

1. **视频**: `rollout_v6_{task}_ep{ep}_seed{seed}.mp4` — agentview + wrist 拼接视频
2. **Force 日志**: `force_pred_v6_{task}_ep{ep}_seed{seed}.json` — 每步的 force 预测 vs 实际值

### Force 日志格式

```json
[
  {
    "step": 0,
    "pred_force": [0.1, 0.2, -8.5],
    "actual_force": [0.12, 0.18, -8.7],
    "pred_torque": [0.01, -0.02, 0.0],
    "actual_torque": [0.02, -0.01, 0.01],
    "force_err": 0.234,
    "torque_err": 0.015
  },
  ...
]
```

---

## 数据集 Force 区间参考

V6 数据集 (`/mnt/hdd/sfy/robosuite/datasets/panda-force`) 的 force 数值区间：

| 维度 | 全局最小 | 全局最大 |
|------|---------|---------|
| Fx | -6.78 | 15.14 |
| Fy | -14.90 | 10.04 |
| Fz | -17.33 | 48.76 |
| Tx | -1.15 | 1.52 |
| Ty | -1.19 | 1.38 |
| Tz | -0.44 | 0.63 |

Force 最大值 48.76 N (Fz)，未超过 50。

---

## 故障排查

### Server 启动失败

- 检查 checkpoint 路径是否存在: `ls /mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_lora_local/local_lora_v6/19999/params/`
- 检查 GPU 显存: `nvidia-smi`
- 如果 OOM，调低 `XLA_PYTHON_CLIENT_MEM_FRACTION`

### Client 连接失败

- 确认 server 已启动并监听正确端口
- Client 会自动重试 60 次（每次间隔 2 秒）

### Force 数据异常

- 检查 `force_pred_*.json` 中的 `force_err` 是否过大
- 如果 force 预测全为 0，检查 server 是否正确加载了 V6 checkpoint
