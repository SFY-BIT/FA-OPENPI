# total_2task 真机部署指南（joint / EEF 双模式）

> 模型: Pi0Force（pi05 主干 + LIMoE 4 专家 + 双头 7+6）
> 任务: Erase Board（擦白板）+ Pump bottle（压泵瓶）双任务
> 数据集: total_2task（100ep，Erase 50 + Pump 50）
> 力历史: 60 帧 wrench_history → FTEncoder → 16 token
> 最后更新: 2026-08-16

---

## 0. 前置条件

| 组件 | 路径 |
|---|---|
| 训练机代码 | `/mnt/hdd/sfy/FA-openpi` |
| conda 环境 | `rlinf`（本地训练/推理机） |
| joint checkpoint | `checkpoints/total_joint/39999`（9.4G） |
| eef checkpoint | `checkpoints/total_eef/39999`（9.5G） |
| joint norm_stats | `/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60` |
| eef norm_stats | `/mnt/hdd/sfy/datasets/total_2task_flexiv_eef` |
| 真机 | Piper 6-DoF + 力传感器（末端 0.211m 含夹爪+传感器） |

**三种模型对应关系**：

| 模型 | checkpoint | server 模式 | 说明 |
|---|---|---|---|
| joint_only | `total_joint/39999` | joint（默认） | 关节空间，对照组 |
| eef_only | `total_eef/39999` | `--action-space=EEF` | EEF 坐标，直接 EEF loss |
| eef_joint | `total_dual/39999`（训练中） | `--action-space=EEF` | joint+EEF 联合 |

---

## 1. 启动 server（训练/推理机）

### 1.1 joint 模式（joint_only 模型）

```bash
cd /mnt/hdd/sfy/FA-openpi
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
nohup python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60 \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_joint_only_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/total_joint/39999 \
    > /tmp/serve_joint.log 2>&1 &
```

### 1.2 EEF 模式（eef_only 模型，FK 输入 / IK 输出）

```bash
cd /mnt/hdd/sfy/FA-openpi
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
nohup python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_eef \
    --port=8000 \
    --action-space=EEF \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_eef_only_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/total_eef/39999 \
    > /tmp/serve_eef.log 2>&1 &
```

### 1.3 确认 server 就绪

```bash
# 日志应显示 listening on 0.0.0.0:8000
tail -5 /tmp/serve_joint.log   # 或 serve_eef.log
# 端口确认
ss -tlnp | grep 8000
```

> ⚠️ **重要**：server 首次推理会触发 JIT 编译（模型 + IK），耗时 ~16s。
> 真机开始任务前，建议先发一帧"热身"请求（可用下方测试脚本跑一次），
> 避免真机首步等待过久。

---

## 2. 启动真机 client（真机侧）

```bash
cd /mnt/hdd/sfy/FA-openpi
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

# joint 模式 / EEF 模式都用同一个 client（server 已做转换，client 无感知）
python scripts/piper_ws_force_client.py \
    --server ws://<server_ip>:8000 \
    --task "perform the task" \
    --fps 30 \
    --num-action-steps 5 \
    --duration-s 120
```

**参数说明**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--server` | ws://127.0.0.1:8000 | server 地址（真机与 server 不同机时填 IP） |
| `--task` | stamp seal | 任务 prompt（此处双任务用 perform the task） |
| `--fps` | 30 | 执行频率（send_action 频率） |
| `--num-action-steps` | 1 | 每次推理后执行的 chunk 步数（1=最响应，30=最省算力） |
| `--duration-s` | 120 | 最大运行时长（秒） |
| `--max-joint-delta` | 0.2 | 单步关节限幅（rad）安全保护 |
| `--no-force-history` | off | 跳过力历史（测试用，server tile 当前帧 60 次） |

---

## 3. 离线测试脚本（不连真机，数据集帧验证）

```bash
cd /mnt/hdd/sfy/FA-openpi
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf
python3 << 'PYEOF'
"""从数据集取真实帧 → 发给 server → 检查响应"""
import numpy as np
import pyarrow.parquet as pq
import cv2, sys
sys.path.insert(0, '/mnt/hdd/sfy/tact/Tabero/benchmarks/openpi/openpi-client/src')
from openpi_client import websocket_client_policy as _ws

# joint 模型用 ft60 数据集, EEF 模型也用 ft60 (client 发 joint, server 转 EEF)
DS = '/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60/data/chunk-000/episode_000000.parquet'
df = pq.read_table(DS).to_pandas()
client = _ws.WebsocketClientPolicy(host="127.0.0.1", port=8000)

for i in [0, 500, 1200]:
    row = df.iloc[i]
    state = np.array(row['observation.state'], dtype=np.float32)
    wrench = np.array(row['observation.wrench_history'], dtype=np.float32).reshape(60, 6)
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(row['observation.image']['bytes'], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    wrist = cv2.cvtColor(cv2.imdecode(np.frombuffer(row['observation.wrist_image']['bytes'], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    resp = client.infer({
        "observation/image": img, "observation/wrist_image": wrist,
        "observation/state": state[:7], "observation/wrench_history": wrench,
        "prompt": "perform the task",
    })
    actions = np.asarray(resp["actions"])
    print(f"帧{i}: state={np.round(state[:6],3)} → h0={np.round(actions[0,:6],3)} delta={np.round(actions[0,:6]-state[:6],3)}")
print("测试完成")
PYEOF
```

---

## 4. 常见问题

| 问题 | 原因/解决 |
|---|---|
| `--policy.config` 报 Unrecognized | 参数顺序：`policy:checkpoint` 放在 `--policy.*` 前，或全用 `=` 形式 |
| server 首步 16s 慢 | JIT 编译；先发一帧热身再开始任务 |
| websocket keepalive timeout | 首次推理超 20s；client 需 `ping_interval=None`（已有）或预热 |
| EEF 模式 IK err~0.001 | 欧拉角奇异区；0.001 rad ≈ 0.06° 物理可忽略 |
| norm_stats 报 Not found（/data/group1/...）| 预期行为；已 fallback 到 `--norm-stats-dir` 加载 |
| 动作抖动 | 检查 `--max-joint-delta` 限幅、`--num-action-steps` 是否过小 |

---

## 5. 快速命令备忘

```bash
# 查看 server 日志
tail -f /tmp/serve_joint.log   # joint
tail -f /tmp/serve_eef.log     # eef

# 停 server
pkill -f "serve_policy.py"

# 看 GPU
nvidia-smi
```

> ⚠️ EEF 模式坐标系：FK/IK 都用 tool_extension=0.211（夹爪 0.13503 + 传感器 0.076），
> 与训练/数据集转换完全一致。若真机传感器未安装，需改 `--tool-extension`。
