# Pi0Force stamp_seal 维度语义与梯度路由

> 配置：`pi05_force_stamp_seal`（本地）/ `pi05_force_stamp_seal_remote`（远端）
> 数据集：`stamp_seal_v2_flexiv`（Piper/Flexiv，13维 state 内嵌 force）
> 最后更新：2026-07-24

---

## 1. 数据集原始格式（stamp_seal_v2_flexiv）

| 字段 | 维度 | 语义 | 说明 |
|------|------|------|------|
| `observation.state` | `[13]` | `[q1, q2, q3, q4, q5, q6, gripper, Fx, Fy, Fz, Tx, Ty, Tz]` | 7 关节+夹爪 + 6 力/力矩 |
| `observation.state_history` | `[K=2, 13]` | 过去 K 帧的完整 state | `ForceHistoryAugmentedDataset` 生成，`[oldest, newest]` |
| `action` | `[7]` | `[target_q1..target_q6, target_gripper]` | 7维控制动作（无 force） |
| `observation.image` | `[480, 640, 3]` | base 相机 | uint8 |
| `observation.wrist_image` | `[480, 640, 3]` | wrist 相机 | uint8 |
| `prompt` | str | `"stamp seal"` | 任务提示 |

**chunked 采样**：data_loader 通过 `delta_timestamps` 把 `observation.state` 和 `action` 采样为未来 chunk：
- `observation.state` → `[H=30, 13]`（当前帧 = 第0行）
- `action` → `[H=30, 7]`

---

## 2. 变换管道各阶段维度

### 2.1 RepackTransform（点号 key → 斜杠 key）

| 输入 key | 输出 key |
|---------|---------|
| `observation.image` | `observation/image` |
| `observation.wrist_image` | `observation/wrist_image` |
| `observation.state` | `observation/state` |
| `observation.state_history` | `observation/state_history` |
| `action` | `actions` |
| `prompt` | `prompt` |

### 2.2 ForceInStatePiperInputs（核心拆分）

输入：`observation/state = [H, 13]`，`observation/state_history = [K=2, 13]`

**拆分逻辑**：
- `proprio = state[0, 0:7]` → `[7]`（当前帧关节+夹爪）
- `current_force = state[0, 7:13]` → `[6]`（当前帧力/力矩）
- `force_history = state_history[:, 7:13]` → `[K=2, 6]` → flatten → `[12]`

**输出**：

| key | 维度 | 语义 |
|-----|------|------|
| `state` | `[19]` | `[proprio(7), oldest_force(6), newest_force(6)]` |
| `actions` | `[H=30, 7]` | 控制动作（绝对值，待 DeltaActions 处理） |
| `force_target` | `[H=30, 6]` | **delta**：`future_force[h] - current_force` |
| `image` | dict | 3 个相机（base/wrist/dummy） |
| `prompt` | str | `"stamp seal"` |

### 2.3 DeltaActions（action delta 变换）

mask = `make_bool_mask(6, -1)` = `(True, True, True, True, True, True, False)`

| 维度 | mask | 变换 |
|------|------|------|
| `actions[0:6]`（6关节） | `True` | `delta = action - current_state[0:6]` |
| `actions[6]`（夹爪） | `False` | 绝对值，不变 |

### 2.4 Normalize（归一化）

对 `state`、`actions`、`force_target` 三个 key 应用 quantile 归一化（pi05 默认 `use_quantile_norm=True`）：
- `force_target` 的 norm_stats 是对 **delta** 计算的（因为 DeltaActions 在 Normalize 之前... 实际上 force_target delta 在 ForceInStatePiperInputs 里已生成，Normalize 在其后）

### 2.5 PadStatesAndActions（padding 到模型维度）

| key | 变换前 | 变换后 | 说明 |
|-----|--------|--------|------|
| `state` | `[19]` | `[32]` | zero-pad 到 `base_action_dim=32`（与 base checkpoint 兼容） |
| `actions` | `[H=30, 7]` | `[H=30, 32]` | zero-pad 到 `action_dim=32` |
| `force_target` | `[H=30, 6]` | `[H=30, 6]` | 不 pad（独立头） |

---

## 3. 模型内部维度

### 3.1 投影层

| 层 | 输入维度 | 输出维度 | 说明 |
|----|---------|---------|------|
| `state_proj` | 32 | 1024 | proprio pad 到 32 后投影（保持 32 以兼容 base checkpoint） |
| `action_in_proj` | 7 | 1024 | noisy action 输入投影（重建为 7 维） |
| `action_out_proj` | 1024 | 7 | 控制动作输出投影（重建为 7 维） |
| `force_in_proj` | 6 | 2048 | 每帧 force history 投影到 token |
| `force_out_proj` | 2048 | 6 | force/torque 预测头（独立） |

### 3.2 关键维度参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `base_action_dim` | 32 | 原始 pi05 action_dim（state_proj 输入维度，保持兼容） |
| `action_dim`（覆盖后） | 7 | `control_action_dim`，flow-matching noise/target 维度 |
| `control_action_dim` | 7 | 6 关节 + 1 夹爪 |
| `force_start_idx` | 7 | state 中 force 起始位置 |
| `force_dim` | 6 | Fx, Fy, Fz, Tx, Ty, Tz |
| `force_history_frames` | 2 | K=2 帧 force history 作为输入 token |
| `action_horizon` | 30 | 预测未来 30 步 |

### 3.3 LIMoE 输入

```
LIMoE input = [prefix_out (VLM输出), force_token_0 (oldest), force_token_1 (newest)]
```
- `prefix_out`：PaliGemma VLM 输出（含图像+文本+state token）
- `force_token_i`：`force_in_proj(force_history[i])` → `[2048]`

---

## 4. 预测方式总结

| 输出 | 维度 | 预测方式 | 训练目标 | 推理还原 |
|------|------|---------|---------|---------|
| 6 关节 | `[H, 6]` | **delta** | `u_t = noise - action`（flow-matching） | `AbsoluteActions: action + current_state[0:6]` |
| 1 夹爪 | `[H, 1]` | **绝对值** | flow-matching | 不变 |
| 6 force | `[H, 6]` | **delta** | `force_target = future_force - current_force`（supervised MSE） | `ForceInStatePiperOutputs: force_pred + current_force` |

**force delta 实现细节**：
- 训练（`ForceInStatePiperInputs`）：`force_target = state_chunk[:, 7:13] - current_force`
- 推理（`ForceInStatePiperOutputs`）：`force_pred_abs = force_pred_delta + state[7 + (K-1)*6 : 7 + K*6]`（从 state 末尾取 newest force）

---

## 5. 梯度路由（方案 B+，单次 value_and_grad）

### 5.1 梯度路由表

| 参数组 | 收到的梯度 | 实现方式 |
|--------|-----------|---------|
| VLM/视觉（prefix_out） | `1.0 × action_loss` | force 路径 `stop_gradient(prefix_out)` 阻断 force |
| LIMoE + action expert | `1.0 × action + 0.1 × force` | action 路径 full gradient + force 路径 Path 2 |
| `action_out_proj` | `1.0 × action_loss` | action 路径 full gradient |
| `force_out_proj` | `1.0 × force_loss` | force 路径 Path 1：`stop_gradient(shared_force)` 阻断上游 |
| `force_in_proj` | `1.0 × action + 0.1 × force` | 两条路径都流经 force_tokens |

### 5.2 双 force loss 路径（互补 stop_gradient）

```
total_loss = 1.0 × action_loss
           + 1.0 × force_loss_head        (Path 1: -> force_out_proj only)
           + 0.1 × force_loss_expert      (Path 2: -> LIMoE + expert only)
```

- **Path 1**（force 头专用）：`force_pred_head = force_out_proj(stop_gradient(shared_force))`
  - `stop_gradient(shared_force)` 阻断 LIMoE+expert，梯度只到 `force_out_proj.kernel/bias`
- **Path 2**（LIMoE+expert 专用）：`force_pred_expert = shared_force @ stop_gradient(kernel) + stop_gradient(bias)`
  - `stop_gradient(kernel/bias)` 阻断 force_out_proj，梯度只到 `shared_force` → LIMoE+expert

### 5.3 NNX Linear 手动计算

```python
# NNX Linear: y = x @ kernel + bias  (kernel shape = (in, out), 无需转置)
force_pred_expert = shared_force @ stop_gradient(force_out_proj.kernel.value) \
                              + stop_gradient(force_out_proj.bias.value)
```

---

## 6. 归一化

### 6.1 norm_stats 内容

`compute_norm_stats.py` 计算三个 key 的统计量：
- `state`：19维（proprio + force history）
- `actions`：7维（delta 后的 action）
- `force_target`：6维（delta 后的 force）

### 6.2 归一化方式

- `use_quantile_norm=True`（pi05 默认）
- quantile 归一化：`(x - q01) / (q99 - q01) * 2.0 - 1.0` → 映射到 `[-1, 1]`

### 6.3 归一化命令

```bash
cd /mnt/hdd/sfy/openpi-force
HF_DATASETS_CACHE=/mnt/hdd/sfy/openpi/.cache/datasets \
PYTHONPATH=src conda run -n rlinf python scripts/compute_norm_stats.py \
    --config-name pi05_force_stamp_seal
```

输出：`assets/pi05_force_stamp_seal/<repo_id>/norm_stats.json`
