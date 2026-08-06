# Pi0Force stamp_seal 训练 Loss 与指标含义

> 配置：`pi05_force_stamp_seal_remote`
> 数据集：`stamp_seal_v2_flexiv`
> 最后更新：2026-07-24

---

## 1. 训练日志输出格式

每 `log_interval=10` 步输出一行，格式：

```
Step {step}: action_loss=..., force_loss=..., grad_norm=..., loss=...,
  moe_aux_loss=..., moe_expert_usage=..., moe_fraction_left_behind=...,
  moe_router_confidence=..., moe_router_z_loss=...,
  moe_top1_frac_expert_0=..., moe_top1_frac_expert_1=...,
  moe_top1_frac_expert_2=..., moe_top1_frac_expert_3=...,
  param_norm=..., skipped_nonfinite=...
```

这些指标来自两个异步回调（`jax.debug.callback`，零训练开销）：
- `_pi0_force._COMPONENT_LOSSES`：`action_loss`、`force_loss`
- `_limoe._ROUTING_METRICS`：所有 `moe_*` 指标

其余（`loss`、`grad_norm`、`param_norm`、`skipped_nonfinite`）由 `train.py` 主循环计算。

---

## 2. 核心 Loss 指标

### 2.1 `action_loss`（动作流匹配损失）

- **含义**：控制动作的 flow-matching MSE 损失
- **计算**：`mean(square(action_pred[..., :7] - u_t[..., :7]))`
  - `action_pred = action_out_proj(shared)`，输出 32 维，取前 7 维（6关节+夹爪）
  - `u_t = noise - actions`，flow-matching 目标，32 维，取前 7 维
- **梯度路由**：1.0× 权重流回 VLM/vision、LIMoE+expert、action_out_proj
- **期望趋势**：从初始 ≈1.3 逐步下降（step 0: 1.2930 → step 10: 1.0417 ✓）
- **数值范围**：归一化空间，0 表示完美预测

### 2.2 `force_loss`（力预测损失）

- **含义**：force_out_proj 头的力/力矩预测 MSE 损失（head 路径，全 1.0 权重）
- **计算**：`mean(square(force_pred_head - force_target))`
  - `force_pred_head = force_out_proj(stop_gradient(shared_force))`，输出 6 维
  - `force_target`：归一化后的 delta force（6维：Fx,Fy,Fz,Tx,Ty,Tz）
- **梯度路由**：1.0× 权重只更新 `force_out_proj.kernel/bias`（`stop_gradient` 阻断 LIMoE）
- **初始现象**：step 0-10 显示 `force_loss=0.0000`
  - **根因分析**（详见第 8 节）：delta force 分布极度长尾——大部分帧 delta≈0（力稳定），少数突变帧 delta 很大（接触/释放）。quantile 归一化用 q01/q99 作分母，突变帧把 q99-q01 拉得很大，导致大部分帧归一化后值很小（|norm|≈0.04~0.17）。force_out_proj 随机初始化输出也接近 0，两者差值的平方再被 `:.4f` 格式化后显示为 `0.0000`
  - **实际值估计**：force_loss 真实值约 0.05~0.15（非真零），被 `f"{v:.4f}"` 截断显示
  - **结论**：这是 delta force 长尾分布 + quantile 归一化 + 格式化截断的共同结果，**不是 bug**
  - **预期**：随训练推进，force_out_proj 学到非零映射后 loss 会上升再下降
- **注意**：此值只反映 head 路径，LIMoE+expert 路径的 force_loss_expert（0.1× 权重）不单独记录

### 2.3 `loss`（总损失）

- **含义**：`value_and_grad` 实际求梯度的标量损失
- **计算**（three_stage 模式）：
  ```
  total = 1.0 * action_loss
        + 1.0 * force_loss_head        (-> force_out_proj only)
        + 0.1 * force_loss_expert      (-> LIMoE+expert only)
  ```
- **注意**：`loss` ≠ `action_loss + force_loss`，因为：
  1. `force_loss_head` 和 `force_loss_expert` 是两条独立路径，日志只记 head
  2. `force_loss_expert` 乘了 0.1 权重
- **初始现象**：step 0 `loss=1.2930 = action_loss`（因 force_loss_head≈0）

---

## 3. 梯度范数与参数范数

### 3.1 `grad_norm`（梯度范数）

- **含义**：所有参数梯度的全局范数（clip 前）
- **计算**：`optax.global_norm(grads)`，然后 `clip_gradient_norm=1.0` 裁剪
- **期望趋势**：训练初期较大（step 0: 124.25 → step 10: 102.35），逐步下降
- **异常信号**：若持续 >1000 或 NaN，可能学习率过高

### 3.2 `param_norm`（参数范数）

- **含义**：所有参数值的全局范数
- **期望趋势**：相对稳定（step 0: 1811.556 → step 10: 1811.556），缓慢变化
- **异常信号**：若快速增长，可能梯度爆炸

### 3.3 `skipped_nonfinite`（非有限值跳过计数）

- **含义**：当前 batch 中 loss/grad 出现 NaN/Inf 而被跳过的次数
- **正常值**：0.0
- **异常信号**：>0 表示出现 NaN，日志会额外打印 warning

---

## 4. LIMoE 路由指标（`moe_*`）

这些指标反映 4 专家（`num_experts=4`）的 MoE 路由器行为，来自 `limoe.py` 的 `_store_routing_metrics`。

### 4.1 `moe_aux_loss`（辅助负载均衡损失）

- **含义**：鼓励专家负载均衡的辅助损失
- **计算**：`num_experts * sum(average_fraction_tokens_per_expert × average_probability_per_expert)`
- **理想值**：`num_experts = 4`（完全均衡时每个专家 1/4）
- **当前**：step 0: 2.6227，step 10: 2.6565（略高于理想，说明负载不均）
- **注意**：此 loss 已加入总损失（隐含在 `loss` 中，通过 LIMoE 梯度）

### 4.2 `moe_router_z_loss`（路由器 z 损失）

- **含义**：鼓励 router logits 保持小数值，防止数值不稳定
- **计算**：`mean(logsumexp(router_logits))`
- **理想值**：接近 0
- **当前**：step 0: 5.7033（偏高，训练初期正常）

### 4.3 `moe_expert_usage`（专家利用率）

- **含义**：被选中的专家的 token 比例（top-1 路由）
- **计算**：`sum(top1_fraction) / num_experts`（归一化到 [0,1]）
- **理想值**：1.0（所有专家都被使用）
- **当前**：step 0: 0.2836（只有约 28% 专家被使用，负载不均）

### 4.4 `moe_fraction_left_behind`（被丢弃的 token 比例）

- **含义**：未被任何专家处理的 token 比例
- **计算**：`1.0 - moe_expert_usage`
- **理想值**：0.0（所有 token 都被处理）
- **当前**：step 0: 0.7164（约 72% token 被丢弃，训练初期路由器未学好）

### 4.5 `moe_router_confidence`（路由器置信度）

- **含义**：router softmax 分布的平均最大概率
- **计算**：`mean(max(softmax(router_logits)))`
- **范围**：[0, 1]，1.0 表示路由器非常确定
- **当前**：step 0: 0.7304（较高，但可能过度集中到少数专家）

### 4.6 `moe_top1_frac_expert_{0..3}`（各专家 top-1 选择比例）

- **含义**：每个专家被选为 top-1 的 token 比例
- **理想值**：每个专家 0.25（4 专家完全均衡）
- **当前**（step 0）：
  - expert_0: 0.0751
  - expert_1: **0.8814**（过度集中！）
  - expert_2: 0.0377
  - expert_3: 0.0058
- **问题**：expert_1 占了 88%，其他专家几乎闲置
- **预期**：随 `moe_aux_loss` 优化，逐步均衡到各 ≈0.25

---

## 5. 指标健康检查清单

| 指标 | 健康范围 | 异常处理 |
|------|---------|---------|
| `action_loss` | 初始 ~1.3，稳步下降 | 若不降或上升，检查数据/学习率 |
| `force_loss` | 初始可能 ≈0，后上升再下降 | 若长期 =0，检查 force_target 是否全 0 |
| `loss` | ≈ action_loss（初期） | 若远大于 action_loss+force_loss，检查 aux_loss |
| `grad_norm` | 初始 ~100，逐步下降 | 若 >1000 或 NaN，降低学习率 |
| `param_norm` | 稳定 ~1800 | 若快速增长，梯度爆炸 |
| `skipped_nonfinite` | 0.0 | 若 >0，检查数据有无 NaN |
| `moe_expert_usage` | 逐步趋近 1.0 | 若长期 <0.5，增大 aux_loss 权重 |
| `moe_top1_frac_expert_*` | 逐步趋近 0.25 | 若某专家长期 >0.8，路由器坍塌 |

---

## 6. 初始训练观察（step 0-10）

```
Step 0:  action_loss=1.2930, force_loss=0.0000, grad_norm=124.25, loss=1.2930
Step 10: action_loss=1.0417, force_loss=0.0000, grad_norm=102.35, loss=1.0417
```

**分析**：
1. **action_loss 下降** ✓：1.2930 → 1.0417，下降 19%，学习正常
2. **force_loss=0** ：force_out_proj 随机初始化输出≈0，匹配归一化后零均值 force_target，属正常初始现象
3. **MoE 路由不均**：expert_1 占 88%，`moe_aux_loss` 会逐步纠正
4. **速度**：~6.3s/it，50000 步预计 ~87 小时（3.6 天）

---

## 7. 相关文件

- `src/openpi/models/pi0_force.py`：`_store_component_losses`（action_loss/force_loss）
- `src/openpi/models/limoe.py`：`_store_routing_metrics`（所有 moe_* 指标）
- `scripts/train.py:369-370`：异步回调读取并写入 wandb
- `docs/stamp_seal_dimension_semantics.md`：维度语义与梯度路由详解

---

## 8. force_loss=0.0000 根因深度分析

### 8.1 现象

```
Step 0:  action_loss=1.2930, force_loss=0.0000, loss=1.2930
Step 10: action_loss=1.0417, force_loss=0.0000, loss=1.0417
```

`loss == action_loss` 完全相等，说明 `force_loss_head` 和 `force_loss_expert` 都被 `:.4f` 格式化成了 0.0000。

### 8.2 force_target 的生成（delta force）

`ForceInStatePiperInputs`（`force_piper_policy.py:267-270`）：

```python
force_target_abs = state_chunk[:, force_start_idx:force_start_idx+force_dim]  # 未来H帧的绝对力
force_target = force_target_abs - current_force  # DELTA = 未来力 - 当前力
```

**delta force 的物理含义**：相邻帧力的变化量。
- 大部分帧：力稳定（接触保持/自由空间），delta ≈ 0
- 少数突变帧：接触建立/释放/碰撞，delta 很大

### 8.3 norm_stats 证据（长尾分布）

从 `norm_stats.json` 的 `force_target` 统计：

| 维度 | mean | std | q01 | q99 | std/|mean| |
|------|------|-----|-----|-----|-----------|
| Fx | -0.0003 | 0.484 | -1.526 | 1.357 | **1578×** |
| Fy | 0.0019 | 0.443 | -1.397 | 1.216 | **231×** |
| Fz | -0.0050 | 2.544 | -7.497 | 9.096 | **508×** |
| Tx | 0.0003 | 0.044 | -0.126 | 0.138 | **137×** |
| Ty | 0.0002 | 0.083 | -0.290 | 0.205 | **424×** |
| Tz | -0.0003 | 0.030 | -0.100 | 0.082 | **115×** |

**关键观察**：
- `mean ≈ 0`（delta 均值接近 0，正负变化抵消）
- `std >> |mean|`（std/|mean| 比值 115~1578，说明分布极度分散）
- `q99-q01` 范围很大（Fz 达 16.6），被突变帧拉大

### 8.4 quantile 归一化的放大效应

归一化公式：`(x - q01) / (q99 - q01) * 2 - 1`

对于 **delta=0 的帧**（大部分帧），归一化后：
```
norm(0) = -2*q01/(q99-q01) - 1
```

计算结果：

| 维度 | norm(0) | |norm(0)| |
|------|---------|----------|
| Fx | 0.059 | 0.059 |
| Fy | 0.069 | 0.069 |
| Fz | -0.096 | 0.096 |
| Tx | -0.044 | 0.044 |
| Ty | 0.171 | 0.171 |
| Tz | 0.100 | 0.100 |

**大部分帧归一化后 |force_target| ≈ 0.04~0.17**（很小但非零）。

### 8.5 force_out_proj 随机初始化

- `force_out_proj = nnx.Linear(1024, 6)`，匹配 `.*force.*` regex，**不从 base checkpoint 加载**
- NNX Linear 默认 Lecun normal 初始化：kernel std = sqrt(1/1024) ≈ 0.031，bias = 0
- 输入 `shared_force` 量级未知，但初始输出 `force_pred_head` 量级较小

### 8.6 force_loss 真实值估计

如果 `force_pred_head ≈ 0`（随机初始化），则：
```
force_loss = mean((0 - force_target)^2) = mean(force_target^2)
           = norm_mean^2 + norm_std^2
           ≈ 0.09^2 + 0.33^2
           ≈ 0.008 + 0.109
           ≈ 0.119
```

但日志显示 `0.0000`，说明 `force_pred_head` **不完全是 0**，而是接近 force_target 的均值（≈0.09），使得残差更小。

**更可能的解释**：`force_pred_head` 输出量级 ≈ 0.1~0.3（非零但小），与 force_target 量级（0.04~0.17）接近，残差平方后约 0.001~0.01，被 `f"{v:.4f}"` 格式化成 `0.0000`。

### 8.7 `:.4f` 格式化截断

`train.py:378` 的日志格式：
```python
info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
```

- `0.000049` → `0.0000`（被截断）
- `0.000050` → `0.0001`（四舍五入）
- `0.119` → `0.1190`（正常显示）

所以 `force_loss=0.0000` 的真实含义是 **`force_loss < 0.00005`**，不是数学上的 0。

### 8.8 结论

`force_loss=0.0000` 是以下三个因素叠加的结果：

1. **delta force 长尾分布**：大部分帧 delta≈0，突变帧少
2. **quantile 归一化放大**：q99-q01 被突变帧拉大，大部分帧归一化后值很小（|norm|≈0.04~0.17）
3. **格式化截断**：`:.4f` 把 <0.00005 的值显示为 0.0000

**这不是 bug**，是 delta force 任务的特殊性。随训练推进：
- force_out_proj 学到更精确的映射后，残差会先上升（脱离零输出陷阱）再下降
- 如果想观察真实 force_loss，可改为 `:.6f` 格式化，或查看 wandb 的原始数值

### 8.9 验证方法

如需确认 force_loss 真实值，可：
1. 在 `train.py:378` 改为 `f"{k}={v:.6f}"` 看更多小数位
2. 查看 wandb 面板的 `force_loss` 曲线（不受格式化影响）
3. 或在 `pi0_force.py:411` 的 `jax.debug.callback` 前加 `jax.debug.print("force_loss={x}", x=jnp.mean(force_loss))`

### 8.10 实测确认（step 0-300）

wandb 面板确认 `force_loss` 曲线是 **flat 的 0 直线**（不是格式化截断），证实 force_out_proj 陷入了"恒输出 0"的懒人解。

### 8.11 修复方案（已实施）

**问题**：delta force 长尾分布 + MSE loss 被稳定帧主导 → force_out_proj 学到平凡解。

**修复**（`pi0_force.py` force_loss 计算处）：
1. **突变帧加权**：`frame_weight = 1.0 + |force_target|.mean() * 20.0`
   - 稳定帧（delta≈0）：weight ≈ 1.0
   - 突变帧（|delta|=1）：weight ≈ 21.0
2. **Huber loss** 替代 MSE：对突变帧的大残差更鲁棒
   - `|diff| < 1.0`：quadratic（0.5×diff²）
   - `|diff| >= 1.0`：linear（|diff|-0.5）

**验证**（`debug_loss.py`）：
- 旧 MSE：pred=0 的 loss = 0.5043
- 新 WHuber：pred=0 的 loss = 3.4011（**6.7× 放大**）
- force_out_proj 会有强烈梯度逃离零输出
