# Config ↔ Checkpoint ↔ Serve 对应总表（权威登记表）

> **目的**: 一张表说清 config 名、checkpoint 目录、serve 模式、norm_stats 来源的对应关系。
> 以后新增 config / checkpoint **必须同步登记到这里**，serve 指令直接从本表复制，不再现场推断。
> 日期: 2026-08-18

---

## 0. 三条铁律（先读这个）

1. **serve 模式只由模型的 action 空间决定**:
   - EEF 模式用于 **action_dim=10 的 EEF 模型**，分两代:
     - **v1** `*_eef_remote`（数据集 `eef_rot6d`，服务端单基准修复 + `--gap-rate` 补偿）
     - **v2** `*_eef_v2_remote`（数据集 `eef_abs`，训练目标已是真实 ramp，**不加 gap-rate**）
     两者都需 `--action-space=EEF --action-rep=delta`
   - 其余一切模型（joint 7 维、dual 联合）**一律默认 joint 模式**，不加任何 `--action-space` 参数
   - dual（eef_joint）虽然训练时有 EEF 辅助 loss，但**模型输出是 joint 7 维，serve 不转 EEF**
2. **config 后缀 `_remote` / `_local` 只代表训练时数据集路径**:
   - `_remote` → repo_id 指向 `/data/group1/junjie008/datasets/...`（SLURM 训练机）
   - `_local` → repo_id 指向 `/mnt/hdd/sfy/datasets/...`（本地）
   - **模型结构/超参完全一致**；本地推理加载 `_remote` config 完全可以，只要用
     `--norm-stats-dir` 把归一化统计指到本地数据集（remote 路径本地找不到，自动 fallback）
3. **norm_stats 永远从训练数据集根目录加载**（`--norm-stats-dir=<数据集根>`，
   其下拼 `norm_stats.json`）。数据集 ↔ 归一化是一一对应的，别交叉用。

---

## 1. 已训好模型总表（serve 指令从此处复制）

| # | checkpoint（本地实际路径） | 训练 config | 家族 | serve 模式 | norm_stats 数据集（本地） | client 发力历史? |
|---|---|---|---|---|---|---|
| 1 | `checkpoints/pi05_eef/36000` | `pi05_plain_total_task_eef_remote` | plain EEF rot6d(10维) | **EEF** | `total_2task_flexiv_eef_rot6d_noforce` | 否 |
| 2 | `checkpoints/pi05_joint/36000` | `pi05_plain_total_task_joint_remote` | plain joint(7维) | joint(默认) | `total_2task_flexiv_ft60_noforce` | 否 |
| 3 | `checkpoints/total_dual/39999` | `pi05_force_total_task_eef_joint_remote` | force dual(joint输出7维+EEF辅loss) | joint(默认) | `total_2task_flexiv_ft60` | **是**(60×6) |
| 4 | `checkpoints/total_joint/39999` | `pi05_force_total_task_joint_only_remote` | force joint_only(7维) | joint(默认) | `total_2task_flexiv_ft60` | **是**(60×6) |

数据集根目录前缀: `/mnt/hdd/sfy/datasets/`
checkpoint 前缀: `/mnt/hdd/sfy/FA-openpi/checkpoints/`

> 训练中（未出最终 ckpt）: **EEF v2 修复轮** `eef_v2_plain` / `eef_v2_force`
> （sbatch: `job_eef_v2_parallel.sbatch`，2026-08-18 起，**单卡并行**（sbatch 内两进程各 45% 显存），bs=8 各 40k 步，
> config: `pi05_plain_total_task_eef_v2_remote` / `pi05_force_total_task_eef_v2_remote`，
> 出 ckpt 后登记步数并复制 §2 ⑤/⑥ 指令改 `--policy.dir` 即可）。
> 回归时期的 `total_eef/39999`（旧 config `pi05_force_total_task_eef_only_remote`，rot6d 改版前）
> 已过时，被 408 的 force_eef_chain 取代。

### 数据集 ↔ 维度速查

| 数据集 | state | action | 用途 |
|---|---|---|---|
| **`total_2task_flexiv_eef_abs`** (v2) | 16 (EEF10 + force6) | **10 绝对** | force EEF v2 训练（训练时合成 delta） |
| **`total_2task_flexiv_eef_abs_noforce`** (v2) | 10 | **10 绝对** | plain EEF v2 训练 / v2 推理归一化 |
| `total_2task_flexiv_eef_rot6d` (v1) | 10 + force6=16 | 10 逐帧 delta ⚠️ | 已废弃（爬行根因，仅存档） |
| `total_2task_flexiv_eef_rot6d_noforce` (v1) | 10 | 10 逐帧 delta ⚠️ | 已废弃（仅 #1 旧 ckpt 归一化还用它） |
| `total_2task_flexiv_ft60` | 13 (6关节+grip+6力) | 7 delta | force joint / dual 训练 |
| `total_2task_flexiv_ft60_noforce` | 7 | 7 delta | plain joint 训练 |

> **v1 vs v2 根因记录 (2026-08-18)**: v1 转换脚本在 parquet 里存**逐帧 delta**
> （相邻帧差），导致 30 步训练 chunk 被压成近似常量 gap → 部署爬行（30 步 ~2cm）。
> v2 (`scripts/convert_dataset_to_eef.py --absolute`) parquet 存**绝对 EEF**，
> 训练时 `EefDeltaActions` 以当前帧 state 为单基准合成 delta（xyz 线性差 +
> rot6d 矩阵合成 `R_state^T @ R_action`，grip 绝对），chunk 内保留完整轨迹 ramp
> （30 帧窗口 max|dx|=0.055m，v1 仅 0.007）。config 开关 `use_eef_delta_actions=True`
> + `use_delta_joint_actions=False`（互斥，同 True 直接 raise）。

---

## 2. 四条 serve 启动指令（一次只开一个，都用 8000 端口）

统一前置:
```bash
cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf
```

### ① pi05_eef —— EEF rot6d 模型（唯一带 EEF 参数的）

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_eef_rot6d_noforce \
    --port=8000 \
    --action-space=EEF --action-rep=delta --gap-rate=0.098 \
    policy:checkpoint \
    --policy.config=pi05_plain_total_task_eef_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/pi05_eef/36000
```

### ② pi05_joint —— plain joint

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60_noforce \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_plain_total_task_joint_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/pi05_joint/36000
```

### ③ total_dual —— force dual（joint 输出，不转 EEF）

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60 \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_eef_joint_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/total_dual/39999
```

### ④ total_joint —— force joint_only

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60 \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_joint_only_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/total_joint/39999
```

### ⑤ EEF v2 plain（训练中，出 ckpt 后改 `--policy.dir` 步数）

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_eef_abs_noforce \
    --port=8000 \
    --action-space=EEF --action-rep=delta \
    policy:checkpoint \
    --policy.config=pi05_plain_total_task_eef_v2_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/eef_v2_plain/40000
```

> ⚠️ v2 模型**不要 `--gap-rate`**：v2 训练目标已是真实轨迹 ramp（不再是常量
> gap 追逐），加 gap 外推会超调。若实测仍慢，再按 §4 坑表调 `--gap-rate`。

### ⑥ EEF v2 force（训练中，出 ckpt 后改 `--policy.dir` 步数；client 需发 wrench_history 60×6）

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_2task_flexiv_eef_abs \
    --port=8000 \
    --action-space=EEF --action-rep=delta \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_eef_v2_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/eef_v2_force/40000
```

### 启动后检查

- 就绪标志: `INFO:websockets.server:server listening on 0.0.0.0:8000`
- norm_stats 加载标志: `Loaded norm_stats from --norm-stats-dir: ...`
  （若先报 `/data/group1/... not found` 属预期——remote config 的 fallback 链）
- 首帧 JIT ~16s（模型+IK），真机前先热身一帧
- #3/#4 client 必须发 `observation/wrench_history` (60×6)；#1/#2 不发
- client: `python scripts/piper_ws_force_client.py --server ws://127.0.0.1:8000 --task "perform the task" ...`

---

## 3. 新增 config / checkpoint 的登记流程（固定步骤）

1. **config 定名规则**: `pi05_{plain|force}_{任务}_{joint|eef|eef_joint}_{local|remote}`
   - `plain` = 无 force（Pi0Config）；`force` = Pi0ForceConfig
   - `eef` = EEF rot6d 模型（action_dim=10, use_delta=False, 数据集 `eef_rot6d`）
   - `eef_joint` = dual（joint 输出 + EEF 辅助 loss）；`joint`/`joint_only` = 纯关节
2. **训练时** 把 sbatch/exp-name → config → checkpoint 输出路径记到 §1 表
3. **训练完成** 后: 更新 §1 表的 checkpoint 步数，并在 §2 复制最近似的一条指令改三处:
   `--policy.config` / `--policy.dir` / `--norm-stats-dir`
4. **禁止**: serve 时临时猜 config 名或用数据集交叉的 norm_stats

## 4. 已踩过的坑（serve 相关）

| 症状 | 原因/解决 |
|---|---|
| `NameError: np is not defined` (serve_policy) | 已修复(2026-08-18): 顶部加 `import numpy as np`（类定义注解在 import 时求值） |
| **EEF 模型冲天/乱飞（chunk 开高严重）** | 已修复(d839622): 旧版对 30 步 chunk 链式复合 delta（cur=target 累加），数据集 delta 含控制超前量 → 复合无界漂移（h29 达 3.16rad）。现改为**单基准**：每步以推理时刻当前 EEF 位姿为基座合成（与 UMI/GR00T/openpi AbsoluteActions 一致），h29 偏差降至 0.054rad。模型权重与 IK 均无问题 |
| **EEF 修复后爬行（30 chunk 才走 ~2cm）** | **根因确认=v1 数据集逐帧 delta**（2026-08-18 已修复）：遥操作 command 领先 state 一个 gap（~0.2rad≈2cm），v1 转换存逐帧差分把 chunk 压成常量 gap → 训练目标无 ramp。**v2 重建**：`convert_dataset_to_eef.py --absolute` 存绝对 EEF + 训练时 `EefDeltaActions` 相对当前帧合成 delta（chunk 保留完整 ramp，30 帧窗口 max dx=0.055m=25×v1）。旧 `--gap-rate 0.098` 是 v1 补丁，v2 模型**不再使用**（会超调）。新数据集: `total_2task_flexiv_eef_abs(_noforce)` |
| `--policy.config` Unrecognized | 参数顺序: `policy:checkpoint` 放 `--policy.*` 前面 |
| norm_stats 找不到 `/data/group1/...` | 预期，`--norm-stats-dir` 指本地数据集根 |
| EEF 模式 IK err~0.001 | 正常（0.001 rad ≈ 0.06°，物理可忽略） |
| OOM / 显存占用满 | `XLA_PYTHON_CLIENT_PREALLOCATE=false` 必带（按需分配） |
