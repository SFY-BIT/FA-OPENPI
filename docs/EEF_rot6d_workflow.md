# EEF rot6d 训练工作流程 (FA-openpi)

> 记录 EEF rot6d 重构后的完整工作流: 数据转换 → 归一化 → 远端部署 → 训练编排 → 监控
> 日期: 2026-08-17 (commit 9b67188 + 2811fd2)

---

## 0. 背景: 为什么 rot6d

### 万向节锁问题 (rpy 表示的病灶)
- 任务姿态经过 pitch≈±90° (万向节锁), rpy 欧拉角分解不唯一
- FK 输出的真实旋转慢速连续, 但 canonical rpy 表示相邻帧可滑动 3+ rad
- v1~v10.3 的 rpy hack 全部是治标: unwrap / sgn 分段 / 共模滤波 (λ=cos²p) / 双表示族选择
- 滤波引入表示误差 (max 0.36-0.54 rad), 锁区内模型旋转监督始终有偏

### rot6d 表示 (UMI / NVIDIA GR00T 标准)
- state  = `[xyz(3), rot6d(6), grip(1)]` 绝对 EEF 位姿 (10 维)
- action = `[d_xyz(3), d_rot6d(6), grip(1)]` 相对 delta (10 维)
  - d_xyz   = xyz_target - xyz_state (线性差)
  - d_rot6d = rot6d( R_state^T @ R_target ) (矩阵合成, 与 UMI "rel"/GR00T "relative" 同)
- rot6d = 旋转矩阵前两行展平 (row-major), Gram-Schmidt 还原
- 连续无奇点, 无万向节锁, 无需任何 unwrap/滤波

---

## 1. 数据集转换

### 输入
- `total_2task_flexiv_ft60` (force 版, state 13 维 = 6关节+grip+6力)
- `total_2task_flexiv_ft60_noforce` (plain 版, state 7 维 = 6关节+grip)

### 命令
```bash
# force 版 (state 13→16 维: xyz3+rot6d6+grip1+force6)
python scripts/convert_dataset_to_eef.py \
    --input /mnt/hdd/sfy/datasets/total_2task_flexiv_ft60 \
    --output /mnt/hdd/sfy/datasets/total_2task_flexiv_eef_rot6d \
    --tool-extension 0.211

# noforce 版 (state 7→10 维: xyz3+rot6d6+grip1)
python scripts/convert_dataset_to_eef.py \
    --input /mnt/hdd/sfy/datasets/total_2task_flexiv_ft60_noforce \
    --output /mnt/hdd/sfy/datasets/total_2task_flexiv_eef_rot6d_noforce \
    --tool-extension 0.211
```

### 核心函数 (scripts/convert_dataset_to_eef.py)
- `mat_to_rot6d(R)` / `rot6d_to_mat(d6)`: 旋转矩阵 ↔ rot6d (Gram-Schmidt)
- `joints_to_eef_abs_batch(joints, tool_extension)`: 关节 → EEF 绝对位姿 [N,9]
- `make_relative_delta(state_eef, action_eef)`: 绝对 → 相对 delta (矩阵合成)
- `convert_episode(...)`: 单集转换 (state/action 前 6 关节 → EEF, 其余列不动)

### 验证要点
- rot6d 还原正交: det(R)=1, R@Rᵀ=I
- delta 合成误差: `R_state @ R_delta ≈ R_target`, 应 ~1e-8
- 帧间增量: rot6d ~0.04 (vs rpy 的 0.19), 无跳变
- 位置合成: `xyz_state + d_xyz = xyz_target` 精确

### 环境 (本地)
- conda env: **rlinf** (`/home/sfy/miniconda3/envs/rlinf`, jax 0.5.3)
- 必须 `PYTHONPATH=/mnt/hdd/sfy/FA-openpi/src` (否则 import 到 /mnt/hdd/sfy/openpi 另一个仓库)

---

## 2. norm_stats 计算

### 命令
```bash
# plain (repo_id 指向 noforce 数据集)
PYTHONPATH=/mnt/hdd/sfy/FA-openpi/src python scripts/compute_norm_stats.py \
    --config-name=pi05_plain_total_task_eef

# force (repo_id 指向 force 数据集)
PYTHONPATH=/mnt/hdd/sfy/FA-openpi/src python scripts/compute_norm_stats.py \
    --config-name=pi05_force_total_task_eef_only_local
```

### 输出 (写入数据集根目录 norm_stats.json)
- plain: `state(10)` + `actions(10)`
- force: `state(10)` + `actions(10)` + `ft_state(360)` + `force_target(6)`
  - state 只有 10 维 (proprio), force 走 ft_state 独立通道 (use_ft_history=True)
- 全量扫描 ~77522 帧, 耗时 ~13 分钟

### 归一化/反归一化机制
- `Normalize`/`Unnormalize` (transforms.py) 按 key 逐维用 quantile q01/q99
- 归一化后 ≈[-1,1], 反归一化精确恢复 (误差 ~1e-16)
- **loss 在归一化数值空间算** (flow matching 天然), 无需改 loss 代码
- **关闭 use_delta_joint_actions** (rot6d 不能线性减, 数据已预处理为 delta)

---

## 3. Config 改动 (src/openpi/training/config.py)

### EEF config (改 rot6d, 4 个)
| config | repo_id | control_action_dim | force_start_idx | action_dim | use_delta |
|---|---|---|---|---|---|
| pi05_force_total_task_eef_only_remote | .../eef_rot6d | 10 | 10 | 10 | False |
| pi05_force_total_task_eef_only_local | /mnt/hdd/sfy/datasets/eef_rot6d | 10 | 10 | 10 | False |
| pi05_force_total_task_eef_only_standard | .../eef_rot6d | 10 | 10 | 10 | False |
| pi05_plain_total_task_eef | /mnt/hdd/sfy/datasets/eef_rot6d_noforce | — | — | 10 | False |
| pi05_plain_total_task_eef_remote | .../eef_rot6d_noforce | — | — | 10 | False |

### 未动 (joint/dual 保持原样)
- pi05_force_total_task_joint_only_remote (7 维, delta=True, ft60)
- pi05_force_total_task_eef_joint_remote (dual 联合, 7 维, delta=True, ft60)
- pi05_plain_total_task_joint(_remote) (7 维, delta=True, ft60_noforce)

### 模型一致性 (EEF 与 joint 对齐)
- plain: 都用 `pi0_config.Pi0Config` (pi05=True, action_horizon=30)
- force: 都用 `pi0_force.Pi0ForceConfig` (pi05=True, action_horizon=30)

---

## 4. serve_policy 改动 (scripts/serve_policy.py)

### EefActionPolicyWrapper
- 输入: client 发 joint(7) → FK → EEF rot6d(10) 喂模型
- 输出: 模型 EEF delta(10) → 矩阵合成绝对 → IK → joint(7) 返回 client
- `action_rep` 参数: "delta" (rot6d EEF 模型) / "abs" (旧 joint baseline)
- force 模型: state(10 proprio) + wrench_history(60帧) 独立输入, client 不改

### piper_fk_jax.eef_ik 改动
- 加 `target_R` 参数: 直接传旋转矩阵, 绕开 rpy (内部已用 R 误差, 只改入口)

---

## 5. 远端部署

### 代码同步 (git, 已推送 origin/main)
```bash
cd /data/group1/junjie008/FA-openpi
git stash          # 保存本地修改 (如 3cfg.sbatch)
git fetch origin
git reset --hard origin/main   # restricted shell 下 pull 常失败, 用这个
git log --oneline -5           # 确认顶部是 2811fd2
git status
```

### 数据集上传 (sftp, put -r)
```bash
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg
cd /data/group1/junjie008/datasets
put -r /mnt/hdd/sfy/datasets/total_2task_flexiv_eef_rot6d
put -r /mnt/hdd/sfy/datasets/total_2task_flexiv_eef_rot6d_noforce
bye
```

### 远端验证
```bash
ls -d /data/group1/junjie008/datasets/total_2task_flexiv_eef_rot6d(_noforce)
ls /data/group1/junjie008/datasets/total_2task_flexiv_eef_rot6d/norm_stats.json
ls /data/group1/junjie008/datasets/total_2task_flexiv_eef_rot6d/data/chunk-000 | wc -l  # 100
```

---

## 6. 训练编排 (job_pi05_chain_eef_force.sbatch)

### SBATCH 头 (保持不变)
```
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1
#SBATCH --cpus-per-task=16 --mem=192G --time=72:00:00
```
- 申请方式与之前 3cfg/plain_pi05 完全一致 (单卡)

### 阶段 (单卡编排)
1. **阶段 1 并行** (单卡 45%×2, bs=8):
   - `pi05_plain_total_task_joint_remote` (exp=plain_joint_chain)
   - `pi05_plain_total_task_eef_remote` (exp=plain_eef_chain)
2. **阶段 2 串行** (独占, bs=8):
   - `pi05_force_total_task_eef_only_remote` (exp=force_eef_chain)

### 关键环境变量
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` 必须在所有阶段前 (防 spawn worker OOM, job 405 教训)
- 并行阶段 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`, 串行阶段 unset

### 提交
```bash
sbatch /data/group1/junjie008/FA-openpi/job_pi05_chain_eef_force.sbatch
```

---

## 7. 监控与检查

### 日志路径
- `/data/group1/junjie008/FA-openpi/logs/total_2task_pi05_chain_plain_joint.log`
- `/data/group1/junjie008/FA-openpi/logs/total_2task_pi05_chain_plain_eef.log`
- `/data/group1/junjie008/FA-openpi/logs/total_2task_pi05_chain_force_eef.log`

### 检查命令
```bash
squeue -u junjie008
tail /data/group1/junjie008/FA-openpi/logs/total_2task_pi05_chain_plain_eef.log
grep -n "Initialized data loader" logs/total_2task_pi05_chain_plain_eef.log
grep -n "\[1\]:" logs/total_2task_pi05_chain_plain_eef.log   # (4, 30, 32) 结构
```

### 本地拉日志
```bash
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg
get -r /data/group1/junjie008/FA-openpi/logs/total_2task_pi05_chain_*.log /mnt/hdd/sfy/FA-openpi/remote_logs/logs/
```

---

## 8. Loss 分析结论 (2026-08-17, 已确认)

### 现象: EEF loss ≈ 3x joint loss (0.015 vs 0.005 @ 4.4k)
### 归因: 不是维度/归一化导致, 是学习难度
- 冷启动下界: joint 1.078 vs EEF 1.120 (32维均值), 只差 4%
- 换归一化 (quantile→min/max) 只降 2%, 非主因
- **主因: rot6d delta 的恒 1 维 (dim3/dim7≈1.0) + gripper 维**, 模型要学"几乎恒定输出", 收敛慢
- UMI/GR00T 同样不优化, 让模型自己学 → 我们也是标准做法

### 结论: 不需要现在优化, 继续训练
- EEF 单调下降 (0.155→0.0157 @ 4.4k), 无平台
- 40k 步外推 EEF loss ≈ 0.005-0.008, 可用
- 40k 后评估: EEF loss <0.01 → 可用; 仍 >0.01 → 再考虑 gripper 独立 + rot6d 固定缩放

---

## 9. 已知坑 (重要)

1. **restricted shell 多行粘连**: 一行一条命令, 逐条粘贴
2. **git pull 常失败**: 用 `git fetch + git reset --hard origin/main`
3. **sftp 不支持 mkdir -p**: 先 SSH 建目录
4. **PYTHONPATH 必须指向 FA-openpi/src**: 否则 import 到 /mnt/hdd/sfy/openpi
5. **XLA_PYTHON_CLIENT_PREALLOCATE=false 必须在所有阶段前**
6. **sftp put -r 需先 cd 到远端父目录** (否则嵌套同名目录)
7. **conda activate 在 restricted shell 不生效**: 用显式路径 `/home/sfy/miniconda3/envs/rlinf/bin/python`

---

## 10. 本地环境速查

| 项 | 值 |
|---|---|
| 本地 conda env | rlinf (/home/sfy/miniconda3/envs/rlinf, jax 0.5.3) |
| 本地数据集 | /mnt/hdd/sfy/datasets/total_2task_flexiv_eef_rot6d(_noforce) |
| 远端主机 | junjie008@mae-cae1-p4103.dynip.ntu.edu.sg (跳板 sfy@10.96.45.13) |
| 远端数据集 | /data/group1/junjie008/datasets/ |
| 远端代码 | /data/group1/junjie008/FA-openpi/ |
| 远端日志 | /data/group1/junjie008/FA-openpi/logs/ |
| 远端 conda | /data/group1/junjie008/miniconda3/bin/activate openpi |
| 远端 GPU | 2x RTX PRO 6000 Blackwell 96G (driver 580, CUDA 13) |
