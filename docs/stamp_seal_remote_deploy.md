# stamp_seal 远端训练部署指令

> 远端服务器：`junjie008@mae-cae1-p4103.dynip.ntu.edu.sg`（via jump `sfy@10.96.45.13`）
> 远端代码路径：`/data/group1/junjie008/openpi-force`
> 远端数据集路径：`/data/group1/junjie008/datasets/stamp_seal_v2_flexiv`
> 最后更新：2026-07-24

---

## 前置条件

- [ ] 本地代码改动已完成（config.py, pi0_force.py, force_piper_policy.py, data_loader.py, pi0_config.py, model.py, weight_loaders.py）
- [ ] 本地已计算 norm_stats（见下方步骤 1）
- [ ] 数据集 `stamp_seal_v2_flexiv` 已准备好（127M）

---

## 步骤 1：本地计算 norm_stats

```bash
cd /mnt/hdd/sfy/openpi-force
conda activate rlinf
HF_DATASETS_CACHE=/mnt/hdd/sfy/openpi/.cache/datasets \
PYTHONPATH=src python scripts/compute_norm_stats.py \
    --config-name pi05_force_stamp_seal
```

**输出**：`assets/pi05_force_stamp_seal/<repo_id>/norm_stats.json`
- `<repo_id>` = `/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv`（本地路径）
- 包含 `state`(19维)、`actions`(7维)、`force_target`(6维 delta) 三个 key 的 quantile 统计量

**验证**：
```bash
ls -la /mnt/hdd/sfy/openpi-force/assets/pi05_force_stamp_seal/
# 应看到 /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv/norm_stats.json
```

---

## 步骤 2：SFTP 上传到远端

### 2.1 连接远端

```bash
# 通过 jump server 连接
ssh -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg
# 或 SFTP
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg
```

### 2.2 上传代码

```bash
# 在 SFTP 中（远端根目录 /data/group1/junjie008/openpi-force）
cd /data/group1/junjie008/openpi-force

# 上传修改过的源码文件
put /mnt/hdd/sfy/openpi-force/src/openpi/training/config.py        src/openpi/training/config.py
put /mnt/hdd/sfy/openpi-force/src/openpi/models/pi0_force.py       src/openpi/models/pi0_force.py
put /mnt/hdd/sfy/openpi-force/src/openpi/models/pi0_config.py      src/openpi/models/pi0_config.py
put /mnt/hdd/sfy/openpi-force/src/openpi/models/model.py           src/openpi/models/model.py
put /mnt/hdd/sfy/openpi-force/src/openpi/policies/force_piper_policy.py   src/openpi/policies/force_piper_policy.py
put /mnt/hdd/sfy/openpi-force/src/openpi/training/data_loader.py   src/openpi/training/data_loader.py
put /mnt/hdd/sfy/openpi-force/src/openpi/training/weight_loaders.py src/openpi/training/weight_loaders.py
put /mnt/hdd/sfy/openpi-force/scripts/train.py                     scripts/train.py
put /mnt/hdd/sfy/openpi-force/scripts/compute_norm_stats.py        scripts/compute_norm_stats.py

# 上传训练脚本
put /mnt/hdd/sfy/openpi-force/job_stamp_seal_force.sbatch          job_stamp_seal_force.sbatch
```

### 2.3 上传数据集

```bash
# 数据集（127M，不大）
mkdir -p /data/group1/junjie008/datasets/stamp_seal_v2_flexiv
put -r /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv/*   /data/group1/junjie008/datasets/stamp_seal_v2_flexiv/
```

### 2.4 上传 norm_stats

**注意**：远端 config 的 `repo_id` 是远端路径，所以 norm_stats 目录名必须是远端路径。

```bash
# 方式 A：上传本地算好的（需重命名目录为远端 repo_id）
mkdir -p /data/group1/junjie008/openpi-force/assets/pi05_force_stamp_seal_remote/data/group1/junjie008/datasets/stamp_seal_v2_flexiv
put /mnt/hdd/sfy/openpi-force/assets/pi05_force_stamp_seal/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv/norm_stats.json \
    /data/group1/junjie008/openpi-force/assets/pi05_force_stamp_seal_remote/data/group1/junjie008/datasets/stamp_seal_v2_flexiv/norm_stats.json

# 方式 B（推荐）：远端重新计算（见步骤 3）
```

---

## 步骤 3：远端计算 norm_stats（推荐，避免路径问题）

```bash
# SSH 到远端
ssh -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg

cd /data/group1/junjie008/openpi-force
source /data/group1/junjie008/miniconda3/bin/activate openpi
export PYTHONPATH=src:${PYTHONPATH:-}

python scripts/compute_norm_stats.py \
    --config-name pi05_force_stamp_seal_remote
```

**输出**：`assets/pi05_force_stamp_seal_remote/<远端repo_id>/norm_stats.json`
- `<远端repo_id>` = `/data/group1/junjie008/datasets/stamp_seal_v2_flexiv`

---

## 步骤 4：提交 SLURM 训练任务

```bash
# SSH 到远端
cd /data/group1/junjie008/openpi-force
sbatch job_stamp_seal_force.sbatch
```

**任务参数**：
- job-name: `stamp_seal`
- partition: `debug`
- GPU: 1 × A100
- 内存: 192G
- 时间: 72h
- config: `pi05_force_stamp_seal_remote`
- exp-name: `stamp_seal_force_v1`
- batch_size: 64, 30K steps, save_interval=2000

**查看任务状态**：
```bash
squeue -u junjie008
# 查看日志
tail -f /data/group1/junjie008/openpi-force/logs/stamp_seal_force_<jobid>.out
```

---

## 步骤 5：下载 checkpoint

训练完成后（30K steps，约每 2000 步存一个 checkpoint）：

```bash
# SFTP 下载
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg

# 下载最终 checkpoint
get -r /data/group1/junjie008/openpi-force/checkpoints/pi05_force_stamp_seal_remote/stamp_seal_force_v1/30000 \
      /mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_stamp_seal_remote/

# 下载 norm_stats（推理需要）
get -r /data/group1/junjie008/openpi-force/assets/pi05_force_stamp_seal_remote/ \
      /mnt/hdd/sfy/openpi-force/assets/
```

---

## 步骤 6：本地推理

```bash
cd /mnt/hdd/sfy/openpi-force
conda activate rlinf
HF_DATASETS_CACHE=/mnt/hdd/sfy/openpi/.cache/datasets \
PYTHONPATH=src python scripts/serve_policy.py \
    --policy.config=pi05_force_stamp_seal \
    --policy.dir=checkpoints/pi05_force_stamp_seal_remote/stamp_seal_force_v1/30000
```

**输出**：
- `actions`：`[H=30, 7]`（6关节 delta + 1夹爪绝对值，经 AbsoluteActions 还原为绝对值）
- `force_pred`：`[H=30, 6]`（delta 还原后的绝对值力/力矩）

---

## 配置对比（local vs remote）

| 配置项 | `pi05_force_stamp_seal`（local） | `pi05_force_stamp_seal_remote`（remote） |
|--------|----------------------------------|----------------------------------------|
| `repo_id` | `/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv` | `/data/group1/junjie008/datasets/stamp_seal_v2_flexiv` |
| 其余所有参数 | 完全相同 | 完全相同 |

**相同参数**：
- model: Pi0ForceConfig(pi05, action_horizon=30, control_action_dim=7, force_history_frames=2, grad_route_mode="three_stage", force_loss_weight=0.1)
- data: LeRobotPiperDataConfig(use_delta_joint_actions=True, use_delta_gripper_actions=False, force_in_state=True, predict_force=True, action_dim=7)
- batch_size=64, lr=5e-5, 30K steps, new_module_lr_multiplier=5.0
- weight_loader: Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")

---

## 故障排查

### norm_stats not found
```
ValueError: Normalization stats not found.
```
**解决**：远端重新跑 `compute_norm_stats.py --config-name pi05_force_stamp_seal_remote`，确保 `assets/pi05_force_stamp_seal_remote/<远端repo_id>/norm_stats.json` 存在。

### 数据集路径错误
```
FileNotFoundError: /data/group1/junjie008/datasets/stamp_seal_v2_flexiv
```
**解决**：确认数据集已上传到远端正确路径。

### checkpoint shape mismatch
```
shape mismatch in action_out_proj
```
**解决**：这是预期的（action_out_proj 从 32 维重建为 7 维），`Pi0ForceWeightLoader._merge_params` 会自动跳过 shape 不一致的权重。

### OOM
```
RESOURCE_EXHAUSTED
```
**解决**：减小 batch_size（64 → 32 → 16），或增加 `XLA_PYTHON_CLIENT_MEM_FRACTION`。
