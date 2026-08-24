# peel 指令（total_task_peel — peel cucumber 单任务）

> 训练线：5 条（joint / joint_only / eef_v2 ×2 / dual），本地 checkpoint 用别名目录
> 已有 3 条完成 (39999)，2 条训练中。
> 统一前置：`cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf`

---

## ① peel_05_joint —— plain joint（✅ 已完成 39999，8-23 06:27）

```bash
cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_ft60_noforce \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_plain_total_task_peel_joint_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_05_joint/39999
```
- joint 模式（7 维），**不加任何 --action-space**
- client 不发 wrench_history

---

## ② peel_FA_joint —— force joint_only（✅ 已完成 39999，8-23 17:44）

```bash
cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_ft60 \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_peel_joint_only_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_FA_joint/39999
```
- joint 模式（7 维），force 模型
- **client 必发 `observation/wrench_history` (60×6)**

---

## ③ peel_05_eef —— plain EEF v2（✅ 已完成 39999，8-24 07:31）

```bash
cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_eef_abs_noforce \
    --port=8000 \
    --action-space=EEF --action-rep=abs \
    policy:checkpoint \
    --policy.config=pi05_plain_total_task_peel_eef_v2_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_05_eef/39999
```
- EEF v2 模式（10 维）：**必须 `--action-rep=abs`**（模型输出已是绝对 EEF，delta 会二次合成爆炸）
- **不加 `--gap-rate`**
- client 不发 wrench_history

---

## ④ peel_FA_eef —— force EEF v2（🕐 训练中，出 ckpt 后改 dir）

```bash
cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_eef_abs \
    --port=8000 \
    --action-space=EEF --action-rep=abs \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_peel_eef_v2_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_FA_eef/39999   # 占位，改成实际步数
```
- force EEF v2（10 维），`--action-rep=abs`，不加 `--gap-rate`
- **client 必发 `observation/wrench_history` (60×6)**

---

## ⑤ peel_FA_daul —— force dual（🕐 训练中，出 ckpt 后改 dir；joint 输出，不转 EEF）

```bash
cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_ft60 \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_peel_eef_joint_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_FA_daul/39999   # 占位，改成实际步数
```
- **joint 模式（7 维）**：dual 本质 joint-only，EFE 只是辅助 loss，serve 不转 EEF
- **client 必发 `observation/wrench_history` (60×6)**
- norm_stats 用 `total_task_peel_ft60`

---

## 备忘

| 目录 | config | serve 模式 | norm_stats | 状态 |
|---|---|---|---|---|
| `peel_05_joint/39999` | `pi05_plain_total_task_peel_joint_remote` | joint | `total_task_peel_ft60_noforce` | ✅ |
| `peel_FA_joint/39999` | `pi05_force_total_task_peel_joint_only_remote` | joint | `total_task_peel_ft60` | ✅ |
| `peel_05_eef/39999` | `pi05_plain_total_task_peel_eef_v2_remote` | **EEF+abs** | `total_task_peel_eef_abs_noforce` | ✅ |
| `peel_FA_eef/?????` | `pi05_force_total_task_peel_eef_v2_remote` | **EEF+abs** | `total_task_peel_eef_abs` | 🕐 |
| `peel_FA_daul/?????` | `pi05_force_total_task_peel_eef_joint_remote` | joint | `total_task_peel_ft60` | 🕐 |
