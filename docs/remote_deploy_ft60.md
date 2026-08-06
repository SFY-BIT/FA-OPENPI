# FA-openpi ft60 部署指令（sftp 上传，本地 LoRA / 远端全参）

> 远端服务器：`junjie008@mae-cae1-p4103.dynip.ntu.edu.sg`（via jump `sfy@10.96.45.13`）
> 远端 FA-openpi 代码路径：`/data/group1/junjie008/FA-openpi`（新建，与主干 openpi-force 分开，不干涉）
> 远端数据集路径：`/data/group1/junjie008/datasets/stamp_seal_v2_flexiv_ft60`
> 分工：**本地只跑 LoRA K=16**；**远端只跑全参 K=16（pi05_base 冷启动）**
> 最后更新：2026-08-01

---

## 1. 配置清单

| 配置名 | 训练 | 初始化 | repo_id | batch/步数 | 跑在哪 |
|--------|------|--------|---------|-----------|--------|
| `pi05_force_stamp_seal_ft60_forcevla_lora_k16` | LoRA K=16 | openpi-force 12000 热启动 | 本地 ft60 | 8 / 30k | **本地** |
| `pi05_force_stamp_seal_ft60_k16_remote` | 全参 K=16（全解冻）| pi05_base 冷启动 | 远端 ft60 | 32 / 50k | **远端** |

- 全参 LR：base=5e-5（PaliGemma/action/time_mlp/**RouterWeights** 1×）；limoe 专家/force/state_proj 5×
- LoRA LR：新模块（ft_encoder/ft_proj）10×；VLM 冻结仅 LoRA；视觉全参；LIMoE/force_out_proj 可训练
- K=16 分段编码：60 帧历史切 16 段（每段 4 帧），共享 FTEncoder → 16 个语义独立 token

---

## 2. 本地：sftp 上传整个 FA-openpi（226M，已清理 smoke checkpoint）

```bash
# 连接（sftp 通过 jump）
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg
```

```text
# 在 sftp 交互中：
# 1) 先 cd 到远端父目录（put -r 会在当前目录下创建同名目录，避免嵌套）
cd /data/group1/junjie008

# 2) 上传整个 FA-openpi（put -r 递归创建 → /data/group1/junjie008/FA-openpi）
put -r /mnt/hdd/sfy/FA-openpi

# 3) 验证
ls -la /data/group1/junjie008/FA-openpi
```

> ⚠️ FA-openpi 已清理：本地 smoke checkpoint（9.4G）已删除，整个目录仅 226M（代码+脚本+assets）。
> ⚠️ 远端 `/data/group1/junjie008/openpi-force`（主干）**不要动**——FA-openpi 独立目录，互不干涉。

---

## 3. 本地：sftp 上传 ft60 数据集（1.2G）

```bash
# 仍在同一 sftp 会话中：
```

```text
# 1) cd 到远端父目录（同样避免嵌套）
cd /data/group1/junjie008/datasets

# 2) 上传数据集（含 parquet + norm_stats.json）
#    → /data/group1/junjie008/datasets/stamp_seal_v2_flexiv_ft60
put -r /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60

# 3) 验证（norm_stats.json 必须在数据集目录里）
ls -la /data/group1/junjie008/datasets/stamp_seal_v2_flexiv_ft60/
# 应看到 norm_stats.json、parquet 文件等
```

> norm_stats 已包含在 ft60 数据集内（`norm_stats.json`），随数据集上传即生效。
> 远端 config `pi05_force_stamp_seal_ft60_k16_remote` 的 repo_id 指向 `/data/group1/junjie008/datasets/stamp_seal_v2_flexiv_ft60`。

---

## 4. 远端：提交全参训练（SLURM）

```bash
ssh -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg
cd /data/group1/junjie008/FA-openpi

# 0) 先跑就绪检查（验证 put 是否完整、无嵌套、数据集/norm_stats/conda/config 都 OK）
bash scripts/remote_ready_check.sh
#    —— 全部 OK（无 FAIL）再往下；脚本会自动创建 logs/ checkpoints/

# 1) 提交全参 K=16（判决性实验，pi05_base 冷启动）
sbatch job_full_ft60.sbatch
```

> ⚠️ **logs 目录必须先存在**：`job_full_ft60.sbatch` 的 `--output` 指向
> `/data/group1/junjie008/FA-openpi/logs/`，SLURM 不会自动创建该目录。
> `remote_ready_check.sh` 已包含 `mkdir -p logs`，跑过它即可。

**查看任务状态**：
```bash
squeue -u junjie008
tail -f /data/group1/junjie008/FA-openpi/logs/ft60_full_k16_<jobid>.out
```

> `job_full_ft60.sbatch` 已含 WANDB key（从主干 sbatch 复制）。
> 远端全参**不需要**上传 12000 checkpoint（pi05_base 冷启动，`ft_encoder`/`ft_proj` 随机初始化）。

---

## 5. 本地：跑 LoRA K=16

```bash
cd /mnt/hdd/sfy/FA-openpi
# 建议在 tmux 中运行（避免断连被杀）
bash scripts/train_lora_local.sh
```

---

## 6. 训练完：下载 checkpoint 回本地分析

```bash
# sftp 下载最终 checkpoint
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg
```

```text
cd /data/group1/junjie008/FA-openpi/checkpoints/pi05_force_stamp_seal_ft60_k16_remote
get -r ft60_full_k16 /mnt/hdd/sfy/FA-openpi/checkpoints/pi05_force_stamp_seal_ft60_k16_remote/
```

**本地 MoE 路由分析**（K=16 时 force token=16）：
```bash
cd /mnt/hdd/sfy/FA-openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src python -u scripts/analyze_moe_ft60.py \
    --checkpoint checkpoints/pi05_force_stamp_seal_ft60_k16_remote/ft60_full_k16/50000 \
    --k-force 16 --output-dir outputs/moe_ft60_k16
```

---

## 7. 故障排查

| 症状 | 解决 |
|------|------|
| `ValueError: Normalization stats not found` | 确认 norm_stats.json 已随 ft60 数据集上传到远端 repo_id 路径 |
| `FileNotFoundError: ...stamp_seal_v2_flexiv_ft60` | 确认第 3 节 put -r 完成 |
| `shape mismatch in action_out_proj` | 预期行为，`Pi0ForceWeightLoader._merge_params` 自动跳过 |
| `shape mismatch in ft_encoder / ft_proj` | 预期行为（K=16 时 input_dim=24），weight_loader regex 排除 → 随机初始化 |
| `RESOURCE_EXHAUSTED` (OOM) | 减小 batch_size（全参 32→16） |
