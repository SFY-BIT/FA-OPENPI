# Deploy openpi-force to A100

## 当前配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| force_history_frames | 5 | 过去 5 帧力输入 |
| force_loss_weight | 0.1 | 力 loss 权重 = 关节的 1/10 |
| predict_force | True | 13-dim 输出 (7 action + 6 force) |
| use_force_data | True | State = 37 dim (7 + 5×3 + 5×3) |
| num_experts | 4 | LIMoE 专家数 |
| batch_size | 128 | 1 GPU |

## 改动文件清单 (vs OPENPI.6dof)

新增:
  src/openpi/models/flaxformer_minimal/   (5 文件)
  src/openpi/models/limoe.py
  src/openpi/models/pi0_force.py
  src/openpi/policies/force_piper_policy.py

修改:
  src/openpi/models/pi0_config.py          (+force/moe 字段, +freeze_filter)
  src/openpi/training/config.py            (+imports, +use_force_data, +predict_force, +2 configs)
  src/openpi/training/weight_loaders.py    (+Pi0ForceWeightLoader)

脚本:
  job_openpi_force_usb_insert.sbatch

## 一步部署

```bash
# === 远端：克隆基础 ===
ssh -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg << 'EOF'
cp -r /data/group1/junjie008/OPENPI.6dof /data/group1/junjie008/openpi-force
mkdir -p /data/group1/junjie008/openpi-force/logs
mkdir -p /data/group1/junjie008/openpi-force/checkpoints
echo "Cloned."
EOF

# === 本地打包差异文件 ===
cd /mnt/hdd/sfy/openpi-force
tar czf /tmp/openpi-force-diff.tar.gz \
    src/openpi/models/flaxformer_minimal/ \
    src/openpi/models/limoe.py \
    src/openpi/models/pi0_force.py \
    src/openpi/policies/force_piper_policy.py \
    src/openpi/models/pi0_config.py \
    src/openpi/training/config.py \
    src/openpi/training/weight_loaders.py \
    job_openpi_force_usb_insert.sbatch

# === 上传 ===
scp -o ProxyJump=sfy@10.96.45.13 /tmp/openpi-force-diff.tar.gz \
    junjie008@mae-cae1-p4103.dynip.ntu.edu.sg:/data/group1/junjie008/

# === 远端解包 + 提交 ===
ssh -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg << 'EOF'
cd /data/group1/junjie008/openpi-force
tar xzf ../openpi-force-diff.tar.gz
rm ../openpi-force-diff.tar.gz
echo "Extracted."

# 提交训练 (norm_stats 由 sbatch step 1 自动计算)
sbatch job_openpi_force_usb_insert.sbatch
squeue -u junjie008
EOF
```

## 单独算 norm_stats (如需要)

```bash
ssh -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg << 'EOF'
source /data/group1/junjie008/miniconda3/bin/activate openpi
cd /data/group1/junjie008/openpi-force
python scripts/compute_norm_stats.py pi05_force_usb_insert
EOF
```

## 日志与 Checkpoint

```bash
# 拉日志
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg << 'EOSFTP'
get /data/group1/junjie008/openpi-force/logs/force_usb_<JOBID>.err /mnt/hdd/sfy/outputs/
EOSFTP

# 拉 checkpoint
sftp -J sfy@10.96.45.13 junjie008@mae-cae1-p4103.dynip.ntu.edu.sg << 'EOSFTP'
get -r /data/group1/junjie008/openpi-force/checkpoints/pi05_force_usb_insert/usb_insert_force_1gpu_v1/<STEP> /mnt/hdd/sfy/outputs/
EOSFTP
```
