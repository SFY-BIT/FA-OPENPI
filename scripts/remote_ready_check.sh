#!/bin/bash
# ============================================================
# FA-openpi 远端就绪检查 — put 上传后，ssh 到远端跑一遍：
#   bash scripts/remote_ready_check.sh
# 全部 OK（无 FAIL）即可直接: sbatch job_full_ft60.sbatch
# ============================================================
set -uo pipefail
FAIL=0

echo "========== [1/6] FA-openpi 路径 =========="
if [ -d /data/group1/junjie008/FA-openpi/src ]; then
  echo "  OK: /data/group1/junjie008/FA-openpi/src 存在"
else
  echo "  FAIL: src 不存在 — 检查是否嵌套 (FA-openpi/FA-openpi)"
  ls -d /data/group1/junjie008/FA-openpi/*/ 2>/dev/null | head
  FAIL=1
fi
if [ -d /data/group1/junjie008/FA-openpi/FA-openpi ]; then
  echo "  WARN: 发现嵌套目录 FA-openpi/FA-openpi — 需用 mv 修正"
fi

echo "========== [2/6] 数据集 + norm_stats =========="
DS=/data/group1/junjie008/datasets/stamp_seal_v2_flexiv_ft60
if [ -d "$DS" ]; then
  echo "  OK: 数据集目录存在"
  ls "$DS" | head -8
else
  echo "  FAIL: 数据集不存在 — 需 put -r 上传"
  FAIL=1
fi
if [ -f "$DS/norm_stats.json" ]; then
  echo "  OK: norm_stats.json 存在"
else
  echo "  FAIL: norm_stats.json 缺失（在数据集目录内）"
  FAIL=1
fi

echo "========== [3/6] conda 环境 =========="
if source /data/group1/junjie008/miniconda3/bin/activate openpi 2>/dev/null; then
  echo "  OK: openpi conda 环境激活 ($(which python))"
else
  echo "  FAIL: openpi 环境不可用"
  FAIL=1
fi

echo "========== [4/6] config 加载 =========="
cd /data/group1/junjie008/FA-openpi
export PYTHONPATH=src:${PYTHONPATH:-}
python -c "
from openpi.training import config as c
x = c.get_config('pi05_force_stamp_seal_ft60_k16_remote')
print('  OK: config =', x.name, '| K =', x.model.ft_num_tokens, '| repo =', x.data.repo_id)
" 2>&1 | tail -2

echo "========== [5/6] 提交目录准备 =========="
mkdir -p /data/group1/junjie008/FA-openpi/logs /data/group1/junjie008/FA-openpi/checkpoints
echo "  OK: logs/ checkpoints/ 已创建（sbatch --output 需要 logs 先存在）"

echo "========== [6/6] sbatch 文件 =========="
if [ -f /data/group1/junjie008/FA-openpi/job_full_ft60.sbatch ]; then
  echo "  OK: job_full_ft60.sbatch 存在"
  grep -q "pi05_force_stamp_seal_ft60_k16_remote" /data/group1/junjie008/FA-openpi/job_full_ft60.sbatch \
    && echo "  OK: 引用 config 正确" || echo "  WARN: config 名核对"
else
  echo "  FAIL: job_full_ft60.sbatch 缺失"
  FAIL=1
fi

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "===== ✅ 全部检查通过：直接执行  sbatch job_full_ft60.sbatch  ====="
else
  echo "===== ❌ 存在 FAIL 项，修复后再提交 ====="
fi
