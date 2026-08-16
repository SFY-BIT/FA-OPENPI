#!/usr/bin/env python3
"""纯 pi05 norm_stats 计算: 只算 state + actions (7 维), 无 force/ft_state。

纯 pi05 模型无 force 头, 归一化只需要 state 和 actions 两个 key。
直接读 parquet 计算 quantile (与 openpi normalize 格式一致), 无需走
完整 transform 管道 (避免 force 相关的 shape mismatch)。

用法:
  python scripts/compute_norm_stats_plain.py \
      <repo_dir> <state_col> <action_col>
"""
import sys
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# 与 openpi/shared/normalize.py 的 NormStats 格式一致
def _stats(arr: np.ndarray) -> dict:
    arr = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
    return {
        "mean": arr.mean(0).tolist(),
        "std": arr.std(0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_norm_stats_plain.py <repo_dir> [state_col] [action_col]")
        sys.exit(1)
    repo_dir = Path(sys.argv[1])
    state_col = sys.argv[2] if len(sys.argv) > 2 else "observation.state"
    action_col = sys.argv[3] if len(sys.argv) > 3 else "action"

    states, actions = [], []
    for f in sorted(repo_dir.glob("data/chunk-*/episode_*.parquet")):
        t = pq.read_table(f, columns=[state_col, action_col])
        df = t.to_pandas()
        states.append(np.stack(df[state_col].to_numpy()))
        actions.append(np.stack(df[action_col].to_numpy()))
    states = np.concatenate(states).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    print(f"加载: {len(states)} 帧, state={states.shape}, action={actions.shape}")

    norm_stats = {
        "state": _stats(states),
        "actions": _stats(actions),
    }
    out = repo_dir / "norm_stats.json"
    out.write_text(json.dumps({"norm_stats": norm_stats}, indent=2))
    print(f"写入: {out}")
    for k, v in norm_stats.items():
        print(f"  {k}: dim={len(v['mean'])} q01[:3]={np.round(v['q01'][:3],3)}")


if __name__ == "__main__":
    main()
