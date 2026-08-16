#!/usr/bin/env python3
"""生成纯 pi05 用的 7 维 state 数据集（去掉 force 列，不动原数据集）。

把 13 维 state 数据集（joint+gripper+force）转成 7 维（joint+gripper），
供纯 pi05 模型（无 force 头）使用。action 保持 7 维不变。

用法:
  python scripts/strip_force_from_dataset.py \
      --input /mnt/hdd/sfy/datasets/total_2task_flexiv_ft60 \
      --output /mnt/hdd/sfy/datasets/total_2task_flexiv_ft60_noforce

注意:
  - 只改 observation.state (13 → 7 维, 截断前 7), 其他列 (action/图像/wrench_history) 不变
  - 生成新数据集, 不修改原数据集
  - norm_stats 需重新计算 (state 维度变化)
"""
import argparse
import shutil
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def convert_episode(ep_file: Path, out_file: Path) -> None:
    t = pq.read_table(ep_file)
    df = t.to_pandas()

    # state: 13 → 7 维 (截断 force)
    states = np.stack(df["observation.state"].to_numpy())
    new_states = states[:, :7].astype(np.float32)
    df["observation.state"] = list(new_states)

    # 其余列不变 (action 已 7 维, wrench_history/图像保留)

    new_t = pa.Table.from_pandas(df, schema=t.schema, preserve_index=False)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_t, out_file)


def main():
    p = argparse.ArgumentParser(description="Strip force dims from dataset state (13→7)")
    p.add_argument("--input", required=True, help="输入数据集目录")
    p.add_argument("--output", required=True, help="输出数据集目录 (新)")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 复制 meta/非数据文件
    for f in in_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
        elif f.name == "data":
            continue
        else:
            shutil.copytree(f, out_dir / f.name, dirs_exist_ok=True)

    # 修正 info.json: state 13 → 7
    info_path = out_dir / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        feats = info.get("features", {})
        if "observation.state" in feats:
            feats["observation.state"]["shape"] = [7]
        info_path.write_text(json.dumps(info, indent=2))
        print(f"info.json: state shape → {feats.get('observation.state', {}).get('shape')}")

    # 转换 parquet
    for ep_file in sorted(in_dir.glob("data/chunk-*/episode_*.parquet")):
        rel = ep_file.relative_to(in_dir)
        out_file = out_dir / rel
        print(f"converting {rel} ...", end=" ", flush=True)
        try:
            convert_episode(ep_file, out_file)
            print("done")
        except Exception as e:
            print(f"SKIP: {e}")

    # norm_stats 提示
    if (in_dir / "norm_stats.json").exists():
        print("\n⚠️ norm_stats.json 已复制但 state 维度变了, 需重新计算!")

    print(f"\n完成: {out_dir}")


if __name__ == "__main__":
    main()
