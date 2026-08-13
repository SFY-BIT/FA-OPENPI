#!/usr/bin/env python3
"""数据集转换脚本: 6dof 关节 + gripper → EEF 6D 坐标 + gripper

把 LeRobot 数据集的 `observation.state` 和 `action` 前 7 维 (6 关节 + 1 gripper)
通过 FK 转换为 EEF 末端 6D 位姿 (xyz + rpy) + gripper, 替换回原位置;
其余内容 (图像/力/时间戳等) 完全不变。

用途: 生成 EEF 空间数据集 (config 1: eef_only 直接用 EEF 坐标算 loss)

用法:
  python convert_dataset_to_eef.py \
      --input /mnt/hdd/sfy/datasets/total_task_flexiv \
      --output /mnt/hdd/sfy/datasets/total_task_flexiv_eef \
      --tool-extension 0.211

注意:
  - state 前 6 维: 关节 → EEF xyz+rpy (FK)
  - action 前 6 维: 绝对关节 → 绝对 EEF 位姿 (FK)
  - gripper (index 6) 不变
  - 力/力矩维度 (state index 7-12) 不变
  - 生成新数据集, 不修改原数据集
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from openpi.models import piper_fk_jax as jfk

# 前 6 维 = 关节角, index 6 = gripper
JOINT_DIMS = slice(0, 6)
GRIPPER_DIM = 6


# JIT 编译的单帧 FK（避免 fk_batch 的 batch 依赖 + 避免非 jit 的 Python 开销）
import jax
import jax.numpy as jnp

_fk_jit = jax.jit(jfk.fk)


def _T_to_eef6(T: np.ndarray) -> np.ndarray:
    """4x4 齐次变换 → [xyz, rpy] (ZYX 欧拉角, 与 pose_from_joints 一致)。"""
    xyz = T[:3, 3]
    R = T[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], sy)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.concatenate([xyz, [roll, pitch, yaw]]).astype(np.float32)


def joints_to_eef6_batch(joints_all: np.ndarray, tool_extension: float) -> np.ndarray:
    """批量: 关节角 [N, 6] → EEF 位姿 [N, 6] (xyz + rpy)。

    逐帧调用 JIT 编译的单帧 fk()：结果与单帧基准完全一致（fk_batch 的
    结果依赖 batch 大小, XLA 浮点重排, 0.6mm 级差异），且 JIT 后
    ~0.2ms/帧, 5 万帧约 10 秒。
    """
    N = joints_all.shape[0]
    eef = np.zeros((N, 6), dtype=np.float32)
    for i in range(N):
        T = np.asarray(_fk_jit(jnp.asarray(joints_all[i], dtype=jnp.float32), tool_extension))
        eef[i] = _T_to_eef6(T)
    return eef


def convert_episode(ep_file: Path, out_file: Path, tool_extension: float) -> None:
    """转换单个 episode parquet: state 和 action 前 6 维都 FK 成 EEF。"""
    t = pq.read_table(ep_file)
    df = t.to_pandas()

    # ── state: 前 7 维 [6 关节 + gripper] → [6 EEF + gripper] ──
    states = np.stack(df["observation.state"].to_numpy())  # [N, 13]
    new_states = states.astype(np.float32).copy()
    eef_states = joints_to_eef6_batch(states[:, JOINT_DIMS], tool_extension)  # [N, 6]
    new_states[:, :6] = eef_states
    # gripper (index 6) 不变
    df["observation.state"] = list(new_states)

    # ── action: 前 7 维 [6 关节 + gripper] → [6 EEF + gripper] ──
    actions = np.stack(df["action"].to_numpy())  # [N, 7]
    new_actions = actions.astype(np.float32).copy()
    eef_actions = joints_to_eef6_batch(actions[:, JOINT_DIMS], tool_extension)  # [N, 6]
    new_actions[:, :6] = eef_actions
    # gripper 不变
    df["action"] = list(new_actions)

    # 写回 (保持 schema)
    new_t = pa.Table.from_pandas(df, schema=t.schema, preserve_index=False)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_t, out_file)


def main():
    p = argparse.ArgumentParser(description="Convert joint-state/action dataset to EEF dataset")
    p.add_argument("--input", required=True, help="输入数据集目录 (LeRobot)")
    p.add_argument("--output", required=True, help="输出数据集目录 (新数据集)")
    p.add_argument("--tool-extension", type=float, default=0.211, help="工具延伸长度 (m)")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    # 复制 meta/非数据文件
    for f in in_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
            print(f"copied {f.name}")
        elif f.name == "data":
            continue
        else:
            shutil.copytree(f, out_dir / f.name, dirs_exist_ok=True)
            print(f"copied dir {f.name}")

    # 转换 data/chunk-*/episode_*.parquet
    for ep_file in sorted(in_dir.glob("data/chunk-*/episode_*.parquet")):
        rel = ep_file.relative_to(in_dir)
        out_file = out_dir / rel
        print(f"converting {rel} ...", end=" ", flush=True)
        try:
            convert_episode(ep_file, out_file, args.tool_extension)
            print("done")
        except Exception as e:
            print(f"SKIP (传输不完整或损坏): {e}")

    # 若存在 norm_stats, 提示需要重新计算
    if (in_dir / "norm_stats.json").exists():
        print("\n⚠️ norm_stats.json 已复制, 但 state/action 维度含义已变 (EEF 位姿).")
        print("   训练前需用 compute_norm_stats 重新计算!")

    print(f"\n完成: {out_dir}")


if __name__ == "__main__":
    main()
