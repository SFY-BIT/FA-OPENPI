#!/usr/bin/env python3
"""数据集转换脚本: 6dof 关节 + gripper → EEF rot6d 位姿 + gripper

把 LeRobot 数据集的 `observation.state` 和 `action` 前 7 维 (6 关节 + 1 gripper)
通过 FK 转换为 EEF 末端位姿 (xyz + rot6d) + gripper, 替换回原位置;
其余内容 (图像/力/时间戳等) 完全不变。

═══════════════════════════════════════════════════════════════════════════════
表示 (rot6d, 参照 UMI / NVIDIA GR00T / Diffusion Policy):
  state   = [xyz(3), rot6d(6), grip(1)]        : 绝对 EEF 位姿 (无万向锁)
  action  = [d_xyz(3), d_rot6d(6), grip(1)]    : 相对 base 的 delta (矩阵合成)
    d_xyz    = xyz_target - xyz_base                       (位置线性差)
    d_rot6d  = rot6d( R_base^T @ R_target )                (旋转矩阵合成)
  其中 base = 当前帧 (chunk 首帧, 即训练时 delta 化的 base)。
  rot6d 是 R 的前两行展平 (row-major), 连续无奇点, 与 UMI 的
  mat_to_rot6d / rot6d_to_mat 完全一致。

═══════════════════════════════════════════════════════════════════════════════
为什么不是 rpy (v1~v10.3 教训):
  rpy 在万向节锁 (|sin p|~1) 病态: 表示滑动 3+ rad / 滤波引入表示误差 /
  双表示族选择做 delta 需共享基准。rot6d 是连续最小表示, 全球无奇点,
  无需任何 unwrap/滤波/sgn 分段 hack。

用法:
  python convert_dataset_to_eef.py \
      --input /mnt/hdd/sfy/datasets/total_2task_flexiv_ft60 \
      --output /mnt/hdd/sfy/datasets/total_2task_flexiv_eef_rot6d \
      --tool-extension 0.211

注意:
  - state 前 6 维: 关节 → EEF xyz+rot6d (FK)
  - action 前 6 维: 绝对关节 → 绝对 EEF 位姿 (FK), delta 由训练侧矩阵合成
  - gripper (index 6) 不变
  - 力/力矩维度 (state index 7-12, 变为 index 10-15) 不变
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


# ═══════════════════════ rot6d 工具 (与 UMI / GR00T 一致) ═══════════════════
def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 6D 表示 (前两行展平, row-major)。与 UMI mat_to_rot6d 一致。"""
    return R[:2, :].reshape(6).astype(np.float64)


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """6D 表示 → 旋转矩阵 (Gram-Schmidt 正交化)。与 UMI rot6d_to_mat 一致。

    rot6d = R 的前两行; 还原时:
      b1 = normalize(a1); b2 = normalize(a2 - (b1·a2)b1); b3 = b1 × b2
    """
    a1 = np.asarray(d6[0:3], dtype=np.float64)
    a2 = np.asarray(d6[3:6], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def _T_to_eef_abs(T: np.ndarray) -> np.ndarray:
    """4x4 齐次变换 → [xyz(3), rot6d(6)]。绝对 EEF 位姿 (无角度表示, 无万向锁)。"""
    xyz = T[:3, 3]
    d6 = mat_to_rot6d(T[:3, :3])
    return np.concatenate([xyz, d6]).astype(np.float32)


def joints_to_eef_abs_batch(joints_all: np.ndarray, tool_extension: float) -> np.ndarray:
    """批量: 关节角 [N, 6] → EEF 绝对位姿 [N, 9] (xyz + rot6d)。

    全程矩阵, 无 rpy/unwrap/共模滤波——rot6d 连续无奇点, 不存在
    万向节锁表示放大 (v1~v10.3 的 rpy 问题全部消失)。
    """
    N = joints_all.shape[0]
    eef = np.zeros((N, 9), dtype=np.float32)
    for i in range(N):
        T = np.asarray(_fk_jit(jnp.asarray(joints_all[i], dtype=jnp.float32), tool_extension))
        eef[i] = _T_to_eef_abs(T)
    return eef


def make_relative_delta(state_eef: np.ndarray, action_eef: np.ndarray) -> np.ndarray:
    """绝对 EEF state/action → 相对 state 的 delta (矩阵合成, 与 UMI 'rel' 一致)。

    state_eef:   [N, 9] 绝对 EEF 位姿 (观测, base)
    action_eef:  [N, 9] 绝对 EEF 位姿 (目标/动作)
    返回:        [N, 9] 相对 delta
      d_xyz   = xyz_action - xyz_state                         (线性差)
      d_rot6d = rot6d( R_state^T @ R_action )                  (前方旋转)
    """
    N = state_eef.shape[0]
    rel = np.zeros_like(action_eef)
    rel[:, :3] = action_eef[:, :3] - state_eef[:, :3]
    for i in range(N):
        Rs = rot6d_to_mat(state_eef[i, 3:9])
        Ra = rot6d_to_mat(action_eef[i, 3:9])
        rel[i, 3:9] = mat_to_rot6d(Rs.T @ Ra)
    return rel.astype(np.float32)


def convert_episode(ep_file: Path, out_file: Path, tool_extension: float) -> None:
    """转换单个 episode parquet: state 和 action 前 6 维都 FK 成 EEF rot6d。

    state  = [xyz(3), rot6d(6), grip(1)]   绝对位姿 (观测)
    action = [xyz(3), rot6d(6), grip(1)]   相对 state 的 delta (矩阵合成)
    剩余维度 (力/扭等) 原样保留, 整体后移。
    """
    t = pq.read_table(ep_file)
    df = t.to_pandas()

    # ── 读取 joints ──
    states = np.stack(df["observation.state"].to_numpy())   # [N, 13+] (ft60 含力)
    actions = np.stack(df["action"].to_numpy())             # [N, 7]

    # ── state: [6 关节 + grip + 力...] → [xyz+rot6d + grip + 力...] ──
    eef_states = joints_to_eef_abs_batch(states[:, JOINT_DIMS], tool_extension)  # [N, 9]
    new_states = np.concatenate([eef_states, states[:, GRIPPER_DIM:]], axis=1).astype(np.float32)

    # ── action: [6 关节 + grip] → 相对 state 的 delta [xyz+rot6d + grip] ──
    eef_actions = joints_to_eef_abs_batch(actions[:, JOINT_DIMS], tool_extension)   # [N, 9]
    rel_delta = make_relative_delta(eef_states, eef_actions)                        # [N, 9]
    new_actions = np.concatenate([rel_delta, actions[:, GRIPPER_DIM:]], axis=1).astype(np.float32)

    # ── 写回 ──
    df["observation.state"] = list(new_states)
    df["action"] = list(new_actions)
    new_t = pa.Table.from_pandas(df, schema=t.schema, preserve_index=False)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_t, out_file)


def main():
    p = argparse.ArgumentParser(description="Convert joint-state/action dataset to EEF rot6d dataset")
    p.add_argument("--input", required=True, help="输入数据集目录 (LeRobot)")
    p.add_argument("--output", required=True, help="输出数据集目录 (新数据集)")
    p.add_argument("--tool-extension", type=float, default=0.211, help="工具延伸长度 (m)")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

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
