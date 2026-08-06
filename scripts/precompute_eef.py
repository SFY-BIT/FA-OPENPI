#!/usr/bin/env python3
"""为擦板数据集生成 EEF 位姿 + EEF delta 字段（新增列，其他字段原样复制）。

新增两个字段到每个 parquet:
    observation.eef:        [6] = FK(state 6 关节, sensor末端 0.211m) 的 [xyz(3), rpy(3)]
    observation.eef_delta:  [6] = 相邻帧 eef 的差（角度回绕到 [-pi, pi]）

计算说明:
  - FK 末端 = link6 沿 z 轴延伸 0.211m (夹爪 0.13503 + 力传感器 0.076)
  - eef_delta[t] = eef[t+1] - eef[t], 其中 rpy 分量用 wrap 处理 ±180° 回绕
  - 其余所有字段(含 action/force/image/wrench_history) 原样保留

用法:
    python scripts/precompute_eef.py \
        --src /mnt/hdd/sfy/datasets/erase_board_flexiv_ft60 \
        --dst /mnt/hdd/sfy/datasets/erase_board_flexiv_ft60_eef
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# 引入 FA-openpi 的 FK（sensor 末端 0.211m）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from piper_fk import tool_pose_from_state, TOOL_EXTENSION


def wrap_angle(d):
    """角度回绕到 [-pi, pi]。"""
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def compute_eef(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """输入 state[N,13] 和 action[N,7], 返回 (eef_abs [N,6], eef_delta [N,6])。

    eef_abs = FK(state 关节) = [x, y, z, roll, pitch, yaw]（sensor 末端 0.211m）
    eef_delta = FK(action 关节目标) - FK(state 关节) = 相对当前帧的 EEF 目标增量。

    ⚠️ 对齐 DeltaActions 语义: 关节 delta = action - state (相对当前帧)。
       EEF delta 也必须是: eef_delta[t] = FK(action[t]) - FK(state[t]),
       而不是相邻帧差 (action[t+1]-action[t])。角度分量回绕到 [-pi, pi]。
    """
    n = len(states)
    eef_abs = np.zeros((n, 6), dtype=np.float32)
    eef_target = np.zeros((n, 6), dtype=np.float32)
    for t in range(n):
        xyz, rpy = tool_pose_from_state(states[t])
        eef_abs[t, :3] = xyz
        eef_abs[t, 3:] = rpy
        # action 前 6 维是绝对关节目标 (与 state 同格式)
        xyz_a, rpy_a = tool_pose_from_state(actions[t])
        eef_target[t, :3] = xyz_a
        eef_target[t, 3:] = rpy_a

    # delta = FK(action) - FK(state), 角度回绕
    delta = np.zeros((n, 6), dtype=np.float32)
    delta[:, :3] = eef_target[:, :3] - eef_abs[:, :3]   # 位置增量
    delta[:, 3] = wrap_angle(eef_target[:, 3] - eef_abs[:, 3])   # roll
    delta[:, 4] = wrap_angle(eef_target[:, 4] - eef_abs[:, 4])   # pitch
    delta[:, 5] = wrap_angle(eef_target[:, 5] - eef_abs[:, 5])   # yaw
    return eef_abs, delta


def load_episodes_data(src_dir: Path) -> dict[int, pd.DataFrame]:
    episodes: dict[int, pd.DataFrame] = {}
    parquet_files = sorted(src_dir.glob("data/**/*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found under {src_dir / 'data'}")
        sys.exit(1)
    for pf in tqdm(parquet_files, desc="Loading parquet files"):
        df = pd.read_parquet(pf)
        for ep in df["episode_index"].unique():
            ep = int(ep)
            ep_df = df[df["episode_index"] == ep].sort_values("frame_index")
            if ep in episodes:
                episodes[ep] = pd.concat([episodes[ep], ep_df], ignore_index=True)
            else:
                episodes[ep] = ep_df.reset_index(drop=True)
    return episodes


def write_new_dataset(src_dir: Path, dst_dir: Path, episodes: dict[int, pd.DataFrame]):
    # Copy meta/
    meta_src = src_dir / "meta"
    meta_dst = dst_dir / "meta"
    if meta_dst.exists():
        shutil.rmtree(meta_dst)
    shutil.copytree(meta_src, meta_dst)

    # Update info.json features
    info_path = meta_dst / "info.json"
    info = json.loads(info_path.read_text())
    info.setdefault("features", {})
    info["features"]["observation.eef"] = {
        "dtype": "float32",
        "shape": [6],
        "names": ["eef_x", "eef_y", "eef_z", "eef_roll", "eef_pitch", "eef_yaw"],
    }
    info["features"]["observation.eef_delta"] = {
        "dtype": "float32",
        "shape": [6],
        "names": ["eef_dx", "eef_dy", "eef_dz", "eef_droll", "eef_dpitch", "eef_dyaw"],
    }
    info_path.write_text(json.dumps(info, indent=2))

    # Write parquet files
    parquet_files = sorted(src_dir.glob("data/**/*.parquet"))
    for pf in tqdm(parquet_files, desc="Writing new parquet files"):
        df = pd.read_parquet(pf)
        n = len(df)

        eef_col = np.zeros((n, 6), dtype=np.float32)
        delta_col = np.zeros((n, 6), dtype=np.float32)

        for ep in df["episode_index"].unique():
            ep = int(ep)
            ep_df = episodes[ep]
            ep_mask = df["episode_index"] == ep
            ep_positions = df.loc[ep_mask, "frame_index"].values
            # 每个 frame 在 ep_df 中的行号
            state_rows = np.array([ep_df.index[ep_df["frame_index"] == f][0] for f in ep_positions])
            ep_states = np.stack(ep_df["observation.state"].values)  # [N_ep, 13]
            ep_actions = np.stack(ep_df["action"].values)            # [N_ep, 7]
            eef_abs, delta = compute_eef(ep_states, ep_actions)
            local_idx = np.searchsorted(ep_df["frame_index"].values, ep_positions)
            eef_col[ep_mask] = eef_abs[local_idx]
            delta_col[ep_mask] = delta[local_idx]

        df["observation.eef"] = [v.tolist() for v in eef_col]
        df["observation.eef_delta"] = [v.tolist() for v in delta_col]

        rel_path = pf.relative_to(src_dir)
        out_path = dst_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_path)

    # Symlink videos
    for video_dir_name in ["videos"]:
        video_src = src_dir / video_dir_name
        video_dst = dst_dir / video_dir_name
        if video_src.exists() and not video_dst.exists():
            video_dst.symlink_to(video_src.resolve())

    print(f"Done! New dataset written to {dst_dir}")
    print(f"  tool extension: {TOOL_EXTENSION*100:.1f}cm (gripper 13.5 + sensor 7.6)")


def main():
    parser = argparse.ArgumentParser(description="Precompute EEF + EEF delta for LeRobot dataset")
    parser.add_argument("--src", required=True, help="Source dataset directory")
    parser.add_argument("--dst", required=True, help="Destination dataset directory")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}")
        sys.exit(1)
    if dst_dir.exists():
        print(f"ERROR: Destination already exists: {dst_dir}")
        sys.exit(1)

    print(f"Source:      {src_dir}")
    print(f"Destination: {dst_dir}")
    print()

    episodes = load_episodes_data(src_dir)
    print(f"Loaded {len(episodes)} episodes, {sum(len(v) for v in episodes.values())} total frames")
    write_new_dataset(src_dir, dst_dir, episodes)


if __name__ == "__main__":
    main()
