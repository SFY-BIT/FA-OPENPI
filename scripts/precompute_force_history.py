#!/usr/bin/env python3
"""Precompute force/torque history for LeRobot datasets.

Reads an existing LeRobot v2.1 dataset, builds a T-frame force/torque history
for every row, and writes a **new** dataset to a separate output directory.
The original dataset is never modified.

Usage:
    python scripts/precompute_force_history.py \
        --src /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv \
        --dst /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60 \
        --history-steps 60

Output:
    A full copy of the source dataset with one extra column per parquet file:
        observation.wrench_history  float32 [T, 6]

    The column contains the force/torque values of the past T frames (oldest
    first), sliced from observation.state[force_start_idx:force_start_idx+6].
    At episode boundaries missing history is zero-padded.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def load_episodes_data(src_dir: Path) -> dict[int, pd.DataFrame]:
    """Load all parquet files, indexed by episode_index."""
    info = json.loads((src_dir / "meta" / "info.json").read_text())
    data_path_template = info["data_path"]  # e.g. "data/chunk-{episode_chunk:03d}/..."

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


def build_wrench_history(
    episodes: dict[int, pd.DataFrame],
    history_steps: int,
    force_start_idx: int = 7,
    force_dim: int = 6,
) -> dict[int, np.ndarray]:
    """Build wrench_history [T, 6] for every frame in every episode.

    Returns a dict mapping episode_index → np.ndarray of shape [N_frames, T, 6].
    """
    zero = np.zeros((force_dim,), dtype=np.float32)
    histories: dict[int, np.ndarray] = {}

    for ep_idx, df in tqdm(episodes.items(), desc="Building wrench histories"):
        n_frames = len(df)
        # Extract force column: observation.state is stored as numpy array in the DF
        states = np.stack(df["observation.state"].values)  # [N, state_dim]
        forces = states[:, force_start_idx : force_start_idx + force_dim].astype(np.float32)  # [N, 6]

        hist = np.zeros((n_frames, history_steps, force_dim), dtype=np.float32)
        for i in range(n_frames):
            for offset in range(history_steps):
                src_idx = i - (history_steps - 1) + offset
                if 0 <= src_idx < n_frames:
                    hist[i, offset] = forces[src_idx]
                # else: leave as zero (episode boundary padding)
        histories[ep_idx] = hist

    return histories


def write_new_dataset(
    src_dir: Path,
    dst_dir: Path,
    episodes: dict[int, pd.DataFrame],
    histories: dict[int, np.ndarray],
    history_steps: int,
):
    """Write the new dataset to dst_dir, adding wrench_history to each parquet."""
    # Copy meta/ directory
    meta_src = src_dir / "meta"
    meta_dst = dst_dir / "meta"
    if meta_dst.exists():
        shutil.rmtree(meta_dst)
    shutil.copytree(meta_src, meta_dst)

    # Update info.json to include the new feature
    info_path = meta_dst / "info.json"
    info = json.loads(info_path.read_text())
    info.setdefault("features", {})["observation.wrench_history"] = {
        "dtype": "float32",
        "shape": [history_steps, 6],
        "names": [
            "wrench_history_fx", "wrench_history_fy", "wrench_history_fz",
            "wrench_history_tx", "wrench_history_ty", "wrench_history_tz",
        ],
    }
    info_path.write_text(json.dumps(info, indent=2))

    # Write parquet files — keep the same chunk structure
    parquet_files = sorted(src_dir.glob("data/**/*.parquet"))
    for pf in tqdm(parquet_files, desc="Writing new parquet files"):
        df = pd.read_parquet(pf)
        n = len(df)

        # Build wrench_history column for this chunk
        wrench_col = []
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            frame = int(row["frame_index"])
            # Find position in episode
            ep_df = episodes[ep]
            pos = ep_df[ep_df["frame_index"] == frame].index
            if len(pos) == 0:
                wrench_col.append(np.zeros((history_steps, 6), dtype=np.float32))
            else:
                wrench_col.append(histories[ep][pos[0]])

        # Add as new column
        df["observation.wrench_history"] = [w.ravel().tolist() for w in wrench_col]

        # Determine output path
        rel_path = pf.relative_to(src_dir)
        out_path = dst_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_path)

    # Symlink videos if they exist
    for video_dir_name in ["videos"]:
        video_src = src_dir / video_dir_name
        video_dst = dst_dir / video_dir_name
        if video_src.exists() and not video_dst.exists():
            video_dst.symlink_to(video_src.resolve())

    print(f"Done! New dataset written to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute force history for LeRobot dataset")
    parser.add_argument("--src", required=True, help="Source dataset directory")
    parser.add_argument("--dst", required=True, help="Destination dataset directory")
    parser.add_argument("--history-steps", type=int, default=60, help="Number of history frames (default: 60)")
    parser.add_argument("--force-start-idx", type=int, default=7, help="Index in observation.state where force starts")
    parser.add_argument("--force-dim", type=int, default=6, help="Force dimension (default: 6)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}")
        sys.exit(1)

    if dst_dir.exists():
        print(f"ERROR: Destination already exists: {dst_dir}")
        print("  Remove it manually or choose a different --dst.")
        sys.exit(1)

    print(f"Source:      {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"History:     {args.history_steps} frames × {args.force_dim} dims = {args.history_steps * args.force_dim}")
    print()

    # 1. Load
    episodes = load_episodes_data(src_dir)
    print(f"Loaded {len(episodes)} episodes, {sum(len(v) for v in episodes.values())} total frames")

    # 2. Build histories
    histories = build_wrench_history(
        episodes, args.history_steps, force_start_idx=args.force_start_idx, force_dim=args.force_dim
    )

    # 3. Write
    write_new_dataset(src_dir, dst_dir, episodes, histories, args.history_steps)


if __name__ == "__main__":
    main()
