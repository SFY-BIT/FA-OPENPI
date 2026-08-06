#!/usr/bin/env python3
"""Compare joint trajectories between force (V6) and no-force rollouts.

Loads joint_log files from both rollout outputs and computes:
  - Per-joint trajectory difference (L2 distance over time)
  - Final joint position difference
  - Gripper behavior difference
  - Success rate comparison

Usage:
  python scripts/compare_rollouts.py \
    --force-dir /mnt/hdd/sfy/outputs/rollouts_v6 \
    --noforce-dir /mnt/hdd/sfy/outputs/rollouts_noforce \
    --task usb
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_joint_logs(directory: Path, task: str, model_tag: str) -> list[dict]:
    """Load all joint_log files matching the task and model tag."""
    pattern = f"joint_log_{model_tag}_{task}_ep*_seed*.json"
    files = sorted(directory.glob(pattern))
    logs = []
    for f in files:
        with open(f) as fh:
            logs.append({
                "file": str(f),
                "data": json.load(fh),
            })
    return logs


def extract_trajectories(log: list[dict]) -> np.ndarray:
    """Extract joint states from a log, shape (steps, 8)."""
    return np.array([entry["state"] for entry in log])


def compute_comparison(force_logs: list[dict], noforce_logs: list[dict]):
    """Compare trajectories between force and no-force models."""
    print(f"\n{'=' * 70}")
    print(f"Force vs No-Force Joint Trajectory Comparison")
    print(f"{'=' * 70}")

    print(f"\nForce model logs:     {len(force_logs)} episodes")
    print(f"No-force model logs:  {len(noforce_logs)} episodes")

    if not force_logs or not noforce_logs:
        print("\n⚠️  Not enough data to compare. Need both force and no-force logs.")
        return

    # Compare episode by episode (match by index)
    n_compare = min(len(force_logs), len(noforce_logs))
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4",
                   "joint_5", "joint_6", "joint_7", "gripper"]

    all_diffs = []
    all_force_final = []
    all_noforce_final = []

    print(f"\nComparing {n_compare} episode pairs:")
    print(f"{'Ep':>3} {'Force steps':>12} {'NoF steps':>12} "
          f"{'Mean L2 diff':>14} {'Max L2 diff':>14} {'Final diff':>14}")
    print("-" * 75)

    for i in range(n_compare):
        f_traj = extract_trajectories(force_logs[i]["data"])
        nf_traj = extract_trajectories(noforce_logs[i]["data"])

        # Align to shorter trajectory
        min_len = min(len(f_traj), len(nf_traj))
        f_aligned = f_traj[:min_len]
        nf_aligned = nf_traj[:min_len]

        # Per-step L2 difference
        step_diffs = np.linalg.norm(f_aligned - nf_aligned, axis=1)
        mean_diff = float(np.mean(step_diffs))
        max_diff = float(np.max(step_diffs))

        # Final position difference
        f_final = f_traj[-1]
        nf_final = nf_traj[-1]
        final_diff = float(np.linalg.norm(f_final - nf_final))

        all_diffs.append(step_diffs)
        all_force_final.append(f_final)
        all_noforce_final.append(nf_final)

        print(f"{i:>3} {len(f_traj):>12} {len(nf_traj):>12} "
              f"{mean_diff:>14.6f} {max_diff:>14.6f} {final_diff:>14.6f}")

    # Per-joint analysis
    print(f"\n{'=' * 70}")
    print("Per-joint final position comparison (averaged over episodes):")
    print(f"{'=' * 70}")
    all_force_final = np.array(all_force_final)
    all_noforce_final = np.array(all_noforce_final)
    mean_force = all_force_final.mean(axis=0)
    mean_noforce = all_noforce_final.mean(axis=0)
    std_force = all_force_final.std(axis=0)
    std_noforce = all_noforce_final.std(axis=0)

    print(f"\n{'Joint':<10} {'Force mean':>12} {'Force std':>12} "
          f"{'NoF mean':>12} {'NoF std':>12} {'|diff|':>12}")
    print("-" * 75)
    for j, name in enumerate(joint_names):
        diff = abs(mean_force[j] - mean_noforce[j])
        print(f"{name:<10} {mean_force[j]:>12.6f} {std_force[j]:>12.6f} "
              f"{mean_noforce[j]:>12.6f} {std_noforce[j]:>12.6f} {diff:>12.6f}")

    # Trajectory divergence over time
    print(f"\n{'=' * 70}")
    print("Trajectory divergence over time (mean L2 diff per step):")
    print(f"{'=' * 70}")
    max_len = max(len(d) for d in all_diffs)
    padded = np.full((len(all_diffs), max_len), np.nan)
    for i, d in enumerate(all_diffs):
        padded[i, :len(d)] = d
    mean_over_eps = np.nanmean(padded, axis=0)

    print(f"\n{'Step':>6} {'Mean L2 diff':>14} {'Std':>14}")
    print("-" * 40)
    for s in range(0, min(len(mean_over_eps), 200), max(1, len(mean_over_eps) // 20)):
        std = np.nanstd(padded[:, s])
        print(f"{s:>6} {mean_over_eps[s]:>14.6f} {std:>14.6f}")
    if len(mean_over_eps) > 0:
        print(f"{'final':>6} {mean_over_eps[-1]:>14.6f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Overall mean L2 trajectory diff: {np.nanmean(padded):.6f}")
    print(f"  Overall max L2 trajectory diff:  {np.nanmax(padded):.6f}")
    print(f"  Mean final position diff:        {np.mean(np.linalg.norm(all_force_final - all_noforce_final, axis=1)):.6f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Compare force vs no-force rollout trajectories")
    parser.add_argument("--force-dir", default="/mnt/hdd/sfy/outputs/rollouts_v6",
                        help="Directory with force (V6) joint logs")
    parser.add_argument("--noforce-dir", default="/mnt/hdd/sfy/outputs/rollouts_noforce",
                        help="Directory with no-force joint logs")
    parser.add_argument("--task", default="usb", choices=["usb", "whiteboard"],
                        help="Task to compare")
    args = parser.parse_args()

    force_dir = Path(args.force_dir)
    noforce_dir = Path(args.noforce_dir)

    force_logs = load_joint_logs(force_dir, args.task, "v6")
    noforce_logs = load_joint_logs(noforce_dir, args.task, "noforce")

    if not force_logs:
        print(f"⚠️  No force logs found in {force_dir} for task '{args.task}'")
        print(f"   Looking for: joint_log_v6_{args.task}_ep*_seed*.json")
    if not noforce_logs:
        print(f"⚠️  No no-force logs found in {noforce_dir} for task '{args.task}'")
        print(f"   Looking for: joint_log_noforce_{args.task}_ep*_seed*.json")

    compute_comparison(force_logs, noforce_logs)


if __name__ == "__main__":
    main()
