#!/usr/bin/env python3
"""末端位置（笛卡尔）分析 — 用 piper_sdk 正运动学把关节对比转成末端位置对比。

读取 offline_replay 生成的 replay_*.json（含 action_gt / action_pred 关节），
用 piper_sdk 的 C_PiperForwardKinematics.CalFK 把 6 关节角度 (radian) 换算成
末端位姿 [x,y,z,r,p,y]（xyz 单位 mm，rpy 单位 degree），然后对比:
  - 末端位置偏差 (预测末端 − 真实末端), 3D 范数
  - 姿态偏差 (roll/pitch/yaw)
  - 轨迹可视化 (XY / XZ 平面 + 3D)

用法:
  conda activate rlinf
  cd /mnt/hdd/sfy/FA-openpi
  python -u scripts/analyze_endpose.py \
      --replay outputs/offline_replay/replay_ep0.json \
      --output-dir outputs/offline_replay/endpose
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

# piper_sdk 的 FK 模块是纯数学，无硬件依赖；直接加载模块绕过 __init__
# （piper_sdk.__init__ 会导入 python-can，没有硬件环境会失败）
_FK_PATH = Path("/mnt/hdd/sfy/piper_sdk/piper_sdk/kinematics/piper_fk.py")
_spec = importlib.util.spec_from_file_location("piper_fk", _FK_PATH)
_piper_fk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_piper_fk)
C_PiperForwardKinematics = _piper_fk.C_PiperForwardKinematics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 数据集关节顺序: j1..j6 (Piper 6 关节) + gripper
# CalFK 接收 6 关节 [j1..j6] radian
def fk_joints(joints6: np.ndarray, fk) -> np.ndarray:
    """6 关节 (radian) → [x,y,z,r,p,y]; xyz mm, rpy degree."""
    pose = fk.CalFK(list(joints6))[-1]  # 最后一组 = 末端
    return np.asarray(pose, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", required=True, help="replay_ep*.json (含 action_gt/action_pred)")
    p.add_argument("--output-dir", default="/mnt/hdd/sfy/FA-openpi/outputs/offline_replay/endpose")
    p.add_argument("--label", default="", help="图例后缀，如 local/remote")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.replay) as f:
        data = json.load(f)

    frames = data["frames"]
    gt_joints = np.array([fr["action_gt"] for fr in frames])[:, :6]     # (N,6) rad
    pred_joints = np.array([fr["action_pred"] for fr in frames])[:, :6]

    fk = C_PiperForwardKinematics(dh_is_offset=0x01)

    # ── FK 换算 ──
    print("[fk] converting %d frames (joint → end pose) ..." % len(frames))
    gt_pose = np.array([fk_joints(j, fk) for j in gt_joints])     # (N,6) [x,y,z,r,p,y]
    pred_pose = np.array([fk_joints(j, fk) for j in pred_joints])

    # ── 偏差 ──
    pos_err = pred_pose[:, :3] - gt_pose[:, :3]          # mm
    pos_norm = np.linalg.norm(pos_err, axis=1)           # 3D 位置偏差 (mm)
    rot_err = pred_pose[:, 3:] - gt_pose[:, 3:]          # degree
    rot_norm = np.linalg.norm(rot_err, axis=1)

    print("\n" + "=" * 66)
    print("  末端位置偏差统计 (预测末端 − 真实末端)")
    print("  xyz: mm | rpy: degree | N=%d" % len(frames))
    print("=" * 66)
    print(f"{'轴':>6} {'MAE':>10} {'RMSE':>10} {'MAX':>10}")
    for i, name in enumerate(["x", "y", "z"]):
        print(f"{name:>6} {np.mean(np.abs(pos_err[:,i])):>10.3f} {np.sqrt(np.mean(pos_err[:,i]**2)):>10.3f} {np.max(np.abs(pos_err[:,i])):>10.3f}")
    print("-" * 66)
    print(f"{'3D范数':>6} {np.mean(pos_norm):>10.3f} {np.sqrt(np.mean(pos_norm**2)):>10.3f} {np.max(pos_norm):>10.3f}")
    print(f"{'roll':>6} {np.mean(np.abs(rot_err[:,0])):>10.3f} {np.sqrt(np.mean(rot_err[:,0]**2)):>10.3f} {np.max(np.abs(rot_err[:,0])):>10.3f}")
    print(f"{'pitch':>6} {np.mean(np.abs(rot_err[:,1])):>10.3f} {np.sqrt(np.mean(rot_err[:,1]**2)):>10.3f} {np.max(np.abs(rot_err[:,1])):>10.3f}")
    print(f"{'yaw':>6} {np.mean(np.abs(rot_err[:,2])):>10.3f} {np.sqrt(np.mean(rot_err[:,2]**2)):>10.3f} {np.max(np.abs(rot_err[:,2])):>10.3f}")

    # 进度相关 (接触段 vs 非接触段)
    if "force_gt" in frames[0]:
        forces = np.array([fr["force_gt"] for fr in frames])
        fz = np.abs(forces[:, 2])
        thr = np.percentile(fz, 80)
        contact = fz > thr
        print("\n  接触段分析 (|Fz|>p80): 3D 位置偏差")
        print(f"  非接触段: {np.mean(pos_norm[~contact]):.3f} mm (n={int((~contact).sum())})")
        print(f"  接触段:   {np.mean(pos_norm[contact]):.3f} mm (n={int(contact.sum())})")

    # ── 图 ──
    label = args.label or Path(args.replay).stem
    fig = plt.figure(figsize=(16, 5.5))
    t = np.arange(len(frames)) * 0.1  # 假想 10Hz

    # (a) XY 轨迹
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(gt_pose[:, 0], gt_pose[:, 1], 'k-', lw=2, label='GT end pose')
    ax.plot(pred_pose[:, 0], pred_pose[:, 1], '--', lw=1.5, label='Pred end pose', color='#E53935')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_title(f'End-Pose XY {label}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.axis('equal')

    # (b) XZ 轨迹
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gt_pose[:, 0], gt_pose[:, 2], 'k-', lw=2, label='GT')
    ax.plot(pred_pose[:, 0], pred_pose[:, 2], '--', lw=1.5, label='Pred', color='#E53935')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (mm)'); ax.set_title(f'End-Pose XZ {label}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.axis('equal')

    # (c) 3D 位置偏差随时间
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, pos_norm, color='#1E88E5', lw=1.5, label='3D pos err')
    ax.axhline(np.mean(pos_norm), color='#E53935', ls='--', lw=1, label=f'mean={np.mean(pos_norm):.2f}mm')
    ax.set_xlabel('time (s)'); ax.set_ylabel('pos err (mm)'); ax.set_title(f'3D End-Pose Error {label}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = out_dir / f"endpose_{label}.png"
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f"\n[save] {out_png}")

    # ── 保存 ──
    out_json = out_dir / f"endpose_{label}.json"
    summary = {
        "label": label, "n_frames": len(frames),
        "pos_mae_xyz": [float(np.mean(np.abs(pos_err[:, i]))) for i in range(3)],
        "pos_rmse_xyz": [float(np.sqrt(np.mean(pos_err[:, i] ** 2))) for i in range(3)],
        "pos_max_xyz": [float(np.max(np.abs(pos_err[:, i]))) for i in range(3)],
        "pos_norm_mean": float(np.mean(pos_norm)),
        "pos_norm_rmse": float(np.sqrt(np.mean(pos_norm ** 2))),
        "pos_norm_max": float(np.max(pos_norm)),
        "rot_mae": [float(np.mean(np.abs(rot_err[:, i]))) for i in range(3)],
        "frames": [
            {
                "progress": fr.get("progress"),
                "gt_pose": gt_pose[i].tolist(),
                "pred_pose": pred_pose[i].tolist(),
                "pos_err_mm": pos_err[i].tolist(),
                "pos_norm_mm": float(pos_norm[i]),
                "rot_err_deg": rot_err[i].tolist(),
            }
            for i, fr in enumerate(frames)
        ],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
