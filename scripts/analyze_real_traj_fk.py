#!/usr/bin/env python3
"""真机日志 FK 末端轨迹分析 — 用 piper_sdk 正运动学把关节轨迹转成末端位置。

读取 client_logs/*.jsonl，提取执行帧的 state(7维关节) 轨迹，
用 piper_sdk FK 换算成末端 [x,y,z,r,p,y]，分析各模型实际飞到哪：
  - 末端轨迹范围/净位移
  - 是否朝印章方向移动
  - 末端是否抖动/原地旋转
  - 夹爪行为

用法:
  cd /mnt/hdd/sfy/FA-VLA-结论
  conda activate rlinf
  python /mnt/hdd/sfy/FA-openpi/scripts/analyze_real_traj_fk.py
"""

import importlib.util
import json
import glob
import sys
from pathlib import Path

import numpy as np

# piper_sdk FK 模块（绕过 __init__ 避免 python-can 依赖）
_FK_PATH = Path("/mnt/hdd/sfy/piper_sdk/piper_sdk/kinematics/piper_fk.py")
_spec = importlib.util.spec_from_file_location("piper_fk", _FK_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FK = _mod.C_PiperForwardKinematics(dh_is_offset=0x01)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = sorted(glob.glob("/mnt/hdd/sfy/FA-VLA-结论/client_logs/*.jsonl"))


def fk_end(joints6):
    """6 关节 (rad) → 末端 [x,y,z,r,p,y]; xyz mm, rpy degree"""
    return np.asarray(FK.CalFK(list(joints6))[-1], dtype=np.float64)


def analyze(f):
    lines = [json.loads(l) for l in open(f)]
    # 执行帧关节轨迹
    exec_states = []   # 每帧 state[0:6]（关节）
    grip = []          # 夹爪 state[6]
    for l in lines:
        if not l.get("query"):
            st = np.asarray(l["state"], dtype=np.float64)
            exec_states.append(st[:6])
            grip.append(st[6])
    J = np.array(exec_states)      # (N,6)
    G = np.array(grip)             # (N,)
    if len(J) < 3:
        return None

    # FK 末端
    P = np.array([fk_end(j) for j in J[::4]])   # 采样减少计算
    idx = np.arange(0, len(J), 4)

    # 末端轨迹统计
    pos = P[:, :3]                       # xyz mm
    start_pt, end_pt = pos[0], pos[-1]
    net_disp = np.linalg.norm(end_pt - start_pt)        # 净位移
    total_path = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))  # 总行程
    span = np.ptp(pos, axis=0)           # 各轴范围

    # 末端速度（mm/s，假设 30Hz 执行）
    dt = 1 / 30.0
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    vel_max, vel_mean = np.max(vel), np.mean(vel)

    # 抖动检测：相邻帧末端位移突变
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    jerk_frac = float(np.mean(d > np.percentile(d, 90) * 3)) if len(d) else 0

    # 旋转分量 (rpy)
    rot = P[:, 3:]
    rot_span = np.ptp(rot, axis=0)

    # 夹爪
    grip_min = float(G.min())
    grip_final = float(G[-1])
    grip_closed = bool(G.min() < 0.02)   # 是否完全闭合过
    # 闭合次数（<0.03 视为闭合）
    closed = G < 0.03
    n_close_seg = 0
    prev = False
    for c in closed:
        if c and not prev:
            n_close_seg += 1
        prev = c

    return {
        "file": Path(f).name,
        "n_frames": len(J), "t_s": len(J) / 30.0,
        "end_start": start_pt.tolist(), "end_end": end_pt.tolist(),
        "net_disp_mm": float(net_disp), "total_path_mm": float(total_path),
        "span_xyz_mm": span.tolist(),
        "vel_max_mms": float(vel_max), "vel_mean_mms": float(vel_mean),
        "jerk_frac": jerk_frac,
        "rot_span_deg": rot_span.tolist(),
        "grip_min": grip_min, "grip_final": grip_final,
        "grip_closed": grip_closed, "n_close_seg": n_close_seg,
    }


def main():
    results = []
    for f in LOGS:
        r = analyze(f)
        if r:
            results.append(r)

    print("=" * 100)
    print("  真机日志 FK 末端轨迹分析 (30Hz 执行假设)")
    print("=" * 100)
    for r in results:
        print(f"\n--- {r['file']} ---")
        print(f"  帧数 {r['n_frames']} ({r['t_s']:.1f}s)")
        print(f"  末端起点: ({r['end_start'][0]:.0f}, {r['end_start'][1]:.0f}, {r['end_start'][2]:.0f}) mm")
        print(f"  末端终点: ({r['end_end'][0]:.0f}, {r['end_end'][1]:.0f}, {r['end_end'][2]:.0f}) mm")
        print(f"  净位移 {r['net_disp_mm']:.1f}mm | 总行程 {r['total_path_mm']:.1f}mm | 行程比 {r['total_path_mm']/max(r['net_disp_mm'],0.1):.1f}")
        print(f"  各轴范围 xyz: {[f'{v:.0f}' for v in r['span_xyz_mm']]} mm")
        print(f"  末端速度: max {r['vel_max_mms']:.0f} mm/s, mean {r['vel_mean_mms']:.0f} mm/s")
        print(f"  姿态范围 rpy: {[f'{v:.0f}' for v in r['rot_span_deg']]} deg")
        print(f"  夹爪: min={r['grip_min']:.3f} final={r['grip_final']:.3f} "
              f"闭合={r['grip_closed']} 闭合段数={r['n_close_seg']}")

    # 汇总表
    print("\n" + "=" * 100)
    print("  汇总对比")
    print("=" * 100)
    print(f"{'模型':<46} {'净移mm':>7} {'行程mm':>7} {'行程/净':>7} {'速度m/s':>7} {'夹爪min':>7} {'闭合段':>6}")
    for r in results:
        name = r['file'].split('_')[0]
        ratio = r['total_path_mm'] / max(r['net_disp_mm'], 0.1)
        print(f"{r['file'][:45]:<46} {r['net_disp_mm']:>7.1f} {r['total_path_mm']:>7.1f} "
              f"{ratio:>7.1f} {r['vel_mean_mms']/1000:>7.2f} {r['grip_min']:>7.3f} {r['n_close_seg']:>6}")

    # 图：各模型末端 XY 轨迹
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, f in zip(axes.flat, LOGS):
        lines = [json.loads(l) for l in open(f)]
        J = np.array([np.asarray(l["state"], dtype=np.float64)[:6]
                      for l in lines if not l.get("query")])
        if len(J) < 3:
            ax.set_visible(False)
            continue
        P = np.array([fk_end(j) for j in J[::3]])
        ax.plot(P[:, 0], P[:, 1], lw=1.0, color='#1E88E5')
        ax.plot(P[0, 0], P[0, 1], 'go', ms=8, label='start')
        ax.plot(P[-1, 0], P[-1, 1], 'r^', ms=8, label='end')
        ax.set_title(Path(f).name[:44], fontsize=9)
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.axis('equal')
    plt.tight_layout()
    out = "/mnt/hdd/sfy/FA-VLA-结论/end_traj_xy.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
