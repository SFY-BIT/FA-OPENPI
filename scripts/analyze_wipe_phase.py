"""整条轨迹逐帧姿态+滑动窗口倾角分析（聚焦擦拭阶段水平度）。"""
import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from piper_fk import fk_tool, pose_to_xyz_rpy


def analyze(logfile: str, label: str) -> None:
    with open(logfile) as f:
        lines = [json.loads(l) for l in f]
    if not lines:
        return
    states = np.array([l["state"][:6] for l in lines])
    fz = np.array([l["force"][2] for l in lines])
    n = min(len(states), len(fz))
    states, fz = states[:n], fz[:n]
    poses = np.array([np.concatenate(pose_to_xyz_rpy(fk_tool(s))) for s in states])
    xyz = poses[:, :3]
    rpy = np.degrees(poses[:, 3:])
    yaw, pitch = rpy[:, 2], rpy[:, 1]

    print(f"\n{'=' * 72}")
    print(f"{label}  [{os.path.basename(logfile)}]  n={n}")
    print(f"  Fz: {fz.min():.1f}~{fz.max():.1f}N | 接触帧(Fz<-15): {int((fz < -15).sum())} | 用力擦(Fz<-19): {int((fz < -19).sum())}")

    # 阶段划分: 用 z 高度 + Fz
    z = xyz[:, 2]
    # 1) 接近段: z 高 (>8cm) 且未用力
    # 2) 下压段: z 快速下降
    # 3) 水平擦拭段: Fz 用力 + z 稳定
    # 4) 抬起/回程: 其他
    touch = fz < -15.0
    wipe = fz < -19.0
    # 滑动窗口位移倾角 (30帧窗口 ≈ 1s)
    W = 30
    ang_win = np.full(n, np.nan)
    dyaw_win = np.full(n, np.nan)
    dx_win = np.full(n, np.nan)
    for i in range(W, n):
        dx = np.linalg.norm(xyz[i, :2] - xyz[i - W, :2])
        dz = xyz[i, 2] - xyz[i - W, 2]
        ang_win[i] = np.degrees(np.arctan2(abs(dz), dx)) if dx > 1e-3 else 0.0
        dyaw_win[i] = abs(yaw[i] - yaw[i - W])
        dx_win[i] = dx * 100
    # 只看擦拭段内的窗口 (窗口中心在用力擦区)
    wipe_win = np.array([wipe[max(0, i - W // 2)] and wipe[min(n - 1, i + W // 2)] for i in range(n)])
    sel = wipe_win & ~np.isnan(ang_win)
    if sel.sum() > 0:
        worst = np.nanargmax(ang_win[sel])
        worst_i = np.where(sel)[0][worst]
        print(f"  擦拭段滑动窗口(30帧)位移倾角: 中位 {np.nanmedian(ang_win[sel]):.1f}° | "
              f"最大 {np.nanmax(ang_win[sel]):.1f}° @帧{worst_i} (Δxy={dx_win[worst_i]:.1f}cm, Δyaw={dyaw_win[worst_i]:.1f}°)")
        # 倾角分布
        p25, p50, p75 = np.percentile(ang_win[sel], [25, 50, 75])
        print(f"  倾角分位数: P25={p25:.1f}° P50={p50:.1f}° P75={p75:.1f}° | "
              f"水平占比(<10°): {100 * np.mean(ang_win[sel] < 10):.0f}%")
        # 擦拭段内 yaw 漂移
        yw = yaw[sel]
        print(f"  擦拭段 yaw: {yw.min():.1f}~{yw.max():.1f}° (总漂移 {yw.max() - yw.min():.1f}°) | "
              f"pitch: {pitch[sel].min():.1f}~{pitch[sel].max():.1f}°")
        # 总位移
        total_dx = np.linalg.norm(xyz[sel][-1, :2] - xyz[sel][0, :2])
        total_dz = xyz[sel][-1, 2] - xyz[sel][0, 2]
        print(f"  擦拭段总位移: Δxy={total_dx * 100:.1f}cm Δz={total_dz * 100:.1f}cm "
              f"({int(sel.sum())}帧, {sel.sum() / 30:.1f}s)")
    else:
        print("  无擦拭段(Fz<-19)")

    # 逐帧概要: 每 50 帧打印一次姿态
    print("  时间线 (每50帧): 帧|z(cm)|Fz|yaw|pitch|30帧倾角")
    for i in range(0, n, 50):
        a = ang_win[i] if not np.isnan(ang_win[i]) else -1
        print(f"    {i:4d} | z={z[i] * 100:5.1f} | Fz={fz[i]:5.1f} | yaw={yaw[i]:7.1f} | "
              f"pitch={pitch[i]:6.1f} | 倾角={a:5.1f}")


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        files = ["logs/20260808/eef_2k_scale11/eef_2k_scale11_20260808_140913.jsonl",
                 "logs/20260808/eef_2k/eef_2k_20260808_133839.jsonl",
                 "logs/20260808/lora_3w/lora_3w_20260808_134308.jsonl"]
    for f in files:
        analyze(f, os.path.basename(os.path.dirname(f)))
