"""动作阶段序列分析：识别 取擦子/放置/下压/水平擦拭/回程 各阶段，输出 EEF 姿态。"""
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
    states = np.array([l["state"] for l in lines])          # [7]: q1-6 + gripper
    fz = np.array([l["force"][2] for l in lines])
    n = min(len(states), len(fz))
    states, fz = states[:n], fz[:n]
    gripper = states[:, 6]
    q = states[:, :6]
    poses = np.array([np.concatenate(pose_to_xyz_rpy(fk_tool(s))) for s in q])
    xyz = poses[:, :3]
    rpy = np.degrees(poses[:, 3:])
    yaw, pitch, roll = rpy[:, 2], rpy[:, 1], rpy[:, 0]

    print(f"\n{'=' * 78}")
    print(f"{label}  [{os.path.basename(logfile)}]  n={n}  ({n/30:.1f}s)")
    print(f"  夹爪范围: {gripper.min():.3f}~{gripper.max():.3f} | Fz: {fz.min():.1f}~{fz.max():.1f}")

    # 阶段划分:
    #  Phase 0 起始(高z, 夹爪开): z>0.12
    #  Phase 1 取擦子: 夹爪从开->闭 (gripper 变化) 且 z 中高
    #  Phase 2 移到白板: z 中 (0.05-0.12), 夹爪闭
    #  Phase 3 下压: z 下降 >0.05 且 Fz 变负
    #  Phase 4 水平擦拭: z 低且稳定 + Fz<-15
    #  Phase 5 回程/结束: z 上升
    z = xyz[:, 2]
    phases = np.full(n, '?')
    for i in range(n):
        if z[i] > 0.12:
            phases[i] = '0_start' if gripper[i] < 0.3 else '5_ret'
        elif z[i] > 0.05:
            phases[i] = '1_pick' if gripper[i] > 0.5 else '2_move'
        else:
            phases[i] = '4_wipe' if fz[i] < -15 else '3_press'
    # 简化展示: 每30帧输出
    print("  阶段序列 (每30帧): 帧|阶段|z(cm)|gripper|Fz|yaw|pitch|roll")
    cur_phase = None
    for i in range(0, n, 30):
        ph = phases[i]
        marker = ""
        if ph != cur_phase:
            marker = " <<<"
            cur_phase = ph
        print(f"    {i:4d}| {ph:8s}| z={z[i]*100:5.1f} | gr={gripper[i]:.2f} | Fz={fz[i]:5.1f} | "
              f"yaw={yaw[i]:6.1f} | pitch={pitch[i]:6.1f} | roll={roll[i]:6.1f}{marker}")

    # 统计各阶段占比
    print("  各阶段帧数:", {p: int((phases == p).sum()) for p in sorted(set(phases))})

    # 擦拭段 (Phase 4) 的欧拉角分布
    wipe = phases == '4_wipe'
    if wipe.sum() > 10:
        print(f"\n  ★ 擦拭段({int(wipe.sum())}帧) 欧拉角:")
        print(f"    yaw:   {yaw[wipe].min():.1f}~{yaw[wipe].max():.1f}° (漂移 {yaw[wipe].max()-yaw[wipe].min():.1f}°)")
        print(f"    pitch: {pitch[wipe].min():.1f}~{pitch[wipe].max():.1f}° (中位 {np.median(pitch[wipe]):.1f}°)")
        print(f"    roll:  {roll[wipe].min():.1f}~{roll[wipe].max():.1f}° (中位 {np.median(roll[wipe]):.1f}°)")
        # 首尾位移
        idx = np.where(wipe)[0]
        s, e = idx[0], idx[-1]
        dx = np.linalg.norm(xyz[e, :2] - xyz[s, :2])
        dz = xyz[e, 2] - xyz[s, 2]
        ang = np.degrees(np.arctan2(abs(dz), dx)) if dx > 1e-3 else float('nan')
        print(f"    位移: Δxy={dx*100:.1f}cm Δz={dz*100:.1f}cm 倾角={ang:.1f}°")


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        import glob
        files = sorted(glob.glob("logs/20260808/eef_2k/*.jsonl"))
    for f in files:
        analyze(f, os.path.basename(os.path.dirname(f)))
