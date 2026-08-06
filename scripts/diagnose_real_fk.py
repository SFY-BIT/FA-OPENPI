#!/usr/bin/env python3
"""真机失败诊断 — 统一计算印章真实位置 + 各模型 FK 判定。"""
import importlib.util, json, glob, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, "/mnt/hdd/sfy/FA-openpi/src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_FK = Path("/mnt/hdd/sfy/piper_sdk/piper_sdk/kinematics/piper_fk.py")
spec = importlib.util.spec_from_file_location("piper_fk", _FK)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
fk = mod.C_PiperForwardKinematics(dh_is_offset=0x01)

def fk_end(j6):
    return np.asarray(fk.CalFK(list(j6))[-1], dtype=np.float64)

# ── 1. 印章真实位置 (30ep 闭合段 FK) ──
import lerobot.common.datasets.lerobot_dataset as ld
ds = ld.LeRobotDataset("/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60")
ds.hf_dataset.set_format("python")
ep_idx = ds.episode_data_index
all_poses = []
for ep in range(30):
    s, e = int(ep_idx["from"][ep]), int(ep_idx["to"][ep])
    for fi in range(s, e, 3):
        f = ds.hf_dataset[fi]
        st = np.asarray(f["observation.state"], dtype=np.float32)
        if st[6] < 0.031:
            all_poses.append(fk_end(st[:6]))
P = np.array(all_poses)
center = np.median(P[:, :3], axis=0)
print("=" * 78)
print(f"印章真实位置 (30ep 闭合段 FK 中位): ({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f}) mm")
print(f"闭合段 xyz 范围: x[{P[:,0].min():.0f}-{P[:,0].max():.0f}] "
      f"y[{P[:,1].min():.0f}-{P[:,1].max():.0f}] z[{P[:,2].min():.0f}-{P[:,2].max():.0f}]")
print(f"闭合段总数: {len(P)}")
print("=" * 78)

# ── 2. 各模型判定 ──
print()
print(f"{'模型':<10} {'到印章区':>7} {'印章区帧':>8} {'夹爪min':>7} {'印章区夹爪min':>11} {'末端净移mm':>9} {'终点距印章':>9}")
for f in sorted(glob.glob("/mnt/hdd/sfy/FA-VLA-结论/client_logs/*.jsonl")):
    base = Path(f).name
    name = base.split("_")[0]
    steps = "30" if "30step" in base else ("15" if "15step" in base else "10")
    lines = [json.loads(l) for l in open(f)]
    grips_all, near_grips, n_near, n_total = [], [], 0, 0
    J = []
    for l in lines:
        if not l.get("query"):
            n_total += 1
            st = np.asarray(l["state"], dtype=np.float64)
            J.append(st[:6]); grips_all.append(st[6])
            end = fk_end(st[:6])
            if np.linalg.norm(end[:3] - center) < 80:
                n_near += 1; near_grips.append(st[6])
    J = np.array(J)
    end_final = fk_end(J[-1])[:3]
    net = np.linalg.norm(end_final - fk_end(J[0])[:3])
    dist = np.linalg.norm(end_final - center)
    print(f"{name:>5}-{steps:<4} {str(n_near>0):>7} {n_near:>5}/{n_total:<3} "
          f"{min(grips_all):>7.3f} {min(near_grips) if near_grips else float('nan'):>11.3f} "
          f"{net:>9.1f} {dist:>9.1f}")
