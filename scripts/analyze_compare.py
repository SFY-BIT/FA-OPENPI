"""统一对比所有 checkpoint 的擦拭段倾斜指标。"""
import sys, json, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from piper_fk import fk_tool, pose_to_xyz_rpy

def analyze(logfile):
    try:
        lines = [json.loads(l) for l in open(logfile)]
    except Exception:
        return None
    if not lines: return None
    st = np.array([l['state'][:6] for l in lines if isinstance(l.get('state'), list) and len(l['state']) >= 6])
    fz = np.array([l['force'][2] for l in lines if isinstance(l.get('force'), list) and len(l['force']) >= 3])
    n = min(len(st), len(fz))
    if n < 30: return None
    st, fz = st[:n], fz[:n]
    poses = np.array([np.concatenate(pose_to_xyz_rpy(fk_tool(s))) for s in st])
    xyz, rpy = poses[:, :3], np.degrees(poses[:, 3:])
    z = xyz[:, 2]
    wipe = (fz < -15) & (z < 0.05)
    if wipe.sum() < 20: return None
    p = rpy[wipe, 1]; y = rpy[wipe, 2]
    idx = np.where(wipe)[0]; s, e = idx[0], idx[-1]
    dx = np.linalg.norm(xyz[e,:2]-xyz[s,:2]); dz = xyz[e,2]-xyz[s,2]
    ang = np.degrees(np.arctan2(abs(dz), dx)) if dx > 1e-3 else float('nan')
    # 夹爪命令闭合段
    cmd_g = []
    for l in lines:
        cr = l.get('chunk_received')
        if isinstance(cr, list) and len(cr) and isinstance(cr[0], list) and len(cr[0]) > 6:
            cmd_g.append(cr[0][6])
        else:
            cmd_g.append(np.nan)
    cmd_g = np.array(cmd_g)
    closed_cmd = int((cmd_g < 0.02).sum()) if len(cmd_g) else 0
    return {
        'n': n, 'wipe': int(wipe.sum()),
        'pitch_med': float(np.median(p)), 'pitch_p10': float(np.percentile(p,10)), 'pitch_p90': float(np.percentile(p,90)),
        'yaw_drift': float(y.max()-y.min()),
        'ang': float(ang) if not np.isnan(ang) else -1,
        'closed_cmd': closed_cmd,
    }

print(f"{'组':<8} {'文件':<32} {'擦拭':>4} {'pitch中':>6} {'p10~p90':>11} {'yaw漂移':>6} {'倾角':>5} {'闭令':>4} | 判定")
groups = [
    ('SUCCESS', glob.glob('client_logs/SUCCESS_*.jsonl')),
    ('lora_1w', glob.glob('client_logs/*1w*.jsonl')),
    ('lora_2w', glob.glob('client_logs/*2w*.jsonl')),
    ('lora_3w_806', glob.glob('client_logs/eraseBoard_3w*.jsonl')),
    ('lora_3w_today', glob.glob('logs/20260808/lora_3w/*.jsonl')),
    ('EEF2k', glob.glob('logs/20260808/eef_2k/*.jsonl')),
    ('EEF2k_s11', glob.glob('logs/20260808/eef_2k_scale11/*.jsonl')),
]
for label, files in groups:
    for f in sorted(files):
        r = analyze(f)
        if r is None:
            print(f"{label:<8} {os.path.basename(f)[:32]:<32} 无擦拭段")
            continue
        # 判定: pitch 接近数据集 17.2° 且倾角<10°
        pitch_ok = 11.5 <= r['pitch_med'] <= 26.7
        ang_ok = r['ang'] < 10 if r['ang'] >= 0 else False
        v = '✅' if pitch_ok and ang_ok else '❌'
        print(f"{label:<8} {os.path.basename(f)[:32]:<32} {r['wipe']:>4} {r['pitch_med']:>5.1f}° {r['pitch_p10']:>4.1f}~{r['pitch_p90']:>4.1f} {r['yaw_drift']:>5.1f}° {r['ang']:>4.1f}° {r['closed_cmd']:>4} | {v}")
print()
print("数据集基准: pitch 中位 17.2° (P10 11.5~P90 26.7), 位移倾角 1-7° (水平)")
