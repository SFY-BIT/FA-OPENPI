"""对比 3w 步(无EEF) vs EEF/2000 vs EEF/2500 的擦拭段倾斜程度。"""
import sys, json, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
from piper_fk import fk_tool, pose_to_xyz_rpy


def analyze(logfile):
    """返回 (n, 擦拭帧数, pitch中位, pitch范围, 位移倾角, yaw漂移, 夹爪闭合段)"""
    try:
        lines = [json.loads(l) for l in open(logfile)]
    except Exception:
        return None
    if not lines:
        return None
    st = np.array([l['state'][:6] for l in lines if isinstance(l.get('state'), list) and len(l['state']) >= 6])
    fz = np.array([l['force'][2] for l in lines if isinstance(l.get('force'), list) and len(l['force']) >= 3])
    n = min(len(st), len(fz))
    if n < 30:
        return None
    st, fz = st[:n], fz[:n]
    poses = np.array([np.concatenate(pose_to_xyz_rpy(fk_tool(s))) for s in st])
    xyz, rpy = poses[:, :3], np.degrees(poses[:, 3:])
    z = xyz[:, 2]
    # 擦拭段: Fz<-15 且 z<5cm
    wipe = (fz < -15) & (z < 0.05)
    if wipe.sum() < 20:
        return None
    p = rpy[wipe, 1]; y = rpy[wipe, 2]
    idx = np.where(wipe)[0]
    s, e = idx[0], idx[-1]
    dx = np.linalg.norm(xyz[e, :2] - xyz[s, :2])
    dz = xyz[e, 2] - xyz[s, 2]
    ang = np.degrees(np.arctan2(abs(dz), dx)) if dx > 1e-3 else float('nan')
    yaw_drift = y.max() - y.min()
    # 夹爪命令闭合段数
    cmd_g = []
    for l in lines:
        cr = l.get('chunk_received')
        if isinstance(cr, list) and len(cr) and isinstance(cr[0], list) and len(cr[0]) > 6:
            cmd_g.append(cr[0][6])
    cmd_g = np.array(cmd_g)
    closed = int((cmd_g < 0.02).sum()) if len(cmd_g) else 0
    return {
        'n': n, 'wipe': int(wipe.sum()),
        'pitch_med': float(np.median(p)), 'pitch_p10': float(np.percentile(p, 10)), 'pitch_p90': float(np.percentile(p, 90)),
        'ang': float(ang) if not np.isnan(ang) else -1,
        'yaw_drift': float(yaw_drift), 'closed_cmd': closed,
        'gripper_min': float(np.min([l['state'][6] for l in lines])), 'gripper_max': float(np.max([l['state'][6] for l in lines])),
    }


def print_group(title, files, labels):
    print(f"\n{'='*88}")
    print(f"### {title}")
    print(f"{'文件':<42} | {'n':>4} | {'擦拭':>4} | {'pitch中位':>8} | {'P10~P90':>11} | {'倾角':>5} | {'yaw漂移':>7} | {'夹闭帧':>4} | 评价")
    for f, lb in zip(files, labels):
        r = analyze(f)
        if r is None:
            print(f"{os.path.basename(f):<42} | 无法分析")
            continue
        pitch_ok = 11.5 <= r['pitch_med'] <= 26.7
        ang_ok = r['ang'] < 10
        v = '✅' if (pitch_ok and ang_ok) else '❌'
        print(f"{os.path.basename(f):<42} | {r['n']:>4} | {r['wipe']:>4} | {r['pitch_med']:>7.1f}° | {r['pitch_p10']:5.1f}~{r['pitch_p90']:5.1f} | {r['ang']:>4.1f}° | {r['yaw_drift']:>6.1f}° | {r['closed_cmd']:>4} | {v}")


# ============ 分组对比 ============
# 1) 3w 步 (无 EEF) - 今天 lora_3w + 之前 8/5-8/6 client_logs
print("="*88)
print("【数据集基准】擦拭段 pitch 中位 17.2° (P10 11.5~P90 26.7), 位移倾角 1-7°(水平), yaw漂移 10-30°(锁定)")

three_w_files = ['logs/20260808/lora_3w/lora_3w_20260808_134308.jsonl']
three_w_files += sorted(glob.glob('client_logs/*3w*.jsonl'))[:4]
three_w_files += ['client_logs/SUCCESS_Erase_Board_20260805_155709.jsonl']
print_group("1️⃣ 3w 步 LoRA (无 EEF) - 今天 + 8/5-8/6", three_w_files, ['today'] * len(three_w_files))

# 2) EEF/2000
eef2k_files = sorted(glob.glob('logs/20260808/eef_2k/*.jsonl'))
print_group("2️⃣ EEF/2000 (无scale) - 今天", eef2k_files, ['eef2k'] * len(eef2k_files))

# 3) EEF/2000 scale11
eef2k11_files = sorted(glob.glob('logs/20260808/eef_2k_scale11/*.jsonl'))
print_group("3️⃣ EEF/2000 scale11 - 今天", eef2k11_files, ['eef2k11'] * len(eef2k11_files))

# 4) EEF/2500 scale11
eef25_files = sorted(glob.glob('logs/20260808/eef_2k5_scale11/*.jsonl'))
print_group("4️⃣ EEF/2500 scale11 - 今天", eef25_files, ['eef25'] * len(eef25_files))

# ============ 汇总统计 ============
print("\n" + "="*88)
print("### 汇总（按组取中位数）")
import statistics
groups = [
    ('3w 步 (无EEF)', three_w_files),
    ('EEF/2000', eef2k_files),
    ('EEF/2000 scale11', eef2k11_files),
    ('EEF/2500 scale11', eef25_files),
]
print(f"{'组别':<18} | {'轨迹数':>4} | {'pitch中位(组中位)':>16} | {'pitch偏差vs基准':>13} | {'倾角中位':>7} | {'yaw漂移中位':>10}")
for name, files in groups:
    res = [r for r in (analyze(f) for f in files) if r]
    if not res:
        continue
    pmed = statistics.median(r['pitch_med'] for r in res)
    ang = statistics.median(r['ang'] for r in res)
    yaw = statistics.median(r['yaw_drift'] for r in res)
    bias = pmed - 17.2
    print(f"{name:<18} | {len(res):>4} | {pmed:>13.1f}° | {bias:>+11.1f}° | {ang:>6.1f}° | {yaw:>9.1f}°")
