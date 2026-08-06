#!/usr/bin/env python3
"""所有真机测试 checkpoint 的统一对比表。

统一指标:
  - 闭合意图: query 中 chunk 夹爪 <0.02 的比例
  - 夹爪闭合保持帧: 执行帧 grip<0.033 的占比
  - 末端净位移: FK 起点→终点
  - 终点距印章: FK 终点到印章中心(325,46,260) 的距离
"""
import importlib.util
import json
import numpy as np
from pathlib import Path

_FK = Path("/mnt/hdd/sfy/piper_sdk/piper_sdk/kinematics/piper_fk.py")
spec = importlib.util.spec_from_file_location("piper_fk", _FK)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fk = mod.C_PiperForwardKinematics(dh_is_offset=0x01)

STAMP = np.array([325.0, 46.0, 260.0])  # 印章中心 (数据集 30ep 闭合段 FK 中位)

def fk_end(j6):
    return np.asarray(fk.CalFK(list(j6))[-1], dtype=np.float64)

def analyze(path):
    lines = [json.loads(l) for l in open(path)]
    nq = sum(1 for l in lines if l.get("query"))
    nc = 0
    for l in lines:
        if l.get("query"):
            cg = np.array(l["chunk_received"])[:, 6]
            if cg.min() < 0.02:
                nc += 1
    grips = [l["state"][6] for l in lines if not l.get("query")]
    execs = [np.asarray(l["state"], dtype=np.float64) for l in lines if not l.get("query")]
    P0 = fk_end(execs[0])
    P1 = fk_end(execs[-1])
    net = np.linalg.norm(P1[:3] - P0[:3])
    dist_stamp = np.linalg.norm(P1[:3] - STAMP)
    grip_min = min(grips)
    grip_final = grips[-1]
    hold = sum(1 for g in grips if g < 0.033)
    return {
        "file": Path(path).name,
        "nq": nq, "nc": nc,
        "close_intent": nc / max(nq, 1),
        "hold": hold, "n_exec": len(grips),
        "grip_min": grip_min, "grip_final": grip_final,
        "net": net, "dist_stamp": dist_stamp,
        "end_final": P1[:3],
    }

CASES = [
    # (标签, 训练方式/步数, 架构, 日志路径)
    ("token1 @1w (成功)", "1token 全参", "client_logs/full1w_20260804_143122.jsonl"),
    ("token1 @1w-2", "1token 全参", "client_logs/full1w_20260804_143011.jsonl"),
    ("token1 @2w #1", "1token 全参", "client_logs/token1_2w_zmq30steps_20260804_152536.jsonl"),
    ("token1 @2w #2", "1token 全参", "client_logs/token1_2w_zmq30steps_20260804_152416.jsonl"),
    ("token1 @4w2-15", "1token 全参", "client_logs/token1_openpiforce42000_zmq15steps_20260803_220316.jsonl"),
    ("token1 @4w2-30", "1token 全参", "client_logs/token1_openpiforce42000_zmq30steps_20260803_220820.jsonl"),
    ("fa16k @1w", "16token 全参", "client_logs/fa16k_1w_20260804_151801.jsonl"),
    ("full16k @2w-10", "16token 全参", "client_logs/full16k_remote_10steps_20260803_222740.jsonl"),
    ("full16k @2w-30", "16token 全参", "client_logs/full16k_remote_30steps_20260803_223309.jsonl"),
    ("lora16k @3w-10", "16token LoRA", "client_logs/lora16k_local_10steps_20260803_223701.jsonl"),
    ("lora16k @3w-30", "16token LoRA", "client_logs/lora16k_local_30steps_20260803_224041.jsonl"),
]

print("=" * 150)
print("  全部真机测试 checkpoint 统一对比表")
print("=" * 150)
hdr = (f"{'模型':<18} {'架构':<12} {'闭合意图':>10} {'闭合帧%':>8} {'夹爪min':>7} "
       f"{'夹爪final':>9} {'末端净移mm':>9} {'距印章mm':>8} {'终点xyz':>18}")
print(hdr)
print("-" * 150)

results = []
for label, arch, path in CASES:
    try:
        r = analyze(path)
    except Exception as e:
        print(f"{label:<18} ERROR: {e}")
        continue
    r["label"] = label
    r["arch"] = arch
    results.append(r)
    end = r["end_final"]
    print(f"{label:<18} {arch:<12} {r['close_intent']*100:>8.0f}% "
          f"{100*r['hold']/max(r['n_exec'],1):>7.0f}% {r['grip_min']:>7.3f} "
          f"{r['grip_final']:>9.3f} {r['net']:>9.0f} {r['dist_stamp']:>8.0f} "
          f"({end[0]:>4.0f},{end[1]:>4.0f},{end[2]:>4.0f})")

print("-" * 150)
print(f"印章中心: ({STAMP[0]:.0f},{STAMP[1]:.0f},{STAMP[2]:.0f}) mm")
print("闭合意图 = query 中 chunk 夹爪<0.02 的比例 (越接近100%越健康)")
print("闭合帧% = 执行帧 grip<0.033 占比 (夹爪实际保持闭合的时间)")
