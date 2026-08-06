#!/usr/bin/env python3
"""离线回放测试 — 用数据集真实轨迹喂 policy server，对比预测关节 vs 真实关节。

从 stamp_seal_v2_flexiv_ft60 数据集读取一条真实轨迹，逐帧把
  observation/image + observation/wrist_image + observation/state + wrench_history + prompt
发送给 websocket policy server，读回预测 action（绝对关节位置，7 维），
与数据集真实 action 对比，监测点：关节数值（6 关节 + 夹爪）。

用法:
  # 1) 先启动 server (见 serve_policy.py)
  # 2) 运行本脚本
  conda activate rlinf
  cd /mnt/hdd/sfy/FA-openpi
  python -u scripts/offline_replay_client.py \
      --host 127.0.0.1 --port 8000 \
      --dataset /mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60 \
      --episode 0 --max-frames 300 \
      --output-dir outputs/offline_replay
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent
_OPENPI_CLIENT_SRC = _OPENPI_ROOT / "packages" / "openpi-client" / "src"

sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

import websockets.sync.client
from openpi_client import msgpack_numpy

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from PIL import Image


def decode_image(img_data) -> np.ndarray:
    """Decode dataset image (dict with bytes/path, or bytes, or ndarray) → RGB uint8."""
    if isinstance(img_data, dict):
        for k in ("bytes", "path"):
            if k in img_data and img_data[k] is not None:
                v = img_data[k]
                img = Image.open(__import__("io").BytesIO(v)) if isinstance(v, bytes) else Image.open(v)
                return np.asarray(img.convert("RGB"), dtype=np.uint8)
    if isinstance(img_data, bytes):
        return np.asarray(Image.open(__import__("io").BytesIO(img_data)).convert("RGB"), dtype=np.uint8)
    if hasattr(img_data, "numpy"):
        arr = img_data.numpy() if callable(img_data.numpy) else np.asarray(img_data)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return np.asarray(arr, dtype=np.uint8)
    if isinstance(img_data, np.ndarray):
        if img_data.ndim == 3 and img_data.shape[0] == 3:
            img_data = img_data.transpose(1, 2, 0)
        return img_data.astype(np.uint8)
    raise TypeError(f"Unknown image type: {type(img_data)}")


def connect_server(host: str, port: int):
    uri = f"ws://{host}:{port}"
    for attempt in range(60):
        try:
            conn = websockets.sync.client.connect(
                uri, compression=None, max_size=None,
                ping_interval=None, ping_timeout=None,
            )
            metadata = msgpack_numpy.unpackb(conn.recv())
            print(f"[connect] server metadata: {metadata}")
            return conn
        except (ConnectionRefusedError, OSError):
            if attempt == 0:
                print(f"[connect] Waiting for server at {uri} ...")
            time.sleep(2)
    raise RuntimeError(f"Server not reachable after 60 attempts: {uri}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dataset", default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=300, help="<=0 表示整条轨迹")
    p.add_argument("--sample-stride", type=int, default=1, help="每 N 帧推理一次")
    p.add_argument("--prompt", default="stamp seal")
    p.add_argument("--output-dir", default="/mnt/hdd/sfy/FA-openpi/outputs/offline_replay")
    p.add_argument("--wrench-key", default="observation.wrench_history",
                   help="60 帧力历史的列名")
    p.add_argument("--ablate", choices=["none", "vision", "force"], default="none",
                   help="消融模式: none=全都有, vision=缺少视觉(图像置零), "
                        "force=缺少力觉(wrench_history+state力置零)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据集 ──
    print(f"[data] Loading dataset: {args.dataset}")
    ds = lerobot_dataset.LeRobotDataset(args.dataset)
    ds.hf_dataset.set_format("python")
    ep_idx = ds.episode_data_index
    s, e = int(ep_idx["from"][args.episode]), int(ep_idx["to"][args.episode])
    length = e - s
    max_f = length if args.max_frames <= 0 else min(args.max_frames, length)
    idxs = list(range(s, e, args.sample_stride))[:max_f]
    print(f"[data] Ep{args.episode}: {length} frames, replay {len(idxs)}")

    # ── 连接 server ──
    conn = connect_server(args.host, args.port)
    packer = msgpack_numpy.Packer()

    # ── 回放 ──
    results = []          # per-frame records
    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    print(f"[replay] {'step':>5} {'progress':>8} | " + " ".join(f"{n:>7}" for n in joint_names) + " | err_norm")
    print("-" * 100)

    for i, fi in enumerate(idxs):
        frm = ds.hf_dataset[fi]
        # 真实观测
        img = decode_image(frm["observation.image"])
        wrist = decode_image(frm["observation.wrist_image"])
        state = np.asarray(frm["observation.state"], dtype=np.float32)  # 13: 7 joints + 6 force
        action_gt = np.asarray(frm["action"], dtype=np.float32)         # 7: absolute joints

        # wrench_history [60,6]（若为一维则 reshape）
        wh = np.asarray(frm[args.wrench_key], dtype=np.float32)
        if wh.ndim == 1:
            wh = wh.reshape(-1, 6)
        if wh.shape[0] < 60:
            pad = np.zeros((60 - wh.shape[0], 6), dtype=np.float32)
            wh = np.concatenate([pad, wh], axis=0)
        elif wh.shape[0] > 60:
            wh = wh[-60:]

        # ── 消融遮挡 (client 端输入处理) ──
        if args.ablate == "vision":
            img = np.zeros_like(img)       # 主相机置零
            wrist = np.zeros_like(wrist)   # 腕相机置零
        elif args.ablate == "force":
            wh = np.zeros_like(wh)                          # wrench_history 置零
            state = state.copy()
            state[7:] = 0.0                                  # state 中力部分置零

        obs = {
            "observation/state": state.tolist(),
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/wrench_history": wh,
            "prompt": args.prompt,
        }

        t0 = time.time()
        conn.send(packer.pack(obs))
        raw = conn.recv()
        if isinstance(raw, str):
            print(f"[error] step {i}: server error: {raw}")
            break
        resp = msgpack_numpy.unpackb(raw)
        infer_ms = (time.time() - t0) * 1000

        action_chunk = np.asarray(resp["actions"])          # (30, 7) absolute
        action_pred = action_chunk[0]                        # 当前步预测
        force_pred = np.asarray(resp.get("force_pred", np.zeros((30, 6))))[0] if "force_pred" in resp else None

        # 关节偏差（绝对关节位置差值）
        err = action_pred - action_gt
        err_norm = float(np.linalg.norm(err[:6]))
        err_abs_max = float(np.max(np.abs(err)))

        # 力偏差（预测 vs 真实当前力）
        force_gt = state[7:13]
        force_err = float(np.linalg.norm(force_pred - force_gt)) if force_pred is not None else float("nan")

        results.append({
            "step": i, "frame_idx": int(fi), "progress": round((fi - s) / length, 4),
            "action_gt": action_gt.tolist(), "action_pred": action_pred.tolist(),
            "joint_err": err.tolist(), "joint_err_norm6": err_norm,
            "joint_err_absmax": err_abs_max,
            "force_gt": force_gt.tolist(),
            "force_pred": force_pred.tolist() if force_pred is not None else None,
            "force_err": force_err,
            "infer_ms": infer_ms,
        })

        if i % 20 == 0 or i == len(idxs) - 1:
            print(f"[{i:>5}] {results[-1]['progress']:>8.3f} | "
                  + " ".join(f"{v:>7.4f}" for v in err) + f" | {err_norm:.4f}")

    conn.close()

    # ── 统计 ──
    if not results:
        print("[stat] no frames")
        return

    jerr = np.array([r["joint_err"] for r in results])       # (N, 7)
    per_joint_mae = np.mean(np.abs(jerr), axis=0)
    per_joint_rmse = np.sqrt(np.mean(jerr ** 2, axis=0))
    per_joint_max = np.max(np.abs(jerr), axis=0)
    overall_mae = np.mean(np.abs(jerr))
    norm6 = np.array([r["joint_err_norm6"] for r in results])

    print("\n" + "=" * 70)
    print("  关节偏差统计 (预测绝对关节 − 数据集真实绝对关节)")
    print("=" * 70)
    print(f"{'关节':>8} {'MAE':>10} {'RMSE':>10} {'MAX':>10}")
    for j, name in enumerate(joint_names):
        print(f"{name:>8} {per_joint_mae[j]:>10.5f} {per_joint_rmse[j]:>10.5f} {per_joint_max[j]:>10.5f}")
    print("-" * 70)
    print(f"{'ALL(6关节)':>8} {overall_mae:>10.5f} {'':>10} {np.mean(norm6):>10.5f}")
    print(f"{'MAX-norm':>8} {'':>10} {'':>10} {np.max(norm6):>10.5f}")
    print(f"  平均 6 关节范数误差: {np.mean(norm6):.5f}")
    print(f"  infer 平均耗时: {np.mean([r['infer_ms'] for r in results]):.1f} ms")

    # 保存
    json_path = out_dir / f"replay_ep{args.episode}.json"
    with open(json_path, "w") as f:
        json.dump({"episode": args.episode, "ablate": args.ablate,
                   "joint_names": joint_names,
                   "per_joint_mae": per_joint_mae.tolist(),
                   "per_joint_rmse": per_joint_rmse.tolist(),
                   "per_joint_max": per_joint_max.tolist(),
                   "overall_mae": float(overall_mae),
                   "frames": results}, f, indent=2)
    print(f"\n[save] {json_path}")


if __name__ == "__main__":
    main()
