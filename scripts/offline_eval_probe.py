"""离线评估准备②: 探测 4 个 peel server 的 websocket 通信。

对每个 port 用 eval 数据集一帧真实观测 (state7 + wrench_history60x6 + 图像),
发送请求, 验证返回 actions/force_pred。EEF server 会自动 FK/IK 转换。
"""
import sys
import glob
import numpy as np
from collections import deque
from pathlib import Path

import pyarrow.parquet as pq

# openpi_client (msgpack) — 与 record_ws_dataset.py 相同引入方式
for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    cand = parent / "packages" / "src"
    if (cand / "openpi_client").is_dir():
        sys.path.insert(0, str(cand))
        break

import websockets.sync.client
from openpi_client import msgpack_numpy

EVAL_DS = "/mnt/hdd/sfy/FA-VLA-结论/checkpoints/test_peel/peel_cucumber_eval_dataset"
FT_HISTORY_STEPS = 60
FORCE_DIM = 6

PORTS = {8000: "peel_05_joint", 8001: "peel_FA_joint", 8002: "peel_05_eef", 8003: "peel_FA_eef"}


def build_wrench_history(force_buf: deque) -> np.ndarray:
    frames = list(force_buf)
    T_avail = len(frames)
    if T_avail == 0:
        return np.zeros((FT_HISTORY_STEPS, FORCE_DIM), dtype=np.float32)
    if T_avail >= FT_HISTORY_STEPS:
        return np.stack(frames[-FT_HISTORY_STEPS:], axis=0)
    pad = np.zeros((FT_HISTORY_STEPS - T_avail, FORCE_DIM), dtype=np.float32)
    return np.concatenate([pad, np.stack(frames, axis=0)], axis=0)


def decode_img(d):
    from PIL import Image
    import io
    if isinstance(d, dict):
        b = d.get("bytes")
        if not b:
            return np.zeros((480, 640, 3), np.uint8)
        d = b
    try:
        return np.asarray(Image.open(io.BytesIO(d)).convert("RGB"))
    except Exception:
        return np.zeros((480, 640, 3), np.uint8)


def main():
    f = sorted(glob.glob(f"{EVAL_DS}/data/chunk-000/episode_*.parquet"))[0]
    t = pq.read_table(f)
    st = np.stack(t.column("observation.state").to_pylist()).astype(np.float32)
    fc = np.stack(t.column("observation.force").to_pylist()).astype(np.float32)
    img = t.column("observation.images.one").to_pylist()
    wrist = t.column("observation.images.two").to_pylist()

    start = 100  # 中段有接触的帧
    force_buf = deque(maxlen=FT_HISTORY_STEPS)
    for i in range(max(0, start - FT_HISTORY_STEPS), start):
        force_buf.append(fc[i])
    obs = {
        "observation/image": decode_img(img[start]),
        "observation/wrist_image": decode_img(wrist[start]),
        "observation/state": st[start, :7].tolist(),
        "observation/wrench_history": build_wrench_history(force_buf),
        "prompt": "peel cucumber",
    }
    packer = msgpack_numpy.Packer()

    print(f"帧 {start}: state[:3]={np.round(st[start,:3],3)} |F|={np.linalg.norm(fc[start]):.2f}N")
    for port, name in PORTS.items():
        try:
            with websockets.sync.client.connect(f"ws://127.0.0.1:{port}") as ws:
                meta = msgpack_numpy.unpackb(ws.recv())  # 先收 metadata
                ws.send(packer.pack(obs))
                raw = ws.recv()
                resp = msgpack_numpy.unpackb(raw)
            act = np.asarray(resp["actions"])
            fp = resp.get("force_pred")
            msg = f"  OK {name:<14} port={port}: actions{act.shape}"
            if fp is not None:
                fp = np.asarray(fp)
                msg += f" force_pred{np.shape(fp)}"
            print(msg)
        except Exception as e:
            print(f"  XX {name:<14} port={port}: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()