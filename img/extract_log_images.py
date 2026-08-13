#!/usr/bin/env python3
"""从 client 日志 (JSONL) 提取图片序列 → 视频/图片文件

日志需用 --save-images 录制 (每帧含 image_main/image_wrist base64)。

用法:
  python extract_log_images.py <日志.jsonl> [--out DIR] [--cam main|wrist|both] [--video]

选项:
  --out DIR    输出目录 (默认: 日志名同目录下 <name>_images/)
  --cam 哪个相机 (main/wrist/both, 默认 both)
  --video      生成 mp4 视频 (需要 ffmpeg 或 cv2 VideoWriter)
  --fps 30     视频帧率 (默认 30)
  --step N     每隔 N 帧取一帧 (默认 1, 全部)

示例:
  python extract_log_images.py eef_2w_toggle_img_20260810_153424.jsonl --video
  python extract_log_images.py eef_2w_toggle_img_20260810_153424.jsonl --cam main --step 2
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def decode_b64(b64: str) -> np.ndarray | None:
    """base64 → BGR ndarray"""
    if not b64:
        return None
    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Extract images from client JSONL log")
    p.add_argument("log", help="日志文件 (jsonl)")
    p.add_argument("--out", default="", help="输出目录 (默认 <log>_images/)")
    p.add_argument("--cam", default="both", choices=["main", "wrist", "both"])
    p.add_argument("--video", action="store_true", help="生成 mp4 视频")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--step", type=int, default=1, help="每隔 N 帧取一帧")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"日志不存在: {log_path}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else Path(f"{log_path.stem}_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = ["main", "wrist"] if args.cam == "both" else [args.cam]

    # 统计
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    total = len(lines)
    print(f"日志: {log_path.name} ({total} 行)")

    # 视频 writer
    writers = {}
    frame_size = None
    frame_count = 0
    saved = 0

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue

        # 只处理含图片的记录 (推理帧/遥控帧都可能)
        if not any(rec.get(f"image_{c}") for c in cams):
            continue
        if i % args.step != 0:
            continue

        for cam in cams:
            b64 = rec.get(f"image_{cam}")
            img = decode_b64(b64) if b64 else None
            if img is None:
                continue

            if args.video:
                if cam not in writers:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vp = out_dir / f"{cam}.mp4"
                    writers[cam] = cv2.VideoWriter(str(vp), fourcc, args.fps, (w, h))
                writers[cam].write(img)
            else:
                fname = out_dir / f"{cam}_{i:06d}.jpg"
                cv2.imwrite(str(fname), img)
                saved += 1

        frame_count += 1
        if frame_count % 200 == 0:
            print(f"  已处理 {frame_count} 帧...")

    for w in writers.values():
        w.release()

    if args.video:
        for cam in cams:
            vp = out_dir / f"{cam}.mp4"
            print(f"✅ 视频: {vp} ({frame_count} 帧 @ {args.fps}fps)")
    else:
        print(f"✅ 图片: {saved} 张 → {out_dir}/")


if __name__ == "__main__":
    main()
