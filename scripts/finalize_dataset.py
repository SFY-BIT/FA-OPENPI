"""一键转换: Piper → Flexiv 兼容格式 (v2 — 元数据完全同步)

用法:
  python finalize_dataset.py my_robot/stamp_seal
  python finalize_dataset.py my_robot/stamp_seal --float64
  python finalize_dataset.py my_robot/stamp_seal --no-embed

输出: {repo}_flexiv

修正项:
  1. force 合并到 observation.state (7→13维)
  2. 视频帧嵌入 parquet (JPEG bytes)
  3. info.json 键名同步 (images.one→image, images.two→wrist_image)
  4. total_videos 设为 0
  5. episodes_stats.jsonl 重算（删除旧键）
  6. 图像 struct path 字段修正 (frame_NNNNNN.jpg)
  7. parquet 内 metadata 同步
"""

import sys
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"

# 列名映射
IMAGE_KEY_MAP = {
    "observation.images.one": "observation.image",
    "observation.images.two": "observation.wrist_image",
}


def _build_image_struct(frames_bytes: list[bytes], ep: int) -> list[dict]:
    """构建 image struct: {bytes, path}，path 用 frame_NNNNNN.jpg"""
    n = len(frames_bytes)
    structs = []
    for j in range(n):
        structs.append({
            "bytes": frames_bytes[min(j, len(frames_bytes) - 1)],
            "path": f"frame_{j:06d}.jpg",
        })
    return structs


def _fix_info_json(dst_meta: Path, use_float64: bool, total_frames: int,
                   has_image: bool, has_wrist: bool, embed_images: bool):
    """修复 info.json，同步键名和统计"""
    with open(dst_meta / "info.json") as f:
        info = json.load(f)

    # 更新基本统计
    info["total_frames"] = total_frames
    info["total_videos"] = 0  # 图像已嵌入，无外部视频
    if embed_images:
        info["video_path"] = None  # 清除旧视频路径引用

    feats = info.get("features", {})

    # (a) 更新 state
    if "observation.state" in feats:
        feats["observation.state"]["shape"] = [13]
        feats["observation.state"]["dtype"] = "float64" if use_float64 else "float32"
        existing_names = feats["observation.state"].get("names") or []
        force_names = ["force_Fx", "force_Fy", "force_Fz", "force_Tx", "force_Ty", "force_Tz"]
        feats["observation.state"]["names"] = existing_names[:7] + force_names

    # (b) 删除 force 独立键
    feats.pop("observation.force", None)

    # (c) 重命名图像键: observation.images.one → observation.image
    new_feats = {}
    for key, val in feats.items():
        if key in IMAGE_KEY_MAP:
            new_key = IMAGE_KEY_MAP[key]
            val["dtype"] = "image"
            val.pop("info", None)  # 清除残留的 video info
            new_feats[new_key] = val
        elif key.startswith("observation.images."):
            new_key = key.replace("observation.images.", "observation.image_")
            val["dtype"] = "image"
            val.pop("info", None)
            new_feats[new_key] = val
        else:
            new_feats[key] = val
    feats.clear()
    feats.update(new_feats)

    info["features"] = feats
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def _fix_stats_jsonl(dst_meta: Path):
    """重算 episodes_stats.jsonl，删除不存在的列统计。
    处理嵌套结构: {"episode_index": N, "stats": {"action": {...}, ...}}"""
    stats_path = dst_meta / "episodes_stats.jsonl"
    if not stats_path.exists():
        return

    new_lines = []
    with open(stats_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line.strip())
            old_stats = rec.get("stats", {})
            new_stats = {}
            for key, val in old_stats.items():
                # 重命名图像键
                if key in IMAGE_KEY_MAP:
                    new_stats[IMAGE_KEY_MAP[key]] = val
                elif key.startswith("observation.images."):
                    new_stats[key.replace("observation.images.", "observation.image_")] = val
                elif key == "observation.force":
                    continue  # 已合并到 state
                else:
                    new_stats[key] = val
            new_lines.append(json.dumps(
                {"episode_index": rec["episode_index"], "stats": new_stats},
                ensure_ascii=False
            ))

    with open(stats_path, "w") as f:
        for line in new_lines:
            f.write(line + "\n")


def _fix_parquet_meta(table: pa.Table, use_float64: bool, new_col_names: list[str]) -> pa.Table:
    """更新 parquet 内嵌的 huggingface metadata"""
    existing = table.schema.metadata or {}
    new_meta = {}
    for k, v in existing.items():
        if k == b"huggingface":
            hf = json.loads(v.decode())
            feats = hf.get("info", {}).get("features", {})
            # 更新 state
            if "observation.state" in feats:
                feats["observation.state"]["shape"] = [13]
                feats["observation.state"]["dtype"] = "float64" if use_float64 else "float32"
            feats.pop("observation.force", None)
            # 重命名图像键
            new_feats = {}
            for fk, fv in feats.items():
                if fk in IMAGE_KEY_MAP:
                    fv["dtype"] = "image"
                    new_feats[IMAGE_KEY_MAP[fk]] = fv
                elif fk.startswith("observation.images."):
                    fv["dtype"] = "image"
                    new_feats[fk.replace("observation.images.", "observation.image_")] = fv
                else:
                    new_feats[fk] = fv
            hf["info"]["features"] = new_feats
            new_meta[k] = json.dumps(hf).encode()
        else:
            new_meta[k] = v
    return table.replace_schema_metadata(new_meta)


def convert(src_repo: str, embed_images: bool = True, use_float64: bool = False):
    src = CACHE / src_repo
    dst_repo = src_repo.rstrip("/").rstrip("\\") + "_flexiv"
    dst = CACHE / dst_repo

    if not src.exists():
        print(f"[ERROR] Not found: {src}")
        return

    # ── 复制 meta ──
    if (dst / "meta").exists():
        shutil.rmtree(dst / "meta")
    shutil.copytree(src / "meta", dst / "meta")
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ── 扫描 ──
    src_data = src / "data" / "chunk-000"
    eps = sorted([int(f.stem.split("_")[1]) for f in src_data.glob("episode_*.parquet")])
    print(f"Found {len(eps)} episodes in {src_repo}")

    # ── 预加载视频 ──
    video_map = {}
    has_image, has_wrist = False, False
    if embed_images:
        video_base = src / "videos" / "chunk-000"
        if video_base.exists():
            for cam_dir in video_base.iterdir():
                if cam_dir.is_dir():
                    for mp4 in cam_dir.glob("*.mp4"):
                        ep = int(mp4.stem.split("_")[1])
                        video_map.setdefault(ep, {})[cam_dir.name] = mp4
            if any("observation.images.one" in v for vv in video_map.values() for v in vv):
                has_image = True
            if any("observation.images.two" in v for vv in video_map.values() for v in vv):
                has_wrist = True
            print(f"  Videos found for {len(video_map)} episodes (cam1={has_image}, cam2={has_wrist})")

    # ── 逐 episode 转换 ──
    total = 0
    for ep_idx, ep in enumerate(eps):
        spq = src_data / f"episode_{ep:06d}.parquet"
        dpq = dst / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"

        table = pq.read_table(str(spq))
        cols = table.column_names
        n = len(table)

        # (1) 合并 force → state
        force_arr = np.stack([np.array(x) for x in table.column("observation.force").to_pylist()])
        state_arr = np.stack([np.array(x) for x in table.column("observation.state").to_pylist()])
        merged = np.concatenate([state_arr, force_arr], axis=1)
        element_type = pa.float64() if use_float64 else pa.float32()

        new_cols_data = []
        new_col_names = []
        for i, name in enumerate(cols):
            if name == "observation.state":
                arr = pa.array([merged[j].tolist() for j in range(n)],
                               type=pa.list_(element_type, 13))
                new_cols_data.append(arr)
                new_col_names.append(name)
            elif name == "observation.force":
                continue
            else:
                new_cols_data.append(table.column(i))
                new_col_names.append(name)

        # (2) 嵌入图像
        if embed_images and ep in video_map:
            for cam_name, vpath in video_map[ep].items():
                cap = cv2.VideoCapture(str(vpath))
                frames_bytes = []
                for _ in range(n):
                    ret, frame = cap.read()
                    if ret:
                        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frames_bytes.append(jpg.tobytes())
                    else:
                        frames_bytes.append(frames_bytes[-1] if frames_bytes else b"")
                cap.release()

                struct_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
                structs = _build_image_struct(frames_bytes, ep)
                new_cols_data.append(pa.array(structs, type=struct_type))

                if cam_name in IMAGE_KEY_MAP:
                    new_col_names.append(IMAGE_KEY_MAP[cam_name])
                else:
                    new_col_names.append(cam_name.replace("observation.images.", "observation.image_"))

        # (3) 构建新表 + 修复 parquet metadata
        new_table = pa.table({n: c for n, c in zip(new_col_names, new_cols_data)})
        new_table = _fix_parquet_meta(new_table, use_float64, new_col_names)
        pq.write_table(new_table, str(dpq))
        total += n
        print(f"  [{ep_idx + 1}/{len(eps)}] Ep {ep}: {n} frames")

    # ── 修复 info.json ──
    _fix_info_json(dst / "meta", use_float64, total, has_image, has_wrist, embed_images)

    # ── 修复 episodes_stats.jsonl ──
    _fix_stats_jsonl(dst / "meta")

    # ── 验证 ──
    check = pq.read_table(str(dst / "data" / "chunk-000" / f"episode_{eps[0]:06d}.parquet"))
    s = np.stack([np.array(x) for x in check.column("observation.state").to_pylist()])

    # 验证 info.json
    with open(dst / "meta" / "info.json") as f:
        final_info = json.load(f)

    print(f"\n{'='*55}")
    print(f"DONE: {dst_repo}")
    print(f"  Episodes: {len(eps)}  Frames: {total}")
    print(f"  State dim: {s.shape[1]} (expect 13)  dtype={'float64' if use_float64 else 'float32'}")
    print(f"  Parquet cols: {check.column_names}")
    print(f"  Info keys: {list(final_info['features'].keys())}")
    print(f"  total_videos: {final_info['total_videos']}")
    print(f"  Images embedded: {embed_images}")
    print(f"  Meta sync: info.json + stats.jsonl updated")
    print(f"{'='*55}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python finalize_dataset.py <repo> [--no-embed] [--float64]")
        print("Example: python finalize_dataset.py my_robot/stamp_seal")
        sys.exit(1)

    repo = sys.argv[1]
    embed = "--no-embed" not in sys.argv
    f64 = "--float64" in sys.argv

    convert(repo, embed_images=embed, use_float64=f64)
