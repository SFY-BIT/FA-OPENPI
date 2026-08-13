#!/usr/bin/env python3
"""真机 log 回放推理测试 — 对比不同 checkpoint 在同一真机轨迹上的推理输出。

用途:
  把真机测试 log (含 state/force/image) 重建为模型输入, 用不同 checkpoint 推理,
  对比动作输出差异 → 定位"哪个 checkpoint / 参数调整"改变了行为。

用法:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/replay_log_infer.py \
    --log img/eef_2w_toggle_img_20260810_152631.jsonl \
    --checkpoints EEF/4500 EEF/10000 EEF/8000 \
    --config pi05_force_erase_board_eef \
    --norm-stats datasets/erase_board_flexiv_ft60 \
    --output-dir outputs/replay
"""
import argparse
import base64
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.shared import normalize as _normalize
from PIL import Image

FT_HISTORY_STEPS = 60  # 60 帧力历史
STATE_DIM = 7          # 6 关节 + 夹爪


def decode_image(b64: str) -> np.ndarray:
    if not b64:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    img = Image.open(__import__("io").BytesIO(base64.b64decode(b64)))
    img = img.resize((224, 224))
    return np.asarray(img, dtype=np.uint8)


def build_wrench_history(force_list: list) -> np.ndarray:
    """把逐帧 force [6] 累积成 60 帧历史 [60, 6] (最新在末尾, 开头补零)。"""
    n = len(force_list)
    if n >= FT_HISTORY_STEPS:
        return np.stack(force_list[-FT_HISTORY_STEPS:]).astype(np.float32)
    # 开头补零
    pad = FT_HISTORY_STEPS - n
    zeros = np.zeros((pad, 6), dtype=np.float32)
    return np.concatenate([zeros, np.stack(force_list)]).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Replay real-robot log through checkpoints")
    p.add_argument("--log", required=True, help="真机 log jsonl")
    p.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint 目录 (如 EEF/4500)")
    p.add_argument("--config", default="pi05_force_erase_board_eef")
    p.add_argument("--norm-stats", default="datasets/erase_board_flexiv_ft60")
    p.add_argument("--output-dir", default="outputs/replay")
    p.add_argument("--sample-step", type=int, default=30, help="每隔多少帧推理一次 (默认30=1s)")
    p.add_argument("--start", type=int, default=60, help="起始帧 (需≥60 凑满力历史)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载真机 log ──
    lines = [json.loads(l) for l in open(args.log)]
    n = len(lines)
    print(f"Log: {args.log}, {n} frames")
    states = [np.asarray(l["state"], dtype=np.float32) for l in lines]
    forces = [np.asarray(l["force"], dtype=np.float32) for l in lines]
    images = [decode_image(l.get("image_main", "")) for l in lines]
    modes = [l.get("mode", "?") for l in lines]

    # ── 加载 norm_stats ──
    norm_stats = _normalize.load(args.norm_stats)
    norm_stats.pop("force_target", None)
    norm_stats.pop("ft_state", None)

    # ── 加载所有 checkpoint ──
    policies = {}
    for ck in args.checkpoints:
        ck_path = Path("checkpoints") / ck
        if not ck_path.exists():
            ck_path = Path(ck)
        print(f"Loading policy {ck} ...", flush=True)
        cfg = _config.get_config(args.config)
        policies[ck] = _policy_config.create_trained_policy(
            cfg, ck_path, norm_stats=norm_stats
        )
        # 单 checkpoint 模式: 跑完立即释放, 避免多模型 OOM
        if len(policies) == 1 and len(args.checkpoints) > 1:
            print("  [single mode] run this checkpoint then exit (use --checkpoints X once per run)")
    print(f"Loaded {len(policies)} policies")

    # ── 逐帧推理对比 ──
    results = {ck: [] for ck in policies}
    idxs = list(range(max(args.start, 60), n, args.sample_step))

    for i in idxs:
        # 构建输入
        obs = {
            "observation/state": states[i],
            "observation/image": images[i],
            "observation/wrist_image": images[i],  # 复用主相机
            "observation/wrench_history": build_wrench_history(forces[: i + 1]),
            "prompt": "erase the board",
        }
        for ck, policy in policies.items():
            try:
                out = policy.infer(obs)
                # 提取动作输出
                act = out["actions"]  # [30, 7] 绝对
                results[ck].append({
                    "fi": i,
                    "mode": modes[i],
                    "state_q4": float(states[i][3]),
                    "act_q4_first": float(act[0, 3]),
                    "act_q4_last": float(act[-1, 3]),
                    "act_delta_q4": float(act[-1, 3] - act[0, 3]),
                    "state_q5": float(states[i][4]),
                    "act_q5_first": float(act[0, 4]),
                    "act_delta_q5": float(act[-1, 4] - act[0, 4]),
                })
            except Exception as e:
                results[ck].append({"fi": i, "error": str(e)})
        if (i - idxs[0]) % 300 == 0:
            print(f"  processed {i}/{n}", flush=True)

    # ── 输出对比 ──
    print("\n=== Checkpoint 推理对比 (真机轨迹回放) ===")
    print(f"{'fi':>5} | {'mode':>6} | {'state_q4':>8} | ", end="")
    for ck in policies:
        print(f"{'act_q4Δ ' + ck:>22} | ", end="")
    print()
    for k in range(len(idxs)):
        fi = idxs[k]
        print(f"{fi:5d} | {modes[fi]:>6} | {states[fi][3]*180/np.pi:8.1f}° | ", end="")
        for ck in policies:
            r = results[ck][k]
            if "act_delta_q4" in r:
                print(f"{(r['act_delta_q4'])*180/np.pi:8.1f}° ({r['act_q4_first']*180/np.pi:6.1f}→{r['act_q4_last']*180/np.pi:6.1f})", end=" | ")
            else:
                print(f"{'ERR':>22}", end=" | ")
        print()

    # 保存 JSON
    out_file = out_dir / f"replay_{Path(args.log).stem}.json"
    with open(out_file, "w") as f:
        json.dump({"idxs": idxs, "results": results}, f, indent=1)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
