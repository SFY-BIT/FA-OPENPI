#!/usr/bin/env python
"""MoE 专家路由分析 —— 参考 ForceVLA analysis.py，本地加载模型和数据集直接分析。

用法:
  cd /mnt/hdd/sfy/openpi-force
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python -u scripts/analyze_expert_routing_client.py \
      --checkpoint checkpoints/12000 \
      --config pi05_force_stamp_seal_remote \
      --dataset /mnt/hdd/sfy/lerobot_datasets/stamp_seal_flexiv
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import moe_routing_capture as _routing
from openpi.shared import normalize as _normalize
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_force_stamp_seal_remote")
    p.add_argument("--checkpoint", default="checkpoints/12000")
    p.add_argument("--dataset", default="/mnt/hdd/sfy/lerobot_datasets/stamp_seal_flexiv")
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=0, help="0=all per episode")
    p.add_argument("--norm-stats", default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv")
    args = p.parse_args()

    # ── 加载数据集 ──
    print(f"Loading dataset: {args.dataset}")
    ds = lerobot_dataset.LeRobotDataset(args.dataset)
    meta = lerobot_dataset.LeRobotDatasetMetadata(args.dataset)
    ep_idx = ds.episode_data_index
    eps_total = len(ep_idx["from"])
    n_eps = min(args.num_episodes, eps_total)
    eps = list(range(n_eps))
    print(f"  Total episodes: {eps_total}, sampling first {n_eps}")

    # ── 加载 norm_stats ──
    norm_stats = None
    ns_path = Path(args.norm_stats) / "norm_stats.json"
    if ns_path.exists():
        norm_stats = _normalize.load(str(args.norm_stats))
        print(f"  norm_stats loaded from {args.norm_stats}")
    else:
        print(f"  WARNING: norm_stats not found at {ns_path}")

    # ── 加载策略 ──
    # ⚠️ enable() 必须在 create_trained_policy() 之前！
    # 因为 model.load() 内部会用 nnx.eval_shape 做抽象求值，
    # 此时就会 trace _scatter_to_experts，决定是否包含 callback。
    print(f"Loading policy: {args.config} (routing capture ENABLED)")
    sys.stdout.flush()
    _routing.enable()
    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        cfg, args.checkpoint, norm_stats=norm_stats,
    )
    print("Policy loaded. Running first JIT inference...\n")
    sys.stdout.flush()

    # ── 逐帧分析 ──
    K_FORCE = 2  # force_history_frames
    V_TOKENS = 256  # SigLIP per image
    all_ep_data = []
    total_frames = 0

    for ep in eps:
        s = int(ep_idx["from"][ep])
        e = int(ep_idx["to"][ep])
        length = e - s
        max_f = length if args.max_frames <= 0 else min(args.max_frames, length)
        step = max(1, length // max_f)
        idxs = list(range(s, e, step))[:max_f]
        recs = []

        print(f"Episode {ep}: {length} frames, sampling {len(idxs)} frames")
        sys.stdout.flush()

        for i, fi in enumerate(idxs):
            if i % 50 == 0 and i > 0:
                print(f"  ... {i}/{len(idxs)}", end="\r")
                sys.stdout.flush()

            frm = ds[fi]
            # ForceInStatePiperInputs 期望的键名
            state_raw = np.asarray(frm["observation.state"], dtype=np.float32)
            obs = {
                "observation/state": state_raw,
                "observation/image": np.asarray(frm["observation.image"]),
                "observation/wrist_image": np.asarray(frm["observation.wrist_image"]),
                "prompt": "stamp seal",
            }

            with _routing.frame() as rr:
                policy.infer(obs)

            if not rr:
                continue

            r = rr[0]
            eids = np.asarray(r["expert"][0]).astype(int)
            sl = int(r["seq_length"])

            # 按模态统计
            nv = V_TOKENS * 2  # base + wrist = 512 visual tokens
            l_end = sl - K_FORCE

            recs.append({
                "fi": fi,
                "rp": (fi - s) / length,
                "sl": sl,
                "ec_all": [int((eids[:sl] == e).sum()) for e in range(4)],
                "ec_vis": [int((eids[:nv] == e).sum()) for e in range(4)],
                "ec_lang": [int((eids[nv:l_end] == e).sum()) for e in range(4)] if l_end > nv else [0]*4,
                "ec_force": [int((eids[l_end:sl] == e).sum()) for e in range(4)],
                "fe": int(eids[sl - 1]),
                "fe0": int(eids[sl - 2]) if K_FORCE >= 2 else -1,
            })
            total_frames += 1

        all_ep_data.append(recs)
        if len(idxs) >= 50:
            print(f"  Episode {ep}: done ({len(recs)} frames)" + " " * 20)
        else:
            print(f"  Episode {ep}: done ({len(recs)} frames)")

    ckpt_step = Path(args.checkpoint).name

    # ── 打印结果 ──
    print(f"\n{'=' * 72}")
    print(f"  MoE Expert Routing Analysis")
    print(f"  Checkpoint: {args.checkpoint} (step {ckpt_step})")
    print(f"  Total frames: {total_frames}  |  Episodes: {n_eps}")
    print(f"  Token layout: V_base(256) + V_wrist(256) + L(~3) + F(2)")
    print(f"{'=' * 72}")

    all_ec_vis = np.array([r["ec_vis"] for ep in all_ep_data for r in ep])
    all_ec_force = np.array([r["ec_force"] for ep in all_ep_data for r in ep])
    all_ec_lang = np.array([r["ec_lang"] for ep in all_ep_data for r in ep])

    mv = all_ec_vis.mean(0); mf = all_ec_force.mean(0); ml = all_ec_lang.mean(0)
    tv = mv.sum(); tf = mf.sum(); tl = ml.sum()

    print(f"\n  {'Modality':<14} {'Tokens/frame':>13}  {'E0':>7} {'E1':>7} {'E2':>7} {'E3':>7}")
    print(f"  {'─' * 14} {'─' * 13}  {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}")
    if tv > 0:
        print(f"  {'Vision':<14} {tv/len(all_ec_vis):>13.0f}  {mv[0]/tv*100:>6.1f}% {mv[1]/tv*100:>6.1f}% {mv[2]/tv*100:>6.1f}% {mv[3]/tv*100:>6.1f}%")
    if tl > 0:
        print(f"  {'Language':<14} {tl/len(all_ec_lang):>13.0f}  {ml[0]/tl*100:>6.1f}% {ml[1]/tl*100:>6.1f}% {ml[2]/tl*100:>6.1f}% {ml[3]/tl*100:>6.1f}%")
    if tf > 0:
        print(f"  {'Force':<14} {tf/len(all_ec_force):>13.0f}  {mf[0]/tf*100:>6.1f}% {mf[1]/tf*100:>6.1f}% {mf[2]/tf*100:>6.1f}% {mf[3]/tf*100:>6.1f}%")

    # Force token 路由详情
    all_fe = [r["fe"] for ep in all_ep_data for r in ep]
    all_fe0 = [r["fe0"] for ep in all_ep_data for r in ep if r["fe0"] >= 0]
    fc = Counter(all_fe)
    fc0 = Counter(all_fe0)
    print(f"\n  Force token routing (F1, last):")
    for e in range(4):
        c = fc.get(e, 0)
        if c > 0:
            print(f"    ->E{e}: {c}/{len(all_fe)} ({c/len(all_fe)*100:.1f}%)")
    if fc0:
        print(f"  Force token routing (F0, second-last):")
        for e in range(4):
            c = fc0.get(e, 0)
            if c > 0:
                print(f"    ->E{e}: {c}/{len(all_fe0)} ({c/len(all_fe0)*100:.1f}%)")

    # 每 episode 详情
    for ep_i, recs in enumerate(all_ep_data):
        print(f"\n  {'─' * 56}")
        print(f"  Episode {eps[ep_i]} ({len(recs)} frames)")
        ec_all = np.array([r["ec_all"] for r in recs])
        mn = ec_all.mean(0)
        tot = mn.sum()
        for e in range(4):
            print(f"    E{e}: {mn[e]/tot*100:5.1f}% ({mn[e]:.0f}/{tot:.0f})")
        fec_ep = Counter(r["fe"] for r in recs)
        print(f"    Force F1: " + " | ".join(f"E{e}={fec_ep.get(e,0)}" for e in range(4) if fec_ep.get(e, 0)))

    # 结论
    print(f"\n{'=' * 72}")
    print(f"  CONCLUSION:")
    if tf > 0 and tv > 0:
        fp = [mf[e]/tf*100 for e in range(4)]
        vp = [mv[e]/tv*100 for e in range(4)]
        max_diff = max(abs(fp[e] - vp[e]) for e in range(4))
        print(f"  Force tokens:  E0={fp[0]:.0f}% E1={fp[1]:.0f}% E2={fp[2]:.0f}% E3={fp[3]:.0f}%")
        print(f"  Vision tokens: E0={vp[0]:.0f}% E1={vp[1]:.0f}% E2={vp[2]:.0f}% E3={vp[3]:.0f}%")
        if max_diff > 30:
            print(f"  Modality separation detected (max diff={max_diff:.0f}%)")
        else:
            print(f"  No significant modality separation (max diff={max_diff:.0f}%)")
            print(f"  -> MoE experts are functionally redundant for this task")
            print(f"  -> Recommend: remove MoE, use dense MLP")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
