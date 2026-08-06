#!/usr/bin/env python
"""K=16 版语言 token 子段（task/state/action/padding）expert 路由分析。

基于 openpi-force/moe_analysis_all/scripts/analyze_lang_split.py 适配：
  - K_FORCE = 16（ft_history 分段编码）
  - obs 增加 observation/wrench_history（K=16 模型需要 ft_state）
  - norm_stats 从 ft60 数据集加载并 pop ft_state
  - config/checkpoint/dataset 默认指向 K=16 本地 LoRA

用法:
  cd /mnt/hdd/sfy/FA-openpi
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python -u scripts/analyze_lang_split_k16.py \
      --checkpoint checkpoints/pi05_force_stamp_seal_ft60_forcevla_lora_k16/ft60_k16/2000 \
      --num-episodes 2 --max-frames 200 --output-dir outputs/lang_split_k16_2000
"""

import argparse, io, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import moe_routing_capture as _routing
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize as _normalize
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from PIL import Image

V_TOKENS_PER_IMAGE = 256
NUM_IMAGES = 3  # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb (dummy)
K_FORCE = 16   # ← K=16 分段编码


def decode_image(img_data):
    if isinstance(img_data, dict):
        for k in ("bytes", "path"):
            if k in img_data and img_data[k] is not None:
                v = img_data[k]
                return Image.open(io.BytesIO(v)) if isinstance(v, bytes) else Image.open(v)
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data))
    if hasattr(img_data, "numpy"):
        arr = img_data.numpy() if callable(img_data.numpy) else np.asarray(img_data)
        if arr.ndim == 3 and arr.shape[0] == 3: arr = arr.transpose(1, 2, 0)
        return Image.fromarray(arr.astype(np.uint8))
    if isinstance(img_data, np.ndarray):
        if img_data.ndim == 3 and img_data.shape[0] == 3: img_data = img_data.transpose(1, 2, 0)
        return Image.fromarray(img_data.astype(np.uint8))
    raise TypeError(f"Unknown: {type(img_data)}")


def tokenize_with_boundaries(prompt_text, state, max_len=200):
    """Tokenize and return token boundaries for task vs state sub-segments."""
    tok = PaligemmaTokenizer(max_len=max_len)
    cleaned = prompt_text.strip().replace("_", " ").replace("\n", " ")
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "

    tokenizer = tok._tokenizer
    bos_id = tokenizer.bos_id()
    bos = [bos_id]
    task_prefix = tokenizer.encode("Task: " + cleaned)
    state_sep = tokenizer.encode(", State: ")
    state_tokens = tokenizer.encode(state_str)
    action_suffix = tokenizer.encode(";\nAction: \n")

    all_tokens = bos + task_prefix + state_sep + state_tokens + action_suffix
    all_tokens = all_tokens[:max_len]

    task_start = len(bos)
    task_end = task_start + len(task_prefix)
    state_start = task_end + len(state_sep)
    state_end = state_start + len(state_tokens)
    action_start = state_end

    n_real = len(all_tokens)
    padded = all_tokens + [0] * (max_len - n_real)

    return {
        "tokens": padded,
        "n_real": n_real,
        "n_pad": max_len - n_real,
        "task_start": task_start,
        "task_end": task_end,
        "state_start": state_start,
        "state_end": state_end,
        "action_start": action_start,
        "action_end": n_real,
        "task_tokens": all_tokens[task_start:task_end],
        "state_tokens": all_tokens[state_start:state_end],
        "action_tokens": all_tokens[action_start:],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/pi05_force_stamp_seal_ft60_forcevla_lora_k16/ft60_k16/2000")
    p.add_argument("--config", default="pi05_force_stamp_seal_ft60_forcevla_lora_k16")
    p.add_argument("--dataset", default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60")
    p.add_argument("--norm-stats", default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60")
    p.add_argument("--num-episodes", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--output-dir", default="outputs/lang_split_k16_2000")
    args = p.parse_args()

    norm_stats = _normalize.load(args.norm_stats)
    norm_stats.pop("force_target", None)
    norm_stats.pop("ft_state", None)  # 推理输出无 ft_state

    _routing.enable()
    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=norm_stats)
    policy._has_flow_breakdown = False
    print("Policy loaded. JIT...", flush=True)

    ds = lerobot_dataset.LeRobotDataset(args.dataset)
    ep_idx = ds.episode_data_index
    n_eps = min(args.num_episodes, len(ep_idx["from"]))

    max_token_len = cfg.model.max_token_len
    print(f"max_token_len={max_token_len}, K_FORCE={K_FORCE}")
    print(f"Token layout: V({V_TOKENS_PER_IMAGE*NUM_IMAGES}) + L({max_token_len}) + F({K_FORCE})")

    task_exp = Counter(); state_exp = Counter(); action_exp = Counter(); pad_exp = Counter()
    task_count = 0; state_count = 0; action_count = 0; pad_count = 0

    for ep in range(n_eps):
        s, e = int(ep_idx["from"][ep]), int(ep_idx["to"][ep])
        length = e - s
        max_f = length if args.max_frames <= 0 else min(args.max_frames, length)
        step = max(1, length // max_f)
        idxs = list(range(s, e, step))[:max_f]

        print(f"\nEp{ep}: {len(idxs)} frames"); sys.stdout.flush()

        for i, fi in enumerate(idxs):
            if i > 0 and i % 50 == 0:
                print(f"  {i}/{len(idxs)}", end="\r"); sys.stdout.flush()

            frm = ds[fi]
            state_raw = np.asarray(frm["observation.state"][:7], dtype=np.float32)
            obs = {
                "observation/state": np.asarray(frm["observation.state"], dtype=np.float32),
                "observation/image": np.asarray(decode_image(frm["observation.image"])),
                "observation/wrist_image": np.asarray(decode_image(frm["observation.wrist_image"])),
                "prompt": "stamp seal",
            }
            # K=16 模型需要 wrench_history → ft_state
            if "observation.wrench_history" in frm:
                wh = np.asarray(frm["observation.wrench_history"], dtype=np.float32)
                if wh.ndim == 1:
                    wh = wh.reshape(-1, 6)
                obs["observation/wrench_history"] = wh
            else:
                obs["observation/wrench_history"] = np.zeros((60, 6), dtype=np.float32)

            with _routing.frame() as rr:
                policy.infer(obs)

            if not rr: continue
            r = rr[0]
            eids = np.asarray(r["expert"][0]).astype(int)
            sl = int(r["seq_length"])

            state_norm = np.clip(state_raw, -1, 1)
            boundaries = tokenize_with_boundaries("stamp seal", state_norm, max_token_len)

            nv = V_TOKENS_PER_IMAGE * NUM_IMAGES  # 768
            lang_start = nv
            lang_end = nv + max_token_len  # 768 + 200 = 968

            if lang_end > sl:
                lang_end = sl - K_FORCE

            lang_eids = eids[lang_start:lang_end]
            n_real = boundaries["n_real"]
            n_total_lang = len(lang_eids)

            ts, te = boundaries["task_start"], boundaries["task_end"]
            if ts < n_total_lang and te <= n_total_lang and te > ts:
                for eid in lang_eids[ts:te]:
                    task_exp[int(eid)] += 1; task_count += 1

            ss, se = boundaries["state_start"], boundaries["state_end"]
            if ss < n_total_lang and se <= n_total_lang and se > ss:
                for eid in lang_eids[ss:se]:
                    state_exp[int(eid)] += 1; state_count += 1

            a_s, ae = boundaries["action_start"], boundaries["action_end"]
            if a_s < n_total_lang and ae <= n_total_lang and ae > a_s:
                for eid in lang_eids[a_s:ae]:
                    action_exp[int(eid)] += 1; action_count += 1

            if n_real < n_total_lang:
                for eid in lang_eids[n_real:]:
                    pad_exp[int(eid)] += 1; pad_count += 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=" * 60)
    lines.append("Language Token Sub-Segment Expert Routing (K=16)")
    lines.append(f"Checkpoint: {Path(args.checkpoint).name}")
    lines.append("=" * 60)
    lines.append(f"\nToken layout: V({V_TOKENS_PER_IMAGE*NUM_IMAGES}) + L({max_token_len}) + F({K_FORCE})")
    lines.append(f"  Real text tokens: ~{boundaries['n_real']}  |  Padding: ~{max_token_len - boundaries['n_real']}")
    lines.append("")
    header = f"  {'Sub-segment':<20} {'Tokens':>8}"
    for e in range(4):
        header += f"  {'E'+str(e):>8}"
    lines.append(header)
    lines.append(f"  {'-'*20} {'-'*8}  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    def fmt_counter(c, n):
        t = sum(c.values()) or 1
        row = [f"{x/t*100:7.1f}%" for x in (c.get(0,0), c.get(1,0), c.get(2,0), c.get(3,0))]
        return f"  {n:<20} {sum(c.values()):>8}" + "".join(f"  {x}" for x in row)

    lines.append(fmt_counter(task_exp, "Task"))
    lines.append(fmt_counter(state_exp, "State"))
    lines.append(fmt_counter(action_exp, "Action"))
    lines.append(fmt_counter(pad_exp, "Padding"))
    lines.append(f"  {'-'*20} {'-'*8}  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    # 汇总：全部语言 token
    all_counter = task_exp + state_exp + action_exp + pad_exp
    lines.append(fmt_counter(all_counter, "Lang-ALL"))

    report = "\n".join(lines)
    print(report)
    (out / "lang_split.txt").write_text(report)
    print(f"\nReport saved: {out}/lang_split.txt")


if __name__ == "__main__":
    main()
