#!/usr/bin/env python
"""从已保存的 JSON 生成图表（无需重新推理）

用法:
  cd /mnt/hdd/sfy/openpi-force
  source /home/sfy/miniconda3/etc/profile.d/conda.sh && conda activate rlinf
  python scripts/analyze_moe_v3_plot.py \
      --data-dir outputs/moe_v3/raw_data \
      --output-dir outputs/moe_v3 \
      --checkpoint-step 12000
"""

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXPERT_COLORS = ["#2196F3","#4CAF50","#FF9800","#F44336"]


def load_data(data_dir):
    all_frames = []
    ep_data = {}
    for jf in sorted(Path(data_dir).glob("ep*_v3.json")):
        ep_name = jf.stem.replace("_v3", "")
        with open(jf) as f:
            frames = json.load(f)
        ep_data[ep_name] = frames
        all_frames.extend(frames)
        print(f"  {jf.name}: {len(frames)} frames")
    return all_frames, ep_data


def fig1_modality_expert(all_frames, out_dir, step):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_i, (mod_name, mod_key) in enumerate([
        ("Vision", "vis_exp"), ("Language", "lang_exp"), ("Force", "force_exp")
    ]):
        ax = axes[ax_i]
        agg = {e: 0 for e in range(4)}
        for fr in all_frames:
            d = fr.get(mod_key, {})
            for e in range(4):
                agg[e] += d.get(f"e{e}", 0)
        total = sum(agg.values()) or 1
        vals = [agg[e] / total * 100 for e in range(4)]
        bars = ax.bar(range(4), vals, color=EXPERT_COLORS, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"E{e}" for e in range(4)])
        ax.set_ylabel("Token Share (%)")
        ax.set_title(f"{mod_name} -> Expert")
        ax.set_ylim(0, max(vals) * 1.25 + 2)
    plt.suptitle(f"Modality -> Expert Distribution (Step {step})", fontsize=13)
    plt.tight_layout()
    path = out_dir / f"fig1_modality_expert_{step}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Fig1: {path}")


def fig2_force_expert(ep_data, out_dir, step):
    n_eps = len(ep_data)
    fig = plt.figure(figsize=(16, 5 * max(n_eps, 2)))
    gs = gridspec.GridSpec(max(n_eps, 2), 2, figure=fig, width_ratios=[3, 1], hspace=0.4)

    for ep_i, ep_name in enumerate(sorted(ep_data.keys())):
        ep_frames = ep_data[ep_name]
        xs = [fr["progress"] for fr in ep_frames]
        fn = [fr["force_norm"] for fr in ep_frames]

        # Left: force token expert share over trajectory
        ax_l = fig.add_subplot(gs[ep_i, 0])
        for e in range(4):
            ys = []
            for fr in ep_frames:
                d = fr["force_exp"]
                total = max(sum(d.values()), 1)
                ys.append(d.get(f"e{e}", 0) / total * 100)
            ax_l.plot(xs, ys, color=EXPERT_COLORS[e], lw=1.2, alpha=0.8, label=f"Force->E{e}")
        ax_l.set_ylabel("Force Token Expert Share (%)", color="#ccc")
        ax_l.set_ylim(-2, 105)
        ax_l.legend(fontsize=7, ncol=4, loc="upper right")
        ax_lf = ax_l.twinx()
        ax_lf.fill_between(xs, 0, fn, alpha=0.12, color="#e74c3c")
        ax_lf.plot(xs, fn, color="#e74c3c", lw=0.6, alpha=0.7)
        ax_lf.set_ylabel("|F_xyz| (N)", color="#e74c3c")
        ax_l.set_title(f"{ep_name}: Force Token Expert Share + Force Profile", fontsize=10)

        # Right: force bucket bar chart
        ax_r = fig.add_subplot(gs[ep_i, 1])
        fn_arr = np.array(fn)
        bins = [0, 2, 5, 10, 20, 50]
        bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)] + [f">{bins[-1]}"]
        bucket_data = defaultdict(list)
        for j, fr in enumerate(ep_frames):
            b = np.digitize(fn_arr[j], bins) - 1
            b = max(0, min(b, len(bins) - 1))
            bucket_data[b].append(fr)
        x_pos = np.arange(len(bin_labels))
        bar_w = 0.2
        for e in range(4):
            means = []
            for b in range(len(bin_labels)):
                brecs = bucket_data.get(b, [])
                vals = [fr["force_exp"].get(f"e{e}", 0) / max(sum(fr["force_exp"].values()), 1) * 100
                        for fr in brecs] if brecs else []
                means.append(np.mean(vals) if vals else 0)
            ax_r.bar(x_pos + e * bar_w, means, bar_w, color=EXPERT_COLORS[e], label=f"E{e}")
        ax_r.set_xticks(x_pos + 1.5 * bar_w)
        ax_r.set_xticklabels(bin_labels, fontsize=7)
        ax_r.set_xlabel("|F_xyz| (N)")
        ax_r.set_ylabel("Force->Expert %")
        ax_r.set_title("Force Level -> Expert", fontsize=9)
        ax_r.legend(fontsize=6, ncol=4)

    plt.suptitle(f"Force Intensity -> Expert Routing (Step {step})", fontsize=13)
    path = out_dir / f"fig2_force_expert_{step}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig2: {path}")


def fig3_contrib_prob(all_frames, ep_data, out_dir, step):
    n_eps = len(ep_data)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 3a: Expert contribution per episode
    ax = axes[0, 0]
    x_pos = np.arange(n_eps)
    bar_w = 0.2
    for e in range(4):
        means = [np.mean([fr["expert_contrib"][f"e{e}"] for fr in ep_data[en]])
                 for en in sorted(ep_data.keys())]
        ax.bar(x_pos + e * bar_w, means, bar_w, color=EXPERT_COLORS[e], label=f"E{e}")
    ax.set_xticks(x_pos + 1.5 * bar_w)
    ax.set_xticklabels(sorted(ep_data.keys()))
    ax.set_ylabel("Sum gate_prob")
    ax.set_title("Expert Contribution (Sum gate_prob)")
    ax.legend(ncol=4)

    # 3b: Gate probability boxplot per expert
    ax = axes[0, 1]
    prob_data = [[fr["expert_prob"][f"e{e}"] for fr in all_frames
                  if fr["expert_count"][f"e{e}"] > 0] for e in range(4)]
    bp = ax.boxplot(prob_data, patch_artist=True)
    for patch, e in zip(bp["boxes"], range(4)):
        patch.set_facecolor(EXPERT_COLORS[e])
        patch.set_alpha(0.6)
    ax.set_xticklabels([f"E{e}" for e in range(4)])
    ax.set_ylabel("Mean Gate Probability")
    ax.set_title("Per-Expert Gate Probability Distribution")
    ax.grid(axis="y", alpha=0.2)

    # 3c: Router entropy over trajectory
    ax = axes[1, 0]
    for ep_name in sorted(ep_data.keys()):
        ep_frames = ep_data[ep_name]
        xs = [fr["progress"] for fr in ep_frames]
        ys = [fr["entropy"] for fr in ep_frames]
        ax.plot(xs, ys, lw=1, alpha=0.7, label=ep_name)
    max_ent = np.log(4)
    ax.axhline(max_ent, color="gray", ls="--", alpha=0.3, label=f"Max ({max_ent:.2f})")
    ax.set_ylabel("Router Entropy (nats)")
    ax.set_xlabel("Progress")
    ax.set_title("Router Entropy Over Trajectory")
    ax.legend(fontsize=7)

    # 3d: Token count distribution per expert
    ax = axes[1, 1]
    for e in range(4):
        cnts = [fr["expert_count"][f"e{e}"] for fr in all_frames]
        ax.hist(cnts, bins=50, alpha=0.5, color=EXPERT_COLORS[e], label=f"E{e}")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Per-Frame Token Count Distribution")
    ax.legend(fontsize=7)

    plt.suptitle(f"Expert Contribution & Router Probability (Step {step})", fontsize=13)
    plt.tight_layout()
    path = out_dir / f"fig3_contrib_prob_{step}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Fig3: {path}")


def write_table(all_frames, out_dir, step):
    path = out_dir / f"table_router_stats_{step}.txt"
    n_total = len(all_frames)
    sep1 = "=" * 72
    sep2 = "-" * 60

    with open(path, "w", encoding="utf-8") as f:
        w = f.write

        w(sep1 + "\n")
        w(f"Router Probability & Entropy Summary -- Step {step}\n")
        w(f"Frames: {n_total}\n")
        w(sep1 + "\n\n")

        # Per-modality gate prob
        w(sep2 + "\n")
        w("Per-Modality Mean Gate Probability\n")
        w(sep2 + "\n")
        for mod, key in [("Vision", "vis_prob"), ("Language", "lang_prob"), ("Force", "force_prob")]:
            vals = [fr[key] for fr in all_frames if fr[key] > 0]
            if vals:
                w(f"  {mod:12s}: {np.mean(vals):.4f} +/-{np.std(vals):.4f}"
                  f"  [{np.min(vals):.4f}-{np.max(vals):.4f}]\n")
        w("\n")

        # Per-expert stats
        w(sep2 + "\n")
        w("Per-Expert Statistics\n")
        w(sep2 + "\n")
        header = f"  {'Expert':<8} {'Tokens/frame':>12} {'Gate Prob':>10} {'Contrib(Sp)':>12} {'Share':>8}"
        w(header + "\n")
        w("  " + "-" * 56 + "\n")
        for e in range(4):
            cnts = [fr["expert_count"][f"e{e}"] for fr in all_frames]
            probs = [fr["expert_prob"][f"e{e}"] for fr in all_frames
                     if fr["expert_count"][f"e{e}"] > 0]
            contribs = [fr["expert_contrib"][f"e{e}"] for fr in all_frames]
            total_tokens = np.mean([fr["sl"] for fr in all_frames])
            w(f"  {'E'+str(e):<8} {np.mean(cnts):>12.1f} "
              f"{np.mean(probs) if probs else 0:>10.4f} "
              f"{np.mean(contribs):>12.1f} "
              f"{np.mean(cnts)/total_tokens*100:>7.1f}%\n")
        w("\n")

        # Router entropy
        w(sep2 + "\n")
        w("Router Entropy Statistics\n")
        w(sep2 + "\n")
        entropies = [fr["entropy"] for fr in all_frames]
        max_ent = np.log(4)
        w(f"  Mean:  {np.mean(entropies):.4f} nats ({np.mean(entropies)/max_ent*100:.1f}% of max)\n")
        w(f"  Std:   {np.std(entropies):.4f}\n")
        w(f"  Min:   {np.min(entropies):.4f}\n")
        w(f"  Max:   {np.max(entropies):.4f}\n")
        w(f"  Max possible: {max_ent:.4f} (4 experts uniform)\n\n")

        # Force-level conditional
        w(sep2 + "\n")
        w("Force-Level Conditional Expert Distribution\n")
        w(sep2 + "\n")
        bins = [0, 2, 5, 10, 20, 50]
        bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)] + [f">{bins[-1]}"]
        header2 = f"  {'Force Range':<12}"
        for e in range(4):
            header2 += f" {'E'+str(e):>8}"
        header2 += f" {'Frames':>8}"
        w(header2 + "\n")
        for b in range(len(bin_labels)):
            brecs = [fr for fr in all_frames
                     if b == min(max(np.digitize(fr["force_norm"], bins) - 1, 0), len(bins) - 1)]
            if not brecs:
                continue
            line = f"  {bin_labels[b]:<12}"
            for e in range(4):
                vals = [fr["force_exp"].get(f"e{e}", 0) /
                        max(sum(fr["force_exp"].values()), 1) * 100
                        for fr in brecs]
                line += f" {np.mean(vals):>7.1f}%"
            line += f" {len(brecs):>8}"
            w(line + "\n")
        w("\n")

    print(f"Table: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="outputs/moe_v3/raw_data")
    p.add_argument("--output-dir", default="outputs/moe_v3")
    p.add_argument("--checkpoint-step", default="12000")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    step = args.checkpoint_step

    print(f"Loading data from {data_dir}")
    all_frames, ep_data = load_data(data_dir)
    print(f"Total: {len(all_frames)} frames, {len(ep_data)} episodes\n")

    fig1_modality_expert(all_frames, out_dir, step)
    fig2_force_expert(ep_data, out_dir, step)
    fig3_contrib_prob(all_frames, ep_data, out_dir, step)
    write_table(all_frames, out_dir, step)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
