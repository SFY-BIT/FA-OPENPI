#!/usr/bin/env python
"""MoE 路由对比柱状图: 主干 (openpi-force 12k/30k/40k) vs FA (ft60@29999).

生成 3 张图到 FA-VLA-结论/03_对比图/:
  figA_overall_expert_share.png   总体专家分布
  figB_modality_expert.png        模态(Vision/Language/Force)→专家
  figC_router_entropy.png         Router 熵 + FA force gate prob
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/hdd/sfy/FA-VLA-结论/03_对比图"
os.makedirs(OUT, exist_ok=True)

EXPERTS = ["E0", "E1", "E2", "E3"]
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
MODELS = ["Backbone 12k", "Backbone 30k", "Backbone 40k", "FA ft60@29999"]
MODEL_COLORS = ["#90A4AE", "#78909C", "#607D8B", "#E53935"]

# 数据 (均为 %)
overall = np.array([
    [50.0, 52.5, 21.1,  0.08],   # E0
    [29.5, 27.1,  0.0, 20.22],   # E1
    [20.5,  0.2, 27.1,  0.15],   # E2
    [ 0.0, 20.2, 51.7, 79.56],   # E3
])

modalities = {
    "Vision (768t)": np.array([
        [62.5, 65.5,  0.9,  0.0],
        [37.1, 34.2,  0.0,  0.0],
        [ 0.4,  0.3, 34.1,  0.1],
        [ 0.0,  0.0, 65.0, 99.9],
    ]),
    "Language (200t)": np.array([
        [ 1.5,  2.0, 98.5,  0.2],
        [ 0.5,  0.0,  0.0, 42.7],
        [98.0,  0.0,  0.5,  0.2],
        [ 0.0, 98.0,  1.0, 56.9],
    ]),
    "Force (2t / 1t)": np.array([
        [79.8, 94.8, 57.2,  0.0],
        [ 5.5,  1.0, 11.2,100.0],
        [ 0.0,  0.0,  0.5,  0.0],
        [14.8,  4.2, 31.0,  0.0],
    ]),
}

# 数值速查 (FA ft60 force 分桶: 所有力区间均 E1=100%)
fa_force_bucket = {"2-5N": [0, 100, 0, 0], "5-10N": [0, 100, 0, 0], "10-20N": [0, 100, 0, 0]}

entropy = [1.0344, 1.0283, 1.0254, 0.515]
force_gate_prob = [0.9999, 0.6780, 0.2515]  # vision/lang/force for FA

BARW = 0.18


def grouped_bar(ax, data, labels=MODELS, title="", ylabel="Token Share (%)"):
    """data: [n_experts, n_models]"""
    n_exp, n_mod = data.shape
    x = np.arange(n_exp)
    for m in range(n_mod):
        ax.bar(x + (m - n_mod / 2 + 0.5) * BARW, data[:, m], BARW,
               label=labels[m], color=MODEL_COLORS[m], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(EXPERTS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_ylim(0, max(data.max() * 1.2, 100))


# ── figA: 总体专家分布 ──
figA, axA = plt.subplots(figsize=(9, 5.2))
grouped_bar(axA, overall, title="Overall Expert Share: Backbone(12k/30k/40k) vs FA(ft60@29999)")
for xi, vals in zip(np.arange(4), overall):
    for m, v in enumerate(vals):
        if v > 2:
            axA.text(xi + (m - 2 + 0.5) * BARW, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
figA.tight_layout()
figA.savefig(os.path.join(OUT, "figA_overall_expert_share.png"), dpi=150)
plt.close(figA)
print("figA done")

# ── figB: 模态→专家 ──
figB, axesB = plt.subplots(1, 3, figsize=(17, 5.2))
for ax, (mod, data) in zip(axesB, modalities.items()):
    grouped_bar(ax, data, title=f"{mod} → Expert", ylabel="Token Share (%)")
figB.suptitle("Modality → Expert: Backbone vs FA", fontsize=13)
figB.tight_layout()
figB.savefig(os.path.join(OUT, "figB_modality_expert.png"), dpi=150)
plt.close(figB)
print("figB done")

# ── figC: Router 熵 + FA force gate prob ──
figC, (axC1, axC2) = plt.subplots(1, 2, figsize=(12, 5.2))

axC1.bar(MODELS, entropy, color=MODEL_COLORS, edgecolor="white", width=0.55)
axC1.axhline(np.log(4), color="gray", ls="--", lw=0.8, label=f"max={np.log(4):.3f}")
for i, v in enumerate(entropy):
    axC1.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
axC1.set_ylabel("Router Entropy (nats)")
axC1.set_title("Router Entropy: Backbone vs FA")
axC1.legend(fontsize=8)
axC1.set_ylim(0, 1.6)

axC2.bar(["Vision", "Language", "Force"], force_gate_prob,
         color=["#42A5F5", "#66BB6A", "#EF5350"], edgecolor="white", width=0.5)
for i, v in enumerate(force_gate_prob):
    axC2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
axC2.set_ylabel("Mean Gate Prob")
axC2.set_title("FA ft60: Router Confidence by Modality")
axC2.set_ylim(0, 1.15)

figC.tight_layout()
figC.savefig(os.path.join(OUT, "figC_router_entropy_prob.png"), dpi=150)
plt.close(figC)
print("figC done")

print(f"ALL FIGURES -> {OUT}")
