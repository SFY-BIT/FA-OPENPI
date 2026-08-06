#!/usr/bin/env python
"""本地 LoRA K=16 训练趋势分析 — 2000 步决策报告。

读取 logs/ft60_k16.log，输出:
  1. loss 趋势（action/force/total 每 200 步窗口均值）
  2. MoE 路由趋势（E0-E3 每 200 步）
  3. grad_norm 尖峰统计（>50 的 step）
  4. router_z_loss 趋势
  5. 末尾快照（最近 200 步）

用法: PYTHONPATH=src python scripts/analyze_training_trend.py [--log logs/ft60_k16.log]
"""
import argparse, re
from collections import defaultdict


def parse(log_path):
    rows = []
    pat = re.compile(
        r"Step (\d+): action_loss=([\d.]+), force_loss=([\d.]+), grad_norm=([\d.]+), "
        r"loss=([\d.]+), .*moe_router_z_loss=([\d.]+), "
        r"moe_top1_frac_expert_0=([\d.]+), moe_top1_frac_expert_1=([\d.]+), "
        r"moe_top1_frac_expert_2=([\d.]+), moe_top1_frac_expert_3=([\d.]+), "
        r".*skipped_nonfinite=([\d.]+)"
    )
    for line in open(log_path, errors="ignore"):
        m = pat.search(line)
        if m:
            g = m.groups()
            rows.append({
                "step": int(g[0]),
                "action": float(g[1]), "force": float(g[2]),
                "grad": float(g[3]), "loss": float(g[4]),
                "z": float(g[5]),
                "e": [float(g[6]), float(g[7]), float(g[8]), float(g[9])],
                "skip": float(g[10]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/ft60_k16.log")
    ap.add_argument("--window", type=int, default=200)
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        print("No Step rows found in", args.log)
        return
    max_step = rows[-1]["step"]
    print(f"=== 训练趋势报告 (共 {len(rows)} 个日志点, 最高 step {max_step}) ===\n")

    # 1) loss 趋势
    print("--- [1] Loss 趋势 (每 {} 步窗口均值) ---".format(args.window))
    print(f"{'step':>7} {'action':>9} {'force':>9} {'total':>9} {'grad':>9} {'z_loss':>9} {'skip%':>7}")
    for w_start in range(0, max_step + 1, args.window):
        w = [r for r in rows if w_start <= r["step"] < w_start + args.window]
        if not w:
            continue
        n = len(w)
        avg = lambda k: sum(r[k] for r in w) / n
        skip_pct = sum(1 for r in w if r["skip"] > 0) / n * 100
        print(f"{w_start:>7} {avg('action'):>9.4f} {avg('force'):>9.4f} {avg('loss'):>9.4f} "
              f"{avg('grad'):>9.2f} {avg('z'):>9.2f} {skip_pct:>6.1f}%")

    # 2) MoE 路由趋势
    print("\n--- [2] MoE 路由趋势 (E0-E3 占比, 每 {} 步窗口均值) ---".format(args.window))
    print(f"{'step':>7} {'E0':>7} {'E1':>7} {'E2':>7} {'E3':>7}   {'活跃专家'}")
    for w_start in range(0, max_step + 1, args.window):
        w = [r for r in rows if w_start <= r["step"] < w_start + args.window]
        if not w:
            continue
        n = len(w)
        es = [sum(r["e"][i] for r in w) / n for i in range(4)]
        active = sum(1 for v in es if v > 0.05)
        print(f"{w_start:>7} {es[0]:>7.1%} {es[1]:>7.1%} {es[2]:>7.1%} {es[3]:>7.1%}   {active}")

    # 3) grad 尖峰
    spikes = [r for r in rows if r["grad"] > 50]
    print(f"\n--- [3] grad_norm 尖峰 (>50): {len(spikes)} 个 ---")
    for r in spikes:
        print(f"  Step {r['step']}: grad={r['grad']:.1f} action={r['action']:.4f} "
              f"force={r['force']:.4f} z={r['z']:.1f} skip={r['skip']}")

    # 4) 末尾快照
    tail = rows[-max(20, len(rows) // 10):]
    print("\n--- [4] 末尾快照 (最近 {} 个日志点) ---".format(len(tail)))
    last = tail[-1]
    print(f"  最后 step {last['step']}: action={last['action']:.4f} force={last['force']:.4f} "
          f"loss={last['loss']:.4f} grad={last['grad']:.2f} z={last['z']:.2f}")
    es = [sum(r["e"][i] for r in tail) / len(tail) for i in range(4)]
    print(f"  E0={es[0]:.1%} E1={es[1]:.1%} E2={es[2]:.1%} E3={es[3]:.1%} "
          f"(活跃专家 {sum(1 for v in es if v > 0.05)})")

    print("\n=== 结论提示 ===")
    print("  - force/total > 90% 说明 force 主导仍严重 → 需调 loss 平衡")
    print("  - 活跃专家 < 3 说明路由有 collapse 趋势")
    print("  - skip% 高或尖峰多 → 需加梯度尖峰过滤")


if __name__ == "__main__":
    main()
