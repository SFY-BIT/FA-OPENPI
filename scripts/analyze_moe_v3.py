#!/usr/bin/env python
"""MoE 深度分析 v3 — 模态→Expert、Force→Expert、Expert贡献、Router概率表

用法:
  cd /mnt/hdd/sfy/openpi-force
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python -u scripts/analyze_moe_v3.py \
      --checkpoint checkpoints/12000 --num-episodes 4 \
      --output-dir outputs/moe_v3
"""

import argparse, io, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
from copy import deepcopy

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import moe_routing_capture as _routing
from openpi.shared import normalize as _normalize
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

K_FORCE = 2; V_TOKENS = 256
EXPERT_COLORS = ["#2196F3","#4CAF50","#FF9800","#F44336"]


def decode_image(img_data):
    if isinstance(img_data, dict):
        for k in ("bytes","path"):
            if k in img_data and img_data[k] is not None:
                v=img_data[k]; return Image.open(io.BytesIO(v)) if isinstance(v,bytes) else Image.open(v)
    if isinstance(img_data, bytes): return Image.open(io.BytesIO(img_data))
    if hasattr(img_data,"numpy"):
        arr=img_data.numpy() if callable(img_data.numpy) else np.asarray(img_data)
        if arr.ndim==3 and arr.shape[0]==3: arr=arr.transpose(1,2,0)
        return Image.fromarray(arr.astype(np.uint8))
    if isinstance(img_data, np.ndarray):
        if img_data.ndim==3 and img_data.shape[0]==3: img_data=img_data.transpose(1,2,0)
        return Image.fromarray(img_data.astype(np.uint8))
    raise TypeError(f"Unknown: {type(img_data)}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="pi05_force_stamp_seal_remote")
    p.add_argument("--checkpoint",default="checkpoints/12000")
    p.add_argument("--dataset",default="/mnt/hdd/sfy/lerobot_datasets/stamp_seal_flexiv")
    p.add_argument("--norm-stats",default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv")
    p.add_argument("--num-episodes",type=int,default=4)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--output-dir",default="outputs/moe_v3")
    args=p.parse_args()

    out_dir=Path(args.output_dir); os.makedirs(out_dir,exist_ok=True)
    raw_dir=out_dir/"raw_data"; os.makedirs(raw_dir,exist_ok=True)
    ckpt_step=Path(args.checkpoint).name

    # ── 数据集 ──
    print(f"Dataset: {args.dataset}")
    ds=lerobot_dataset.LeRobotDataset(args.dataset)
    ep_idx=ds.episode_data_index
    n_eps=min(args.num_episodes,len(ep_idx["from"])); eps=list(range(n_eps))
    print(f"  {len(ep_idx['from'])} episodes, using {n_eps}")

    # ── 加载策略 ──
    norm_stats=None
    if Path(args.norm_stats,"norm_stats.json").exists():
        norm_stats=_normalize.load(args.norm_stats)
        norm_stats.pop("force_target",None)

    print(f"Loading policy: {args.config}")
    _routing.enable()
    cfg=_config.get_config(args.config)
    policy=_policy_config.create_trained_policy(cfg,args.checkpoint,norm_stats=norm_stats)
    policy._has_flow_breakdown=False
    print("  First JIT..."); sys.stdout.flush()

    # ═══════════════════════════════════════════════
    # 数据采集
    # ═══════════════════════════════════════════════
    all_frames=[]  # 扁平列表
    ep_data={}     # 按 episode 分组

    for ep in eps:
        s,e=int(ep_idx["from"][ep]),int(ep_idx["to"][ep])
        length=e-s
        max_f=length if args.max_frames<=0 else min(args.max_frames,length)
        step=max(1,length//max_f)
        idxs=list(range(s,e,step))[:max_f]
        ep_frames=[]

        print(f"\nEp{ep}: {length}f, sampling {len(idxs)}"); sys.stdout.flush()

        for i,fi in enumerate(idxs):
            if i>0 and i%80==0: print(f"  {i}/{len(idxs)}",end="\r"); sys.stdout.flush()

            frm=ds[fi]
            obs={
                "observation/state":np.asarray(frm["observation.state"],dtype=np.float32),
                "observation/image":np.asarray(decode_image(frm["observation.image"])),
                "observation/wrist_image":np.asarray(decode_image(frm["observation.wrist_image"])),
                "prompt":"stamp seal",
            }

            with _routing.frame() as rr: policy.infer(obs)
            if not rr: continue

            # ── 从 routing capture 提取数据 ──
            # rr[0] = 扩散第一步的 routing（所有步相同）
            r=rr[0]
            eids=np.asarray(r["expert"][0]).astype(int)    # [seq_len] expert id
            probs=np.asarray(r["prob"][0])                  # [seq_len] gate prob
            sl=int(r["seq_length"])

            nv=V_TOKENS*2; l_end=sl-K_FORCE

            # ① 各模态 token 数量 & ② 每个 token 的 router probability
            vis_mask=np.arange(sl)<nv
            lang_mask=(np.arange(sl)>=nv)&(np.arange(sl)<l_end) if l_end>nv else np.zeros(sl,dtype=bool)
            force_mask=np.arange(sl)>=l_end

            n_vis=int(vis_mask.sum()); n_lang=int(lang_mask.sum()); n_force=int(force_mask.sum())

            # Per modality: expert distribution + mean gate prob
            vis_exp=Counter(eids[vis_mask].tolist()) if n_vis>0 else {}
            lang_exp=Counter(eids[lang_mask].tolist()) if n_lang>0 else {}
            force_exp=Counter(eids[force_mask].tolist()) if n_force>0 else {}

            vis_prob_mean=float(probs[vis_mask].mean()) if n_vis>0 else 0
            lang_prob_mean=float(probs[lang_mask].mean()) if n_lang>0 else 0
            force_prob_mean=float(probs[force_mask].mean()) if n_force>0 else 0

            # Per expert: mean gate prob, token count
            expert_prob={}; expert_count={}
            for e in range(4):
                emask=eids==e
                if emask.any():
                    expert_prob[f"e{e}"]=float(probs[emask].mean())
                    expert_count[f"e{e}"]=int(emask.sum())
                else:
                    expert_prob[f"e{e}"]=0.0
                    expert_count[f"e{e}"]=0

            # ③ Expert 贡献 (gate probability × token count 作为 proxy)
            #   real ||gate × output|| 需要改模型，这里用 "sum of gate probs" 近似
            expert_contrib={}
            for e in range(4):
                emask=eids==e
                expert_contrib[f"e{e}"]=float(probs[emask].sum()) if emask.any() else 0.0

            # Router entropy per frame
            total_tokens=sl
            expert_fracs=np.array([expert_count[f"e{e}"]/max(total_tokens,1) for e in range(4)])
            # Entropy of expert distribution
            valid=expert_fracs>0
            entropy=float(-np.sum(expert_fracs[valid]*np.log(expert_fracs[valid])))

            # ④ Force values
            force_vals=np.asarray(frm["observation.state"][7:13],dtype=np.float32)
            force_norm=float(np.linalg.norm(force_vals[:3]))
            force_norm_torque=float(np.linalg.norm(force_vals))

            frame_rec={
                "ep":ep,"fi":fi,"progress":round((fi-s)/length,4),"sl":sl,
                # ①
                "n_vis":n_vis,"n_lang":n_lang,"n_force":n_force,
                # ②
                "vis_exp":{f"e{k}":v for k,v in vis_exp.items()},
                "lang_exp":{f"e{k}":v for k,v in lang_exp.items()},
                "force_exp":{f"e{k}":v for k,v in force_exp.items()},
                "vis_prob":vis_prob_mean,"lang_prob":lang_prob_mean,"force_prob":force_prob_mean,
                "expert_prob":expert_prob,
                "expert_count":expert_count,
                # ③
                "expert_contrib":expert_contrib,
                # Router entropy
                "entropy":entropy,
                "expert_fracs":expert_fracs.tolist(),
                # ④
                "fx":float(force_vals[0]),"fy":float(force_vals[1]),"fz":float(force_vals[2]),
                "tx":float(force_vals[3]),"ty":float(force_vals[4]),"tz":float(force_vals[5]),
                "force_norm":force_norm,"force_norm_torque":force_norm_torque,
            }
            ep_frames.append(frame_rec)
            all_frames.append(frame_rec)

        ep_data[f"ep{ep}"]=ep_frames
        json_path=raw_dir/f"ep{ep}_v3.json"
        with open(json_path,"w") as f: json.dump(ep_frames,f,indent=2)
        print(f"  Ep{ep}: {len(ep_frames)} frames -> {json_path}"+" "*30)

    n_total=len(all_frames)
    print(f"\nTotal: {n_total} frames")

    # ═══════════════════════════════════════════════
    # 📊 图 1: 模态 → Expert 条件分布
    # ═══════════════════════════════════════════════
    fig1,axes1=plt.subplots(1,3,figsize=(18,5))
    for ax_i,(mod_name,mod_key) in enumerate([
        ("Vision","vis_exp"),("Language","lang_exp"),("Force","force_exp")
    ]):
        ax=axes1[ax_i]
        # Aggregate across all frames
        agg={e:0 for e in range(4)}
        for fr in all_frames:
            d=fr.get(mod_key,{})
            for e in range(4):
                agg[e]+=d.get(f"e{e}",0)
        total=sum(agg.values()) or 1
        vals=[agg[e]/total*100 for e in range(4)]
        bars=ax.bar(range(4),vals,color=EXPERT_COLORS,edgecolor="white",linewidth=0.5)
        for bar,v in zip(bars,vals):
            if v>0.5: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                              f"{v:.1f}%",ha="center",fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"E{e}" for e in range(4)])
        ax.set_ylabel("Token Share (%)"); ax.set_title(f"{mod_name} → Expert")
        ax.set_ylim(0,max(vals)*1.25+2)
    plt.suptitle(f"Modality → Expert Distribution (Step {ckpt_step})",fontsize=13)
    plt.tight_layout()
    fig1.savefig(out_dir/f"fig1_modality_expert_{ckpt_step}.png",dpi=150)
    plt.close(fig1)
    print(f"Fig1: {out_dir}/fig1_modality_expert_{ckpt_step}.png")

    # ═══════════════════════════════════════════════
    # 📊 图 2: Force 强度 → Expert 路由
    # ═══════════════════════════════════════════════
    #  左: force→expert per episode 的曲线
    #  右: force 分桶 bar chart
    fig2=plt.figure(figsize=(16,5*max(n_eps,2)))
    gs2=gridspec.GridSpec(max(n_eps,2),2,figure=fig2,width_ratios=[3,1],hspace=0.4)

    for ep_i in range(n_eps):
        ep_frames=ep_data[f"ep{ep_i}"]
        xs=[fr["progress"] for fr in ep_frames]
        fn=[fr["force_norm"] for fr in ep_frames]

        # 左: 轨迹中 force token 的 expert 分布变化
        ax_l=fig2.add_subplot(gs2[ep_i,0])
        for e in range(4):
            ys=[]
            for fr in ep_frames:
                d=fr["force_exp"]
                total=max(sum(d.values()),1)
                ys.append(d.get(f"e{e}",0)/total*100)
            ax_l.plot(xs,ys,color=EXPERT_COLORS[e],lw=1.2,alpha=0.8,label=f"Force→E{e}")
        ax_l.set_ylabel("Force Token Expert Share (%)",color="#ccc")
        ax_l.set_ylim(-2,105)
        ax_l.legend(fontsize=7,ncol=4,loc="upper right")

        # Force overlay
        ax_lf=ax_l.twinx()
        ax_lf.fill_between(xs,0,fn,alpha=0.12,color="#e74c3c")
        ax_lf.plot(xs,fn,color="#e74c3c",lw=0.6,alpha=0.7)
        ax_lf.set_ylabel("|F_xyz| (N)",color="#e74c3c")
        ax_l.set_title(f"Ep{ep_i}: Force Token Expert Share + Force Profile",fontsize=10)

        # 右: force 分桶 bar
        ax_r=fig2.add_subplot(gs2[ep_i,1])
        fn_arr=np.array(fn)
        bins=[0,2,5,10,20,50]
        bin_labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]+[f">{bins[-1]}"]
        bucket_data=defaultdict(list)
        for j,fr in enumerate(ep_frames):
            b=np.digitize(fn_arr[j],bins)-1
            b=max(0,min(b,len(bins)-1))
            bucket_data[b].append(fr)
        x_pos=np.arange(len(bin_labels)); bar_w=0.2
        for e in range(4):
            means=[]
            for b in range(len(bin_labels)):
                brecs=bucket_data.get(b,[])
                if brecs:
                    vals=[fr["force_exp"].get(f"e{e}",0)/max(sum(fr["force_exp"].values()),1)*100 for fr in brecs]
                    means.append(np.mean(vals))
                else:
                    means.append(0)
            ax_r.bar(x_pos+e*bar_w,means,bar_w,color=EXPERT_COLORS[e],label=f"E{e}")
        ax_r.set_xticks(x_pos+1.5*bar_w); ax_r.set_xticklabels(bin_labels,fontsize=7)
        ax_r.set_xlabel("|F_xyz| (N)"); ax_r.set_ylabel("Force→Expert %")
        ax_r.set_title("Force Level → Expert",fontsize=9); ax_r.legend(fontsize=6,ncol=4)

    plt.suptitle(f"Force Intensity → Expert Routing (Step {ckpt_step})",fontsize=13)
    fig2.savefig(out_dir/f"fig2_force_expert_{ckpt_step}.png",dpi=150,bbox_inches="tight")
    plt.close(fig2)
    print(f"Fig2: {out_dir}/fig2_force_expert_{ckpt_step}.png")

    # ═══════════════════════════════════════════════
    # 📊 图 3: Expert 实际贡献 + Router 概率分布
    # ═══════════════════════════════════════════════
    fig3,axes3=plt.subplots(2,2,figsize=(14,10))

    # 3a: Expert contribution (sum of gate probs) — per episode
    ax3a=axes3[0,0]
    x_pos=np.arange(n_eps); bar_w=0.2
    for e in range(4):
        means=[np.mean([fr["expert_contrib"][f"e{e}"] for fr in ep_data[f"ep{ep_i}"]])
               for ep_i in range(n_eps)]
        ax3a.bar(x_pos+e*bar_w,means,bar_w,color=EXPERT_COLORS[e],label=f"E{e}")
    ax3a.set_xticks(x_pos+1.5*bar_w); ax3a.set_xticklabels([f"Ep{i}" for i in range(n_eps)])
    ax3a.set_ylabel("Σ gate_prob"); ax3a.set_title("Expert Contribution (Σ gate_prob)")
    ax3a.legend(ncol=4)

    # 3b: Gate probability distribution per expert (violin/box)
    ax3b=axes3[0,1]
    prob_data=[[fr["expert_prob"][f"e{e}"] for fr in all_frames if fr["expert_count"][f"e{e}"]>0]
               for e in range(4)]
    bp=ax3b.boxplot(prob_data,patch_artist=True)
    ax3b.set_xticklabels([f"E{e}" for e in range(4)])
    for patch,e in zip(bp["boxes"],range(4)):
        patch.set_facecolor(EXPERT_COLORS[e]); patch.set_alpha(0.6)
    ax3b.set_ylabel("Mean Gate Probability"); ax3b.set_title("Per-Expert Gate Probability Distribution")
    ax3b.grid(axis="y",alpha=0.2)

    # 3c: Router entropy over trajectory (per episode)
    ax3c=axes3[1,0]
    for ep_i in range(n_eps):
        ep_frames=ep_data[f"ep{ep_i}"]
        xs=[fr["progress"] for fr in ep_frames]
        ys=[fr["entropy"] for fr in ep_frames]
        ax3c.plot(xs,ys,lw=1,alpha=0.7,label=f"Ep{ep_i}")
    max_entropy=np.log(4)  # uniform over 4 experts
    ax3c.axhline(max_entropy,color="gray",ls="--",alpha=0.3,label=f"Max entropy ({max_entropy:.2f})")
    ax3c.set_ylabel("Router Entropy (nats)"); ax3c.set_xlabel("Progress")
    ax3c.set_title("Router Entropy Over Trajectory"); ax3c.legend(fontsize=7)

    # 3d: Token count per expert (log scale)
    ax3d=axes3[1,1]
    for e in range(4):
        cnts=[fr["expert_count"][f"e{e}"] for fr in all_frames]
        ax3d.hist(cnts,bins=50,alpha=0.5,color=EXPERT_COLORS[e],label=f"E{e}")
    ax3d.set_xlabel("Token Count"); ax3d.set_ylabel("Frequency")
    ax3d.set_title("Per-Frame Token Count Distribution"); ax3d.legend(fontsize=7)

    plt.suptitle(f"Expert Contribution & Router Probability (Step {ckpt_step})",fontsize=13)
    plt.tight_layout()
    fig3.savefig(out_dir/f"fig3_contrib_prob_{ckpt_step}.png",dpi=150)
    plt.close(fig3)
    print(f"Fig3: {out_dir}/fig3_contrib_prob_{ckpt_step}.png")

    # ═══════════════════════════════════════════════
    # 📋 表: Router Probability / Entropy Summary
    # ═══════════════════════════════════════════════
    table_path=out_dir/f"table_router_stats_{ckpt_step}.txt"
    with open(table_path,"w",encoding="utf-8") as f:
        f.write("="*75+"\n")
        f.write(f"Router Probability & Entropy Summary — Step {ckpt_step}\n")
        f.write(f"Frames: {n_total} | Episodes: {n_eps}\n")
        f.write("="*75+"\n\n")

        f.write("─"*60+"\n")
        f.write("Per-Modality Mean Gate Probability\n")
        f.write("─"*60+"\n")
        for mod,key in [("Vision","vis_prob"),("Language","lang_prob"),("Force","force_prob")]:
            vals=[fr[key] for fr in all_frames if fr[key]>0]
            if vals:
                f.write(f"  {mod:12s}: {np.mean(vals):.4f} ±{np.std(vals):.4f}  "
                        f"[{np.min(vals):.4f}-{np.max(vals):.4f}]\n")
        f.write("\n")

        f.write("─"*60+"\n")
        f.write("Per-Expert Statistics (across all frames)\n")
        f.write("─"*60+"\n")
        f.write(f"  {'Expert':<8} {'Tokens/frame':>12} {'Gate Prob':>10} {'Contrib(Σp)':>12} {'Share':>8}\n")
        f.write(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*12} {'─'*8}\n")
        for e in range(4):
            cnts=[fr["expert_count"][f"e{e}"] for fr in all_frames]
            probs=[fr["expert_prob"][f"e{e}"] for fr in all_frames if fr["expert_count"][f"e{e}"]>0]
            contribs=[fr["expert_contrib"][f"e{e}"] for fr in all_frames]
            total_tokens=np.mean([fr["sl"] for fr in all_frames])
            f.write(f"  {'E'+str(e):<8} {np.mean(cnts):>12.1f} "
                    f"{np.mean(probs) if probs else 0:>10.4f} "
                    f"{np.mean(contribs):>12.1f} "
                    f"{np.mean(cnts)/total_tokens*100:>7.1f}%\n")
        f.write("\n")

        f.write("─"*60+"\n")
        f.write("Router Entropy Statistics\n")
        f.write("─"*60+"\n")
        entropies=[fr["entropy"] for fr in all_frames]
        max_ent=np.log(4)
        f.write(f"  Mean:  {np.mean(entropies):.4f} nats ({np.mean(entropies)/max_ent*100:.1f}% of max)\n")
        f.write(f"  Std:   {np.std(entropies):.4f}\n")
        f.write(f"  Min:   {np.min(entropies):.4f} (most concentrated)\n")
        f.write(f"  Max:   {np.max(entropies):.4f} (most uniform)\n")
        f.write(f"  Max possible: {max_ent:.4f} (4 experts, uniform)\n\n")

        f.write("─"*60+"\n")
        f.write("Force-Level Conditional Expert Distribution\n")
        f.write("─"*60+"\n")
        bins=[0,2,5,10,20,50]
        bin_labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]+[f">{bins[-1]}"]
        f.write(f"  {'Force Range':<12}")
        for e in range(4): f.write(f" {'E'+str(e):>8}")
        f.write(f" {'Frames':>8}\n")
        for b in range(len(bin_labels)):
            brecs=[fr for fr in all_frames if b==min(max(np.digitize(fr["force_norm"],bins)-1,0),len(bins)-1)]
            if not brecs: continue
            f.write(f"  {bin_labels[b]:<12}",end="")
            for e in range(4):
                vals=[fr["force_exp"].get(f"e{e}",0)/max(sum(fr["force_exp"].values()),1)*100 for fr in brecs]
                f.write(f" {np.mean(vals):>7.1f}%",end="")
            f.write(f" {len(brecs):>8}\n")
        f.write("\n")

        f.write("─"*60+"\n")
        f.write("Notes\n")
        f.write("─"*60+"\n")
        f.write("  * 'Expert Contribution' = Σ gate_prob over all tokens routed to that expert.\n")
        f.write("    This is a PROXY for true ||gate × expert_output|| (requires model code change).\n")
        f.write("  * Gate probability = softmax(router_logits) after top-1 selection.\n")
        f.write("  * Raw per-frame JSON data is in raw_data/ep*_v3.json\n")

    print(f"Table: {table_path}")
    print(f"\nDone. All outputs in {out_dir}/")
    print(f"  fig1_modality_expert_{ckpt_step}.png")
    print(f"  fig2_force_expert_{ckpt_step}.png")
    print(f"  fig3_contrib_prob_{ckpt_step}.png")
    print(f"  table_router_stats_{ckpt_step}.txt")
    print(f"  raw_data/ep*_v3.json")


if __name__=="__main__": main()
