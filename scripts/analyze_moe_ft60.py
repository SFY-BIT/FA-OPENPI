#!/usr/bin/env python
"""MoE 深度分析 — FA-openpi ft60 版 (K=1 force token)

分析 FA-openpi 训练的 ft_history 模型的路由行为:
  - 模态(视觉/语言/Force)→Expert 分布
  - Force 强度→Expert 路由
  - Router 概率/熵统计

用法:
  cd /mnt/hdd/sfy/FA-openpi
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python -u scripts/analyze_moe_ft60.py \
      --checkpoint checkpoints/pi05_force_stamp_seal_ft60_forcevla_lora/ft60_fv_lora/29999 \
      --num-episodes 4 --output-dir outputs/moe_ft60
"""

import argparse, io, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

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

K_FORCE_DEFAULT = 1   # 默认 1; 可用 --k-force 覆盖 (K=16 时用 16)
V_TOKENS = 256
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
    p.add_argument("--config",default="pi05_force_stamp_seal_ft60_forcevla_lora")
    p.add_argument("--checkpoint",default="checkpoints/pi05_force_stamp_seal_ft60_forcevla_lora/ft60_fv_lora/29999")
    p.add_argument("--dataset",default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60")
    p.add_argument("--norm-stats",default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60")
    p.add_argument("--num-episodes",type=int,default=4)
    p.add_argument("--ep-ids",default=None,
                   help="comma-separated episode ids to sample, e.g. --ep-ids 0 (parallel mode: one proc per episode)")
    p.add_argument("--only-analyze",action="store_true",
                   help="skip sampling; load existing raw_data/ep*_v3.json and only produce figures/tables")
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--k-force",type=int,default=K_FORCE_DEFAULT,
                   help="Number of force tokens at the end of the sequence (1 for K=1, 16 for K=16)")
    p.add_argument("--output-dir",default="outputs/moe_ft60")
    args=p.parse_args()

    K_FORCE = args.k_force

    out_dir=Path(args.output_dir); os.makedirs(out_dir,exist_ok=True)
    raw_dir=out_dir/"raw_data"; os.makedirs(raw_dir,exist_ok=True)
    ckpt_step=Path(args.checkpoint).name

    # ── 数据集 (only-analyze 不需要) ──
    eps=None
    if not args.only_analyze:
        print(f"Dataset: {args.dataset}")
        ds=lerobot_dataset.LeRobotDataset(args.dataset)
        ep_idx=ds.episode_data_index
        n_eps=min(args.num_episodes,len(ep_idx["from"]))
        if args.ep_ids:
            eps=[int(x.strip()) for x in args.ep_ids.split(",")]
            n_eps=len(eps)
        else:
            eps=list(range(n_eps))
        print(f"  {len(ep_idx['from'])} episodes, using {n_eps}: {eps}")

    # ── 加载策略 (only-analyze 不需要) ──
    if not args.only_analyze:
        norm_stats=None
        if Path(args.norm_stats,"norm_stats.json").exists():
            norm_stats=_normalize.load(args.norm_stats)
            # These keys exist in norm_stats but NOT in the model output tree.
            # Unnormalize(strict=True) would fail on them.
            norm_stats.pop("force_target",None)
            norm_stats.pop("ft_state",None)
            print(f"  norm_stats keys: {list(norm_stats.keys())}")

        print(f"Loading policy: {args.config}")
        _routing.enable()
        cfg=_config.get_config(args.config)
        policy=_policy_config.create_trained_policy(cfg,args.checkpoint,norm_stats=norm_stats)
        policy._has_flow_breakdown=False
        print("  First JIT..."); sys.stdout.flush()

    # ═══════════════════════════════════════════════
    # 数据采集 (或 only-analyze 从已有 json 加载)
    # ═══════════════════════════════════════════════
    all_frames=[]  # 扁平列表
    ep_data={}     # 按 episode 分组

    if args.only_analyze:
        for jp in sorted(raw_dir.glob("ep*_v3.json")):
            ep=jp.stem.split("ep")[1].split("_")[0]
            frames=json.loads(jp.read_text())
            ep_data[f"ep{ep}"]=frames
            all_frames+=frames
        n_eps=len(ep_data)
        n_total=len(all_frames)
        print(f"Only-analyze: loaded {n_eps} episodes, {n_total} frames from {raw_dir}")
    else:
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

                # ── 构建 obs（ft_history 模式需要 wrench_history）──
                obs={
                    "observation/state":np.asarray(frm["observation.state"],dtype=np.float32),
                    "observation/image":np.asarray(decode_image(frm["observation.image"])),
                    "observation/wrist_image":np.asarray(decode_image(frm["observation.wrist_image"])),
                    "prompt":"stamp seal",
                }
                # ft60 数据集新增了 wrench_history 列 [T,6]
                if "observation.wrench_history" in frm:
                    wh = np.asarray(frm["observation.wrench_history"], dtype=np.float32)
                    if wh.ndim == 1:
                        wh = wh.reshape(-1, 6)
                    obs["observation/wrench_history"] = wh
                else:
                    print(f"  WARN: no wrench_history in frame {fi}, using state force fallback")
                    obs["observation/wrench_history"] = np.zeros((60,6), dtype=np.float32)

                with _routing.frame() as rr: policy.infer(obs)
                if not rr: continue

                # ── 从 routing capture 提取数据 ──
                r=rr[0]
                eids=np.asarray(r["expert"][0]).astype(int)    # [seq_len] expert id
                probs=np.asarray(r["prob"][0])                  # [seq_len] gate prob
                sl=int(r["seq_length"])

                nv=V_TOKENS*2; l_end=sl-K_FORCE

                # ① 各模态 token 数量
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

                # ③ Expert 贡献 (sum of gate probs 近似)
                expert_contrib={}
                for e in range(4):
                    emask=eids==e
                    expert_contrib[f"e{e}"]=float(probs[emask].sum()) if emask.any() else 0.0

                # Router entropy per frame
                total_tokens=sl
                expert_fracs=np.array([expert_count[f"e{e}"]/max(total_tokens,1) for e in range(4)])
                valid=expert_fracs>0
                entropy=float(-np.sum(expert_fracs[valid]*np.log(expert_fracs[valid])))

                # ④ Force values
                force_vals=np.asarray(frm["observation.state"][7:13],dtype=np.float32)
                force_norm=float(np.linalg.norm(force_vals[:3]))
                force_norm_torque=float(np.linalg.norm(force_vals))

                frame_rec={
                    "ep":ep,"fi":fi,"progress":round((fi-s)/length,4),"sl":sl,
                    "n_vis":n_vis,"n_lang":n_lang,"n_force":n_force,
                    "vis_exp":{f"e{k}":v for k,v in vis_exp.items()},
                    "lang_exp":{f"e{k}":v for k,v in lang_exp.items()},
                    "force_exp":{f"e{k}":v for k,v in force_exp.items()},
                    "vis_prob":vis_prob_mean,"lang_prob":lang_prob_mean,"force_prob":force_prob_mean,
                    "expert_prob":expert_prob,
                    "expert_count":expert_count,
                    "expert_contrib":expert_contrib,
                    "entropy":entropy,
                    "expert_fracs":expert_fracs.tolist(),
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
    fig2=plt.figure(figsize=(16,5*max(n_eps,2)))
    gs2=gridspec.GridSpec(max(n_eps,2),2,figure=fig2,width_ratios=[3,1],hspace=0.4)

    ep_keys=list(ep_data.keys())
    for ep_i in range(n_eps):
        ep_frames=ep_data[ep_keys[ep_i]]
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
        ax_l.set_title(f"ep{ep_i}: Force Token Expert Share + Force Profile",fontsize=10)

        # 右: force 分桶 bar chart
        ax_r=fig2.add_subplot(gs2[ep_i,1])
        fn_arr=np.array(fn)
        bins=[0,2,5,10,20,50]
        labels=["0-2N","2-5N","5-10N","10-20N",">20N"]
        for e in range(4):
            vals=[]
            for b_i in range(len(bins)-1):
                mask=(fn_arr>=bins[b_i])&(fn_arr<bins[b_i+1])
                if mask.sum()==0: vals.append(0); continue
                shares=[]
                for fr in [ep_frames[i] for i in np.where(mask)[0]]:
                    d=fr["force_exp"]
                    total=max(sum(d.values()),1)
                    shares.append(d.get(f"e{e}",0)/total)
                vals.append(np.mean(shares)*100)
            ax_r.plot(range(len(bins)-1),vals,marker="o",color=EXPERT_COLORS[e],label=f"E{e}")
        ax_r.set_xticks(range(len(bins)-1)); ax_r.set_xticklabels(labels,fontsize=8)
        ax_r.set_ylabel("Force Token Expert Share (%)")
        ax_r.set_title("Force Bucket → Expert")
        ax_r.legend(fontsize=7)
    plt.suptitle(f"Force Strength → Expert Routing (Step {ckpt_step})",fontsize=13)
    plt.tight_layout()
    fig2.savefig(out_dir/f"fig2_force_expert_{ckpt_step}.png",dpi=150)
    plt.close(fig2)
    print(f"Fig2: {out_dir}/fig2_force_expert_{ckpt_step}.png")

    # ═══════════════════════════════════════════════
    # 📊 图 3: Expert 贡献 + Router 概率分布
    # ═══════════════════════════════════════════════
    fig3,axes3=plt.subplots(1,2,figsize=(14,5))

    # 左: Expert 贡献 (sum gate probs)
    ax=axes3[0]
    contrib=[0.0]*4
    for fr in all_frames:
        for e in range(4):
            contrib[e]+=fr["expert_contrib"].get(f"e{e}",0)
    total_c=sum(contrib) or 1
    vals=[c/total_c*100 for c in contrib]
    bars=ax.bar(range(4),vals,color=EXPERT_COLORS,edgecolor="white")
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{v:.1f}%",ha="center")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"E{e}" for e in range(4)])
    ax.set_title("Expert Contribution (Σ gate_prob)")
    ax.set_ylabel("Share (%)")

    # 右: force token router prob 分布
    ax=axes3[1]
    force_probs=[fr["force_prob"] for fr in all_frames if fr["force_prob"]>0]
    if force_probs:
        ax.hist(force_probs,bins=20,color="#2196F3",alpha=0.7,edgecolor="white")
        ax.axvline(np.mean(force_probs),color="red",ls="--",label=f"mean={np.mean(force_probs):.3f}")
        ax.legend()
    ax.set_xlabel("Force Token Router Prob")
    ax.set_ylabel("Count")
    ax.set_title("Force Token Router Prob Distribution")
    plt.suptitle(f"Expert Contribution & Router Prob (Step {ckpt_step})",fontsize=13)
    plt.tight_layout()
    fig3.savefig(out_dir/f"fig3_contrib_prob_{ckpt_step}.png",dpi=150)
    plt.close(fig3)
    print(f"Fig3: {out_dir}/fig3_contrib_prob_{ckpt_step}.png")

    # ═══════════════════════════════════════════════
    # 📊 表: Router 统计
    # ═══════════════════════════════════════════════
    lines=[]
    lines.append(f"=== MoE Router Statistics (Step {ckpt_step}, {n_total} frames) ===")
    lines.append("")
    lines.append("--- Overall Expert Distribution ---")
    overall=[0]*4
    for fr in all_frames:
        for e in range(4):
            overall[e]+=fr["expert_count"].get(f"e{e}",0)
    tot=sum(overall) or 1
    for e in range(4):
        lines.append(f"  E{e}: {overall[e]/tot*100:6.2f}%  ({overall[e]} tokens)")
    lines.append("")
    lines.append("--- Modality → Expert ---")
    for mod_key,mod_name in [("vis_exp","Vision"),("lang_exp","Language"),("force_exp","Force")]:
        agg=[0]*4
        for fr in all_frames:
            for e in range(4):
                agg[e]+=fr[mod_key].get(f"e{e}",0)
        t=sum(agg) or 1
        lines.append(f"  {mod_name}: " + "  ".join([f"E{e}={agg[e]/t*100:5.1f}%" for e in range(4)]))
    lines.append("")
    lines.append("--- Router Confidence ---")
    vp=[fr["vis_prob"] for fr in all_frames if fr["vis_prob"]>0]
    lp=[fr["lang_prob"] for fr in all_frames if fr["lang_prob"]>0]
    fp=[fr["force_prob"] for fr in all_frames if fr["force_prob"]>0]
    lines.append(f"  Vision mean gate prob: {np.mean(vp):.4f}" if vp else "  Vision: N/A")
    lines.append(f"  Lang   mean gate prob: {np.mean(lp):.4f}" if lp else "  Lang: N/A")
    lines.append(f"  Force  mean gate prob: {np.mean(fp):.4f}" if fp else "  Force: N/A")
    lines.append("")
    lines.append("--- Router Entropy ---")
    ents=[fr["entropy"] for fr in all_frames]
    lines.append(f"  mean={np.mean(ents):.3f}  min={np.min(ents):.3f}  max={np.max(ents):.3f}  (max possible for 4 experts={np.log(4):.3f})")
    lines.append("")
    lines.append("--- Force Levels → Force Token Expert (avg) ---")
    for b_i in range(len(bins)-1):
        mask=np.array([fr["force_norm"]>=bins[b_i] and fr["force_norm"]<bins[b_i+1] for fr in all_frames])
        if mask.sum()==0: continue
        agg=[0]*4
        for fr in [all_frames[i] for i in np.where(mask)[0]]:
            for e in range(4):
                agg[e]+=fr["force_exp"].get(f"e{e}",0)
        t=sum(agg) or 1
        lines.append(f"  {labels[b_i]:>8}: " + "  ".join([f"E{e}={agg[e]/t*100:5.1f}%" for e in range(4)]))

    report=out_dir/f"table_router_stats_{ckpt_step}.txt"
    with open(report,"w") as f: f.write("\n".join(lines))
    print(f"Table: {report}")

    print("\n=== DONE ===")


if __name__=="__main__":
    main()
