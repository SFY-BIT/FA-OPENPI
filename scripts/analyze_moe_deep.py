#!/usr/bin/env python
"""深度分析：轨迹级别 expert 演化 + force 关联 + token-type×expert 条件分布

用法: 先跑 analyze_moe_full.py 产出一轮结果，再跑本脚本
  cd /mnt/hdd/sfy/openpi-force
  source /home/sfy/miniconda3/etc/profile.d/conda.sh && conda activate rlinf
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python -u scripts/analyze_moe_deep.py \
      --checkpoint checkpoints/12000 --num-episodes 4 \
      --output-dir outputs/moe_analysis
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
    raise TypeError(f"Unknown image: {type(img_data)}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="pi05_force_stamp_seal_remote")
    p.add_argument("--checkpoint",default="checkpoints/12000")
    p.add_argument("--dataset",default="/mnt/hdd/sfy/lerobot_datasets/stamp_seal_flexiv")
    p.add_argument("--norm-stats",default="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv")
    p.add_argument("--num-episodes",type=int,default=4)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--output-dir",default="outputs/moe_analysis")
    args=p.parse_args()

    out_dir=Path(args.output_dir); os.makedirs(out_dir,exist_ok=True)
    ckpt_step=Path(args.checkpoint).name; raw_dir=out_dir/"raw_data"; os.makedirs(raw_dir,exist_ok=True)

    print(f"Dataset: {args.dataset}")
    ds=lerobot_dataset.LeRobotDataset(args.dataset)
    ep_idx=ds.episode_data_index
    n_eps=min(args.num_episodes,len(ep_idx["from"])); eps=list(range(n_eps))
    print(f"  {len(ep_idx['from'])} episodes, using {n_eps}")

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

    all_data=[]; total_frames=0

    for ep in eps:
        s,e=int(ep_idx["from"][ep]),int(ep_idx["to"][ep])
        length=e-s
        max_f=length if args.max_frames<=0 else min(args.max_frames,length)
        step=max(1,length//max_f)
        idxs=list(range(s,e,step))[:max_f]
        ep_records=[]

        print(f"\nEp{ep}: {length} frames, sampling {len(idxs)}"); sys.stdout.flush()

        for i,fi in enumerate(idxs):
            if i>0 and i%50==0: print(f"  {i}/{len(idxs)}",end="\r"); sys.stdout.flush()

            frm=ds[fi]
            obs={
                "observation/state":np.asarray(frm["observation.state"],dtype=np.float32),
                "observation/image":np.asarray(decode_image(frm["observation.image"])),
                "observation/wrist_image":np.asarray(decode_image(frm["observation.wrist_image"])),
                "prompt":"stamp seal",
            }

            with _routing.frame() as rr: policy.infer(obs)

            if not rr: continue

            r=rr[0]; eids=np.asarray(r["expert"][0]).astype(int); sl=int(r["seq_length"])
            probs=np.asarray(r["prob"][0]); nv=V_TOKENS*2; l_end=sl-K_FORCE

            # ── 精细记录 ──
            force_vals=np.asarray(frm["observation.state"][7:13],dtype=np.float32)
            proprio=np.asarray(frm["observation.state"][:7],dtype=np.float32)
            progress=(fi-s)/length
            force_norm_xyz=float(np.linalg.norm(force_vals[:3]))
            force_norm_all=float(np.linalg.norm(force_vals))

            # 各模态各 expert 计数
            vis_eids = eids[:nv] if nv<=sl else eids[:sl]
            lang_eids = eids[nv:l_end] if l_end>nv else np.array([],dtype=int)
            force_eids = eids[l_end:sl]

            record={
                "ep":ep,"fi":fi,"progress":round(progress,4),"sl":sl,
                # Per-modality per-expert counts
                "vis":  {f"e{e}":int((vis_eids==e).sum()) for e in range(4)},
                "lang": {f"e{e}":int((lang_eids==e).sum()) for e in range(4)} if len(lang_eids)>0 else None,
                "force":{f"e{e}":int((force_eids==e).sum()) for e in range(4)},
                # Totals
                "ec_all":[int((eids[:sl]==e).sum()) for e in range(4)],
                # Force values
                "fx":float(force_vals[0]),"fy":float(force_vals[1]),"fz":float(force_vals[2]),
                "tx":float(force_vals[3]),"ty":float(force_vals[4]),"tz":float(force_vals[5]),
                "force_norm_xyz":force_norm_xyz,"force_norm_all":force_norm_all,
                # Proprio
                "gripper":float(proprio[6]),
                # Router confidence for force tokens
                "fp_last":float(probs[sl-1]) if sl>0 else 0,
                "fe_last":int(eids[sl-1]) if sl>0 else -1,
                "fp_2nd":float(probs[sl-2]) if sl>=2 else 0,
                "fe_2nd":int(eids[sl-2]) if sl>=2 else -1,
            }
            ep_records.append(record)
            total_frames+=1

        all_data.append(ep_records)
        # 保存原始 JSON
        json_path=raw_dir/f"ep{ep}_deep.json"
        with open(json_path,"w") as f: json.dump(ep_records,f,indent=2)
        print(f"  Ep{ep}: {len(ep_records)} frames -> {json_path}"+" "*30)

    if total_frames==0: print("ERROR: No data!"); sys.exit(1)
    print(f"\nTotal: {total_frames} frames")

    # ══════════════════════════════════════════════════════
    # 深度分析 1: 轨迹级 Expert 演化 + Force 关联
    # ══════════════════════════════════════════════════════
    fig=plt.figure(figsize=(16,4*n_eps))
    gs=gridspec.GridSpec(n_eps,2,figure=fig,width_ratios=[3,1],hspace=0.4,wspace=0.3)

    for ep_i,recs in enumerate(all_data):
        xs=[r["progress"] for r in recs]

        # 左侧: expert share + force over trajectory
        ax1=fig.add_subplot(gs[ep_i,0])
        for e in range(4):
            ys=[r["ec_all"][e]/max(sum(r["ec_all"]),1)*100 for r in recs]
            ax1.plot(xs,ys,color=EXPERT_COLORS[e],lw=1,alpha=0.8,label=f"E{e}")
        ax1.set_ylabel("Expert Share (%)",color="#ccc")
        ax1.set_ylim(0,70)
        ax1.axhline(25,color="gray",ls="--",alpha=0.2)
        ax1.legend(fontsize=7,ncol=4,loc="upper right")

        # Force on twin axis
        ax1f=ax1.twinx()
        fz=[r["fz"] for r in recs]
        fn=[r["force_norm_xyz"] for r in recs]
        ax1f.fill_between(xs,0,fn,alpha=0.15,color="#e74c3c",label="|F|")
        ax1f.plot(xs,fz,color="#e74c3c",lw=0.8,alpha=0.6,label="Fz")
        ax1f.set_ylabel("Force (N/Nm)",color="#e74c3c")
        ax1f.legend(fontsize=7,loc="upper left")
        ax1.set_title(f"Episode {ep_i}: Expert Share + Force Profile",fontsize=11)

        # 右侧: Token-Type × Expert 条件分布
        ax2=fig.add_subplot(gs[ep_i,1])
        token_types=["Vis","Lang","Force"]
        x_pos=np.arange(3); bar_w=0.2
        for e in range(4):
            means=[]
            for tt in token_types:
                key={"Vis":"vis","Lang":"lang","Force":"force"}[tt]
                vals=[]
                for r in recs:
                    d=r.get(key)
                    if d:
                        total_tt=sum(d.values())
                        if total_tt>0: vals.append(d[f"e{e}"]/total_tt*100)
                means.append(np.mean(vals) if vals else 0)
            ax2.bar(x_pos+e*bar_w,means,bar_w,color=EXPERT_COLORS[e],label=f"E{e}",alpha=0.85)
        ax2.set_xticks(x_pos+1.5*bar_w); ax2.set_xticklabels(token_types)
        ax2.set_ylabel("Expert Share (%)")
        ax2.set_title("Token-Type × Expert (avg)",fontsize=10)
        ax2.legend(fontsize=6,ncol=4)

    plt.suptitle(f"MoE Deep Analysis — Step {ckpt_step}",fontsize=13,y=1.01)
    plt.savefig(out_dir/f"deep_analysis_{ckpt_step}.png",dpi=150,bbox_inches="tight")
    plt.close()
    print(f"Deep analysis plot: {out_dir}/deep_analysis_{ckpt_step}.png")

    # ══════════════════════════════════════════════════════
    # 深度分析 2: Force 分桶分析 — 不同力级下的 expert 行为
    # ══════════════════════════════════════════════════════
    all_recs=[r for ep_data in all_data for r in ep_data]
    fz_vals=np.array([r["fz"] for r in all_recs])
    fn_vals=np.array([r["force_norm_xyz"] for r in all_recs])

    # 按 force magnitude 分桶
    bins=[0,2,5,10,20,50,100]
    labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    digitized=np.digitize(fn_vals,bins)-1
    digitized=np.clip(digitized,0,len(bins)-2)

    bucket_data=defaultdict(list)
    for i,r in enumerate(all_recs):
        b=digitized[i]
        if b>=0 and b<len(labels):
            bucket_data[b].append(r)

    fig2,axes2=plt.subplots(2,1,figsize=(14,10))

    # 2a: Force token expert distribution by force bucket
    ax2a=axes2[0]
    x_pos=np.arange(len(labels)); bar_w=0.2
    for e in range(4):
        means=[]
        for b in range(len(labels)):
            brecs=bucket_data.get(b,[])
            if brecs:
                vals=[r["force"][f"e{e}"]/max(sum(r["force"].values()),1)*100 for r in brecs]
                means.append(np.mean(vals))
            else:
                means.append(0)
        ax2a.bar(x_pos+e*bar_w,means,bar_w,color=EXPERT_COLORS[e],label=f"E{e}")
    ax2a.set_xticks(x_pos+1.5*bar_w); ax2a.set_xticklabels(labels)
    ax2a.set_xlabel("Force Magnitude |F_xyz| (N)"); ax2a.set_ylabel("Force Token Expert Share (%)")
    ax2a.set_title("Force Token → Expert Distribution by Force Level"); ax2a.legend(ncol=4)
    ax2a.axhline(0,color="white",lw=0.5)

    # 2b: Overall expert share by force bucket
    ax2b=axes2[1]
    for e in range(4):
        means=[]
        for b in range(len(labels)):
            brecs=bucket_data.get(b,[])
            if brecs:
                vals=[r["ec_all"][e]/max(sum(r["ec_all"]),1)*100 for r in brecs]
                means.append(np.mean(vals))
            else:
                means.append(0)
        ax2b.bar(x_pos+e*bar_w,means,bar_w,color=EXPERT_COLORS[e],label=f"E{e}")
    ax2b.set_xticks(x_pos+1.5*bar_w); ax2b.set_xticklabels(labels)
    ax2b.set_xlabel("Force Magnitude |F_xyz| (N)"); ax2b.set_ylabel("Overall Expert Share (%)")
    ax2b.set_title("Overall Expert Distribution by Force Level"); ax2b.legend(ncol=4)

    plt.suptitle(f"Force-Level Analysis — Step {ckpt_step}",fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir/f"force_bucket_{ckpt_step}.png",dpi=150,bbox_inches="tight")
    plt.close()
    print(f"Force bucket plot: {out_dir}/force_bucket_{ckpt_step}.png")

    # ══════════════════════════════════════════════════════
    # 深度分析 3: TXT 详细报告
    # ══════════════════════════════════════════════════════
    report_path=out_dir/f"deep_report_{ckpt_step}.txt"
    with open(report_path,"w",encoding="utf-8") as f:
        f.write("="*70+"\n")
        f.write(f"MoE Deep Analysis Report — Step {ckpt_step}\n")
        f.write(f"Frames: {total_frames}, Episodes: {n_eps}\n")
        f.write("="*70+"\n\n")

        # Token-Type × Expert 条件分布
        f.write("─"*50+"\n")
        f.write("1. Token-Type × Expert Conditional Distribution\n")
        f.write("─"*50+"\n\n")
        for tt,key in [("Vision","vis"),("Language","lang"),("Force","force")]:
            all_e=[]
            for r in all_recs:
                d=r.get(key)
                if d:
                    total=max(sum(d.values()),1)
                    for e in range(4):
                        all_e.append(d[f"e{e}"]/total*100)
            if all_e:
                f.write(f"  {tt} tokens:\n")
                for e in range(4):
                    vals=[r.get(key,{}).get(f"e{e}",0)/max(sum(r.get(key,{}).values()),1)*100
                          for r in all_recs if r.get(key)]
                    if vals:
                        f.write(f"    E{e}: {np.mean(vals):5.1f}% ±{np.std(vals):.1f}%  "
                                f"[{np.min(vals):.0f}-{np.max(vals):.0f}] N={len(vals)}\n")
                f.write("\n")

        # Force 分桶分析
        f.write("─"*50+"\n")
        f.write("2. Force-Level Bucket Analysis\n")
        f.write("─"*50+"\n\n")
        for b in range(len(labels)):
            brecs=bucket_data.get(b,[])
            if not brecs: continue
            f.write(f"  |F| in [{labels[b]}] N: {len(brecs)} frames\n")
            force_e={e:[] for e in range(4)}
            for r in brecs:
                for e in range(4):
                    force_e[e].append(r["force"][f"e{e}"]/max(sum(r["force"].values()),1)*100)
            for e in range(4):
                if force_e[e]:
                    f.write(f"    Force→E{e}: {np.mean(force_e[e]):5.1f}% ±{np.std(force_e[e]):.1f}%\n")
            f.write("\n")

        # 每个 episode 的 dominant expert 切换
        f.write("─"*50+"\n")
        f.write("3. Per-Episode Dominant Expert Switching\n")
        f.write("─"*50+"\n\n")
        for ep_i,recs in enumerate(all_data):
            f.write(f"  Episode {ep_i} ({len(recs)} frames):\n")
            # 检测主导 expert 切换（连续 N 帧主导变化）
            dominants=[]
            for r in recs:
                ec=r["ec_all"]; dom=np.argmax(ec); dominants.append(dom)
            # 找切换点
            switches=[]
            prev=dominants[0]
            for i,d in enumerate(dominants):
                if d!=prev:
                    switches.append((i,recs[i]["progress"],prev,d))
                    prev=d
            if switches:
                f.write(f"    Switches detected: {len(switches)}\n")
                for si,sp,sf,st in switches[:20]:  # 最多显示 20 个
                    f.write(f"      frame ~{si:4d} (progress {sp:.2f}): E{sf} → E{st}\n")
            else:
                f.write(f"    No dominant expert switches\n")

            # Force token routing stability
            fe_hist=[r["fe_last"] for r in recs]
            fe_counts=Counter(fe_hist)
            f.write(f"    Force token (F1) routing stability:\n")
            for e in range(4):
                if fe_counts.get(e,0)>0:
                    f.write(f"      →E{e}: {fe_counts[e]}/{len(recs)} ({fe_counts[e]/len(recs)*100:.1f}%)\n")
            f.write("\n")

        f.write("─"*50+"\n")
        f.write("4. Conclusions\n")
        f.write("─"*50+"\n\n")
        f.write("  Key question: Does MoE routing change with force/task phase?\n\n")

        # Check if force bucket changes expert distribution
        bucket_expert_means={}
        for b in range(len(labels)):
            brecs=bucket_data.get(b,[])
            if not brecs: continue
            for e in range(4):
                if e not in bucket_expert_means: bucket_expert_means[e]=[]
                bucket_expert_means[e].append(
                    np.mean([r["ec_all"][e]/max(sum(r["ec_all"]),1)*100 for r in brecs]))

        f.write("  Force level vs Expert share correlation:\n")
        for e in range(4):
            if e in bucket_expert_means and len(bucket_expert_means[e])>1:
                change=max(bucket_expert_means[e])-min(bucket_expert_means[e])
                f.write(f"    E{e}: varies {change:.1f}% across force levels "
                        f"[{min(bucket_expert_means[e]):.1f}-{max(bucket_expert_means[e]):.1f}]%")

    print(f"Deep report: {report_path}")
    print(f"\nDone. Raw data: {raw_dir}/")
    print(f"  {out_dir}/deep_analysis_{ckpt_step}.png  — trajectory overview")
    print(f"  {out_dir}/force_bucket_{ckpt_step}.png   — force-level analysis")
    print(f"  {out_dir}/deep_report_{ckpt_step}.txt    — detailed report")


if __name__=="__main__": main()
