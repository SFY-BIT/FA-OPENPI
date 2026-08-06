#!/usr/bin/env python
"""MoE 专家路由分析 → PNG帧 + TXT报告 + 本地matplotlib浏览器

用法:
  cd /mnt/hdd/sfy/openpi-force
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python -u scripts/analyze_moe_full.py \
      --checkpoint checkpoints/12000 --num-episodes 4

输出:
  outputs/moe_analysis/moe_report_12000.txt
  outputs/moe_analysis/expert_share_12000.png     ← 汇总曲线
  outputs/moe_analysis/frames/ep0/00001.png ...   ← 逐帧图
  outputs/moe_analysis/viewer.py                  ← cd 进去 python viewer.py 浏览
"""

import argparse, io, json, os, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import moe_routing_capture as _routing
from openpi.shared import normalize as _normalize
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

K_FORCE = 2; V_TOKENS = 256
EXPERT_COLORS = ["#2196F3","#4CAF50","#FF9800","#F44336"]
MOD_COLORS = {"V":"#90CAF9","L":"#A5D6A7","F":"#EF9A9A"}


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


def draw_frame_png(out_path, img_base, force_vals, eids_full, sl, nv, l_end, ep, fi, ec_all, fe, ckpt_step):
    fig=plt.figure(figsize=(18,8))
    gs=GridSpec(2,2,figure=fig,width_ratios=[1,2.5],height_ratios=[1,1.5],hspace=0.3,wspace=0.2)

    ax_img=fig.add_subplot(gs[0,0]); ax_img.imshow(img_base)
    ax_img.set_title(f"Base Camera (ep{ep} f{fi})",fontsize=10); ax_img.axis("off")

    ax_f=fig.add_subplot(gs[1,0])
    labels=["Fx","Fy","Fz","Tx","Ty","Tz"]
    colors_f=["#e74c3c" if v>=0 else "#3498db" for v in force_vals]
    bars=ax_f.barh(range(6),force_vals,color=colors_f,height=0.6)
    ax_f.set_yticks(range(6)); ax_f.set_yticklabels(labels)
    ax_f.axvline(0,color="white",lw=0.8); ax_f.set_title("6-Axis Force",fontsize=10)
    for i,(bar,v) in enumerate(zip(bars,force_vals)):
        ax_f.text(v+(0.3 if v>=0 else -0.3),i,f"{v:.1f}",va="center",fontsize=8,
                  ha="left" if v>=0 else "right",color="white")

    n_show=min(sl,800); step=max(1,sl//n_show)
    eids=eids_full[::step][:n_show]; n_real=len(eids)
    nv_frac=nv/sl*n_real; l_frac=l_end/sl*n_real

    ax_mod=fig.add_subplot(gs[0,1])
    mc=[]; 
    for i in range(n_real):
        if i<nv_frac: mc.append(MOD_COLORS["V"])
        elif i<l_frac: mc.append(MOD_COLORS["L"])
        else: mc.append(MOD_COLORS["F"])
    ax_mod.bar(range(n_real),[1]*n_real,color=mc,width=1,edgecolor="none")
    ax_mod.set_xlim(0,n_real); ax_mod.set_ylim(0,1); ax_mod.set_yticks([])
    ax_mod.set_title(f"Token Modality (blue=V green=L red=F) | seq={sl}",fontsize=10)
    ax_mod.axvline(nv_frac,color="white",ls="--",lw=0.8,alpha=0.5)
    ax_mod.axvline(l_frac,color="white",ls="--",lw=0.8,alpha=0.5)
    ax_mod.text(nv_frac/2,0.85,"Vision",ha="center",fontsize=7,color="white")
    if l_frac>nv_frac: ax_mod.text((nv_frac+l_frac)/2,0.85,"Lang",ha="center",fontsize=7,color="white")
    ax_mod.text((l_frac+n_real)/2,0.85,"Force",ha="center",fontsize=7,color="white")

    ax_exp=fig.add_subplot(gs[1,1])
    ec=[EXPERT_COLORS[min(e,3)] for e in eids]
    ax_exp.bar(range(n_real),[1]*n_real,color=ec,width=1,edgecolor="none")
    ax_exp.set_xlim(0,n_real); ax_exp.set_ylim(0,1); ax_exp.set_yticks([])
    ax_exp.set_title("Token -> Expert (E0=blue E1=green E2=orange E3=red)",fontsize=10)
    ax_exp.axvline(nv_frac,color="white",ls="--",lw=0.8,alpha=0.5)
    ax_exp.axvline(l_frac,color="white",ls="--",lw=0.8,alpha=0.5)
    total=max(sum(ec_all),1)
    for e in range(4):
        ax_exp.text(n_real*(e+0.5)/4,-0.25,f"E{e}:{ec_all[e]/total*100:.0f}%",
                    ha="center",fontsize=8,color=EXPERT_COLORS[e])
    ax_exp.text(n_real*0.98,-0.25,f"F1->E{fe}",ha="right",fontsize=8,
                color=EXPERT_COLORS[fe],fontweight="bold")

    fig.suptitle(f"MoE Routing - Ep{ep} Frame {fi} (step {ckpt_step})",fontsize=12,y=0.98)
    plt.savefig(out_path,dpi=100,bbox_inches="tight",facecolor="#1a1a2e",edgecolor="none")
    plt.close(fig)


def write_viewer(viewer_path, all_metadata, ckpt_step):
    viewer_code=f'''#!/usr/bin/env python
"""MoE Routing local viewer. Keys: left/right arrows to navigate, digits 0-9 to switch episode, q to quit."""
import json,os; from pathlib import Path
import matplotlib; matplotlib.use("TkAgg"); import matplotlib.pyplot as plt

DATA_DIR=Path(__file__).parent
META={json.dumps(all_metadata)}; CKPT="{ckpt_step}"

class Viewer:
    def __init__(self):
        self.ep=0; self.idx=0
        self.fig,self.ax=plt.subplots(figsize=(14,8))
        self.fig.canvas.mpl_connect("key_press_event",self.on_key)
        self.load(); plt.show()
    def load(self):
        ep_meta=META[self.ep]; f=ep_meta["frames"][self.idx]
        path=DATA_DIR/f"frames/ep{{self.ep}}/{{f['fi']:05d}}.png"
        if not path.exists(): print(f"Not found: {{path}}"); return
        img=plt.imread(str(path))
        self.ax.clear(); self.ax.imshow(img); self.ax.axis("off")
        self.ax.set_title(
            f"Ep{{self.ep}} Frame {{f['fi']}} ({{self.idx+1}}/{{len(ep_meta['frames'])}})  "
            f"E0={{f['ec_all'][0]/max(sum(f['ec_all']),1)*100:.0f}}% "
            f"E1={{f['ec_all'][1]/max(sum(f['ec_all']),1)*100:.0f}}% "
            f"E2={{f['ec_all'][2]/max(sum(f['ec_all']),1)*100:.0f}}% "
            f"E3={{f['ec_all'][3]/max(sum(f['ec_all']),1)*100:.0f}}%  F1->E{{f['fe']}}",
            fontsize=10,color="white")
        self.fig.patch.set_facecolor("#1a1a2e"); self.fig.canvas.draw()
    def on_key(self,event):
        ep_meta=META[self.ep]
        if event.key=="right": self.idx=min(self.idx+1,len(ep_meta["frames"])-1)
        elif event.key=="left": self.idx=max(self.idx-1,0)
        elif event.key in "0123456789":
            ep=int(event.key)
            if ep<len(META): self.ep=ep; self.idx=0
        elif event.key=="q": plt.close(); return
        elif event.key=="escape": plt.close(); return
        self.load()
if __name__=="__main__": Viewer()
'''
    with open(viewer_path,"w") as f: f.write(viewer_code)
    os.chmod(viewer_path,0o755)


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
    ckpt_step=Path(args.checkpoint).name

    print(f"Dataset: {args.dataset}")
    ds=lerobot_dataset.LeRobotDataset(args.dataset)
    ep_idx=ds.episode_data_index
    n_eps=min(args.num_episodes,len(ep_idx["from"])); eps=list(range(n_eps))
    print(f"  {len(ep_idx['from'])} episodes, using {n_eps}")

    norm_stats=None
    if Path(args.norm_stats,"norm_stats.json").exists():
        norm_stats=_normalize.load(args.norm_stats)
        # 推理时没有 force_target，删掉避免 Unnormalize 报错
        norm_stats.pop("force_target", None)
        print("  norm_stats loaded")

    print(f"Loading policy: {args.config}")
    _routing.enable()
    cfg=_config.get_config(args.config)
    policy=_policy_config.create_trained_policy(cfg,args.checkpoint,norm_stats=norm_stats)
    policy._has_flow_breakdown=False
    print("  First JIT..."); sys.stdout.flush()

    all_data=[]; all_metadata=[]; total_frames=0

    for ep in eps:
        s,e=int(ep_idx["from"][ep]),int(ep_idx["to"][ep])
        length=e-s
        max_f=length if args.max_frames<=0 else min(args.max_frames,length)
        step=max(1,length//max_f)
        idxs=list(range(s,e,step))[:max_f]
        ep_frames=[]; ep_meta={"ep":ep,"frames":[]}
        frame_dir=out_dir/"frames"/f"ep{ep}"; os.makedirs(frame_dir,exist_ok=True)

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
            ec_all=[int((eids[:sl]==e).sum()) for e in range(4)]; fe=int(eids[sl-1])
            force_vals=np.asarray(frm["observation.state"][7:13],dtype=np.float32)
            img_base=np.asarray(decode_image(frm["observation.image"]))

            png_path=frame_dir/f"{fi:05d}.png"
            draw_frame_png(str(png_path),img_base,force_vals,eids,sl,nv,l_end,ep,fi,ec_all,fe,ckpt_step)

            ep_frames.append({"fi":fi,"rp":round((fi-s)/length,4),"sl":sl,"ec_all":ec_all,"fe":fe})
            ep_meta["frames"].append({"fi":fi,"ec_all":ec_all,"fe":fe})
            total_frames+=1

        all_data.append(ep_frames); all_metadata.append(ep_meta)
        print(f"  Ep{ep}: done ({len(ep_frames)} frames)"+" "*30)

    if total_frames==0: print("ERROR: No routing data!"); sys.exit(1)
    print(f"\nTotal: {total_frames} frames")

    # 汇总曲线
    nep_plot=len(all_data)
    fig,axes=plt.subplots(nep_plot,1,figsize=(14,3*nep_plot))
    if nep_plot==1: axes=[axes]
    for ep_i,recs in enumerate(all_data):
        xs=[r["rp"] for r in recs]
        for e in range(4):
            ys=[r["ec_all"][e]/max(sum(r["ec_all"]),1)*100 for r in recs]
            axes[ep_i].plot(xs,ys,color=EXPERT_COLORS[e],lw=1,alpha=.8,label=f"E{e}")
        axes[ep_i].axhline(25,color="gray",ls="--",alpha=.3)
        axes[ep_i].set_xlabel("Progress"); axes[ep_i].set_ylabel("Share (%)")
        axes[ep_i].set_title(f"Episode {ep_i} ({len(recs)} frames)")
        axes[ep_i].legend(fontsize=7,ncol=4); axes[ep_i].set_ylim(0,70)
    plt.suptitle(f"MoE Expert Share (step {ckpt_step})"); plt.tight_layout()
    plt.savefig(out_dir/f"expert_share_{ckpt_step}.png",dpi=150); plt.close()
    print(f"Summary: {out_dir}/expert_share_{ckpt_step}.png")

    # TXT 报告
    txt_path=out_dir/f"moe_report_{ckpt_step}.txt"
    with open(txt_path,"w",encoding="utf-8") as f:
        all_frames=[r for ep_data in all_data for r in ep_data]
        all_ec=np.array([r["ec_all"] for r in all_frames]); mn=all_ec.mean(0); tot=mn.sum()
        f.write("="*60+f"\nPi0Force MoE Routing - Step {ckpt_step}\nEpisodes: {n_eps}, Frames: {total_frames}\n"+"="*60+"\n\n")
        f.write("Overall Expert Distribution:\n")
        for e in range(4): f.write(f"  E{e}: {mn[e]/tot*100:5.1f}% ({mn[e]:.0f}/{tot:.0f})\n")
        all_fe=[r["fe"] for r in all_frames]; fc=Counter(all_fe)
        f.write("\nForce Token (F1) Routing:\n")
        for e in range(4): f.write(f"  ->E{e}: {fc.get(e,0)}/{len(all_fe)} ({fc.get(e,0)/len(all_fe)*100:.1f}%)\n")
        f.write("\nPer-Episode:\n")
        for ep_i,recs in enumerate(all_data):
            ec_ep=np.array([r["ec_all"] for r in recs]); mn_ep=ec_ep.mean(0)
            f.write(f"  Ep{ep_i}: E0={mn_ep[0]/mn_ep.sum()*100:.1f}% E1={mn_ep[1]/mn_ep.sum()*100:.1f}% E2={mn_ep[2]/mn_ep.sum()*100:.1f}% E3={mn_ep[3]/mn_ep.sum()*100:.1f}%\n")
        f.write("\nConclusion: ")
        if tot>0:
            mp=max(mn)/tot*100; de=list(mn).index(max(mn))
            f.write(f"E{de} dominates ({mp:.0f}%). "+("Highly unbalanced.\n" if mp>70 else "Somewhat balanced.\n"))
    print(f"Report: {txt_path}")

    # 本地浏览器
    viewer_path=out_dir/"viewer.py"
    write_viewer(str(viewer_path),all_metadata,ckpt_step)
    print(f"Viewer: {viewer_path}")
    print(f"\nDone. Browse: cd {out_dir} && python viewer.py  (arrows:nav  digits:ep  q:quit)")

if __name__=="__main__": main()
