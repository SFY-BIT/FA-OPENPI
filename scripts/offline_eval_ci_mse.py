"""离线评估: CI-MSE 风格 chunk 评估 (ICRA 论文, force-aware VLA).

对 eval 验证集, 只推理 value=1 关键段 (段内 stride=2 抽样), 每帧得到完整
[30,7] 动作 chunk, 与真值 chunk (action[t:t+30], 尾部越界重复最后 action 扩充)
对比. 输出三套独立指标:
  - joint:  joint 7 维直接 MSE/MAE
  - EEF:    pred/gt chunk 用 piper_fk_jax.fk 转 EEF 位姿 (xyz+rot6d) 再比
  - force:  force_pred chunk vs force_gt chunk (force[t:t+30])
支持 temporal ensemble (多帧 chunk 加权, pred/force_pred 都做), 与 CI-MSE 一致.
ablate=force 时 wrench_history 全 0 (等价真机 noforce 变体).

用法:
  python offline_eval_ci_mse.py --task peel --model peel_05_eef --port 8000 \
      --out-dir offline_eval_results --ablate force
  # 或 --all 批量 (读取 EVAL_MATRIX)

协议 (与真机 record_ws_dataset.py 一致):
  connect → recv metadata → send {state[7], wrench_history[60,6], images, prompt}
  → recv {actions[30,7], force_pred[30,6]?}
"""
import argparse
import glob
import json
import sys
from collections import deque
from pathlib import Path

# 崩溃时不丢缓冲输出 (此前子进程日志全空 = 缓冲未刷)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

# openpi_client msgpack (同 record_ws_dataset.py)
for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    cand = parent / "packages" / "src"
    if (cand / "openpi_client").is_dir():
        sys.path.insert(0, str(cand))
        break
import websockets.sync.client
from openpi_client import msgpack_numpy

from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from openpi.models import piper_fk_jax as _jfk

FT_HISTORY_STEPS = 60
FORCE_DIM = 6
ACTION_DIM = 7
HORIZON = 30
TOOL_EXT = 0.211

EVAL_DATASETS = {
    "peel": "/mnt/hdd/sfy/FA-VLA-结论/checkpoints/test_peel/peel_cucumber_eval_dataset",
    "board": "/mnt/hdd/sfy/FA-VLA-结论/checkpoints/test_board/Erase_Board_eval_dataset",
    "bottle": "/mnt/hdd/sfy/FA-VLA-结论/checkpoints/test_bottle/Pump_bottle_eval_dataset",
}
TASK_PROMPTS = {
    "peel": "peel cucumber",
    "board": "Erase Board",
    "bottle": "Pump bottle",
}


def decode_image(d):
    if isinstance(d, dict):
        b = d.get("bytes")
        if not b:
            return np.zeros((480, 640, 3), np.uint8)
        d = b
    try:
        return np.asarray(Image.open(io.BytesIO(d)).convert("RGB"))
    except Exception:
        return np.zeros((480, 640, 3), np.uint8)


def build_wrench_history(force_buf: deque, ablate: bool) -> np.ndarray:
    if ablate:
        return np.zeros((FT_HISTORY_STEPS, FORCE_DIM), dtype=np.float32)
    frames = list(force_buf)
    if not frames:
        return np.zeros((FT_HISTORY_STEPS, FORCE_DIM), dtype=np.float32)
    if len(frames) >= FT_HISTORY_STEPS:
        return np.stack(frames[-FT_HISTORY_STEPS:], axis=0)
    pad = np.zeros((FT_HISTORY_STEPS - len(frames), FORCE_DIM), dtype=np.float32)
    return np.concatenate([pad, np.stack(frames, axis=0)], axis=0)


def temporal_ensemble(preds: np.ndarray, K: int) -> np.ndarray:
    """CI-MSE temporal ensemble: a_hat[t,h] = (1/m) sum_{m=0}^{m-1} a[t-m, h+m].
    preds: [T, H, D]. K=1 → no-op. 未推理帧 (全 NaN 行) 保持 NaN 不参与."""
    if K <= 1:
        return preds
    T, H, D = preds.shape
    out = np.zeros_like(preds)
    cnt = np.zeros((T, H), dtype=np.float32)
    for m in range(min(K, H, T)):
        valid = ~np.isnan(preds[m:, : H - m, 0])[..., None]  # [T-m, H-m, 1]
        out[m:, : H - m] = np.where(
            valid, out[m:, : H - m] + preds[m:, : H - m], out[m:, : H - m]
        )
        cnt[m:, : H - m] += valid[..., 0].astype(np.float32)
    cnt = np.maximum(cnt, 1)
    out /= cnt[..., None]
    # 未推理帧保持 NaN (aggregate 时跳过)
    row_valid = ~np.isnan(preds[:, 0, 0])
    out[~row_valid] = np.nan
    return out


def load_episode(ep_parquet: str):
    import pyarrow.parquet as pq
    t = pq.read_table(ep_parquet)
    st = np.stack(t.column("observation.state").to_pylist()).astype(np.float32)
    ac = np.stack(t.column("action").to_pylist()).astype(np.float32)
    fc = np.stack(t.column("observation.force").to_pylist()).astype(np.float32)
    val = np.array(t.column("observation.value").to_pylist(), dtype=np.int64)
    img = t.column("observation.images.one").to_pylist()
    wrist = t.column("observation.images.two").to_pylist()
    return st, ac, fc, val, img, wrist


def fk_to_eef(joint_chunk: np.ndarray) -> np.ndarray:
    """joint chunk [H,7] -> EEF 位姿 [H,9] (xyz+rot6d, tool_ext 0.211)."""
    import jax.numpy as jnp
    q = joint_chunk[:, :6].astype(np.float32)
    T = _jfk.fk_batch(jnp.asarray(q), TOOL_EXT)
    xyz = np.asarray(T[..., :3, 3])
    R = np.asarray(T[..., :3, :3])
    d6 = R[:, :2, :].reshape(-1, 6)  # rot6d 前两行
    return np.concatenate([xyz, d6], axis=-1)


class ServerClient:
    """websocket client, 加大握手超时 + 自动重试 (首帧 JIT 编译会阻塞事件循环,
    同 server 并发连接可能握手超时)."""

    def __init__(self, port: int, open_timeout: float = 120.0, retries: int = 3):
        self.port = port
        self.url = f"ws://127.0.0.1:{port}"
        self.packer = msgpack_numpy.Packer()
        self.open_timeout = open_timeout
        self.retries = retries

    def _infer_once(self, obs: dict) -> dict:
        with websockets.sync.client.connect(
            self.url, max_size=300 * 1024 * 1024, open_timeout=self.open_timeout
        ) as ws:
            ws.recv()  # metadata
            ws.send(self.packer.pack(obs))
            raw = ws.recv()
        return msgpack_numpy.unpackb(raw)

    def infer(self, obs: dict) -> dict:
        import time
        last = None
        for attempt in range(self.retries):
            try:
                return self._infer_once(obs)
            except Exception as e:
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"port {self.port} infer failed after {self.retries} attempts: {last}")


def run_model(task: str, port: int, out_path: Path, ablate: bool,
              stride: int = 2, ensemble_horizon: int = 4,
              max_episodes: int | None = None) -> dict:
    """对一个模型跑 eval 验证集 (只 V=1 段, stride 抽样), 返回聚合指标. 逐帧明细写 jsonl."""
    ds = EVAL_DATASETS[task]
    fs = sorted(glob.glob(f"{ds}/data/chunk-000/episode_*.parquet"))
    if max_episodes is not None:
        fs = fs[:max_episodes]
    client = ServerClient(port)

    rows = []  # 每帧明细
    # 每 ep 的 (frame, pred_chunk, force_pred_chunk) 用于 ensemble + 指标
    ep_data = []  # list of dict per ep

    infer_set_cache = {}
    for ep_i, f in enumerate(fs):
        try:
            st, ac, fc, val, img, wrist = load_episode(f)
            T = len(st)
            # 找 V=1 连续段
            in_seg = False
            seg_frames = []
            for t in range(T):
                if val[t] == 1 and not in_seg:
                    in_seg = True
                    seg_start = t
                if val[t] == 0 and in_seg:
                    in_seg = False
                    seg_frames.append(list(range(seg_start, t)))
            if in_seg:
                seg_frames.append(list(range(seg_start, T)))
            # 段内 stride 抽样
            infer_frames = []
            for seg in seg_frames:
                infer_frames.extend(seg[::stride])
            infer_set = set(infer_frames)

            preds = np.full((T, HORIZON, ACTION_DIM), np.nan, dtype=np.float32)
            fpreds = np.full((T, HORIZON, FORCE_DIM), np.nan, dtype=np.float32)

            force_buf = deque(maxlen=FT_HISTORY_STEPS)
            for t in range(T):
                if t in infer_set:
                    obs = {
                        "observation/image": decode_image(img[t]),
                        "observation/wrist_image": decode_image(wrist[t]),
                        "observation/state": st[t, :7].tolist(),
                        "observation/wrench_history": build_wrench_history(force_buf, ablate),
                        "prompt": TASK_PROMPTS[task],
                    }
                    resp = client.infer(obs)
                    preds[t] = np.asarray(resp["actions"])[:HORIZON, :ACTION_DIM]
                    if "force_pred" in resp and resp["force_pred"] is not None:
                        fpreds[t] = np.asarray(resp["force_pred"])[:HORIZON, :FORCE_DIM]
                    else:
                        fpreds[t] = 0.0
                force_buf.append(fc[t])  # 每帧都进历史 (与推理无关, 与真机采集一致)

            # temporal ensemble (对已推帧)
            preds_ens = temporal_ensemble(preds, ensemble_horizon)
            fpreds_ens = temporal_ensemble(fpreds, ensemble_horizon)

            # target chunk (越界重复最后 action 扩充)
            gt_chunk = np.zeros((T, HORIZON, ACTION_DIM), dtype=np.float32)
            fgt_chunk = np.zeros((T, HORIZON, FORCE_DIM), dtype=np.float32)
            for t in range(T):
                rem = T - t
                if rem >= HORIZON:
                    gt_chunk[t] = ac[t : t + HORIZON]
                    fgt_chunk[t] = fc[t : t + HORIZON]
                else:
                    base = ac[t:T]
                    tail = np.tile(base[-1], (HORIZON - rem, 1))
                    gt_chunk[t] = np.concatenate([base, tail], axis=0)
                    base_f = fc[t:T]
                    tail_f = np.tile(base_f[-1], (HORIZON - rem, 1))
                    fgt_chunk[t] = np.concatenate([base_f, tail_f], axis=0)

            # 只对 V=1 且已推帧算误差
            for t in infer_frames:
                if val[t] != 1:
                    continue
                if np.isnan(preds_ens[t]).any():
                    continue
                rows.append({
                    "episode": ep_i, "frame": t, "value": int(val[t]),
                    "pred_joint": preds_ens[t].tolist(),
                    "gt_joint": gt_chunk[t].tolist(),
                    "pred_eef": fk_to_eef(preds_ens[t]).tolist(),
                    "gt_eef": fk_to_eef(gt_chunk[t]).tolist(),
                    "pred_force": fpreds_ens[t].tolist(),
                    "gt_force": fgt_chunk[t].tolist(),
                    "force_norm": float(np.linalg.norm(fc[t])),
                })
        except Exception as e:
            # 单集异常不中断整个任务
            print(f"  [warn] episode {ep_i} 处理失败, 跳过: {type(e).__name__}: {e}")
            continue
    print(f"  [{task}] 完成 {len(fs)} 个 episode, 有效帧 {len(rows)}")

    # 写 jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # 聚合
    agg = aggregate(rows)
    return agg


def _mse_mae(pred, gt):
    d = np.asarray(pred) - np.asarray(gt)
    return float(np.mean(d ** 2)), float(np.mean(np.abs(d)))


def aggregate(rows):
    """每模型独立聚合. 输出:
      joint_mse/mae, eef_mse/mae (综合)
      force_mse/mae (综合) + force_mse_by_force (分桶, 带样本数, 加权可还原综合)
    """
    res = {"n_frames": len(rows)}
    jm, jma, em, ema = [], [], [], []
    fm, fma = [], []
    # 关节分组 MSE (Q1-Q3 = joint dim 0-2, Q4-Q6 = joint dim 3-5, 不含 gripper)
    j_q1q3, j_q4q6 = [], []
    j_q1q3_n, j_q4q6_n = [], []  # 归一化空间 (norm_stats actions q01/q99 前 6 维)
    q_scale = None
    # 力幅分桶: 每桶存 (mse_sum, count)
    buckets = {"lt2": [0.0, 0], "2to5": [0.0, 0], "gt5": [0.0, 0]}
    for r in rows:
        pj = np.asarray(r["pred_joint"]); gj = np.asarray(r["gt_joint"])
        m, a = _mse_mae(pj, gj); jm.append(m); jma.append(a)
        pe = np.asarray(r["pred_eef"]); ge = np.asarray(r["gt_eef"])
        m, a = _mse_mae(pe, ge); em.append(m); ema.append(a)
        pf = np.asarray(r["pred_force"]); gf = np.asarray(r["gt_force"])
        m, a = _mse_mae(pf, gf); fm.append(m); fma.append(a)
        # Q1-Q3 / Q4-Q6 (原始 rad)
        j_q1q3.append(float(np.mean((pj[:, :3] - gj[:, :3]) ** 2)))
        j_q4q6.append(float(np.mean((pj[:, 3:6] - gj[:, 3:6]) ** 2)))
        # 归一化空间 (若 q_scale 可用): 关节 0-5 用 actions norm_stats 缩放到 [-1,1]
        if q_scale is None:
            q_scale = _load_joint_norm_scale()
        if q_scale is not None:
            pn = (pj[:, :6] - q_scale["mid"]) * q_scale["inv"]  # 近似归一化
            gn = (gj[:, :6] - q_scale["mid"]) * q_scale["inv"]
            j_q1q3_n.append(float(np.mean((pn[:, :3] - gn[:, :3]) ** 2)))
            j_q4q6_n.append(float(np.mean((pn[:, 3:6] - gn[:, 3:6]) ** 2)))
        n = r["force_norm"]
        key = "lt2" if n < 2 else ("2to5" if n <= 5 else "gt5")
        buckets[key][0] += m
        buckets[key][1] += 1
    res["joint_mse"] = float(np.mean(jm)) if jm else None
    res["joint_mae"] = float(np.mean(jma)) if jma else None
    res["eef_mse"] = float(np.mean(em)) if em else None
    res["eef_mae"] = float(np.mean(ema)) if ema else None
    res["force_mse"] = float(np.mean(fm)) if fm else None
    res["force_mae"] = float(np.mean(fma)) if fma else None
    # Q1-Q3 / Q4-Q6 拆分 (原始 rad + 归一化空间)
    res["joint_mse_q1q3"] = float(np.mean(j_q1q3)) if j_q1q3 else None
    res["joint_mse_q4q6"] = float(np.mean(j_q4q6)) if j_q4q6 else None
    res["joint_mse_q1q3_norm"] = float(np.mean(j_q1q3_n)) if j_q1q3_n else None
    res["joint_mse_q4q6_norm"] = float(np.mean(j_q4q6_n)) if j_q4q6_n else None
    # 分桶: 输出 综合(force_mse) + 分桶两组, 分桶按帧数加权 → 加权平均可还原综合
    res["force_mse_by_force"] = {
        k: {"mse": (v[0] / v[1] if v[1] else None), "n_frames": v[1]}
        for k, v in buckets.items()
    }
    return res


_JOINT_NORM_CACHE = {}


def _load_joint_norm_scale():
    """从训练数据集 norm_stats 读关节 (actions 前 6 维) 的 q01/q99, 用于归一化拆分.

    归一化: x_norm = (x - mid) / (range/2), mid=(q01+q99)/2, inv = 2/(q99-q01).
    返回 None 若读取失败 (此时跳过归一化拆分).
    """
    global _JOINT_NORM_CACHE
    if "done" in _JOINT_NORM_CACHE:
        return _JOINT_NORM_CACHE.get("scale")
    try:
        # 按任务/模型自动定位训练 norm_stats (与 verify_mapping 一致)
        import os
        task = os.environ.get("EVAL_TASK", "")
        model = os.environ.get("EVAL_MODEL", "")
        cand = None
        if task in ("board", "bottle"):
            if model in ("PI05_JOINT",):
                cand = "/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60_noforce"
            elif model in ("PI05_EEF",):
                cand = "/mnt/hdd/sfy/datasets/total_2task_flexiv_eef_abs_noforce"
            elif model in ("FORCE_JOINT", "FORCE_JOINT_NOFORCE", "FORCE_DUAL", "FORCE_DUAL_NOFORCE"):
                cand = "/mnt/hdd/sfy/datasets/total_2task_flexiv_ft60"
            elif model in ("FORCE_EEF", "FORCE_EEF_NOFORCE"):
                cand = "/mnt/hdd/sfy/datasets/total_2task_flexiv_eef_abs"
        else:  # peel
            if model in ("PI05_JOINT",):
                cand = "/mnt/hdd/sfy/datasets/total_task_peel_ft60_noforce"
            elif model in ("PI05_EEF",):
                cand = "/mnt/hdd/sfy/datasets/total_task_peel_eef_abs_noforce"
            elif model in ("FORCE_JOINT", "FORCE_JOINT_NOFORCE", "FORCE_DUAL", "FORCE_DUAL_NOFORCE"):
                cand = "/mnt/hdd/sfy/datasets/total_task_peel_ft60"
            elif model in ("FORCE_EEF", "FORCE_EEF_NOFORCE"):
                cand = "/mnt/hdd/sfy/datasets/total_task_peel_eef_abs"
        if cand is None:
            _JOINT_NORM_CACHE["done"] = True
            return None
        import json
        ns = json.load(open(f"{cand}/norm_stats.json"))["norm_stats"]["actions"]
        q01 = np.asarray(ns["q01"], dtype=np.float64)[:6]
        q99 = np.asarray(ns["q99"], dtype=np.float64)[:6]
        scale = {"mid": (q01 + q99) / 2.0, "inv": 2.0 / np.maximum(q99 - q01, 1e-6)}
        _JOINT_NORM_CACHE["done"] = True
        _JOINT_NORM_CACHE["scale"] = scale
        return scale
    except Exception:
        _JOINT_NORM_CACHE["done"] = True
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(EVAL_DATASETS))
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True, help="模型别名 (用于输出命名)")
    ap.add_argument("--out-dir", default="offline_eval_results")
    ap.add_argument("--ablate", action="store_true", help="wrench_history 全 0")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ensemble-horizon", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, default=None, help="小批量: 只跑前 N 个 episode")
    args = ap.parse_args()

    # 供 _load_joint_norm_scale 定位训练 norm_stats
    import os
    os.environ["EVAL_TASK"] = args.task
    os.environ["EVAL_MODEL"] = args.model

    suffix = "_ablate" if args.ablate else "_force"
    out = Path(args.out_dir) / f"{args.task}_{args.model}{suffix}.jsonl"
    agg = run_model(args.task, args.port, out, args.ablate,
                    stride=args.stride, ensemble_horizon=args.ensemble_horizon,
                    max_episodes=args.max_episodes)
    # 每模型独立 summary json
    summ = dict(agg)
    summ["task"] = args.task
    summ["model"] = args.model
    summ["ablate"] = bool(args.ablate)
    summ["jsonl"] = str(out)
    summ_path = Path(args.out_dir) / f"{args.task}_{args.model}{suffix}_summary.json"
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    print(f"[{args.task}/{args.model}{suffix}] n_frames={agg['n_frames']} (episodes<= {args.max_episodes})")
    print(f"  joint: mse={agg['joint_mse']:.6f} mae={agg['joint_mae']:.6f}")
    print(f"  eef  : mse={agg['eef_mse']:.6f} mae={agg['eef_mae']:.6f}")
    print(f"  force(综合): mse={agg['force_mse']:.6f} mae={agg['force_mae']:.6f}")
    b = agg["force_mse_by_force"]
    print(f"  force分桶: lt2={b['lt2']['mse'] and round(b['lt2']['mse'],6)} n={b['lt2']['n_frames']}"
          f" | 2to5={b['2to5']['mse'] and round(b['2to5']['mse'],6)} n={b['2to5']['n_frames']}"
          f" | gt5={b['gt5']['mse'] and round(b['gt5']['mse'],6)} n={b['gt5']['n_frames']}")
    print(f"  → {out} | {summ_path}")


if __name__ == "__main__":
    main()