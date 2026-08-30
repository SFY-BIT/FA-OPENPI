#!/usr/bin/env python3
"""离线评估批次编排: 每模型独占 server, server 间并行, server 内任务串行.

用法:
  python scripts/offline_eval_batch.py --batch a      # board+bottle 5 权重并行
  python scripts/offline_eval_batch.py --batch b      # peel 5 权重并行

批次 a: pi05_joint/pi05_eef/total_joint/total_eef/total_dual → board + bottle
批次 b: peel 5 权重 → peel
每个 server 独占 port; 线程间并行, 线程内串行跑该模型所有任务变体.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = "/mnt/hdd/sfy/datasets"
CK = "/mnt/hdd/sfy/FA-openpi/checkpoints"
ENV = dict(os.environ)
ENV.update({"CUDA_VISIBLE_DEVICES": "0", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONPATH": "src"})

# model → server 参数 + 任务列表
# tasks: (task, 输出模型名, ablate)
# device: server 挂在哪张 GPU (0=GPU0, 1=GPU1); 每 GPU 最多 4 个
BATCHES = {
    "a": [  # board + bottle 共用权重
        dict(port=8010, name="pi05_joint", device=0,
             norm=f"{DS}/total_2task_flexiv_ft60_noforce",
             cfg="pi05_plain_total_task_joint_remote", ckpt=f"{CK}/pi05_joint/36000",
             action=None,
             tasks=[("board", "PI05_JOINT", False), ("bottle", "PI05_JOINT", False)]),
        dict(port=8011, name="pi05_eef", device=0,
             norm=f"{DS}/total_2task_flexiv_eef_abs_noforce",
             cfg="pi05_plain_total_task_eef_v2_remote", ckpt=f"{CK}/pi05_eef/39999",
             action=["--action-space=EEF", "--action-rep=abs"],
             tasks=[("board", "PI05_EEF", False), ("bottle", "PI05_EEF", False)]),
        dict(port=8012, name="total_joint", device=0,
             norm=f"{DS}/total_2task_flexiv_ft60",
             cfg="pi05_force_total_task_joint_only_remote", ckpt=f"{CK}/total_joint/39999",
             action=None,
             tasks=[("board", "FORCE_JOINT", False), ("board", "FORCE_JOINT_NOFORCE", True),
                    ("bottle", "FORCE_JOINT", False), ("bottle", "FORCE_JOINT_NOFORCE", True)]),
        dict(port=8013, name="total_eef", device=0,
             norm=f"{DS}/total_2task_flexiv_eef_abs",
             cfg="pi05_force_total_task_eef_v2_remote", ckpt=f"{CK}/total_eef/39999",
             action=["--action-space=EEF", "--action-rep=abs"],
             tasks=[("board", "FORCE_EEF", False), ("board", "FORCE_EEF_NOFORCE", True),
                    ("bottle", "FORCE_EEF", False), ("bottle", "FORCE_EEF_NOFORCE", True)]),
        dict(port=8014, name="total_dual", device=1,
             norm=f"{DS}/total_2task_flexiv_ft60",
             cfg="pi05_force_total_task_eef_joint_remote", ckpt=f"{CK}/total_dual/39999",
             action=None,
             tasks=[("board", "FORCE_DUAL", False), ("board", "FORCE_DUAL_NOFORCE", True),
                    ("bottle", "FORCE_DUAL", False), ("bottle", "FORCE_DUAL_NOFORCE", True)]),
    ],
    "b": [  # peel 5 权重
        dict(port=8015, name="peel_05_joint", device=0,
             norm=f"{DS}/total_task_peel_ft60_noforce",
             cfg="pi05_plain_total_task_peel_joint_remote", ckpt=f"{CK}/peel_05_joint/39999",
             action=None,
             tasks=[("peel", "PI05_JOINT", False)]),
        dict(port=8016, name="peel_05_eef", device=0,
             norm=f"{DS}/total_task_peel_eef_abs_noforce",
             cfg="pi05_plain_total_task_peel_eef_v2_remote", ckpt=f"{CK}/peel_05_eef/39999",
             action=["--action-space=EEF", "--action-rep=abs"],
             tasks=[("peel", "PI05_EEF", False)]),
        dict(port=8017, name="peel_FA_joint", device=0,
             norm=f"{DS}/total_task_peel_ft60",
             cfg="pi05_force_total_task_peel_joint_only_remote", ckpt=f"{CK}/peel_FA_joint/39999",
             action=None,
             tasks=[("peel", "FORCE_JOINT", False), ("peel", "FORCE_JOINT_NOFORCE", True)]),
        dict(port=8018, name="peel_FA_eef", device=0,
             norm=f"{DS}/total_task_peel_eef_abs",
             cfg="pi05_force_total_task_peel_eef_v2_remote", ckpt=f"{CK}/peel_FA_eef/39999",
             action=["--action-space=EEF", "--action-rep=abs"],
             tasks=[("peel", "FORCE_EEF", False), ("peel", "FORCE_EEF_NOFORCE", True)]),
        dict(port=8019, name="peel_FA_daul", device=1,
             norm=f"{DS}/total_task_peel_ft60",
             cfg="pi05_force_total_task_peel_eef_joint_remote", ckpt=f"{CK}/peel_FA_daul/39999",
             action=None,
             tasks=[("peel", "FORCE_DUAL", False), ("peel", "FORCE_DUAL_NOFORCE", True)]),
    ],
}


def start_server(cfg: dict) -> subprocess.Popen:
    env = dict(ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["device"])
    cmd = [sys.executable, "-u", "scripts/serve_policy.py",
           "--norm-stats-dir", cfg["norm"], "--port", str(cfg["port"])]
    if cfg["action"]:
        cmd += cfg["action"]
    cmd += ["policy:checkpoint", "--policy.config", cfg["cfg"],
            "--policy.dir", cfg["ckpt"]]
    log = open(f"/tmp/batch_{cfg['port']}_server.log", "w")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_server(port: int, timeout: int = 600):
    log = f"/tmp/batch_{port}_server.log"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if Path(log).exists():
            try:
                if open(log).read().count("server listening") >= 1:
                    return True
            except Exception:
                pass
        time.sleep(10)
    return False


def run_tasks(cfg: dict, out_dir: Path):
    """线程内串行跑该模型所有任务变体."""
    results = []
    for task, model, ablate in cfg["tasks"]:
        cmd = [sys.executable, "scripts/offline_eval_ci_mse.py",
               "--task", task, "--model", model, "--port", str(cfg["port"]),
               "--out-dir", str(out_dir)]
        if ablate:
            cmd.append("--ablate")
        logf = open(f"/tmp/batch_{cfg['port']}_{model}.log", "w")
        print(f"[{cfg['name']}:{cfg['port']}] 跑 {task}/{model}{' ablate' if ablate else ''} ...", flush=True)
        log_path = f"/tmp/batch_{cfg['port']}_{model}.log"
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=logf, stderr=subprocess.STDOUT, timeout=7200)
            ok = r.returncode == 0
            if not ok:
                tail = open(log_path).read()[-300:]
                print(f"  !! {task}/{model} 退出码 {r.returncode}; 日志尾部: {tail}", flush=True)
            results.append((task, model, ablate, ok))
        except subprocess.TimeoutExpired:
            print(f"  !! {task}/{model} 超时 (7200s)", flush=True)
            results.append((task, model, ablate, False))
        except Exception as e:
            print(f"  !! {task}/{model} 异常: {type(e).__name__}: {e}", flush=True)
            results.append((task, model, ablate, False))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=list(BATCHES))
    ap.add_argument("--out-dir", default="offline_eval_results")
    ap.add_argument("--no-server", action="store_true", help="只跑任务(server 已挂)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = BATCHES[args.batch]

    # 全部并行: 每模型挂在自己 device (GPU0 ≤4, GPU1 放第5个), 线程间并行, 线程内串行
    servers = {}
    if not args.no_server:
        for cfg in models:
            print(f"启动 server {cfg['name']} :{cfg['port']} (GPU{cfg['device']})")
            servers[cfg["port"]] = start_server(cfg)
        for cfg in models:
            ok = wait_server(cfg["port"])
            print(f"  {cfg['name']} :{cfg['port']} {'就绪' if ok else '超时!'}")

    threads = []
    results = {}
    for cfg in models:
        t = threading.Thread(target=lambda c=cfg: results.update({c["port"]: run_tasks(c, out_dir)}))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if not args.no_server:
        for p, proc in servers.items():
            proc.terminate()

    print("\n══════ 批次完成 ══════")
    for cfg in models:
        for task, model, ablate, ok in results.get(cfg["port"], []):
            print(f"  {cfg['name']:14} {task:6}/{model:<22} {'ablate' if ablate else 'force '} {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()