"""离线评估准备①(轻量): 验证 10 模型 的 config 存在 + norm_stats 维度匹配 + ckpt 完整。
秒级完成; 完整模型加载放 4-server GPU 测试。
"""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from openpi.training import config as _config
from openpi.shared import normalize as _normalize

DS = "/mnt/hdd/sfy/datasets"

MODELS = [
    ("peel_05_joint/39999", "pi05_plain_total_task_peel_joint_remote",      "total_task_peel_ft60_noforce",      "joint", False),
    ("peel_FA_joint/39999", "pi05_force_total_task_peel_joint_only_remote", "total_task_peel_ft60",              "joint", True),
    ("peel_05_eef/39999",   "pi05_plain_total_task_peel_eef_v2_remote",     "total_task_peel_eef_abs_noforce",   "EEF+abs", False),
    ("peel_FA_eef/39999",   "pi05_force_total_task_peel_eef_v2_remote",     "total_task_peel_eef_abs",           "EEF+abs", True),
    ("peel_FA_daul/39999",  "pi05_force_total_task_peel_eef_joint_remote",  "total_task_peel_ft60",              "joint", True),
    ("pi05_joint/36000",    "pi05_plain_total_task_joint_remote",           "total_2task_flexiv_ft60_noforce",   "joint", False),
    ("pi05_eef/39999",      "pi05_plain_total_task_eef_v2_remote",          "total_2task_flexiv_eef_abs_noforce","EEF+abs", False),
    ("total_joint/39999",   "pi05_force_total_task_joint_only_remote",      "total_2task_flexiv_ft60",           "joint", True),
    ("total_eef/39999",     "pi05_force_total_task_eef_v2_remote",          "total_2task_flexiv_eef_abs",        "EEF+abs", True),
    ("total_dual/39999",    "pi05_force_total_task_eef_joint_remote",       "total_2task_flexiv_ft60",           "joint", True),
]

CKPT = "/mnt/hdd/sfy/FA-openpi/checkpoints"


def main():
    print("=== 轻量映射验证 (config + norm_stats + ckpt) ===")
    ok = 0
    for ckpt, cfg_name, ds, action, force in MODELS:
        errs = []
        try:
            cfg = _config.get_config(cfg_name)
        except Exception as e:
            print(f"  X {ckpt}: config '{cfg_name}' 不存在 ({e})")
            continue
        cdir = f"{CKPT}/{ckpt}"
        if not os.path.isdir(cdir):
            errs.append("ckpt 目录缺失")
        if not os.path.isdir(f"{cdir}/params"):
            errs.append("params 缺失")
        if not os.path.isfile(f"{cdir}/_CHECKPOINT_METADATA"):
            errs.append("metadata 缺失")
        try:
            ns = _normalize.load(f"{DS}/{ds}")
            want_state = cfg.data.action_dim
            st_dim = len(ns["state"].q01)
            if st_dim != want_state:
                errs.append(f"norm_stats state dim={st_dim} != 期望 {want_state}")
        except Exception as e:
            errs.append(f"norm_stats 加载失败: {e}")
        if errs:
            print(f"  X {ckpt:<22} <- {cfg_name:<46} {errs}")
        else:
            print(f"  OK {ckpt:<22} <- {cfg_name:<46} action={action:<8} force={force}")
            ok += 1
    print(f"\n通过 {ok}/{len(MODELS)}")


if __name__ == "__main__":
    main()
