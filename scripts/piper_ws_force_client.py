#!/usr/bin/env python3
"""Piper 真机推理客户端 — websocket 版（适配 60 帧力历史 K=16 模型）

运行位置: 真机端 (或任何能采集真机 obs 的机器)
Server 位置: 训练/推理机 (serve_policy.py, websocket 端口 8000)

┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️ 关键设计: 60 帧力历史由 CLIENT 端拼接, server 无状态不拼        │
│                                                                     │
│  Server 的 ForceInStatePiperInputs 期望收到完整 wrench_history:     │
│    observation/wrench_history: [60, 6]  (60 帧力/力矩历史)          │
│  如果 client 不发, server 会用"当前帧力重复 60 次"兜底 ——          │
│  那会丢失真实历史, 本 client 负责真正攒 60 帧。                     │
│                                                                     │
│  不足 60 帧时: 前面补零 (与数据集 stamp_seal_v2_flexiv_ft60 一致):  │
│    数据集 wrench_history 语义 = 当前帧往前 60 帧 (含当前),           │
│    轨迹开头不足 60 帧 → 最早的帧前面补 0。                          │
└─────────────────────────────────────────────────────────────────────┘

发送给 server 的观测 (与 offline_replay 完全一致):
  observation/image          : 主相机 RGB (H,W,3) uint8, server 会 resize 224
  observation/wrist_image    : 腕相机 RGB (H,W,3) uint8
  observation/state          : 7 维 [j1..j6, gripper]  (proprio, 单位 rad)
  observation/wrench_history : [60, 6] float32
                              列序 = [Fx,Fy,Fz,Tx,Ty,Tz] (与数据集一致)
  prompt                     : 任务文本, 如 "stamp seal"

接收 (server → client):
  actions    : (30, 7) 绝对关节位置 [j1..j6, gripper] rad
  force_pred : (30, 6) 预测力 (仅 predict_force 模型, 可选)

执行语义 (重要):
  - 模型内部输出的是 DELTA (动作增量, DeltaActions 输入侧做 delta 精度控制)
  - 但 server 的 output_transform 链里 AbsoluteActions 会把 delta 加回 state:
        actions_abs = actions_delta + state   (mask 前6关节)
    因此 client 收到的是"绝对目标关节位置" (30,7)
  - send_action 传绝对目标即可 (勿再转 delta, 否则双重转换出错)
  - 6×30 force_pred 仅预测力, 不用于控制
  - 每帧执行前做限幅 (相对当前关节, 防猛动)

频率控制 (解耦 执行/推理):
  --fps             执行频率 (send_action 调用频率, Hz)
  --num-action-steps 每次推理后连续执行的 chunk 步数 (1~30)
                     → 推理频率 = fps / num_action_steps
                     例: --fps 30 --num-action-steps 5 → 执行 30Hz, 推理 6Hz
  --infer-fps       可选推理频率上限 (留作未来限流, 当前由 chunk 步数决定)

用法 (真机端):
  # 最响应 (每步推理): 
  python piper_ws_force_client.py --server ws://<IP>:8000 --task "stamp seal" \
      --fps 30 --num-action-steps 1 --duration-s 120
  # 省算力 (推理 6Hz):
  python piper_ws_force_client.py --server ws://<IP>:8000 --task "stamp seal" \
      --fps 30 --num-action-steps 5 --duration-s 120

依赖 (真机端):
  websockets  msgpack_numpy  numpy  torch  lerobot(含 Piper 机器人)
"""

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# ── openpi-client 的 websocket + msgpack 编码 ──
_OPENPI_CLIENT_SRC = Path(__file__).resolve().parent.parent / "packages" / "openpi-client" / "src"
sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

import websockets.sync.client
from openpi_client import msgpack_numpy

# ── 真机机器人 (复用旧版 piper_force_client 的 lerobot 采集) ──
from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.utils import busy_wait

# =========================================================================
# 配置常量 — 必须与 server 端训练配置一致
# =========================================================================
FT_HISTORY_STEPS = 60      # 力历史帧数 T (训练配置 ft_history_steps=60)
FORCE_DIM = 6              # 力/力矩维数 (Fx,Fy,Fz,Tx,Ty,Tz)
PROPRIO_DIM = 7            # 关节+夹爪 (force_start_idx=7)
ACTION_DIM = 7             # 模型输出关节动作维 (joints6 + gripper)
ACTION_HORIZON = 30        # server 返回的动作块长度

# 真机 lerobot obs 键 → server 期望键 (按你的真机环境调整!)
#   旧版 client 用 observation.images.one/two, 这里映射到 server 键
CAMERA_MAIN = "observation.images.one"      # 主相机 (本地键)
CAMERA_WRIST = "observation.images.two"     # 腕相机 (本地键)
STATE_KEY = "observation.state"             # 7 维关节+夹爪
FORCE_KEY = "observation.force"             # 6 维力/力矩 (真机自定义)


def build_wrench_history(force_buf: deque) -> np.ndarray:
    """把 60 帧力历史 deque 拼成 [60, 6] ndarray。

    与数据集处理一致:
      - deque 尾部 = 最新帧
      - 不足 60 帧 → 前面补零 (轨迹开头的历史未知, 用 0 填充)
    """
    frames = list(force_buf)                    # 旧→新 (最旧在前)
    T_avail = len(frames)
    if T_avail >= FT_HISTORY_STEPS:
        return np.stack(frames[-FT_HISTORY_STEPS:], axis=0)  # 只留最近 60
    # 补零到 60: [pad_zeros, ...真实帧] → 最早的帧补 0
    pad = np.zeros((FT_HISTORY_STEPS - T_avail, FORCE_DIM), dtype=np.float32)
    real = np.stack(frames, axis=0)
    return np.concatenate([pad, real], axis=0)  # [60, 6]


def build_observation(obs: dict[str, Any], force_buf: deque,
                      prompt: str,
                      camera_main: str = CAMERA_MAIN,
                      camera_wrist: str = CAMERA_WRIST) -> dict[str, Any]:
    """把真机 obs + 力 buffer 组装成 server 期望的 payload。"""
    # 相机: 转成 uint8 ndarray (server 端会 resize 到 224)
    img = _to_uint8(obs[camera_main])
    wrist = _to_uint8(obs[camera_wrist])

    # state: 7 维 proprio (只取前 7, 若真机返回 13 维含力则截断)
    state = np.asarray(obs[STATE_KEY], dtype=np.float32)
    if state.ndim > 1:
        state = state[0]                      # 兼容 (1,7) 形状
    proprio = state[:PROPRIO_DIM]

    # wrench_history: client 端拼接的 60 帧历史 [60,6]
    wrench_history = build_wrench_history(force_buf)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": proprio.tolist(),
        "observation/wrench_history": wrench_history,   # [60,6] ← client 拼!
        "prompt": prompt,
    }


def _to_uint8(img: Any) -> np.ndarray:
    """把 torch.Tensor / ndarray 统一成 uint8 RGB ndarray。"""
    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 4:
        arr = arr[0]
    return np.ascontiguousarray(arr)


def clamp_action_delta(target: list[float], current: list[float],
                       max_joint_delta: float, max_gripper_delta: float,
                       ) -> tuple[list[float], bool]:
    """限幅动作: 防止单步关节/夹爪变化过大 (安全)。"""
    clamped = list(target)
    was_clamped = False
    for i, (t, c) in enumerate(zip(target, current)):
        max_d = max_gripper_delta if i == len(target) - 1 else max_joint_delta
        lo, hi = c - max_d, c + max_d
        bounded = max(lo, min(hi, float(t)))
        if bounded != float(t):
            was_clamped = True
        clamped[i] = bounded
    return clamped, was_clamped


def main() -> None:
    p = argparse.ArgumentParser(description="Piper websocket force client (60-frame history)")
    p.add_argument("--server", default="ws://127.0.0.1:8000",
                   help="server 地址: ws://<IP>:8000 (serve_policy.py 端口)")
    p.add_argument("--task", default="stamp seal", help="任务指令 (prompt)")
    p.add_argument("--fps", type=int, default=30, help="执行频率 (send_action 频率, Hz)")
    p.add_argument("--num-action-steps", type=int, default=1,
                   help="每次推理后连续执行的 chunk 步数 (1=每步都推理最响应; "
                        "30=一次推理执行完整 30 步最省算力)")
    p.add_argument("--infer-fps", type=float, default=0,
                   help="推理频率上限 (0=不限制)。配合 num-action-steps 使用: "
                        "实际推理频率 = min(fps, infer_fps) 由 chunk 步数决定")
    p.add_argument("--duration-s", type=float, default=120.0, help="最大运行时长")
    p.add_argument("--robot-type", default="piper", help="lerobot 机器人类型")
    p.add_argument("--max-joint-delta", type=float, default=0.2, help="单步关节最大变化 (rad)")
    p.add_argument("--max-gripper-delta", type=float, default=0.02, help="单步夹爪最大变化")
    p.add_argument("--camera-main", default=CAMERA_MAIN, help="主相机 obs 键")
    p.add_argument("--camera-wrist", default=CAMERA_WRIST, help="腕相机 obs 键")
    p.add_argument("--no-force-history", action="store_true",
                   help="不攒历史, 只发当前帧力 (测试用, server 会 tile 60 次)")
    args = p.parse_args()

    camera_main, camera_wrist = args.camera_main, args.camera_wrist

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [CLIENT] %(message)s", force=True)

    # ── 1. 连接真机机器人 ──
    logging.info("Connecting to Piper robot...")
    robot = make_robot(args.robot_type, inference_time=True)
    robot.connect()
    logging.info("Robot connected")

    # ── 2. 连接 server (websocket) ──
    logging.info("Connecting to server %s ...", args.server)
    ws = websockets.sync.client.connect(
        args.server, compression=None, max_size=None,
        ping_interval=None, ping_timeout=None,
    )
    metadata = msgpack_numpy.unpackb(ws.recv())
    logging.info("Server connected, metadata=%s", metadata)
    packer = msgpack_numpy.Packer()

    # ── 3. 60 帧力历史 buffer (client 端拼接) ──
    force_buf = deque(maxlen=FT_HISTORY_STEPS)

    try:
        # 先采集一帧用于初始化 buffer (轨迹开头: 补零逻辑见 build_wrench_history)
        init_obs = robot.capture_observation()
        init_force = np.asarray(init_obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
        force_buf.append(init_force)
        logging.info("Initialized force buffer (1 frame); will pad zeros until %d frames",
                     FT_HISTORY_STEPS)

        start_t = time.perf_counter()
        step = 0
        action_chunk = None          # 当前缓存的 (30,7) 动作块
        chunk_pos = 0                # 已消费的 chunk 内步数
        n_infer = 0                  # 推理次数统计
        while time.perf_counter() - start_t < args.duration_s:
            t0 = time.perf_counter()

            # (a) 采集真机观测 (每执行步都采集, 用于 state 限幅 + 力历史)
            obs = robot.capture_observation()

            # (b) 把当前帧力 push 进历史 buffer (最新在末尾)
            if not args.no_force_history:
                force = np.asarray(obs[FORCE_KEY], dtype=np.float32).reshape(-1)[:FORCE_DIM]
                force_buf.append(force)

            # (c) 需要推理时: chunk 耗尽 或 达到推理频率上限
            need_infer = (action_chunk is None) or (chunk_pos >= args.num_action_steps)
            if need_infer:
                payload = build_observation(obs, force_buf, args.task,
                                            camera_main, camera_wrist)

                # (c1) 发送 → server 推理 → 接收
                ws.send(packer.pack(payload))
                raw = ws.recv()
                if isinstance(raw, str):
                    logging.error("Server error: %s", raw)
                    break
                resp = msgpack_numpy.unpackb(raw)

                action_chunk = np.asarray(resp["actions"])       # (30,7) 绝对关节
                chunk_pos = 0
                n_infer += 1

            # (d) 取当前步目标 (绝对关节位置)
            target = action_chunk[min(chunk_pos, len(action_chunk) - 1)].tolist()
            chunk_pos += 1
            force_pred = np.asarray(resp["force_pred"])[min(chunk_pos - 1, len(action_chunk) - 1)].tolist() if "force_pred" in resp else None

            logging.info("[step %d] n_hist=%d infer#%d chunk[%d/%d] target=%s",
                         step, len(force_buf), n_infer, chunk_pos, args.num_action_steps,
                         [f"{v:+.4f}" for v in target])

            # (e) 安全限幅 (相对当前关节位置)
            current = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)[:PROPRIO_DIM]
            safe, was_clamped = clamp_action_delta(
                target, current.tolist(),
                args.max_joint_delta, args.max_gripper_delta,
            )
            if was_clamped:
                logging.warning("[step %d] clamped: %s → %s",
                                step, [f"{v:+.4f}" for v in target],
                                [f"{v:+.4f}" for v in safe])

            # (f) 执行 (绝对关节目标)
            robot.send_action(np.asarray(safe, dtype=np.float32))

            # (g) 帧率控制
            dt = time.perf_counter() - t0
            logging.info("[step %d] dt=%.1fms fps=%.1f%s",
                         step, dt * 1000, 1.0 / dt if dt > 0 else 0,
                         f" force_pred={[f'{v:+.2f}' for v in force_pred]}" if force_pred else "")
            if args.fps:
                busy_wait(1.0 / args.fps - dt)
            step += 1

        # ── 会话统计 ──
        elapsed = time.perf_counter() - start_t
        logging.info("Session: %d steps in %.1fs (exec %.1f Hz), %d inferences "
                     "(infer %.1f Hz, chunk=%d)",
                     step, elapsed, step / elapsed if elapsed else 0,
                     n_infer, n_infer / elapsed if elapsed else 0,
                     args.num_action_steps)

    finally:
        logging.info("Disconnecting...")
        try:
            ws.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        logging.info("Done")


if __name__ == "__main__":
    main()
