"""力感知 Piper 推理客户端

发送: images + joint state + force → server
接收: joint action (+ force prediction 可选)

用法:
  python piper_force_client.py --server tcp://192.168.1.100:5555 --task "stamp seal"
"""

import argparse
import logging
import time
from typing import Any

import numpy as np
import torch
import zmq

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.utils import busy_wait

# ── 相机映射: 本地键 → 远程策略期望键 ──
DEFAULT_CAMERA_MAPPING = {
    "observation.images.one": "observation.images.camera0",
    "observation.images.two": "observation.images.camera1",
}


def parse_camera_mappings(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CAMERA_MAPPING)
    mapping: dict[str, str] = {}
    for value in values:
        local_key, remote_key = value.split("=", maxsplit=1)
        if not local_key.startswith("observation.images."):
            local_key = f"observation.images.{local_key}"
        if not remote_key.startswith("observation.images."):
            remote_key = f"observation.images.{remote_key}"
        mapping[local_key] = remote_key
    return mapping


def build_request(
    observation: dict[str, Any],
    task: str,
    camera_mapping: dict[str, str],
    include_force: bool = True,
) -> dict[str, Any]:
    """构建发送给 server 的请求。

    请求格式:
      {
        "type": "infer",
        "task": "...",
        "state": [j1..j7],           # 7维关节+夹爪
        "force": [Fx,Fy,Fz,Tx,Ty,Tz], # 6轴力 (可选)
        "images": {camera0: ..., camera1: ...}
      }
    """
    state = observation["observation.state"].tolist()
    images: dict[str, Any] = {}
    for local_key, remote_key in camera_mapping.items():
        if local_key not in observation:
            continue
        img = observation[local_key]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        images[remote_key] = img

    if not images:
        raise ValueError(
            f"No mapped images found. obs keys: {list(observation.keys())}, "
            f"mapping: {camera_mapping}"
        )

    payload: dict[str, Any] = {
        "type": "infer",
        "task": task,
        "state": state,
        "images": images,
    }

    if include_force and "observation.force" in observation:
        force = observation["observation.force"]
        if isinstance(force, torch.Tensor):
            force = force.tolist()
        payload["force"] = force

    return payload


def clamp_action_delta(
    target_action: list[float],
    current_state: list[float],
    max_abs_joint_delta: float,
    max_abs_gripper_delta: float,
) -> tuple[list[float], bool]:
    """限幅动作增量，防止机械臂剧烈运动"""
    clamped = list(target_action)
    was_clamped = False
    for i, (target, current) in enumerate(zip(target_action, current_state)):
        max_d = max_abs_gripper_delta if i == len(target_action) - 1 else max_abs_joint_delta
        lower, upper = current - max_d, current + max_d
        bounded = max(lower, min(upper, float(target)))
        if bounded != float(target):
            was_clamped = True
        clamped[i] = bounded
    return clamped, was_clamped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Piper force-aware remote inference client"
    )
    parser.add_argument("--server", default="tcp://127.0.0.1:5555",
                        help="ZeroMQ server endpoint")
    parser.add_argument("--task", required=True,
                        help="Task instruction string")
    parser.add_argument("--fps", type=int, default=30,
                        help="Control loop rate (default 30)")
    parser.add_argument("--duration-s", type=float, default=60.0,
                        help="Max run duration")
    parser.add_argument("--robot-type", default="piper",
                        help="Robot type")
    parser.add_argument("--camera-mapping", action="append",
                        help="Map local→remote image keys, e.g. one=camera0")
    parser.add_argument("--max-abs-joint-delta", type=float, default=0.2,
                        help="Max per-step joint change")
    parser.add_argument("--max-abs-gripper-delta", type=float, default=0.02,
                        help="Max per-step gripper change")
    parser.add_argument("--no-force", action="store_true",
                        help="Disable force data sending")
    parser.add_argument("--predict-force", action="store_true",
                        help="Server also predicts force (action includes 6 force dims)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [CLIENT] %(message)s"
    )

    camera_mapping = parse_camera_mappings(args.camera_mapping)
    logging.info("Camera mapping: %s", camera_mapping)
    logging.info("Force sending: %s, Force prediction: %s",
                 not args.no_force, args.predict_force)

    # ── 连接机器人 ──
    logging.info("Connecting to Piper robot...")
    robot = make_robot(args.robot_type, inference_time=True)
    robot.connect()
    logging.info("Robot connected")

    # ── 连接 server ──
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(args.server)
    logging.info("Connected to server: %s", args.server)

    try:
        # 重置策略
        socket.send_pyobj({"type": "reset"})
        reply = socket.recv_pyobj()
        if reply.get("status") != "ok":
            raise RuntimeError(f"Server reset failed: {reply}")
        logging.info("Policy reset OK")

        start_t = time.perf_counter()
        step_idx = 0

        while time.perf_counter() - start_t < args.duration_s:
            loop_start = time.perf_counter()

            # (1) 采集观测
            observation = robot.capture_observation()

            # (2) 构建请求
            request = build_request(
                observation, args.task, camera_mapping,
                include_force=not args.no_force,
            )

            # (3) 发送 → 推理 → 接收
            socket.send_pyobj(request)
            reply = socket.recv_pyobj()
            if reply.get("status") != "ok":
                raise RuntimeError(f"Inference failed: {reply}")

            raw_action = reply["action"]
            logging.info("step=%s action=%s", step_idx,
                         [f"{v:+.4f}" for v in raw_action])

            # (4) 提取关节动作（忽略可能的力预测尾缀）
            joint_dim = len(observation["observation.state"])
            joint_action = raw_action[:joint_dim]

            # (5) 安全限幅
            current_state = observation["observation.state"].tolist()
            safe_action, was_clamped = clamp_action_delta(
                joint_action, current_state,
                args.max_abs_joint_delta, args.max_abs_gripper_delta,
            )
            if was_clamped:
                logging.warning("step=%s clamped: raw=%s → safe=%s",
                                step_idx,
                                [f"{v:+.4f}" for v in joint_action],
                                [f"{v:+.4f}" for v in safe_action])

            # (6) 执行动作
            robot.send_action(torch.tensor(safe_action, dtype=torch.float32))

            # (7) 计时
            dt_s = time.perf_counter() - loop_start
            logging.info("step=%s dt=%.1fms fps=%.1f", step_idx, dt_s * 1000,
                         1.0 / dt_s if dt_s > 0 else 0)
            if args.fps:
                busy_wait(1 / args.fps - dt_s)
            step_idx += 1

    finally:
        logging.info("Disconnecting...")
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            socket.close(0)
        except Exception:
            pass
        try:
            context.term()
        except Exception:
            pass
        logging.info("Done")


if __name__ == "__main__":
    main()
