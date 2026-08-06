"""ZMQ 力感知推理服务器 —— Pi0Force 双头模型 (stamp seal)

接收 piper_force_client.py 的 ZMQ 请求，桥接为 policy.infer() 调用。

用法:
  cd /mnt/hdd/sfy/openpi-force
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src python zmq_force_server.py \
      --checkpoint checkpoints/30000 --port 5555

  # 启用路由捕获（专家分配分析）:
  ... --capture-routing
"""

import argparse
import logging
import signal
import time

import jax
import numpy as np
import zmq

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(message)s",
)
logger = logging.getLogger("zmq_force_server")

# ── 客户端图像 key → ForceInStatePiperInputs 期望的 key ──
# 客户端发送: images["observation.images.camera0"], images["observation.images.camera1"]
# ForceInStatePiperInputs 直接读取 (斜杠格式，无需 repack):
#   data["observation/image"] → base image
#   data["observation/wrist_image"] → wrist image
#   data["observation/state"] → state
CAMERA_KEY_MAP = {
    "observation.images.camera0": "observation/image",
    "observation.images.camera1": "observation/wrist_image",
}


def build_obs_from_request(
    request: dict,
    force_history_buffer: list | None = None,
    ft_history_steps: int = 60,
    use_ft_history: bool = False,
) -> dict:
    """将客户端 ZMQ 请求转换为 policy.infer() 期望的 dict。

    客户端发送:
      {"state": [7], "force": [6], "images": {"observation.images.camera0": ..., ...}, "task": "..."}

    Legacy (force_in_state):
      {"observation/image": ndarray, "observation/wrist_image": ndarray,
       "observation/state": ndarray(13), "prompt": str}

    FT History mode (use_ft_history=True):
      {"observation/image": ndarray, "observation/wrist_image": ndarray,
       "observation/state": ndarray(7), "observation/wrench_history": ndarray(T,6), "prompt": str}
    """
    state_7d = np.asarray(request["state"], dtype=np.float32)
    force_6d = np.asarray(request.get("force", [0.0] * 6), dtype=np.float32)

    images = {}
    for client_key, policy_key in CAMERA_KEY_MAP.items():
        if client_key in request.get("images", {}):
            img = np.asarray(request["images"][client_key])
            images[policy_key] = img

    if not images:
        raise ValueError(
            f"No images found in request. Got keys: {list(request.get('images', {}))}"
        )

    if use_ft_history:
        # New path: state = proprio only, ft_state from rolling buffer
        obs = {
            "observation/state": state_7d,  # 7-dim proprio only
            "prompt": request.get("task", "stamp seal"),
            **images,
        }
        # Build wrench_history from rolling buffer
        if force_history_buffer is not None:
            # Update buffer with current force
            force_history_buffer.append(force_6d.copy())
            if len(force_history_buffer) > ft_history_steps:
                force_history_buffer.pop(0)
            # Pad if not enough history yet
            wrench = np.zeros((ft_history_steps, 6), dtype=np.float32)
            start = ft_history_steps - len(force_history_buffer)
            for i, f in enumerate(force_history_buffer):
                wrench[start + i] = f
            obs["observation/wrench_history"] = wrench
        else:
            # Fallback: all zeros (will be tiled by ForceInStatePiperInputs)
            obs["observation/wrench_history"] = np.zeros((ft_history_steps, 6), dtype=np.float32)
    else:
        # Legacy path: state = [proprio(7), force(6)]
        state_13d = np.concatenate([state_7d, force_6d])
        obs = {
            "observation/state": state_13d,
            "prompt": request.get("task", "stamp seal"),
            **images,
        }

    return obs


def load_policy(checkpoint_dir: str, config_name: str):
    """加载 Pi0Force 策略。

    norm_stats 优先从本地数据集加载（兼容 checkpoint 中不含 assets 的情况）。
    """
    logger.info("Loading config '%s'...", config_name)
    config = _config.get_config(config_name)

    logger.info("Creating policy from checkpoint '%s'...", checkpoint_dir)

    # 加载 norm_stats——优先从本地路径尝试
    norm_stats = None
    local_norm_stats_dir = "/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv"
    try:
        from openpi.shared import normalize as _normalize
        norm_stats = _normalize.load(local_norm_stats_dir)
        logger.info("Loaded norm_stats from %s", local_norm_stats_dir)
    except FileNotFoundError:
        logger.warning("norm_stats not found at %s, will try checkpoint assets", local_norm_stats_dir)

    policy = _policy_config.create_trained_policy(
        config, checkpoint_dir, norm_stats=norm_stats,
    )
    logger.info("Policy loaded successfully.")
    return policy


def main():
    parser = argparse.ArgumentParser(description="Pi0Force ZMQ inference server")
    parser.add_argument(
        "--checkpoint", default="checkpoints/30000",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--config", default="pi05_force_stamp_seal",
        help="Config name (pi05_force_stamp_seal or pi05_force_stamp_seal_remote)",
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="ZMQ server port",
    )
    parser.add_argument(
        "--capture-routing", action="store_true",
        help="Enable per-token expert routing capture (for MoE analysis)",
    )
    args = parser.parse_args()

    # 加载策略 (首次推理触发 JIT 编译，需要几十秒)
    t0 = time.monotonic()
    policy = load_policy(args.checkpoint, args.config)
    logger.info("Load time: %.1fs", time.monotonic() - t0)

    # ── 检测 FT history 模式 ──
    use_ft_history = getattr(getattr(policy, 'model', None), 'use_ft_history', False)
    ft_history_steps = getattr(getattr(policy, 'model', None), 'ft_history_steps', 60)
    if use_ft_history:
        logger.info("FT History mode ENABLED (T=%d, encoder=%s)",
                     ft_history_steps,
                     getattr(getattr(policy, 'model', None), 'ft_encoder_type', 'unknown'))
    else:
        logger.info("Legacy force-in-state mode (K=%d)",
                     getattr(getattr(policy, 'model', None), 'force_history_frames', 1))

    # ── MoE 路由捕获（可选） ──
    capture_routing = args.capture_routing
    if capture_routing:
        # 路由捕获已通过 _routing_capture.enable() 启用，
        # 数据由 limoe._POSITION_EXPERTS 自动存储（通过 jax.debug.callback）
        from openpi.models import moe_routing_capture as _routing_capture
        from openpi.models import limoe as _limoe
        _routing_capture.enable()  # 必须在首次 JIT 之前！
        logger.info("MoE routing capture ENABLED")

    # ── Force history rolling buffer (FT history mode) ──
    force_history_buffer: list = []  # list of [6] ndarrays, oldest first

    # ZMQ 上下文
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    bind_addr = f"tcp://*:{args.port}"
    socket.bind(bind_addr)
    logger.info("ZMQ server listening on %s", bind_addr)

    # 用 Poller 代替阻塞 recv，才能响应 Ctrl+C
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    # 优雅退出
    running = True

    def shutdown(signum, frame):
        nonlocal running
        logger.info("Received signal %d, shutting down...", signum)
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    step_count = 0
    try:
        while running:
            # 轮询等待（100ms 超时，可被 Ctrl+C 打断）
            socks = dict(poller.poll(timeout=100))
            if socket not in socks:
                continue

            try:
                request = socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                continue

            msg_type = request.get("type", "")

            if msg_type == "reset":
                socket.send_pyobj({"status": "ok"})
                logger.info("Policy reset")
                continue

            if msg_type != "infer":
                socket.send_pyobj({
                    "status": "error",
                    "message": f"Unknown message type: {msg_type}",
                })
                continue

            # ── 推理 ──
            t_start = time.monotonic()
            try:
                obs = build_obs_from_request(
                    request,
                    force_history_buffer=force_history_buffer if use_ft_history else None,
                    ft_history_steps=ft_history_steps,
                    use_ft_history=use_ft_history,
                )
                result = policy.infer(obs)

                # 提取路由数据（如果有）
                routing_data = None
                if capture_routing:
                    # 强制同步，确保 jax.debug.callback 已执行
                    _ = jax.block_until_ready(
                        jnp.asarray(result["actions"])
                    )
                    pos_data = _limoe.get_position_experts()
                    if pos_data is not None:
                        expert_flat = pos_data["expert"]  # flat array, int
                        valid_flat = pos_data["valid"]     # flat array, bool
                        valid_mask = valid_flat & (expert_flat >= 0) & (expert_flat < 4)
                        experts = expert_flat[valid_mask].tolist()
                        routing_data = {
                            "expert": experts,
                            "seq_length": len(experts),
                        }
                    else:
                        logger.warning(
                            "routing capture enabled but no position data; "
                            "ENABLED=%s", _routing_capture.ENABLED,
                        )

                # 取完整 action chunk (action_horizon=30, control_action_dim=7)
                actions = np.asarray(result["actions"])  # shape: (30, 7)
                action_chunk = actions.tolist()          # 30×7 完整序列

                # ── 诊断日志：打印 action 值供客户端对照 ──
                if step_count < 5 or step_count % 10 == 0:
                    logger.info(
                        "DIAG step=%d | step0=%s | abs_mean=%.4f | dx_mean=%.4f std=%.4f",
                        step_count,
                        [f"{v:+.4f}" for v in action_chunk[0]],
                        float(np.abs(actions).mean()),
                        float(np.abs(actions - actions[0:1]).mean()),
                        float(actions.std()),
                    )

                infer_ms = (time.monotonic() - t_start) * 1000
                reply = {
                    "status": "ok",
                    "action": action_chunk,
                }
                # 可选：返回力预测
                if "force_pred" in result:
                    force_pred = np.asarray(result["force_pred"])  # (30, 6)
                    reply["force_pred"] = force_pred.tolist()
                # 可选：返回路由捕获数据
                if routing_data is not None:
                    reply["routing"] = routing_data

                step_count += 1

            except Exception as e:
                logger.error("Inference error: %s", e, exc_info=True)
                reply = {"status": "error", "message": str(e)}

            socket.send_pyobj(reply)

    finally:
        logger.info("Closing ZMQ socket...")
        socket.close(0)
        context.term()
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
