#!/usr/bin/env python3
"""Panda no-force rollout for openpi pi05_panda_noforce checkpoint.

No-force config (pi05_panda_noforce_local):
  - robot: Panda (7 arm joints + 1 gripper = 8-dim state)
  - NO force/torque input, NO force prediction
  - Pure Pi0.5 baseline for ablation comparison
  - state = [joint(7), gripper(1)] = 8 dims
  - action = [joint(7), gripper(1)] = 8 dims (delta during training, abs from server)
  - Server output: AbsoluteActions converts delta→abs, client receives ABSOLUTE actions

Checkpoint: /mnt/hdd/sfy/outputs/panda-no-force/29999
Config: pi05_panda_noforce_local (local dataset path for norm_stats)

Usage:
  # ── Start server (Terminal 1) ──
  cd /mnt/hdd/sfy/openpi-force
  conda activate rlinf
  python scripts/serve_policy.py \
    --port 8001 \
    --jax-mem-fraction 0.3 \
    --jax-preallocate false \
    policy:checkpoint \
    --policy.config=pi05_panda_noforce_local \
    --policy.dir=/mnt/hdd/sfy/outputs/panda-no-force/29999

  # ── Rollout (Terminal 2) ──
  conda activate rlinf
  cd /mnt/hdd/sfy/openpi-force
  PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
    python scripts/rollout_panda_noforce.py --task usb --episodes 5 --max-steps 400
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent
_OPENPI_CLIENT_SRC = _OPENPI_ROOT / "packages" / "openpi-client" / "src"
_ROBOSUITE_ROOT = _OPENPI_ROOT.parent / "robosuite"

sys.path.insert(0, str(_OPENPI_CLIENT_SRC))
sys.path.insert(0, str(_ROBOSUITE_ROOT))

import websockets.sync.client
from openpi_client import msgpack_numpy

import robosuite as suite
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config
import imageio

CAMERA_HEIGHT = 224
CAMERA_WIDTH = 224
FPS = 20
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# ──────────────────────────────────────────────────────────────────────────
# Task definitions
# ──────────────────────────────────────────────────────────────────────────

TASKS = {
    "usb": {
        "env_name": "USBInsert",
        "prompt": "Insert the USB plug into the socket.",
        "gripper_open_steps": 3,
        "max_steps_default": 400,
    },
    "whiteboard": {
        "env_name": "WhiteboardWipe",
        "prompt": "Wipe the whiteboard with the eraser.",
        "gripper_open_steps": 5,
        "max_steps_default": 1200,
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────

def make_env(task: str = "usb"):
    """Create Panda env with JOINT_POSITION controller."""
    task_cfg = TASKS[task]
    controller_config = load_composite_controller_config(robot="Panda")
    arm_cfg = controller_config["body_parts"]["right"]
    arm_cfg["type"] = "JOINT_POSITION"
    arm_cfg["input_type"] = "delta"
    arm_cfg["input_max"] = 1
    arm_cfg["input_min"] = -1
    arm_cfg["output_max"] = 0.1
    arm_cfg["output_min"] = -0.1
    arm_cfg["kp"] = 150
    arm_cfg["damping_ratio"] = 1.0

    return suite.make(
        env_name=task_cfg["env_name"],
        robots="Panda",
        controller_configs=controller_config,
        gripper_types="default",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
        control_freq=FPS,
        ignore_done=True,
        horizon=2500,
    )


def gripper_qpos_addrs(env) -> list[int]:
    addrs = []
    for joint_id in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(joint_id)
        if name and "gripper" in name and "finger_joint" in name:
            addrs.append(env.sim.model.jnt_qposadr[joint_id])
    if len(addrs) != 2:
        raise RuntimeError(f"Expected 2 Panda gripper finger joints, found {len(addrs)}")
    return addrs


# ──────────────────────────────────────────────────────────────────────────
# State extraction
# ──────────────────────────────────────────────────────────────────────────

def panda_state(env) -> np.ndarray:
    """Panda 8-DOF state: joints[0:7] + gripper absolute."""
    robot = env.robots[0]
    jq = np.array(robot._joint_positions, dtype=np.float32)
    arm = jq[:7].copy()

    addrs = gripper_qpos_addrs(env)
    g0 = env.sim.data.qpos[addrs[0]]
    g1 = env.sim.data.qpos[addrs[1]]
    gripper_open = (abs(float(g0)) + abs(float(g1))) / 0.08
    gripper = float(np.clip(gripper_open, 0.0, 1.0)) * GRIPPER_OPEN
    return np.array([*arm, gripper], dtype=np.float32)


def get_task_positions(env) -> dict:
    """Get USB plug tip, socket mouth, and EEF positions for Z-alignment debugging."""
    sim = env.sim
    positions = {}

    # EEF position
    try:
        eef_sid = env.robots[0].eef_site_id["right"]
        positions["eef"] = sim.data.site_xpos[eef_sid].copy()
    except Exception as e:
        logging.warning("get_task_positions: eef error: %s", e)
        positions["eef"] = np.zeros(3)

    # USB plug tip & socket mouth (USBInsert task)
    try:
        plug_sid = sim.model.site_name2id("plug_tip_site")
        positions["plug_tip"] = sim.data.site_xpos[plug_sid].copy()
    except Exception as e:
        logging.warning("get_task_positions: plug_tip error: %s", e)
        positions["plug_tip"] = None
    try:
        socket_sid = sim.model.site_name2id("socket_mouth_site")
        positions["socket_mouth"] = sim.data.site_xpos[socket_sid].copy()
    except Exception as e:
        logging.warning("get_task_positions: socket_mouth error: %s", e)
        positions["socket_mouth"] = None

    # Plug body center
    try:
        plug_bid = sim.model.body_name2id("plug_main")
        positions["plug_center"] = sim.data.xpos[plug_bid].copy()
    except Exception as e:
        positions["plug_center"] = None

    return positions


# ──────────────────────────────────────────────────────────────────────────
# Observation building (no force)
# ──────────────────────────────────────────────────────────────────────────

def build_observation(env, obs: dict, prompt: str) -> dict[str, Any]:
    """Build observation dict for PiperInputs (no force).

    No-force version: state is just 8-dim [joint(7), gripper(1)].
    No force/torque history is sent.
    """
    state = panda_state(env)

    return {
        "observation/image": obs["agentview_image"],
        "observation/wrist_image": obs["robot0_eye_in_hand_image"],
        "observation/state": state.tolist(),  # 8 dims only
        "prompt": prompt,
    }


# ──────────────────────────────────────────────────────────────────────────
# Server connection
# ──────────────────────────────────────────────────────────────────────────

def connect_server(host: str, port: int) -> websockets.sync.client.ClientConnection:
    uri = f"ws://{host}:{port}"
    for attempt in range(60):
        try:
            conn = websockets.sync.client.connect(
                uri, compression=None, max_size=None,
                ping_interval=None, ping_timeout=None,
            )
            metadata = msgpack_numpy.unpackb(conn.recv())
            logging.info("Connected to server. metadata=%s", metadata)
            return conn
        except (ConnectionRefusedError, OSError):
            if attempt == 0:
                logging.info("Waiting for server at %s ...", uri)
            time.sleep(2)
    raise RuntimeError(f"Server not reachable after 60 attempts: {uri}")


# ──────────────────────────────────────────────────────────────────────────
# Gripper helpers
# ──────────────────────────────────────────────────────────────────────────

def set_gripper_open(env, addrs: list[int]):
    env.sim.data.qpos[addrs[0]] = GRIPPER_OPEN
    env.sim.data.qpos[addrs[1]] = -GRIPPER_OPEN
    env.sim.forward()


# ──────────────────────────────────────────────────────────────────────────
# Main rollout loop
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Panda no-force rollout (USB insert / Whiteboard wipe)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--task", choices=list(TASKS.keys()), default="usb",
        help="Task: 'usb' or 'whiteboard'",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0,
        help="Max steps (0=task default: usb=400, whiteboard=1200)")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-action-steps", type=int, default=1)
    parser.add_argument(
        "--output-dir", default="/mnt/hdd/sfy/outputs/rollouts_noforce",
        help="Directory for rollout videos",
    )
    parser.add_argument("--gripper-open-steps", type=int, default=-1,
        help="Initial gripper-open steps (-1=task default)")
    args = parser.parse_args()

    task_cfg = TASKS[args.task]
    prompt = task_cfg["prompt"]
    max_steps = args.max_steps if args.max_steps > 0 else task_cfg["max_steps_default"]
    gripper_open_steps = args.gripper_open_steps if args.gripper_open_steps >= 0 else task_cfg["gripper_open_steps"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", force=True)

    conn = connect_server(args.host, args.port)
    packer = msgpack_numpy.Packer()

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        seed = args.seed_offset + ep
        env = make_env(args.task)
        obs = env.reset()

        if args.task == "whiteboard":
            env.allow_dot_erasure = True

        addrs = gripper_qpos_addrs(env)
        set_gripper_open(env, addrs)

        video_path = output_dir / f"rollout_noforce_{args.task}_ep{ep}_seed{seed}.mp4"
        video_writer = imageio.get_writer(
            video_path, fps=FPS, format="FFMPEG",
            codec="libx264", quality=8, macro_block_size=1,
            output_params=["-preset", "ultrafast"],
        )

        success = False
        total_reward = 0.0
        action_buffer = []
        joint_log = []  # track joint trajectories for comparison
        t0 = time.time()

        for step in range(max_steps):
            agent = obs["agentview_image"]
            wrist = obs["robot0_eye_in_hand_image"]
            video_writer.append_data(np.hstack([agent, wrist]))

            if not action_buffer:
                payload = build_observation(env, obs, prompt)
                conn.send(packer.pack(payload))
                raw = conn.recv()
                if isinstance(raw, str):
                    logging.error("Server error: %s", raw)
                    break
                response = msgpack_numpy.unpackb(raw)

                # No-force: response["actions"] is (H, 8) absolute [joint(7), gripper(1)]
                action_chunk = np.array(response["actions"])
                n = min(args.num_action_steps, len(action_chunk))
                action_buffer = [action_chunk[i] for i in range(n)]

            action_abs = action_buffer.pop(0)  # 8-dim absolute

            current_state = panda_state(env)
            delta = action_abs.copy()
            delta[:7] = action_abs[:7] - current_state[:7]

            # Gripper: PandaGripper.format_action maps sign(action) to
            #   +1 => close, -1 => open.  But gripper position is 0=closed,
            #   0.04=open (larger = more open).  Negate the position error so
            #   that wanting to close (target < current) sends +1 (close).
            if step < gripper_open_steps:
                delta[7] = -(GRIPPER_OPEN - current_state[7])  # negate → open
            else:
                delta[7] = -(action_abs[7] - current_state[7])  # negate

            delta = np.clip(delta, -1.0, 1.0)

            # Log joint state for comparison
            joint_log.append({
                "step": step,
                "state": current_state.tolist(),
                "action_abs": action_abs.tolist(),
                "delta": delta.tolist(),
            })

            obs, reward, done, info = env.step(delta)
            total_reward += reward or 0.0

            if step < 10 or step % 50 == 0:
                task_pos = get_task_positions(env)
                plug_tip = task_pos.get("plug_tip")
                socket = task_pos.get("socket_mouth")
                eef = task_pos.get("eef")
                plug_str = f"plug=[{plug_tip[0]:+.3f},{plug_tip[1]:+.3f},{plug_tip[2]:+.3f}]" if plug_tip is not None else "plug=N/A"
                sock_str = f"sock=[{socket[0]:+.3f},{socket[1]:+.3f},{socket[2]:+.3f}]" if socket is not None else "sock=N/A"
                eef_str = f"eef=[{eef[0]:+.3f},{eef[1]:+.3f},{eef[2]:+.3f}]"
                logging.info(
                    "  [step %d] %s %s %s | grip=%.4f",
                    step, eef_str, plug_str, sock_str,
                    current_state[7],
                )

            if done or info.get("success", False):
                success = True
                break

        dt = time.time() - t0
        video_writer.close()
        env.close()

        # Save joint log for comparison
        joint_path = output_dir / f"joint_log_noforce_{args.task}_ep{ep}_seed{seed}.json"
        with open(joint_path, "w") as f:
            json.dump(joint_log, f, indent=2)

        results.append({
            "ep": ep, "seed": seed, "success": success,
            "reward": float(total_reward), "steps": step + 1, "time": dt,
            "video": str(video_path), "joint_log": str(joint_path),
        })
        logging.info(
            "[%d] seed=%d %s  reward=%.1f  steps=%d  (%.1fs)",
            ep, seed, "✓" if success else "✗",
            total_reward, step + 1, dt,
        )

    conn.close()

    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'=' * 60}")
    print(f"NoForce [{args.task}] RESULTS: {n_success}/{len(results)} = {n_success / len(results):.0%}")
    print(f"Avg time: {np.mean([r['time'] for r in results]):.1f}s/ep")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
