#!/usr/bin/env python3
"""Force-aware Panda USB insertion rollout for openpi LoRA V6 checkpoint.

V6 config (pi05_force_lora_local):
  - robot: Panda (7 arm joints + 1 gripper = 8-dim state)
  - force_start_idx=8, force_history_frames=3
  - state = [joint(7), gripper(1), force_t0(3), force_t1(3), force_t2(3),
             torque_t0(3), torque_t1(3), torque_t2(3)] = 26 dims
  - predict_force=True → action = [joint(7), gripper(1), force(3), torque(3)] = 14 dims
  - use_delta_gripper_actions=True → all 14 dims are delta during training
  - Server output: AbsoluteActions converts delta→abs, so client receives ABSOLUTE actions

Controller: JOINT_POSITION with input_type="delta"
  - Model outputs absolute joint targets; client converts to delta for the controller
  - Alternatively, can use input_type="absolute" and send targets directly

Usage:
  # ── Start server (Terminal 1) ──
  cd /mnt/hdd/sfy/openpi-force
  conda activate rlinf
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=src:packages/openpi-client/src \
    python scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_force_lora_local \
    --policy.dir=/mnt/hdd/sfy/openpi-force/checkpoints/pi05_force_lora_local/local_lora_v6/19999

  # ── Rollout (Terminal 2) ──
  conda activate rlinf
  cd /mnt/hdd/sfy/openpi-force
  PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
    python scripts/rollout_panda_v6.py --episodes 5 --max-steps 400 --num-action-steps 1
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
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
FORCE_HISTORY_FRAMES = 3  # V6: must match config force_history_frames=3
GRIPPER_OPEN = 0.04       # Panda gripper open position
GRIPPER_CLOSED = 0.0      # Panda gripper closed position


# ──────────────────────────────────────────────────────────────────────────
# Task definitions
# ──────────────────────────────────────────────────────────────────────────

TASKS = {
    "usb": {
        "env_name": "USBInsert",
        "prompt": "Insert the USB plug into the socket.",
        "gripper_open_steps": 3,   # brief open at start
        "max_steps_default": 400,
    },
    "whiteboard": {
        "env_name": "WhiteboardWipe",
        "prompt": "Wipe the whiteboard with the eraser.",
        "gripper_open_steps": 5,   # need open to approach eraser
        "max_steps_default": 1200,
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────

def make_env(task: str = "usb"):
    """Create Panda env with JOINT_POSITION controller for the given task."""
    task_cfg = TASKS[task]
    controller_config = load_composite_controller_config(robot="Panda")
    arm_cfg = controller_config["body_parts"]["right"]
    # Use JOINT_POSITION: input is delta relative to current joint positions
    arm_cfg["type"] = "JOINT_POSITION"
    arm_cfg["input_type"] = "delta"
    arm_cfg["input_max"] = 1
    arm_cfg["input_min"] = -1
    # output_max/min: max joint movement per step (rad)
    # 0.1 rad ≈ 5.7° per step at 20Hz → reasonable for insertion
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
    """Get Panda gripper finger joint qpos addresses."""
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
    """Panda 8-DOF state: joints[0:7] + gripper absolute.

    Matches dataset _panda_state in collect_usb_lerobot_v21.py:
      arm = joint_positions[:7]
      gripper = (|finger1| + |finger2|) / 0.08 * 0.04, clipped to [0, 0.04]
    """
    robot = env.robots[0]
    jq = np.array(robot._joint_positions, dtype=np.float32)  # 7 arm joints
    arm = jq[:7].copy()

    addrs = gripper_qpos_addrs(env)
    g0 = env.sim.data.qpos[addrs[0]]
    g1 = env.sim.data.qpos[addrs[1]]
    gripper_open = (abs(float(g0)) + abs(float(g1))) / 0.08
    gripper = float(np.clip(gripper_open, 0.0, 1.0)) * GRIPPER_OPEN
    return np.array([*arm, gripper], dtype=np.float32)


def read_wrist_force(obs: dict) -> np.ndarray:
    return np.array(obs.get("robot0_wrist_force", np.zeros(3)), dtype=np.float32)


def read_wrist_torque(obs: dict) -> np.ndarray:
    return np.array(obs.get("robot0_wrist_torque", np.zeros(3)), dtype=np.float32)


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
# Force history buffer
# ──────────────────────────────────────────────────────────────────────────

class ForceHistory:
    """Circular buffer for force/torque history (K frames)."""

    def __init__(self, n_frames: int = FORCE_HISTORY_FRAMES):
        self.n = n_frames
        self.force_buf = deque(
            [np.zeros(3, dtype=np.float32) for _ in range(n_frames)], maxlen=n_frames
        )
        self.torque_buf = deque(
            [np.zeros(3, dtype=np.float32) for _ in range(n_frames)], maxlen=n_frames
        )

    def push(self, force: np.ndarray, torque: np.ndarray):
        self.force_buf.append(force.astype(np.float32))
        self.torque_buf.append(torque.astype(np.float32))

    def get_force_hist(self) -> np.ndarray:
        """Returns (n_frames, 3) array, oldest first."""
        return np.stack(list(self.force_buf), axis=0)

    def get_torque_hist(self) -> np.ndarray:
        """Returns (n_frames, 3) array, oldest first."""
        return np.stack(list(self.torque_buf), axis=0)


# ──────────────────────────────────────────────────────────────────────────
# Observation building
# ──────────────────────────────────────────────────────────────────────────

def build_observation(env, obs: dict, force_history: ForceHistory, prompt: str) -> dict[str, Any]:
    """Build observation dict matching ForcePiperInputs repack transform.

    ForcePiperInputs concatenates:
      state = [joint_state(8), force_hist(K*3), torque_hist(K*3)]
    So we send raw 8-dim state (7 joints + 1 gripper), plus separate
    force/torque history arrays. The transform handles concatenation.
    """
    state = panda_state(env)

    return {
        "observation/image": obs["agentview_image"],
        "observation/wrist_image": obs["robot0_eye_in_hand_image"],
        "observation/state": state.tolist(),  # 8 dims: 7 joints + 1 gripper
        "observation/wrist_force": force_history.get_force_hist(),   # (K, 3)
        "observation/wrist_torque": force_history.get_torque_hist(),  # (K, 3)
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
    """Force Panda gripper to fully open position."""
    env.sim.data.qpos[addrs[0]] = GRIPPER_OPEN
    env.sim.data.qpos[addrs[1]] = -GRIPPER_OPEN
    env.sim.forward()


def set_gripper_closed(env, addrs: list[int]):
    """Force Panda gripper to closed position."""
    env.sim.data.qpos[addrs[0]] = GRIPPER_CLOSED
    env.sim.data.qpos[addrs[1]] = GRIPPER_CLOSED
    env.sim.forward()


# ──────────────────────────────────────────────────────────────────────────
# Main rollout loop
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Panda V6 force-aware rollout (USB insert / Whiteboard wipe)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--task", choices=list(TASKS.keys()), default="usb",
        help="Task to roll out: 'usb' (USBInsert) or 'whiteboard' (WhiteboardWipe)",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0,
        help="Max steps per episode (0=use task default: usb=400, whiteboard=1200)")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--num-action-steps", type=int, default=1,
        help="How many predicted actions to execute before re-inferring (1=most responsive)",
    )
    parser.add_argument(
        "--output-dir", default="/mnt/hdd/sfy/outputs/rollouts_v6",
        help="Directory for rollout videos and force logs",
    )
    parser.add_argument(
        "--gripper-open-steps", type=int, default=-1,
        help="Initial steps to force gripper open (-1=use task default)",
    )
    args = parser.parse_args()

    task_cfg = TASKS[args.task]
    prompt = task_cfg["prompt"]
    max_steps = args.max_steps if args.max_steps > 0 else task_cfg["max_steps_default"]
    gripper_open_steps = args.gripper_open_steps if args.gripper_open_steps >= 0 else task_cfg["gripper_open_steps"]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", force=True
    )

    conn = connect_server(args.host, args.port)
    packer = msgpack_numpy.Packer()

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        seed = args.seed_offset + ep
        env = make_env(args.task)
        obs = env.reset()

        # WhiteboardWipe needs dot erasure enabled after reset
        if args.task == "whiteboard":
            env.allow_dot_erasure = True

        addrs = gripper_qpos_addrs(env)
        # Force gripper fully open at start
        set_gripper_open(env, addrs)

        # Force history — initialise as [0, ..., 0, current] to match training
        force_history = ForceHistory(FORCE_HISTORY_FRAMES)
        init_force = read_wrist_force(obs)
        init_torque = read_wrist_torque(obs)
        force_history.push(init_force, init_torque)

        # Video recording
        video_path = output_dir / f"rollout_v6_{args.task}_ep{ep}_seed{seed}.mp4"
        video_writer = imageio.get_writer(
            video_path, fps=FPS, format="FFMPEG",
            codec="libx264", quality=8, macro_block_size=1,
            output_params=["-preset", "ultrafast"],
        )

        success = False
        total_reward = 0.0
        action_buffer = []
        force_buffer = []
        force_predictions = []
        joint_log = []
        t0 = time.time()

        for step in range(max_steps):
            # Record frame
            agent = obs["agentview_image"]
            wrist = obs["robot0_eye_in_hand_image"]
            video_writer.append_data(np.hstack([agent, wrist]))

            # Query server if action buffer empty
            if not action_buffer:
                payload = build_observation(env, obs, force_history, prompt)
                conn.send(packer.pack(payload))
                raw = conn.recv()
                if isinstance(raw, str):
                    logging.error("Server error: %s", raw)
                    break
                response = msgpack_numpy.unpackb(raw)

                # Server returns:
                #   response["actions"]: (action_horizon, 8) — absolute [joint(7), gripper(1)]
                #   response["force_pred"]: (action_horizon, 6) — absolute [force(3), torque(3)]
                # (ForcePiperOutputs splits the 14-dim output into actions[:8] and force_pred[8:14])
                action_chunk = np.array(response["actions"])       # (H, 8)
                force_chunk = np.array(response.get("force_pred", []))  # (H, 6) or empty

                n = min(args.num_action_steps, len(action_chunk))
                action_buffer = [action_chunk[i] for i in range(n)]
                force_buffer = [force_chunk[i] for i in range(n)] if len(force_chunk) else [
                    np.zeros(6) for _ in range(n)
                ]

            action_abs = action_buffer.pop(0)  # 8-dim absolute [joint(7), gripper(1)]
            ft_pred = force_buffer.pop(0)      # 6-dim [force(3), torque(3)]
            force_pred = ft_pred[:3]
            torque_pred = ft_pred[3:6]

            # Convert absolute action → delta for JOINT_POSITION(delta) controller
            current_state = panda_state(env)
            delta = action_abs.copy()
            delta[:7] = action_abs[:7] - current_state[:7]  # joint deltas

            # Gripper: PandaGripper.format_action maps sign(action) to
            #   +1 => close, -1 => open.  But gripper position is 0=closed,
            #   0.04=open (larger = more open).  So the position error must
            #   be NEGATED: if model wants to close (target < current), we
            #   need to send +1 (close command), not -1.
            if step < gripper_open_steps:
                delta[7] = -(GRIPPER_OPEN - current_state[7])  # negate → open
            else:
                delta[7] = -(action_abs[7] - current_state[7])  # negate

            # Clip delta to controller's input range [-1, 1]
            delta = np.clip(delta, -1.0, 1.0)

            # Log joint state for comparison with no-force version
            joint_log.append({
                "step": step,
                "state": current_state.tolist(),
                "action_abs": action_abs.tolist(),
                "delta": delta.tolist(),
            })

            obs, reward, done, info = env.step(delta)
            total_reward += reward or 0.0

            # Update force history with new reading
            new_force = read_wrist_force(obs)
            new_torque = read_wrist_torque(obs)
            force_history.push(new_force, new_torque)

            # Track force predictions vs actual
            force_predictions.append({
                "step": step,
                "pred_force": force_pred.tolist(),
                "actual_force": new_force.tolist(),
                "pred_torque": torque_pred.tolist(),
                "actual_torque": new_torque.tolist(),
                "force_err": float(np.linalg.norm(force_pred - new_force)),
                "torque_err": float(np.linalg.norm(torque_pred - new_torque)),
            })

            # Debug logging
            if step < 10 or step % 50 == 0:
                task_pos = get_task_positions(env)
                plug_tip = task_pos.get("plug_tip")
                socket = task_pos.get("socket_mouth")
                eef = task_pos.get("eef")
                plug_str = f"plug=[{plug_tip[0]:+.3f},{plug_tip[1]:+.3f},{plug_tip[2]:+.3f}]" if plug_tip is not None else "plug=N/A"
                sock_str = f"sock=[{socket[0]:+.3f},{socket[1]:+.3f},{socket[2]:+.3f}]" if socket is not None else "sock=N/A"
                eef_str = f"eef=[{eef[0]:+.3f},{eef[1]:+.3f},{eef[2]:+.3f}]"
                logging.info(
                    "  [step %d] %s %s %s | grip=%.4f | force=[%+.2f,%+.2f,%+.2f]",
                    step, eef_str, plug_str, sock_str,
                    current_state[7],
                    force_pred[0], force_pred[1], force_pred[2],
                )

            if done or info.get("success", False):
                success = True
                break

        dt = time.time() - t0
        video_writer.close()
        env.close()

        # Save force predictions
        force_path = output_dir / f"force_pred_v6_{args.task}_ep{ep}_seed{seed}.json"
        with open(force_path, "w") as f:
            json.dump(force_predictions, f, indent=2)

        # Save joint log for comparison with no-force version
        joint_path = output_dir / f"joint_log_v6_{args.task}_ep{ep}_seed{seed}.json"
        with open(joint_path, "w") as f:
            json.dump(joint_log, f, indent=2)

        results.append({
            "ep": ep, "seed": seed, "success": success,
            "reward": float(total_reward), "steps": step + 1, "time": dt,
            "video": str(video_path), "force_log": str(force_path),
        })
        logging.info(
            "[%d] seed=%d %s  reward=%.1f  steps=%d  (%.1fs)  force=%s",
            ep, seed, "✓" if success else "✗",
            total_reward, step + 1, dt, force_path,
        )

    conn.close()

    # Summary
    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'=' * 60}")
    print(f"V6 [{args.task}] RESULTS: {n_success}/{len(results)} = {n_success / len(results):.0%}")
    print(f"Avg time: {np.mean([r['time'] for r in results]):.1f}s/ep")
    avg_force_err = 0.0
    for r in results:
        with open(r["force_log"]) as f:
            fp = json.load(f)
        avg_force_err += np.mean([p["force_err"] for p in fp])
    avg_force_err /= max(len(results), 1)
    print(f"Avg force pred error: {avg_force_err:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
