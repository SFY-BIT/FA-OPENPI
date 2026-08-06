#!/usr/bin/env python3
"""Force-aware USB insertion rollout via openpi websocket policy server.

Key differences from standard rollout:
  - State includes wrist force/torque history (37 dims)
  - Model outputs 13 dims: joints(6)+grip(1)+force(3)+torque(3)
  - Force predictions logged for diagnostics

Usage:
  # ── Start server ──
  cd /mnt/hdd/sfy/openpi-force
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=src:packages/openpi-client/src \
    /home/sfy/miniconda3/envs/rlinf/bin/python scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_force_usb_insert \
    --policy.dir=/mnt/hdd/sfy/outputs/4000

  # ── Rollout ──
  conda activate rlinf
  cd /mnt/hdd/sfy/openpi-force
  PYTHONPATH=/mnt/hdd/sfy/robosuite:scripts:packages/openpi-client/src \
    python scripts/rollout_force_client.py --episodes 5 --max-steps 400 --num-action-steps 1
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

CAMERA_HEIGHT = 128
CAMERA_WIDTH = 128
FPS = 20
FORCE_HISTORY_FRAMES = 5  # must match config force_history_frames


def make_env():
    controller_config = load_composite_controller_config(robot="Piper")
    arm_cfg = controller_config["body_parts"]["right"]
    arm_cfg["type"] = "JOINT_POSITION"
    arm_cfg["input_max"] = 1
    arm_cfg["input_min"] = -1
    arm_cfg["output_max"] = 1
    arm_cfg["output_min"] = -1
    return suite.make(
        env_name="USBInsert",
        robots="Piper",
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


def joint_state(env) -> np.ndarray:
    """7-DOF: joints[0:6] + gripper absolute. Matches dataset _piper_state."""
    robot = env.robots[0]
    jq = robot._joint_positions  # 8 values
    arm = jq[:6]
    grip = (abs(float(jq[6])) + abs(float(jq[7]))) / 0.07 * 0.065
    return np.array([*arm, float(np.clip(grip, 0.0, 0.065))], dtype=np.float32)


def read_wrist_force(obs: dict) -> np.ndarray:
    return np.array(obs.get("robot0_wrist_force", np.zeros(3)), dtype=np.float32)


def read_wrist_torque(obs: dict) -> np.ndarray:
    return np.array(obs.get("robot0_wrist_torque", np.zeros(3)), dtype=np.float32)


class ForceHistory:
    """Circular buffer for force/torque history."""

    def __init__(self, n_frames: int = FORCE_HISTORY_FRAMES):
        self.n = n_frames
        self.force_buf = deque([np.zeros(3, dtype=np.float32) for _ in range(n_frames)], maxlen=n_frames)
        self.torque_buf = deque([np.zeros(3, dtype=np.float32) for _ in range(n_frames)], maxlen=n_frames)

    def push(self, force: np.ndarray, torque: np.ndarray):
        self.force_buf.append(force.astype(np.float32))
        self.torque_buf.append(torque.astype(np.float32))

    def get_force_hist(self) -> np.ndarray:
        """Returns (n_frames, 3) array."""
        return np.stack(list(self.force_buf), axis=0)

    def get_torque_hist(self) -> np.ndarray:
        """Returns (n_frames, 3) array."""
        return np.stack(list(self.torque_buf), axis=0)

    def get_state(self, joints: np.ndarray) -> np.ndarray:
        """Build full state: [joints(7), force_hist(n*3), torque_hist(n*3)]."""
        fh = self.get_force_hist().reshape(-1)
        th = self.get_torque_hist().reshape(-1)
        return np.concatenate([joints, fh, th], axis=-1)


def build_observation(env, obs: dict, force_history: ForceHistory) -> dict[str, Any]:
    """Build observation dict matching ForcePiperInputs repack transform.

    ForcePiperInputs concatenates: state = [joints(7), force_hist(K*3), torque_hist(K*3)].
    So we send raw 7-dim joints as state, plus separate force/torque history arrays.
    """
    joints = joint_state(env)

    return {
        "observation/image": obs["agentview_image"],
        "observation/wrist_image": obs["robot0_eye_in_hand_image"],
        "observation/state": joints.tolist(),  # 7 dims only — transform adds force
        "observation/wrist_force": force_history.get_force_hist(),  # (K, 3)
        "observation/wrist_torque": force_history.get_torque_hist(),  # (K, 3)
        "prompt": "Insert the USB into the port",
    }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-action-steps", type=int, default=1,
                        help="How many predicted actions to execute before re-inferring (1=most responsive)")
    parser.add_argument("--output-dir", default="/mnt/hdd/sfy/outputs/rollouts",
                        help="Directory for rollout videos")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", force=True)

    conn = connect_server(args.host, args.port)
    packer = msgpack_numpy.Packer()

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        seed = args.seed_offset + ep
        env = make_env()
        obs = env.reset()
        # Force gripper fully open
        env.robots[0].gripper["right"].current_action = np.array([1.0, -1.0])
        env.sim.data.qpos[env.sim.model.jnt_qposadr[env.sim.model.joint_name2id("robot0_joint7")]] = 0.035
        env.sim.data.qpos[env.sim.model.jnt_qposadr[env.sim.model.joint_name2id("robot0_joint8")]] = -0.035

        # Force history — initialise as [0, ..., 0, current] to match training
        force_history = ForceHistory(FORCE_HISTORY_FRAMES)
        init_force = read_wrist_force(obs)
        init_torque = read_wrist_torque(obs)
        force_history.push(init_force, init_torque)

        # Video recording
        video_path = output_dir / f"rollout_force_ep{ep}_seed{seed}.mp4"
        video_writer = imageio.get_writer(video_path, fps=FPS, format="FFMPEG",
                                          codec="libx264", quality=8,
                                          macro_block_size=1,
                                          output_params=["-preset", "ultrafast"])

        success = False
        total_reward = 0.0
        action_buffer = []
        force_open_steps = 3
        t0 = time.time()

        # Force prediction tracking
        force_predictions = []

        for step in range(args.max_steps):
            # Record frame
            if video_writer is not None:
                agent = obs["agentview_image"]
                wrist = obs["robot0_eye_in_hand_image"]
                video_writer.append_data(np.hstack([agent, wrist]))

            if not action_buffer:
                payload = build_observation(env, obs, force_history)
                conn.send(packer.pack(payload))
                raw = conn.recv()
                if isinstance(raw, str):
                    logging.error("Server error: %s", raw)
                    break
                response = msgpack_numpy.unpackb(raw)

                action_chunk = np.array(response["actions"])  # (30, 13)
                action_buffer = [action_chunk[i] for i in range(min(args.num_action_steps, len(action_chunk)))]

            action_full = action_buffer.pop(0)  # 13-dim
            action_abs = action_full[:7]         # joints(6)+grip(1) for control
            force_pred = action_full[7:10]       # predicted next force
            torque_pred = action_full[10:13]     # predicted next torque

            current_state = joint_state(env)
            delta = action_abs.copy()
            delta[:6] = action_abs[:6] - current_state[:6]

            # Grip: model predicts delta → AbsoluteActions → abs → convert back
            if step < force_open_steps:
                delta[6] = -1.0  # brief open
            else:
                grip_err = action_abs[6] - current_state[6]
                delta[6] = -np.sign(grip_err) if abs(grip_err) > 5e-4 else 0.0

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

            # Debug
            if step < 5 or step % 50 == 0:
                logging.info(
                    "  [step %d] grip=%.4f cmd=%.0f | "
                    "force_pred=[%+.4f,%+.4f,%+.4f] actual=[%+.4f,%+.4f,%+.4f] err=%.4f",
                    step, current_state[6], delta[6],
                    force_pred[0], force_pred[1], force_pred[2],
                    new_force[0], new_force[1], new_force[2],
                    force_predictions[-1]["force_err"],
                )

            if done or info.get("success", False):
                success = True
                break

        dt = time.time() - t0
        if video_writer is not None:
            video_writer.close()
        env.close()

        # Save force predictions
        force_path = output_dir / f"force_pred_ep{ep}_seed{seed}.json"
        with open(force_path, "w") as f:
            json.dump(force_predictions, f, indent=2)

        results.append({"ep": ep, "seed": seed, "success": success,
                        "reward": float(total_reward), "steps": step + 1, "time": dt,
                        "video": str(video_path), "force_log": str(force_path)})
        logging.info("[%d] seed=%d %s  reward=%.1f  steps=%d  (%.1fs)  force=%s",
                     ep, seed, "✓" if success else "✗", total_reward, step + 1, dt, force_path)

    conn.close()

    # Summary
    n_success = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_success}/{len(results)} = {n_success/len(results):.0%}")
    print(f"Avg time: {np.mean([r['time'] for r in results]):.1f}s/ep")
    avg_force_err = 0
    for r in results:
        with open(r["force_log"]) as f:
            fp = json.load(f)
        avg_force_err += np.mean([p["force_err"] for p in fp])
    print(f"Avg force pred error: {avg_force_err/len(results):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
