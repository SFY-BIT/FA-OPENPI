#!/usr/bin/env python3
"""Diagnostic rollout: saves force predictions vs actual force for analysis."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/mnt/hdd/sfy/robosuite")
import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config

CAMERA_HEIGHT = 128
CAMERA_WIDTH = 128
FPS = 20


def state_vector(env, obs):
    robot = env.robots[0]
    jq = robot._joint_positions
    arm = jq[:6]
    grip = (abs(float(jq[6])) + abs(float(jq[7]))) / 0.07 * 0.065
    return np.array([*arm, float(np.clip(grip, 0.0, 0.065))], dtype=np.float32)


def build_observation(env, obs):
    return {
        "observation/image": obs["agentview_image"],
        "observation/wrist_image": obs["robot0_eye_in_hand_image"],
        "observation/state": state_vector(env, obs).tolist(),
        "prompt": "Insert the USB into the port",
    }


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


def connect_server(host, port):
    uri = f"ws://{host}:{port}"
    for attempt in range(60):
        try:
            conn = websockets.sync.client.connect(
                uri, compression=None, max_size=None,
                ping_interval=None, ping_timeout=None,
            )
            metadata = msgpack_numpy.unpackb(conn.recv())
            logger.info(f"Connected. metadata={metadata}")
            return conn
        except (ConnectionRefusedError, OSError):
            if attempt == 0:
                logger.info(f"Waiting for server at {uri} ...")
            time.sleep(2)
    raise RuntimeError(f"Server not reachable: {uri}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--output", default="/mnt/hdd/sfy/outputs/rollouts")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_server(args.host, args.port)
    env = make_env()
    obs = env.reset()
    env.robots[0].gripper["right"].current_action = np.array([1.0, -1.0])
    packer = msgpack_numpy.Packer()

    joint_states = []
    delta_actions = []
    force_preds = []
    actual_forces = []
    rewards = []

    current_state = state_vector(env, obs)
    gripper_state = obs["robot0_gripper_qpos"][0]
    force_open_steps = 3

    for step in range(args.max_steps):
        payload = build_observation(env, obs)
        conn.send(packer.pack(payload))
        raw = conn.recv()
        response = msgpack_numpy.unpackb(raw)

        if isinstance(response, str):
            logger.error(f"Server error: {response}")
            break
        if "actions" not in response:
            logger.error(f"No actions. keys={list(response.keys())}")
            break

        action_chunk = np.array(response["actions"])
        action_abs = action_chunk[0] if action_chunk.ndim == 2 else action_chunk[:7]

        delta = action_abs.copy()
        delta[:6] = action_abs[:6] - current_state[:6]

        if step < force_open_steps:
            delta[6] = -1
        else:
            delta[6] = action_abs[6] - gripper_state

        obs, reward, done, info = env.step(delta)
        current_state = state_vector(env, obs)
        gripper_state = obs["robot0_gripper_qpos"][0]

        joint_states.append(current_state)
        delta_actions.append(delta)

        if "force_pred" in response:
            fp = np.array(response["force_pred"])
            force_preds.append(fp[0] if fp.ndim == 2 else fp[:6])

        try:
            ef = np.atleast_1d(np.asarray(env.robots[0].ee_force, dtype=np.float32))
            if ef.ndim == 0:
                ef = np.zeros(6)
            elif ef.shape[0] != 6:
                ef = np.zeros(6)
            actual_forces.append(ef)
        except Exception:
            actual_forces.append(np.zeros(6))

        rewards.append(reward)

        if done:
            break
        if step % 100 == 0:
            logger.info(f"  [step {step}] grip={gripper_state:.4f} reward={reward:.1f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"diag_seed{args.seed}_{ts}.npz"
    np.savez(
        out_path,
        joint_states=np.array(joint_states),
        delta_actions=np.array(delta_actions),
        force_preds=np.array(force_preds) if force_preds else np.zeros((0, 6)),
        actual_forces=np.array(actual_forces) if actual_forces else np.zeros((0, 6)),
        rewards=np.array(rewards),
        seed=args.seed,
    )
    logger.info(f"Saved to {out_path}")

    avg_reward = np.mean(rewards) if rewards else 0
    logger.info(f"Steps={len(rewards)}, avg_reward={avg_reward:.1f}")

    if force_preds and actual_forces:
        fp = np.array(force_preds)
        af = np.array(actual_forces)
        n = min(len(fp), len(af))
        fp, af = fp[:n], af[:n]
        mse = np.mean((fp - af) ** 2)
        logger.info(f"Force MSE: {mse:.4f}")
        for i, name in enumerate(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
            if fp[:, i].std() > 0 and af[:, i].std() > 0:
                corr = np.corrcoef(fp[:, i], af[:, i])[0, 1]
                logger.info(f"  {name}: corr={corr:.3f}")
            else:
                logger.info(f"  {name}: corr=N/A")


if __name__ == "__main__":
    main()
