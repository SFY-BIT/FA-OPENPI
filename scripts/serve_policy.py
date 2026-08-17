import argparse
import dataclasses
import enum
import logging
import os
import socket
import sys

import jax
import jax.numpy as jnp
import tyro


def _parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def _apply_jax_runtime_env(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--jax-cuda-visible-devices",
        type=str,
        default=None,
        help="Value to export to CUDA_VISIBLE_DEVICES before JAX is imported.",
    )
    parser.add_argument(
        "--jax-preallocate",
        type=_parse_bool_flag,
        default=None,
        help="Whether to export XLA_PYTHON_CLIENT_PREALLOCATE before JAX is imported.",
    )
    parser.add_argument(
        "--jax-mem-fraction",
        type=float,
        default=None,
        help="Optional XLA_PYTHON_CLIENT_MEM_FRACTION value to cap JAX GPU memory preallocation.",
    )
    parser.add_argument(
        "--jax-allocator",
        choices=("default", "platform"),
        default=None,
        help="Optional XLA_PYTHON_CLIENT_ALLOCATOR value. 'platform' uses on-demand allocation.",
    )
    early_args, remaining = parser.parse_known_args(argv)

    applied_settings: dict[str, str] = {}
    if early_args.jax_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = early_args.jax_cuda_visible_devices
        applied_settings["CUDA_VISIBLE_DEVICES"] = early_args.jax_cuda_visible_devices
    if early_args.jax_preallocate is not None:
        value = "true" if early_args.jax_preallocate else "false"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = value
        applied_settings["XLA_PYTHON_CLIENT_PREALLOCATE"] = value
    if early_args.jax_mem_fraction is not None:
        value = str(early_args.jax_mem_fraction)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = value
        applied_settings["XLA_PYTHON_CLIENT_MEM_FRACTION"] = value
    if early_args.jax_allocator is not None and early_args.jax_allocator != "default":
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = early_args.jax_allocator
        applied_settings["XLA_PYTHON_CLIENT_ALLOCATOR"] = early_args.jax_allocator

    return remaining, applied_settings


_REMAINING_ARGV, _APPLIED_JAX_ENV = _apply_jax_runtime_env(sys.argv[1:])

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


class ActionSpace(enum.Enum):
    """动作空间: joint（关节 delta/绝对）或 eef（末端 6D 位姿）。"""

    JOINT = "joint"
    EEF = "eef"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Local directory containing norm_stats.json, used when the checkpoint's own
    # assets (or the config's dataset path, e.g. a SLURM-only path) are unavailable.
    norm_stats_dir: str | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Action space: "joint" (default, 关节绝对位置) or "eef" (末端 EEF 位姿).
    # In EEF mode the server converts client joint obs -> EEF via FK before inference,
    # and converts model EEF actions -> joint via numerical IK after inference.
    # Requires piper_fk_jax (Piper 6-DoF, tool_extension 0.211 incl. sensor).
    action_space: ActionSpace = ActionSpace.JOINT
    # Model action semantics in EEF mode:
    #   "abs"   : model outputs absolute EEF pose (joint baseline / old EEF models)
    #   "delta" : model outputs relative delta w.r.t. current state (rot6d EEF models)
    action_rep: str = "abs"
    # Tool extension used for FK/IK when action_space=eef (default 0.211 = gripper + sensor).
    tool_extension: float = 0.211


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _load_norm_stats(checkpoint_dir: str, config_name: str,
                     norm_stats_dir: str | None = None) -> dict | None:
    """Load the full norm stats for a force config.

    Pi0Force models normalize extra *input-only* keys (`ft_state`, `force_target`)
    that are absent from the model output tree. `Unnormalize(strict=True)` in the
    output pipeline fails when such keys are present, so the full stats are used
    for input normalization and a filtered copy for output unnormalization.
    """
    import pathlib

    from openpi.shared import normalize as _normalize
    from openpi.training import checkpoints as _checkpoints

    # 1) Explicit --norm-stats-dir (e.g. local copy of a SLURM-only dataset path).
    if norm_stats_dir:
        try:
            norm_stats = _normalize.load(norm_stats_dir)
            logging.info("Loaded norm_stats from --norm-stats-dir: %s", norm_stats_dir)
            return norm_stats
        except Exception:
            logging.warning("Failed to load norm_stats from --norm-stats-dir: %s", norm_stats_dir)

    cfg = _config.get_config(config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_config.asset_id is None:
        return None
    try:
        norm_stats = _checkpoints.load_norm_stats(
            pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id
        )
    except Exception:
        norm_stats = None
    if norm_stats is None:
        # Fallback: try the config assets dir (e.g. dataset root for local repos).
        try:
            norm_stats = data_config.norm_stats
        except Exception:
            norm_stats = None
    if norm_stats is not None:
        logging.info("Loaded norm_stats keys: %s", list(norm_stats.keys()))
    return norm_stats


def _filter_output_norm_stats(norm_stats: dict | None) -> dict | None:
    """Return output-side norm stats for Unnormalize.

    ft_state is an *input* key (wrench_history, 360-dim), but it IS echoed back
    into the output data, so we keep it in the output norm_stats so that
    Unnormalize restores it to real-force space (ForceInStatePiperOutputs reads
    the last frame of ft_state as current_force for delta-mode force pred).
    force_target is the training-time force target; the model emits
    ``force_pred`` instead, so we RENAME the stats (not drop) so that
    Unnormalize can restore force_pred to real-force space.
    """
    if norm_stats is None:
        return None
    filtered = dict(norm_stats)
    if "force_target" in filtered:
        filtered["force_pred"] = filtered.pop("force_target")
    logging.info("Output norm_stats keys: %s", list(filtered.keys()))
    return filtered


class EefActionPolicyWrapper(_policy.Policy):
    """Wrap a policy for EEF action space (input FK, output IK).

    输入（client 发 joint）→ FK → EEF rot6d 给模型推理
    模型输出 EEF delta → 合成绝对 → IK → joint 返回 client

    坐标系与训练完全一致: piper_fk_jax 的 fk()/eef_ik() 都带 tool_extension
    (0.211 = 夹爪 0.13503 + 传感器 0.076), 与数据集转换 (rot6d 版) 同一套。

    支持两种模型输出语义:
      - action_rep="abs"   : 模型输出绝对 EEF 位姿 (joint baseline 模型的
                              joint 输出也走这里, 先 FK 成绝对)
      - action_rep="delta" : 模型输出相对当前 state 的 delta (rot6d 版
                              EEF 模型), 用 R_cur^T @ R_target 合成绝对
    """

    def __init__(self, policy: _policy.Policy, *, tool_extension: float = 0.211,
                 action_rep: str = "abs"):
        self._inner = policy
        self._tool_extension = tool_extension
        self._action_rep = action_rep
        # metadata 透传（用 _metadata 内部字段，metadata 是只读 property）
        self._metadata = policy.metadata
        # JIT 编译 IK（非 jit 单次 1.4s, jit 后 ~3ms）——首次调用时编译。
        from openpi.models import piper_fk_jax as _jfk
        self._eef_ik_jit = jax.jit(
            lambda tp, qi, Rm: _jfk.eef_ik(tp, qi, tool_extension=tool_extension, target_R=Rm)
        )

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _fk_abs(self, joints: np.ndarray) -> np.ndarray:
        """joint [6] → EEF 绝对位姿 [9] (xyz + rot6d)。无角度表示, 无不连续。"""
        import numpy as np
        from openpi.models import piper_fk_jax as _jfk
        T = np.asarray(_jfk.fk(jnp.asarray(joints, dtype=jnp.float32), self._tool_extension))
        xyz = T[:3, 3]
        R = T[:3, :3]
        d6 = R[:2, :].reshape(6)  # rot6d (与训练转换一致)
        return np.concatenate([xyz, d6]).astype(np.float32)

    def _rot6d_to_R(self, d6: np.ndarray) -> np.ndarray:
        """rot6d → 旋转矩阵 (Gram-Schmidt, 与 UMI/训练一致)。"""
        a1 = d6[0:3]
        a2 = d6[3:6]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=0)

    def _abs_to_rel(self, target_abs: np.ndarray, base_abs: np.ndarray) -> np.ndarray:
        """绝对 target → 相对 base 的 delta (矩阵合成)。target/base: [9] xyz+rot6d。"""
        d_xyz = target_abs[:3] - base_abs[:3]
        R_base = self._rot6d_to_R(base_abs[3:9])
        R_tgt = self._rot6d_to_R(target_abs[3:9])
        dR = R_base.T @ R_tgt
        return np.concatenate([d_xyz, dR[:2, :].reshape(6)]).astype(np.float32)

    def infer(self, obs: dict, *, noise=None) -> dict:
        import numpy as np
        from openpi.models import piper_fk_jax as _jfk

        # ── 输入转换: joint state [7] -> EEF state [10] (xyz+rot6d+grip) ──
        obs = dict(obs)
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.ndim > 1:
            state = state[0]
        joints = state[:6]
        eef_abs = self._fk_abs(joints)  # [9]
        # 保持 gripper 与力维度 (force 版 state 是 [10..16], grip 在 index 9)
        new_state = np.concatenate([eef_abs, state[6:]]).astype(np.float32)
        obs["observation/state"] = new_state
        logging.info("[EEF-mode] FK state: %s", np.round(new_state[:10], 4).tolist())

        result = self._inner.infer(obs, noise=noise)

        # ── 输出转换: EEF delta/abs [30, 10] -> joint action [30, 7] (IK) ──
        if "actions" in result:
            actions = np.asarray(result["actions"], dtype=np.float32)  # [30, 10]
            eef_delta = actions[..., :9]  # [30, 9]
            # 从当前绝对 EEF 出发迭代 (delta) 或直接用目标 (abs)。
            q_cur = joints
            cur_abs = eef_abs
            ik_joints = []
            n_fail = 0
            max_err = 0.0
            for h in range(eef_delta.shape[0]):
                if self._action_rep == "delta":
                    # delta: 模型输出相对当前 state 的 delta → 合成绝对
                    d = eef_delta[h]
                    target = np.zeros(9, dtype=np.float32)
                    target[:3] = cur_abs[:3] + d[:3]
                    R_cur = self._rot6d_to_R(cur_abs[3:9])
                    dR = self._rot6d_to_R(d[3:9])
                    target[3:9] = (R_cur @ dR)[:2, :].reshape(6)
                else:
                    # abs: 模型输出绝对 EEF 位姿 (或 joint 输出 FK 成绝对)
                    target = eef_delta[h]
                # IK: 传 [xyz] + target_R (rot6d→R), 完全绕开 rpy。
                R_tgt = self._rot6d_to_R(target[3:9])
                q_sol, converged, err = self._eef_ik_jit(
                    jnp.asarray(target[:3], dtype=jnp.float32),
                    jnp.asarray(q_cur, dtype=jnp.float32),
                    jnp.asarray(R_tgt, dtype=jnp.float32),
                )
                q_cur = np.asarray(q_sol, dtype=np.float32)
                if self._action_rep == "delta":
                    # 更新当前绝对位姿 (合成)
                    cur_abs = target
                ik_joints.append(q_cur)
                err_norm = float(np.asarray(np.linalg.norm(err)))
                max_err = max(max_err, err_norm)
                if not bool(np.asarray(converged)):
                    n_fail += 1
            ik_joints = np.stack(ik_joints)  # [30, 6]
            # 保持 gripper 列不变 (index 9)
            result["actions"] = np.concatenate([ik_joints, actions[..., 9:10]], axis=-1)
            logging.info(
                "[EEF-mode] IK done: %d/30 not-converged, max_err=%.5f",
                n_fail, max_err,
            )
        return result

    def reset(self) -> None:
        self._inner.reset()


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            norm_stats = _load_norm_stats(args.policy.dir, args.policy.config,
                                          args.norm_stats_dir)
            base_policy = _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                norm_stats=norm_stats,
                output_norm_stats=_filter_output_norm_stats(norm_stats),
            )
        case Default():
            base_policy = create_default_policy(args.env, default_prompt=args.default_prompt)

    # EEF action space: wrap with FK/IK conversion.
    if args.action_space == ActionSpace.EEF:
        logging.info("EEF action space enabled (FK input / IK output, tool_ext=%.3f, rep=%s)",
                     args.tool_extension, args.action_rep)
        return EefActionPolicyWrapper(
            base_policy,
            tool_extension=args.tool_extension,
            action_rep=args.action_rep,
        )
    return base_policy


def main(args: Args) -> None:
    if _APPLIED_JAX_ENV:
        logging.info("Applied JAX runtime env overrides: %s", _APPLIED_JAX_ENV)

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args, args=_REMAINING_ARGV))
