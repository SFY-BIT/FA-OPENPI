import argparse
import dataclasses
import enum
import logging
import os
import socket
import sys

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


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            norm_stats = _load_norm_stats(args.policy.dir, args.policy.config,
                                          args.norm_stats_dir)
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                norm_stats=norm_stats,
                output_norm_stats=_filter_output_norm_stats(norm_stats),
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


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
