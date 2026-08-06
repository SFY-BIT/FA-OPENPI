"""Force-aware Piper policy — passes wrist force/torque data into the model.

Extends the standard Piper policy to:
  1. Concatenate wrist force/torque into state (13+ dims)
  2. Optionally use K frames of force history as input
  3. Optionally predict next-frame force alongside actions (13-dim output)

With force_history_frames=K, state becomes:
  [joints(7), force_t0(3), torque_t0(3), ..., force_t{K-1}(3), torque_t{K-1}(3)]
  = 7 + K*6 dims
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _extract_force_history(data, key: str, history_frames: int) -> np.ndarray:
    """Extract K frames of force/torque history from (possibly chunked) data.

    Priority:
      1. Explicit past history field `<key>_history`: expected shape [K, D]
      2. Chunked field [H, D]: takes the first K frames
      3. Single-frame [D]: tiles it K times (legacy fallback)
    """
    history_key = f"{key}_history"
    if history_key in data:
        raw_history = np.asarray(data[history_key], dtype=np.float32)
        if raw_history.ndim != 2:
            raise ValueError(f"{history_key} must have shape [K, D], got {raw_history.shape}")
        K = min(history_frames, raw_history.shape[0])
        result = raw_history[-K:].reshape(-1)
        if K < history_frames:
            pad = np.zeros((history_frames - K, raw_history.shape[-1]), dtype=raw_history.dtype)
            result = np.concatenate([pad.reshape(-1), result])
        return result

    raw = np.asarray(data.get(key, np.zeros(3)))
    if raw.ndim == 2:
        # Chunked: [action_horizon, D] → take first K frames
        K = min(history_frames, raw.shape[0])
        result = raw[:K].reshape(-1)
        # Pad if fewer than history_frames
        if K < history_frames:
            pad = np.tile(raw[0], history_frames - K)
            result = np.concatenate([result, pad])
        return result
    else:
        # Single frame: [D] → tile K times
        return np.tile(raw, history_frames)


@dataclasses.dataclass(frozen=True)
class ForcePiperInputs(transforms.DataTransformFn):
    """Converts Piper observations/actions into model input format.

    Builds state with optional force history and optionally expands the
    action target to include next-frame force prediction.
    """

    model_type: _model.ModelType
    predict_force: bool = False
    force_history_frames: int = 1

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        K = self.force_history_frames

        # Build state: joints(7) + K frames of force(3) + K frames of torque(3)
        joint_state = np.asarray(data["observation/state"])
        force_hist = _extract_force_history(data, "observation/wrist_force", K)
        torque_hist = _extract_force_history(data, "observation/wrist_torque", K)
        state = np.concatenate([joint_state, force_hist, torque_hist], axis=-1)
        # state = [joints(7), force_t0(3), ..., force_t{K-1}(3),
        #          torque_t0(3), ..., torque_t{K-1}(3)] = 7 + K*6

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions: optionally include force prediction target
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if self.predict_force:
                force_next = np.asarray(data["observation/wrist_force"])
                torque_next = np.asarray(data["observation/wrist_torque"])
                if force_next.ndim == 1:
                    force_next = force_next[None, :]
                if torque_next.ndim == 1:
                    torque_next = torque_next[None, :]
                # concat: [actions(7), force_next(3), torque_next(3)] = 13
                actions = np.concatenate([actions, force_next, torque_next], axis=-1)
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ForcePiperOutputs(transforms.DataTransformFn):
    """Converts model outputs back to Piper action + optional force format."""

    predict_force: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        result = {"actions": actions[:, :8]}  # 8-dim for robot control (7 joints + 1 gripper)
        if self.predict_force:
            result["force_pred"] = actions[:, 8:14]  # 6-dim force for comparison
        return result


@dataclasses.dataclass(frozen=True)
class ForceInStatePiperInputs(transforms.DataTransformFn):
    """Converts Piper observations/actions into model input format when the
    force/torque data is already stored inside ``observation.state``.

    Expected dataset format (e.g. stamp_seal_flexiv on Piper/Flexiv):
        observation.state = [q1..q6, gripper, Fx, Fy, Fz, Tx, Ty, Tz]  (13-dim)
        action             = [target_q1..target_q6, target_gripper]   (7-dim)

    This transform splits the state into proprio (joints+gripper) and force,
    and builds the model input state as:
        [proprio(force_start_idx), K frames of force(force_dim)]
        = force_start_idx + K*force_dim dims
    which is identical to the layout produced by ``ForcePiperInputs``, so the
    model-side force extraction (``obs.state[force_start_idx:...]``) is unchanged.

    When ``predict_force=True`` the force prediction target is taken from the
    future state chunk (``observation/state`` sampled with delta_timestamps to
    shape [action_horizon, state_dim]) and emitted as a SEPARATE ``force_target``
    key of shape [action_horizon, force_dim], instead of being concatenated onto
    the action target. This enables the dual-head model architecture.

    When ``use_ft_history=True``, precomputed force history is read from the
    ``observation.wrench_history`` column [T, 6], flattened to [T*6], and
    emitted as a SEPARATE ``ft_state`` key. The model ``state`` is then just
    proprio (``force_start_idx`` dims, no force history appended). This enables
    the FTEncoder-based temporal encoding path.
    """

    model_type: _model.ModelType
    predict_force: bool = False
    force_history_frames: int = 1
    force_start_idx: int = 7
    force_dim: int = 6
    use_ft_history: bool = False       # If True, read wrench_history → ft_state
    ft_history_steps: int = 60         # T: number of history frames in wrench_history

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        K = self.force_history_frames
        T = self.ft_history_steps

        # The raw state already contains force in dims [force_start_idx:force_start_idx+force_dim].
        # When observation/state is listed in action_sequence_keys, the data loader samples it
        # to a future chunk [H, state_dim] via delta_timestamps. The current frame is the first
        # row of that chunk (t=0); the full chunk is used below as the force_target source.
        raw_state = np.asarray(data["observation/state"], dtype=np.float32)
        if raw_state.ndim == 2:
            # Chunked [H, state_dim]: current frame = first row.
            state_chunk = raw_state
            current_state = raw_state[0]
        else:
            # Single frame [state_dim].
            state_chunk = None
            current_state = raw_state
        proprio = current_state[: self.force_start_idx]
        current_force = current_state[self.force_start_idx : self.force_start_idx + self.force_dim]

        # ---- Force history path selection ----
        if self.use_ft_history:
            # NEW: precomputed wrench_history [T, 6] → flattened ft_state [T*6]
            wrench_key = "observation/wrench_history"
            if wrench_key in data:
                wrench = np.asarray(data[wrench_key], dtype=np.float32)  # [T, 6]
                if wrench.ndim < 2:
                    wrench = wrench.reshape(T, self.force_dim)
                T_avail = wrench.shape[0]
                if T_avail < T:
                    pad = np.zeros((T - T_avail, self.force_dim), dtype=np.float32)
                    wrench = np.concatenate([pad, wrench], axis=0)
                else:
                    wrench = wrench[-T:]
                ft_state_val = wrench.reshape(-1)  # [T*6]
            else:
                # Fallback: tile current force T times
                ft_state_val = np.tile(current_force, T).reshape(-1)

            # state = proprio only (no force appended)
            state = proprio.astype(np.float32)

        else:
            # LEGACY: build K-frame force history from state_history or tiled current frame
            history_key = "observation/state_history"
            if history_key in data:
                raw_history = np.asarray(data[history_key], dtype=np.float32)
                if raw_history.ndim == 2:
                    force_hist_frames = raw_history[:, self.force_start_idx : self.force_start_idx + self.force_dim]
                else:
                    force_hist_frames = raw_history.reshape(K, self.force_dim)
                K_avail = force_hist_frames.shape[0]
                if K_avail < K:
                    pad = np.zeros((K - K_avail, self.force_dim), dtype=np.float32)
                    force_hist_frames = np.concatenate([pad, force_hist_frames], axis=0)
                else:
                    force_hist_frames = force_hist_frames[-K:]
                force_hist = force_hist_frames.reshape(-1)
            else:
                force_hist = np.tile(current_force, K)

            # Model input state = [proprio, K frames of force] (matches ForcePiperInputs layout).
            state = np.concatenate([proprio, force_hist], axis=-1)
            ft_state_val = None

        # ---- Build model inputs dict ----
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if ft_state_val is not None:
            inputs["ft_state"] = ft_state_val

        # Action target: robot control only (force_start_idx dims), NOT concatenated with force.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # Keep only the control dims (joints+gripper). If the dataset action already
            # has exactly force_start_idx dims this is a no-op; if it is wider (e.g. legacy
            # 13-dim) we truncate to the control dims.
            if actions.shape[-1] > self.force_start_idx:
                actions = actions[..., : self.force_start_idx]
            inputs["actions"] = actions

        # Force prediction target: separate key, aligned with the action chunk.
        # When observation/state was chunked ([H, state_dim]), slice the force cols
        # from the full chunk. Otherwise fall back to the current-frame force.
        #
        # DELTA mode (mirrors action DeltaActions): the force target is stored as
        # a DELTA relative to the current frame's force (force_target - current_force).
        # This improves precision because the model predicts force *changes* rather
        # than absolute magnitudes (which vary across episodes/contacts). The
        # inverse (adding current_force back) is done in ForceInStatePiperOutputs
        # at inference time. norm_stats are computed on the delta, so Normalize /
        # Unnormalize operate in delta space consistently.
        if self.predict_force:
            if state_chunk is not None:
                force_target_abs = state_chunk[:, self.force_start_idx : self.force_start_idx + self.force_dim]
            else:
                force_target_abs = current_force[None, :]
            # Convert to delta: force_target_delta[h] = force_target_abs[h] - current_force
            # current_force shape [force_dim], broadcasts over horizon H.
            force_target = force_target_abs - current_force
            inputs["force_target"] = force_target

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ForceInStatePiperOutputs(transforms.DataTransformFn):
    """Converts dual-head model outputs back to Piper action + force format.

    Unlike ``ForcePiperOutputs`` (which slices force out of a single action
    tensor), this expects the dual-head outputs where ``actions`` is the
    control action (force_start_idx dims) and ``force_pred`` is an independent
    key (force_dim dims).

    DELTA mode: the model predicts force *deltas* relative to the current
    frame's force. Here we add the current force back to recover absolute
    force. The current force is taken from the input ``state`` which has the
    layout [proprio(force_start_idx), K frames of force(force_dim)] produced
    by ForceInStatePiperInputs — the LAST K force values are the newest frame.
    """

    predict_force: bool = False
    control_action_dim: int = 7
    force_start_idx: int = 7
    force_dim: int = 6
    force_history_frames: int = 1

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        result = {"actions": actions[:, : self.control_action_dim]}
        if self.predict_force and "force_pred" in data:
            force_pred_delta = np.asarray(data["force_pred"])
            # Recover absolute force: force_abs = force_delta + current_force.
            #
            # use_ft_history (16-token) models carry the real force history in
            # ft_state (T*force_dim; the output Unnormalize restores it to real
            # force space). The newest frame is the LAST force_dim values.
            # Fall back to the legacy state[force_start_idx:...] layout otherwise.
            if "ft_state" in data:
                ft_state = np.asarray(data["ft_state"])
                flat = ft_state.reshape(-1)
                if flat.shape[-1] >= self.force_dim:
                    current_force = flat[-self.force_dim :].astype(np.float32)
                else:
                    current_force = np.zeros(self.force_dim, dtype=np.float32)
            else:
                state = np.asarray(data["state"])
                current_force = state[self.force_start_idx + (self.force_history_frames - 1) * self.force_dim :
                                      self.force_start_idx + self.force_history_frames * self.force_dim]
            result["force_pred"] = force_pred_delta + current_force
        return result
