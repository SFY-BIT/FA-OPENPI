"""Pi0Force: Pi0 model augmented with LIMoE (Sparse MoE) and force input.

Extends the pi05-based Pi0 model with:
  - LIMoE (Latent-Input Mixture of Experts) block
  - Multi-frame force history input (K separate force tokens → LIMoE)
  - Weighted loss (joints vs force)
  - Optional force prediction head (13-dim action+force output)
"""

import dataclasses
import logging

import numpy as np
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import ft_encoder as _ft_encoder
from openpi.models.pi0 import Pi0 as _Pi0
from openpi.models.pi0 import posemb_sincos
from openpi.models.pi0 import make_attn_mask
from openpi.models import piper_fk_jax as _jfk
import openpi.models.gemma as _gemma
import openpi.models.limoe as _limoe
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Async component loss storage for wandb logging (zero training overhead).
# Written by jax.debug.callback inside compute_loss, read by training loop.
_COMPONENT_LOSSES: dict[str, float] = {}


def _store_component_losses(action_loss: float, force_loss: float,
                            eef_loss: float = 0.0, eef_pos_loss: float = 0.0,
                            eef_rot_loss: float = 0.0):
    _COMPONENT_LOSSES["action_loss"] = float(action_loss)
    _COMPONENT_LOSSES["force_loss"] = float(force_loss)
    _COMPONENT_LOSSES["eef_loss"] = float(eef_loss)
    _COMPONENT_LOSSES["eef_pos_loss"] = float(eef_pos_loss)
    _COMPONENT_LOSSES["eef_rot_loss"] = float(eef_rot_loss)


@dataclasses.dataclass(frozen=True)
class Pi0ForceConfig(pi0_config.Pi0Config):
    """Configuration for Pi0Force model."""

    # If True, the action output includes force prediction (14-dim).
    # The loss will include a weighted component on force dims 8:14.
    predict_force: bool = False

    # ---- Force history sequence encoding (FAWAM-style) ----
    # When True, use FTEncoder to encode a T-frame force history sequence into
    # a single global force feature vector, instead of K independent force tokens.
    use_ft_history: bool = False
    ft_history_steps: int = 60         # T: number of history frames
    ft_input_dim: int = 360            # T * 6
    ft_output_dim: int = 256           # FTEncoder output dimension
    ft_encoder_type: str = "mlp"       # "mlp" | "causal_conv" | "lstm" | "tcn"
    ft_num_tokens: int = 1             # K: number of force tokens emitted (1=single global, >1=multi-token)

    # ---- EEF pose loss (dual-space supervision) ----
    # When True, an extra loss on the end-effector pose (FK of predicted joints
    # vs FK of target joints) is added. This amplifies small wrist-joint errors
    # (q4-q6) that are drowned out in joint space. The model still outputs the
    # same 7-dim control action; EEF loss is computed in compute_loss via a
    # differentiable FK. No dataset changes needed — gt is FK(target joints).
    use_eef_loss: bool = False
    # Weight of the joint-space action loss (the rest goes to EEF loss).
    # total_action = action_joint_weight * joint_loss + (1 - action_joint_weight) * eef_loss
    action_joint_weight: float = 0.7
    # If True, only EEF loss is used for the action term (joint loss is computed
    # and logged but NOT added to the total loss). Ablation for EEF-only training.
    eef_only_mode: bool = False
    # EEF warmup steps: during the first `eef_warmup_steps` training steps the
    # EEF loss is scaled by min(1, step / eef_warmup_steps) (linear ramp).
    # This lets the model first learn a reasonable joint-space behavior before
    # the EEF pose supervision kicks in (avoids large initial EEF gradients).
    # 0 disables warmup.
    eef_warmup_steps: int = 0
    # Tool extension (m): link6 → gripper(0.13503) + force sensor(0.076) = 0.211
    tool_extension: float = 0.211
    # Weight of the rotation (orientation) part of the EEF loss relative to
    # position. 1.0 = balanced; >1 boosts orientation (wrist) supervision.
    eef_angle_weight: float = 1.0
    # Weight of the position part of the EEF loss.
    # eef_loss = eef_pos_weight * pos_loss + eef_angle_weight * rot_loss
    eef_pos_weight: float = 0.3
    # Quantile norm stats injected by LeRobotPiperDataConfig (physical-space
    # unnormalization before FK). Shapes [6] each (first 6 joint dims).
    eef_action_q01: np.ndarray | None = None
    eef_action_q99: np.ndarray | None = None
    eef_state_q01: np.ndarray | None = None
    eef_state_q99: np.ndarray | None = None

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Force":
        return Pi0Force(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        filters = []
        has_lora = False
        import openpi.shared.nnx_utils as nnx_utils

        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
            filters.append(nnx.Not(nnx_utils.PathRegex(".*limoe.*")))
            filters.append(nnx.Not(nnx_utils.PathRegex(".*force.*")))
            filters.append(nnx.Not(nnx_utils.PathRegex(".*state_proj.*")))
            filters.append(nnx.Not(nnx_utils.PathRegex(".*ft_encoder.*")))
            filters.append(nnx.Not(nnx_utils.PathRegex(".*ft_proj.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0Force(_Pi0):
    """Pi0 model augmented with LIMoE sparse MoE and force input."""

    def __init__(self, config: Pi0ForceConfig, rngs: nnx.Rngs):
        _Pi0.__init__(self, config, rngs)

        self.use_force = config.use_force
        self.predict_force = config.predict_force
        self.force_dim = config.force_dim                     # per-frame force dim (usually 6)
        self.force_start_idx = config.force_start_idx
        self.force_history_frames = config.force_history_frames  # K frames
        self.force_loss_weight = config.force_loss_weight
        self.force_head_loss_weight = config.force_head_loss_weight
        self.force_frame_spike_weight = config.force_frame_spike_weight
        # FT history sequence encoding config
        self.use_ft_history = config.use_ft_history
        self.ft_history_steps = config.ft_history_steps
        self.ft_input_dim = config.ft_input_dim
        self.ft_output_dim = config.ft_output_dim
        self.ft_encoder_type = config.ft_encoder_type
        self.ft_num_tokens = config.ft_num_tokens
        # EEF pose loss config
        self.use_eef_loss = config.use_eef_loss
        self.action_joint_weight = config.action_joint_weight
        self.eef_only_mode = config.eef_only_mode
        self.eef_warmup_steps = config.eef_warmup_steps
        self.tool_extension = config.tool_extension
        self.eef_angle_weight = config.eef_angle_weight
        self.eef_pos_weight = config.eef_pos_weight
        # Quantile norm stats for physical-space unnormalization before FK
        # (injected by LeRobotPiperDataConfig when use_eef_loss=True).
        # Stored as tuples: NNX flattens bare numpy/jax arrays as param leaves
        # ("Arrays leaves are not supported"), while tuples/lists are static.
        # compute_loss converts back with jnp.asarray(...).
        self.eef_action_q01 = tuple(np.asarray(config.eef_action_q01, dtype=np.float32)) if config.eef_action_q01 is not None else None
        self.eef_action_q99 = tuple(np.asarray(config.eef_action_q99, dtype=np.float32)) if config.eef_action_q99 is not None else None
        self.eef_state_q01 = tuple(np.asarray(config.eef_state_q01, dtype=np.float32)) if config.eef_state_q01 is not None else None
        self.eef_state_q99 = tuple(np.asarray(config.eef_state_q99, dtype=np.float32)) if config.eef_state_q99 is not None else None
        # Keep the original config.action_dim (e.g. 32) for state_proj input padding
        # and for the action head (action_in_proj / action_out_proj). The action head
        # ALWAYS operates in the full action_dim space (32), exactly like the base
        # pi05 — the real 7-dim control action is zero-padded to 32 by
        # PadStatesAndActions and the first 7 dims of action_out_proj's output are
        # taken as the control action. We do NOT rebuild action_in_proj /
        # action_out_proj so they stay shape-compatible with the base checkpoint.
        self.base_action_dim = config.action_dim

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Dual-head configuration.
        # When predict_force is True, a separate force_out_proj head is added that
        # predicts force_dim dims from the shared action-horizon features. The
        # action head (action_out_proj) is left UNTOUCHED at action_dim (32) — it
        # still outputs 32 dims, of which the first control_action_dim are the
        # control action and the rest are padding (ignored by the loss mask).
        # control_action_dim is only used to (a) mask the action loss to the real
        # control dims and (b) slice the final action output during inference.
        self.control_action_dim = config.control_action_dim
        self.force_head_stop_grad = config.force_head_stop_grad
        # Three-stage gradient routing config.
        self.grad_route_mode = config.grad_route_mode
        self.action_loss_weight = config.action_loss_weight

        # Independent force prediction head: shared features -> force_dim.
        # Named force_out_proj so it matches the .*force.* regex used by
        # get_freeze_filter (trainable under LoRA) and Pi0ForceWeightLoader
        # (kept randomly initialised when loading a base pi05 checkpoint).
        # action_in_proj / action_out_proj are NOT rebuilt — they keep the
        # base action_dim (32) from _Pi0.__init__ so the pretrained weights load.
        if self.predict_force:
            self.force_out_proj = nnx.Linear(
                action_expert_config.width, self.force_dim, rngs=rngs
            )
        else:
            self.force_out_proj = None

        # Per-frame force projection: each of the K history frames is projected
        # independently to a D-dim token via the same Linear layer.
        # When use_ft_history=True, this is replaced by ft_encoder + ft_proj.
        if self.use_force and not self.use_ft_history:
            self.force_in_proj = nnx.Linear(
                config.force_dim, paligemma_config.width, rngs=rngs
            )
        else:
            self.force_in_proj = None

        # FT history sequence encoder: replaces K independent force tokens with
        # K segment tokens produced by a shared temporal encoder (segment encoding).
        # The history (T frames) is split into ft_num_tokens segments of
        # ft_seg_frames frames each; every segment is encoded with the SAME encoder
        # (mirroring the legacy K per-frame tokens sharing one force_in_proj).
        # K=1 → seg_frames = T → identical to the original single-global-token path.
        if self.use_force and self.use_ft_history:
            self.ft_seg_frames = -(-config.ft_history_steps // config.ft_num_tokens)  # ceil(T/K)
            self.ft_seg_input_dim = self.ft_seg_frames * config.force_dim
            self.ft_encoder = _ft_encoder.FTEncoder(
                encoder_type=config.ft_encoder_type,
                input_dim=self.ft_seg_input_dim,   # frames-per-segment * 6 (K=1 → 360)
                output_dim=config.ft_output_dim,
                rngs=rngs,
            )
            self.ft_proj = nnx.Linear(
                config.ft_output_dim, paligemma_config.width, rngs=rngs
            )  # shared projection applied per segment token
        else:
            self.ft_encoder = None
            self.ft_proj = None
            self.ft_seg_frames = 0
            self.ft_seg_input_dim = 0

        # Pi05 mode does not create state_proj in _Pi0.__init__, but we need it
        # to inject proprioceptive state (joints+gripper) as a suffix token.
        # state_proj keeps config.action_dim (32) as its input dim to stay
        # shape-compatible with the base pi05 checkpoint (which has a
        # 32->width state_proj). The proprio (7-dim) is zero-padded to 32 inside
        # embed_suffix before being projected.
        if config.pi05 and not hasattr(self, 'state_proj'):
            self.state_proj = nnx.Linear(
                config.action_dim, action_expert_config.width, rngs=rngs
            )

        # LIMoE block
        self.limoe = nnx_bridge.ToNNX(
            _limoe.LIMoEBlock(
                mlp_dim=paligemma_config.width,
                num_experts=config.num_experts,
                num_top_k=config.num_top_k,
                num_heads=paligemma_config.num_heads,
                out_dim=action_expert_config.width,
            )
        )
        self.limoe.lazy_init(
            jnp.zeros((32, 200, paligemma_config.width)), True, rngs=rngs
        )

    # ---- Override embed_suffix: multi-frame force tokens ----

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
        at.Float[at.Array, "b k femb"] | None,  # force_tokens: [B, K, paligemma_width]
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]
        else:
            # Pi05: inject proprio state (joints+gripper only, no force) as a suffix token.
            # Force data enters exclusively via force_tokens → LIMoE.
            # proprio (force_start_idx dims) is zero-padded to base_action_dim (32)
            # so it matches the pretrained state_proj (32->width). self.action_dim
            # may be control_action_dim (7) in dual-head mode, which is NOT the
            # state_proj input dim — use base_action_dim for padding.
            proprio = obs.state[:, :self.force_start_idx]
            proprio_padded = jnp.pad(proprio, ((0, 0), (0, self.base_action_dim - self.force_start_idx)))
            state_token = self.state_proj(proprio_padded)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        # Force tokens: two paths —
        #   (A) use_ft_history:  FTEncoder → single global token
        #   (B) legacy:           K independent per-frame tokens via force_in_proj
        force_tokens = None
        if self.use_force:
            if self.use_ft_history:
                # Path A: temporal encoding of force history sequence
                # obs.ft_state: [B, T*6] flattened history
                ft_state = obs.ft_state
                if ft_state is None:
                    # Fallback: extract from state (backward compat)
                    K = self.force_history_frames
                    ft_state = obs.state[:, self.force_start_idx : self.force_start_idx + K * self.force_dim]
                # Segment encoding: split history into ft_num_tokens segments and
                # encode each with the shared encoder → [B, K, ft_output_dim].
                seg_feats = self.ft_encoder.encode_segments(
                    ft_state, self.ft_num_tokens
                )
                # Shared projection per segment token → [B, K, paligemma_width].
                # K=1 exactly reproduces the original single-global-token behavior.
                force_tokens = jax.vmap(self.ft_proj)(seg_feats)
            else:
                # Path B: K independent per-frame projections (legacy)
                K = self.force_history_frames
                B = obs.state.shape[0]
                force_flat = obs.state[:, self.force_start_idx : self.force_start_idx + K * self.force_dim]
                force_per_frame = force_flat.reshape(B, K, self.force_dim)
                force_tokens = jax.vmap(jax.vmap(self.force_in_proj))(force_per_frame)

        return tokens, input_mask, ar_mask, adarms_cond, force_tokens

    # ---- Override compute_loss: LIMoE + weighted loss ----

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        train_step: at.Array | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Training step for EEF warmup (passed from train_step; None e.g. during eval).
        # NOTE: do NOT stash this on the model instance (e.g. self._train_step = ...);
        # nnx value_and_grad splits model attributes as leaves and a traced array
        # attribute breaks the graph flatten.
        eef_cur_step = train_step

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, force_tokens = self.embed_suffix(
            observation, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # LIMoE: prefix_out + K force history tokens
        limoe_inputs = [prefix_out]
        if force_tokens is not None:
            # force_tokens: [B, K, D] — flatten to [B, K, D] as separate sequence positions
            limoe_inputs.append(force_tokens)
        limoe_input = jnp.concatenate(limoe_inputs, axis=1)
        limoe_out = self.limoe(limoe_input)

        # Shared action-horizon features: LIMoE output + action expert suffix output.
        shared = limoe_out[0][:, -self.action_horizon :] + suffix_out[:, -self.action_horizon :]

        if self.force_out_proj is not None:
            # ---- Dual-head mode ----
            # Gradient routing (grad_route_mode="three_stage", scheme B+):
            #   * VLM / vision (prefix_out)        <- 1.0 * action_loss  (pure action)
            #   * LIMoE + action expert            <- 1.0*action + force_loss_weight*force
            #   * action_out_proj                  <- 1.0 * action_loss
            #   * force_out_proj                   <- 1.0 * force_loss  (FULL weight, not 0.1)
            #
            # Implementation (single value_and_grad, no train_step change):
            #   - action path: full gradient through shared (reaches VLM + LIMoE
            #     + expert + action_out_proj).
            #   - force path: recompute LIMoE on stop_gradient(prefix_out) so
            #     force_loss CANNOT reach the PaliGemma backbone (VLM/vision),
            #     but still flows into LIMoE + action expert (via suffix_out and
            #     the re-computed limoe_out) and force_out_proj.
            #   - To give force_out_proj a FULL 1.0 force weight while LIMoE/expert
            #     only get force_loss_weight (0.1), we split the force loss into
            #     two paths with complementary stop_gradient on the Linear params:
            #       * head path:   stop_gradient(shared_force) blocks LIMoE/expert,
            #                      force_out_proj gets 1.0 * force_loss.
            #       * expert path: stop_gradient(force_out_proj.kernel/bias) blocks
            #                      force_out_proj, LIMoE/expert get force_loss_weight*force_loss.
            if self.grad_route_mode == "three_stage":
                # Force path: block gradient to VLM/vision (prefix_out).
                prefix_out_sg = jax.lax.stop_gradient(prefix_out)
                limoe_input_force = jnp.concatenate(
                    [prefix_out_sg] + ([force_tokens] if force_tokens is not None else []), axis=1
                )
                limoe_out_force = self.limoe(limoe_input_force)
                shared_force = limoe_out_force[0][:, -self.action_horizon :] + suffix_out[:, -self.action_horizon :]
                # action path: full gradient through shared.
                shared_action = shared

                # --- Split force loss into two isolated paths ---
                # Path 1 (force_out_proj only): stop_gradient on shared_force so
                # gradient flows ONLY into force_out_proj.kernel/bias.
                force_pred_head = self.force_out_proj(jax.lax.stop_gradient(shared_force))
                # Path 2 (LIMoE + action expert only): manual linear with
                # stop_gradient'd weights so gradient flows ONLY into shared_force
                # (-> LIMoE + suffix_out), NOT into force_out_proj.
                fop_kernel_sg = jax.lax.stop_gradient(self.force_out_proj.kernel.value)
                fop_bias_sg = jax.lax.stop_gradient(self.force_out_proj.bias.value)
                force_pred_expert = shared_force @ fop_kernel_sg + fop_bias_sg
            elif self.force_head_stop_grad:
                # Legacy binary stop_gradient: force_loss only updates force_out_proj.
                shared_action = shared
                shared_force = jax.lax.stop_gradient(shared)
                force_pred_head = self.force_out_proj(shared_force)
                force_pred_expert = None
            else:
                # No routing: both losses update everything.
                shared_action = shared
                shared_force = shared
                force_pred_head = self.force_out_proj(shared_force)
                force_pred_expert = None

            # Action head: outputs full action_dim (32) like the base pi05.
            # The real control action occupies the first control_action_dim dims;
            # the rest are zero-padding. Loss is masked to the control dims only
            # so the model is not penalised for the padding dimensions.
            action_pred = self.action_out_proj(shared_action)

            # Action flow-matching loss on the control action dims only.
            # u_t (noise - actions) is 32-wide (padded); mask to control dims.
            ctrl = self.control_action_dim if self.control_action_dim is not None else self.action_dim
            action_sq_error = jnp.square(action_pred[..., :ctrl] - u_t[..., :ctrl])
            action_loss = jnp.mean(action_sq_error, axis=-1)

            # ---- EEF pose loss (dual-space supervision, optional) ----
            # Amplify small wrist-joint errors (q4-q6) that are drowned out in
            # joint space, by comparing the end-effector pose reached by the
            # predicted joints vs the target joints. No dataset changes: both
            # sides are FK'ed on the fly from physical-space joints.
            #
            # Pipeline per horizon step h:
            #   pred_delta_norm  -> unnorm -> pred_delta_real     (joint delta)
            #   target_delta     -> unnorm -> target_delta_real   (joint delta)
            #   q_cur_norm       -> unnorm -> q_cur_real          (current joints)
            #   pred_pose = FK(q_cur_real + pred_delta_real)      (4x4)
            #   gt_pose   = FK(q_cur_real + target_delta_real)    (4x4)
            #   pos_loss  = ||xyz_pred - xyz_gt||^2
            #   rot_loss  = ||R_pred @ R_gt^T - I||^2_F  (no wrap, no gimbal)
            eef_loss = jnp.zeros_like(action_loss)
            eef_pos_loss = jnp.zeros_like(action_loss)
            eef_rot_loss = jnp.zeros_like(action_loss)
            if self.use_eef_loss:
                if (
                    self.eef_action_q01 is not None
                    and self.eef_state_q01 is not None
                ):
                    # Physical-space unnormalization (quantile): real = (n+1)/2*(q99-q01)+q01.
                    a_q01 = jnp.asarray(self.eef_action_q01)  # [6]
                    a_q99 = jnp.asarray(self.eef_action_q99)
                    s_q01 = jnp.asarray(self.eef_state_q01)
                    s_q99 = jnp.asarray(self.eef_state_q99)

                    # Current absolute joints: observation.state is [B, force_start_idx].
                    # First 6 dims are the joint angles (normalized).
                    q_cur_norm = observation.state[..., :6]                       # [B, 6]
                    q_cur_real = (q_cur_norm + 1.0) / 2.0 * (s_q99 - s_q01) + s_q01

                    # Targets: actions are [B, H, action_dim]; first 6 dims are the
                    # normalized joint deltas (DeltaActions already applied).
                    target_norm = actions[..., :6]                                # [B, H, 6]
                    # Predicted velocity u_t is d(noise-actions); recover predicted
                    # action as (noise - u_t), which for flow matching equals the
                    # target action at training time and the predicted one otherwise.
                    pred_norm = noise[..., :6] - action_pred[..., :6]             # [B, H, 6]

                    target_real = (target_norm + 1.0) / 2.0 * (a_q99 - a_q01) + a_q01
                    pred_real = (pred_norm + 1.0) / 2.0 * (a_q99 - a_q01) + a_q01

                    # Absolute target joints per horizon step: q_cur + cumulative delta.
                    # (DeltaActions stores delta relative to current state; each row of
                    # the chunk is a target *absolute* joint position, so the delta for
                    # step h is actions[h] — not cumulative.)
                    gt_joints = q_cur_real[:, None, :] + target_real              # [B, H, 6]
                    pred_joints = q_cur_real[:, None, :] + pred_real              # [B, H, 6]

                    # FK both: [B, H, 4, 4]
                    T_pred = _jfk.fk_batch(pred_joints.reshape(-1, 6), self.tool_extension).reshape(
                        -1, actions.shape[-3], 4, 4
                    )
                    T_gt = _jfk.fk_batch(gt_joints.reshape(-1, 6), self.tool_extension).reshape(
                        -1, actions.shape[-3], 4, 4
                    )

                    # Position loss (m).
                    pos_diff = T_pred[..., :3, 3] - T_gt[..., :3, 3]
                    pos_loss = jnp.mean(jnp.sum(pos_diff**2, axis=-1), axis=-1)   # [B]

                    # Rotation loss: ||R_pred @ R_gt^T - I||_F^2, no wrap/gimbal.
                    R_pred = T_pred[..., :3, :3]
                    R_gt = T_gt[..., :3, :3]
                    R_diff = R_pred @ jnp.swapaxes(R_gt, -1, -2) - jnp.eye(3)
                    rot_loss = jnp.mean(jnp.sum(R_diff**2, axis=(-2, -1)), axis=-1)  # [B]

                    eef_loss = self.eef_pos_weight * pos_loss + self.eef_angle_weight * rot_loss
                    # In eef_only_mode, use UNWEIGHTED EEF loss (pos+rot each 1.0)
                    # so the ablation isolates pure EEF supervision without the
                    # internal 0.3*pos + 2.0*rot weighting skewing the scale.
                    eef_loss_unweighted = pos_loss + rot_loss
                    eef_pos_loss = pos_loss
                    eef_rot_loss = rot_loss

            # Weighted action loss: joint-space + EEF (if enabled).
            if self.use_eef_loss and self.eef_only_mode:
                # EEF-only ablation: joint loss is computed/logged but NOT in total.
                # Use unweighted EEF (pos+rot @ 1.0 each) for a clean ablation.
                action_loss_weighted = eef_loss_unweighted
            elif self.use_eef_loss:
                # EEF warmup: scale EEF loss by min(1, step / eef_warmup_steps).
                # During warmup the joint loss dominates; EEF ramps in linearly.
                eef_scale = 1.0
                if self.eef_warmup_steps > 0:
                    if eef_cur_step is None:
                        eef_scale = 0.0  # no step info (e.g. eval) -> EEF off
                    else:
                        eef_scale = jnp.clip(
                            eef_cur_step / self.eef_warmup_steps, 0.0, 1.0
                        )
                joint_part = self.action_joint_weight * action_loss
                eef_part = (1.0 - self.action_joint_weight) * eef_loss * eef_scale
                action_loss_weighted = joint_part + eef_part
            else:
                action_loss_weighted = action_loss
                eef_loss = jnp.zeros_like(action_loss)

            # Force supervised loss against the separate force_target.
            #
            # IMPORTANT: delta force has a long-tailed distribution — most frames
            # have delta ≈ 0 (force stable), few frames have large delta (contact
            # events). Plain MSE is dominated by the stable frames, so force_out_proj
            # collapses to the trivial "always output 0" solution. We fix this with:
            #   1. Frame weighting: frames with large |force_target| get higher weight
            #      so the loss pays attention to contact events.
            #   2. Huber loss: robust to large residuals on contact frames, prevents
            #      stable frames from drowning out the gradient.
            if observation.force_target is not None:
                force_target = observation.force_target
                # u_t (flow-matching target) has shape [B, H, action_dim=32];
                # force_target has shape [B, H, force_dim]. They are independent.

                # Frame weight: emphasize contact-event frames (large |delta force|).
                # delta_mag: [B, H, 1] — mean absolute force_target across dims.
                # weight = 1 + delta_mag * spike_weight  (spike_weight from config)
                #   stable frame (delta≈0):   weight ≈ 1.0
                #   contact frame (|delta|=1): weight ≈ 1 + spike_weight
                delta_mag = jnp.mean(jnp.abs(force_target), axis=-1, keepdims=True)
                frame_weight = 1.0 + delta_mag * self.force_frame_spike_weight

                # Huber loss (delta=1.0): quadratic for |x|<1, linear for |x|>=1.
                # More robust than MSE for the long-tailed delta force distribution.
                def _huber(pred, target):
                    diff = pred - target
                    abs_diff = jnp.abs(diff)
                    quadratic = 0.5 * diff ** 2
                    linear = abs_diff - 0.5
                    return jnp.where(abs_diff < 1.0, quadratic, linear)

                # Path 1: force_out_proj gets FULL 1.0 weight (weighted Huber).
                force_loss_head = jnp.mean(
                    frame_weight * _huber(force_pred_head, force_target), axis=-1
                )
                # Path 2: LIMoE + action expert get force_loss_weight (0.1).
                if force_pred_expert is not None:
                    force_loss_expert = jnp.mean(
                        frame_weight * _huber(force_pred_expert, force_target), axis=-1
                    )
                else:
                    force_loss_expert = jnp.zeros_like(force_loss_head)
                # For wandb logging, report the head-path force loss (full weight).
                force_loss = force_loss_head
            else:
                force_loss_head = jnp.zeros_like(action_loss)
                force_loss_expert = jnp.zeros_like(action_loss)
                force_loss = jnp.zeros_like(action_loss)

            # Weighted total loss (scheme B+, single value_and_grad):
            #   total = 1.0 * action_loss
            #         + force_head_loss_weight * force_loss_head  (-> force_out_proj only)
            #         + force_loss_weight * force_loss_expert    (-> LIMoE+expert only)
            # action_loss_weight is implicit 1.0 here (VLM gets pure action).
            # force_head_loss_weight (default 1.0) scales the force_out_proj head
            # path; force_loss_weight (default 0.1) scales the LIMoE+expert group.
            # erase_board config sets both to 0.01 + frame spike 2.0.
            if self.grad_route_mode == "three_stage":
                total_loss = (
                    action_loss_weighted
                    + self.force_head_loss_weight * force_loss_head
                    + self.force_loss_weight * force_loss_expert
                )
            else:
                total_loss = action_loss_weighted + self.force_loss_weight * force_loss_head

            # Async component losses for wandb (zero training overhead).
            jax.debug.callback(
                _store_component_losses,
                jnp.mean(action_loss_weighted),
                jnp.mean(force_loss),
                jnp.mean(eef_loss),
                jnp.mean(eef_pos_loss),
                jnp.mean(eef_rot_loss),
            )
            return total_loss

        # ---- Legacy single-head mode (force occupies action output dims) ----        v_t = self.action_out_proj(shared)

        # Weighted MSE loss
        # action_dim=32: joints(7) + grip(1) [+ force(6) if predict_force]
        action_dim = self.action_dim
        loss_weights = jnp.zeros(action_dim)
        loss_weights = loss_weights.at[:7].set(1.0)        # joints: weight 1.0
        loss_weights = loss_weights.at[7].set(1.0)          # gripper: weight 1.0
        if self.predict_force:
            # Force dims 8:13 (fx,fy,fz,tx,ty,tz): lower weight
            loss_weights = loss_weights.at[8:14].set(self.force_loss_weight)

        sq_error = jnp.square(v_t - u_t)
        weighted_sq = sq_error * loss_weights
        loss = jnp.sum(weighted_sq, axis=-1) / jnp.maximum(jnp.sum(loss_weights), 1.0)

        # Split loss for wandb logging (async callback, zero training overhead)
        action_mask = jnp.zeros(action_dim).at[:8].set(1.0)
        force_mask = jnp.zeros(action_dim).at[8:14].set(1.0)
        action_loss = jnp.sum(sq_error * action_mask, axis=-1) / jnp.maximum(jnp.sum(action_mask), 1.0)
        force_loss = jnp.sum(sq_error * force_mask, axis=-1) / jnp.maximum(jnp.sum(force_mask), 1.0)
        jax.debug.callback(_store_component_losses,
                          jnp.mean(action_loss), jnp.mean(force_loss))

        return loss

    # ---- Override sample_actions_flow_loop: unpack force_tokens ----
    # NOTE: This override exists to handle the 5-value return of Pi0Force.embed_suffix.
    # It is NOT used by the default Policy inference path (which calls sample_actions
    # -> _flow_loop_with_limoe, because Pi0Force does not define the full
    # sample_actions_embed_image/embed_prompt/prefix_prefill breakdown). It is kept
    # for API completeness and direct callers. Unlike _flow_loop_with_limoe it does
    # NOT apply LIMoE fusion (prefix_out is not available in this signature); use
    # sample_actions for full LIMoE + dual-head inference.

    @override
    def sample_actions_flow_loop(
        self,
        observation: _model.Observation,
        prefix_mask,
        kv_cache,
        noise,
        *,
        num_steps: int = 10,
    ):
        """Override to handle 5 return values from Pi0Force.embed_suffix."""
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        prefix_len = prefix_mask.shape[1]

        def step(carry):
            x_t, time, _force_pred = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, _force_tokens = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            shared = suffix_out[:, -self.action_horizon :]
            v_t = self.action_out_proj(shared)
            if self.force_out_proj is not None:
                force_shared = jax.lax.stop_gradient(shared) if self.force_head_stop_grad else shared
                force_pred = self.force_out_proj(force_shared).astype(x_t.dtype)
            else:
                force_pred = None
            return x_t + dt * v_t, time + dt, force_pred

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        if self.force_out_proj is not None:
            init_force_pred = jnp.zeros((batch_size, self.action_horizon, self.force_dim), dtype=noise.dtype)
        else:
            init_force_pred = None
        x_0, _, force_pred = jax.lax.while_loop(cond, step, (noise, 1.0, init_force_pred))
        if self.force_out_proj is not None:
            return {"actions": x_0, "force_pred": force_pred}
        return x_0

    # ---- Override sample_actions ----

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions | dict:
        """Sample actions via flow matching with LIMoE fusion.

        Returns:
            In legacy single-head mode: an Actions tensor [b, ah, action_dim].
            In dual-head mode (force_out_proj is not None): a dict
                {"actions": [b, ah, control_action_dim],
                 "force_pred": [b, ah, force_dim] | None}.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        if noise is None:
            batch_size = observation.state.shape[0]
            # noise is in the full action_dim (32) space, matching action_in_proj.
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.sample_actions_embed_prefix(observation)
        prefix_mask, kv_cache, prefix_out_fix = self._prefill_with_prefix_out(
            prefix_tokens, prefix_mask, prefix_ar_mask
        )
        x_0, force_pred = self._flow_loop_with_limoe(
            observation, prefix_mask, kv_cache, prefix_out_fix, noise, num_steps=num_steps
        )
        if self.force_out_proj is not None:
            # x_0 is 32-wide (padded); slice to the real control action dims.
            ctrl = self.control_action_dim if self.control_action_dim is not None else self.action_dim
            return {"actions": x_0[..., :ctrl], "force_pred": force_pred}
        return x_0

    def _prefill_with_prefix_out(
        self,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, " s"],
    ):
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return prefix_mask, kv_cache, prefix_out

    def _flow_loop_with_limoe(
        self,
        observation: _model.Observation,
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache,
        prefix_out_fix: at.Float[at.Array, "b s emb"],
        noise: at.Float[at.Array, "b ah ad"],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        def step(carry):
            x_t, time, _force_pred = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, force_tokens = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )

            limoe_inputs = [prefix_out_fix]
            if force_tokens is not None:
                limoe_inputs.append(force_tokens)
            limoe_input = jnp.concatenate(limoe_inputs, axis=1)
            limoe_out = self.limoe(limoe_input)

            shared = limoe_out[0][:, -self.action_horizon :] + suffix_out[:, -self.action_horizon :]
            v_t = self.action_out_proj(shared)

            # In dual-head mode, also compute the force prediction each step so the
            # final carry carries the last-step force_pred. The force head is a single
            # Linear so the overhead is negligible. In legacy mode force_pred is None.
            if self.force_out_proj is not None:
                force_shared = jax.lax.stop_gradient(shared) if self.force_head_stop_grad else shared
                force_pred = self.force_out_proj(force_shared).astype(x_t.dtype)
            else:
                force_pred = None

            return (x_t + dt * v_t, time + dt, force_pred)

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        # Initial force_pred placeholder matching the step output structure.
        if self.force_out_proj is not None:
            init_force_pred = jnp.zeros((batch_size, self.action_horizon, self.force_dim), dtype=noise.dtype)
        else:
            init_force_pred = None
        x_0, _, force_pred = jax.lax.while_loop(cond, step, (noise, 1.0, init_force_pred))
        return x_0, force_pred
