import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # ForceVLA / LIMoE (Sparse MoE) configuration
    use_force: bool = False           # Whether to use force/torque input
    force_dim: int = 6                # Force/torque dimension (fx,fy,fz,tx,ty,tz)
    force_start_idx: int = 7          # Index in state where force data starts
    force_history_frames: int = 1     # Number of past force frames to use as input
    force_loss_weight: float = 1.0    # Weight for force loss (relative to joints=1.0)
    # force_head_loss_weight: weight applied to the force_out_proj HEAD path in
    #   grad_route_mode="three_stage". Default 1.0 (full weight, legacy behaviour).
    #   In the three_stage split, the head path was hardcoded to 1.0; this field
    #   makes it configurable (e.g. 0.01 to down-weight force prediction).
    force_head_loss_weight: float = 1.0
    # force_frame_spike_weight: scaling for the contact-frame spike weighting in
    #   the force supervised loss: frame_weight = 1 + |force_target| * spike_weight.
    #   Default 20.0 (legacy). Lower it (e.g. 2.0) to weaken contact-event emphasis.
    force_frame_spike_weight: float = 20.0
    num_experts: int = 4              # Number of experts in LIMoE
    num_top_k: int = 1                # Number of experts per token (top-k routing)

    # Dual-head configuration (Pi0Force).
    # control_action_dim: dimension of the robot control action (joints+gripper),
    #   NOT including force. When predict_force=True with a dual-head model,
    #   action_out_proj outputs control_action_dim dims and force_out_proj outputs
    #   force_dim dims separately. Defaults to 7 for single-arm Piper (6 joints + 1 gripper).
    #   Set to 8 for Panda (7 joints + 1 gripper). If None, falls back to action_dim
    #   (legacy single-head behaviour where force occupies action output dims).
    control_action_dim: int | None = None
    # force_head_stop_grad: if True, stop_gradient is applied to the shared features
    #   feeding the force head, so force_loss does NOT backprop into the PaliGemma
    #   backbone / action expert / LIMoE. Only force_out_proj receives force_loss
    #   gradients. Default False (force loss updates shared trunk).
    # NOTE: superseded by grad_route_mode="three_stage" when set; kept for the
    #   legacy binary stop_gradient behaviour.
    force_head_stop_grad: bool = False

    # ---- Three-stage gradient routing (dual-head multi-task) ----
    # grad_route_mode controls how action_loss and force_loss backprop into the
    # three parameter groups:
    #   "none"        : no routing; both losses update everything (legacy).
    #   "stop_grad"   : force_head_stop_grad-style — force_loss only updates
    #                   force_out_proj (stop_gradient on shared for the force path).
    #   "three_stage" : fine-grained routing:
    #       * VLM / vision (prefix_out)        <- action_loss only
    #       * action expert + LIMoE (suffix_out+limoe_out) <- action_loss + force_loss (weighted)
    #       * force_out_proj                   <- force_loss only
    #       * action_out_proj                  <- action_loss only
    #   Implemented by stopping gradient on prefix_out in the force-loss path so
    #   force_loss cannot reach the PaliGemma backbone, while still flowing into
    #   the action expert + LIMoE via suffix_out / limoe_out.
    grad_route_mode: str = "none"
    # Loss weights for the three-stage route. action_loss_weight scales the
    # action flow-matching loss; force_loss_weight scales the force supervised
    # loss. Both are configurable. Defaults 0.9 / 0.1 per the stamp_seal spec.
    action_loss_weight: float = 0.9
    # force_loss_weight above is reused as the force loss weight (default 0.1
    # when grad_route_mode="three_stage"; set explicitly in the config).

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
            # Always train limoe and force parameters -- they are new and not in base checkpoint.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*limoe.*")),
            )
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*force.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
