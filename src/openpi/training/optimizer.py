import dataclasses
import re
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig,
    lr_schedule: LRScheduleConfig,
    weight_decay_mask: at.PyTree | None = None,
    new_module_lr_multiplier: float = 1.0,
) -> optax.GradientTransformation:
    """Create optimizer, optionally with higher LR for new (from-scratch) modules.

    When new_module_lr_multiplier > 1.0, params matching ``.*limoe.*``,
    ``.*force.*`` or ``.*state_proj.*`` get multiplied LR. These are the modules
    that are NOT in the base pi05 checkpoint and must be trained from scratch.
    Everything else (PaliGemma, action_in_proj, action_out_proj, time_mlp_*)
    gets base LR since they load pretrained weights.
    """
    base_lr = lr_schedule.create()

    if new_module_lr_multiplier == 1.0:
        return optimizer.create(base_lr, weight_decay_mask=weight_decay_mask)

    # Differential LR: new modules × multiplier.
    # V2: RouterWeights gets base LR (1×) to prevent router_z_loss explosion.
    #     MlpBlock_0 (per-expert FFNs) + other limoe + force + state_proj get 5× LR
    #     to keep all experts competitive and prevent the "rusty expert" death spiral.
    new_lr = lambda step: base_lr(step) * new_module_lr_multiplier  # noqa: E731

    # Patterns for parameter group partitioning (higher priority checked first).
    router_pattern = re.compile(r".*RouterWeights.*")
    high_lr_pattern = re.compile(r".*limoe.*|.*force.*|.*state_proj.*")

    def partition_fn(params):
        # optax >= 0.2: param_labels maps the full pytree, not per-leaf.
        return jax.tree_util.tree_map_with_path(
            lambda path, _: (
                "base" if router_pattern.match("/".join(str(k) for k in path))
                else "new" if high_lr_pattern.match("/".join(str(k) for k in path))
                else "base"
            ),
            params,
        )

    tx = optax.multi_transform(
        {
            "base": optimizer.create(base_lr, weight_decay_mask=weight_decay_mask),
            "new": optimizer.create(new_lr, weight_decay_mask=weight_decay_mask),
        },
        partition_fn,
    )
    return tx
