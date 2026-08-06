"""Smoke test: verify ft_num_tokens (K) segment-encoding.

Checks that:
  - K=1: ft_proj is Linear(256, 2048), force_tokens shape [B, 1, 2048]
  - K=2: force_tokens shape [B, 2, 2048]
  - K=16: ft_encoder input_dim = ceil(60/16)*6 = 24, force_tokens shape [B, 16, 2048]
No training, no checkpoint loading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp
from openpi.models import pi0_force as _pf


def make_obs():
    from openpi.models import model as _model
    rng = jax.random.PRNGKey(0)
    return _model.Observation(
        images={},                                     # embed_suffix does not touch images
        image_masks={},
        state=jnp.zeros((2, 7), dtype=jnp.float32),      # proprio only (ft_history mode)
        # Non-zero random history so different segments carry different content.
        ft_state=jax.random.normal(rng, (2, 360), dtype=jnp.float32),  # [B, T*6]
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )


for K in (1, 2, 16):
    cfg = _pf.Pi0ForceConfig(
        pi05=True, action_horizon=30, discrete_state_input=False,
        paligemma_variant="gemma_2b",   # dense variant: build without pretrained weights
        action_expert_variant="gemma_300m",
        use_force=True, predict_force=True,
        control_action_dim=7,
        force_start_idx=7, force_dim=6, force_history_frames=2,
        use_ft_history=True, ft_history_steps=60,
        ft_input_dim=360, ft_output_dim=256, ft_encoder_type="mlp",
        ft_num_tokens=K,
        grad_route_mode="three_stage",
        num_experts=4, num_top_k=1,
    )
    model = cfg.create(jax.random.PRNGKey(0))
    print(f"K={K}: ft_encoder.input_dim = {model.ft_encoder.input_dim}  "
          f"(expect {((60 + K - 1)//K)*6})  ft_proj kernel = {model.ft_proj.kernel.shape}")

    obs = make_obs()
    # action_in_proj expects base action_dim (32); in the real pipeline
    # PadStatesAndActions zero-pads the 7-dim control action to 32.
    actions = jnp.zeros((2, 30, 32), dtype=jnp.float32)
    t = jnp.zeros((2,), dtype=jnp.float32)
    tokens, mask, ar_mask, adarms, force_tokens = model.embed_suffix(obs, actions, t)
    print(f"K={K}: force_tokens shape = {force_tokens.shape}  (expect [2, {K}, 2048])")
    assert force_tokens.shape == (2, K, 2048), force_tokens.shape

    # Segment tokens must be semantically different (not identical projections).
    if K > 1:
        ft = force_tokens  # [2, K, 2048]
        diffs = jnp.abs(ft[:, 0] - ft[:, 1]).sum()
        print(f"K={K}: |tok0 - tok1| sum = {diffs:.4f} (should be > 0)")
        assert diffs > 0
    print(f"K={K}: OK")
    print()

print("ALL PASSED")

