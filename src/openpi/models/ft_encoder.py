"""Force/torque (F/T) history sequence encoder for JAX/Flax.

Encodes a temporally-flattened F/T history sequence into a single global feature
vector, which can then be injected into the model (e.g. via LIMoE or AdaLN).

Input shape:  (B, T*6)  where T = history steps, 6 = [Fx,Fy,Fz,Tx,Ty,Tz] per step.
Output shape: (B, output_dim)

Supported encoder types:
  - mlp:         Directly encodes the flattened vector through MLP layers.
  - causal_conv: Reshapes to (B,T,6) and applies stacked causal 1D convolutions.
  - tcn:         Temporal Convolutional Network with dilated causal convs + residuals.
  - lstm:        Reshapes to (B,T,6) and applies LSTM, taking the last hidden state.
"""

from typing import Sequence
import flax.nnx as nnx
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv1D(nnx.Module):
    """Causal 1D convolution (left-only padding) in Flax/nnx.

    Equivalent to FAWAM's torch CausalConv1D: pad only on the left so that
    output at time t depends only on inputs at times ≤ t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        rngs: nnx.Rngs | None = None,
    ):
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(kernel_size,),
            strides=(1,),
            padding="VALID",
            feature_group_count=1,
            kernel_init=nnx.initializers.kaiming_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, C) → pad left → conv → (B, T_new, C_out)
        x = jnp.pad(x, ((0, 0), (self.padding, 0), (0, 0)))
        return self.conv(x)


class TemporalBlock(nnx.Module):
    """TCN block: two dilated causal convs + LayerNorm + ReLU + dropout + residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs | None = None,
    ):
        self.conv1 = CausalConv1D(in_channels, out_channels, kernel_size, dilation, rngs=rngs)
        self.norm1 = nnx.LayerNorm(out_channels, rngs=rngs)
        self.conv2 = CausalConv1D(out_channels, out_channels, kernel_size, dilation, rngs=rngs)
        self.norm2 = nnx.LayerNorm(out_channels, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs) if dropout_rate > 0 else None

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(1,),
                strides=(1,),
                padding="VALID",
                kernel_init=nnx.initializers.kaiming_uniform(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        residual = x
        out = jax.nn.relu(self.norm1(self.conv1(x)))
        if self.dropout is not None and train:
            out = self.dropout(out)
        out = jax.nn.relu(self.norm2(self.conv2(out)))
        if self.dropout is not None and train:
            out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return jax.nn.relu(out + residual)


# ---------------------------------------------------------------------------
# Encoder implementations
# ---------------------------------------------------------------------------

class FTMLPEncoder(nnx.Module):
    """MLP encoder: (B, input_dim) → Linear+SiLU → ... → (B, output_dim)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        rngs: nnx.Rngs | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        in_dim = input_dim
        for i, hd in enumerate(hidden_dims):
            setattr(self, f"linear_{i}", nnx.Linear(in_dim, hd, rngs=rngs))
            in_dim = hd
        self.num_layers = len(hidden_dims)
        self.output_proj = nnx.Linear(in_dim, output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        for i in range(self.num_layers):
            x = jax.nn.silu(getattr(self, f"linear_{i}")(x))
        x = self.output_proj(x)
        return x[0] if squeeze else x


class FTCausalConvEncoder(nnx.Module):
    """Stacked causal conv encoder: (B,T,6) → causal convs → (B, output_dim)."""

    def __init__(self, output_dim: int, rngs: nnx.Rngs | None = None):
        self.output_dim = output_dim
        self.input_dim = 6  # per-step channels

        self.conv1 = CausalConv1D(6, 32, kernel_size=3, dilation=1, rngs=rngs)
        self.conv2 = CausalConv1D(32, 64, kernel_size=3, dilation=2, rngs=rngs)
        self.conv3 = CausalConv1D(64, 64, kernel_size=3, dilation=4, rngs=rngs)
        self.conv4 = CausalConv1D(64, 128, kernel_size=3, dilation=8, rngs=rngs)
        self.conv5 = CausalConv1D(128, output_dim, kernel_size=3, dilation=16, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        B = x.shape[0]
        x = x.reshape(B, -1, self.input_dim)  # (B, T, 6)

        x = jax.nn.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = jax.nn.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = jax.nn.leaky_relu(self.conv3(x), negative_slope=0.1)
        x = jax.nn.leaky_relu(self.conv4(x), negative_slope=0.1)
        x = jax.nn.leaky_relu(self.conv5(x), negative_slope=0.1)

        x = x[:, -1, :]  # take last timestep
        return x[0] if squeeze else x


class FTLSTMEncoder(nnx.Module):
    """LSTM encoder: (B,T,6) → LSTM → last hidden → (B, output_dim).

    Uses Flax's nnx.LSTMCell in a scan for simplicity.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        rngs: nnx.Rngs | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Stacked LSTM cells
        self.lstm_cells = [
            nnx.LSTMCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                rngs=rngs,
            )
            for i in range(num_layers)
        ]
        self.output_proj = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        B = x.shape[0]
        x = x.reshape(B, -1, self.input_dim)  # (B, T, 6)
        T = x.shape[1]

        # Multi-layer LSTM via scan over time
        carry = [
            (
                nnx.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (B,), self.lstm_cells[i].hidden_size)
                if i == 0
                else nnx.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (B,), self.lstm_cells[i].hidden_size)
            )
            for i in range(self.num_layers)
        ]

        for t in range(T):
            inp = x[:, t, :]
            for i in range(self.num_layers):
                carry[i], inp = self.lstm_cells[i](carry[i], inp)

        # Final hidden state from top layer
        x = carry[-1][0]  # (B, hidden_dim)
        x = self.output_proj(x)
        return x[0] if squeeze else x


class FTTCNEncoder(nnx.Module):
    """TCN encoder: (B,T,6) → dilated causal convs + residual → pooling → (B, output_dim)."""

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 64,
        num_channels: Sequence[int] = (32, 64, 64, 128),
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        pooling: str = "last",
        rngs: nnx.Rngs | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling

        in_ch = input_dim
        blocks = []
        for i, out_ch in enumerate(num_channels):
            blocks.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, dilation=2**i, dropout_rate=dropout_rate, rngs=rngs
                )
            )
            in_ch = out_ch
        self.blocks = blocks
        self.output_proj = nnx.Linear(num_channels[-1], output_dim, rngs=rngs)
        self.output_norm = nnx.LayerNorm(output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        B = x.shape[0]
        x = x.reshape(B, -1, self.input_dim)

        for block in self.blocks:
            x = block(x, train=train)

        if self.pooling == "last":
            x = x[:, -1, :]
        elif self.pooling == "mean":
            x = x.mean(axis=1)
        elif self.pooling == "max":
            x = x.max(axis=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        x = self.output_norm(self.output_proj(x))
        return x[0] if squeeze else x


# ---------------------------------------------------------------------------
# Unified FTEncoder
# ---------------------------------------------------------------------------

class FTEncoder(nnx.Module):
    """Unified F/T history encoder.

    Input:  (B, input_dim)   flattened F/T history, input_dim = T * 6
    Output: (B, output_dim)  global force feature vector

    Args:
        encoder_type: one of 'mlp', 'lstm', 'tcn', or 'causal_conv'.
        input_dim: flattened input dimension (must be divisible by 6).
        output_dim: output feature dimension (e.g. 256).
    """

    def __init__(
        self,
        encoder_type: str = "mlp",
        input_dim: int = 360,
        output_dim: int = 256,
        mlp_hidden_dims: Sequence[int] = (512, 256),
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        tcn_num_channels: Sequence[int] = (32, 64, 64, 128),
        rngs: nnx.Rngs | None = None,
    ):
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        if encoder_type == "mlp":
            self.encoder = FTMLPEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=mlp_hidden_dims,
                rngs=rngs,
            )
        elif encoder_type == "lstm":
            self.encoder = FTLSTMEncoder(
                input_dim=6,
                hidden_dim=lstm_hidden_dim,
                output_dim=output_dim,
                num_layers=lstm_num_layers,
                rngs=rngs,
            )
        elif encoder_type == "tcn":
            self.encoder = FTTCNEncoder(
                input_dim=6,
                output_dim=output_dim,
                num_channels=tcn_num_channels,
                rngs=rngs,
            )
        elif encoder_type == "causal_conv":
            self.encoder = FTCausalConvEncoder(output_dim=output_dim, rngs=rngs)
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                "Use 'mlp', 'lstm', 'tcn', or 'causal_conv'."
            )

    def __call__(self, ft_data: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        """Encode flattened F/T history to a global feature vector.

        Args:
            ft_data: (B, input_dim) or (input_dim,).
        Returns:
            (B, output_dim) or (output_dim,).
        """
        # Pad/truncate to input_dim
        if ft_data.ndim == 1:
            cur = ft_data.shape[0]
            if cur < self.input_dim:
                ft_data = jnp.pad(ft_data, (self.input_dim - cur, 0))
            ft_data = ft_data[-self.input_dim :]
        else:
            cur = ft_data.shape[1]
            if cur < self.input_dim:
                ft_data = jnp.pad(ft_data, ((0, 0), (self.input_dim - cur, 0)))
            ft_data = ft_data[:, -self.input_dim :]

        if hasattr(self.encoder, "__call__") and "train" in self.encoder.__call__.__code__.co_varnames:
            return self.encoder(ft_data, train=train)
        return self.encoder(ft_data)

    def encode_segments(
        self, ft_data: jnp.ndarray, num_segments: int, *, train: bool = False
    ) -> jnp.ndarray:
        """Encode flattened F/T history into ``num_segments`` independent segment tokens.

        The history is split into ``num_segments`` equal chunks (tail zero-padded to
        ``ceil(T/K)`` frames each); every chunk is encoded with the SAME encoder, so
        the resulting ``(B, K, output_dim)`` tokens are semantically different but
        share one feature extractor — mirroring the legacy K independent per-frame
        tokens which shared a single ``force_in_proj``.

        The encoder must be built with ``input_dim = ceil(T/K) * 6`` (frames per
        segment). For K=1 this reduces to the standard single-global-token behavior.

        Args:
            ft_data: (B, input_total) or (input_total,), flattened F/T history.
            num_segments: K, number of output tokens.

        Returns:
            (B, K, output_dim) segment features.
        """
        was_1d = ft_data.ndim == 1
        if was_1d:
            ft_data = ft_data[None, :]
        B = ft_data.shape[0]
        frames_per_seg = self.input_dim // 6           # frames per segment (ceil(T/K))
        target = frames_per_seg * num_segments * 6      # flattened frames across all segments
        cur = ft_data.shape[1]
        if cur < target:
            ft_data = jnp.pad(ft_data, ((0, 0), (target - cur, 0)))
        ft_data = ft_data[:, -target:]
        # [B, K, frames_per_seg, 6]
        segs = ft_data.reshape(B, num_segments, frames_per_seg, 6)

        accepts_train = (
            hasattr(self.encoder, "__call__")
            and "train" in self.encoder.__call__.__code__.co_varnames
        )

        def _encode(x):
            if accepts_train:
                return self.encoder(x, train=train)
            return self.encoder(x)

        if self.encoder_type == "mlp":
            # MLP expects flattened (B, frames_per_seg*6) per segment.
            segs_flat = segs.reshape(B, num_segments, frames_per_seg * 6)
        else:
            # Sequence encoders (lstm/tcn/causal_conv) expect (B, frames_per_seg, 6).
            segs_flat = segs

        out = jax.vmap(_encode, in_axes=1, out_axes=1)(segs_flat)  # [B, K, output_dim]
        if was_1d:
            out = out[0]
        return out
