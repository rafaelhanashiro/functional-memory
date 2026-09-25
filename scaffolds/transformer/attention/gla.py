"""Vector-gated linear attention adapted from OFM; see backends/gla_sources.json.

Each head has S_t = exp(log_alpha_t)[:,None] * S_{t-1} + k_t v_t^T
and an inclusive read q_t^T S_t, with no attention denominator or score scale.
The shared Transformer owns residual blocks, FFNs, training, and checkpoints.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from scaffolds.transformer.attention.positional import apply_rope


class HeadRMSNorm(nnx.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        self.scale = nnx.Param(jnp.ones((num_heads, head_dim)), weight_decay=False)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        normed = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return normed * self.scale


class GatedLinearAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        gate_rank: int = 16,
        gate_tau: float = 16.0,
        chunk_size: int = 64,
        inner_chunk_size: int = 16,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_rope: bool = True,
        backend: str = "blocked",
        *,
        rngs: nnx.Rngs,
    ):
        if min(hidden_size, num_heads, key_dim, value_dim, gate_rank) <= 0:
            raise ValueError("GLA dimensions and gate_rank must be positive")
        if not math.isfinite(gate_tau) or gate_tau <= 0:
            raise ValueError("gate_tau must be finite and positive")
        if use_rope and (key_dim // num_heads) % 2:
            raise ValueError("GLA key head dimension must be even for RoPE")
        if key_dim % num_heads != 0:
            raise ValueError("key_dim must be divisible by num_heads")
        if value_dim % num_heads != 0:
            raise ValueError("value_dim must be divisible by num_heads")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if inner_chunk_size <= 0:
            raise ValueError("inner_chunk_size must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_head_dim = key_dim // num_heads
        self.value_head_dim = value_dim // num_heads
        self.gate_tau = gate_tau
        self.chunk_size = chunk_size
        self.inner_chunk_size = inner_chunk_size
        self.rope_theta = rope_theta
        self.use_rope = use_rope
        if backend not in {"blocked", "fused"}:
            raise ValueError(f"Unsupported GLA backend: {backend}")
        self.backend = backend
        self.dtype = jnp.float32

        self.q_proj = nnx.Linear(hidden_size, key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, key_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, value_dim, use_bias=False, rngs=rngs)
        self.output_gate_proj = nnx.Linear(hidden_size, value_dim, use_bias=False, rngs=rngs)
        self.gate_low_rank = nnx.Linear(hidden_size, gate_rank, use_bias=False, rngs=rngs)
        self.gate_proj = nnx.Linear(gate_rank, key_dim, use_bias=True, rngs=rngs)
        self.head_norm = HeadRMSNorm(num_heads, self.value_head_dim, rms_norm_eps)
        self.dropout = nnx.Dropout(dropout)
        self.o_proj = nnx.Linear(value_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
        return_diagnostics: bool = False,
        coefficients: jax.Array | None = None,
        attention_parameters: dict[str, jax.Array] | None = None,
    ):
        if coefficients is not None or attention_parameters:
            raise ValueError("GLA does not accept polynomial attention parameters")
        result = self.forward_heads(
            x, deterministic=deterministic, rngs=rngs,
            return_diagnostics=return_diagnostics,
        )
        y, diagnostics = result if return_diagnostics else (result, None)
        batch, seq_len, _, _ = y.shape
        output = self.o_proj(y.reshape(batch, seq_len, self.value_dim))
        if return_diagnostics:
            diagnostics["invalid_output_count"] = jnp.sum(~jnp.isfinite(output), dtype=jnp.int32)
            return output, diagnostics
        return output

    def forward_heads(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
        return_diagnostics: bool = False,
    ):
        """Return the gated per-head read before the shared output projection."""
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).astype(self.dtype)
        k = self.k_proj(x).astype(self.dtype)
        v = self.v_proj(x).astype(self.dtype)

        gate_logits = self.gate_proj(self.gate_low_rank(x)).astype(jnp.float32)
        log_alpha = -jax.nn.softplus(-gate_logits) / self.gate_tau

        q = q.reshape(batch, seq_len, self.num_heads, self.key_head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.key_head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.value_head_dim)
        log_alpha = log_alpha.reshape(batch, seq_len, self.num_heads, self.key_head_dim)

        if self.use_rope:
            positions = jnp.arange(seq_len)
            q = apply_rope(q, positions, self.rope_theta)
            k = apply_rope(k, positions, self.rope_theta)

        if self.backend == "fused":
            # Import lazily so CPU/reference installations do not require the
            # optional Triton/JAX bridge unless this backend is selected.
            from scaffolds.transformer.attention.backends.chunked_gla_triton import CHUNK_SIZE, SUBCHUNK_SIZE, chunked_gla

            if self.chunk_size != CHUNK_SIZE or self.inner_chunk_size != SUBCHUNK_SIZE:
                raise ValueError(
                    "Fused GLA requires gla_chunk_size=64 and "
                    "gla_inner_chunk_size=16"
                )
            y, _ = chunked_gla(q, k, v, log_alpha)
        else:
            y = self._chunkwise_gla(q, k, v, log_alpha)
        raw_read = y
        y = self.head_norm(y)
        gate_projection = self.output_gate_proj(x).astype(self.dtype)
        gate = nnx.silu(gate_projection).reshape(
            batch,
            seq_len,
            self.num_heads,
            self.value_head_dim,
        )
        y = gate * y.astype(self.dtype)
        y = self.dropout(y, deterministic=deterministic, rngs=rngs)
        if return_diagnostics:
            diagnostics = {
                "log_decay_min": jnp.min(log_alpha),
                "log_decay_max": jnp.max(log_alpha),
                "read_abs_max": jnp.max(jnp.abs(raw_read.astype(jnp.float32))),
                "invalid_projection_count": sum(
                    jnp.sum(~jnp.isfinite(value), dtype=jnp.int32)
                    for value in (q, k, v, gate_logits, gate_projection)
                ),
                "invalid_gate_count": jnp.sum(
                    ~jnp.isfinite(log_alpha) | (log_alpha > 0), dtype=jnp.int32,
                ),
                "invalid_read_count": jnp.sum(~jnp.isfinite(raw_read), dtype=jnp.int32),
                "invalid_head_output_count": jnp.sum(~jnp.isfinite(y), dtype=jnp.int32),
            }
            return y, diagnostics
        return y

    def _chunkwise_gla(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        log_alpha: jax.Array,
    ) -> jax.Array:
        return chunkwise_gla(q, k, v, log_alpha, chunk_size=self.chunk_size)[0].astype(self.dtype)


def chunkwise_gla(q, k, v, log_alpha, *, chunk_size=64):
    """OFM associative reference with fp32 recurrent state and inclusive reads."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if q.ndim != 4 or q.shape != k.shape or q.shape != log_alpha.shape:
        raise ValueError("q, k, and log_alpha must have identical rank-4 shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3] or min(*q.shape, v.shape[-1]) <= 0:
        raise ValueError("GLA operands must have positive compatible [B,T,H,D] shapes")
    batch, seq_len, num_heads, key_head_dim = q.shape
    value_head_dim = v.shape[-1]
    state = jnp.zeros((batch, num_heads, key_head_dim, value_head_dim), dtype=jnp.float32)
    outputs = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        q_chunk = q[:, start:end].astype(jnp.float32)
        k_chunk = k[:, start:end].astype(jnp.float32)
        v_chunk = v[:, start:end].astype(jnp.float32)
        log_alpha_chunk = log_alpha[:, start:end].astype(jnp.float32)
        alpha = jnp.exp(log_alpha_chunk)[..., None]
        update = k_chunk[..., :, None] * v_chunk[..., None, :]

        def compose(left, right):
            alpha_left, update_left = left
            alpha_right, update_right = right
            return (
                alpha_right * alpha_left,
                alpha_right * update_left + update_right,
            )

        prefix_alpha, prefix_update = jax.lax.associative_scan(
            compose,
            (alpha, update),
            axis=1,
        )
        states = prefix_alpha * state[:, None] + prefix_update
        y_chunk = jnp.einsum("bthk,bthkv->bthv", q_chunk, states)
        state = states[:, -1]
        outputs.append(y_chunk)

    return jnp.concatenate(outputs, axis=1).astype(q.dtype), state
