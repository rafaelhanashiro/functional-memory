"""Causal attention with a matched blocked path for Taylor and exact kernels."""

import math
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from experiments.polynomial_attention.models.components.positional import apply_rope


AttentionBackend = Literal["blocked", "fused", "manual", "auto", "xla", "cudnn"]
QUERY_BLOCK_SIZE = 128


def _validate_degree(attention_degree: int | None) -> None:
    if attention_degree is not None and (
        isinstance(attention_degree, bool)
        or not isinstance(attention_degree, int)
        or attention_degree < 0
    ):
        raise ValueError("attention_degree must be None or a nonnegative integer")


def _taylor_polynomial(scores: jax.Array, degree: int) -> jax.Array:
    """Nested Horner evaluation without underflowing factorial coefficients."""
    result = jnp.ones_like(scores)
    for ell in range(degree, 0, -1):
        result = 1.0 + scores * (result / ell)
    return result


def _quadratic_polynomial(scores: jax.Array, coefficients: jax.Array, mode: str) -> jax.Array:
    """Evaluate raw learned parameters over [batch, heads, queries, keys] scores."""
    coefficients = jnp.broadcast_to(coefficients, (scores.shape[1], 2)).astype(jnp.float32)
    first, second = coefficients[:, 0][None, :, None, None], coefficients[:, 1][None, :, None, None]
    if mode == "positive":
        return 0.5 * (1.0 + jnp.square(first * scores + second))
    return 1.0 + scores * (first + scores * second)


def blocked_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    attention_degree: int | None = None,
    coefficients: jax.Array | None = None,
    quadratic_mode: str = "fixed",
    query_block_size: int = QUERY_BLOCK_SIZE,
    return_diagnostics: bool = False,
) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    """Attend to the full causal prefix using bounded, recomputed score blocks.

    Inputs are post-RoPE [batch, time, heads, channels]. Polynomial and reduction
    arithmetic is FP32, and the result has V's dtype. The same blocked path is
    used for every degree and the exact-exponential control. No scores are
    shifted in the Taylor branch. A checkpointed scan body prevents retaining
    the [batch, heads, query_block, time] intermediates for every block/layer.

    Invalid causal scores, nonfinite kernels, and invalid normalizers poison
    their rows with NaN. Even Taylor and positive learned kernels must stay
    positive; odd Taylor and unrestricted quadratics permit signed/zero kernels
    and negative nonzero normalizers. Stable exact exponentials may underflow to zero.
    Diagnostics exclude padded queries and masked future entries.
    """
    _validate_degree(attention_degree)
    if quadratic_mode not in ("fixed", "positive", "unrestricted"):
        raise ValueError(f"Unknown quadratic mode: {quadratic_mode}")
    if quadratic_mode != "fixed" and (attention_degree != 2 or coefficients is None):
        raise ValueError("Learned quadratic attention requires degree=2 and coefficients")
    if quadratic_mode == "fixed" and coefficients is not None:
        raise ValueError("Fixed attention does not accept learned coefficients")
    signed = quadratic_mode == "unrestricted" or (
        quadratic_mode == "fixed" and attention_degree is not None and attention_degree % 2 == 1
    )
    extended_diagnostics = quadratic_mode != "fixed" or signed
    if not isinstance(query_block_size, int) or isinstance(query_block_size, bool) or query_block_size <= 0:
        raise ValueError("query_block_size must be a positive integer")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must have shape [batch, time, heads, channels]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K shapes and Q/K/V batch, time, and head dimensions must match")
    batch, time, heads, qk_dim = q.shape
    if coefficients is not None and coefficients.shape not in ((2,), (heads, 2)):
        raise ValueError("Quadratic coefficients must have shape (2,) or (heads, 2)")
    if min(batch, time, heads, qk_dim, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    # Avoid excessive padding for short CPU/reference checks; production uses 128.
    block_size = min(query_block_size, time)
    block_count = (time + block_size - 1) // block_size
    pad = block_count * block_size - time
    queries = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
    queries = queries.reshape(batch, block_count, block_size, heads, qk_dim)
    queries = jnp.swapaxes(queries, 0, 1)
    starts = jnp.arange(block_count, dtype=jnp.int32) * block_size
    key_positions = jnp.arange(time)

    def attend_block(_, inputs):
        query, start = inputs
        query_positions = start + jnp.arange(block_size)
        causal = key_positions[None, :] <= query_positions[:, None]
        actual_rows = query_positions < time
        measured = causal & actual_rows[:, None]
        # Preserve BF16 input compute with FP32 dot-product accumulation.
        scores = jnp.einsum(
            "bqhd,bkhd->bhqk", query, k,
            preferred_element_type=jnp.float32,
        ) * (1.0 / math.sqrt(qk_dim))
        if attention_degree is None:
            # Shifting is an exact identity only for the exponential control.
            maximum = jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1, keepdims=True)
            safe_scores = jnp.where(causal, scores - maximum, 0.0)
            kernel = jnp.exp(safe_scores)
            invalid_kernel = (~jnp.isfinite(kernel)) | (kernel < 0)
        else:
            # Evaluate masked positions at zero, never at masking sentinels.
            safe_scores = jnp.where(causal, scores, 0.0)
            kernel = (_taylor_polynomial(safe_scores, attention_degree) if quadratic_mode == "fixed"
                      else _quadratic_polynomial(safe_scores, coefficients, quadratic_mode))
            invalid_kernel = ~jnp.isfinite(kernel)
            if not signed:
                invalid_kernel = invalid_kernel | (kernel <= 0)
        kernel = jnp.where(causal, kernel, 0.0)
        normalizer = jnp.sum(kernel, axis=-1, keepdims=True, dtype=jnp.float32)
        invalid_score = ~jnp.isfinite(scores)
        invalid_normalizer = (~jnp.isfinite(normalizer)) | (
            (normalizer == 0) if signed else (normalizer <= 0)
        )
        invalid_row = (
            jnp.any(causal & (invalid_score | invalid_kernel), axis=-1, keepdims=True)
            | invalid_normalizer
        )
        weights = kernel / normalizer
        output = jnp.einsum(
            "bhqk,bkhd->bqhd", weights, v.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        output = jnp.where(jnp.swapaxes(invalid_row, 1, 2), jnp.nan, output)
        output = output.astype(v.dtype)
        if signed:
            output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
        if not return_diagnostics:
            return None, output
        diagnostics = {
            "score_min": jnp.min(jnp.where(measured, scores, jnp.inf)),
            "score_max": jnp.max(jnp.where(measured, scores, -jnp.inf)),
            "kernel_min": jnp.min(jnp.where(measured, kernel, jnp.inf)),
            "kernel_max": jnp.max(jnp.where(measured, kernel, -jnp.inf)),
            "normalizer_min": jnp.min(jnp.where(actual_rows[None, None, :, None], normalizer, jnp.inf)),
            "normalizer_max": jnp.max(jnp.where(actual_rows[None, None, :, None], normalizer, -jnp.inf)),
            "invalid_score_count": jnp.sum(measured & invalid_score, dtype=jnp.int32),
            "invalid_kernel_count": jnp.sum(measured & invalid_kernel, dtype=jnp.int32),
            "invalid_normalizer_count": jnp.sum(actual_rows[None, None, :, None] & invalid_normalizer, dtype=jnp.int32),
        }
        if extended_diagnostics:
            actual = actual_rows[None, None, :, None]
            cancellation = jnp.sum(jnp.abs(kernel), axis=-1, keepdims=True) / jnp.abs(normalizer)
            diagnostics.update({
                "min_abs_normalizer": jnp.min(jnp.where(actual, jnp.abs(normalizer), jnp.inf)),
                "negative_weight_count": jnp.sum(measured & (weights < 0), dtype=jnp.int32),
                "negative_kernel_count": jnp.sum(measured & (kernel < 0), dtype=jnp.int32),
                "causal_entry_count": jnp.sum(measured, dtype=jnp.int32) * batch * heads,
                "cancellation_ratio_max": jnp.max(jnp.where(actual, cancellation, -jnp.inf)),
                "cancellation_sum": jnp.sum(jnp.where(actual, cancellation, 0.0)),
                "row_count": jnp.sum(actual_rows, dtype=jnp.int32) * batch * heads,
                "invalid_output_count": jnp.sum(actual_rows[None, :, None, None] & ~jnp.isfinite(output), dtype=jnp.int32),
            })
        return None, (output, jax.lax.stop_gradient(diagnostics))

    _, result = jax.lax.scan(
        jax.checkpoint(attend_block, prevent_cse=False), None, (queries, starts),
    )
    if return_diagnostics:
        output_blocks, per_block = result
        diagnostics = {
            name: (jnp.min(values) if name.endswith("_min") or name == "min_abs_normalizer" else
                   jnp.max(values) if name.endswith("_max") else jnp.sum(values))
            for name, values in per_block.items()
        }
        if extended_diagnostics:
            diagnostics["negative_weight_fraction"] = diagnostics["negative_weight_count"] / diagnostics["causal_entry_count"]
            diagnostics["negative_kernel_fraction"] = diagnostics["negative_kernel_count"] / diagnostics["causal_entry_count"]
            diagnostics["cancellation_ratio_mean"] = diagnostics.pop("cancellation_sum") / diagnostics["row_count"]
    else:
        output_blocks = result
    output = jnp.swapaxes(output_blocks, 0, 1).reshape(batch, block_count * block_size, heads, v.shape[-1])
    output = output[:, :time]
    return (output, diagnostics) if return_diagnostics else output


def padded_dot_product_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, backend: AttentionBackend = "xla",
) -> jax.Array:
    """Original public SDPA path retained solely for baseline parity checks."""
    qk_dim, value_dim = q.shape[-1], v.shape[-1]
    if k.shape[-1] != qk_dim:
        raise ValueError("Query and key channel dimensions must match")
    if backend not in ("auto", "xla", "cudnn"):
        raise ValueError(f"Unknown fused attention backend: {backend}")
    width = ((max(qk_dim, value_dim) + 7) // 8) * 8

    def pad(array):
        return jnp.pad(array, ((0, 0), (0, 0), (0, 0), (0, width - array.shape[-1])))

    y = jax.nn.dot_product_attention(
        pad(q), pad(k), pad(v), scale=1.0 / math.sqrt(qk_dim),
        is_causal=True, implementation=None if backend == "auto" else backend,
    )
    return y[..., :value_dim]


class MultiHeadAttention(nnx.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        qk_head_dim: int, value_head_dim: int, rope_theta: float = 10000.0,
        dropout: float = 0.0, backend: AttentionBackend = "blocked",
        attention_degree: int | None = None, quadratic_mode: str = "fixed", *, rngs: nnx.Rngs,
    ):
        _validate_degree(attention_degree)
        if quadratic_mode not in ("fixed", "positive", "unrestricted"):
            raise ValueError(f"Unknown quadratic mode: {quadratic_mode}")
        if quadratic_mode != "fixed" and attention_degree != 2:
            raise ValueError("Learned quadratic attention requires degree=2")
        if min(hidden_size, num_heads, num_kv_heads, qk_head_dim, value_head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive")
        if num_heads != num_kv_heads:
            raise ValueError("This experiment requires full MHA (num_heads == num_kv_heads)")
        if qk_head_dim % 2:
            raise ValueError("qk_head_dim must be even for RoPE")
        if backend not in ("blocked", "fused", "manual", "auto", "xla", "cudnn"):
            raise ValueError(f"Unknown attention backend: {backend}")
        if backend not in ("blocked", "fused") and attention_degree is not None:
            raise ValueError("Taylor attention requires backend='blocked' or 'fused'")
        if backend != "manual" and dropout != 0.0:
            raise ValueError("Blocked/fused attention requires dropout=0")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.rope_theta = rope_theta
        self.backend = backend
        self.attention_degree = attention_degree
        self.quadratic_mode = quadratic_mode
        self.dtype = jnp.float32
        # Preserve the reference constructor order/default initialization exactly.
        self.q_proj = nnx.Linear(hidden_size, num_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, num_kv_heads * qk_head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, num_kv_heads * value_head_dim, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout)
        self.o_proj = nnx.Linear(num_heads * value_head_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(
        self, x: jax.Array, *, deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None, return_diagnostics: bool = False,
        coefficients: jax.Array | None = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        batch, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.backend != "manual":
            q, k, v = (operand.astype(self.dtype) for operand in (q, k, v))
        q = q.reshape(batch, seq_len, self.num_heads, self.qk_head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.qk_head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.value_head_dim)
        positions = jnp.arange(seq_len)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        if self.backend != "manual":
            q, k = q.astype(self.dtype), k.astype(self.dtype)
        if self.backend == "blocked":
            result = blocked_dot_product_attention(
                q, k, v, attention_degree=self.attention_degree,
                coefficients=coefficients, quadratic_mode=self.quadratic_mode,
                return_diagnostics=return_diagnostics,
            )
            y, diagnostics = result if return_diagnostics else (result, None)
        elif self.backend == "fused":
            if self.quadratic_mode != "fixed":
                from experiments.polynomial_attention.models.components.fused_quadratic_attention import fused_quadratic_attention
                y = fused_quadratic_attention(
                    q, k, v, jnp.broadcast_to(coefficients, (self.num_heads, 2)),
                    quadratic_mode=self.quadratic_mode,
                )
            elif self.attention_degree is None:
                y = padded_dot_product_attention(q, k, v, backend="cudnn")
            else:
                from experiments.polynomial_attention.models.components.fused_attention import fused_polynomial_attention
                y = fused_polynomial_attention(q, k, v, attention_degree=self.attention_degree)
            diagnostics = None
            if return_diagnostics:
                # Collect reference score/kernel statistics on the actual layer
                # inputs while preserving the fused output for following layers.
                _, diagnostics = blocked_dot_product_attention(
                    q, k, v, attention_degree=self.attention_degree,
                    coefficients=coefficients, quadratic_mode=self.quadratic_mode,
                    return_diagnostics=True,
                )
        elif return_diagnostics:
            raise ValueError("Diagnostics require backend='blocked' or 'fused'")
        elif self.backend != "manual":
            y = padded_dot_product_attention(q, k, v, backend=self.backend)
        else:
            scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.qk_head_dim)
            causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(causal[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
            weights = nnx.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
            weights = self.dropout(weights, deterministic=deterministic, rngs=rngs)
            y = jnp.einsum("bhts,bshd->bthd", weights, v)
        output = self.o_proj(y.reshape(batch, seq_len, self.num_heads * self.value_head_dim))
        return (output, diagnostics) if return_diagnostics else output
