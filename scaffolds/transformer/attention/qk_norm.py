"""Cosine attention scores: unit-normalized Q/K with a learned log scale.

With ``qk_norm`` every dot-product mechanism scores ``s_ij = exp(gamma) cos(q_i, k_j)``,
so ``|s_ij| <= exp(gamma)`` and the kernel sees a bounded argument. The mechanisms
keep their ``QK^T / sqrt(d_K)`` arithmetic and fused kernels: the attention module
feeds them ``q' = exp(gamma) sqrt(d_K) q / ||q||`` and ``k' = k / ||k||``. RoPE is
orthogonal, so normalizing before it equals normalizing after it.
"""

import math

import jax
import jax.numpy as jnp


def _positive_finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def initial_scale(model_config) -> float:
    """``qk_norm_initial_scale``, or sqrt(d_K) when null: RMSNorm QK-norm's scale at initialization."""
    value = model_config.qk_norm_initial_scale
    return math.sqrt(model_config.qk_head_dim) if value is None else float(value)


def parameter_specs(*, initial_scale: float, sharing: str):
    """The log scale gamma, alpha = exp(gamma), initialized at log(alpha_0), without weight decay."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    if not _positive_finite(initial_scale):
        raise ValueError("QK-norm requires a finite positive initial scale")
    return (AttentionParameterSpec(
        name="qk_norm_log_scale", argument="qk_norm_log_scale",
        initial=(math.log(initial_scale),), sharing=sharing, weight_decay=False,
    ),)


def unit_normalize(x: jax.Array, eps: float) -> jax.Array:
    """``x / sqrt(||x||^2 + eps)`` over the channel axis in FP32, returned in x's dtype."""
    if not _positive_finite(eps):
        raise ValueError("QK-norm requires a finite positive eps")
    values = x.astype(jnp.float32)
    normalized = values * jax.lax.rsqrt(jnp.sum(jnp.square(values), axis=-1, keepdims=True) + eps)
    return normalized.astype(x.dtype)


def cosine_score_operands(q: jax.Array, k: jax.Array, log_scale: jax.Array, *, eps: float):
    """Return ``(q', k')`` whose mechanism score ``q'·k' / sqrt(d_K)`` is ``exp(gamma) cos(q, k)``.

    Inputs are ``[batch, time, heads, channels]``; ``log_scale`` is ``[heads, 1]`` (per
    head) or ``[1]`` (shared). The scale multiplies the normalized queries in FP32;
    keys are only normalized. Both results keep their input dtype.
    """
    if q.ndim != 4 or q.shape != k.shape:
        raise ValueError("Q and K must share the shape [batch, time, heads, channels]")
    heads, qk_dim = q.shape[2], q.shape[3]
    if log_scale.shape not in ((1,), (heads, 1)):
        raise ValueError("QK-norm log scale must have shape (1,) or (heads, 1)")
    scale = jnp.exp(log_scale.astype(jnp.float32)) * math.sqrt(qk_dim)
    scaled = unit_normalize(q, eps).astype(jnp.float32) * scale.reshape(1, 1, -1, 1)
    return scaled.astype(q.dtype), unit_normalize(k, eps)
