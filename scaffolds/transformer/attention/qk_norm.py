"""Q/K normalization outside the attention kernels.

The historical ``cosine_log_scale`` mode scores ``s_ij = exp(gamma) cos(q_i, k_j)``,
so ``|s_ij| <= exp(gamma)`` and the kernel sees a bounded argument. The mechanisms
keep their ``QK^T / sqrt(d_K)`` arithmetic and fused kernels: the attention module
feeds them ``q' = exp(gamma) sqrt(d_K) q / ||q||`` and ``k' = k / ||k||``. RoPE is
orthogonal, so its scalar normalization commutes with RoPE. ``rms_norm`` instead
uses independent learned Q/K channel gains, shared across heads in each layer.
Those gains apply before RoPE and do not in general commute with it. In
``rms_norm_cosine``, the final rotated queries and keys are then unit-normalized
without a learned temperature. All modes preserve the kernels' ordinary
``QK^T / sqrt(d_K)`` arithmetic.
RMSNorm may additionally multiply each head's queries by its own positive scalar
``exp(gamma_h)``, initially one, preserving the existing channel gains. The optional
per-token gate multiplies that scalar by ``exp(a_h cos(w_h, x_t))``: a bounded log
scale read from the attention input, which gives the kernel back the query-norm
freedom that RMSNorm removes; ``a_h`` starts at zero.
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


def rms_parameter_specs(*, qk_head_dim: int):
    """Independent per-layer Q/K channel gains, shared across heads, without decay."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    if not isinstance(qk_head_dim, int) or isinstance(qk_head_dim, bool) or qk_head_dim <= 0:
        raise ValueError("QK RMSNorm requires a positive integer head dimension")
    return tuple(AttentionParameterSpec(
        name=name, argument=name, initial=(1.0,) * qk_head_dim,
        sharing="per_layer", weight_decay=False,
    ) for name in ("qk_norm_q_gain", "qk_norm_k_gain"))


def rms_head_scale_specs():
    """An independent positive score multiplier per layer/head, initially one."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (AttentionParameterSpec(
        name="qk_norm_head_log_scale", argument="qk_norm_head_log_scale",
        initial=(0.0,), sharing="per_head", weight_decay=False,
    ),)


TOKEN_SCALE_PARAMETERS = ("qk_norm_token_scale_kernel", "qk_norm_token_scale_amplitude")


def token_scale_specs(*, hidden_size: int, kernel_weight_decay: bool = False):
    """The per-token gate's direction w_h, drawn N(0, 1/hidden_size), and amplitude a_h, initially zero."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
        raise ValueError("The per-token query scale requires a positive integer hidden size")
    if not isinstance(kernel_weight_decay, bool):
        raise ValueError("Token-scale projection weight decay must be a boolean")
    return (
        AttentionParameterSpec(
            name="qk_norm_token_scale_kernel", argument="qk_norm_token_scale_kernel",
            initial=(0.0,) * hidden_size, initial_shape=(hidden_size,), sharing="per_head",
            weight_decay=kernel_weight_decay, initial_std=1.0 / math.sqrt(hidden_size),
        ),
        AttentionParameterSpec(
            name="qk_norm_token_scale_amplitude", argument="qk_norm_token_scale_amplitude",
            initial=(0.0,), sharing="per_head", weight_decay=False,
        ),
    )


def token_log_scale(x: jax.Array, kernel: jax.Array, amplitude: jax.Array, *, eps: float) -> jax.Array:
    """FP32 delta [batch,time,heads] = a_h cos(w_h, x_t) from x [B,T,D], kernel [H,D], amplitude [H,1].

    Both vectors are normalized as ``v / sqrt(||v||^2 + eps)``, so ``|delta| <= |a_h|``
    without saturation, and ``a_h = 0`` leaves every query scale exactly at exp(gamma_h).
    """
    if (x.ndim != 3 or kernel.ndim != 2 or kernel.shape[1] != x.shape[-1]
            or amplitude.shape != (kernel.shape[0], 1)):
        raise ValueError("The per-token query scale requires x [B,T,D], kernel [H,D] and amplitude [H,1]")
    cosine = jnp.einsum("btd,hd->bth", unit_normalize(x.astype(jnp.float32), eps),
                        unit_normalize(kernel.astype(jnp.float32), eps),
                        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return amplitude[:, 0].astype(jnp.float32) * cosine


def scale_rms_queries(q: jax.Array, head_log_scale: jax.Array,
                      token_log_scale: jax.Array | None = None) -> jax.Array:
    """Multiply RMS-normalized queries by exp(gamma_h), or exp(gamma_h + delta_th), without renormalizing them.

    q is [batch,time,heads,channels], after its existing channel gains. Scaling
    queries alone multiplies the resulting QK score once by the query's scale.
    """
    if q.ndim != 4 or head_log_scale.shape != (q.shape[2], 1):
        raise ValueError("QK RMS head scaling requires queries [B,T,H,D] and log scales [H,1]")
    log_scale = head_log_scale.astype(jnp.float32).reshape(1, 1, q.shape[2], 1)
    if token_log_scale is not None:
        if token_log_scale.shape != q.shape[:3]:
            raise ValueError("A per-token log scale must have shape [B,T,H]")
        log_scale = log_scale + token_log_scale.astype(jnp.float32)[..., None]
    return (q.astype(jnp.float32) * jnp.exp(log_scale)).astype(q.dtype)


def rms_score_operands(q: jax.Array, k: jax.Array, q_gain: jax.Array, k_gain: jax.Array, *, eps: float):
    """Normalize each token/head over channels in FP32, then apply its layer's gains.

    Q/K have shape ``[batch, time, heads, channels]``. Each gain has shape
    ``[channels]`` and broadcasts over tokens and heads; statistics never do.
    Return the input dtype after the FP32 normalization and gain multiplication.
    """
    if q.ndim != 4 or q.shape != k.shape:
        raise ValueError("Q and K must share the shape [batch, time, heads, channels]")
    if q_gain.shape != (q.shape[-1],) or k_gain.shape != (k.shape[-1],):
        raise ValueError("QK RMSNorm gains must each have shape (channels,)")
    if not _positive_finite(eps):
        raise ValueError("QK RMSNorm requires a finite positive eps")

    def normalize(values, gain):
        fp32 = values.astype(jnp.float32)
        normalized = fp32 * jax.lax.rsqrt(jnp.mean(jnp.square(fp32), axis=-1, keepdims=True) + eps)
        return (normalized * gain.astype(jnp.float32)).astype(values.dtype)

    return normalize(q, q_gain), normalize(k, k_gain)


def unit_normalize(x: jax.Array, eps: float) -> jax.Array:
    """``x / sqrt(||x||^2 + eps)`` over the channel axis in FP32, returned in x's dtype."""
    if not _positive_finite(eps):
        raise ValueError("QK-norm requires a finite positive eps")
    values = x.astype(jnp.float32)
    normalized = values * jax.lax.rsqrt(jnp.sum(jnp.square(values), axis=-1, keepdims=True) + eps)
    return normalized.astype(x.dtype)


def unit_cosine_score_operands(q: jax.Array, k: jax.Array, *, eps: float):
    """Unit-normalize the final Q/K vectors, compensating only the backend's fixed 1/sqrt(d_K)."""
    def direction(x):
        fp32 = x.astype(jnp.float32)
        squared_norm = jnp.sum(jnp.square(fp32), axis=-1, keepdims=True)
        return fp32 * jax.lax.rsqrt(jnp.maximum(squared_norm, eps * eps))

    return (direction(q) * math.sqrt(q.shape[-1])).astype(q.dtype), direction(k).astype(k.dtype)


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
