"""Differentiable exact polynomial GDN without an expanded feature-space state.

F_t(q) = exp(g_t) F_{t-1}((I-erase_t k_t k_t^T)q)
         + v_t sum_{r=0}^m beta_{t,r} (q^T k_t)^r, m in {2, 3}.

The public read is scale*F_t(q_t). Its erase geometry is in the original key
space; decay multiplies every degree equally. beta may be signed. FP32 chunk
algebra and TF32x3 products follow the repaired linear GDN backend. Saved
residuals remain linear in sequence length. Backward-only, packed triangular
query/adjoint workspaces are reused between layers, avoiding quadratic states.
"""

from functools import partial
import os

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends.fla_gated_delta_rule import (
    CHUNK_SIZE, _call, _kernels, _shape,
)


def _default_num_warps():
    """Avoid the unqualified Blackwell tensor-memory path in recurrent dots."""
    device = jax.local_devices()[0]
    if device.platform not in ("gpu", "cuda"):
        return 4
    capability = getattr(device, "compute_capability", "")
    capability = capability() if callable(capability) else capability
    return 2 if str(capability).split(".")[0] in ("10", "12") else 4


def resolve_polynomial_gdn_options(precision=None, num_warps=None):
    """Resolve static runtime options; explicit arguments override the environment.

    ``bf16_read`` rounds only value-reading products to BF16; ``bf16_values``
    additionally rounds the score-adjoint product. Both retain FP32 accumulation
    and all transport/inverse geometry. They are experimental precision modes.
    Options are resolved when the public operation is traced, not on each token.
    Blackwell defaults to two warps and rejects four/eight: four-warps backward
    failed device qualification, while two warps avoids its tensor-memory path.
    """
    if precision is None:
        precision = os.environ.get("POLY_GDN_VALUE_PRECISION", "tf32x3")
    if precision not in ("tf32x3", "bf16_read", "bf16_values"):
        raise ValueError(f"Unsupported polynomial GDN precision: {precision!r}")
    default_warps = _default_num_warps()
    if num_warps is None:
        raw_warps = os.environ.get("POLY_GDN_NUM_WARPS", str(default_warps))
        try:
            num_warps = int(raw_warps)
        except ValueError as error:
            raise ValueError(f"Invalid POLY_GDN_NUM_WARPS: {raw_warps!r}") from error
    if type(num_warps) is not int or num_warps not in (1, 2, 4, 8):
        raise ValueError("Polynomial GDN num_warps must be one of 1, 2, 4, 8")
    if default_warps == 2 and num_warps > 2:
        raise ValueError("Polynomial GDN on Blackwell requires num_warps=1 or 2; "
                         "four/eight-warps backward is not qualified")
    return precision, num_warps


def resolve_polynomial_gdn_min_ctas(min_ctas_per_sm=None):
    """Resolve experimental launch guidance; zero leaves PTX unchanged.

    An omitted value reads ``POLY_GDN_MIN_CTAS_PER_SM`` (default zero).
    Explicit zero disables guidance even when the environment enables it.
    The resolved integer is static through the forward and backward passes.
    """
    if min_ctas_per_sm is None:
        raw = os.environ.get("POLY_GDN_MIN_CTAS_PER_SM", "0")
        try:
            min_ctas_per_sm = int(raw)
        except ValueError as error:
            raise ValueError(f"Invalid POLY_GDN_MIN_CTAS_PER_SM: {raw!r}") from error
    if type(min_ctas_per_sm) is not int or min_ctas_per_sm not in (0, 1, 2):
        raise ValueError("Polynomial GDN min_ctas_per_sm must be one of 0, 1, 2")
    return min_ctas_per_sm


def _chunk_view(x):
    batch, time, heads, width = x.shape
    return x.reshape(batch, time // CHUNK_SIZE, CHUNK_SIZE, heads, width).transpose(0, 3, 1, 2, 4)


def _token_view(x):
    batch, heads, chunks, chunk, width = x.shape
    return x.transpose(0, 2, 3, 1, 4).reshape(batch, chunks * chunk, heads, width)


def _inverse_forward(k, erase):
    batch, length, heads, key_dim = k.shape
    return _call(_kernels()[0]["kkt_solve"], k.astype(jnp.float32), jnp.zeros_like(erase, dtype=jnp.float32),
                 erase.astype(jnp.float32), length,
                 out=_shape((batch, length, heads, CHUNK_SIZE), jnp.float32),
                 grid=(length // CHUNK_SIZE, batch * heads), name="polynomial_kkt_solve", zeroed=(0,),
                 H=heads, HV=heads, K=key_dim, BT=CHUNK_SIZE, BC=16)


@jax.custom_vjp
def _inverse(k, erase):
    return _inverse_forward(k, erase)


def _inverse_fwd(k, erase):
    inverse = _inverse_forward(k, erase)
    return inverse, (k, erase, inverse)


def _inverse_bwd(residuals, dinverse):
    k, erase, inverse = residuals
    keys = _chunk_view(k.astype(jnp.float32))
    a = _chunk_view(inverse)
    da = _chunk_view(dinverse)
    at = jnp.swapaxes(a, -1, -2)
    with jax.default_matmul_precision("highest"):
        dlower = -at @ da @ at
        lower = jnp.arange(CHUNK_SIZE)[:, None] > jnp.arange(CHUNK_SIZE)[None, :]
        dlower = jnp.where(lower, dlower, 0.0)
        gram = keys @ jnp.swapaxes(keys, -1, -2)
        derase = jnp.sum(dlower * gram, axis=-1, keepdims=True)
        erasure = _chunk_view(erase.astype(jnp.float32)[..., None])
        dgram = dlower * erasure
        dk = (dgram + jnp.swapaxes(dgram, -1, -2)) @ keys
    return _token_view(dk).astype(k.dtype), _token_view(derase)[..., 0].astype(erase.dtype)


_inverse.defvjp(_inverse_fwd, _inverse_bwd)


def _metadata(q, v):
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    return dict(T=length, H=heads, K=key_dim, V=value_dim, BT=CHUNK_SIZE,
                BK=max(16, 1 << (key_dim - 1).bit_length()),
                BV=max(16, 1 << (value_dim - 1).bit_length()))


def _polynomial_call(kernel, *args, out, q, v, name,
                     precision="tf32x3", num_warps=None, min_ctas_per_sm=None, **extra):
    precision, num_warps = resolve_polynomial_gdn_options(precision, num_warps)
    min_ctas_per_sm = resolve_polynomial_gdn_min_ctas(min_ctas_per_sm)
    from scaffolds.transformer.attention.backends.ejkernel_bridge import triton_call

    return triton_call(*args, kernel=kernel, out_shape=out,
                       grid=(q.shape[1] // CHUNK_SIZE, q.shape[0] * q.shape[2]),
                       name=f"polynomial_gdn::{name}", default_dot_input_precision="tf32x3",
                       num_warps=num_warps, num_stages=1,
                       min_ctas_per_sm=min_ctas_per_sm or None,
                       VALUE_PRECISION=precision, **_metadata(q, v), **extra)


def _forward(q, k, v, prefix, beta, erase, inverse, transformed, scale,
             precision="tf32x3", num_warps=None, min_ctas_per_sm=None):
    from scaffolds.transformer.attention.backends.polynomial_gdn_kernels import forward

    output, _ = _polynomial_call(
        forward, q, k, v, prefix, beta, erase, inverse, transformed, float(scale),
        out=[_shape(v.shape, v.dtype), _shape((), jnp.float32)], q=q, v=v, name="forward",
        SAVE_HISTORY=False, COMPUTE_OUTPUT=True, NBETA=beta.shape[-1],
        precision=precision, num_warps=num_warps, min_ctas_per_sm=min_ctas_per_sm)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11))
def _attention(q, k, v, prefix, beta, erase, inverse, transformed, scale, precision, num_warps,
               min_ctas_per_sm):
    return _forward(q, k, v, prefix, beta, erase, inverse, transformed, scale,
                    precision, num_warps, min_ctas_per_sm)


def _attention_fwd(q, k, v, prefix, beta, erase, inverse, transformed, scale, precision, num_warps,
                   min_ctas_per_sm):
    return _forward(q, k, v, prefix, beta, erase, inverse, transformed, scale,
                    precision, num_warps, min_ctas_per_sm), (
        q, k, v, prefix, beta, erase, inverse, transformed)


def _attention_bwd(scale, precision, num_warps, min_ctas_per_sm, residuals, dout):
    from scaffolds.transformer.attention.backends.polynomial_gdn_kernels import (
        forward, backward_queries, backward_keys,
    )

    q, k, v, prefix, beta, erase, inverse, transformed = residuals
    options = dict(precision=precision, num_warps=num_warps, min_ctas_per_sm=min_ctas_per_sm)
    batch, length, heads, key_dim = q.shape
    chunks = length // CHUNK_SIZE
    history_shape = _shape((batch, heads, chunks * (chunks + 1) // 2, CHUNK_SIZE, key_dim), jnp.float32)
    _, history = _polynomial_call(
        # The history-only kernel does not read its value operand. Supplying
        # dout makes its scheduling depend on this layer's backward pass, so
        # XLA cannot hoist all layers' quadratic workspaces into the forward.
        forward, q, k, dout, prefix, beta, erase, inverse, transformed, float(scale),
        out=[_shape((), jnp.float32), history_shape], q=q, v=v, name="recompute_queries",
        SAVE_HISTORY=True, COMPUTE_OUTPUT=False, NBETA=beta.shape[-1], **options)
    dq, dquery_prefix, adjoint_history = _polynomial_call(
        backward_queries, k, v, prefix, beta, erase, inverse, transformed, dout, history, float(scale),
        out=[_shape(q.shape, q.dtype), _shape(prefix.shape, jnp.float32), history_shape],
        q=q, v=v, name="backward_queries", NBETA=beta.shape[-1], **options)
    dummy = _shape((), jnp.float32)
    _, dv, dbeta, _, _, _, dkey_prefix = _polynomial_call(
        backward_keys, k, v, prefix, beta, erase, inverse, transformed, dout, history, adjoint_history, float(scale),
        out=[dummy, _shape(v.shape, v.dtype), _shape(beta.shape, beta.dtype), dummy, dummy, dummy,
             _shape(prefix.shape, jnp.float32)], q=q, v=v, name="backward_key_values", GROUP=1, BQ=64, MODE=0,
        NBETA=beta.shape[-1], **options)
    dk_diagonal, _, _, derase_diagonal, dinverse, _, _ = _polynomial_call(
        backward_keys, k, v, prefix, beta, erase, inverse, transformed, dout, history, adjoint_history, float(scale),
        out=[_shape(k.shape, jnp.float32), dummy, dummy, _shape(erase.shape, erase.dtype),
             _shape(inverse.shape, jnp.float32), dummy, dummy],
        q=q, v=v, name="backward_key_geometry_diagonal", GROUP=2, BQ=64, MODE=1,
        NBETA=beta.shape[-1], **options)
    dk, _, _, derase, _, dtransformed, _ = _polynomial_call(
        backward_keys, k, v, prefix, beta, erase, inverse, transformed, dout, history, adjoint_history, float(scale),
        out=[_shape(k.shape, jnp.float32), dummy, dummy, _shape(erase.shape, erase.dtype),
             dummy, _shape(transformed.shape, jnp.float32), dummy],
        q=q, v=v, name="backward_key_geometry_cross", GROUP=2, BQ=64, MODE=2,
        NBETA=beta.shape[-1], **options)
    dk, derase = dk + dk_diagonal, derase + derase_diagonal
    return dq, dk.astype(k.dtype), dv, dquery_prefix + dkey_prefix, dbeta, derase, dinverse, dtransformed


_attention.defvjp(_attention_fwd, _attention_bwd)


def polynomial_gated_delta_rule(q, k, v, g, beta, erase, scale, *, precision=None, num_warps=None,
                                min_ctas_per_sm=None):
    """Degree-two/three GDN read: q/k[B,T,H,K], v[B,T,H,V], beta[B,T,H,m+1].

    g and erase have shape [B,T,H]. Inputs q/k/v may be BF16 or FP32;
    g, beta, and erase use FP32 arithmetic. q/k normalization belongs to the
    model. This GPU backend supports lengths divisible by 64 and widths <=64.

    The default is FP32/TF32x3 with two warps on Blackwell and four elsewhere.
    Optional settings resolve through :func:`resolve_polynomial_gdn_options`.
    Minimum-CTA guidance defaults to disabled; see
    :func:`resolve_polynomial_gdn_min_ctas`. It affects only the polynomial
    kernels, leaving the inverse preparation and its differentiation unchanged.
    """
    precision, num_warps = resolve_polynomial_gdn_options(precision, num_warps)
    min_ctas_per_sm = resolve_polynomial_gdn_min_ctas(min_ctas_per_sm)
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Polynomial GDN expects matching q/k/v[B,T,H,D] dimensions")
    if (beta.ndim != 4 or beta.shape[:3] != q.shape[:3] or beta.shape[-1] not in (3, 4)
            or g.shape != q.shape[:3] or erase.shape != g.shape):
        raise ValueError("Polynomial GDN expects beta[B,T,H,3 or 4] and g/erase[B,T,H]")
    if q.shape[1] % CHUNK_SIZE or not 0 < q.shape[-1] <= 64 or not 0 < v.shape[-1] <= 64:
        raise ValueError("Polynomial GDN fused backend needs T divisible by 64 and 1<=K,V<=64")
    # Keep the two key gradient paths in FP32 until after their sum. Saving k32
    # costs linear memory and avoids BF16 rounding of cancellation-prone terms.
    k32 = k.astype(jnp.float32)
    erase32, beta32 = erase.astype(jnp.float32), beta.astype(jnp.float32)
    inverse = _inverse(k32, erase32)
    # Outside the query's own chunk, E=Q K^T A=Q R^T with R=A^T K.
    # Prepare R once per key chunk instead of multiplying by A for every query
    # chunk; autodiff adds its local key/inverse adjoints in FP32.
    with jax.default_matmul_precision("highest"):
        transformed = _token_view(jnp.swapaxes(_chunk_view(inverse), -1, -2) @ _chunk_view(k32))
    prefix = jnp.cumsum(g.astype(jnp.float32), axis=1)
    return _attention(q, k32, v, prefix, beta32, erase32, inverse, transformed,
                      float(scale), precision, num_warps, min_ctas_per_sm)
