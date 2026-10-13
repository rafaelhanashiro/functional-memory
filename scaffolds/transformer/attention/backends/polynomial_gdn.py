"""Differentiable exact quadratic GDN without an expanded feature-space state.

F_t(q) = exp(g_t) F_{t-1}((I-erase_t k_t k_t^T)q)
         + v_t sum_{r=0}^2 beta_{t,r} (q^T k_t)^r.

The public read is scale*F_t(q_t). Its erase geometry is in the original key
space; decay multiplies every degree equally. beta may be signed. FP32 chunk
algebra and TF32x3 products follow the repaired linear GDN backend. Saved
residuals remain linear in sequence length. Backward-only, packed triangular
query/adjoint workspaces are reused between layers, avoiding quadratic states.
"""

from functools import partial

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends.fla_gated_delta_rule import (
    CHUNK_SIZE, _call, _kernels, _shape,
)


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


def _polynomial_call(kernel, *args, out, q, v, name, **extra):
    from scaffolds.transformer.attention.backends.ejkernel_bridge import triton_call

    return triton_call(*args, kernel=kernel, out_shape=out,
                       grid=(q.shape[1] // CHUNK_SIZE, q.shape[0] * q.shape[2]),
                       name=f"polynomial_gdn::{name}", default_dot_input_precision="tf32x3",
                       num_warps=4, num_stages=1, **_metadata(q, v), **extra)


def _forward(q, k, v, prefix, beta, erase, inverse, scale):
    from scaffolds.transformer.attention.backends.polynomial_gdn_kernels import forward

    output, _ = _polynomial_call(
        forward, q, k, v, prefix, beta, erase, inverse, float(scale),
        out=[_shape(v.shape, v.dtype), _shape((), jnp.float32)], q=q, v=v, name="forward",
        SAVE_HISTORY=False, COMPUTE_OUTPUT=True)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _attention(q, k, v, prefix, beta, erase, inverse, scale):
    return _forward(q, k, v, prefix, beta, erase, inverse, scale)


def _attention_fwd(q, k, v, prefix, beta, erase, inverse, scale):
    return _forward(q, k, v, prefix, beta, erase, inverse, scale), (q, k, v, prefix, beta, erase, inverse)


def _attention_bwd(scale, residuals, dout):
    from scaffolds.transformer.attention.backends.polynomial_gdn_kernels import (
        forward, backward_queries, backward_keys,
    )

    q, k, v, prefix, beta, erase, inverse = residuals
    batch, length, heads, key_dim = q.shape
    chunks = length // CHUNK_SIZE
    history_shape = _shape((batch, heads, chunks * (chunks + 1) // 2, CHUNK_SIZE, key_dim), jnp.float32)
    _, history = _polynomial_call(
        # The history-only kernel does not read its value operand. Supplying
        # dout makes its scheduling depend on this layer's backward pass, so
        # XLA cannot hoist all layers' quadratic workspaces into the forward.
        forward, q, k, dout, prefix, beta, erase, inverse, float(scale),
        out=[_shape((), jnp.float32), history_shape], q=q, v=v, name="recompute_queries",
        SAVE_HISTORY=True, COMPUTE_OUTPUT=False)
    dq, dquery_prefix, adjoint_history = _polynomial_call(
        backward_queries, k, v, prefix, beta, erase, inverse, dout, history, float(scale),
        out=[_shape(q.shape, q.dtype), _shape(prefix.shape, jnp.float32), history_shape],
        q=q, v=v, name="backward_queries")
    dummy = _shape((), jnp.float32)
    _, dv, dbeta, _, _, dkey_prefix = _polynomial_call(
        backward_keys, k, v, prefix, beta, erase, inverse, dout, history, adjoint_history, float(scale),
        out=[dummy, _shape(v.shape, v.dtype), _shape(beta.shape, beta.dtype), dummy, dummy,
             _shape(prefix.shape, jnp.float32)], q=q, v=v, name="backward_key_values", GROUP=1)
    dk, _, _, derase, dinverse, _ = _polynomial_call(
        backward_keys, k, v, prefix, beta, erase, inverse, dout, history, adjoint_history, float(scale),
        out=[_shape(k.shape, jnp.float32), dummy, dummy, _shape(erase.shape, erase.dtype),
             _shape(inverse.shape, jnp.float32), dummy], q=q, v=v, name="backward_key_geometry", GROUP=2)
    return dq, dk.astype(k.dtype), dv, dquery_prefix + dkey_prefix, dbeta, derase, dinverse


_attention.defvjp(_attention_fwd, _attention_bwd)


def polynomial_gated_delta_rule(q, k, v, g, beta, erase, scale):
    """Quadratic GDN read, with q/k[B,T,H,K], v[B,T,H,V], beta[B,T,H,3].

    g and erase have shape [B,T,H]. Inputs q/k/v may be BF16 or FP32;
    g, beta, and erase use FP32 arithmetic. q/k normalization belongs to the
    model. This GPU backend supports lengths divisible by 64 and widths <=64.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Polynomial GDN expects matching q/k/v[B,T,H,D] dimensions")
    if beta.shape != (*q.shape[:3], 3) or g.shape != q.shape[:3] or erase.shape != g.shape:
        raise ValueError("Polynomial GDN expects beta[B,T,H,3] and g/erase[B,T,H]")
    if q.shape[1] % CHUNK_SIZE or not 0 < q.shape[-1] <= 64 or not 0 < v.shape[-1] <= 64:
        raise ValueError("Polynomial GDN fused backend needs T divisible by 64 and 1<=K,V<=64")
    # Keep the two key gradient paths in FP32 until after their sum. Saving k32
    # costs linear memory and avoids BF16 rounding of cancellation-prone terms.
    k32 = k.astype(jnp.float32)
    erase32, beta32 = erase.astype(jnp.float32), beta.astype(jnp.float32)
    inverse = _inverse(k32, erase32)
    prefix = jnp.cumsum(g.astype(jnp.float32), axis=1)
    return _attention(q, k32, v, prefix, beta32, erase32, inverse, float(scale))
