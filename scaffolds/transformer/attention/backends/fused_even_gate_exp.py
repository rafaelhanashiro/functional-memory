"""Causal degree-16 even powers plus exp with nine query gates and one common numerical shift.

The query-tile program accumulates V online. Its companion recomputes shifted basis tiles to return
dQ and all nine gate derivatives. A key-tile program accumulates dK/dV. No pass stores pair weights.
"""
from functools import partial
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from scaffolds.transformer.attention.backends.fused_unnormalized import (
    NUM_STAGES, NUM_WARPS, _dot, _full_shift_spec, _full_spec, _load, _shift_spec, _spec, _tile_sizes,
)
from scaffolds.transformer.attention.even_gate_exp import BASIS_COUNT, row_shift, shifted_basis


def _scores(q, k, qi, ki, time, dq):
    scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
    causal = (ki[None, :] <= qi[:, None]) & (ki[None, :] < time) & (qi[:, None] < time)
    return jnp.where(causal, scores, 0.0), causal


def _weighted(bases, gates):
    value = bases[0] * gates[0][:, None]
    for base, gate in zip(bases[1:], gates[1:], strict=True):
        value = value + base * gate[:, None]
    return value


def _forward_kernel(q_ref, k_ref, v_ref, *refs, time, dq, dv, bq, bk):
    *gate_refs, output_ref, shift_ref = refs
    qi = pl.program_id(0) * bq + jnp.arange(bq)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    q = plt.load(q_ref, mask=qmask, other=0)
    gates = tuple(plt.load(ref, mask=qi < time, other=0) for ref in gate_refs)

    def body(tile, carry):
        output, previous = carry
        ki = tile * bk + jnp.arange(bk)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        shift = jnp.maximum(previous, row_shift(scores, gates[-1] != 0.0))
        bases = shifted_basis(scores, shift[:, None], gates[-1][:, None] != 0.0)
        weights = jnp.where(causal, _weighted(bases, gates), 0.0)
        output = output * jnp.exp(previous - shift)[:, None] + _dot(weights, v)
        return output, shift

    end = lax.div((pl.program_id(0) + 1) * bq + bk - 1, bk)
    output, shift = lax.fori_loop(0, end, body, (jnp.zeros((bq, v_ref.shape[-1]), jnp.float32),
                                                 jnp.zeros((bq,), jnp.float32)))
    plt.store(output_ref, output, mask=(qi[:, None] < time) & (jnp.arange(output_ref.shape[-1])[None, :] < dv))
    plt.store(shift_ref, shift, mask=qi < time)


def _dq_kernel(q_ref, k_ref, v_ref, do_ref, shift_ref, *refs, time, dq, dv, bq, bk):
    gate_refs, dq_ref, dg_refs = refs[:BASIS_COUNT], refs[BASIS_COUNT], refs[BASIS_COUNT + 1:]
    qi = pl.program_id(0) * bq + jnp.arange(bq)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=(qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv), other=0)
    shift = plt.load(shift_ref, mask=qi < time, other=0)
    gates = tuple(plt.load(ref, mask=qi < time, other=0) for ref in gate_refs)

    def body(tile, carry):
        grad_q, grad_gates = carry
        ki = tile * bk + jnp.arange(bk)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        bases, slopes = shifted_basis(scores, shift[:, None], gates[-1][:, None] != 0.0, slopes=True)
        dp = jnp.where(causal, _dot(do, v.T), 0.0)
        grad_q += _dot(dp * _weighted(slopes, gates), k) * (1.0 / math.sqrt(dq))
        return grad_q, tuple(g + jnp.sum(dp * base, axis=1) for g, base in zip(grad_gates, bases, strict=True))

    end = lax.div((pl.program_id(0) + 1) * bq + bk - 1, bk)
    grad_q, grad_gates = lax.fori_loop(0, end, body, (jnp.zeros(q_ref.shape, jnp.float32),
                                                      (jnp.zeros((bq,), jnp.float32),) * BASIS_COUNT))
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    for ref, grad in zip(dg_refs, grad_gates, strict=True):
        plt.store(ref, grad, mask=qi < time)


def _dkv_kernel(q_ref, k_ref, v_ref, do_ref, shift_ref, *refs, time, dq, dv, bq, bk):
    *gate_refs, dk_ref, dv_ref = refs
    ki = pl.program_id(0) * bk + jnp.arange(bk)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k, v = plt.load(k_ref, mask=kmask, other=0), plt.load(v_ref, mask=vmask, other=0)

    def body(tile, carry):
        grad_k, grad_v = carry
        qi = tile * bq + jnp.arange(bq)
        q, do = _load(q_ref, qi, dq, time), _load(do_ref, qi, dv, time)
        shift = plt.load(shift_ref.at[qi], mask=qi < time, other=0)
        gates = tuple(plt.load(ref.at[qi], mask=qi < time, other=0) for ref in gate_refs)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        bases, slopes = shifted_basis(scores, shift[:, None], gates[-1][:, None] != 0.0, slopes=True)
        weights = jnp.where(causal, _weighted(bases, gates), 0.0)
        ds = jnp.where(causal, _dot(do, v.T) * _weighted(slopes, gates), 0.0)
        return grad_k + _dot(ds.T, q) * (1.0 / math.sqrt(dq)), grad_v + _dot(weights.T, do)

    start = lax.div(pl.program_id(0) * bk, bq)
    grad_k, grad_v = lax.fori_loop(start, pl.cdiv(time, bq), body, (jnp.zeros(k_ref.shape, jnp.float32),
                                                                    jnp.zeros(v_ref.shape, jnp.float32)))
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


def _launch(kernel, args, outputs, specs, out_specs, *, query_tiles, q, v, interpret):
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    bq, bk = _tile_sizes(dq, dv)
    return pl.pallas_call(partial(kernel, time=time, dq=dq, dv=dv, bq=bq, bk=bk),
                          out_shape=outputs, in_specs=specs, out_specs=out_specs,
                          grid=(pl.cdiv(time, bq if query_tiles else bk), batch, heads),
                          compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
                          interpret=interpret, name=f"even_gate_exp_{kernel.__name__.strip('_')}")(*args)


def _forward(q, k, v, gates, interpret):
    bq, _ = _tile_sizes(q.shape[-1], v.shape[-1])
    time = q.shape[1]
    query, key, value = _spec(bq, q.shape[-1]), _full_spec(time, k.shape[-1]), _full_spec(time, v.shape[-1])
    row = _shift_spec(bq)
    output, shift = _launch(_forward_kernel, (q, k, v, *gates),
                            (jax.ShapeDtypeStruct(v.shape, jnp.float32), jax.ShapeDtypeStruct(q.shape[:3], jnp.float32)),
                            (query, key, value, *(row,) * BASIS_COUNT), (_spec(bq, v.shape[-1]), row),
                            query_tiles=True, q=q, v=v, interpret=interpret)
    # Keep the scaled sum and its incoming cotangent in FP32 through head RMSNorm. In particular the
    # small residual gradient of a nearly dominant degree is sensitive to rounding before normalization.
    return output, lax.stop_gradient(shift)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _attention(q, k, v, gates, interpret):
    return _forward(q, k, v, gates, interpret)


def _fwd(q, k, v, gates, interpret):
    output, shift = _forward(q, k, v, gates, interpret)
    return (output, shift), (q, k, v, gates, shift)


def _bwd(interpret, saved, cotangents):
    q, k, v, gates, shift = saved
    do, _ = cotangents
    time, dq, dv = q.shape[1], q.shape[-1], v.shape[-1]
    bq, bk = _tile_sizes(dq, dv)
    row, full_row = _shift_spec(bq), _full_shift_spec(time)
    dq_type, dk_type, dv_type = (jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, k, v))
    gate_type = jax.ShapeDtypeStruct(q.shape[:3], jnp.float32)
    args = (q, k, v, do, shift, *gates)
    grad_q, *grad_gates = _launch(_dq_kernel, args, (dq_type, *(gate_type,) * BASIS_COUNT),
                                  (_spec(bq, dq), _full_spec(time, dq), _full_spec(time, dv),
                                   _spec(bq, dv), row, *(row,) * BASIS_COUNT),
                                  (_spec(bq, dq), *(row,) * BASIS_COUNT), query_tiles=True, q=q, v=v, interpret=interpret)
    grad_k, grad_v = _launch(_dkv_kernel, args, (dk_type, dv_type),
                             (_full_spec(time, dq), _spec(bk, dq), _spec(bk, dv), _full_spec(time, dv),
                              full_row, *(full_row,) * BASIS_COUNT),
                             (_spec(bk, dq), _spec(bk, dv)), query_tiles=False, q=q, v=v, interpret=interpret)
    return grad_q, grad_k, grad_v, tuple(grad_gates)


_attention.defvjp(_fwd, _bwd)


def fused_even_gate_exp(q, k, v, coefficients, exp_gate, *, interpret=False):
    gates = tuple(coefficients[..., order] for order in range(2, 17, 2)) + (exp_gate,)
    return _attention(q, k, v, gates, interpret)
