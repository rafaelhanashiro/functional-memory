"""Causal degree-16 even powers plus exp with nine query gates and one common numerical shift.

The query-tile program accumulates V online. Its companion recomputes the shifted weights' slopes to return dQ
and all nine gate derivatives. A key-tile program accumulates dK/dV. No pass stores pair weights. With BF16 V the
output cotangent enters both backward programs as its three BF16 components, split once
(fused_unnormalized._bf16_components): the products are those of splitting it in every tile.

The even powers are one polynomial in w = u^2, u = s exp(-shift/16) (|u| <= 1, even_gate_exp.shifted_basis):
sum_r c_r s^r e^-shift = w (a_2 + w (a_4 + ... + w a_16)), a_r = c_r exp(-(16 - r) shift / 16) per query row,
evaluated by Horner's rule in seven multiply-adds per pair, and its slope with the shift fixed is
u e^(-shift/16) (2 a_2 + w (4 a_4 + ... + w 16 a_16)). Gate r's derivative is exp(-(16 - r) shift / 16) times the
row's moment sum_i dp_i w_i^(r/2), accumulated from successive products dp w^k.
"""
from functools import partial
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from scaffolds.transformer.attention.backends.fused_unnormalized import (
    NUM_STAGES, NUM_WARPS, _bf16_components, _dot, _full_shift_spec, _full_spec, _load, _shift_spec, _spec,
    _tile_sizes,
)
from scaffolds.transformer.attention.even_gate_exp import BASIS_COUNT, DEGREE, ORDERS, row_shift


def _scores(q, k, qi, ki, time, dq):
    scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
    causal = (ki[None, :] <= qi[:, None]) & (ki[None, :] < time) & (qi[:, None] < time)
    return jnp.where(causal, scores, 0.0), causal


def _cotangent_dot(do, v):
    """do v^T for the output cotangent as one FP32 tile, or as its three BF16 components with BF16 v: then the
    three products _dot forms after splitting do itself."""
    if len(do) == 1:
        return _dot(do[0], v.T)
    high, middle, low = do
    return (pl.dot(high, v.T, precision=lax.Precision.HIGHEST) + pl.dot(middle, v.T, precision=lax.Precision.HIGHEST)
            + pl.dot(low, v.T, precision=lax.Precision.HIGHEST))


def _cotangent(do):
    """The FP32 cotangent tile: the components' sum, exact where they split it."""
    if len(do) == 1:
        return do[0]
    high, middle, low = (part.astype(jnp.float32) for part in do)
    return (high + middle) + low


def _row_units(shift, gates):
    """Per query row: half = exp(-shift/32), the even powers' row units exp(-(16 - r) shift / 16), and the
    Horner coefficients a_r = c_r times those units."""
    half = jnp.exp(shift * (-0.5 / DEGREE))
    units = tuple(1.0 if r == DEGREE else jnp.exp(shift * (-(DEGREE - r) / DEGREE)) for r in ORDERS)
    return half, units, tuple(gate * unit for gate, unit in zip(gates[:-1], units, strict=True))


def _horner(coefficients, w):
    """sum_k coefficients[k] w^k for per-row coefficients [rows] and w [rows, keys]."""
    value = coefficients[-1][:, None]
    for coefficient in reversed(coefficients[:-1]):
        value = value * w + coefficient[:, None]
    return value


def _exponential(scores, shift, exp_gate):
    # A sigmoid rounded to zero is flat in the model: its basis and its logit gradient vanish. Mask before exp, so a
    # switched-off basis neither overflows nor sets the row's units.
    return jnp.exp(jnp.where(exp_gate[:, None] != 0.0, scores, -jnp.inf) - shift[:, None])


def _polynomial(scores, half, coefficients, *, slope=False):
    """Shifted even polynomial w * sum_k a_2k w^(k-1) and, if asked, its slope with the shift fixed."""
    u = (scores * half[:, None]) * half[:, None]
    w = u * u
    value = _horner(coefficients, w) * w
    if not slope:
        return value, w
    scaled = tuple(r * a * half * half for r, a in zip(ORDERS, coefficients, strict=True))
    return value, w, _horner(scaled, w) * u


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
        half, _, coefficients = _row_units(shift, gates)
        polynomial, _ = _polynomial(scores, half, coefficients)
        weights = polynomial + gates[-1][:, None] * _exponential(scores, shift, gates[-1])
        weights = jnp.where(causal, weights, 0.0)
        output = output * jnp.exp(previous - shift)[:, None] + _dot(weights, v)
        return output, shift

    end = lax.div((pl.program_id(0) + 1) * bq + bk - 1, bk)
    output, shift = lax.fori_loop(0, end, body, (jnp.zeros((bq, v_ref.shape[-1]), jnp.float32),
                                                 jnp.zeros((bq,), jnp.float32)))
    plt.store(output_ref, output, mask=(qi[:, None] < time) & (jnp.arange(output_ref.shape[-1])[None, :] < dv))
    plt.store(shift_ref, shift, mask=qi < time)


def _dq_kernel(q_ref, k_ref, v_ref, *refs, parts, time, dq, dv, bq, bk):
    do_refs, shift_ref, refs = refs[:parts], refs[parts], refs[parts + 1:]
    gate_refs, dq_ref, dg_refs = refs[:BASIS_COUNT], refs[BASIS_COUNT], refs[BASIS_COUNT + 1:]
    qi = pl.program_id(0) * bq + jnp.arange(bq)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = tuple(plt.load(ref, mask=(qi[:, None] < time) & (jnp.arange(ref.shape[-1])[None, :] < dv), other=0)
               for ref in do_refs)
    shift = plt.load(shift_ref, mask=qi < time, other=0)
    gates = tuple(plt.load(ref, mask=qi < time, other=0) for ref in gate_refs)
    half, units, coefficients = _row_units(shift, gates)

    def body(tile, carry):
        grad_q, moments = carry
        ki = tile * bk + jnp.arange(bk)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        _, w, slope = _polynomial(scores, half, coefficients, slope=True)
        exponential = _exponential(scores, shift, gates[-1])
        dp = jnp.where(causal, _cotangent_dot(do, v), 0.0)
        grad_q += _dot(dp * (slope + gates[-1][:, None] * exponential), k) * (1.0 / math.sqrt(dq))
        product, updated = dp, []
        for moment in moments[:-1]:
            product = product * w
            updated.append(moment + jnp.sum(product, axis=1))
        return grad_q, (*updated, moments[-1] + jnp.sum(dp * exponential, axis=1))

    end = lax.div((pl.program_id(0) + 1) * bq + bk - 1, bk)
    grad_q, moments = lax.fori_loop(0, end, body, (jnp.zeros(q_ref.shape, jnp.float32),
                                                   (jnp.zeros((bq,), jnp.float32),) * BASIS_COUNT))
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    for ref, grad in zip(dg_refs, (*(m * unit for m, unit in zip(moments[:-1], units, strict=True)), moments[-1]),
                         strict=True):
        plt.store(ref, grad, mask=qi < time)


def _dkv_kernel(q_ref, k_ref, v_ref, *refs, parts, time, dq, dv, bq, bk):
    do_refs, shift_ref, refs = refs[:parts], refs[parts], refs[parts + 1:]
    *gate_refs, dk_ref, dv_ref = refs
    ki = pl.program_id(0) * bk + jnp.arange(bk)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k, v = plt.load(k_ref, mask=kmask, other=0), plt.load(v_ref, mask=vmask, other=0)

    def body(tile, carry):
        grad_k, grad_v = carry
        qi = tile * bq + jnp.arange(bq)
        q, do = _load(q_ref, qi, dq, time), tuple(_load(ref, qi, dv, time) for ref in do_refs)
        shift = plt.load(shift_ref.at[qi], mask=qi < time, other=0)
        gates = tuple(plt.load(ref.at[qi], mask=qi < time, other=0) for ref in gate_refs)
        half, _, coefficients = _row_units(shift, gates)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        polynomial, _, slope = _polynomial(scores, half, coefficients, slope=True)
        exponential = gates[-1][:, None] * _exponential(scores, shift, gates[-1])
        weights = jnp.where(causal, polynomial + exponential, 0.0)
        ds = jnp.where(causal, _cotangent_dot(do, v) * (slope + exponential), 0.0)
        return grad_k + _dot(ds.T, q) * (1.0 / math.sqrt(dq)), grad_v + _dot(weights.T, _cotangent(do))

    start = lax.div(pl.program_id(0) * bk, bq)
    grad_k, grad_v = lax.fori_loop(start, pl.cdiv(time, bq), body, (jnp.zeros(k_ref.shape, jnp.float32),
                                                                    jnp.zeros(v_ref.shape, jnp.float32)))
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


# Query tile, key tile, warps and pipeline stages of each program at head widths up to 64: measured at Q/K width 4,
# value width 64 and 2,048 tokens on B200 (validation README). Narrower key tiles relieve the key-tile program's
# registers; neither change alters any sum.
TILING = {"forward": (64, 64, 8, NUM_STAGES), "dq": (64, 64, NUM_WARPS, NUM_STAGES),
          "dkv": (64, 32, NUM_WARPS, NUM_STAGES)}


def _tiling(program, dq, dv):
    if max(dq, dv) <= 64:
        return TILING[program]
    return (*_tile_sizes(dq, dv), NUM_WARPS, NUM_STAGES)


def _launch(kernel, program, args, outputs, specs, out_specs, *, query_tiles, q, v, interpret, **static):
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    bq, bk, warps, stages = _tiling(program, dq, dv)
    return pl.pallas_call(partial(kernel, time=time, dq=dq, dv=dv, bq=bq, bk=bk, **static),
                          out_shape=outputs, in_specs=specs, out_specs=out_specs,
                          grid=(pl.cdiv(time, bq if query_tiles else bk), batch, heads),
                          compiler_params=plt.CompilerParams(num_warps=warps, num_stages=stages),
                          interpret=interpret, name=f"even_gate_exp_{kernel.__name__.strip('_')}")(*args)


def _forward(q, k, v, gates, interpret):
    bq = _tiling("forward", q.shape[-1], v.shape[-1])[0]
    time = q.shape[1]
    query, key, value = _spec(bq, q.shape[-1]), _full_spec(time, k.shape[-1]), _full_spec(time, v.shape[-1])
    row = _shift_spec(bq)
    output, shift = _launch(_forward_kernel, "forward", (q, k, v, *gates),
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
    bq, bk = _tiling("dq", dq, dv)[0], _tiling("dkv", dq, dv)[1]
    row, full_row = _shift_spec(bq), _full_shift_spec(time)
    dq_type, dk_type, dv_type = (jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, k, v))
    gate_type = jax.ShapeDtypeStruct(q.shape[:3], jnp.float32)
    cotangent = _bf16_components(do) if v.dtype == jnp.bfloat16 else (do,)
    parts = len(cotangent)
    args = (q, k, v, *cotangent, shift, *gates)
    grad_q, *grad_gates = _launch(_dq_kernel, "dq", args, (dq_type, *(gate_type,) * BASIS_COUNT),
                                  (_spec(bq, dq), _full_spec(time, dq), _full_spec(time, dv),
                                   *(_spec(bq, dv),) * parts, row, *(row,) * BASIS_COUNT),
                                  (_spec(bq, dq), *(row,) * BASIS_COUNT), query_tiles=True, q=q, v=v,
                                  interpret=interpret, parts=parts)
    grad_k, grad_v = _launch(_dkv_kernel, "dkv", args, (dk_type, dv_type),
                             (_full_spec(time, dq), _spec(bk, dq), _spec(bk, dv), *(_full_spec(time, dv),) * parts,
                              full_row, *(full_row,) * BASIS_COUNT),
                             (_spec(bk, dq), _spec(bk, dv)), query_tiles=False, q=q, v=v, interpret=interpret,
                             parts=parts)
    return grad_q, grad_k, grad_v, tuple(grad_gates)


_attention.defvjp(_fwd, _bwd)


def fused_even_gate_exp(q, k, v, coefficients, exp_gate, *, interpret=False):
    gates = tuple(coefficients[..., order] for order in ORDERS) + (exp_gate,)
    return _attention(q, k, v, gates, interpret)
