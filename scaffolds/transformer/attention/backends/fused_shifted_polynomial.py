"""Causal per-query polynomials sum_r c_r(t) z^r of degree M, z = s or softplus(s), with one common row shift.

The query-tile program accumulates V online. Its companion recomputes the shifted weights' slopes to return dQ and
every coefficient's derivative. A key-tile program accumulates dK/dV. No pass stores pair weights. With BF16 V the
output cotangent enters both backward programs as its three BF16 components, split once
(fused_unnormalized._bf16_components); dV's product takes the components' FP32 sum.

With u = z exp(-shift/M) (|u| <= 1, shifted_polynomial.shifted_terms) the shifted polynomial is sum_r a_r u^r,
a_r = c_r exp(-(M - r) shift / M) per query row, evaluated by Horner's rule in M multiply-adds per pair; its slope in
z with the shift fixed is exp(-shift/M) sum_r r a_r u^(r-1), times dz/ds for softplus. Coefficient r's derivative is
exp(-(M - r) shift / M) times the row's moment sum_i dp_i u_i^r, accumulated from successive products dp u^k.

With a raw-score prefactor f the key bound is b = log f(s) + M log S with S = max(1, |z|).
In units of the largest causal bound the weight is exp(b - shift) p(z)/S^M. Its score
slope in these units is exp(b - shift) [(f'/f) p(z) + p'(z) dz/ds]/S^M. Coefficient
derivatives use z^r/S^M, with reversed powers of 1/z where |z| > 1.
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
from scaffolds.transformer.attention.shifted_polynomial import PREFACTORS, TRANSFORMS


def _scores(q, k, qi, ki, time, dq):
    scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
    causal = (ki[None, :] <= qi[:, None]) & (ki[None, :] < time) & (qi[:, None] < time)
    return jnp.where(causal, scores, 0.0), causal


def _transformed(scores, transform):
    """z and dz/ds (None for the identity). A masked score is zero, so its z has magnitude below one."""
    if transform == "identity":
        return scores, None
    # softplus(s) = max(s, 0) + log(1 + e^-|s|) and its derivative, the logistic sigmoid, without overflow.
    small = jnp.exp(-jnp.abs(scores))
    return (jnp.maximum(scores, 0.0) + jnp.log1p(small),
            jnp.where(scores >= 0, 1.0 / (1.0 + small), small / (1.0 + small)))


def _row_shift(z, degree):
    return degree * jnp.log(jnp.maximum(1.0, jnp.max(jnp.abs(z), axis=1)))


def _key_bound(scores, prefactor, degree, *, z=None):
    """b = log f(s) + M log max(1, |z|) and f'(s)/f(s) (None for exp, where it is one). For softplus below s = -20 both
    take their limits, s and one, within e^-20."""
    z = scores if z is None else z
    magnitude = degree * jnp.log(jnp.maximum(1.0, jnp.abs(z)))
    if prefactor == "exp":
        return scores + magnitude, None
    softplus, sigmoid = _transformed(scores, "softplus")
    tail = scores <= -20.0
    safe = jnp.maximum(softplus, 1e-30)
    return (jnp.where(tail, scores, jnp.log(safe)) + magnitude,
            jnp.where(tail, 1.0, sigmoid / safe))


def _keyed_polynomial(scores, coefficients, *, slope=False):
    """h = p(s) / max(1, |s|)^M and, if asked, p'(s) / max(1, |s|)^M: Horner in s where |s| <= 1, in 1/s over the
    reversed coefficients elsewhere, times sign(s)^M."""
    degree = len(coefficients) - 1
    small = jnp.abs(scores) <= 1.0
    inner = jnp.where(small, scores, 0.0)
    inverse = 1.0 / jnp.where(small, 1.0, scores)
    sign = jnp.where(scores < 0, -1.0, 1.0) if degree % 2 else 1.0
    value = jnp.where(small, _horner(coefficients, inner), sign * _horner(coefficients[::-1], inverse))
    if not slope:
        return value
    weighted = tuple(r * c for r, c in enumerate(coefficients))
    derivative = jnp.where(small, _horner(weighted[1:], inner), sign * inverse * _horner(weighted[::-1], inverse))
    return value, derivative


def _cotangent_dot(do, v):
    """do v^T for the output cotangent as one FP32 tile, or as its three BF16 components with BF16 v: then the
    three products _dot forms after splitting do itself."""
    if len(do) == 1:
        return _dot(do[0], v.T)
    high, middle, low = do
    return (pl.dot(high, v.T, precision=lax.Precision.HIGHEST) + pl.dot(middle, v.T, precision=lax.Precision.HIGHEST)
            + pl.dot(low, v.T, precision=lax.Precision.HIGHEST))


def _cotangent(do):
    """The FP32 cotangent tile: the components' sum."""
    if len(do) == 1:
        return do[0]
    high, middle, low = (part.astype(jnp.float32) for part in do)
    return (high + middle) + low


def _row_units(shift, coefficients):
    """Per query row: half = exp(-shift/(2M)), the units exp(-(M - r) shift / M) for r = 0..M, and the Horner
    coefficients a_r = c_r times those units."""
    degree = len(coefficients) - 1
    half = jnp.exp(shift * (-0.5 / degree))
    units = tuple(1.0 if r == degree else jnp.exp(shift * (-(degree - r) / degree)) for r in range(degree + 1))
    return half, units, tuple(c * unit for c, unit in zip(coefficients, units, strict=True))


def _horner(coefficients, x):
    """sum_k coefficients[k] x^k for per-row coefficients [rows] and x [rows, keys]."""
    value = coefficients[-1][:, None]
    for coefficient in reversed(coefficients[:-1]):
        value = value * x + coefficient[:, None]
    return value


def _polynomial(z, half, coefficients, *, slope=False):
    """Shifted polynomial sum_r a_r u^r and, if asked, its slope in z with the shift fixed."""
    u = (z * half[:, None]) * half[:, None]
    value = _horner(coefficients, u)
    if not slope:
        return value, u
    scaled = tuple(r * a * half * half for r, a in enumerate(coefficients) if r)
    return value, u, _horner(scaled, u)


def _forward_kernel(q_ref, k_ref, v_ref, *refs, transform, prefactor, time, dq, dv, bq, bk):
    *coefficient_refs, output_ref, shift_ref = refs
    degree = len(coefficient_refs) - 1
    qi = pl.program_id(0) * bq + jnp.arange(bq)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    q = plt.load(q_ref, mask=qmask, other=0)
    coefficients = tuple(plt.load(ref, mask=qi < time, other=0) for ref in coefficient_refs)

    def body(tile, carry):
        output, previous = carry
        ki = tile * bk + jnp.arange(bk)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        if prefactor == "none":
            z, _ = _transformed(scores, transform)
            shift = jnp.maximum(previous, _row_shift(z, degree))
            half, _, horner = _row_units(shift, coefficients)
            weights, _ = _polynomial(z, half, horner)
        else:
            # A masked score is zero; its bound, at most zero, never raises the shift above its floor.
            z, _ = _transformed(scores, transform)
            bound, _ = _key_bound(scores, prefactor, degree, z=z)
            shift = jnp.maximum(previous, jnp.max(bound, axis=1))
            weights = jnp.exp(bound - shift[:, None]) * _keyed_polynomial(z, coefficients)
        weights = jnp.where(causal, weights, 0.0)
        output = output * jnp.exp(previous - shift)[:, None] + _dot(weights, v)
        return output, shift

    end = lax.div((pl.program_id(0) + 1) * bq + bk - 1, bk)
    output, shift = lax.fori_loop(0, end, body, (jnp.zeros((bq, v_ref.shape[-1]), jnp.float32),
                                                 jnp.zeros((bq,), jnp.float32)))
    plt.store(output_ref, output, mask=(qi[:, None] < time) & (jnp.arange(output_ref.shape[-1])[None, :] < dv))
    plt.store(shift_ref, shift, mask=qi < time)


def _dq_kernel(q_ref, k_ref, v_ref, *refs, parts, transform, prefactor, time, dq, dv, bq, bk):
    do_refs, shift_ref, refs = refs[:parts], refs[parts], refs[parts + 1:]
    count = (len(refs) - 1) // 2  # the coefficients, dQ, and the coefficients' derivatives
    coefficient_refs, dq_ref, dc_refs = refs[:count], refs[count], refs[count + 1:]
    qi = pl.program_id(0) * bq + jnp.arange(bq)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = tuple(plt.load(ref, mask=(qi[:, None] < time) & (jnp.arange(ref.shape[-1])[None, :] < dv), other=0)
               for ref in do_refs)
    shift = plt.load(shift_ref, mask=qi < time, other=0)
    coefficients = tuple(plt.load(ref, mask=qi < time, other=0) for ref in coefficient_refs)
    degree = count - 1
    half, units, horner = _row_units(shift, coefficients)

    def body(tile, carry):
        grad_q, moments = carry
        ki = tile * bk + jnp.arange(bk)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        if prefactor == "none":
            z, dz = _transformed(scores, transform)
            _, u, slope = _polynomial(z, half, horner, slope=True)
            dp = jnp.where(causal, _cotangent_dot(do, v), 0.0)
            ds = dp * slope if dz is None else dp * slope * dz
            grad_q += _dot(ds, k) * (1.0 / math.sqrt(dq))
            product, updated = dp, [moments[0] + jnp.sum(dp, axis=1)]
            for moment in moments[1:]:
                product = product * u
                updated.append(moment + jnp.sum(product, axis=1))
            return grad_q, tuple(updated)
        z, dz = _transformed(scores, transform)
        bound, ratio = _key_bound(scores, prefactor, degree, z=z)
        value, slope = _keyed_polynomial(z, coefficients, slope=True)
        dp = jnp.where(causal, _cotangent_dot(do, v), 0.0) * jnp.exp(bound - shift[:, None])
        slope = slope if dz is None else slope * dz
        ds = dp * (value + slope if ratio is None else ratio * value + slope)
        grad_q += _dot(ds, k) * (1.0 / math.sqrt(dq))
        # Coefficient r's moment of z^r / max(1, |z|)^M: z^r where |z| <= 1, sign(z)^M (1/z)^(M - r) elsewhere.
        small = jnp.abs(z) <= 1.0
        inverse = 1.0 / jnp.where(small, 1.0, z)
        ascending = jnp.where(small, dp, 0.0)
        descending = jnp.where(small, 0.0, dp) * (jnp.where(z < 0, -1.0, 1.0) if degree % 2 else 1.0)
        updated = list(moments)
        for step in range(degree + 1):
            if step:
                ascending, descending = ascending * z, descending * inverse
            updated[step] = updated[step] + jnp.sum(ascending, axis=1)
            updated[degree - step] = updated[degree - step] + jnp.sum(descending, axis=1)
        return grad_q, tuple(updated)

    end = lax.div((pl.program_id(0) + 1) * bq + bk - 1, bk)
    grad_q, moments = lax.fori_loop(0, end, body, (jnp.zeros(q_ref.shape, jnp.float32),
                                                   (jnp.zeros((bq,), jnp.float32),) * count))
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    for ref, moment, unit in zip(dc_refs, moments, units, strict=True):
        # The keyed moments carry each key's units already.
        plt.store(ref, moment * unit if prefactor == "none" else moment, mask=qi < time)


def _dkv_kernel(q_ref, k_ref, v_ref, *refs, parts, transform, prefactor, time, dq, dv, bq, bk):
    do_refs, shift_ref, refs = refs[:parts], refs[parts], refs[parts + 1:]
    *coefficient_refs, dk_ref, dv_ref = refs
    ki = pl.program_id(0) * bk + jnp.arange(bk)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k, v = plt.load(k_ref, mask=kmask, other=0), plt.load(v_ref, mask=vmask, other=0)

    def body(tile, carry):
        grad_k, grad_v = carry
        qi = tile * bq + jnp.arange(bq)
        q, do = _load(q_ref, qi, dq, time), tuple(_load(ref, qi, dv, time) for ref in do_refs)
        shift = plt.load(shift_ref.at[qi], mask=qi < time, other=0)
        coefficients = tuple(plt.load(ref.at[qi], mask=qi < time, other=0) for ref in coefficient_refs)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        dp = _cotangent_dot(do, v)
        if prefactor == "none":
            half, _, horner = _row_units(shift, coefficients)
            z, dz = _transformed(scores, transform)
            weights, _, slope = _polynomial(z, half, horner, slope=True)
            ds = jnp.where(causal, dp * slope if dz is None else dp * slope * dz, 0.0)
        else:
            z, dz = _transformed(scores, transform)
            bound, ratio = _key_bound(scores, prefactor, len(coefficients) - 1, z=z)
            unit = jnp.exp(bound - shift[:, None])
            value, slope = _keyed_polynomial(z, coefficients, slope=True)
            weights = unit * value
            slope = slope if dz is None else slope * dz
            ds = jnp.where(causal, dp * unit * (value + slope if ratio is None else ratio * value + slope), 0.0)
        weights = jnp.where(causal, weights, 0.0)
        return grad_k + _dot(ds.T, q) * (1.0 / math.sqrt(dq)), grad_v + _dot(weights.T, _cotangent(do))

    start = lax.div(pl.program_id(0) * bk, bq)
    grad_k, grad_v = lax.fori_loop(start, pl.cdiv(time, bq), body, (jnp.zeros(k_ref.shape, jnp.float32),
                                                                    jnp.zeros(v_ref.shape, jnp.float32)))
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


# Query tile, key tile, warps and pipeline stages of each program at head widths up to 64: measured at degree 17
# (softplus), Q/K width 4, value width 64 and 2,048 tokens on B200 (validation README). Narrow tiles relieve the
# backward programs' registers, which hold every order's row vectors; tiles change only the order of FP32 sums.
TILING = {"forward": (64, 64, 8, NUM_STAGES), "dq": (64, 32, NUM_WARPS, NUM_STAGES),
          "dkv": (16, 32, NUM_WARPS, NUM_STAGES)}


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
                          interpret=interpret, name=f"shifted_polynomial_{kernel.__name__.strip('_')}")(*args)


def _forward(q, k, v, coefficients, transform, prefactor, interpret):
    bq = _tiling("forward", q.shape[-1], v.shape[-1])[0]
    time = q.shape[1]
    query, key, value = _spec(bq, q.shape[-1]), _full_spec(time, k.shape[-1]), _full_spec(time, v.shape[-1])
    row = _shift_spec(bq)
    output, shift = _launch(_forward_kernel, "forward", (q, k, v, *coefficients),
                            (jax.ShapeDtypeStruct(v.shape, jnp.float32), jax.ShapeDtypeStruct(q.shape[:3], jnp.float32)),
                            (query, key, value, *(row,) * len(coefficients)), (_spec(bq, v.shape[-1]), row),
                            query_tiles=True, q=q, v=v, interpret=interpret, transform=transform,
                            prefactor=prefactor)
    # Keep the shifted sum and its incoming cotangent in FP32 through head RMSNorm.
    return output, lax.stop_gradient(shift)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _attention(q, k, v, coefficients, transform, prefactor, interpret):
    return _forward(q, k, v, coefficients, transform, prefactor, interpret)


def _fwd(q, k, v, coefficients, transform, prefactor, interpret):
    output, shift = _forward(q, k, v, coefficients, transform, prefactor, interpret)
    return (output, shift), (q, k, v, coefficients, shift)


def _bwd(transform, prefactor, interpret, saved, cotangents):
    q, k, v, coefficients, shift = saved
    do, _ = cotangents
    time, dq, dv = q.shape[1], q.shape[-1], v.shape[-1]
    bq, bk = _tiling("dq", dq, dv)[0], _tiling("dkv", dq, dv)[1]
    row, full_row = _shift_spec(bq), _full_shift_spec(time)
    dq_type, dk_type, dv_type = (jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (q, k, v))
    coefficient_type = jax.ShapeDtypeStruct(q.shape[:3], jnp.float32)
    cotangent = _bf16_components(do) if v.dtype == jnp.bfloat16 else (do,)
    parts, count = len(cotangent), len(coefficients)
    args = (q, k, v, *cotangent, shift, *coefficients)
    grad_q, *grad_coefficients = _launch(_dq_kernel, "dq", args, (dq_type, *(coefficient_type,) * count),
                                         (_spec(bq, dq), _full_spec(time, dq), _full_spec(time, dv),
                                          *(_spec(bq, dv),) * parts, row, *(row,) * count),
                                         (_spec(bq, dq), *(row,) * count), query_tiles=True, q=q, v=v,
                                         interpret=interpret, parts=parts, transform=transform,
                                         prefactor=prefactor)
    grad_k, grad_v = _launch(_dkv_kernel, "dkv", args, (dk_type, dv_type),
                             (_full_spec(time, dq), _spec(bk, dq), _spec(bk, dv), *(_full_spec(time, dv),) * parts,
                              full_row, *(full_row,) * count),
                             (_spec(bk, dq), _spec(bk, dv)), query_tiles=False, q=q, v=v, interpret=interpret,
                             parts=parts, transform=transform, prefactor=prefactor)
    return grad_q, grad_k, grad_v, tuple(grad_coefficients)


_attention.defvjp(_fwd, _bwd)


def fused_shifted_polynomial(q, k, v, coefficients, *, transform="identity", prefactor="none", interpret=False):
    """The shifted output e^(-m) o [B,T,H,dv] in FP32 and the stopped shifts m [B,T,H]; coefficients [B,T,H,M+1]."""
    if transform not in TRANSFORMS:
        raise ValueError(f"The shifted polynomial takes the transforms {TRANSFORMS}, not {transform!r}")
    if prefactor not in PREFACTORS or (prefactor == "softplus" and transform != "identity"):
        raise ValueError(f"The shifted polynomial takes raw-score prefactors {PREFACTORS}; "
                         "the softplus prefactor requires identity polynomial input")
    rows = tuple(coefficients[..., r] for r in range(coefficients.shape[-1]))
    return _attention(q, k, v, rows, transform, prefactor, interpret)
