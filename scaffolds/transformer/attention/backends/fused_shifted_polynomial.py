"""Causal per-query polynomials sum_r c_r(t) z^r of degree M, z = s or softplus(s), with one common row shift.

The query-tile program accumulates V online. Its companion recomputes the shifted weights' slopes to return dQ and
every coefficient's derivative. A key-tile program accumulates dK/dV. No pass stores pair weights. With BF16 V the
output cotangent enters both backward programs as its three BF16 components, split once
(fused_unnormalized._bf16_components); dV's product takes the components' FP32 sum.

With u = z exp(-shift/M) (|u| <= 1, shifted_polynomial.shifted_terms) the shifted polynomial is sum_r a_r u^r,
a_r = c_r exp(-(M - r) shift / M) per query row, evaluated by Horner's rule in M multiply-adds per pair; its slope in
z with the shift fixed is exp(-shift/M) sum_r r a_r u^(r-1), times dz/ds for softplus. Coefficient r's derivative is
exp(-(M - r) shift / M) times the row's moment sum_i dp_i u_i^r, accumulated from successive products dp u^k.

With a prefactor f of the raw score (z = s) the row's units come from its largest score (shifted_polynomial.key_shift),
which the query-tile program tracks online, and each pair is evaluated in one pass (shifted_polynomial.keyed_weights):
chunks of p by Horner's rule in s, staged by an n-th root g of the exponential part of f(s) e^(-shift), with no
logarithm or reciprocal of a score. The slope stages the chunks F1 P_j + F P_j' for the factors F of f and F1 of f'
outside g (for exp the chunks of p + p'), and coefficient r's derivative is the row sum of dp F g^n s^r
(_keyed_moments).
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
from scaffolds.transformer.attention.shifted_polynomial import (
    NO_SCORE, PREFACTORS, TRANSFORMS, key_shift, key_terms, keyed_weights,
)


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


def _row_top(scores, causal, previous):
    """The largest causal score of each row so far."""
    return jnp.maximum(previous, jnp.max(jnp.where(causal, scores, NO_SCORE), axis=1))


def _keyed(scores, coefficients, terms, *, value=True, slope=False):
    """(The weights f(s) p(s) e^(-shift) of a tile, their slopes in s), each None unless asked for; terms are
    key_terms'."""
    return keyed_weights(scores, [c[:, None] for c in coefficients], terms, value=value, slope=slope)


def _keyed_moments(scores, dp, terms, degree):
    """Coefficient r's row sum of dp f(s) e^(-shift) s^r, r = j d + i. Chunk j starts from dp F (g s^d)^j g^a, steps
    by s, and meets the remaining g^b at each sum (a + b = n - j, b = ceil((n - j) / 2)): no factor underflows before
    the product does."""
    count, width, g, w, factor, _ = terms
    powers = [None, g]
    for _ in range(count - count // 2 - 1):
        powers.append(powers[-1] * g)
    scaled, lift, moments = dp if factor is None else dp * factor, None, []
    for j in range(count):
        a = (count - j) // 2
        product = scaled if lift is None else scaled * lift
        if a:
            product = product * powers[a]
        for i in range(width if j < count - 1 else degree + 1 - j * width):
            if i:
                product = product * scores
            moments.append(jnp.sum(product * powers[count - j - a], axis=1))
        lift = w if lift is None else lift * w
    return moments


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
        output, previous, *top = carry
        ki = tile * bk + jnp.arange(bk)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores, causal = _scores(q, k, qi, ki, time, dq)
        if prefactor == "none":
            z, _ = _transformed(scores, transform)
            shift = jnp.maximum(previous, _row_shift(z, degree))
            half, _, horner = _row_units(shift, coefficients)
            weights, _ = _polynomial(z, half, horner)
        else:
            # The units follow the row's largest causal score so far, which the carry holds; they never decrease.
            top = [_row_top(scores, causal, *top)]
            shift = jnp.maximum(previous, key_shift(*top, prefactor, degree))
            weights, _ = _keyed(scores, coefficients, key_terms(scores, shift[:, None], prefactor, degree))
        weights = jnp.where(causal, weights, 0.0)
        output = output * jnp.exp(previous - shift)[:, None] + _dot(weights, v)
        return output, shift, *top

    end = lax.div((pl.program_id(0) + 1) * bq + bk - 1, bk)
    start = (jnp.zeros((bq, v_ref.shape[-1]), jnp.float32), jnp.zeros((bq,), jnp.float32))
    if prefactor != "none":
        start = (*start, jnp.full((bq,), NO_SCORE, jnp.float32))
    output, shift, *_ = lax.fori_loop(0, end, body, start)
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
        terms = key_terms(scores, shift[:, None], prefactor, degree)
        _, slope = _keyed(scores, coefficients, terms, value=False, slope=True)
        dp = jnp.where(causal, _cotangent_dot(do, v), 0.0)
        grad_q += _dot(dp * slope, k) * (1.0 / math.sqrt(dq))
        return grad_q, tuple(moment + update for moment, update in
                             zip(moments, _keyed_moments(scores, dp, terms, degree), strict=True))

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
            weights, slope = _keyed(scores, coefficients,
                                    key_terms(scores, shift[:, None], prefactor, len(coefficients) - 1), slope=True)
            ds = jnp.where(causal, dp * slope, 0.0)
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
    if prefactor not in PREFACTORS or (prefactor != "none" and transform != "identity"):
        raise ValueError(f"The shifted polynomial takes the prefactors {PREFACTORS}, of the raw score with z = s")
    rows = tuple(coefficients[..., r] for r in range(coefficients.shape[-1]))
    return _attention(q, k, v, rows, transform, prefactor, interpret)
