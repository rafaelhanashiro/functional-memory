"""Tiled normalized learned polynomial attention and its CUDA reverse pass.

``direct`` learns every coefficient of p(s); signed weights and signed nonzero
row sums are valid. Its two-pass forward first computes final row sums, then
recomputes score tiles to accumulate p(s)/sum(p(s)) times V. ``rational`` takes
numerator coefficients a_0..a_m followed by the denominator's parameters: b_1..b_d
of 1 + sum_r b_r s^r (``direct``), or u_1..u_k, w_1..w_k of
(1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2 with d = 2k (``sum_of_squares``). The
first pass sums p_b(s), the second accumulates p_a(s)/sum(p_b(s)) times V, and the
reverse pass carries the numerator sensitivity (dY . V) / Z and the denominator
sensitivity -(dY . Y) / Z separately through their Horner recurrences.
``sum_of_squares`` evaluates
q1(s)^2 + q2(s)^2 directly, preserving nonnegativity even near shared roots,
and uses a single-pass online normalized accumulation. No path stores a
sequence-square array. Individual zero SOS weights are valid; a zero or
nonfinite row sum, or a nonfinite causal score/kernel, poisons the row.

All polynomial arithmetic, normalizers, outputs retained for differentiation,
and parameter-gradient partial sums use FP32. Products use the shared tensor
core policy: TF32x3 for FP32 and a three-component decomposition for mixed
FP32/BF16. The dQ program accumulates parameter-gradient partials, reduced over
query tiles and batch afterwards without atomics. Parameters remain per-head
FP32 arrays; broadcasting and reductions for parameter sharing belong outside
this backend.
"""

from functools import partial
import math
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from scaffolds.transformer.attention.backends.fused_taylor import (
    NUM_STAGES, NUM_WARPS, _dot, _full_spec, _load, _norm_spec, _spec, _tile_sizes,
)
from scaffolds.transformer.attention.backends.fused_learned_polynomial import (
    _coefficient_spec, _columns, _values,
)


PARAMETERIZATIONS = ("direct", "sum_of_squares", "rational")
DENOMINATOR_FORMS = ("direct", "sum_of_squares")


class _Layout(NamedTuple):
    """Static description of each head's parameter vector."""

    parameterization: str
    numerator_count: int
    denominator_form: str | None = None


def _split(parameters, layout):
    """(numerator, denominator factors); the denominator is None unless rational.

    A direct denominator is one factor (1, b_1, ..., b_d) evaluated by Horner's rule;
    a sum-of-squares denominator is two, (1, u_1, ..., u_k) and (0, w_1, ..., w_k),
    whose squares add.
    """
    numerator = parameters[:layout.numerator_count]
    if layout.parameterization != "rational":
        return numerator, None
    rest = parameters[layout.numerator_count:]
    if layout.denominator_form == "direct":
        return numerator, ((1.0, *rest),)
    half = len(rest) // 2
    return numerator, ((1.0, *rest[:half]), (0.0, *rest[half:]))


def _denominator_values(scores, denominator):
    if len(denominator) == 1:
        return _values(scores, denominator[0])
    first, second = (_values(scores, factor) for factor in denominator)
    return first * first + second * second


def _polynomial(scores, parameters, parameterization):
    if parameterization != "sum_of_squares":
        return _values(scores, parameters)
    split = (len(parameters) + 1) // 2
    first = _values(scores, parameters[:split])
    second = _values(scores, parameters[split:])
    return first * first + second * second


def _horner_and_scaled_derivative(scores, parameters, sensitivity):
    # Reverse the Horner recurrence with its normalized cotangent. Visiting
    # low powers first avoids both an overflowing raw p'(s) and underflow of
    # dp * c_m before later powers of s could bring that product into range.
    value = jnp.zeros_like(scores) + parameters[-1]
    intermediates = []
    for parameter in parameters[-2::-1]:
        intermediates.append(value)
        value = value * scores + parameter
    derivative = jnp.zeros_like(scores)
    for intermediate in intermediates[::-1]:
        derivative += sensitivity * intermediate
        sensitivity = sensitivity * scores
    return value, derivative


def _polynomial_and_scaled_derivative(scores, parameters, parameterization, sensitivity):
    if parameterization == "direct":
        value, derivative = _horner_and_scaled_derivative(scores, parameters, sensitivity)
        return value, derivative, None, None
    split = (len(parameters) + 1) // 2
    first = _values(scores, parameters[:split])
    second = _values(scores, parameters[split:])
    # Scale each factor before differentiating its square. Forming either
    # q*q' or 2*(q*q') first can overflow even for valid forward values.
    _, dfirst = _horner_and_scaled_derivative(scores, parameters[:split], (sensitivity * first) * 2.0)
    _, dsecond = _horner_and_scaled_derivative(scores, parameters[split:], (sensitivity * second) * 2.0)
    return first * first + second * second, dfirst + dsecond, first, second


def _tile_sensitivities(safe_scores, parameters, layout, dot_do_v, delta, normalizer):
    """Kernel values, the score cotangent, and the parameter sensitivities of one tile.

    With one polynomial the normalized cotangent is dp = (dY . V - dY . Y) / Z and
    every parameter's sensitivity is dp times a power of s (times a scaled factor for
    sum of squares). A rational kernel differentiates the numerator with
    (dY . V) / Z and the denominator with -(dY . Y) / Z: the output's dependence
    on Z enters only through the denominator polynomial.

    The sensitivities are returned as (start, length) segments in parameter order:
    the r-th parameter of a segment has sensitivity start * s^r.
    """
    numerator, denominator = _split(parameters, layout)
    if denominator is None:
        dp = (dot_do_v - delta[:, None]) / normalizer[:, None]
        values, ds, first, second = _polynomial_and_scaled_derivative(
            safe_scores, numerator, layout.parameterization, dp)
        if layout.parameterization == "direct":
            return values, ds, ((dp, len(numerator)),)
        split = (len(numerator) + 1) // 2
        return values, ds, (((dp * first) * 2.0, split), ((dp * second) * 2.0, len(numerator) - split))
    dp = dot_do_v / normalizer[:, None]
    dz = jnp.broadcast_to((-delta / normalizer)[:, None], dp.shape)
    values, ds = _horner_and_scaled_derivative(safe_scores, numerator, dp)
    segments = [(dp, len(numerator))]
    if len(denominator) == 1:
        _, dz_ds = _horner_and_scaled_derivative(safe_scores, denominator[0], dz)
        # Denominator parameters start at b_1, whose sensitivity is dz * s.
        segments.append((dz * safe_scores, len(denominator[0]) - 1))
    else:
        first, second = (_values(safe_scores, factor) for factor in denominator)
        # Scale each factor before differentiating its square, as for SOS kernels.
        scaled_first, scaled_second = (dz * first) * 2.0, (dz * second) * 2.0
        _, dfirst = _horner_and_scaled_derivative(safe_scores, denominator[0], scaled_first)
        _, dsecond = _horner_and_scaled_derivative(safe_scores, denominator[1], scaled_second)
        dz_ds = dfirst + dsecond
        # u_r and w_r start at r = 1: their sensitivities are 2 dz q1 s^r and 2 dz q2 s^r.
        segments += [(scaled_first * safe_scores, len(denominator[0]) - 1),
                     (scaled_second * safe_scores, len(denominator[1]) - 1)]
    return values, ds + dz_ds, tuple(segments)


def _forward_kernel(q_ref, k_ref, v_ref, *refs, count, layout, time, dq, dv, block_q, block_k):
    parameters = tuple(ref[...] for ref in refs[:count])
    numerator, denominator = _split(parameters, layout)
    parameterization = layout.parameterization
    out_ref, norm_ref = refs[count:]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)

    def numerator_values(scores):
        return _polynomial(scores, numerator, parameterization)

    def denominator_values(scores):
        return numerator_values(scores) if denominator is None else _denominator_values(scores, denominator)

    def kernel_tile(tile, evaluate):
        ki = tile * block_k + jnp.arange(block_k)
        k = _load(k_ref, ki, dq, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        values = evaluate(jnp.where(causal, scores, 0.0))
        invalid_entries = ~jnp.isfinite(scores) | ~jnp.isfinite(values)
        if parameterization == "sum_of_squares":
            invalid_entries |= values < 0
        invalid = jnp.max((causal & invalid_entries).astype(jnp.int32), axis=1) != 0
        return ki, jnp.where(causal, values, 0.0), invalid

    zeros = jnp.zeros((block_q,), jnp.float32)
    invalid = jnp.zeros((block_q,), jnp.bool_)
    output = jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32)
    if parameterization != "sum_of_squares":
        # Dividing by intermediate signed sums would fail for zero partial
        # sums, even when the final row sum is nonzero.
        def denominator_body(tile, carry):
            normalizer, invalid = carry
            _, values, invalid_entries = kernel_tile(tile, denominator_values)
            normalizer += jnp.sum(values, axis=1)
            return normalizer, invalid | invalid_entries | ~jnp.isfinite(normalizer)

        normalizer, invalid = lax.fori_loop(0, end, denominator_body, (zeros, invalid))
        invalid |= ~jnp.isfinite(normalizer) | (normalizer == 0)
        normalizer = jnp.where(invalid, jnp.nan, normalizer)

        def output_body(tile, carry):
            output, invalid = carry
            ki, values, invalid_entries = kernel_tile(tile, numerator_values)
            v = _load(v_ref, ki, dv, time)
            return output + _dot(values / normalizer[:, None], v), invalid | invalid_entries

        output, invalid = lax.fori_loop(0, end, output_body, (output, invalid))
    else:
        def online_body(tile, carry):
            output, normalizer, invalid = carry
            ki, values, invalid_entries = kernel_tile(tile, numerator_values)
            v = _load(v_ref, ki, dv, time)
            updated = normalizer + jnp.sum(values, axis=1)
            # An all-zero prefix contributes zero until a positive weight
            # arrives. This is an internal guard, never a final-row epsilon.
            denominator = jnp.where(updated > 0, updated, 1.0)
            output = output * (normalizer / denominator)[:, None]
            output += _dot(values / denominator[:, None], v)
            return output, updated, invalid | invalid_entries

        output, normalizer, invalid = lax.fori_loop(0, end, online_body, (output, zeros, invalid))
        invalid |= ~jnp.isfinite(normalizer) | (normalizer <= 0)

    # A failed output invalidates the whole row. Check the public output dtype
    # too: a finite FP32 accumulator can overflow when returned in BF16.
    invalid |= jnp.max((~jnp.isfinite(output.astype(v_ref.dtype))).astype(jnp.int32), axis=1) != 0
    output = jnp.where(invalid[:, None], jnp.nan, output)
    normalizer = jnp.where(invalid, jnp.nan, normalizer)
    plt.store(out_ref, output, mask=(qi[:, None] < time) & (jnp.arange(out_ref.shape[-1])[None, :] < dv))
    plt.store(norm_ref, normalizer, mask=qi < time)


def _dq_parameter_kernel(q_ref, k_ref, v_ref, *refs, count, layout, time, dq, dv, block_q, block_k):
    parameters = tuple(ref[...] for ref in refs[:count])
    do_ref, out_ref, norm_ref, dq_ref = refs[count:count + 4]
    grad_refs = refs[count + 4:]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0)
    output = plt.load(out_ref, mask=vmask, other=0)
    normalizer = plt.load(norm_ref, mask=qi < time, other=1)
    delta = jnp.sum(do.astype(jnp.float32) * output, axis=1)

    def body(tile, carry):
        grad_q, grads = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        safe_scores = jnp.where(causal, scores, 0.0)
        _, ds, segments = _tile_sensitivities(safe_scores, parameters, layout, _dot(do, v.T), delta, normalizer)
        ds = jnp.where(causal, ds, 0.0)
        grad_q += _dot(ds, k) * (1.0 / math.sqrt(dq))
        if sum(length for _, length in segments) != count:
            raise AssertionError("Parameter sensitivities do not cover every parameter")
        # Carry each normalized sensitivity through the power recurrence.
        # Forming s**r first could overflow even when dp * s**r is finite.
        updated, remaining = [], iter(grads)
        for start, length in segments:
            sensitivity = jnp.where(causal, start, 0.0)
            for _ in range(length):
                updated.append(next(remaining) + jnp.sum(sensitivity))
                sensitivity = sensitivity * safe_scores
        return grad_q, tuple(updated)

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    grad_q, grads = lax.fori_loop(
        0, end, body,
        (jnp.zeros(q_ref.shape, jnp.float32), tuple(jnp.zeros((), jnp.float32) for _ in range(count))),
    )
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    for grad_ref, grad in zip(grad_refs, grads, strict=True):
        grad_ref[...] = grad


def _dkv_kernel(q_ref, k_ref, v_ref, *refs, count, layout, time, dq, dv, block_q, block_k):
    parameters = tuple(ref[...] for ref in refs[:count])
    do_ref, out_ref, norm_ref, dk_ref, dv_ref = refs[count:count + 5]
    ki = pl.program_id(0) * block_k + jnp.arange(block_k)
    kmask = (ki[:, None] < time) & (jnp.arange(k_ref.shape[-1])[None, :] < dq)
    vmask = (ki[:, None] < time) & (jnp.arange(v_ref.shape[-1])[None, :] < dv)
    k = plt.load(k_ref, mask=kmask, other=0)
    v = plt.load(v_ref, mask=vmask, other=0)
    initial = (jnp.zeros(k_ref.shape, jnp.float32), jnp.zeros(v_ref.shape, jnp.float32))

    def body(tile, carry):
        grad_k, grad_v = carry
        qi = tile * block_q + jnp.arange(block_q)
        q = _load(q_ref, qi, dq, time)
        do = _load(do_ref, qi, dv, time)
        output = _load(out_ref, qi, dv, time)
        normalizer = plt.load(norm_ref.at[qi], mask=qi < time, other=1)
        delta = jnp.sum(do.astype(jnp.float32) * output, axis=1)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        values, ds, _ = _tile_sensitivities(
            jnp.where(causal, scores, 0.0), parameters, layout, _dot(do, v.T), delta, normalizer)
        weights = jnp.where(causal, values / normalizer[:, None], 0.0)
        grad_v += _dot(weights.T, do)
        ds = jnp.where(causal, ds, 0.0)
        grad_k += _dot(ds.T, q) * (1.0 / math.sqrt(dq))
        return grad_k, grad_v

    start = lax.div(pl.program_id(0) * block_k, block_q)
    grad_k, grad_v = lax.fori_loop(start, pl.cdiv(time, block_q), body, initial)
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


def _name(layout, count):
    name = f"normalized_polynomial_{layout.parameterization}_{layout.numerator_count - 1}"
    if layout.parameterization == "rational":
        name += f"_{layout.denominator_form}_{count - layout.numerator_count}"
    return name


def _forward(q, k, v, parameters, layout, interpret):
    batch, time, heads, dq = q.shape
    dv, count = v.shape[-1], parameters.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    return pl.pallas_call(
        partial(_forward_kernel, count=count, layout=layout, time=time,
                dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(v.shape, jnp.float32),
                   jax.ShapeDtypeStruct((batch, time, heads), jnp.float32)),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(),) * count),
        out_specs=(_spec(block_q, dv), _norm_spec(block_q)),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        interpret=interpret, name=f"{_name(layout, count)}_forward",
    )(q, k, v, *_columns(parameters))


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _attention(q, k, v, parameters, layout, interpret):
    output, _ = _forward(q, k, v, parameters, layout, interpret)
    return output.astype(v.dtype)


def _attention_fwd(q, k, v, parameters, layout, interpret):
    output, norm = _forward(q, k, v, parameters, layout, interpret)
    return output.astype(v.dtype), (q, k, v, parameters, output, norm)


def _attention_bwd(layout, interpret, residuals, do):
    q, k, v, parameters, output, norm = residuals
    batch, time, heads, dq = q.shape
    dv, count = v.shape[-1], parameters.shape[-1]
    block_q, block_k = _tile_sizes(dq, dv)
    query_tiles = pl.cdiv(time, block_q)
    compiler = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    columns = _columns(parameters)
    kernel_args = dict(count=count, layout=layout, time=time,
                       dq=dq, dv=dv, block_q=block_q, block_k=block_k)
    grad_q, *partials = pl.pallas_call(
        partial(_dq_parameter_kernel, **kernel_args),
        out_shape=(jax.ShapeDtypeStruct(q.shape, q.dtype),
                   *(jax.ShapeDtypeStruct((query_tiles, batch, heads), jnp.float32),) * count),
        grid=(query_tiles, batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(),) * count, _spec(block_q, dv), _spec(block_q, dv), _norm_spec(block_q)),
        out_specs=(_spec(block_q, dq), *(pl.BlockSpec((None, None, None), lambda i, b, h: (i, b, h)),) * count),
        compiler_params=compiler, interpret=interpret,
        name=f"{_name(layout, count)}_dq_parameters",
    )(q, k, v, *columns, do, output, norm)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, **kernel_args),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  *(_coefficient_spec(),) * count, _full_spec(time, dv), _full_spec(time, dv),
                  pl.BlockSpec((None, time, None), lambda i, b, h: (b, 0, h))),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)), compiler_params=compiler, interpret=interpret,
        name=f"{_name(layout, count)}_dkv",
    )(q, k, v, *columns, do, output, norm)
    grad_parameters = jnp.stack([jnp.sum(part, axis=(0, 1), dtype=jnp.float32) for part in partials], axis=-1)
    return grad_q, grad_k, grad_v, grad_parameters


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_normalized_polynomial_attention(q, k, v, parameters, *, parameterization="direct",
                                          denominator_degree=None, denominator_form="direct", interpret=False):
    """Causal row-normalized p_h(QK^T / sqrt(d_k))V, with learned FP32 parameters.

    Parameters have shape [heads, n]. ``direct`` supplies ascending coefficients
    c_0..c_m (n = m+1, m >= 1). ``sum_of_squares`` requires even m >= 2 and supplies
    ascending q1 coefficients (m/2+1), followed by q2 coefficients (m/2), for q1^2+q2^2.
    ``rational`` supplies ascending numerator coefficients a_0..a_m followed by the
    denominator's ``denominator_degree`` = d parameters, for p_a(s) / sum_j p_b(s_j):
    b_1..b_d of p_b(s) = 1 + sum_r b_r s^r with ``denominator_form="direct"``, or, for
    even d = 2k, u_1..u_k then w_1..w_k of p_b(s) = (1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2
    with ``denominator_form="sum_of_squares"``. ``denominator_degree=None`` means d = m.
    Q/K/V use [batch, time, heads, channels], a common FP32/BF16 dtype, and
    widths up to 128. Compiled execution requires Ampere-or-newer NVIDIA GPUs;
    ``interpret=True`` runs the Pallas programs on CPU for tests.
    """
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError("parameterization must be 'direct', 'sum_of_squares', or 'rational'")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must have shape [batch, time, heads, channels]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K shapes and Q/K/V batch, time, and head dimensions must match")
    if min(*q.shape, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    if max(q.shape[-1], v.shape[-1]) > 128:
        raise ValueError("Fused attention supports Q/K and value head dimensions up to 128")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("Fused attention requires a common float32 or bfloat16 dtype")
    if (parameters.ndim != 2 or parameters.shape[0] != q.shape[2] or parameters.shape[1] < 2
            or parameters.dtype != jnp.float32):
        raise ValueError("Polynomial parameters must be float32 with shape [heads, m+1], m >= 1")
    count = parameters.shape[-1]
    if parameterization != "rational":
        if denominator_degree is not None or denominator_form != "direct":
            raise ValueError("Denominator options apply to the rational parameterization only")
        if parameterization == "sum_of_squares" and (count < 3 or count % 2 == 0):
            raise ValueError("A sum-of-squares polynomial requires an even degree >= 2")
        layout = _Layout(parameterization, count)
    else:
        if denominator_form not in DENOMINATOR_FORMS:
            raise ValueError("denominator_form must be 'direct' or 'sum_of_squares'")
        if denominator_degree is None:
            if count < 3 or count % 2 == 0:
                raise ValueError("A rational kernel requires 2m+1 parameters (a_0..a_m, b_1..b_m) with m >= 1")
            denominator_degree = (count - 1) // 2
        if (isinstance(denominator_degree, bool) or not isinstance(denominator_degree, int)
                or denominator_degree < 1 or count - denominator_degree < 2):
            raise ValueError("A rational kernel needs numerator degree >= 1 and denominator degree >= 1")
        if denominator_form == "sum_of_squares" and denominator_degree % 2:
            raise ValueError("A sum-of-squares denominator requires an even degree")
        layout = _Layout("rational", count - denominator_degree, denominator_form)
    return _attention(q, k, v, parameters, layout, bool(interpret))
