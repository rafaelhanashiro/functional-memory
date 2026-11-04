"""Fused causal score-MLP attention, never divided by a row sum.

For Y_t = sum_(i<=t) f_ti V_i with s_ti = Q_t . K_i / sqrt(d_K) and J units per query and
head (``attention/score_mlp.py``),

    z_tij = w_j s_ti + b_tj,    u_ti = sum_j a_tj SiLU(z_tij),    SiLU(z) = z sigmoid(z),
    kernel:        f_ti = u_ti
    exp_residual:  f_ti = exp(s_ti) (1 + u_ti).

Each Pallas/Triton program owns a query tile (forward; dQ and the unit gradients) or a key
tile (dK, dV) and visits only causal tiles. Scores, units, and weights stay on chip; no
sequence-square array or per-unit tile is stored. Scores, units, and accumulators are FP32,
and exp is unshifted. The per-query a_tj, b_tj and the per-head w_j (broadcast over queries)
enter as FP32 columns, one per unit, like the query-polynomial coefficients. Products follow
``fused_unnormalized._dot``.

The custom VJP retains only Q, K, V, a, b, w and recomputes every unit. With dY the output
cotangent, dP_ti = dY_t . V_i on causal pairs (zero elsewhere), and
SiLU'(z) = sigmoid(z) (1 + z (1 - sigmoid(z))):

    G_ti  = dL/du_ti  = dP_ti                    (kernel)
                      = dP_ti exp(s_ti)          (exp_residual)
    D_tij = dL/dz_tij = G_ti a_tj SiLU'(z_tij)
    da_tj = sum_(i<=t) G_ti SiLU(z_tij)          db_tj = sum_(i<=t) D_tij
    dw_j  = sum_(b,t) sum_(i<=t) D_tij s_ti      (per-query partials, summed over batch and time in JAX)
    dS_ti = dL/ds_ti  = sum_j w_j D_tij          (kernel)
                      = sum_j w_j D_tij + dP_ti f_ti
                        (exp_residual: d/ds [exp(s) (1 + u)] = f + exp(s) du/ds, and dP exp(s) du/ds
                         = sum_j w_j D_tij)
    dQ_t = sum_(i<=t) dS_ti K_i / sqrt(d_K)      dK_i = sum_(t>=i) dS_ti Q_t / sqrt(d_K)
    dV_i = sum_(t>=i) f_ti dY_t

The query pass computes dQ, da, db, and the dw partials; the key pass dK and dV.
"""

from functools import partial
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from scaffolds.transformer.attention.backends.fused_query_polynomial import (
    _cast_output, _coefficient_spec, _columns, _full_coefficient_spec,
)
from scaffolds.transformer.attention.backends.fused_unnormalized import (
    NUM_STAGES, NUM_WARPS, _dot, _full_spec, _load, _spec,
)
from scaffolds.transformer.attention.score_mlp import FORMS

# Each unit keeps several FP32 score tiles live: at 64x64 the J = 8 backward ran about 4x slower on
# H100 than at 32x32 (validation/README.md), so all three programs use 32x32 tiles at every width.
BLOCK_Q = 32
BLOCK_K = 32


def _activation(pre, *, derivative=False):
    """SiLU(z) = z sigmoid(z) in FP32; with ``derivative`` also SiLU'(z) = sigmoid(z) (1 + z (1 - sigmoid(z)))."""
    gate = jax.nn.sigmoid(pre)
    value = pre * gate
    return (value, gate * (1.0 + pre * (1.0 - gate))) if derivative else value


def _units(columns, width):
    """(a_j, b_j, w_j) [block] per unit from the 3J columns: amplitudes, offsets, then slopes."""
    return tuple(zip(columns[:width], columns[width:2 * width], columns[2 * width:], strict=True))


def _values(scores, units, form):
    """f on a tile of masked FP32 scores."""
    total = jnp.zeros_like(scores)
    for amplitude, offset, slope in units:
        total += amplitude[:, None] * _activation(slope[:, None] * scores + offset[:, None])
    return total if form == "kernel" else jnp.exp(scores) * (1.0 + total)


def _unit_terms(scores, cotangent, units):
    """Per unit: a_j, w_j, SiLU(z_j), and D_j = dL/dz_j = G a_j SiLU'(z_j) for the unit cotangent G."""
    for amplitude, offset, slope in units:
        value, derivative = _activation(slope[:, None] * scores + offset[:, None], derivative=True)
        yield amplitude, slope, value, cotangent * amplitude[:, None] * derivative


def _forward_kernel(q_ref, k_ref, v_ref, *refs, form, width, time, dq, dv, block_q, block_k):
    unit_refs, out_ref = refs[:3 * width], refs[3 * width]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = plt.load(q_ref, mask=(qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq), other=0)
    units = _units(tuple(plt.load(ref, mask=qi < time, other=0) for ref in unit_refs), width)
    initial = (jnp.zeros((block_q, v_ref.shape[-1]), jnp.float32), jnp.zeros((block_q,), jnp.bool_))

    def body(tile, carry):
        output, invalid = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        # Masked entries are evaluated at score zero, never at sentinels, and zeroed below.
        values = _values(jnp.where(causal, scores, 0.0), units, form)
        invalid_entries = causal & (~jnp.isfinite(scores) | ~jnp.isfinite(values))
        invalid |= jnp.max(invalid_entries.astype(jnp.int32), axis=1) != 0
        output += _dot(jnp.where(causal, values, 0.0), v)
        return output, invalid

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    output, invalid = lax.fori_loop(0, end, body, initial)
    output = jnp.where(invalid[:, None], jnp.nan, output)
    plt.store(out_ref, output, mask=(qi[:, None] < time) & (jnp.arange(out_ref.shape[-1])[None, :] < dv))


def _dq_unit_kernel(q_ref, k_ref, v_ref, *refs, form, width, time, dq, dv, block_q, block_k):
    unit_refs = refs[:3 * width]
    do_ref, dq_ref = refs[3 * width], refs[3 * width + 1]
    grad_refs = refs[3 * width + 2:]
    qi = pl.program_id(0) * block_q + jnp.arange(block_q)
    qmask = (qi[:, None] < time) & (jnp.arange(q_ref.shape[-1])[None, :] < dq)
    vmask = (qi[:, None] < time) & (jnp.arange(do_ref.shape[-1])[None, :] < dv)
    q = plt.load(q_ref, mask=qmask, other=0)
    do = plt.load(do_ref, mask=vmask, other=0)
    units = _units(tuple(plt.load(ref, mask=qi < time, other=0) for ref in unit_refs), width)

    def body(tile, carry):
        grad_q, grads = carry
        ki = tile * block_k + jnp.arange(block_k)
        k, v = _load(k_ref, ki, dq, time), _load(v_ref, ki, dv, time)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        safe_scores = jnp.where(causal, scores, 0.0)
        dp = jnp.where(causal, _dot(do, v.T), 0.0)
        growth = jnp.exp(safe_scores) if form == "exp_residual" else None
        cotangent = dp if growth is None else dp * growth
        dscores, total, updated = jnp.zeros_like(scores), jnp.zeros_like(scores), []
        for (amplitude, slope, value, dpre), (grad_a, grad_b, grad_w) in zip(
                _unit_terms(safe_scores, cotangent, units), grads, strict=True):
            dscores += dpre * slope[:, None]
            if growth is not None:
                total += amplitude[:, None] * value
            updated.append((grad_a + jnp.sum(cotangent * value, axis=1), grad_b + jnp.sum(dpre, axis=1),
                            grad_w + jnp.sum(dpre * safe_scores, axis=1)))
        if growth is not None:
            dscores += dp * (growth * (1.0 + total))
        grad_q += _dot(dscores, k) * (1.0 / math.sqrt(dq))
        return grad_q, tuple(updated)

    end = lax.div((pl.program_id(0) + 1) * block_q + block_k - 1, block_k)
    zero = jnp.zeros((block_q,), jnp.float32)
    grad_q, grads = lax.fori_loop(0, end, body, (jnp.zeros(q_ref.shape, jnp.float32), ((zero,) * 3,) * width))
    plt.store(dq_ref, grad_q.astype(dq_ref.dtype), mask=qmask)
    # Outputs are every da_j, then every db_j, then every per-query dw_j partial.
    for grad_ref, grad in zip(grad_refs, (*(g[0] for g in grads), *(g[1] for g in grads), *(g[2] for g in grads)),
                              strict=True):
        plt.store(grad_ref, grad, mask=qi < time)


def _dkv_kernel(q_ref, k_ref, v_ref, *refs, form, width, time, dq, dv, block_q, block_k):
    unit_refs = refs[:3 * width]
    do_ref, dk_ref, dv_ref = refs[3 * width:3 * width + 3]
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
        units = _units(tuple(plt.load(ref.at[qi], mask=qi < time, other=0) for ref in unit_refs), width)
        scores = _dot(q, k.T) * (1.0 / math.sqrt(dq))
        causal = (qi[:, None] >= ki[None, :]) & (qi[:, None] < time) & (ki[None, :] < time)
        safe_scores = jnp.where(causal, scores, 0.0)
        dp = jnp.where(causal, _dot(do, v.T), 0.0)
        growth = jnp.exp(safe_scores) if form == "exp_residual" else None
        cotangent = dp if growth is None else dp * growth
        dscores, total = jnp.zeros_like(scores), jnp.zeros_like(scores)
        for amplitude, slope, value, dpre in _unit_terms(safe_scores, cotangent, units):
            dscores += dpre * slope[:, None]
            total += amplitude[:, None] * value
        values = total if growth is None else growth * (1.0 + total)
        if growth is not None:
            dscores += dp * values
        grad_v += _dot(jnp.where(causal, values, 0.0).T, do)
        grad_k += _dot(dscores.T, q) * (1.0 / math.sqrt(dq))
        return grad_k, grad_v

    start = lax.div(pl.program_id(0) * block_k, block_q)
    grad_k, grad_v = lax.fori_loop(start, pl.cdiv(time, block_q), body, initial)
    plt.store(dk_ref, grad_k.astype(dk_ref.dtype), mask=kmask)
    plt.store(dv_ref, grad_v.astype(dv_ref.dtype), mask=vmask)


def _unit_columns(amplitude, offset, slope):
    """3J FP32 [batch, time, heads] columns: every a_j, every b_j, then every w_j broadcast over batch and time."""
    return (*_columns(amplitude), *_columns(offset), *_columns(jnp.broadcast_to(slope, amplitude.shape)))


def _forward(q, k, v, amplitude, offset, slope, form, interpret):
    batch, time, heads, dq = q.shape
    dv, width = v.shape[-1], amplitude.shape[-1]
    block_q, block_k = BLOCK_Q, BLOCK_K
    return pl.pallas_call(
        partial(_forward_kernel, form=form, width=width, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct(v.shape, jnp.float32),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(block_q),) * (3 * width)),
        out_specs=_spec(block_q, dv),
        compiler_params=plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        interpret=interpret, name=f"score_mlp_{form}_{width}_unnormalized_forward",
    )(q, k, v, *_unit_columns(amplitude, offset, slope))


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _attention(q, k, v, amplitude, offset, slope, form, interpret):
    return _cast_output(_forward(q, k, v, amplitude, offset, slope, form, interpret), v.dtype)


def _attention_fwd(q, k, v, amplitude, offset, slope, form, interpret):
    output = _cast_output(_forward(q, k, v, amplitude, offset, slope, form, interpret), v.dtype)
    return output, (q, k, v, amplitude, offset, slope)


def _attention_bwd(form, interpret, residuals, do):
    q, k, v, amplitude, offset, slope = residuals
    batch, time, heads, dq = q.shape
    dv, width = v.shape[-1], amplitude.shape[-1]
    block_q, block_k = BLOCK_Q, BLOCK_K
    compiler = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    columns = _unit_columns(amplitude, offset, slope)
    grad_q, *grad_columns = pl.pallas_call(
        partial(_dq_unit_kernel, form=form, width=width, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(q.shape, q.dtype),
                   *(jax.ShapeDtypeStruct(amplitude.shape[:3], jnp.float32),) * (3 * width)),
        grid=(pl.cdiv(time, block_q), batch, heads),
        in_specs=(_spec(block_q, dq), _full_spec(time, dq), _full_spec(time, dv),
                  *(_coefficient_spec(block_q),) * (3 * width), _spec(block_q, dv)),
        out_specs=(_spec(block_q, dq), *(_coefficient_spec(block_q),) * (3 * width)),
        compiler_params=compiler, interpret=interpret, name=f"score_mlp_{form}_{width}_unnormalized_dq_units",
    )(q, k, v, *columns, do)
    grad_k, grad_v = pl.pallas_call(
        partial(_dkv_kernel, form=form, width=width, time=time, dq=dq, dv=dv, block_q=block_q, block_k=block_k),
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(pl.cdiv(time, block_k), batch, heads),
        in_specs=(_full_spec(time, dq), _spec(block_k, dq), _spec(block_k, dv),
                  *(_full_coefficient_spec(time),) * (3 * width), _full_spec(time, dv)),
        out_specs=(_spec(block_k, dq), _spec(block_k, dv)),
        compiler_params=compiler, interpret=interpret, name=f"score_mlp_{form}_{width}_unnormalized_dkv",
    )(q, k, v, *columns, do)
    grad_amplitude, grad_offset, slope_partials = (
        jnp.stack(grad_columns[group * width:(group + 1) * width], axis=-1) for group in range(3))
    # w is shared by every query of a head: its gradient sums the per-query partials.
    return grad_q, grad_k, grad_v, grad_amplitude, grad_offset, jnp.sum(slope_partials, axis=(0, 1))


_attention.defvjp(_attention_fwd, _attention_bwd)


def fused_score_mlp_attention(q, k, v, amplitude, offset, slope, *, form, interpret=False):
    """Unnormalized causal sum_(i<=t) f(s_ti) V_i of a score MLP with a, b [B,T,H,J] and w [H,J].

    ``form`` is ``kernel`` (f = sum_j a_tj SiLU(w_j s + b_tj)) or ``exp_residual``
    (f = exp(s) (1 + that sum)). Q/K/V share FP32 or BF16 dtype and head widths up to 128;
    a, b, and w are FP32 with J >= 1. Arithmetic and the a, b, w gradients are FP32; the
    output and Q/K/V gradients have their operand dtypes. Nonfinite causal scores or kernel
    values poison the complete output row, even if V is zero. ``interpret=True`` runs the
    Pallas CPU interpreter for qualification.
    """
    if form not in FORMS:
        raise ValueError(f"Unknown score-MLP form: {form}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must have shape [batch,time,heads,channels]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K shapes and Q/K/V batch, time, and head dimensions must match")
    if min(*q.shape, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    if max(q.shape[-1], v.shape[-1]) > 128:
        raise ValueError("Fused attention supports Q/K and value head dimensions up to 128")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.float32, jnp.bfloat16):
        raise ValueError("Fused attention requires a common float32 or bfloat16 dtype")
    if (amplitude.ndim != 4 or amplitude.shape[:3] != q.shape[:3] or amplitude.shape[-1] < 1
            or offset.shape != amplitude.shape or amplitude.dtype != jnp.float32 or offset.dtype != jnp.float32):
        raise ValueError("Score-MLP amplitudes and offsets must be float32 with shape [batch,time,heads,J], J>=1")
    if slope.shape != (q.shape[2], amplitude.shape[-1]) or slope.dtype != jnp.float32:
        raise ValueError("Score-MLP slopes must be float32 with shape [heads,J]")
    return _attention(q, k, v, amplitude, offset, slope, form, interpret)
