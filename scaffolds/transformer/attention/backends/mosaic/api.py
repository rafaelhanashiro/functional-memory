"""Causal polynomial attention on Mosaic GPU kernels (Hopper and Blackwell).

Two operations, each with the mathematics and failure behavior of its Pallas-Triton
counterpart. FP32 scores, polynomials, sums, and accumulators. FP32 weights and score
gradients enter the tensor cores at ``precision``: ``"bf16"`` rounds them once to BF16, as
FlashAttention and cuDNN do for probabilities and score gradients, and the Taylor backward
reads the BF16 output, as theirs does; ``"bf16x3"`` splits them into three BF16 components
whose sum is the FP32 value and keeps the FP32 output, as the Triton kernels do. Each setup
defaults to ``default_precision``.

- ``taylor_attention`` (``fused_taylor.fused_polynomial_attention``): normalized fixed
  Taylor T_m. A row whose sum is nonfinite, nonpositive (even degree) or zero (odd degree),
  or whose output overflows, is NaN. The output is divided by its row sum once, after
  accumulation, so signed odd degrees need a single pass.
- ``query_polynomial_attention`` (``fused_query_polynomial``): unnormalized
  sum_r c_(t,r) s^r with FP32 coefficients for every query and head. A row with a
  nonfinite score or weight, or whose BF16 output overflows, is NaN; coefficient
  gradients are FP32 and stay per query.
- ``query_decay_attention`` and ``endpoint_decay_attention`` (the blocked reference's
  query-distance forget gates): the query polynomial times exp((t-i) l), with a per-query
  log rate l_t <= 0 or the endpoint rate l_ti = -exp(A_h) softplus(u_t + v_i). Decay is
  evaluated inside the causal tiles and its gradients are accumulated there: per-query sums
  in the dQ pass, per-key sums in the dK pass. An invalid gate makes the affected causal
  rows NaN; the diagonal is never attenuated.

Supported: BF16 Q/K/V of shape [batch, time, heads, channels], Q/K width 8, 16, 32, 64, or
128 (width 8 is zero-padded to the tensor cores' minimum of 16, which leaves every score
unchanged; narrower tiles use narrower shared-memory swizzles), value width 64, degree
>= 1 (query coefficients: degree <= 8), and time a multiple of 128 (H100) or 256 (B300).
Anything else, and any GPU other than compute capability 9.0 or 10.x, raises ValueError.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends.mosaic import common
from scaffolds.transformer.attention.backends.mosaic.common import (
    BF16, BF16X3, ENDPOINT_DECAY, PRECISIONS, QUERY, QUERY_DECAY, TAYLOR,
)

WIDTHS = (8, 16, 32, 64, 128)
VALUE_WIDTH = 64
MIN_WIDTH = 16  # BF16 tensor-core contraction step on Hopper and Blackwell
MAX_QUERY_DEGREE = 12  # largest qualified degree; B300 must also fit it (query_degree_supported)


# Per-query vectors a forget gate adds to the coefficients, and the shared memory its B300 dK pass
# adds, by gate conditioning: query (query_decay) or query_key (endpoint_decay).
DECAY_FAMILY = {"query": QUERY_DECAY, "query_key": ENDPOINT_DECAY}


def query_degree_supported(degree, width, time, *, gate=None):
    """Whether the query-coefficient kernels run degree m at this key width and sequence length
    on both GPUs: B300's dK/dV holds (m+1) FP32 coefficients (and a forget gate's vectors and
    buffers, for gate conditioning ``gate``) for every query in shared memory."""
    family = QUERY if gate is None else DECAY_FAMILY[gate]
    return (1 <= degree <= MAX_QUERY_DEGREE and time > 0 and common.blackwell_dkv_stages(
        max(width, MIN_WIDTH), VALUE_WIDTH, common.vector_counts(family, degree)[1], time,
        extra_bytes=common.dkv_extra_bytes(family, "dk")) is not None)


def default_precision(family: str, degree: int) -> str:
    """BF16 weight products, as standard attention kernels use, for every setup but normalized
    odd-degree Taylor. Its signed weights are divided by a row sum that can cancel, which
    amplifies BF16 rounding: with sharp scores its gradients err about twice as much as
    cuDNN softmax's, so it keeps FP32-precision weights."""
    return BF16X3 if family == TAYLOR and degree % 2 else BF16


def _kernels():
    capability = jax.devices()[0].compute_capability
    major = int(capability.split(".")[0])
    if capability == "9.0":
        from scaffolds.transformer.attention.backends.mosaic import hopper
        return "hopper", hopper
    if major == 10:
        from scaffolds.transformer.attention.backends.mosaic import blackwell
        return "blackwell", blackwell
    raise ValueError(f"Mosaic GPU attention supports compute capability 9.0 and 10.x, not {capability}")


def _check_operands(q, k, v, name):
    if q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"{name}: Q and K must share the shape [batch, time, heads, channels], with V matching to channels")
    if not q.dtype == k.dtype == v.dtype == jnp.bfloat16:
        raise ValueError(f"{name} requires BF16 Q/K/V, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.shape[-1] not in WIDTHS:
        raise ValueError(f"{name} supports Q/K widths {WIDTHS}, got {q.shape[-1]}")
    if v.shape[-1] != VALUE_WIDTH:
        raise ValueError(f"{name} supports value width {VALUE_WIDTH}, got {v.shape[-1]}")
    arch, _ = _kernels()
    multiple = 128 if arch == "hopper" else 256
    if q.shape[1] % multiple:
        raise ValueError(f"{name} on {arch} requires time to be a multiple of {multiple}, got {q.shape[1]}")


def _pad(x):
    width = x.shape[-1]
    return x if width >= MIN_WIDTH else jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, MIN_WIDTH - width)))


def _scale(q):
    return 1.0 / math.sqrt(q.shape[-1])


def _check_precision(precision):
    if precision not in PRECISIONS:
        raise ValueError(f"Mosaic attention precision must be one of {PRECISIONS}, got {precision!r}")


def _backward(family, degree, precision, q, k, v, do, vectors, key_vectors=None):
    """(dQ, per-query sums or None, dK, dV) at the unpadded width, and per-key sums for
    families that have them."""
    arch, kernels = _kernels()
    width = q.shape[-1]
    args = (_pad(q), _pad(k), v, do, vectors, key_vectors)
    options = dict(family=family, degree=degree, scale=_scale(q), precision=precision)
    key_sums = None
    if arch == "hopper":
        grad_q, sums = kernels.dq(*args, **options)
        grad_k, grad_v, *key_sums = kernels.dkv(*args, **options)
    else:
        stages = 2 if width > 64 else 3  # shared memory at width 128
        grad_q, sums = kernels.dq(*args, **options, stages=stages)
        dk_stages = stages
        if family in common.DECAY_FAMILIES:  # their extra vectors and buffers may need a shallower ring
            dk_stages = min(stages, common.blackwell_dkv_stages(
                max(width, MIN_WIDTH), v.shape[-1], common.vector_counts(family, degree)[1], q.shape[1],
                extra_bytes=common.dkv_extra_bytes(family, "dk")))
        grad_k = kernels.dkv(*args, which="dk", **options, stages=dk_stages)
        if isinstance(grad_k, tuple):
            grad_k, *key_sums = grad_k
        grad_v = kernels.dkv(*args, which="dv", **options)
    result = (grad_q[..., :width], sums, grad_k[..., :width], grad_v)
    return (*result, key_sums[0]) if key_sums else result


# ----------------------------------------------------------------------------------------
# Normalized fixed Taylor attention.

def _taylor_forward(q, k, v, degree, precision):
    _, kernels = _kernels()
    out, row_sum = kernels.forward(_pad(q), _pad(k), v, None, family=TAYLOR, degree=degree, scale=_scale(q),
                                   precision=precision)
    return out, row_sum  # row_sum is [batch, heads, time]


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _taylor(q, k, v, degree, precision):
    return _taylor_forward(q, k, v, degree, precision)[0].astype(v.dtype)


def _taylor_fwd(q, k, v, degree, precision):
    out, row_sum = _taylor_forward(q, k, v, degree, precision)
    output = out.astype(v.dtype)
    return output, (q, k, v, output if precision == BF16 else out, row_sum)


def _taylor_bwd(degree, precision, residuals, do):
    q, k, v, out, row_sum = residuals
    delta = jnp.sum(do.astype(jnp.float32) * out.astype(jnp.float32), axis=-1).transpose(0, 2, 1)
    vectors = jnp.stack([delta, 1.0 / row_sum])
    grad_q, _, grad_k, grad_v = _backward(TAYLOR, degree, precision, q, k, v, do, vectors)
    return grad_q, grad_k, grad_v


_taylor.defvjp(_taylor_fwd, _taylor_bwd)


def taylor_attention(q, k, v, *, attention_degree: int, precision: str | None = None):
    """Normalized causal T_m(QK^T / sqrt(d_K)) V for [batch, time, heads, channels] arrays."""
    degree = attention_degree
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError(f"Mosaic Taylor attention requires an integer degree >= 1, got {degree!r}")
    _check_operands(q, k, v, "Mosaic Taylor attention")
    precision = default_precision(TAYLOR, degree) if precision is None else precision
    _check_precision(precision)
    return _taylor(q, k, v, degree, precision)


# ----------------------------------------------------------------------------------------
# Unnormalized attention with per-query polynomial coefficients.

def _coefficient_vectors(coefficients):
    return coefficients.transpose(3, 0, 2, 1)  # [m+1, batch, heads, time]


def _cast_output(output, dtype):
    output = output.astype(dtype)  # as fused_query_polynomial: an overflowing row becomes NaN
    return jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)


def _query_forward(q, k, v, coefficients, precision):
    _, kernels = _kernels()
    degree = coefficients.shape[-1] - 1
    out, _ = kernels.forward(_pad(q), _pad(k), v, _coefficient_vectors(coefficients),
                             family=QUERY, degree=degree, scale=_scale(q), precision=precision)
    return _cast_output(out, v.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _query(q, k, v, coefficients, precision):
    return _query_forward(q, k, v, coefficients, precision)


def _query_fwd(q, k, v, coefficients, precision):
    return _query_forward(q, k, v, coefficients, precision), (q, k, v, coefficients)


def _query_bwd(precision, residuals, do):
    q, k, v, coefficients = residuals
    degree = coefficients.shape[-1] - 1
    grad_q, sums, grad_k, grad_v = _backward(QUERY, degree, precision, q, k, v, do,
                                             _coefficient_vectors(coefficients))
    return grad_q, grad_k, grad_v, sums.transpose(1, 3, 2, 0)


_query.defvjp(_query_fwd, _query_bwd)


def query_polynomial_attention(q, k, v, coefficients, *, precision: str | None = None):
    """Unnormalized causal p_(b,t,h)(QK^T / sqrt(d_K)) V with FP32 coefficients c[B, T, H, m+1]."""
    _check_operands(q, k, v, "Mosaic query-polynomial attention")
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    if (coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3] or coefficients.dtype != jnp.float32
            or not 2 <= coefficients.shape[-1] <= MAX_QUERY_DEGREE + 1):
        raise ValueError("Mosaic query-polynomial attention requires FP32 coefficients [batch, time, heads, m+1] "
                         f"with 1 <= m <= {MAX_QUERY_DEGREE}, got {coefficients.dtype} {coefficients.shape}")
    return _query(q, k, v, coefficients, precision)


# ----------------------------------------------------------------------------------------
# Query-distance forget gates: the query polynomial times exp((t-i) l) inside each tile.

def _check_query_operands(q, k, v, coefficients, name):
    _check_operands(q, k, v, name)
    if (coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3] or coefficients.dtype != jnp.float32
            or not 2 <= coefficients.shape[-1] <= MAX_QUERY_DEGREE + 1):
        raise ValueError(f"{name} requires FP32 coefficients [batch, time, heads, m+1] with 1 <= m <= "
                         f"{MAX_QUERY_DEGREE}, got {coefficients.dtype} {coefficients.shape}")


def _token_vector(x):
    return x.astype(jnp.float32).transpose(0, 2, 1)[None]  # [1, batch, heads, time]


def _decay_forward(family, q, k, v, vectors, key_vectors, precision, degree):
    _, kernels = _kernels()
    out, _ = kernels.forward(_pad(q), _pad(k), v, vectors, key_vectors, family=family, degree=degree,
                             scale=_scale(q), precision=precision)
    return _cast_output(out, v.dtype)


def _query_decay_vectors(coefficients, log_decay):
    # Validate rates per token (the kernels never test them): an invalid one becomes NaN, which
    # poisons every weight of its query. The kernels exponentiate in base 2.
    rate = jnp.where(jnp.isfinite(log_decay) & (log_decay <= 0), log_decay * common.LOG2E, jnp.nan)
    return jnp.concatenate([_coefficient_vectors(coefficients), _token_vector(rate)])


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _query_decay(q, k, v, coefficients, log_decay, precision):
    return _decay_forward(QUERY_DECAY, q, k, v, _query_decay_vectors(coefficients, log_decay), None, precision,
                          coefficients.shape[-1] - 1)


def _query_decay_fwd(q, k, v, coefficients, log_decay, precision):
    return _query_decay(q, k, v, coefficients, log_decay, precision), (q, k, v, coefficients, log_decay)


def _query_decay_bwd(precision, residuals, do):
    q, k, v, coefficients, log_decay = residuals
    degree = coefficients.shape[-1] - 1
    grad_q, sums, grad_k, grad_v = _backward(QUERY_DECAY, degree, precision, q, k, v, do,
                                             _query_decay_vectors(coefficients, log_decay))
    return (grad_q, grad_k, grad_v, sums[:degree + 1].transpose(1, 3, 2, 0),
            sums[degree + 1].transpose(0, 2, 1).astype(log_decay.dtype))


_query_decay.defvjp(_query_decay_fwd, _query_decay_bwd)


def query_decay_attention(q, k, v, coefficients, log_decay, *, precision: str | None = None):
    """Unnormalized causal sum_i p_(b,t,h)(s_ti) exp((t-i) l_(b,t,h)) v_i with FP32 coefficients
    c[B, T, H, m+1] and raw FP32 log rates l[B, T, H] (``forget_gate.log_decay``). A query whose
    rate is nonfinite or positive is NaN."""
    _check_query_operands(q, k, v, coefficients, "Mosaic query-decay attention")
    if log_decay.shape != q.shape[:3] or log_decay.dtype != jnp.float32:
        raise ValueError(f"Mosaic query-decay attention requires FP32 log rates [batch, time, heads], "
                         f"got {log_decay.dtype} {log_decay.shape}")
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    return _query_decay(q, k, v, coefficients, log_decay, precision)


def _finite_or_nan(x):
    # Validate endpoint inputs per token (the kernels never test them): a nonfinite one, even
    # -inf whose softplus would be a valid zero rate, becomes NaN and poisons the pairs it enters.
    return jnp.where(jnp.isfinite(x), x, jnp.nan)


def _endpoint_vectors(coefficients, query_input, rate):
    batch, time = query_input.shape[:2]
    scale = jnp.broadcast_to(rate[None, None, :], (batch, time, rate.shape[0]))
    return jnp.concatenate([_coefficient_vectors(coefficients), _token_vector(_finite_or_nan(query_input)),
                            _token_vector(scale)])


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _endpoint_decay(q, k, v, coefficients, query_input, key_input, rate, precision):
    vectors = _endpoint_vectors(coefficients, query_input, rate)
    return _decay_forward(ENDPOINT_DECAY, q, k, v, vectors, _token_vector(_finite_or_nan(key_input)), precision,
                          coefficients.shape[-1] - 1)


def _endpoint_decay_fwd(q, k, v, coefficients, query_input, key_input, rate, precision):
    output = _endpoint_decay(q, k, v, coefficients, query_input, key_input, rate, precision)
    return output, (q, k, v, coefficients, query_input, key_input, rate)


def _endpoint_decay_bwd(precision, residuals, do):
    q, k, v, coefficients, query_input, key_input, rate = residuals
    degree = coefficients.shape[-1] - 1
    vectors = _endpoint_vectors(coefficients, query_input, rate)
    grad_q, sums, grad_k, grad_v, key_sums = _backward(
        ENDPOINT_DECAY, degree, precision, q, k, v, do, vectors, _token_vector(_finite_or_nan(key_input)))
    # l = a softplus(u + v): dl/du = dl/dv = a sigmoid(u + v), dl/da = softplus(u + v), which the
    # kernels sum as log2(1 + e^x).
    head_rate = rate[None, :, None]
    return (grad_q, grad_k, grad_v, sums[:degree + 1].transpose(1, 3, 2, 0),
            (head_rate * sums[degree + 1]).transpose(0, 2, 1), (head_rate * key_sums[0]).transpose(0, 2, 1),
            common.LN2 * sums[degree + 2].sum(axis=(0, 2)))


_endpoint_decay.defvjp(_endpoint_decay_fwd, _endpoint_decay_bwd)


def endpoint_decay_attention(q, k, v, coefficients, query_gate_input, key_gate_input, gate_log_rate, *,
                             precision: str | None = None):
    """Unnormalized causal sum_i p_(b,t,h)(s_ti) exp(-(t-i) exp(A_h) softplus(u_t + v_i)) v_i with
    FP32 endpoint gate inputs u, v [B, T, H] (``forget_gate.project_gate_inputs``) and log rate
    A [H, 1]. Nonfinite gate inputs or rates make the affected causal rows NaN."""
    _check_query_operands(q, k, v, coefficients, "Mosaic endpoint-decay attention")
    if (query_gate_input.shape != q.shape[:3] or key_gate_input.shape != q.shape[:3]
            or gate_log_rate.shape != (q.shape[2], 1)
            or not query_gate_input.dtype == key_gate_input.dtype == jnp.float32):
        raise ValueError("Mosaic endpoint-decay attention requires FP32 gate inputs [batch, time, heads] "
                         "and a log rate [heads, 1]")
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    log_rate = gate_log_rate[:, 0].astype(jnp.float32)
    # A nonfinite log rate is invalid even where exp would hide it (exp(-inf) = 0).
    rate = jnp.where(jnp.isfinite(log_rate), -jnp.exp(log_rate), jnp.nan)
    return _endpoint_decay(q, k, v, coefficients, query_gate_input, key_gate_input, rate, precision)
