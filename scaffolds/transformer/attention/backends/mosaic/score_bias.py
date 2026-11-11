"""Per-degree score biases on H100 Mosaic GPU kernels.

The unnormalized query polynomial, ungated (``api.query_polynomial_attention``) or with its
per-order endpoint gate (``api.per_order_endpoint_decay_attention``), with every order r = 1..m
of the score shifted by its own bias and order 0 unshifted:

    W(t, i) = c_0(t) G_0(t, i) + sum_{r=1..m} c_r(t) G_r(t, i) (s_ti + beta_r(t, i))^r,  o_t = sum_{i<=t} W v_i,

with G_r = 1 (``score_bias_query_polynomial_attention``) or
G_r = exp(-(t - i) exp(A_r) softplus(u_r(t) + v_r(i))) (``score_bias_per_order_endpoint_attention``).
The shift arrives in one of two forms:

- rows (static and query biases): beta_r(t) [B, T, H, m] for orders 1..m, computed by the caller;
  its gradient is R_r(t) = sum_i dbeta_r(t, i), which JAX maps to the bias parameters.
- pairs (query-key biases): beta_r(t, i) = b_r + a_r tanh(p_r(t) + q_r(i)) with token inputs
  p, q [B, T, H, m] and head scalars b, a [H, m], combined inside the tiles (tanh from exp2).
  Their gradients are dp_r(t) = a_r sum_i dbeta (1 - tau^2), dq_r(i) = a_r sum_t dbeta (1 - tau^2),
  da_r = sum dbeta tau and db_r = sum dbeta over every causal pair.

Each order's z_r = s + beta_r is raised to z^(r-1) and z^r by repeated multiplication inside the
causal tiles: never expanded into powers of s (which would change how the orders cancel), never
divided by z, so exactly zero shifted scores are safe. The backward recomputes S, dP and every
order's gate and shift from the inputs (the residuals are the inputs only). FP32 scores, shifts,
powers, sums and accumulators; weights and score gradients enter the tensor cores at
``precision`` (default BF16, as ``api.default_precision``). A token whose shift input, coefficient,
gate input or weight is nonfinite makes the affected causal rows NaN, as in the other Mosaic
operations; every reduction of a tile crossing the diagonal selects its causal entries first, so
a future token's input never reaches an earlier row.

H100 only: these kernels stage every per-order vector per tile, which B300's kernels do not. On
any other GPU, and for shapes H100's shared memory cannot hold (``score_bias_supported``), the
entry points raise ValueError before tracing a kernel; there is no fallback.
"""
from functools import partial

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends.mosaic import api, common

GATES = ("none", "endpoint")  # no forget gate, or the per-order endpoint (query-key) gate
SHIFTS = ("row", "pair")  # beta_r(t) given per query, or b_r + a_r tanh(p_r(t) + q_r(i))
SHIFT_OF_MODE = {"static": "row", "query": "row", "query_key": "pair"}  # model.attention_score_bias
HOPPER_TIME_MULTIPLE = 128  # rows per H100 CTA

FAMILY = {("none", "row"): common.SCORE_BIAS_ROW, ("none", "pair"): common.SCORE_BIAS_PAIR,
          ("endpoint", "row"): common.SCORE_BIAS_ENDPOINT_ROW, ("endpoint", "pair"): common.SCORE_BIAS_ENDPOINT_PAIR}
IMPLEMENTATION = {
    ("none", "row"): "pallas_mosaic_gpu_query_polynomial_score_bias_row",
    ("none", "pair"): "pallas_mosaic_gpu_query_polynomial_score_bias_pair",
    ("endpoint", "row"): "pallas_mosaic_gpu_query_polynomial_per_order_endpoint_decay_score_bias_row",
    ("endpoint", "pair"): "pallas_mosaic_gpu_query_polynomial_per_order_endpoint_decay_score_bias_pair",
}


def score_bias_family(gate: str, shift: str) -> str:
    """The kernel family (``common``) of a gate ("none" or "endpoint") and shift ("row" or "pair")."""
    if gate not in GATES or shift not in SHIFTS:
        raise ValueError(f"Mosaic score biases take gate in {GATES} and shift in {SHIFTS}, got {gate!r}, {shift!r}")
    return FAMILY[gate, shift]


def implementation_name(gate: str, shift: str) -> str:
    """Provenance name of the kernels that run a gate and shift (``score_bias_family``)."""
    score_bias_family(gate, shift)
    return IMPLEMENTATION[gate, shift]


def score_bias_supported(degree, width, time, *, gate, shift) -> bool:
    """Whether H100's kernels run degree m at this Q/K width and sequence length: 1 <= m <=
    ``api.MAX_QUERY_DEGREE``, a Mosaic Q/K width, time a positive multiple of 128, and the forward,
    dQ and dK/dV kernels within H100's shared memory (``common.hopper_smem_bytes``, which the
    sequence length does not enter). B300 runs none of these families at any shape."""
    family = score_bias_family(gate, shift)
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= api.MAX_QUERY_DEGREE:
        return False
    if width not in api.WIDTHS or time <= 0 or time % HOPPER_TIME_MULTIPLE:
        return False
    return api.hopper_supported(family, degree, width)


def _require_hopper(name):
    """Refuse every GPU but H100 before any operand is inspected or kernel traced."""
    try:
        arch, _ = api._kernels()
    except (AttributeError, ValueError) as error:  # not a GPU, or an unsupported one
        raise ValueError(f"{name} runs on H100 (compute capability 9.0) only") from error
    if arch != "hopper":
        raise ValueError(f"{name} runs on H100 only: {arch}'s Mosaic kernels have no score-bias family, "
                         "and there is no fallback")


def _shift(name, q, coefficients, row_shift, query_input, key_input, offset, amplitude):
    """(shift kind, operands) of exactly one shift form, validated."""
    degree = coefficients.shape[-1] - 1
    tokens = (*q.shape[:3], degree)
    pair = (query_input, key_input, offset, amplitude)
    if row_shift is not None and all(x is None for x in pair):
        if row_shift.shape != tokens or row_shift.dtype != jnp.float32:
            raise ValueError(f"{name} requires an FP32 row shift [batch, time, heads, m] = {tokens}, "
                             f"got {row_shift.dtype} {row_shift.shape}")
        return "row", {"row_shift": row_shift}
    if row_shift is None and all(x is not None for x in pair):
        heads = (q.shape[2], degree)
        if (query_input.shape != tokens or key_input.shape != tokens or offset.shape != heads
                or amplitude.shape != heads or not all(x.dtype == jnp.float32 for x in pair)):
            raise ValueError(f"{name} requires FP32 shift inputs [batch, time, heads, m] = {tokens} and FP32 "
                             f"offsets and amplitudes [heads, m] = {heads}")
        return "pair", {"shift_query": query_input, "shift_key": key_input, "shift_offset": offset,
                        "shift_amplitude": amplitude}
    raise ValueError(f"{name} takes either row_shift, or all of query_shift_input, key_shift_input, shift_offset "
                     "and shift_amplitude")


def _prepare(name, gate, shift, q, coefficients, precision):
    """(family, precision) once the shape fits H100's kernels."""
    degree = coefficients.shape[-1] - 1
    if not score_bias_supported(degree, q.shape[-1], q.shape[1], gate=gate, shift=shift):
        raise ValueError(f"{name}: degree {degree} at Q/K width {q.shape[-1]} and {q.shape[1]} tokens does not "
                         "fit H100's kernels")
    family = score_bias_family(gate, shift)
    precision = api.default_precision(family, degree) if precision is None else precision
    api._check_precision(precision)
    return family, precision


def _pack(family, coefficients, operands):
    """(per-query vectors, per-key vectors or None, head scalars or None) in the family's layout
    (``common``): coefficients, then the gate's inputs, then the shift's. Token inputs are
    validated per token (``api._finite_or_nan``): a nonfinite one becomes NaN."""
    vectors, keys, heads = [api._coefficient_vectors(coefficients)], [], []
    if family in common.GATED_SCORE_BIAS_FAMILIES:
        vectors.append(api._order_vectors(api._finite_or_nan(operands["gate_query"])))
        keys.append(api._order_vectors(api._finite_or_nan(operands["gate_key"])))
        heads.append(operands["gate_rate"])
    if family in common.PAIR_SHIFT_FAMILIES:
        vectors.append(api._order_vectors(api._finite_or_nan(operands["shift_query"])))
        keys.append(api._order_vectors(api._finite_or_nan(operands["shift_key"])))
        heads += [api._finite_or_nan(operands["shift_offset"]), api._finite_or_nan(operands["shift_amplitude"])]
    else:
        vectors.append(api._order_vectors(api._finite_or_nan(operands["row_shift"])))
    key_vectors = jnp.concatenate(keys) if keys else None
    head_constants = None
    if heads:
        head_constants = jnp.concatenate([x.astype(jnp.float32) for x in heads], axis=1)
        count = head_constants.shape[1]
        head_constants = jnp.pad(head_constants, ((0, 0), (0, common.head_width(count) - count)))
    return jnp.concatenate(vectors), key_vectors, head_constants


def _forward(q, k, v, coefficients, operands, family, precision):
    _, kernels = api._kernels()
    vectors, key_vectors, head_constants = _pack(family, coefficients, operands)
    out, _ = kernels.forward(api._pad(q), api._pad(k), v, vectors, key_vectors, head_constants, family=family,
                             degree=coefficients.shape[-1] - 1, scale=api._scale(q), precision=precision)
    return api._cast_output(out, v.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _score_bias(q, k, v, coefficients, operands, family, precision):
    return _forward(q, k, v, coefficients, operands, family, precision)


def _score_bias_fwd(q, k, v, coefficients, operands, family, precision):
    return _forward(q, k, v, coefficients, operands, family, precision), (q, k, v, coefficients, operands)


def _score_bias_bwd(family, precision, residuals, do):
    q, k, v, coefficients, operands = residuals
    orders = coefficients.shape[-1]
    degree = orders - 1
    vectors, key_vectors, head_constants = _pack(family, coefficients, operands)
    grad_q, sums, grad_k, grad_v, *key_sums = api._backward(family, degree, precision, q, k, v, do, vectors,
                                                            key_vectors, head_constants)
    key_sums = key_sums[0] if key_sums else None
    tokens = lambda x: x.transpose(1, 3, 2, 0)  # [n, batch, heads, time] -> [batch, time, heads, n]
    totals = lambda x: x.sum(axis=(1, 3)).T  # summed over batch and time -> [heads, n]
    grads, row, key = {}, orders, 0
    if family in common.GATED_SCORE_BIAS_FAMILIES:
        # l_r = a_r softplus(u_r + v_r) as per-order endpoint gates (api._per_order_endpoint_decay_bwd).
        order_rate = operands["gate_rate"].T[:, None, :, None]
        grads["gate_query"] = tokens(order_rate * sums[orders:2 * orders])
        grads["gate_key"] = tokens(order_rate * key_sums[:orders])
        grads["gate_rate"] = common.LN2 * totals(sums[2 * orders:3 * orders])
        row, key = 3 * orders, orders
    if family in common.PAIR_SHIFT_FAMILIES:
        # beta = b + a tanh(p + q): dbeta/dp = dbeta/dq = a (1 - tau^2), dbeta/da = tau, dbeta/db = 1.
        order_amplitude = api._finite_or_nan(operands["shift_amplitude"]).T[:, None, :, None]
        grads["shift_query"] = tokens(order_amplitude * sums[row:row + degree])
        grads["shift_key"] = tokens(order_amplitude * key_sums[key:key + degree])
        grads["shift_amplitude"] = totals(sums[row + degree:row + 2 * degree])
        grads["shift_offset"] = totals(sums[row + 2 * degree:row + 3 * degree])
    else:
        grads["row_shift"] = tokens(sums[row:row + degree])
    return grad_q, grad_k, grad_v, tokens(sums[:orders]), grads


_score_bias.defvjp(_score_bias_fwd, _score_bias_bwd)


def score_bias_query_polynomial_attention(q, k, v, coefficients, *, row_shift=None, query_shift_input=None,
                                         key_shift_input=None, shift_offset=None, shift_amplitude=None,
                                         precision: str | None = None):
    """Unnormalized causal sum_i [c_0 + sum_{r>=1} c_r (s_ti + beta_r)^r] v_i with FP32 coefficients
    c[B, T, H, m+1] and either a row shift beta[B, T, H, m] (orders 1..m) or pair-shift inputs p, q
    [B, T, H, m] with offsets b and amplitudes a [H, m]: beta_r = b_r + a_r tanh(p_r(t) + q_r(i))."""
    name = "Mosaic score-bias query-polynomial attention"
    _require_hopper(name)
    api._check_query_operands(q, k, v, coefficients, name)
    shift, operands = _shift(name, q, coefficients, row_shift, query_shift_input, key_shift_input, shift_offset,
                             shift_amplitude)
    family, precision = _prepare(name, "none", shift, q, coefficients, precision)
    return _score_bias(q, k, v, coefficients, operands, family, precision)


def score_bias_per_order_endpoint_attention(q, k, v, coefficients, query_gate_input, key_gate_input, gate_log_rate,
                                            *, row_shift=None, query_shift_input=None, key_shift_input=None,
                                            shift_offset=None, shift_amplitude=None, precision: str | None = None):
    """``score_bias_query_polynomial_attention`` with every order's weight decayed by its endpoint gate,
    exp(-(t-i) exp(A_hr) softplus(u_tr + v_ir)), from FP32 gate inputs u, v [B, T, H, m+1] and log
    rates A [H, m+1, 1], as ``api.per_order_endpoint_decay_attention``."""
    name = "Mosaic score-bias per-order endpoint attention"
    _require_hopper(name)
    api._check_query_operands(q, k, v, coefficients, name)
    orders = coefficients.shape[-1]
    if (query_gate_input.shape != coefficients.shape or key_gate_input.shape != coefficients.shape
            or gate_log_rate.shape != (q.shape[2], orders, 1)
            or not query_gate_input.dtype == key_gate_input.dtype == jnp.float32):
        raise ValueError(f"{name} requires FP32 gate inputs [batch, time, heads, m+1] and log rates "
                         "[heads, m+1, 1]")
    shift, operands = _shift(name, q, coefficients, row_shift, query_shift_input, key_shift_input, shift_offset,
                             shift_amplitude)
    family, precision = _prepare(name, "endpoint", shift, q, coefficients, precision)
    log_rate = gate_log_rate[..., 0].astype(jnp.float32)
    rate = jnp.where(jnp.isfinite(log_rate), -jnp.exp(log_rate), jnp.nan)  # [heads, orders]
    operands = {**operands, "gate_query": query_gate_input, "gate_key": key_gate_input, "gate_rate": rate}
    return _score_bias(q, k, v, coefficients, operands, family, precision)
