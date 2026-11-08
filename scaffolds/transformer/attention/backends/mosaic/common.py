"""Elementwise helpers shared by the Hopper and Blackwell Mosaic GPU Taylor kernels."""
import math
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import lax


def truncate(x):
    """Clear the low 16 bits of an FP32 value: an exact BF16 value with a same-sign residual."""
    bits = lax.bitcast_convert_type(x, jnp.uint32)
    return lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), jnp.float32)


# Precision of the FP32 weights and score gradients that enter the tensor cores.
#   bf16:   one round-to-nearest BF16 value, as FlashAttention and cuDNN feed probabilities
#           and score gradients to their matrix products.
#   bf16x3: three truncated BF16 components whose FP32 sum reproduces the FP32 value, as the
#           Pallas-Triton kernels do (three matrix products per weight product).
BF16, BF16X3 = "bf16", "bf16x3"
PRECISIONS = (BF16, BF16X3)


# B300 per-block shared memory, less a margin for barriers and alignment.
BLACKWELL_SMEM_BYTES = 232448 - 2048


def blackwell_dkv_stages(width, value_width, vector_count, time, *, block_q=64, block_k=128, warpgroups=2,
                         extra_bytes=0):
    """Q/dO ring depth (3, else 2) of Blackwell's dK/dV kernel, which keeps every query's FP32
    vectors in shared memory beside its BF16 K/V block (and a decay family's ``extra_bytes``,
    ``dkv_extra_bytes``); None when even two stages do not fit."""
    fixed = 2 * (warpgroups * block_k * (width + value_width) + vector_count * time * 2) + extra_bytes
    for stages in (3, 2):
        if fixed + stages * 2 * block_q * (width + value_width) <= BLACKWELL_SMEM_BYTES:
            return stages
    return None


def blackwell_query_pass_bytes(width, value_width, vector_count, sum_count, key_count, time, *, head=False,
                               rows=128, block_k=64, warpgroups=2):
    """Shared memory of Blackwell's forward and dQ kernels (the larger), which hold a CTA's Q, its
    K/V ring (three stages forward; two dQ at width 128, else three), per-query vectors, the
    forward's FP32 output staging and the dQ sums, every key's vectors for the whole sequence, and
    their barriers and TMEM address (at most 256 bytes; BLACKWELL_SMEM_BYTES already sets aside
    the 2 KiB reduction scratch)."""
    vectors = 4 * (warpgroups * vector_count * rows + key_count * time) + (1024 if head else 0) + 256
    forward = 2 * (warpgroups * rows * width + 3 * block_k * (width + value_width)) + 4 * warpgroups * rows * (
        value_width + 1) + vectors
    stages = 2 if width > 64 else 3
    dq = 2 * (warpgroups * rows * (width + value_width) + stages * block_k * (width + value_width)) + (
        4 * warpgroups * sum_count * rows) + vectors
    return max(forward, dq)


def part_count(precision):
    return {BF16: 1, BF16X3: 3}[precision]


def weight_parts(x, precision):
    """BF16 operands whose products with a BF16 tile sum to the weight product."""
    if precision == BF16:
        return [x.astype(jnp.bfloat16)]
    high = truncate(x)
    rest = x - high
    middle = truncate(rest)
    return [y.astype(jnp.bfloat16) for y in (high, middle, rest - middle)]


def finite(x):
    return jnp.abs(x) < jnp.float32(jnp.inf)  # NaN compares false; Mosaic GPU lacks is_finite


def taylor(s, degree):
    """T_degree(s) = sum_r s^r / r! by Horner on the coefficients."""
    value = jnp.full_like(s, 1.0 / math.factorial(degree))
    for order in range(degree - 1, -1, -1):
        value = value * s + 1.0 / math.factorial(order)
    return value


def finalize(acc, rowsum, degree):
    """Divide by the row sum; a row is NaN when its sum is nonfinite, nonpositive (even
    degree) or zero (odd, signed degree), or its output overflows, as in fused_taylor."""
    invalid = ~finite(rowsum) | ((rowsum <= 0) if degree % 2 == 0 else (rowsum == 0))
    denominator = jnp.where(invalid, jnp.nan, rowsum)
    out = acc / lax.broadcast_in_dim(denominator, acc.shape, [0])
    overflow = jnp.where(finite(out), 0.0, 1.0).sum(axis=1)
    out = jnp.where(lax.broadcast_in_dim(overflow, out.shape, [0]) > 0, jnp.nan, out)
    return out, jnp.where(overflow > 0, jnp.nan, denominator)


def polynomial(s, coefficients):
    """sum_r c_r s^r by Horner; coefficients lowest order first, broadcastable to s."""
    value = jnp.zeros_like(s) + coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        value = value * s + coefficient
    return value


def polynomial_and_derivative(s, coefficients):
    """p(s) and p'(s), differentiating the Horner recurrence as fused_query_polynomial does."""
    value = jnp.zeros_like(s) + coefficients[-1]
    derivative = jnp.zeros_like(s)
    for coefficient in coefficients[-2::-1]:
        derivative = derivative * s + value
        value = value * s + coefficient
    return value, derivative


# Kernel families. Each kernel owns tiling and synchronization; these functions own the
# elementwise mathematics. ``vectors`` are per-query FP32 vectors already broadcast to the
# tile; masked tiles pass scores and dP with masked entries set to zero, and the kernels
# zero masked entries of every returned tile.
#   taylor: normalized fixed Taylor (forward vectors none; backward vectors delta, 1/Z).
#   query:  unnormalized polynomial with per-query coefficients c_0..c_m (vectors).
#   query_decay:    the query polynomial times D = exp((t-i) l_t) = 2^((t-i) l_t log2(e)), with
#                   the per-query l_t log2(e) as vector m+1 (forget_gate's query-distance decay).
#   endpoint_decay: the query polynomial times D = exp((t-i) l_ti), l_ti = a softplus(u_t + v_i),
#                   with u_t and a = -exp(A_h) as vectors m+1 and m+2 and v_i as the key vector.
#   per_order_query_decay:    sum_r c_r s^r D_r, with every order r its own query-distance decay
#                             D_r = 2^((t-i) l_tr log2(e)) (vectors m+1..2m+1 hold l_tr log2(e)).
#   per_order_endpoint_decay: D_r = exp((t-i) a_r softplus(u_tr + v_ir)): u_t0..u_tm as vectors
#                             m+1..2m+1, v_i0..v_im as key vectors, a_r = -exp(A_hr) as head scalars.
# Callers validate gates per token: an invalid rate or endpoint input arrives as NaN, which makes
# every weight it touches NaN. Decay kernels compile with approximate exp2/log2 (``approx_math``).
# They also receive a ``Pair``: the distance t-i and the per-key vectors of the tile. Their dQ row
# sums append sum_i G W (t-i) (query) or sum_i G W (t-i) sigmoid(x) and
# sum_i G W (t-i) log2(1 + e^x) (endpoint), where G = dO V^T, W = p D and x = u_t + v_i; the
# endpoint dK pass also sums G W (t-i) sigmoid(x) over queries for every key. Per-order families
# evaluate one order's decay at a time; their dQ sums are dc_r = sum_i G D_r s^r, then per order
# sum_i G W_r (t-i) (query) or sum_i G W_r (t-i) sigmoid(x_r) and sum_i G W_r (t-i) log2(1 + e^x_r)
# (endpoint), with W_r = c_r s^r D_r; the endpoint dK pass sums G W_r (t-i) sigmoid(x_r) per order.
TAYLOR, QUERY = "taylor", "query"
QUERY_DECAY, ENDPOINT_DECAY = "query_decay", "endpoint_decay"
PER_ORDER_QUERY_DECAY, PER_ORDER_ENDPOINT_DECAY = "per_order_query_decay", "per_order_endpoint_decay"
PER_ORDER_FAMILIES = (PER_ORDER_QUERY_DECAY, PER_ORDER_ENDPOINT_DECAY)
DECAY_FAMILIES = (QUERY_DECAY, ENDPOINT_DECAY, *PER_ORDER_FAMILIES)


class Pair(NamedTuple):
    """Per-element context of a decay family's tile: FP32 query-minus-key distance (zero on
    masked entries), per-key FP32 vectors broadcast to the tile, in tiles crossing the
    diagonal the causal mask (else None), and per-head FP32 scalars."""
    distance: Any
    keys: tuple = ()
    causal: Any = None
    heads: tuple = ()


class Lazy:
    """A sequence whose items are built on access: kernels pass column-broadcast vectors this
    way to per-order families, so each occupies registers only while one order uses it (a
    64x64 Hopper tile spreads a column vector over 16 registers per thread)."""

    def __init__(self, count, build):
        self.count, self.build = count, build

    def __len__(self):
        return self.count

    def __iter__(self):
        return (self.build(i) for i in range(self.count))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if isinstance(index, int):  # static: Python semantics; traced indices are taken as in range
            if not -self.count <= index < self.count:
                raise IndexError(index)
            index %= self.count
        return self.build(index)


class Decay(NamedTuple):
    factor: Any  # D: NaN where a gate is invalid, zero on masked entries
    sigmoid: Any = None  # endpoint: sigmoid(x), zero on masked entries
    log2_softplus: Any = None  # endpoint: log2(1 + e^x), zero on masked entries


LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
_SERIES = 2.0 ** -6  # below it, log2(1 + y) uses its series instead of the approximate log2


def vector_counts(family, degree):
    """(forward vectors, backward vectors, dQ row sums) for a family."""
    if family == TAYLOR:
        return 0, 2, 0
    if family in PER_ORDER_FAMILIES:
        orders = degree + 1
        return 2 * orders, 2 * orders, (3 if family == PER_ORDER_ENDPOINT_DECAY else 2) * orders
    extra = {QUERY: 0, QUERY_DECAY: 1, ENDPOINT_DECAY: 2}[family]
    return degree + 1 + extra, degree + 1 + extra, degree + 1 + extra


def approx_math(family):
    """Whether a family's kernels use the GPU's approximate exp2/log2 (ex2/lg2.approx: about
    2^-22 relative and absolute error). Only decay families evaluate transcendentals."""
    return family in DECAY_FAMILIES


def key_counts(family, degree):
    """(per-key vectors, per-key dK-pass sums) for a family."""
    if family == PER_ORDER_ENDPOINT_DECAY:
        return degree + 1, degree + 1
    return (1, 1) if family == ENDPOINT_DECAY else (0, 0)


def head_count(family, degree):
    """Per-head FP32 scalars of a family (per-order endpoint rates a_r)."""
    return degree + 1 if family == PER_ORDER_ENDPOINT_DECAY else 0


def head_width(count):
    """Stored width of ``count`` per-head scalars: whole 128-byte rows, since a copy issued by a
    warpgroup must split its bytes evenly over the warpgroup's 128 threads."""
    return -(-count // 32) * 32


def dkv_extra_bytes(family, which, degree):
    """Shared memory a family's B300 dK ("dk") or dV ("dv") pass adds to ``blackwell_dkv_stages``'s
    formula: 1 KiB-aligned per-key vector, per-key sum, and head-scalar buffers, and the reduction
    scratch of the per-key sums."""
    n_key, n_key_sums = key_counts(family, degree)
    if which == "dv":
        n_key_sums = 0
    buffer = lambda count: -(-count * 2 * 128 * 4 // 1024) * 1024  # [warpgroups, count, keys] FP32
    return (buffer(n_key) if n_key else 0) + (buffer(n_key_sums) if n_key_sums else 0) + (
        1024 if head_count(family, degree) else 0) + (2048 if n_key_sums else 0)


def log2_softplus(x):
    """log2(1 + e^x) = max(x, 0) log2(e) + log2(1 + y), y = e^-|x| in (0, 1], from one exp2 and
    one log2, stable for every x. Below y = 2^-6 the cubic series of log2(1 + y) (relative error
    < 1e-6) replaces the approximate log2, whose absolute error (2^-22) would dominate tiny y."""
    y = jnp.exp2(jnp.abs(x) * -LOG2E)
    series = y * LOG2E * (1.0 - y * (0.5 - y * (1.0 / 3.0)))
    return jnp.maximum(x, 0.0) * LOG2E + jnp.where(y < _SERIES, series, jnp.log2(1.0 + y))


def decay(family, degree, vectors, pair, *, slopes=False):
    """D = 2^((t-i) l log2(e)) of a decay family's tile (``Decay``): one on the diagonal, zero on
    masked entries, and free to underflow to zero at large distances. A NaN gate (an invalid one,
    see above) makes every causal weight it touches NaN; sigmoid(x) = 2^(x log2(e) - log2(1 + e^x))."""
    sigmoid = log2_sp = None
    if family == QUERY_DECAY:
        exponent = vectors[degree + 1] * pair.distance
    else:
        x = vectors[degree + 1] + pair.keys[0]
        log2_sp = log2_softplus(x)
        exponent = vectors[degree + 2] * log2_sp * pair.distance
        if slopes:
            sigmoid = jnp.exp2(x * LOG2E - log2_sp)
    factor = jnp.exp2(exponent)
    if pair.causal is not None:  # future keys' gates may be invalid or overflow; drop them exactly
        factor = jnp.where(pair.causal, factor, 0.0)
        if slopes:
            sigmoid, log2_sp = jnp.where(pair.causal, sigmoid, 0.0), jnp.where(pair.causal, log2_sp, 0.0)
    return Decay(factor, sigmoid, log2_sp if slopes else None)


def _order_decay(family, degree, vectors, pair, order, *, slopes=False):
    """One order's ``Decay`` in a per-order family's tile, as ``decay``."""
    sigmoid = log2_sp = None
    if family == PER_ORDER_QUERY_DECAY:
        exponent = vectors[degree + 1 + order] * pair.distance
    else:
        x = vectors[degree + 1 + order] + pair.keys[order]
        log2_sp = log2_softplus(x)
        exponent = pair.heads[order] * log2_sp * pair.distance
        if slopes:
            sigmoid = jnp.exp2(x * LOG2E - log2_sp)
    factor = jnp.exp2(exponent)
    if pair.causal is not None:
        factor = jnp.where(pair.causal, factor, 0.0)
        if slopes:
            sigmoid, log2_sp = jnp.where(pair.causal, sigmoid, 0.0), jnp.where(pair.causal, log2_sp, 0.0)
    return Decay(factor, sigmoid, log2_sp if slopes else None)


def _per_order_weights(family, degree, s, vectors, pair):
    """W = sum_r c_r D_r s^r by Horner over the orders. The orders form a rolled loop with one
    order's decay per iteration, which keeps a kernel's code within the instruction cache;
    ``vectors``, ``pair.keys`` and ``pair.heads`` must accept traced indices (``Lazy``)."""
    def term(order):
        return vectors[order] * _order_decay(family, degree, vectors, pair, order).factor

    return lax.fori_loop(0, degree, lambda i, w: w * s + term(degree - 1 - i), term(degree))


def _per_order_backward(family, degree, s, dp, vectors, pair, *, rows, weights=False):
    """(dS, W or None, sums) of a per-order tile in one ascending pass over the orders, one decay
    at a time: dS = G sum_r r c_r D_r s^(r-1) and W = sum_r c_r D_r s^r. ``rows`` gives the dQ
    sums (see above); otherwise the per-key sums of the endpoint dK pass. Orders 0 and 1 start
    every accumulator in its layout; later orders form a rolled loop (``_per_order_weights``)
    whose sums reach their order's accumulator through 0/1 masks. A NaN sum at a rolled order (an
    invalid gate, whose row is NaN already) therefore reaches every rolled order's accumulator."""
    endpoint = family == PER_ORDER_ENDPOINT_DECAY
    orders = degree + 1

    def evaluate(order, power):
        """Order r's weight c_r D_r and sums, given power = s^r (None for r = 0)."""
        terms = _order_decay(family, degree, vectors, pair, order, slopes=endpoint)
        weight = vectors[order] * terms.factor
        gds = dp * terms.factor if power is None else dp * terms.factor * power  # G D_r s^r
        sums = [gds.sum(axis=1)] if rows else []
        if rows or endpoint:  # query-only rates have no per-key gradients
            gwd = gds * vectors[order] * pair.distance  # G W_r (t-i)
            if pair.causal is not None:  # a masked query's polynomial may be nonfinite at s = 0
                gwd = jnp.where(pair.causal, gwd, 0.0)
            sums.append((gwd * terms.sigmoid if endpoint else gwd).sum(axis=1))
            if rows and endpoint:
                sums.append((gwd * terms.log2_softplus).sum(axis=1))
        return weight, sums

    weight0, sums0 = evaluate(0, None)
    weight1, sums1 = evaluate(1, s)
    carry = (weight1, s, weight0 + weight1 * s if weights else None,
             [[first, second, *([first * 0.0] * (orders - 2))] for first, second in zip(sums0, sums1)])

    def body(order, carry):
        derivative, power, value, accumulators = carry
        previous, power = power, power * s
        weight, sums = evaluate(order, power)
        derivative = derivative + weight * previous * order.astype(jnp.float32)
        value = None if value is None else value + weight * power
        masks = [(order == r).astype(jnp.float32) for r in range(2, orders)]
        accumulators = [[*kind[:2], *(a + mask * x for a, mask in zip(kind[2:], masks))]
                        for kind, x in zip(accumulators, sums)]
        return derivative, power, value, accumulators

    derivative, _, value, accumulators = lax.fori_loop(2, orders, body, carry)
    return dp * derivative, value, [x for kind in accumulators for x in kind]


def forward_weights(family, degree, s, vectors, pair=None, terms=None):
    """Weights of a tile; decay families may pass the tile's ``Decay`` as ``terms``."""
    if family in PER_ORDER_FAMILIES:
        return _per_order_weights(family, degree, s, vectors, pair)
    if family in DECAY_FAMILIES:
        terms = decay(family, degree, vectors, pair) if terms is None else terms
        return polynomial(s, vectors[:degree + 1]) * terms.factor
    return taylor(s, degree) if family == TAYLOR else polynomial(s, vectors)


def forward_statistic(family, p):
    """Per-element contribution to the per-row forward statistic: the row sum (Taylor),
    or p - p, which sums to zero unless some weight is nonfinite (query)."""
    return p if family == TAYLOR else p - p


def forward_finish(family, degree, acc, statistic):
    """(output, statistic output) from the FP32 accumulator and the per-row statistic."""
    if family == TAYLOR:
        return finalize(acc, statistic, degree)
    bad = jnp.where(finite(statistic), 0.0, 1.0)  # float flags: Mosaic broadcasts floats, not booleans
    return jnp.where(lax.broadcast_in_dim(bad, acc.shape, [0]) > 0, jnp.nan, acc), statistic


def score_gradient(family, degree, s, dp, vectors):
    """dS for the dQ and dK products."""
    if family == TAYLOR:
        delta, inv_z = vectors
        return (dp - delta) * taylor(s, degree - 1) * inv_z
    return dp * polynomial_and_derivative(s, vectors)[1]


def value_weights(family, degree, s, vectors, pair=None, terms=None):
    """W for the dV product."""
    if family in PER_ORDER_FAMILIES:
        return _per_order_weights(family, degree, s, vectors, pair)
    if family in DECAY_FAMILIES:
        return forward_weights(family, degree, s, vectors, pair, terms)
    if family == TAYLOR:
        return taylor(s, degree) * vectors[1]
    return polynomial(s, vectors)


def row_sums(family, degree, s, dp):
    """Per-query reductions emitted by the dQ kernel: dc_r = sum_j dP s^r (query)."""
    if family == TAYLOR:
        return []
    sums, power = [], jnp.ones_like(s)
    for _ in range(degree + 1):
        sums.append((dp * power).sum(axis=1))
        power = power * s
    return sums


def _decay_gradient(family, degree, s, dp, vectors, pair, terms=None):
    """(dS, G W (t-i), G D, Decay) of a decay family: dS = G D p'(s); G D is the cotangent
    of the polynomial."""
    if terms is None:
        terms = decay(family, degree, vectors, pair, slopes=family == ENDPOINT_DECAY)
    gd = dp * terms.factor
    p, slope = polynomial_and_derivative(s, vectors[:degree + 1])
    gwd = gd * p * pair.distance
    if pair.causal is not None:  # p of a masked entry may be nonfinite (e.g. inf * 0 in Horner)
        gwd = jnp.where(pair.causal, gwd, 0.0)
    return gd * slope, gwd, gd, terms


def dq_terms(family, degree, s, dp, vectors, pair=None):
    """(dS, dQ-kernel row sums) of a tile."""
    if family in PER_ORDER_FAMILIES:
        ds, _, sums = _per_order_backward(family, degree, s, dp, vectors, pair, rows=True)
        return ds, sums
    if family not in DECAY_FAMILIES:
        return score_gradient(family, degree, s, dp, vectors), row_sums(family, degree, s, dp)
    ds, gwd, gd, terms = _decay_gradient(family, degree, s, dp, vectors, pair)
    sums = row_sums(QUERY, degree, s, gd)
    if family == QUERY_DECAY:
        return ds, [*sums, gwd.sum(axis=1)]
    return ds, [*sums, (gwd * terms.sigmoid).sum(axis=1), (gwd * terms.log2_softplus).sum(axis=1)]


def dk_terms(family, degree, s, dp, vectors, pair=None, terms=None):
    """(dS, per-key sums over the tile's queries) of a transposed dK tile; ``terms`` may hold
    the tile's ``Decay`` from ``decay(..., slopes=True)``."""
    if family in PER_ORDER_FAMILIES:
        ds, _, sums = _per_order_backward(family, degree, s, dp, vectors, pair, rows=False)
        return ds, sums
    if family not in DECAY_FAMILIES:
        return score_gradient(family, degree, s, dp, vectors), []
    ds, gwd, _, terms = _decay_gradient(family, degree, s, dp, vectors, pair, terms)
    return ds, ([(gwd * terms.sigmoid).sum(axis=1)] if family == ENDPOINT_DECAY else [])


def dkv_terms(family, degree, s, dp, vectors, pair):
    """(W, dS, per-key sums) of a per-order family's transposed tile in one pass, for kernels that
    form dV and dK together after S and dP."""
    ds, w, sums = _per_order_backward(family, degree, s, dp, vectors, pair, rows=False, weights=True)
    return w, ds, sums
