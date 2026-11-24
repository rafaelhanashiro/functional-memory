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


# H100 per-block shared memory: the 227 KiB opt-in maximum that Mosaic GPU lowering checks.
HOPPER_SMEM_BYTES = 232448


def hopper_smem_bytes(family, degree, width, value_width, *, rows=64, block=64, warpgroups=2):
    """Shared memory of each Hopper kernel for a family ({"forward", "dq", "dkv"}), as Mosaic GPU
    lowering allocates it: every scratch buffer rounded up to 1 KiB (the BF16 Q/K/V/dO tiles of a
    CTA and their rings, three stages forward and two backward; the FP32 per-query and per-key
    vectors, sums and head scalars of a CTA or a tile; the forward's FP32 output and statistic), the
    2 KiB reduction scratch of a kernel that reduces, and 8-byte barriers. Vectors are staged per
    tile or per CTA, never for the whole sequence, so the sequence length does not enter."""
    def aligned(count, size):
        return -(-count * size // 1024) * 1024

    n_forward, n_backward, n_sums = vector_counts(family, degree)
    n_key, n_key_sums = key_counts(family, degree)
    n_head = head_count(family, degree)
    # A CTA's BF16 rows at the Q/K and value widths, and the BF16 tiles of a K/V or Q/dO ring.
    cta_qk, cta_v = aligned(warpgroups * rows * width, 2), aligned(warpgroups * rows * value_width, 2)
    ring = lambda stages: aligned(stages * block * width, 2) + aligned(stages * block * value_width, 2)
    per_cta = lambda count: aligned(warpgroups * count * rows, 4)  # FP32 vectors or sums of a CTA's rows
    per_tile = lambda stages, count: aligned(stages * count * block, 4)  # FP32 vectors of each ring tile
    heads = aligned(warpgroups * head_width(n_head), 4) if n_head else 0
    forward = (cta_qk + ring(3) + per_cta(max(n_forward, 1)) + per_tile(3, n_key) + heads
               + aligned(warpgroups * rows * value_width, 4) + per_cta(1) + 2048 + 8 * (warpgroups + 4 * 3))
    dq = (cta_qk + cta_v + ring(2) + per_cta(n_backward) + per_cta(max(n_sums, 1)) + per_tile(2, n_key) + heads
          + (2048 if n_sums else 0) + 8 * (warpgroups + 4 * 2))
    dkv = (cta_qk + cta_v + ring(2) + per_tile(2, n_backward) + per_cta(n_key) + per_cta(n_key_sums) + heads
           + (2048 if n_key_sums else 0) + 8 * (warpgroups + 2 * 2))
    return {"forward": forward, "dq": dq, "dkv": dkv}


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
#   per_order_cumulative_decay: D_r = exp(L_tr - L_ir), every order r its own cumulative log decay:
#                             L_t0..L_tm as vectors m+1..2m+1 and L_i0..L_im as key vectors, whose
#                             FP32 difference (exact for nearby tokens, however large |L|) the tile
#                             scales by log2(e).
# Callers validate gates per token: an invalid rate, endpoint input or cumulative sum arrives as
# NaN, which makes every weight it touches NaN. Decay kernels compile with approximate exp2/log2
# (``approx_math``). They also receive a ``Pair``: the distance t-i (none for the cumulative family)
# and the per-key vectors of the tile. Their dQ row sums append sum_i G W (t-i) (query) or
# sum_i G W (t-i) sigmoid(x) and sum_i G W (t-i) log2(1 + e^x) (endpoint), where G = dO V^T,
# W = p D and x = u_t + v_i; the endpoint dK pass also sums G W (t-i) sigmoid(x) over queries for
# every key. Per-order families evaluate one order's decay at a time; their dQ sums are
# dc_r = sum_i G D_r s^r, then per order sum_i G W_r (t-i) (query) or sum_i G W_r (t-i) sigmoid(x_r)
# and sum_i G W_r (t-i) log2(1 + e^x_r) (endpoint), with W_r = c_r s^r D_r; the endpoint dK pass
# sums G W_r (t-i) sigmoid(x_r) per order. The cumulative family's dQ sums are the dc_r alone and
# its dK pass sums G W_r per order: L_tr's gradient is c_tr dc_tr less that sum at key t (the
# caller forms it).
#   score_bias_*: per-degree score biases, sum_r c_r G_r z_r^r with z_r = s + beta_r for r >= 1 and
#                 order 0 unshifted, G_r = 1 (score_bias_row, score_bias_pair) or the per-order endpoint
#                 gate D_r (score_bias_endpoint_*, laid out as per_order_endpoint_decay). Each order's
#                 powers z^(r-1) and z^r are formed by repeated multiplication inside the tile: never
#                 expanded into powers of s, never divided by z. A row shift (*_row) takes
#                 beta_1(t)..beta_m(t) as the next m vectors. A pair shift (*_pair) takes p_1..p_m as
#                 the next m vectors, q_1..q_m as the next m key vectors, and b_1..b_m, a_1..a_m as the
#                 next 2m head scalars: beta_r = b_r + a_r tanh(p_r(t) + q_r(i)), tanh built from exp2.
#                 With g = G dbeta_r = G c_r G_r r z_r^(r-1), the dQ sums are dc_r = sum_i G G_r z_r^r,
#                 the gate's two kinds with W_r = c_r G_r z_r^r, then per order r >= 1 sum_i dbeta_r
#                 (row) or sum_i dbeta_r (1 - tau_r^2), sum_i dbeta_r tau_r, sum_i dbeta_r (pair); the dK
#                 pass sums the gate's kind, then sum_t dbeta_r (1 - tau_r^2) per order (pair). Every
#                 summand of a tile crossing the diagonal is selected with the causal mask first.
#   query_softplus, query_abs, query_sigmoid, query_swish: sum_r c_r z^r at the per-pair transform phi(s), with
#                 the vectors of query (attention/score_transform.py); fixed vectors [0, 1] give phi(s) alone.
#                 dS = G p'(z) phi'(s): softplus uses sigmoid(s), abs uses sign(s), sigmoid uses its stable
#                 logistic derivative, swish sigmoid(s) + s sigmoid'(s) (respectively 1/2, 0, 1/4, 1/2 at s = 0).
#                 The dQ sums are dc_r = sum_i G z^r (``transformed_score``). Masked entries
#                 evaluate phi at their zero score, finite for finite coefficients, and the kernels zero their
#                 weights and score gradients as for query, so these families need no ``Pair``. A nonfinite
#                 causal score gives a nonfinite z (softplus's -inf too), so its row is NaN as for query.
#   <base>_times_<f>: the score prefactor f (softplus, sigmoid, exp or swish of the raw score) times the
#                 polynomial of the base family, query (z = s) or query_softplus: W = f(s) p(z), and
#                 dS = G f(s) p'(z) phi'(s) + G f'(s) p(z), with dc_r = sum_i G f(s) z^r. f shares the transform's
#                 e^-|s| (``score_prefactor``); with phi = softplus, f = softplus is z itself and the sigmoid
#                 and swish prefactors reuse phi' = sigmoid(s). exp is 2^(s log2(e)) and overflows to +inf for
#                 large scores, which makes the row NaN. Masked entries enter at s = 0, where every f is finite,
#                 and are zeroed; a nonfinite causal score makes its row NaN, since p(z) is nonfinite there
#                 (and f too, but for exp at -inf, whose zero times the nonfinite p is NaN).
#   query_softplus_row_power: query_softplus in units of its row's largest power (attention/shifted_polynomial.py):
#                 sum_r a_r u^r at u = (z h) h, with a_0..a_m and h = e^(-m_t / (2m)) as vectors 0..m+1 (h twice,
#                 as the fused kernel, so that no unit is subnormal), the caller forming a_r = c_r e^(-(m - r) m_t / m)
#                 and m_t = m log max(1, max_i z_ti), so |u| <= 1 and every term stays within its coefficient's scale.
#                 dS = G p_a'(u) h h phi'(s), and the dQ sums are sum_i G u^r, from which the caller forms
#                 dc_r = e^(-(m - r) m_t / m) sum_i G u^r; m_t is a constant to differentiation.
TAYLOR, QUERY = "taylor", "query"
QUERY_SOFTPLUS, QUERY_ABS = "query_softplus", "query_abs"
QUERY_SIGMOID, QUERY_SWISH = "query_sigmoid", "query_swish"
# The kernel family of each parameter-free score transform (attention/score_transform.py); the identity is the
# query family itself.
TRANSFORM_BASES = {"identity": QUERY, "softplus": QUERY_SOFTPLUS, "abs": QUERY_ABS, "sigmoid": QUERY_SIGMOID,
                   "swish": QUERY_SWISH}
QUERY_SOFTPLUS_ROW_POWER = "query_softplus_row_power"
# The row-power family of each transformed family that has one.
ROW_POWER_FAMILIES = {QUERY_SOFTPLUS: QUERY_SOFTPLUS_ROW_POWER}
# Score prefactors of the transformed families, and the transforms whose families take one (the qualified ones).
SCORE_PREFACTORS = ("softplus", "sigmoid", "exp", "swish")
PREFACTOR_TRANSFORMS = ("identity", "softplus")


def score_function_family(transform, prefactor="none"):
    """The kernel family of W = f(s) p(phi(s)) for a transform phi and prefactor f, or None when no family
    evaluates them: ``query`` itself for the identity without a prefactor, a transform's family for prefactor
    "none" (query_softplus, query_abs, query_sigmoid, query_swish), and "<base>_times_<f>" for a prefactor, e.g.
    query_times_exp or query_softplus_times_sigmoid, whose base takes the identity or softplus transform."""
    base = TRANSFORM_BASES.get(transform)
    if base is None or prefactor == "none":
        return base
    if transform not in PREFACTOR_TRANSFORMS or prefactor not in SCORE_PREFACTORS:
        return None
    return f"{base}_times_{prefactor}"


# Every transformed family by its (transform, prefactor): all but the plain query family.
SCORE_FUNCTION_FAMILIES = {
    (transform, prefactor): score_function_family(transform, prefactor)
    for transform in TRANSFORM_BASES for prefactor in ("none", *SCORE_PREFACTORS)
    if score_function_family(transform, prefactor) not in (None, QUERY)}
FAMILY_SCORE_FUNCTIONS = {family: functions for functions, family in SCORE_FUNCTION_FAMILIES.items()}
TRANSFORMED_QUERY_FAMILIES = tuple(SCORE_FUNCTION_FAMILIES.values())
# A row-power family evaluates its base's score functions; it holds one vector more than the query family.
FAMILY_SCORE_FUNCTIONS.update({shifted: FAMILY_SCORE_FUNCTIONS[base] for base, shifted in ROW_POWER_FAMILIES.items()})
PREFACTOR_FAMILIES = tuple(family for (_, prefactor), family in SCORE_FUNCTION_FAMILIES.items() if prefactor != "none")
QUERY_DECAY, ENDPOINT_DECAY = "query_decay", "endpoint_decay"
PER_ORDER_QUERY_DECAY, PER_ORDER_ENDPOINT_DECAY = "per_order_query_decay", "per_order_endpoint_decay"
PER_ORDER_CUMULATIVE_DECAY = "per_order_cumulative_decay"
SCORE_BIAS_ROW, SCORE_BIAS_PAIR = "score_bias_row", "score_bias_pair"
SCORE_BIAS_ENDPOINT_ROW, SCORE_BIAS_ENDPOINT_PAIR = "score_bias_endpoint_row", "score_bias_endpoint_pair"
GATED_SCORE_BIAS_FAMILIES = (SCORE_BIAS_ENDPOINT_ROW, SCORE_BIAS_ENDPOINT_PAIR)
PAIR_SHIFT_FAMILIES = (SCORE_BIAS_PAIR, SCORE_BIAS_ENDPOINT_PAIR)
SCORE_BIAS_FAMILIES = (SCORE_BIAS_ROW, SCORE_BIAS_PAIR, *GATED_SCORE_BIAS_FAMILIES)
PER_ORDER_FAMILIES = (PER_ORDER_QUERY_DECAY, PER_ORDER_ENDPOINT_DECAY, PER_ORDER_CUMULATIVE_DECAY,
                      *SCORE_BIAS_FAMILIES)
DECAY_FAMILIES = (QUERY_DECAY, ENDPOINT_DECAY, PER_ORDER_QUERY_DECAY, PER_ORDER_ENDPOINT_DECAY,
                  PER_ORDER_CUMULATIVE_DECAY, *GATED_SCORE_BIAS_FAMILIES)


def has_pair(family):
    """Whether a family's tiles see a ``Pair``: decay families (distance and gate operands) and
    score-bias families (the causal mask, and a pair shift's key vectors and head scalars)."""
    return family in DECAY_FAMILIES or family in SCORE_BIAS_FAMILIES


def accumulates(family):
    """Whether a family's dQ and dK terms add every order's sums to the kernel's running sums
    themselves (score-bias families), through ``sums.add(index, x)``: x is a row vector of the
    tile and ``index`` the sum's position, static or traced. The kernels keep these sums in shared
    memory, so a rolled order loop neither carries nor routes them through registers."""
    return family in SCORE_BIAS_FAMILIES


class Pair(NamedTuple):
    """Per-element context of a decay family's tile: FP32 query-minus-key distance (zero on
    masked entries; None for the cumulative family, which never reads it), per-key FP32 vectors
    broadcast to the tile, in tiles crossing the diagonal the causal mask (else None), and
    per-head FP32 scalars."""
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
    if family in SCORE_BIAS_FAMILIES:
        orders, gated = degree + 1, family in GATED_SCORE_BIAS_FAMILIES
        vectors = (2 if gated else 1) * orders + degree
        return vectors, vectors, (3 if gated else 1) * orders + (3 if family in PAIR_SHIFT_FAMILIES else 1) * degree
    if family in PER_ORDER_FAMILIES:
        orders = degree + 1
        sums = {PER_ORDER_QUERY_DECAY: 2, PER_ORDER_ENDPOINT_DECAY: 3, PER_ORDER_CUMULATIVE_DECAY: 1}[family]
        return 2 * orders, 2 * orders, sums * orders
    if family in ROW_POWER_FAMILIES.values():  # the coefficients a_r, then lambda; the sums are those of query
        return degree + 2, degree + 2, degree + 1
    # Transformed families hold the query family's vectors: phi(s) and f(s) are evaluated in their tiles.
    extra = 0 if family == QUERY or family in TRANSFORMED_QUERY_FAMILIES else {QUERY_DECAY: 1, ENDPOINT_DECAY: 2}[family]
    return degree + 1 + extra, degree + 1 + extra, degree + 1 + extra


def approx_math(family):
    """Whether a family's kernels use the GPU's approximate exp2/log2 (ex2/lg2.approx: about
    2^-22 relative and absolute error). Only decay families, pair shifts (tanh from exp2), the
    softplus/sigmoid/swish transforms and every prefactor evaluate transcendentals; a row shift evaluates none,
    so the flag does not change it, and the abs transform, which evaluates none either, keeps query's compiler
    settings."""
    if family in FAMILY_SCORE_FUNCTIONS:
        transform, prefactor = FAMILY_SCORE_FUNCTIONS[family]
        return transform in ("softplus", "sigmoid", "swish") or prefactor != "none"
    return family in DECAY_FAMILIES or family in SCORE_BIAS_FAMILIES


def key_counts(family, degree):
    """(per-key vectors, per-key dK-pass sums) for a family."""
    if family in SCORE_BIAS_FAMILIES:
        count = (degree + 1 if family in GATED_SCORE_BIAS_FAMILIES else 0) + (
            degree if family in PAIR_SHIFT_FAMILIES else 0)
        return count, count
    if family in (PER_ORDER_ENDPOINT_DECAY, PER_ORDER_CUMULATIVE_DECAY):
        return degree + 1, degree + 1
    return (1, 1) if family == ENDPOINT_DECAY else (0, 0)


def head_count(family, degree):
    """Per-head FP32 scalars of a family (per-order endpoint rates a_r; a pair shift's b_r, a_r)."""
    if family in SCORE_BIAS_FAMILIES:
        return (degree + 1 if family in GATED_SCORE_BIAS_FAMILIES else 0) + (
            2 * degree if family in PAIR_SHIFT_FAMILIES else 0)
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


def transformed_score(family, s, *, slope=False):
    """(z, phi'(s) or None) of a transformed query family's tile, as attention/score_transform.py defines them;
    for a prefactor family, its base transform's (the identity's z = s, with phi' None: one, never multiplied).
    Softplus: z = max(s, 0) + ln(1 + y) with y = e^-|s| in (0, 1], from one exp2 and one log (below y = 2^-6
    the cubic series of ln(1 + y), relative error < 1e-6, replaces the approximate log, whose absolute error
    would dominate small y), and sigmoid(s) = e^(min(s, 0) - ln(1 + y)) from one more exp2: 1/2 at s = 0, and
    far below zero e^s to the FP32 rounding of its exponent (about 6e-8 |s| relative), zero once that
    underflows. Abs: z = |s| and sign(s), 0 at s = 0. z is finite for every finite score and nonfinite for every
    nonfinite one: abs gives +inf at +-inf, softplus +inf at +inf and NaN at -inf, where its limit 0 would give
    the pair the finite weight c_0, and both NaN at NaN. A causal score that overflowed therefore makes its row
    NaN, as for query and the blocked reference. Sigmoid uses y = e^-|s|, z = 1/(1+y) for s >= 0 and
    y/(1+y) otherwise, with slope y/(1+y)^2 so that a value rounded to one still has its positive-tail
    gradient. Nonfinite sigmoid inputs explicitly become NaN rather than saturating to valid weights. Swish is
    s times that sigmoid, z = s/(1+y) for s >= 0 and s y/(1+y) otherwise, with slope sigmoid(s) + s y/(1+y)^2
    (1/2 at zero); a nonfinite input is NaN, value and slope."""
    transform = FAMILY_SCORE_FUNCTIONS[family][0]
    if transform == "identity":
        return s, None
    return _transform(transform, s, None if transform == "abs" else _tail(s), slope=slope)


def _tail(s):
    """y = e^-|s| in (0, 1], from one exp2: the transforms' and prefactors' shared transcendental."""
    return jnp.exp2(jnp.abs(s) * -LOG2E)


def _log1p_tail(y):
    """ln(1 + y) for y in (0, 1]; below y = 2^-6 its cubic series (relative error < 1e-6)."""
    return jnp.where(y < _SERIES, y * (1.0 - y * (0.5 - y * (1.0 / 3.0))), jnp.log(1.0 + y))


def _transform(transform, s, y, *, slope=False):
    """(phi(s), phi'(s) or None) of a non-identity transform, given y = ``_tail(s)`` (None for abs)."""
    if transform == "abs":
        z = jnp.abs(s)
        return z, (jnp.where(s > 0.0, 1.0, 0.0) - jnp.where(s < 0.0, 1.0, 0.0) if slope else None)
    if transform == "sigmoid":
        inverse = 1.0 / (1.0 + y)
        z = jnp.where(finite(s), jnp.where(s >= 0.0, 1.0, y) * inverse, jnp.nan)
        return z, (jnp.where(finite(s), y * inverse * inverse, jnp.nan) if slope else None)
    if transform == "swish":
        inverse = 1.0 / (1.0 + y)
        logistic = jnp.where(s >= 0.0, 1.0, y) * inverse
        z = jnp.where(finite(s), s * logistic, jnp.nan)
        return z, (jnp.where(finite(s), logistic + s * y * inverse * inverse, jnp.nan) if slope else None)
    log1p = _log1p_tail(y)
    z = (s + jnp.maximum(-s, 0.0)) + log1p  # s + max(-s, 0) is max(s, 0) exactly for finite s, NaN at s = -inf
    return z, (jnp.exp2((jnp.minimum(s, 0.0) - log1p) * LOG2E) if slope else None)


def score_prefactor(family, s, *, derivative=False, y=None, z=None, slope=None):
    """(f(s), f'(s) or None) of a prefactor family's tile, as attention/score_transform.py defines them. ``y``,
    ``z`` and ``slope`` are the tile's ``_tail(s)`` and base transform value and derivative when already formed:
    with phi = softplus, f = softplus is z and f' = phi' = sigmoid(s), the sigmoid prefactor is phi' (its slope
    y/(1+y)^2), and swish is s phi'. Otherwise softplus is max(s, 0) + ln(1 + y) with f' = 2^((min(s, 0) -
    ln(1 + y)) log2(e)), sigmoid and swish take the stable logistic from y, and exp is 2^(s log2(e)) with f' = f.
    A nonfinite score gives a nonfinite or NaN f, but for exp at -inf (zero); the row is NaN regardless, since
    p(z) is nonfinite there."""
    transform, prefactor = FAMILY_SCORE_FUNCTIONS[family]
    if prefactor == "exp":
        value = jnp.exp2(s * LOG2E)
        return value, (value if derivative else None)
    if transform == "softplus" and prefactor == "softplus":
        return z, (slope if derivative else None)
    if y is None:
        y = _tail(s)
    if prefactor == "softplus":
        return _transform("softplus", s, y, slope=derivative)
    if transform == "softplus" and slope is not None:  # phi' = sigmoid(s): reuse it as the logistic
        inverse = 1.0 / (1.0 + y)
        if prefactor == "sigmoid":
            return slope, (y * inverse * inverse if derivative else None)
        return s * slope, (slope + s * y * inverse * inverse if derivative else None)
    return _transform(prefactor, s, y, slope=derivative)


def _score_functions(family, s, *, derivatives):
    """(z, phi'(s) or None, f(s), f'(s) or None) of a transformed family's tile, its transcendental shared."""
    transform, prefactor = FAMILY_SCORE_FUNCTIONS[family]
    y = None if transform in ("identity", "abs") and prefactor == "exp" else _tail(s)
    if transform == "identity":
        z, slope = s, None
    else:
        z, slope = _transform(transform, s, y, slope=derivatives or prefactor in ("sigmoid", "swish"))
    if prefactor == "none":
        return z, slope, None, None
    value, value_slope = score_prefactor(family, s, derivative=derivatives, y=y, z=z, slope=slope)
    return z, slope, value, value_slope


def _transformed_weights(family, s, vectors):
    """W = f(s) p(phi(s)) of a transformed family's tile (f = 1 without a prefactor); a row-power family's is
    sum_r a_r u^r at u = (phi(s) h) h."""
    if family in ROW_POWER_FAMILIES.values():
        degree = len(vectors) - 2
        half = vectors[degree + 1]
        return polynomial((transformed_score(family, s)[0] * half) * half, vectors[:degree + 1])
    if FAMILY_SCORE_FUNCTIONS[family][1] == "none":
        return polynomial(transformed_score(family, s)[0], vectors)
    z, _, value, _ = _score_functions(family, s, derivatives=False)
    return value * polynomial(z, vectors)


def _transformed_terms(family, degree, s, dp, vectors, *, rows):
    """(dS, dQ row sums or []) of a transformed query family's tile, dS = p'(z) (G phi'(s)). In program order the
    sums dc_r = sum_i G z^r come first, then G phi'(s), then the derivative's recurrence, which therefore holds
    neither G nor phi'(s) beside it. With a prefactor, G_f = G f(s) takes G's place in the sums and the chain
    rule, and the product rule adds G f'(s) p(z): dS = p'(z) G_f phi'(s) + G f'(s) p(z). A row-power family takes
    u = (z h) h in z's place, its sums dc_r = sum_i G u^r and its chain rule h h phi'(s)."""
    if family in ROW_POWER_FAMILIES.values():
        z, slope = transformed_score(family, s, slope=True)
        half = vectors[degree + 1]
        u = (z * half) * half
        sums = row_sums(QUERY, degree, u, dp) if rows else []
        gradient = ((dp * slope) * half) * half
        return polynomial_and_derivative(u, vectors[:degree + 1])[1] * gradient, sums
    if FAMILY_SCORE_FUNCTIONS[family][1] == "none":
        z, slope = transformed_score(family, s, slope=True)
        sums = row_sums(QUERY, degree, z, dp) if rows else []
        gradient = dp * slope
        return polynomial_and_derivative(z, vectors)[1] * gradient, sums
    z, slope, value, value_slope = _score_functions(family, s, derivatives=True)
    weighted = dp * value  # G_f
    sums = row_sums(QUERY, degree, z, weighted) if rows else []
    gradient = weighted if slope is None else weighted * slope
    polynomial_value, derivative = polynomial_and_derivative(z, vectors)
    return derivative * gradient + dp * value_slope * polynomial_value, sums


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
    """One order's ``Decay`` in a per-order family's tile, as ``decay``. The cumulative family's
    exponent is the FP32 difference L_tr - L_ir of the two endpoints' sums (exact between tokens
    whose sums are within a factor of two), times log2(e): scaling each sum before subtracting
    would round each by up to |L| log2(e) 2^-24, a weight error of 3e-4 where |L| nears 3,000 (a
    fast gate at 2,048 tokens). A future key's exponent is positive and may overflow; the causal
    mask drops it."""
    sigmoid = log2_sp = None
    if family == PER_ORDER_QUERY_DECAY:
        exponent = vectors[degree + 1 + order] * pair.distance
    elif family == PER_ORDER_CUMULATIVE_DECAY:
        exponent = (vectors[degree + 1 + order] - pair.keys[order]) * LOG2E
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
    sums (see above); otherwise the per-key sums of the endpoint and cumulative dK passes. Orders
    0 and 1 start every accumulator in its layout; later orders form a rolled loop
    (``_per_order_weights``) whose sums reach their order's accumulator through 0/1 masks. A NaN
    sum at a rolled order (an invalid gate, whose row is NaN already) therefore reaches every
    rolled order's accumulator."""
    endpoint = family == PER_ORDER_ENDPOINT_DECAY
    cumulative = family == PER_ORDER_CUMULATIVE_DECAY
    orders = degree + 1

    def evaluate(order, power):
        """Order r's weight c_r D_r and sums, given power = s^r (None for r = 0)."""
        terms = _order_decay(family, degree, vectors, pair, order, slopes=endpoint)
        weight = vectors[order] * terms.factor
        gds = dp * terms.factor if power is None else dp * terms.factor * power  # G D_r s^r
        sums = [gds.sum(axis=1)] if rows else []
        if cumulative:
            if not rows:  # the key side of L_r's gradient: sum over queries of G W_r
                gw = gds * vectors[order]
                if pair.causal is not None:  # a masked query's coefficients may be nonfinite
                    gw = jnp.where(pair.causal, gw, 0.0)
                sums.append(gw.sum(axis=1))
        elif rows or endpoint:  # query-only rates have no per-key gradients
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


def _shift_operands(family, degree):
    """Indices of order 1's shift operands in a score-bias family's (vectors, key vectors, head
    scalars): they follow the coefficients and, for gated families, the endpoint gate's inputs."""
    orders = degree + 1
    gated = family in GATED_SCORE_BIAS_FAMILIES
    return (2 if gated else 1) * orders, orders if gated else 0, orders if gated else 0


def _order_shift(family, degree, vectors, pair, order, *, slopes=False):
    """(beta_r, tau_r, 1 - tau_r^2) of order r >= 1 in a score-bias tile (tau terms None for a row
    shift; 1 - tau^2 only with ``slopes``). tau = sign(h) (1 - e) / (1 + e) with e = 2^(-2 log2(e) |h|),
    h = p_r(t) + q_r(i), and 1 - tau^2 = 4 e / (1 + e)^2 are stable for every h: tau is exactly zero
    at h = 0 and +-1 once e underflows, and NaN for a NaN input. On masked entries every term is
    zero, so z = s = 0 there and a future token's shift input never reaches a sum."""
    vector, key, head = _shift_operands(family, degree)
    tau = sech2 = None
    if family in PAIR_SHIFT_FAMILIES:
        h = vectors[vector + order - 1] + pair.keys[key + order - 1]
        e = jnp.exp2(jnp.abs(h) * (-2.0 * LOG2E))
        inverse = 1.0 / (1.0 + e)
        magnitude = (1.0 - e) * inverse
        tau = jnp.where(h < 0.0, -magnitude, magnitude)
        beta = pair.heads[head + order - 1] + pair.heads[head + degree + order - 1] * tau
        if slopes:
            sech2 = 4.0 * e * (inverse * inverse)
    else:
        beta = vectors[vector + order - 1]
    if pair.causal is not None:
        beta = jnp.where(pair.causal, beta, 0.0)
        if tau is not None:
            tau = jnp.where(pair.causal, tau, 0.0)
        if sech2 is not None:
            sech2 = jnp.where(pair.causal, sech2, 0.0)
    return beta, tau, sech2


def _order_terms(family, degree, vectors, pair, order, *, slopes=False):
    """(c_r G_r, ``Decay`` or None) of one order in a score-bias tile; c_r G_r is zero on masked
    entries, where a transposed tile's earlier query may hold any coefficient."""
    gate = None
    weight = vectors[order]
    if family in GATED_SCORE_BIAS_FAMILIES:  # the per-order endpoint gate, laid out as that family's
        gate = _order_decay(PER_ORDER_ENDPOINT_DECAY, degree, vectors, pair, order, slopes=slopes)
        weight = weight * gate.factor
    if pair.causal is not None:
        weight = jnp.where(pair.causal, weight, 0.0)
    return weight, gate


def _shifted_powers(s, beta, order):
    """(z^(r-1), z^r) of z = s + beta for a rolled order r >= 2: r - 1 multiplications."""
    z = s + beta
    previous = lax.fori_loop(0, order - 2, lambda _, power: power * z, z)
    return previous, previous * z


def _score_bias_weights(family, degree, s, vectors, pair):
    """W = c_0 G_0 + sum_{r >= 1} c_r G_r (s + beta_r)^r, one order per iteration of a rolled loop
    (as ``_per_order_weights``), each power formed from its own base."""
    def term(order):
        return _order_terms(family, degree, vectors, pair, order)[0]

    def body(order, w):
        beta, _, _ = _order_shift(family, degree, vectors, pair, order)
        power = _shifted_powers(s, beta, order)[1]  # before the gate: its tiles need not outlive the powers
        return w + term(order) * power

    beta, _, _ = _order_shift(family, degree, vectors, pair, 1)
    return lax.fori_loop(2, degree + 1, body, term(0) + term(1) * (s + beta))


def _score_bias_backward(family, degree, s, dp, vectors, pair, sums, *, rows, weights=False):
    """(dS, W or None) of a score-bias tile in one ascending pass over the orders: dS = G
    sum_r r c_r G_r z_r^(r-1), whose order-r term is dbeta_r = G r c_r G_r z_r^(r-1), and W. Each
    order's dQ (``rows``) or dK sums (see above) are added to the kernel's running sums as they
    are formed (``accumulates``). Orders 0 and 1 are evaluated explicitly; later orders form a
    rolled loop (as ``_per_order_backward``) that carries only dS's factor and W."""
    gated = family in GATED_SCORE_BIAS_FAMILIES
    paired = family in PAIR_SHIFT_FAMILIES
    orders = degree + 1
    # Index of order 0 in each kind of sums (shift kinds start at order 1).
    if rows:
        gate_bases = (orders, 2 * orders) if gated else ()
        start = (3 if gated else 1) * orders - 1
        shift_bases = tuple(start + j * degree for j in range(3 if paired else 1))
    else:
        gate_bases = (0,) if gated else ()
        shift_bases = (orders - 1,) if paired and gated else (-1,) if paired else ()

    def evaluate(order, previous, power, beta_terms):
        """Order r's weight c_r G_r, derivative term r c_r G_r z^(r-1) (None for r = 0), and
        its (index of order 0 in a kind, sum) pairs, given z^(r-1) (None: one) and z^r."""
        weight, gate = _order_terms(family, degree, vectors, pair, order, slopes=gated)
        static = isinstance(order, int)
        term = None
        if not (static and order == 0):
            term = weight if previous is None else weight * previous
            if not static:
                term = term * order.astype(jnp.float32)
            elif order > 1:
                term = term * float(order)
        contributions = []
        if rows:
            gds = dp if gate is None else dp * gate.factor
            if power is not None:
                gds = gds * power
            contributions.append((0, gds.sum(axis=1)))  # dc_r
        if gated:
            gwd = dp * weight * pair.distance if power is None else dp * power * weight * pair.distance
            contributions.append((gate_bases[0], (gwd * gate.sigmoid).sum(axis=1)))
            if rows:
                contributions.append((gate_bases[1], (gwd * gate.log2_softplus).sum(axis=1)))
        if term is not None and shift_bases:
            _, tau, sech2 = beta_terms
            dbeta = dp * term
            if paired:
                contributions.append((shift_bases[0], (dbeta * sech2).sum(axis=1)))
                if rows:
                    contributions.append((shift_bases[1], (dbeta * tau).sum(axis=1)))
                    contributions.append((shift_bases[2], dbeta.sum(axis=1)))
            else:
                contributions.append((shift_bases[0], dbeta.sum(axis=1)))
        return weight, term, contributions

    def add(order, contributions):
        for base, x in contributions:
            sums.add(base + order, x)

    weight0, _, contributions = evaluate(0, None, None, None)
    add(0, contributions)
    terms1 = _order_shift(family, degree, vectors, pair, 1, slopes=paired)
    z1 = s + terms1[0]
    weight1, derivative, contributions = evaluate(1, None, z1, terms1)
    add(1, contributions)
    value = weight0 + weight1 * z1 if weights else None

    def body(order, carry):
        derivative, value = carry
        terms = _order_shift(family, degree, vectors, pair, order, slopes=paired)
        previous, power = _shifted_powers(s, terms[0], order)
        weight, term, contributions = evaluate(order, previous, power, terms)
        add(order, contributions)
        return derivative + term, None if value is None else value + weight * power

    derivative, value = lax.fori_loop(2, orders, body, (derivative, value))
    return dp * derivative, value


def forward_weights(family, degree, s, vectors, pair=None, terms=None):
    """Weights of a tile; decay families may pass the tile's ``Decay`` as ``terms``."""
    if family in SCORE_BIAS_FAMILIES:
        return _score_bias_weights(family, degree, s, vectors, pair)
    if family in PER_ORDER_FAMILIES:
        return _per_order_weights(family, degree, s, vectors, pair)
    if family in DECAY_FAMILIES:
        terms = decay(family, degree, vectors, pair) if terms is None else terms
        return polynomial(s, vectors[:degree + 1]) * terms.factor
    if family in FAMILY_SCORE_FUNCTIONS:
        return _transformed_weights(family, s, vectors)
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
    if family in SCORE_BIAS_FAMILIES:
        return _score_bias_weights(family, degree, s, vectors, pair)
    if family in PER_ORDER_FAMILIES:
        return _per_order_weights(family, degree, s, vectors, pair)
    if family in DECAY_FAMILIES:
        return forward_weights(family, degree, s, vectors, pair, terms)
    if family == TAYLOR:
        return taylor(s, degree) * vectors[1]
    if family in FAMILY_SCORE_FUNCTIONS:
        return _transformed_weights(family, s, vectors)
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


def dq_terms(family, degree, s, dp, vectors, pair=None, sums=None):
    """(dS, dQ-kernel row sums) of a tile; families that ``accumulates`` add the tile's sums to the
    running ``sums`` instead and return (dS, None)."""
    if family in SCORE_BIAS_FAMILIES:
        return _score_bias_backward(family, degree, s, dp, vectors, pair, sums, rows=True)[0], None
    if family in PER_ORDER_FAMILIES:
        ds, _, sums = _per_order_backward(family, degree, s, dp, vectors, pair, rows=True)
        return ds, sums
    if family in FAMILY_SCORE_FUNCTIONS:
        return _transformed_terms(family, degree, s, dp, vectors, rows=True)
    if family not in DECAY_FAMILIES:
        return score_gradient(family, degree, s, dp, vectors), row_sums(family, degree, s, dp)
    ds, gwd, gd, terms = _decay_gradient(family, degree, s, dp, vectors, pair)
    sums = row_sums(QUERY, degree, s, gd)
    if family == QUERY_DECAY:
        return ds, [*sums, gwd.sum(axis=1)]
    return ds, [*sums, (gwd * terms.sigmoid).sum(axis=1), (gwd * terms.log2_softplus).sum(axis=1)]


def dk_terms(family, degree, s, dp, vectors, pair=None, terms=None, sums=None):
    """(dS, per-key sums over the tile's queries) of a transposed dK tile; ``terms`` may hold
    the tile's ``Decay`` from ``decay(..., slopes=True)``. Families that ``accumulates`` add the
    tile's sums to the running ``sums`` instead and return (dS, None)."""
    if family in SCORE_BIAS_FAMILIES:
        return _score_bias_backward(family, degree, s, dp, vectors, pair, sums, rows=False)[0], None
    if family in PER_ORDER_FAMILIES:
        ds, _, sums = _per_order_backward(family, degree, s, dp, vectors, pair, rows=False)
        return ds, sums
    if family in FAMILY_SCORE_FUNCTIONS:
        return _transformed_terms(family, degree, s, dp, vectors, rows=False)
    if family not in DECAY_FAMILIES:
        return score_gradient(family, degree, s, dp, vectors), []
    ds, gwd, _, terms = _decay_gradient(family, degree, s, dp, vectors, pair, terms)
    return ds, ([(gwd * terms.sigmoid).sum(axis=1)] if family == ENDPOINT_DECAY else [])


def dkv_terms(family, degree, s, dp, vectors, pair, sums=None):
    """(W, dS, per-key sums) of a per-order family's transposed tile in one pass, for kernels that
    form dV and dK together after S and dP; families that ``accumulates`` add the tile's sums to
    the running ``sums`` instead and return (W, dS, None)."""
    if family in SCORE_BIAS_FAMILIES:
        ds, w = _score_bias_backward(family, degree, s, dp, vectors, pair, sums, rows=False, weights=True)
        return w, ds, None
    ds, w, sums = _per_order_backward(family, degree, s, dp, vectors, pair, rows=False, weights=True)
    return w, ds, sums
