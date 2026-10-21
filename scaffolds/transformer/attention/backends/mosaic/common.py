"""Elementwise helpers shared by the Hopper and Blackwell Mosaic GPU Taylor kernels."""
import math

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


def blackwell_dkv_stages(width, value_width, vector_count, time, *, block_q=64, block_k=128, warpgroups=2):
    """Q/dO ring depth (3, else 2) of Blackwell's dK/dV kernel, which keeps every query's FP32
    vectors in shared memory beside its BF16 K/V block; None when even two stages do not fit."""
    fixed = 2 * (warpgroups * block_k * (width + value_width) + vector_count * time * 2)
    for stages in (3, 2):
        if fixed + stages * 2 * block_q * (width + value_width) <= BLACKWELL_SMEM_BYTES:
            return stages
    return None


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
TAYLOR, QUERY = "taylor", "query"


def vector_counts(family, degree):
    """(forward vectors, backward vectors, dQ row sums) for a family."""
    if family == TAYLOR:
        return 0, 2, 0
    return degree + 1, degree + 1, degree + 1


def forward_weights(family, degree, s, vectors):
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


def value_weights(family, degree, s, vectors):
    """W for the dV product."""
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
