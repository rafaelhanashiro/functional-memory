"""Elementwise helpers shared by the Hopper and Blackwell Mosaic GPU Taylor kernels."""
import math

import jax.numpy as jnp
from jax import lax


def truncate(x):
    """Clear the low 16 bits of an FP32 value: an exact BF16 value with a same-sign residual."""
    bits = lax.bitcast_convert_type(x, jnp.uint32)
    return lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), jnp.float32)


def bf16_parts(x):
    """Three BF16 components whose FP32 sum reproduces x (as in the Triton kernels)."""
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
