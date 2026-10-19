"""Framework-independent operands for upstream attention timing comparisons."""
from __future__ import annotations

import numpy as np


def kernel_inputs(batch: int, length: int, heads: int, width: int, seed: int = 46,
                  *, value_width: int | None = None):
    """Return reproducible FP32 operands; framework casts happen outside timing.

    GDN uses unit Q/K, sigmoid beta/erase, and nonpositive log decay. Dot-product
    attention uses ``q * width`` to give order-one scaled scores, also outside
    timing. Both frameworks receive identical arrays from the same NumPy version.
    """
    value_width = width if value_width is None else value_width
    if min(batch, length, heads, width, value_width) < 1:
        raise ValueError("Operand dimensions must be positive")
    rng = np.random.default_rng(seed)
    shape = (batch, length, heads, width)
    q, k = (rng.standard_normal(shape, dtype=np.float32) for _ in range(2))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    value_shape = (batch, length, heads, value_width)
    v = rng.standard_normal(value_shape, dtype=np.float32)
    rate = np.exp(rng.uniform(0, np.log(16), (1, 1, heads)).astype(np.float32))
    g = -rate * np.logaddexp(0, rng.standard_normal(shape[:-1], dtype=np.float32) * .5 - 4)
    beta, erase = (1 / (1 + np.exp(-rng.standard_normal(shape[:-1], dtype=np.float32)))
                   for _ in range(2))
    do = rng.standard_normal(value_shape, dtype=np.float32)
    return dict(q=q, k=k, v=v, do=do, g=g, beta=beta, erase=erase)


def polynomial_coefficients(batch: int, length: int, heads: int, degree: int = 3, seed: int = 46):
    """Signed tanh write coefficients, independent of the shared linear-GDN fixture.

    A separate generator preserves every shared operand hash when adding this
    polynomial-only input. Coefficient generation is outside all timed calls.
    """
    if min(batch, length, heads) < 1 or degree not in (2, 3):
        raise ValueError("Polynomial fixtures require positive dimensions and degree two or three")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x504F4C59]))
    return np.tanh(.2 * rng.standard_normal((batch, length, heads, degree + 1), dtype=np.float32))
