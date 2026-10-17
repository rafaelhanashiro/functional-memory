"""Framework-independent operands for upstream attention timing comparisons."""
from __future__ import annotations

import numpy as np


def kernel_inputs(batch: int, length: int, heads: int, width: int, seed: int = 46):
    """Return reproducible FP32 operands; framework casts happen outside timing.

    GDN uses unit Q/K, sigmoid beta/erase, and nonpositive log decay. Dot-product
    attention uses ``q * width`` to give order-one scaled scores, also outside
    timing. Both frameworks receive identical arrays from the same NumPy version.
    """
    if min(batch, length, heads, width) < 1:
        raise ValueError("Operand dimensions must be positive")
    rng = np.random.default_rng(seed)
    shape = (batch, length, heads, width)
    q, k = (rng.standard_normal(shape, dtype=np.float32) for _ in range(2))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal(shape, dtype=np.float32)
    rate = np.exp(rng.uniform(0, np.log(16), (1, 1, heads)).astype(np.float32))
    g = -rate * np.logaddexp(0, rng.standard_normal(shape[:-1], dtype=np.float32) * .5 - 4)
    beta, erase = (1 / (1 + np.exp(-rng.standard_normal(shape[:-1], dtype=np.float32)))
                   for _ in range(2))
    do = rng.standard_normal(shape, dtype=np.float32)
    return dict(q=q, k=k, v=v, do=do, g=g, beta=beta, erase=erase)
