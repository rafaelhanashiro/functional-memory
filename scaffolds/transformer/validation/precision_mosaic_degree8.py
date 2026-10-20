"""Worst relative L2 error against NumPy FP64 of query-coefficient attention at degree 8 and,
for comparison, degree 3, at both weight precisions, with the GPU tests' positive coefficients
and with signed ones (each c_r of either sign, at the Taylor magnitude 1/r!).

    python scaffolds/transformer/validation/precision_mosaic_degree8.py   # in an H100 or B300 allocation
"""
import math
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
from scaffolds.transformer.benchmarks.mosaic_precision import _errors
from scaffolds.transformer.tests.test_fused_attention import _compiled_forward_and_vjp
from scaffolds.transformer.tests.test_mosaic_query import _inputs, _numpy_fp64

print(jax.devices()[0].device_kind)
for degree in (3, 8):
    for signed in (False, True):
        for time, heads in ((512, 2), (2048, 4)):
            operands, cotangent = _inputs(degree, 64, time=time, heads=heads)
            if signed:  # learned coefficients of either sign, magnitudes like the Taylor ones
                rng = np.random.default_rng(5 + degree)
                taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)])
                c = jnp.asarray(taylor * rng.uniform(-1.5, 1.5, (1, time, heads, degree + 1)), jnp.float32)
                operands = (*operands[:3], c)
            reference = _numpy_fp64(operands, cotangent)
            for precision in ("bf16", "bf16x3"):
                errors = _errors(_compiled_forward_and_vjp(partial(query_polynomial_attention, precision=precision))(operands, cotangent), reference)
                print(f"m={degree} {'signed' if signed else 'positive'} T={time} {precision:6s} " + "  ".join(
                    f"{n} {e[0]:.1e}" for n, e in zip(("out", "dq", "dk", "dv", "dc"), errors)), flush=True)
