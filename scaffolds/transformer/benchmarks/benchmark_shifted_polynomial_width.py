"""Time the row-power shifted kernel (attention/backends/fused_shifted_polynomial.py) on the allocated GPU at a training
micro-batch: 16 x 2,048 tokens and 14 heads by default, Q/K width 64, value width 64, BF16 Q/K/V and per-query FP32
coefficients tanh(g + atanh(1/2))/r!, through head RMSNorm's max-abs prescale. Prints one JSON object: the median
forward and forward-plus-backward milliseconds per degree, after warm-up.

    python scaffolds/transformer/benchmarks/benchmark_shifted_polynomial_width.py --degrees 8 16
"""
import argparse
import json
import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.output_scaling import max_abs_head_rms
from scaffolds.transformer.attention.shifted_polynomial import shifted_polynomial_attention


def _milliseconds(function, args, repeats):
    jax.block_until_ready(function(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(*args))
        samples.append(1e3 * (time.perf_counter() - start))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrees", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--value-width", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    rows = (args.batch, args.tokens, args.heads)
    # Scores of standard deviation 3 at unit-variance keys, as QK RMSNorm with a learned scale gives.
    q = jnp.asarray(3 * rng.normal(size=(*rows, args.width)), jnp.bfloat16)
    k = jnp.asarray(rng.normal(size=(*rows, args.width)), jnp.bfloat16)
    v = jnp.asarray(rng.normal(size=(*rows, args.value_width)), jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(*rows, args.value_width)), jnp.float32)
    report = {"device": jax.devices()[0].device_kind, "jax": jax.__version__, "shape": vars(args), "degrees": {}}
    for degree in args.degrees:
        factorials = np.array([math.factorial(r) for r in range(degree + 1)])
        c = jnp.asarray(np.tanh(rng.normal(size=(*rows, degree + 1)) + math.atanh(0.5)) / factorials, jnp.float32)

        def attention(q, k, v, c):
            result, shift = shifted_polynomial_attention(q, k, v, c, transform="identity", backend="fused")
            return max_abs_head_rms(result, 1e-6, shift[..., None])

        def loss(q, k, v, c):
            return jnp.sum(attention(q, k, v, c).astype(jnp.float32) * cotangent)

        forward = jax.jit(attention)
        backward = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))
        report["degrees"][degree] = {
            "forward_ms": round(_milliseconds(forward, (q, k, v, c), args.repeats), 3),
            "forward_backward_ms": round(_milliseconds(backward, (q, k, v, c), args.repeats), 3),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
