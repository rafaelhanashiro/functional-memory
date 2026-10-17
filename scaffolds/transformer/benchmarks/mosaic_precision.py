"""Error of Mosaic polynomial attention at both weight precisions, and of cuDNN softmax,
against independent NumPy FP64 references on the same inputs.

For each case: the worst relative L2 error over (batch, head) and the worst max-element
error relative to the reference's largest element, for the output and every VJP. Inputs are
the GPU tests' (q, k ~ 0.7 N(0, 1)); ``sharp`` multiplies q by 4, so scores reach about +-8
and odd Taylor degrees put large negative weight on many keys.

    python -m scaffolds.transformer.benchmarks.mosaic_precision --output errors.json   # H100 or B300
"""
import argparse
import json
import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends.mosaic.api import WIDTHS, query_polynomial_attention, taylor_attention
from scaffolds.transformer.attention.backends.mosaic.common import PRECISIONS
from scaffolds.transformer.attention.softmax import padded_dot_product_attention
from scaffolds.transformer.tests.test_fused_attention import _compiled_forward_and_vjp, _inputs
from scaffolds.transformer.tests.test_mosaic_query import _inputs as _query_inputs, _numpy_fp64 as _query_fp64
from scaffolds.transformer.tests.test_mosaic_taylor import _numpy_fp64_forward_and_vjp as _taylor_fp64


def _softmax_fp64(operands, cotangent):
    q, k, v, do = (np.asarray(x.astype(jnp.float32), np.float64) for x in (*operands, cotangent))
    time, width = q.shape[1], q.shape[-1]
    s = np.einsum("bthd,bshd->bhts", q, k) / math.sqrt(width)
    s = np.where(np.tril(np.ones((time, time), bool)), s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    out = np.einsum("bhts,bshd->bthd", p, v)
    do_h = do.transpose(0, 2, 1, 3)
    dp = np.einsum("bhtd,bshd->bhts", do_h, v)
    ds = p * (dp - np.einsum("bhtd,bthd->bht", do_h, out)[..., None])
    return out, (np.einsum("bhts,bshd->bthd", ds, k) / math.sqrt(width),
                 np.einsum("bhts,bthd->bshd", ds, q) / math.sqrt(width),
                 np.einsum("bhts,bhtd->bshd", p, do_h))


def _errors(got, reference):
    """(worst relative L2 over batch/head, worst max error / max |reference|) per array."""
    rows = []
    for actual, want in zip((got[0], *got[1]), (reference[0], *reference[1])):
        actual = np.asarray(jnp.asarray(actual, jnp.float32), np.float64)
        l2, peak = 0.0, 0.0
        for b in range(want.shape[0]):
            for h in range(want.shape[2]):
                a, w = actual[b, :, h], want[b, :, h]
                if np.linalg.norm(w) == 0:
                    continue
                l2 = max(l2, np.linalg.norm(a - w) / np.linalg.norm(w))
                peak = max(peak, np.max(np.abs(a - w)) / np.max(np.abs(w)))
        rows.append((float(l2), float(peak)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {"device": jax.devices()[0].device_kind, "cases": []}

    def record(family, degree, width, shape_name, scores, precision, names, errors):
        case = dict(family=family, degree=degree, width=width, shape=shape_name, scores=scores, precision=precision,
                    errors={n: {"relative_l2": e[0], "max_over_peak": e[1]} for n, e in zip(names, errors)})
        results["cases"].append(case)
        print(json.dumps(case), flush=True)

    shapes = {"tile": lambda w: (1, 512, 2, w, 64), "production": lambda w: (1, 2048, 4, w, 64)}
    taylor_names = ("out", "dq", "dk", "dv")
    softmax = partial(padded_dot_product_attention, backend="cudnn")  # the production softmax path
    for shape_name, widths in (("tile", WIDTHS), ("production", (64,))):
        for width in widths:
            for scores in ("normal", "sharp"):
                operands, cotangent = _inputs(shapes[shape_name](width), dtype=jnp.bfloat16, seed=94)
                if scores == "sharp":
                    operands = (operands[0] * 4, *operands[1:])
                record("softmax", None, width, shape_name, scores, "cudnn", taylor_names,
                       _errors(_compiled_forward_and_vjp(softmax)(operands, cotangent),
                               _softmax_fp64(operands, cotangent)))
                for degree in (1, 2, 3, 4, 8, 16):
                    reference = _taylor_fp64(operands, cotangent, degree)
                    for precision in PRECISIONS:
                        attend = partial(taylor_attention, attention_degree=degree, precision=precision)
                        record("taylor", degree, width, shape_name, scores, precision, taylor_names,
                               _errors(_compiled_forward_and_vjp(attend)(operands, cotangent), reference))
        for degree in (1, 2, 3, 4):
            time, heads = shapes[shape_name](64)[1:3]
            operands, cotangent = _query_inputs(degree, 64, time=time, heads=heads)
            reference = _query_fp64(operands, cotangent)
            for precision in PRECISIONS:
                attend = partial(query_polynomial_attention, precision=precision)
                record("query", degree, 64, shape_name, "normal", precision, taylor_names + ("dc",),
                       _errors(_compiled_forward_and_vjp(attend)(operands, cotangent), reference))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
