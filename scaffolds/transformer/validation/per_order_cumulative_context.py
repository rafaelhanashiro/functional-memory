"""Mosaic per-order cumulative kernels against the causal strips at the production context.

One micro-batch of the degree-twelve per-order cumulative recipe's attention, [16, 2048, 14, 64]:
BF16 Q/K/V with unit-RMS rows (QK RMSNorm's scale; ``sharp`` doubles Q), FP32 query coefficients
near Taylor, and each order's FP32 cumulative sum L of per-token log alpha in two regimes:

- ``initialization``: the recipe's gate as drawn (``forget_gate.log_decay`` of its initializers'
  A_log, dt_bias and w_alpha on unit-RMS inputs of width 896, every order a copy of one draw);
- ``trained``: per head and order a mean |log alpha| log-uniform on [1e-3, 70] and a log-normal
  token spread sigma uniform on [0.7, 2.3], the range of a trained per-order model's gates (the
  degree-eight per-order cumulative run's final diagnostics: means -71 to -1e-5, median -0.52,
  spreads 0.7-2.3 between the 5th and 95th percentiles), so |L| reaches about 1e5.

``attention_core`` runs both backends as the model calls them (BF16 Q/K/V; Mosaic's BF16 weights,
the strips' FP32 arithmetic). For the output and the gradients of Q, K, V, the coefficients and L,
the record holds the relative L2 difference of Mosaic from the strips and each implementation's
error against FP64 over the micro-batch, beside the BF16 rounding floor (the FP64 values rounded
to BF16). The FP32 gradients are compared order by order (the worst order is reported); L's
against the larger of its own norm and its query side c dc, which closed gates cancel. The FP64
reference is test_per_order_cumulative's analytic oracle evaluated on the GPU in FP64, one head
at a time; one head of the first case is checked against that host oracle itself. H100's
per-order query-distance kernels, given the same per-token log alpha, are compared with the
strips the same way for reference.

    # In one H100 allocation, from the worktree root:
    PYTHONPATH=$PWD python scaffolds/transformer/validation/per_order_cumulative_context.py --output record.json
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.forget_gate import log_decay
from scaffolds.transformer.attention.parameters import RANDOM_INITIALIZERS
from scaffolds.transformer.tests.test_per_order_cumulative import _numpy_vjp

BATCH, TIME, HEADS, WIDTH, DEGREE, HIDDEN = 16, 2048, 14, 64, 12, 896
NAMES = ("output", "dq", "dk", "dv", "coefficients", "log_decay")
BF16_NAMES = NAMES[:4]


def _initial_rates(rng, orders):
    """Per-token log alpha [B, T, H, R] of the recipe's gate at initialization on unit-RMS inputs."""
    keys = jax.random.split(jax.random.PRNGKey(int(rng.integers(2 ** 31))), 3)
    kernel = RANDOM_INITIALIZERS["uniform_fan_in"](keys[0], (HEADS, 1, HIDDEN))
    log_rate = RANDOM_INITIALIZERS["mamba2_log_rate"](keys[1], (HEADS, 1, 1))
    dt_bias = RANDOM_INITIALIZERS["mamba2_dt_bias"](keys[2], (HEADS, 1, 1))
    repeat = lambda x: jnp.repeat(x, orders, axis=1)  # every order starts as a copy of the shared gate
    x = jnp.asarray(rng.standard_normal((BATCH, TIME, HIDDEN)), jnp.float32)
    return log_decay(x, repeat(kernel), repeat(log_rate), repeat(dt_bias))


def _trained_rates(rng, orders):
    """Per-token log alpha [B, T, H, R]: log-normal magnitudes with a log-uniform mean per head and order."""
    mean = np.exp(rng.uniform(math.log(1e-3), math.log(70.0), (HEADS, orders)))
    sigma = rng.uniform(0.7, 2.3, (HEADS, orders))
    magnitude = np.exp(np.log(mean) - sigma ** 2 / 2 + sigma * rng.standard_normal((BATCH, TIME, HEADS, orders)))
    return jnp.asarray(-magnitude, jnp.float32)


def _inputs(regime, sharp, seed):
    rng = np.random.default_rng(seed)
    orders = DEGREE + 1
    q, k = (jnp.asarray(rng.standard_normal((BATCH, TIME, HEADS, WIDTH)), jnp.bfloat16) for _ in range(2))
    if sharp:
        q = q * 2  # exact in BF16: scores of standard deviation 2
    v, cotangent = (jnp.asarray(rng.standard_normal((BATCH, TIME, HEADS, 64)), jnp.bfloat16) for _ in range(2))
    taylor = np.asarray([1 / math.factorial(r) for r in range(orders)])
    c = jnp.asarray(taylor * (1 + 0.3 * np.tanh(rng.standard_normal((BATCH, TIME, HEADS, orders)))), jnp.float32)
    rates = (_initial_rates if regime == "initialization" else _trained_rates)(rng, orders)
    return (q, k, v, c), rates, cotangent


def _attend(backend, mode):
    def attend(q, k, v, c, gate):
        return attention_core(q, k, v, backend=backend, attention_degree=DEGREE, attention_normalization=False,
                              query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                              query_log_decay=gate, query_decay_mode=mode, blocked_implementation="strips")

    def output_and_gradients(operands, cotangent):
        output, pullback = jax.vjp(attend, *operands)
        return (output, *pullback(cotangent))

    return jax.jit(output_and_gradients)


def _head_vjp(q, k, v, c, sums, do):
    """One head's output and analytic VJP (dQ, dK, dV, dc, dL), test_per_order_cumulative._numpy_vjp's
    formulas: q, k [T, d], v and do [T, dv], c and sums [T, R], in the inputs' precision."""
    time_, orders = q.shape[0], c.shape[-1]
    causal = jnp.arange(time_)[:, None] >= jnp.arange(time_)[None, :]
    scores = q @ k.T / math.sqrt(q.shape[-1])
    dweights = jnp.where(causal, do @ v.T, 0.0)
    weights = dscores = 0.0
    power, previous, dc, dsums = jnp.ones_like(scores), None, [], []
    for r in range(orders):
        factor = jnp.where(causal, jnp.exp(jnp.where(causal, sums[:, None, r] - sums[None, :, r], 0.0)), 0.0)
        weights = weights + c[:, r, None] * power * factor
        term = dweights * power * factor
        dc.append(term.sum(axis=-1))
        if r:
            dscores = dscores + dweights * c[:, r, None] * r * previous * factor
        pair = c[:, r, None] * term  # L_tr enters every pair (t, i) with sign +1 as the query and -1 as the key
        dsums.append(pair.sum(axis=-1) - pair.sum(axis=0))
        previous, power = power, power * scores
    return (weights @ v, dscores @ k / math.sqrt(q.shape[-1]), dscores.T @ q / math.sqrt(q.shape[-1]),
            weights.T @ do, jnp.stack(dc, -1), jnp.stack(dsums, -1))


def _fp64(operands, sums, cotangent):
    """FP64 output and gradients [B, T, H, ...] on the GPU, one head at a time."""
    host = [np.asarray(x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, np.float64)
            for x in (*operands, sums, cotangent)]
    result = [np.zeros(shape, np.float64) for shape in (host[2].shape, host[0].shape, host[1].shape, host[2].shape,
                                                       host[3].shape, host[4].shape)]
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        head_vjp = jax.jit(_head_vjp)
        for b in range(host[0].shape[0]):
            for h in range(host[0].shape[2]):
                parts = head_vjp(*(jnp.asarray(x[b, :, h], jnp.float64) for x in host))
                for array, part in zip(result, parts, strict=True):
                    array[b, :, h] = np.asarray(part)
    return tuple(result)


def _as64(x):
    return np.asarray(x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, np.float64)


def _differences(actual, expected, coefficients):
    """Relative L2 of each quantity; FP32 gradients also per order (and the worst order), the log-decay
    gradient's against the larger of its norm and c dc's (the expected coefficient gradient)."""
    actual = tuple(_as64(x) for x in actual)
    query_side = np.asarray(coefficients, np.float64) * expected[4]
    result = {}
    for name, got, want in zip(NAMES, actual, expected, strict=True):
        entry = {"finite": bool(np.isfinite(got).all()),
                 "relative_l2": float(np.linalg.norm(got - want) / np.linalg.norm(want))}
        if name not in BF16_NAMES:
            per_order = []
            for order in range(want.shape[-1]):
                norm = np.linalg.norm(want[..., order])
                if name == "log_decay":
                    norm = max(norm, np.linalg.norm(query_side[..., order]))
                per_order.append(0.0 if norm == 0 else float(np.linalg.norm(got[..., order] - want[..., order]) / norm))
            entry["worst_order_relative_l2"] = max(per_order)
            entry["per_order_relative_l2"] = per_order
        result[name] = entry
    return result


def _floor(expected):
    """Relative L2 of the FP64 output and Q/K/V gradients rounded to BF16 (as the kernels return them)."""
    return {name: float(np.linalg.norm(_as64(jnp.asarray(want, jnp.bfloat16)) - want) / np.linalg.norm(want))
            for name, want in zip(BF16_NAMES, expected)}


def _summary(differences):
    return {name: round(value.get("worst_order_relative_l2", value["relative_l2"]), 9)
            for name, value in differences.items()}


def _oracle_check(operands, sums, cotangent, expected):
    """The GPU FP64 oracle against the host oracle on the first head of the first sequence."""
    q, k, v = (x[:1, :, :1].astype(jnp.float32) for x in operands[:3])
    output, gradients = _numpy_vjp(q, k, v, operands[3][:1, :, :1], sums[:1, :, :1],
                                   cotangent[:1, :, :1].astype(jnp.float32))
    return max(float(np.max(np.abs(got[:1, :, :1] - want)) / max(float(np.max(np.abs(want))), 1e-300))
               for got, want in zip(expected, (output, *gradients), strict=True))


def _case(regime, sharp, seed, mosaic, strips, check_oracle=False):
    operands, rates, cotangent = _inputs(regime, sharp, seed)
    sums = jnp.cumsum(rates, axis=1)  # each order's FP32 cumulative sum, as the model forms it
    got = mosaic((*operands, sums), cotangent)
    reference = tuple(_as64(x) for x in strips((*operands, sums), cotangent))
    expected = _fp64(operands, sums, cotangent)
    c = np.asarray(operands[3], np.float64)
    record = {
        "regime": regime, "scores": "sharp (Q doubled)" if sharp else "unit scale",
        "log_alpha_quantiles": {str(p): float(x) for p, x in zip(
            (0, 1, 50, 99, 100), np.percentile(np.asarray(rates), (0, 1, 50, 99, 100)))},
        "log_decay_min": float(jnp.min(sums)),
        "mosaic_vs_strips": _differences(got, reference, c),
        "bf16_floor": _floor(expected),
        "mosaic_vs_fp64": _differences(got, expected, c),
        "strips_vs_fp64": _differences(reference, expected, c),
    }
    if check_oracle:
        record["fp64_oracle_max_relative_deviation_from_host_oracle"] = _oracle_check(operands, sums, cotangent,
                                                                                      expected)
    print(regime, record["scores"], json.dumps({
        "L_min": record["log_decay_min"], "floor": record["bf16_floor"],
        **{key: _summary(record[key]) for key in ("mosaic_vs_strips", "mosaic_vs_fp64", "strips_vs_fp64")},
        **({"oracle_check": record["fp64_oracle_max_relative_deviation_from_host_oracle"]} if check_oracle else {})}),
        flush=True)
    return record, (operands, rates, cotangent)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = jax.devices()[0]
    if device.platform != "gpu" or device.compute_capability != "9.0":
        raise SystemExit("Run in an H100 allocation: the per-order cumulative kernels run on H100 alone")
    mosaic, strips = _attend("mosaic", "cumulative"), _attend("blocked", "cumulative")
    cases = []
    for seed, (regime, sharp) in enumerate((("initialization", False), ("trained", False), ("trained", True))):
        record, (operands, rates, cotangent) = _case(regime, sharp, seed, mosaic, strips, check_oracle=seed == 0)
        cases.append(record)
    # The per-order query-distance kernels (the qualified family) on the last case's per-token log alpha.
    distance = _attend("mosaic", "query_distance")((*operands, rates), cotangent)
    reference = _attend("blocked", "query_distance")((*operands, rates), cotangent)
    query_distance = _differences(distance, tuple(_as64(x) for x in reference), np.asarray(operands[3], np.float64))
    print("per-order query distance, mosaic vs strips", json.dumps(_summary(query_distance)), flush=True)
    report = {
        "format": "per-order-cumulative-mosaic-context", "device": device.device_kind,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
        "jax": jax.__version__,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        # Scaffold files, tracked or new, except the validation records checks write.
        "source_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", "scaffolds",
                                             ":(exclude)scaffolds/transformer/validation/runs"],
                                            capture_output=True, text=True).stdout.strip()),
        "shape": {"batch": BATCH, "time": TIME, "heads": HEADS, "width": WIDTH, "value_width": 64, "degree": DEGREE},
        "cases": cases,
        "per_order_query_distance_mosaic_vs_strips": {"case": "trained, sharp", **query_distance},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
