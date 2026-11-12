"""Throughput of forget-gate attention on the blocked and Mosaic backends.

``attention`` times the attention operation alone (warmed forward, and forward plus the VJP
of every operand) at the production per-GPU shape: 16 sequences of 2,048 tokens, 14 heads,
widths 64, degree 4 (``--degree``), for the gate-free query polynomial and the query-only and
endpoint query-distance decays, shared or per polynomial order, and per-order cumulative decay
(H100's Mosaic kernels against the causal strips). ``kernels`` times each Mosaic kernel of those
families alone (forward, dQ, dK/dV) at the same shape. ``--families`` restricts both to some
families; a family this GPU's kernels do not run at the degree is skipped. ``update`` times
complete accumulated training updates (every microbatch's forward and backward, then the
optimizer) of one recipe, with its attention backend optionally replaced, and reports the
allocator's peak; run each recipe in its own process so peaks do not mix. Synthetic tokens
replace the dataset; diagnostics and checkpointing are excluded.

    python -m scaffolds.transformer.benchmarks.benchmark_decay attention --output a.json
    python -m scaffolds.transformer.benchmarks.benchmark_decay kernels --degree 12 --output k.json
    python -m scaffolds.transformer.benchmarks.benchmark_decay update RECIPE --backend mosaic --output u.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

FAMILIES = ("query_polynomial", "query_decay", "endpoint_decay", "per_order_query_decay", "per_order_endpoint_decay",
            "per_order_cumulative_decay")


def _runs(family, degree, width, time_):
    """Whether this GPU's Mosaic kernels run a family at this shape: H100's run every family; B300
    refuses the per-order shapes it cannot hold and every family qualified on H100 alone."""
    from scaffolds.transformer.attention.backends.mosaic import api, common

    family = common.QUERY if family == "query_polynomial" else family
    return (api._kernels()[0] == "hopper" or family not in common.PER_ORDER_FAMILIES
            or api.blackwell_supported(family, degree, width, time_))


def _timed(function, *args, repeats):
    import jax

    jax.block_until_ready(function(*args))  # compile and warm
    jax.block_until_ready(function(*args))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(function(*args))
        samples.append((time.perf_counter() - started) * 1e3)
    return samples


def attention(repeats, degree=4, families=FAMILIES):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scaffolds.transformer.attention.backends.mosaic import api
    from scaffolds.transformer.attention.backends.reference import (
        blocked_dot_product_attention, strip_dot_product_attention,
    )

    batch, time_, heads, width = 16, 2048, 14, 64
    rng = np.random.default_rng(0)
    q, k = (jnp.asarray(rng.standard_normal((batch, time_, heads, width)) * .7, jnp.bfloat16) for _ in range(2))
    v, do = (jnp.asarray(rng.standard_normal((batch, time_, heads, 64)), jnp.bfloat16) for _ in range(2))
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)])
    c = jnp.asarray(taylor * (1 + .5 * np.tanh(rng.standard_normal((batch, time_, heads, degree + 1)))), jnp.float32)
    u = jnp.asarray(rng.uniform(-5, -1, (batch, time_, heads)), jnp.float32)
    key = jnp.asarray(rng.standard_normal((batch, time_, heads)) * .5, jnp.float32)
    log_rate = jnp.asarray(rng.uniform(-1, 1, (heads, 1)), jnp.float32)
    log_decay = -jnp.exp(log_rate[:, 0]) * jax.nn.softplus(u)
    orders = degree + 1
    u_order = jnp.asarray(rng.uniform(-5, -1, (batch, time_, heads, orders)), jnp.float32)
    key_order = jnp.asarray(rng.standard_normal((batch, time_, heads, orders)) * .5, jnp.float32)
    log_rate_order = jnp.asarray(rng.uniform(-1, 1, (heads, orders, 1)), jnp.float32)
    log_decay_order = -jnp.exp(log_rate_order[..., 0]) * jax.nn.softplus(u_order)
    cumulative_log_decay = jnp.cumsum(log_decay_order, axis=1)  # every order's own sum L_tr

    def blocked(**gate):
        return lambda q, k, v, c, *g: blocked_dot_product_attention(
            q, k, v, attention_degree=degree, attention_normalization=False, query_polynomial_coefficients=c,
            **({name: x for name, x in zip(gate, g)} | ({"query_decay_mode": "query_distance"} if gate else {})))

    def strips(q, k, v, c, log_decay):
        return strip_dot_product_attention(q, k, v, attention_degree=degree, attention_normalization=False,
                                           query_polynomial_coefficients=c, query_log_decay=log_decay,
                                           query_decay_mode="cumulative")

    cases = {
        "query_polynomial/mosaic": (api.query_polynomial_attention, ()),
        "query_polynomial/blocked": (blocked(), ()),
        "query_decay/mosaic": (api.query_decay_attention, (log_decay,)),
        "query_decay/blocked": (blocked(query_log_decay=None), (log_decay,)),
        "endpoint_decay/mosaic": (api.endpoint_decay_attention, (u, key, log_rate)),
        "endpoint_decay/blocked": (blocked(query_gate_input=None, key_gate_input=None, gate_log_rate=None),
                                   (u, key, log_rate)),
        "per_order_query_decay/mosaic": (api.per_order_query_decay_attention, (log_decay_order,)),
        "per_order_query_decay/blocked": (blocked(query_log_decay=None), (log_decay_order,)),
        "per_order_endpoint_decay/mosaic": (api.per_order_endpoint_decay_attention,
                                            (u_order, key_order, log_rate_order)),
        "per_order_endpoint_decay/blocked": (blocked(query_gate_input=None, key_gate_input=None,
                                                     gate_log_rate=None), (u_order, key_order, log_rate_order)),
        "per_order_cumulative_decay/mosaic": (api.per_order_cumulative_decay_attention, (cumulative_log_decay,)),
        "per_order_cumulative_decay/strips": (strips, (cumulative_log_decay,)),
    }
    results = {}
    for name, (function, gate) in cases.items():
        family, backend = name.split("/")
        if family not in families or backend == "mosaic" and not _runs(family, degree, width, time_):
            continue
        operands = (q, k, v, c, *gate)
        forward = jax.jit(function)

        def backward(*x, function=function):
            output, pullback = jax.vjp(function, *x)
            return pullback(do)

        results[name] = {"forward_ms": _timed(forward, *operands, repeats=repeats),
                         "forward_backward_ms": _timed(jax.jit(backward), *operands, repeats=repeats)}
        print(name, {key: round(float(np.median(value)), 3) for key, value in results[name].items()}, flush=True)
    return {"shape": {"batch": batch, "time": time_, "heads": heads, "width": width, "value_width": 64,
                      "degree": degree}, "cases": results}


def kernels(repeats, degree=4, families=FAMILIES):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scaffolds.transformer.attention.backends.mosaic import api, common

    arch, module = api._kernels()
    batch, time_, heads, width = 16, 2048, 14, 64
    selected = {common.QUERY if family == "query_polynomial" else family for family in families}
    rng = np.random.default_rng(0)
    q, k = (jnp.asarray(rng.standard_normal((batch, time_, heads, width)) * .7, jnp.bfloat16) for _ in range(2))
    v, do = (jnp.asarray(rng.standard_normal((batch, time_, heads, 64)), jnp.bfloat16) for _ in range(2))
    taylor = np.array([1 / math.factorial(r) for r in range(degree + 1)])
    coefficients = jnp.asarray(np.broadcast_to(taylor[:, None, None, None], (degree + 1, batch, heads, time_)),
                               jnp.float32)
    token = lambda low, high: jnp.asarray(rng.uniform(low, high, (1, batch, heads, time_)), jnp.float32)
    orders = degree + 1
    order_tokens = lambda low, high: jnp.concatenate([token(low, high) for _ in range(orders)])
    head_rates = jnp.pad(jnp.full((heads, orders), -1.3, jnp.float32), ((0, 0), (0, common.head_width(orders) - orders)))
    inputs = {
        "query": (coefficients, None, None),
        "query_decay": (jnp.concatenate([coefficients, -token(.001, .5) * common.LOG2E]), None, None),
        "endpoint_decay": (jnp.concatenate([coefficients, token(-5, -1), jnp.full((1, batch, heads, time_), -1.3)]),
                           jnp.asarray(rng.standard_normal((1, batch, heads, time_)) * .5, jnp.float32), None),
        "per_order_query_decay": (jnp.concatenate([coefficients, -order_tokens(.001, .5) * common.LOG2E]), None, None),
        "per_order_endpoint_decay": (jnp.concatenate([coefficients, order_tokens(-5, -1)]),
                                     jnp.asarray(rng.standard_normal((orders, batch, heads, time_)) * .5, jnp.float32),
                                     head_rates),
    }
    sums = jnp.cumsum(-order_tokens(.001, .5), axis=-1)  # every order's cumulative log decay, for queries and keys
    inputs["per_order_cumulative_decay"] = (jnp.concatenate([coefficients, sums]), sums, None)
    results = {}
    for family, (vectors, key_vectors, head_constants) in inputs.items():
        if family not in selected or not _runs(family, degree, width, time_):
            continue
        options = dict(family=family, degree=degree, scale=1 / math.sqrt(width), precision="bf16")
        extra = (key_vectors, head_constants)
        passes = {"forward": lambda: module.forward(q, k, v, vectors, *extra, **options),
                  "dq": lambda: module.dq(q, k, v, do, vectors, *extra, **options)}
        if arch == "hopper":
            passes["dkv"] = lambda: module.dkv(q, k, v, do, vectors, *extra, **options)
        else:
            passes["dk"] = lambda: module.dkv(q, k, v, do, vectors, *extra, which="dk", **options)
            passes["dv"] = lambda: module.dkv(q, k, v, do, vectors, *extra, which="dv", **options)
        results[family] = {name: _timed(function, repeats=repeats) for name, function in passes.items()}
        print(family, {name: round(float(np.median(x)), 3) for name, x in results[family].items()}, flush=True)
    return {"architecture": arch, "shape": {"batch": batch, "time": time_, "heads": heads, "width": width,
                                            "value_width": 64, "degree": degree}, "kernels_ms": results}


def update(recipe, backend, warmup, iterations):
    import jax
    import numpy as np
    from scaffolds.transformer.configs import config_to_dict, load_config
    from scaffolds.transformer.training import loop, state as states

    config = load_config(recipe)
    if backend:
        config.model.attention_backend = backend
        config.model.__post_init__()  # validate the replaced backend
    config.data.source = "synthetic"
    config.data.synthetic_vocab_size = config.model.vocab_size
    global_batch, accumulation = loop.resolve_gradient_accumulation(config)
    micro = global_batch
    rng = jax.random.PRNGKey(config.train.seed)
    state = states.create_state(config, rng)
    batches = jax.numpy.asarray(np.random.default_rng(config.train.seed).integers(
        0, config.model.vocab_size, (accumulation, micro, config.data.sequence_length), dtype=np.int32))
    rngs = jax.random.split(rng, accumulation)
    started = time.perf_counter()
    compiled = jax.jit(loop.accumulated_train_step, donate_argnums=(0,)).lower(state, batches, rngs).compile()
    compile_seconds = time.perf_counter() - started
    samples, losses = [], []
    for index in range(warmup + iterations):
        started = time.perf_counter()
        state, loss = compiled(state, batches, rngs)
        jax.block_until_ready((state, loss))
        if index >= warmup:
            samples.append(time.perf_counter() - started)
        losses.append(float(loss))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"Nonfinite loss at benchmark update {index + 1}")
    stats = jax.local_devices()[0].memory_stats() or {}
    total = config.optim.total_steps
    return {"recipe": str(recipe), "attention_backend": config.model.attention_backend,
            "microbatches": accumulation, "microbatch_sequences": micro,
            "sequence_length": config.data.sequence_length, "compile_seconds": compile_seconds,
            "warmup": warmup, "seconds_per_update": samples, "median_seconds_per_update": float(np.median(samples)),
            "projected_training_hours": float(np.median(samples)) * total / 3600, "total_steps": total,
            "losses": losses, "allocator_peak_bytes": stats.get("peak_bytes_in_use"),
            "allocator_limit_bytes": stats.get("bytes_limit"), "config": config_to_dict(config)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("attention", "kernels", "update"))
    parser.add_argument("recipe", nargs="?", type=Path)
    parser.add_argument("--backend", choices=("blocked", "mosaic"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--degree", type=int, default=4, help="Polynomial degree of attention and kernels")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES,
                        help="Families timed by attention and kernels")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    import jax

    device = jax.local_devices()[0]
    report = {"format": "decay-attention-benchmark", "mode": args.mode, "device": device.device_kind,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
              "jax": jax.__version__,
              "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "source_dirty": bool(subprocess.run(["git", "status", "--porcelain", "scaffolds"], capture_output=True,
                                                  text=True).stdout.strip())}
    if args.mode == "attention":
        report.update(attention(args.repeats, args.degree, args.families))
    elif args.mode == "kernels":
        report.update(kernels(args.repeats, args.degree, args.families))
    else:
        if args.recipe is None:
            parser.error("update requires a recipe")
        report.update(update(args.recipe, args.backend, args.warmup, args.iterations))
        print(json.dumps({name: report[name] for name in (
            "recipe", "attention_backend", "median_seconds_per_update", "projected_training_hours",
            "allocator_peak_bytes", "compile_seconds")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
