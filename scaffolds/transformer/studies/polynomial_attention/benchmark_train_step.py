"""Compare complete blocked/fused Transformer updates on allocated GPUs.

Production defaults: four GPUs, global microbatch 64, four accumulation steps,
2048 prediction positions, and the existing 200M BF16 model/optimizer recipe.
Synthetic tokens are resident on the devices before timing. Data loading,
validation, diagnostics, checkpointing, and compilation are excluded from the
timed updates. All variants begin from the same initial parameter arrays and
optimizer state. Optional profiles capture three additional warmed updates
after the timed updates. Run this in an allocation, never on the login node.

  python -m scaffolds.transformer.studies.polynomial_attention.benchmark_train_step \
    --output /path/on/nfs/train-step-benchmark.json

  python -m scaffolds.transformer.studies.polynomial_attention.benchmark_train_step \
    --degrees 2 16 softmax --implementations fused \
    --trace-dir /path/on/nfs/profiles --output /path/on/nfs/profiled-benchmark.json
"""
from __future__ import annotations

import argparse
import copy
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scaffolds.transformer.studies.polynomial_attention.benchmark_attention import provenance, write_report
from scaffolds.transformer.studies.polynomial_attention.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.training.distributed import make_data_parallel_functions, put_train_batches
from scaffolds.transformer.training.state import create_state


TRACE_UPDATES = 3


def degree_argument(value: str) -> int | None:
    if value.lower() == "softmax":
        return None
    try:
        degree = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected 'softmax' or a nonnegative even integer") from error
    if degree < 0 or degree % 2:
        raise argparse.ArgumentTypeError("Expected 'softmax' or a nonnegative even integer")
    return degree


def parameter_digest(params) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(path).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
        count += array.size
    return digest.hexdigest(), count


def backend_state(reference, model_config):
    """Change static dispatch only; reuse the one initialized parameter tree."""
    model = nnx.merge(reference.graphdef, reference.params)
    model.config = copy.deepcopy(model_config)
    for name in model.layer_names:
        attention = getattr(model, name).attention
        attention.backend = model_config.attention_backend
        attention.attention_degree = model_config.attention_degree
    graphdef, _ = nnx.split(model, nnx.Param)
    return reference.replace(graphdef=graphdef)


def immutable_host_state(state):
    """Detach the reusable initialization from all donatable device buffers."""
    def snapshot(value):
        if not isinstance(value, (jax.Array, np.ndarray)):
            return value
        array = np.array(jax.device_get(value), copy=True)
        array.setflags(write=False)
        return array

    return jax.tree.map(snapshot, state)


def fresh_replicated_state(state, sharding):
    # Copy from host-owned arrays for each trial. Keeping the reference on the
    # host avoids dependence on device-buffer ownership during resharding and
    # donation; the earlier device-backed reference was deleted by donation.
    def restore(value):
        if isinstance(value, jax.Array):
            raise TypeError("A reusable benchmark state must contain only host arrays")
        if isinstance(value, np.ndarray):
            return jax.device_put(np.array(value, copy=True), sharding, may_alias=False)
        return value

    return jax.tree.map(restore, state)


def time_updates(state, functions, batches, rngs, *, warmup: int, iterations: int) -> tuple[dict, object]:
    warmup_seconds, warmup_losses = [], []
    for index in range(warmup):
        start = time.perf_counter()
        state, loss = functions.train_step(state, batches, rngs[index])
        jax.block_until_ready((state, loss))
        value = float(loss)
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite warmup loss at update {index + 1}: {value}")
        warmup_seconds.append(time.perf_counter() - start)
        warmup_losses.append(value)
        print(f"warmup_update={index + 1} seconds={warmup_seconds[-1]:.4f} loss={value:.7f}", flush=True)
    samples_ms, losses = [], []
    for index in range(iterations):
        start = time.perf_counter()
        state, loss = functions.train_step(state, batches, rngs[warmup + index])
        jax.block_until_ready((state, loss))
        value = float(loss)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite measured loss at update {warmup + index + 1}: {value}")
        samples_ms.append(elapsed_ms)
        losses.append(value)
    final_step = int(state.step)
    if final_step != warmup + iterations:
        raise RuntimeError(f"Unexpected optimizer step after benchmark: {final_step}")
    return {
        "initial_loss": warmup_losses[0],
        "first_warmup_including_compilation_seconds": warmup_seconds[0],
        "warmup_seconds": warmup_seconds,
        "warmup_losses": warmup_losses,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "final_optimizer_step": final_step,
        "median_ms": float(np.median(samples_ms)),
        "p10_ms": float(np.quantile(samples_ms, 0.1)),
        "p90_ms": float(np.quantile(samples_ms, 0.9)),
        "min_ms": min(samples_ms), "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "losses": losses,
    }, state


def trace_updates(state, functions, batches, rngs, *, trace_dir: Path,
                  degree: int | None, backend: str, device_count: int) -> tuple[dict, object]:
    """Profile complete already-compiled updates without changing timing samples."""
    if len(rngs) != TRACE_UPDATES:
        raise ValueError(f"Expected exactly {TRACE_UPDATES} profiling RNGs")
    trace_dir.mkdir(parents=True, exist_ok=False)
    initial_step = int(state.step)
    losses = []
    started = time.perf_counter()
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        with jax.profiler.TraceAnnotation(
            "benchmark_train_step", degree="softmax" if degree is None else degree,
            implementation=backend, device_count=device_count,
        ):
            for index, rng in enumerate(rngs):
                with jax.profiler.StepTraceAnnotation("train", step_num=initial_step + index + 1):
                    state, loss = functions.train_step(state, batches, rng)
                    jax.block_until_ready((state, loss))
                    value = float(loss)
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Nonfinite profiled loss at update {initial_step + index + 1}: {value}")
                    losses.append(value)
    final_step = int(state.step)
    if final_step != initial_step + TRACE_UPDATES:
        raise RuntimeError(f"Unexpected optimizer step after profiling: {final_step}")
    return {
        "directory": str(trace_dir), "updates": TRACE_UPDATES,
        "initial_optimizer_step": initial_step, "final_optimizer_step": final_step,
        "losses": losses, "total_including_profiler_export_seconds": time.perf_counter() - started,
        "position": "after timed updates; continuing the same warmed state and executable",
        "create_perfetto_trace": True,
    }, state


def benchmark(args) -> dict:
    visible = jax.local_devices(backend="gpu")
    if len(visible) < args.device_count:
        raise ValueError(f"Benchmark requires {args.device_count} GPUs; found {len(visible)}")
    devices = tuple(visible[:args.device_count])
    config = load_config(args.config)
    config.data.source = "synthetic"
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.data.batch_size = args.global_micro_batch_size * args.accumulation_steps
    config.train.micro_batch_size = args.global_micro_batch_size
    config.train.dtype = "bfloat16"
    config.train.seed = args.seed
    config.train.wandb_mode = "disabled"
    config.model.attention_degree = args.degrees[0]
    config.model.attention_backend = "blocked"
    validate_config(config)
    token_shape = (args.accumulation_steps, args.global_micro_batch_size, config.data.sequence_length)
    tokens = np.random.default_rng(args.seed).integers(
        0, config.model.vocab_size, size=token_shape, dtype=np.int32,
    )
    tokens_per_step = config.data.batch_size * (config.data.sequence_length - 1)
    trace_root = (
        args.trace_dir.resolve() / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        if args.trace_dir is not None else None
    )
    report = {
        "format": "polynomial-attention-train-step-benchmark", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(devices[0]),
        "configuration": {
            "model_and_training_config": config_to_dict(config),
            "device_count": args.device_count,
            "devices": [{"id": device.id, "kind": device.device_kind} for device in devices],
            "global_batch_size": config.data.batch_size,
            "global_micro_batch_size": args.global_micro_batch_size,
            "local_micro_batch_size": args.global_micro_batch_size // args.device_count,
            "accumulation_steps": args.accumulation_steps,
            "tokens_per_step": tokens_per_step,
            "token_shape": list(token_shape), "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
            "degrees": args.degrees, "implementations": args.implementations,
            "warmup": args.warmup, "iterations": args.iterations,
            "profile": {
                "directory": str(trace_root) if trace_root is not None else None,
                "updates_per_trial": TRACE_UPDATES if trace_root is not None else 0,
                "included_in_timing": False,
            },
            "input_distribution": "fixed independent uniform token IDs; same packed-shaped batch for every update",
            "timing": "existing compiled distributed forward/backward/accumulation/all-reduce/optimizer; complete state and loss synchronized",
            "excluded": ["compilation", "input loading/transfers", "validation", "diagnostics", "checkpointing", "profiling"],
            "comparison": "fresh identical initial parameters and optimizer state for each trial; loss sequences are diagnostic, not gradient correctness proof",
        },
        "results": [],
    }
    root = Path(__file__).resolve().parents[4]
    for relative in (
        "scaffolds/transformer/studies/polynomial_attention/benchmark_train_step.py",
        "scaffolds/transformer/training/distributed.py",
        "scaffolds/transformer/training/state.py",
        "scaffolds/transformer/models/architectures/transformer.py",
        "scaffolds/transformer/configs/base.py",
    ):
        report["provenance"]["source_sha256"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    write_report(args.output, report)
    try:
        with jax.default_device(devices[0]):
            print("Initializing shared model parameters and optimizer state", flush=True)
            initialized = create_state(config, jax.random.PRNGKey(args.seed))
            jax.block_until_ready(initialized)
            reference = immutable_host_state(initialized)
            del initialized
            digest, parameter_count = parameter_digest(reference.params)
            report["initialization"] = {"parameter_sha256": digest, "parameter_count": parameter_count,
                                        "optimizer_step": int(reference.step), "initialized_once": True,
                                        "reusable_snapshot": "read-only NumPy copies of every parameter/optimizer/step array"}
            write_report(args.output, report)
            for degree in args.degrees:
                degree_label = "softmax" if degree is None else f"m{degree:03d}"
                entry = {"degree": degree, "attention": degree_label, "implementations": {}}
                report["results"].append(entry)
                for backend in args.implementations:
                    report["active_stage"] = {"degree": degree, "implementation": backend}
                    write_report(args.output, report)
                    current_config = copy.deepcopy(config)
                    current_config.model.attention_degree = degree
                    current_config.model.attention_backend = backend
                    validate_config(current_config)
                    template = backend_state(reference, current_config.model)
                    functions = make_data_parallel_functions(template, devices)
                    state = fresh_replicated_state(template, functions.replicated_sharding)
                    batches = put_train_batches(tokens, functions.train_batch_sharding)
                    master_rng = jax.random.PRNGKey(args.seed)
                    rngs = [jax.device_put(jax.random.fold_in(master_rng, index + 1), functions.replicated_sharding)
                            for index in range(args.warmup + args.iterations + (TRACE_UPDATES if trace_root is not None else 0))]
                    jax.block_until_ready((state, batches, rngs))
                    print(f"degree={degree} implementation={backend} devices={args.device_count} "
                          f"global_batch={config.data.batch_size}: compiling and warming up", flush=True)
                    timing, state = time_updates(state, functions, batches, rngs, warmup=args.warmup, iterations=args.iterations)
                    timing["tokens_per_second_at_median"] = tokens_per_step * 1000 / timing["median_ms"]
                    entry["implementations"][backend] = timing
                    print(f"degree={degree} implementation={backend} median_ms={timing['median_ms']:.4f} "
                          f"tokens_per_second={timing['tokens_per_second_at_median']:.1f}", flush=True)
                    write_report(args.output, report)
                    if trace_root is not None:
                        trace_dir = trace_root / f"{degree_label}-{backend}-g{args.device_count}-mb{args.global_micro_batch_size}"
                        report["active_stage"]["phase"] = "profiling"
                        write_report(args.output, report)
                        print(f"degree={degree} implementation={backend}: profiling {TRACE_UPDATES} warmed updates into {trace_dir}", flush=True)
                        timing["profile"], state = trace_updates(
                            state, functions, batches, rngs[args.warmup + args.iterations:],
                            trace_dir=trace_dir, degree=degree, backend=backend, device_count=args.device_count,
                        )
                        write_report(args.output, report)
                    del state, template, functions, batches, rngs
                if "blocked" in entry["implementations"] and "fused" in entry["implementations"]:
                    blocked = entry["implementations"]["blocked"]
                    fused = entry["implementations"]["fused"]
                    differences = np.abs(np.asarray(blocked["losses"]) - np.asarray(fused["losses"]))
                    entry["loss_comparison"] = {
                        "initial_absolute_error": abs(blocked["initial_loss"] - fused["initial_loss"]),
                        "max_timed_absolute_error": float(differences.max()),
                        "mean_timed_absolute_error": float(differences.mean()),
                    }
                    entry["speedup_blocked_over_fused"] = blocked["median_ms"] / fused["median_ms"]
                write_report(args.output, report)
        report["status"] = "complete"
        report.pop("active_stage", None)
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(args.output, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("scaffolds/transformer/configs/base.yaml"))
    parser.add_argument("--degrees", type=degree_argument, nargs="+", default=[2, 16],
                        help="Nonnegative even polynomial degrees or 'softmax' (fused backend uses cuDNN)")
    parser.add_argument("--implementations", choices=("blocked", "fused"), nargs="+", default=["blocked", "fused"])
    parser.add_argument("--device-count", type=int, choices=(2, 4), default=4)
    parser.add_argument("--global-micro-batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace-dir", type=Path,
                        help="NFS directory for three additional warmed, annotated updates per trial; excluded from timings")
    args = parser.parse_args()
    if args.global_micro_batch_size < 1 or args.global_micro_batch_size % args.device_count or args.accumulation_steps < 1:
        parser.error("Require a positive microbatch divisible by device-count and positive accumulation-steps")
    if args.warmup < 3 or args.iterations < 10:
        parser.error("Require at least 3 warmup updates and 10 timed updates")
    if len(set(args.degrees)) != len(args.degrees) or len(set(args.implementations)) != len(args.implementations):
        parser.error("Degrees and implementations must not contain duplicates")
    benchmark(args)


if __name__ == "__main__":
    main()
