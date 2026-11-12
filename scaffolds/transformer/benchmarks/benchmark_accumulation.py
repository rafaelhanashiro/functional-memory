"""Time one single-GPU accumulation configuration with resident synthetic tokens.

Run each configuration in a fresh process so allocator peaks are independent.
This measures complete optimizer updates, excluding data loading, evaluation,
checkpointing and compilation. It is a runtime benchmark, not a training run.
The Python implementation reproduces the pre-scan accumulator for comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scaffolds.transformer.benchmarks.benchmark_attention import write_report
from scaffolds.transformer.benchmarks.benchmark_train_step import parameter_digest
from scaffolds.transformer.benchmarks.benchmark_training_throughput import (
    _execution_provenance, digest, summarize,
)
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.numerics import checked_update_loss, update_numerics
from scaffolds.transformer.training.state import create_state


_compiled_microbatch = nnx.jit(
    loop.microbatch_loss_and_grads,
    static_argnames=("coefficient_penalty", "coefficient_penalty_denominator"),
)


def python_accumulation(state, batches, rngs, *, coefficient_penalty=0.0,
                        coefficient_penalty_denominator=True):
    """Baseline: separately compiled microbatches and Python-dispatched reductions."""
    grad_sum = loss_sum = None
    for batch, rng in zip(batches, rngs, strict=True):
        loss, grads = _compiled_microbatch(
            state, batch, rng, coefficient_penalty=coefficient_penalty,
            coefficient_penalty_denominator=coefficient_penalty_denominator)
        if grad_sum is None:
            grad_sum, loss_sum = grads, loss
        else:
            grad_sum = jax.tree.map(jnp.add, grad_sum, grads)
            loss_sum = loss_sum + loss
    scale = jnp.asarray(1.0 / len(batches), dtype=loss_sum.dtype)
    grads = jax.tree.map(lambda value: value * scale, grad_sum)
    loss = loss_sum * scale
    updated = loop.apply_gradients_step(state, grads)
    return (updated, checked_update_loss(state.params, grads, updated.params, loss,
                                         check_all_gradients=True),
            update_numerics(grads, updated.params, loss))


def benchmark(args):
    config = load_config(args.config)
    original = config_to_dict(config)
    if min(args.micro_batch_size, args.warmup, args.iterations) < 1:
        raise ValueError("Microbatch, warmup, and iterations must be positive")
    if getattr(args, "backend", None):
        config.model.attention_backend = args.backend
    config.train.micro_batch_size = args.micro_batch_size
    validate_config(config)
    if config.data.batch_size % args.micro_batch_size:
        raise ValueError("Microbatch must divide the global batch")
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("This benchmark requires exactly one GPU")
    source = _execution_provenance(devices[0], "attention_scaling")
    source["benchmark_source_sha256"] = digest(Path(__file__))
    accumulation_steps = config.data.batch_size // args.micro_batch_size
    report = {
        "format": "single-gpu-accumulation", "schema_version": 1, "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(), "provenance": source,
        "input_recipe": {"path": str(args.config.resolve()), "sha256": digest(args.config),
                         "configuration": original},
        "configuration": config_to_dict(config), "implementation": args.implementation,
        "warmup_updates": args.warmup, "timed_updates": args.iterations,
        "accumulation_steps": accumulation_steps, "samples": [],
        "measurement_notes": [
            "Resident synthetic tokens; model, optimizer, and global batch match the input recipe.",
            "Microbatch size, accumulation implementation and optional attention backend are execution variants.",
            "Every sample synchronizes the complete optimizer state, loss, and numerical checks.",
            "Compilation, data loading, evaluation, logging and checkpoints are excluded from steady timing.",
            "Allocator peak includes initialization and warmup in this fresh process."],
    }
    if config.model.attention_backend == "triton_scalar":
        from scaffolds.transformer.attention.basis import projection_from_config
        from scaffolds.transformer.attention.backends.triton_scalar import basis_kernel, Tuning
        kernel = basis_kernel(projection_from_config(config.model.attention_basis), implementation="auto")
        fused_rms = not config.model.attention_normalization and config.model.attention_output_scaling == "head_rms_norm"
        report['scalar_execution'] = {'callback': kernel.name, 'tuning': asdict(Tuning()),
            'fused_head_rms_norm': fused_rms, 'streaming': config.model.attention_normalization or fused_rms}
    write_report(args.output, report)
    try:
        state = create_state(config, jax.random.PRNGKey(config.train.seed))
        report["initial_parameter_sha256"], report["parameter_count"] = parameter_digest(state.params)
        tokens = np.random.default_rng(config.train.seed).integers(
            0, config.model.vocab_size, (config.data.batch_size, config.data.sequence_length), dtype=np.int32)
        batches = jax.device_put(tokens.reshape(accumulation_steps, args.micro_batch_size, -1))
        rngs = [jax.random.split(jax.random.fold_in(jax.random.PRNGKey(config.train.seed), step),
                                 accumulation_steps)
                for step in range(args.warmup + args.iterations)]
        jax.block_until_ready((state, batches, rngs))
        kwargs = {"coefficient_penalty": config.optim.coefficient_penalty,
                  "coefficient_penalty_denominator": config.optim.coefficient_penalty_denominator}
        if args.implementation == "compiled":
            function = loop.accumulated_train_step
            kwargs["check_all_gradients"] = True
        else:
            function = python_accumulation
        for index, keys in enumerate(rngs):
            started = time.perf_counter()
            state, loss, numerics = function(state, batches, keys, **kwargs)
            jax.block_until_ready((state, loss, numerics))
            seconds = time.perf_counter() - started
            if not np.isfinite(float(loss)) or not np.all(np.isfinite(np.asarray(numerics))):
                raise FloatingPointError(f"Nonfinite loss or optimizer update at step {index + 1}")
            sample = {"step": int(state.step), "seconds": seconds, "loss": float(loss),
                      "update_numerics": np.asarray(numerics).tolist(), "warmup": index < args.warmup}
            report["samples"].append(sample)
            print(f"step={sample['step']} seconds={seconds:.4f} loss={float(loss):.7f}", flush=True)
            write_report(args.output, report)
        if int(state.step) != args.warmup + args.iterations:
            raise RuntimeError("Expected one optimizer update per global batch")
        report["tokens_per_update"] = config.data.batch_size * (config.data.sequence_length - 1)
        report["steady_compute"] = summarize(
            [sample["seconds"] for sample in report["samples"] if not sample["warmup"]],
            report["tokens_per_update"])
        report["status"] = "timed"
        write_report(args.output, report)
        if getattr(args, "trace_dir", None):
            report['profile_samples'] = []
            with jax.profiler.trace(str(args.trace_dir), create_perfetto_trace=True):
                for index in range(3):
                    with jax.profiler.StepTraceAnnotation("train", step_num=int(state.step) + 1):
                        state, loss, numerics = function(state, batches, rngs[-1], **kwargs)
                        jax.block_until_ready((state, loss, numerics))
                        if not np.isfinite(float(loss)) or not np.all(np.isfinite(np.asarray(numerics))):
                            raise FloatingPointError("Nonfinite update during profiling")
                        report['profile_samples'].append({'step': int(state.step), 'loss': float(loss),
                                                         'update_numerics': np.asarray(numerics).tolist()})
            report["profile"] = str(args.trace_dir)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["gpu_memory_stats"] = devices[0].memory_stats()
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--implementation", choices=("python", "compiled"), default="compiled")
    parser.add_argument("--backend", choices=("blocked", "triton_scalar"))
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
