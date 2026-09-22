"""Measure synchronized training updates without writing training artifacts.

The optional package selects a historical scaffold exported into a temporary
checkout, allowing the same driver to compare old and new implementations.
Compilation and token generation are excluded from timed samples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", default="scaffolds.transformer")
    parser.add_argument("--source-commit", help="Recorded commit when benchmarking an exported source tree.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")

    import jax
    import jax.numpy as jnp
    import numpy as np

    configs = importlib.import_module(f"{args.package}.configs")
    states = importlib.import_module(f"{args.package}.training.state")
    loop = importlib.import_module(f"{args.package}.training.loop")
    config = configs.load_config(args.config)
    config.data.source = "synthetic"
    config.data.synthetic_vocab_size = config.model.vocab_size
    rng = jax.random.PRNGKey(config.train.seed)
    state = states.create_state(config, rng)
    batch = jnp.asarray(np.random.default_rng(config.train.seed).integers(
        0, config.model.vocab_size,
        (config.train.micro_batch_size, config.data.sequence_length), dtype=np.int32,
    ))
    started = time.perf_counter()
    compiled = jax.jit(loop.train_step, donate_argnums=(0,)).lower(state, batch, rng).compile()
    compile_seconds = time.perf_counter() - started
    memory = compiled.memory_analysis()
    memory_fields = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
    estimates = {name: int(getattr(memory, name)) for name in memory_fields} if memory else {}
    samples, losses = [], []
    for index in range(args.warmup + args.iterations):
        started = time.perf_counter()
        state, loss = compiled(state, batch, rng)
        jax.block_until_ready((state, loss))
        elapsed = (time.perf_counter() - started) * 1000
        value = float(loss)
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite benchmark loss at update {index + 1}")
        losses.append(value)
        if index >= args.warmup:
            samples.append(elapsed)
    device = jax.local_devices()[0]
    stats = device.memory_stats() or {}
    source_directory = Path(states.__file__).resolve().parents[1]
    source_digest = hashlib.sha256()
    for source in sorted(source_directory.rglob("*.py")):
        source_digest.update(str(source.relative_to(source_directory)).encode())
        source_digest.update(source.read_bytes())
    commit = args.source_commit
    if commit is None:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except subprocess.CalledProcessError:
            pass
    report = {
        "format": "shared-transformer-update-benchmark", "version": 1,
        "package": args.package, "source_commit": commit,
        "source_directory": str(source_directory), "source_tree_sha256": source_digest.hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": device.device_kind, "config": configs.config_to_dict(config),
        "warmup": args.warmup, "iterations": args.iterations,
        "compile_seconds": compile_seconds,
        "median_ms": float(np.median(samples)), "samples_ms": samples,
        "losses": losses, "compiled_buffer_bytes": estimates,
        "allocator_peak_bytes": stats.get("peak_bytes_in_use"),
        "timing_scope": "One complete microbatch forward/backward/optimizer update; compilation, data loading, accumulation, diagnostics and checkpointing excluded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: report[name] for name in ("package", "median_ms", "compiled_buffer_bytes", "allocator_peak_bytes")}))


if __name__ == "__main__":
    main()
