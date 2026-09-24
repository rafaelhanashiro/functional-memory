"""Measure the actual packed-data training loop in a fresh, bounded pilot.

Keep the production model, batch, optimizer, and learning-rate horizon. Only
validation/checkpoint cadence is compressed to observe their first and warmed
costs. Reads and writes use the configured paths directly, including NFS.
Run inside a GPU allocation, never on the login node.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import time

import jax

from scaffolds.transformer.benchmarks.benchmark_attention import provenance, write_report
from scaffolds.transformer.benchmarks.benchmark_train_step import degree_argument
from scaffolds.transformer.studies.key_dimension_taylor.degree_sweep.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.training.loop import run_training


def benchmark(args) -> dict:
    workdir = args.workdir.resolve()
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError(f"Pilot requires a fresh workdir: {workdir}")
    config = load_config(args.config)
    if config.data.source != "packed":
        raise ValueError("This benchmark requires real packed data")
    if args.steps >= config.optim.total_steps:
        raise ValueError("Pilot must stop before the unchanged production horizon")
    devices = jax.local_devices(backend="gpu")
    source = provenance(devices[0])
    if not source["source_commit"] or source["source_dirty"]:
        raise ValueError("Run the packed-data pilot from a clean committed checkout")
    config.model.attention_backend = "fused"
    config.model.attention_degree = args.degree
    config.train.micro_batch_size = args.micro_batch_size
    config.train.workdir = workdir
    config.train.checkpoint_dir = None  # Isolate pilot artifacts from production.
    config.train.source_commit = source["source_commit"]
    config.train.wandb_mode = "disabled"
    config.train.eval_every = args.evaluation_every
    config.train.ckpt_every = args.checkpoint_every
    validate_config(config)
    root = Path(__file__).resolve().parents[3]
    for relative in (
        "scaffolds/transformer/benchmarks/benchmark_packed_run.py",
        "scaffolds/transformer/training/loop.py",
        "scaffolds/transformer/training/distributed.py",
        "scaffolds/transformer/models/architectures/transformer.py",
    ):
        source["source_sha256"][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    report = {
        "format": "polynomial-attention-packed-run-benchmark", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": source, "configuration": config_to_dict(config),
        "stop_after_step": args.steps,
        "measurement_notes": [
            "Actual packed-data loading, transfers, updates, diagnostics, evaluation, and checkpoint writes.",
            "Production learning-rate horizon is unchanged; validation/checkpoint cadence is compressed.",
            "Training loop retains the first eight synchronized step timings; first step includes compilation.",
            "First evaluation includes compilation; later evaluations use warmed executables.",
            "Validation checkpoints are waited for; other checkpoints can overlap subsequent training.",
            "Full wall time includes initialization and final checkpoint completion; no NFS staging.",
        ],
    }
    write_report(args.output, report)
    started = time.perf_counter()
    try:
        report["outcome"] = run_training(config, resume=False, stop_after_step=args.steps)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("scaffolds/transformer/configs/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--degree", type=degree_argument, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--evaluation-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    args = parser.parse_args()
    if min(args.steps, args.evaluation_every, args.checkpoint_every, args.micro_batch_size) < 1:
        parser.error("Steps, intervals, and microbatch must be positive")
    if args.steps < 2 * args.evaluation_every:
        parser.error("Require at least two evaluations to observe warmed overhead")
    benchmark(args)


if __name__ == "__main__":
    main()
