"""Time a bounded, isolated replay through the actual single-GPU training loop.

The source run is read-only. A disposable copy of its checkpoint is restored with
the original optimizer, data cursor, global batch, and learning-rate horizon.
Five warmup updates exclude compilation. Completion-to-completion intervals then
include data loading, transfer, numerical checks, and ordinary training logging.
No evaluation or periodic checkpoint may fall inside the measured interval.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import jax
import numpy as np
import yaml

from scaffolds.launch.frozen import frozen_hash
from scaffolds.transformer.benchmarks.benchmark_attention import provenance, write_report
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.validation.replay_checkpoint import prepare


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def summarize(seconds, tokens):
    values = np.asarray(seconds)
    return {"updates": len(seconds), "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)),
            "p10_seconds": float(np.quantile(values, .1)),
            "p90_seconds": float(np.quantile(values, .9)),
            "tokens_per_second": float(tokens * len(values) / values.sum())}


def benchmark(args):
    run, scratch = args.run.resolve(), args.scratch.resolve()
    if scratch.exists() or scratch == run or scratch.is_relative_to(run):
        raise ValueError("Replay scratch must be a new directory outside the source run")
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("This replay benchmark requires exactly one GPU")
    source = provenance(devices[0])
    source["frozen_execution_sha256"] = frozen_hash(
        Path.cwd(), source["source_commit"], "attention_scaling", "transformer")
    for name in ("jax-cuda13-plugin", "nvidia-cublas", "nvidia-cuda-runtime"):
        try:
            source["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    source["runtime_environment"] = {name: os.environ.get(name) for name in (
        "TRANSFORMER_CUDA_ENVIRONMENT", "CUDA_HOME", "TRITON_PTXAS_PATH",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS", "TRANSFORMER_HOST_STAGING")}
    raw_recipe = yaml.safe_load((run / "config.yaml").read_text())
    original_record = json.loads((run / "record.json").read_text())
    checkpoint = run / "checkpoints" / str(args.checkpoint)
    protected = [run / name for name in ("config.yaml", "record.json", "metrics.jsonl", "data_identity.json")]
    protected += sorted(path for path in checkpoint.rglob("*") if path.is_file())
    hashes = {str(path.relative_to(run)): digest(path) for path in protected}
    baseline_rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
    baseline = []
    for step in args.baseline_steps:
        matches = [row for row in baseline_rows if row.get("step") == step and "tokens_per_second" in row]
        if len(matches) != 1:
            raise ValueError(f"Expected one recorded throughput at baseline step {step}")
        row = matches[0]
        baseline.append({key: row[key] for key in (
            "step", "tokens_per_second", "training_interval_seconds", "training_window_start_step")})
    report = {"format": "checkpoint-replay-throughput", "schema_version": 1,
              "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
              "home_cluster": "kennel", "provenance": source,
              "source_run": {"run_id": original_record["run_id"],
                             "source_commit": original_record["source_commit"],
                             "recipe": raw_recipe, "checkpoint_step": args.checkpoint,
                             "input_sha256": hashes,
                             "runtime": original_record["execution"]["runtime"]},
              "historical_baseline": baseline, "warmup_updates": args.warmup,
              "timed_updates": args.iterations, "samples": [],
              "limitations": ["Single-GPU trainer omits the per-group telemetry used by data-parallel training.",
                              "Historical four-GPU measurements are not a simultaneous hardware-controlled comparison.",
                              "Steady intervals exclude validation, checkpointing, compilation, and startup."]}
    write_report(args.output, report)
    started = time.perf_counter()
    recipe_path = prepare(run, args.checkpoint, scratch)
    recipe = yaml.safe_load(recipe_path.read_text())
    legacy_degree = recipe["model"].pop("gdn_polynomial_degree", None)
    if legacy_degree is not None and (type(legacy_degree) is not int or legacy_degree != 1
                                    or recipe["model"]["attention_mechanism"] != "gated_deltanet"):
        raise ValueError("Only the legacy degree-one GDN recipe is equivalent to the current linear implementation")
    if legacy_degree is not None:
        report["recipe_compatibility"] = "Removed explicit gdn_polynomial_degree: 1; linear GDN mathematics unchanged."
    if args.micro_batch_size is not None:
        recipe["train"]["micro_batch_size"] = args.micro_batch_size
    recipe["train"]["source_commit"] = source["source_commit"]
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config = load_config(recipe_path)
    end_step = args.checkpoint + args.warmup + args.iterations
    if end_step >= config.optim.total_steps:
        raise ValueError("Benchmark must finish before the training horizon")
    for step in range(args.checkpoint + 1, end_step + 1):
        if step % config.train.eval_every == 0 or step % config.train.ckpt_every == 0:
            raise ValueError("Choose an interval without scheduled evaluation or checkpointing")
    tokens = config.data.batch_size * (config.data.sequence_length - 1)
    report["configuration"] = config_to_dict(config)
    report["tokens_per_update"] = tokens
    report["accumulation_steps"] = config.data.batch_size // config.train.micro_batch_size
    last_end = None
    originals = {name: getattr(loop, name) for name in ("train_step", "accumulated_train_step")}

    def measured(function):
        def wrapped(*positional, **keywords):
            nonlocal last_end
            begin = time.perf_counter()
            result = function(*positional, **keywords)
            jax.block_until_ready(result)
            finish = time.perf_counter()
            sample = {"step": int(result[0].step), "loss": float(result[1]),
                      "compute_seconds": finish - begin,
                      "interval_seconds": None if last_end is None else finish - last_end}
            last_end = finish
            report["samples"].append(sample)
            return result
        return wrapped

    try:
        for name, function in originals.items():
            setattr(loop, name, measured(function))
        report["outcome"] = loop.run_training(config, resume=True, stop_after_step=end_step)
        samples = report["samples"][args.warmup:]
        if len(samples) != args.iterations or samples[-1]["step"] != end_step:
            raise RuntimeError("Unexpected replay update count")
        report["steady_training"] = summarize([s["interval_seconds"] for s in samples], tokens)
        report["steady_compute"] = summarize([s["compute_seconds"] for s in samples], tokens)
        report["gpu_memory_stats"] = devices[0].memory_stats()
        report["source_files_unchanged"] = all(digest(path) == hashes[str(path.relative_to(run))] for path in protected)
        if not report["source_files_unchanged"]:
            raise RuntimeError("Source run changed during replay")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for name, function in originals.items():
            setattr(loop, name, function)
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        if (scratch / "runtime.json").exists():
            report["runtime_audit"] = json.loads((scratch / "runtime.json").read_text())
        write_report(args.output, report)
        if report["status"] == "complete":
            shutil.rmtree(scratch)
    print("REPLAY_THROUGHPUT " + json.dumps(report["steady_training"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--baseline-steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if min(args.checkpoint, args.warmup, args.iterations) < 1:
        parser.error("Checkpoint, warmup, and iterations must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
