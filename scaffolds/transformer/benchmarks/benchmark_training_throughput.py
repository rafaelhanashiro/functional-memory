"""Time fresh, bounded training through the actual single-GPU production loop.

Keep the supplied architecture, backend, batches, data, and learning-rate horizon.
Warmup excludes compilation. Steady completion intervals include data loading,
transfers, numerical checks, and normal logging; evaluation and checkpoint writes
are excluded. Successful pilots delete their disposable state after recording it.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import jax
import numpy as np

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.transformer.benchmarks.benchmark_attention import provenance, write_report
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.training import loop


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def summarize(seconds, tokens):
    values = np.asarray(seconds, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Timing samples must be finite and positive")
    return {"updates": len(values), "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)),
            "p10_seconds": float(np.quantile(values, .1)),
            "p90_seconds": float(np.quantile(values, .9)),
            "min_seconds": float(values.min()), "max_seconds": float(values.max()),
            "tokens_per_second": float(tokens * len(values) / values.sum())}


def _execution_provenance(device, experiment):
    root = Path(__file__).resolve().parents[3]
    source = provenance(device)
    commit = source["source_commit"]
    if not commit:
        raise ValueError("Benchmark requires committed source")
    modified = changes(root, commit, experiment, "transformer")
    if modified:
        raise ValueError(f"Frozen execution files differ from the source commit: {modified}")
    source["frozen_execution_sha256"] = frozen_hash(root, commit, experiment, "transformer")
    source["frozen_execution_scope"] = {"experiment": experiment, "scaffold": "transformer"}
    source["dependency_lock_sha256"] = digest(root / "scaffolds/transformer/uv.lock")
    source["benchmark_source_sha256"] = digest(Path(__file__))
    for name in ("jax-cuda13-plugin", "nvidia-cublas", "nvidia-cuda-runtime", "nvidia-cuda-nvcc",
                 "nvidia-cudnn-cu13", "nvidia-nccl-cu13", "triton", "flash-linear-attention"):
        try:
            source["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    source["runtime_environment"] = {name: os.environ.get(name) for name in (
        "TRANSFORMER_CUDA_ENVIRONMENT", "CUDA_HOME", "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS",
        "TRANSFORMER_HOST_STAGING", "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE",
        "POLY_GDN_VALUE_PRECISION", "POLY_GDN_NUM_WARPS")}
    source["device"]["uuid"] = None
    try:
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        selector = os.environ.get("CUDA_VISIBLE_DEVICES", str(device.id)).split(",")[0].strip()
        for line in inventory.splitlines():
            row = dict(zip(("physical_index", "uuid", "name", "pci_bus_id", "driver_version"),
                           (part.strip() for part in line.split(",")), strict=True))
            if row["physical_index"] == selector or row["uuid"] == selector:
                source["device"].update(row)
                break
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        source["device"]["inventory_error"] = f"{type(error).__name__}: {error}"
    return source


def benchmark(args):
    source_path, scratch, output = args.config.resolve(), args.scratch.resolve(), args.output.resolve()
    if min(args.warmup, args.iterations) < 1:
        raise ValueError("Warmup and iterations must be positive")
    if scratch.exists() or source_path.is_relative_to(scratch) or output.is_relative_to(scratch):
        raise ValueError("Scratch must be a new directory containing neither the source recipe nor the report")
    source_digest = digest(source_path)
    config = load_config(source_path)
    original_config = config_to_dict(config)
    end_step = args.warmup + args.iterations
    if end_step >= config.optim.total_steps:
        raise ValueError("Benchmark must finish before the unchanged training horizon")
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("This benchmark requires exactly one GPU")
    source = _execution_provenance(devices[0], getattr(args, "experiment", "attention_scaling"))
    config.train.workdir = scratch
    config.train.checkpoint_dir = scratch / "checkpoints"
    config.train.wandb_mode = "disabled"
    config.train.wandb_dir = None
    config.train.source_commit = source["source_commit"]
    # The trainer still saves a final checkpoint after the last timed update.
    config.train.eval_every = max(config.train.eval_every, end_step + 1)
    config.train.ckpt_every = max(config.train.ckpt_every, end_step + 1)
    validate_config(config)
    tokens = config.data.batch_size * (config.data.sequence_length - 1)
    report = {"format": "fresh-training-throughput", "schema_version": 1, "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(), "provenance": source,
              "input_recipe": {"path": str(source_path), "sha256": source_digest,
                               "configuration": original_config},
              "configuration": config_to_dict(config), "tokens_per_update": tokens,
              "warmup_updates": args.warmup, "timed_updates": args.iterations,
              "accumulation_steps": config.data.batch_size // config.train.micro_batch_size,
              "samples": [], "scratch_removed": False,
              "measurement_notes": [
                  "Fresh initialization and optimizer; the scientific recipe and learning-rate horizon are unchanged.",
                  "Steady completion intervals include data loading, transfer, update checks, and ordinary logging.",
                  "Warmup, initialization, evaluation, and checkpoint writes are outside steady measurements.",
                  "Single-GPU training omits the per-group telemetry used by data-parallel training."]}
    write_report(output, report)
    originals = {name: getattr(loop, name) for name in ("train_step", "accumulated_train_step")}
    last_end = None

    def measured(function):
        def wrapped(*positional, **keywords):
            nonlocal last_end
            begin = time.perf_counter()
            result = function(*positional, **keywords)
            jax.block_until_ready(result)
            finish = time.perf_counter()
            loss = float(result[1])
            numerics = np.asarray(jax.device_get(result[2]), dtype=np.float64) if len(result) > 2 else None
            finite = math.isfinite(loss) and (numerics is None or bool(np.all(np.isfinite(numerics))))
            report["samples"].append({
                "step": int(result[0].step), "loss": loss if math.isfinite(loss) else None,
                "finite_loss_and_update_numerics": finite,
                "update_numerics": ([float(value) if math.isfinite(value) else None for value in numerics]
                                    if numerics is not None else None),
                "compute_seconds": finish - begin,
                "interval_seconds": None if last_end is None else finish - last_end})
            last_end = finish
            return result
        return wrapped

    started = time.perf_counter()
    try:
        for name, function in originals.items():
            setattr(loop, name, measured(function))
        report["outcome"] = loop.run_training(config, resume=False, stop_after_step=end_step)
        samples = report["samples"]
        if [sample["step"] for sample in samples] != list(range(1, end_step + 1)):
            raise RuntimeError("Unexpected training update sequence")
        report["all_finite"] = all(sample["finite_loss_and_update_numerics"] for sample in samples)
        if not report["all_finite"]:
            raise FloatingPointError("Nonfinite training loss or update numerics")
        measured_samples = samples[args.warmup:]
        report["steady_training"] = summarize([sample["interval_seconds"] for sample in measured_samples], tokens)
        report["steady_compute"] = summarize([sample["compute_seconds"] for sample in measured_samples], tokens)
        report["gpu_memory_stats"] = devices[0].memory_stats()
        report["source_recipe_unchanged"] = digest(source_path) == source_digest
        if not report["source_recipe_unchanged"]:
            raise RuntimeError("Source recipe changed during the benchmark")
        report["resolved_configuration"] = config_to_dict(config)
        report["runtime_audit"] = report["outcome"]
        for filename, key in (("runtime.json", "runtime_audit"), ("data_identity.json", "data_identity"),
                              ("metadata.json", "training_metadata")):
            path = scratch / filename
            if path.exists():
                report[key] = json.loads(path.read_text())
                report[f"{key}_sha256"] = digest(path)
        report["parameter_count"] = report.get("training_metadata", {}).get("parameter_count")
        report["resolved_backend"] = config.model.attention_backend
        report["resolved_implementation"] = report.get("training_metadata", {}).get("attention_implementation")
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
        # Persist the successful measurements before deleting their disposable checkpoint.
        write_report(output, report)
        if report["status"] == "complete":
            shutil.rmtree(scratch)
            report["scratch_removed"] = True
            write_report(output, report)
    print("TRAINING_THROUGHPUT " + json.dumps(report["steady_training"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--experiment", default="attention_scaling", help="Study used for the frozen execution hash")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
