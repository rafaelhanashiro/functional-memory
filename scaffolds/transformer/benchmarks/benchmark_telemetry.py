"""Time complete production updates with and without training telemetry.

Loads a frozen run recipe and replaces only its data with resident synthetic tokens of
the recipe's batch shape (global batch, microbatch, sequence length). Telemetry-off and
telemetry-on trials alternate on the same devices, each from the same initial
parameters and optimizer state with the same tokens and update RNGs. Every update
synchronizes the loss and the update numerics as the training loop does; telemetry
trials also fetch the statistics every ``--log-every`` updates. Compilation and warmup
updates are excluded from the timings. Run this in an allocation, never on the login node.

  python -m scaffolds.transformer.benchmarks.benchmark_telemetry \\
    --config experiments/attention_scaling/runs/<run_id>/config.yaml \\
    --device-count 2 --output /path/on/nfs/telemetry-benchmark.json
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
import time

import jax
import numpy as np

from scaffolds.transformer.benchmarks.benchmark_attention import provenance, write_report
from scaffolds.transformer.benchmarks.benchmark_train_step import (
    fresh_replicated_state, immutable_host_state, parameter_digest,
)
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.training.distributed import make_data_parallel_functions, put_train_batches
from scaffolds.transformer.training.state import create_state


def run_trial(state, functions, batches, rngs, *, collect: bool, warmup: int, iterations: int, log_every: int):
    samples_ms, losses, warmup_seconds, statistics = [], [], [], None
    for index in range(warmup + iterations):
        start = time.perf_counter()
        outputs = functions.train_step(state, batches, rngs[index])
        state = outputs[0]
        loss = float(outputs[1])
        np.asarray(jax.device_get(outputs[2]))
        if collect and (index + 1) % log_every == 0:
            statistics = {key: float(value) for key, value in jax.device_get(outputs[3]).items()}
        elapsed = time.perf_counter() - start
        if not math.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {index + 1}: {loss}")
        if index < warmup:
            warmup_seconds.append(elapsed)
        else:
            samples_ms.append(elapsed * 1000)
            losses.append(loss)
    return {
        "warmup_seconds": warmup_seconds,
        "median_ms": float(np.median(samples_ms)),
        "p10_ms": float(np.quantile(samples_ms, 0.1)), "p90_ms": float(np.quantile(samples_ms, 0.9)),
        "samples_ms": samples_ms, "losses": losses,
    }, statistics


def benchmark(args) -> dict:
    visible = jax.local_devices(backend="gpu")
    if len(visible) < args.device_count:
        raise ValueError(f"Benchmark requires {args.device_count} GPUs; found {len(visible)}")
    devices = tuple(visible[:args.device_count])
    config = load_config(args.config)
    config.data.source = "synthetic"
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.train.wandb_mode = "disabled"
    validate_config(config)
    micro = config.train.micro_batch_size
    if config.data.batch_size % micro or micro % args.device_count:
        raise ValueError("The recipe's batch must split into microbatches divisible by the device count")
    token_shape = (config.data.batch_size // micro, micro, config.data.sequence_length)
    tokens = np.random.default_rng(args.seed).integers(0, config.model.vocab_size, size=token_shape, dtype=np.int32)
    report = {
        "format": "transformer-telemetry-overhead-benchmark", "schema_version": 1,
        "status": "running", "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(devices[0]),
        "configuration": {
            "recipe": str(args.config), "recipe_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "model_and_training_config": config_to_dict(config),
            "device_count": args.device_count,
            "devices": [{"id": device.id, "kind": device.device_kind} for device in devices],
            "token_shape": list(token_shape), "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
            "warmup": args.warmup, "iterations": args.iterations, "trials": args.trials, "log_every": args.log_every,
            "timing": ("one update: compiled data-parallel step, loss and update-numerics synchronization, and "
                       "(telemetry trials) the statistics fetch every log_every updates"),
            "order": "trials alternate which variant runs first",
        },
        "trials": [],
    }
    write_report(args.output, report)
    try:
        with jax.default_device(devices[0]):
            initialized = create_state(config, jax.random.PRNGKey(args.seed))
            jax.block_until_ready(initialized)
            reference = immutable_host_state(initialized)
            del initialized
        digest, parameter_count = parameter_digest(reference.params)
        report["initialization"] = {"parameter_sha256": digest, "parameter_count": parameter_count}
        functions = {collect: make_data_parallel_functions(reference, devices, check_all_gradients=True,
                                                           collect_telemetry=collect)
                     for collect in (False, True)}
        master = jax.random.PRNGKey(args.seed)
        for trial in range(args.trials):
            order = (False, True) if trial % 2 == 0 else (True, False)
            entry = {"trial": trial, "order": ["on" if collect else "off" for collect in order]}
            for collect in order:
                current = functions[collect]
                state = fresh_replicated_state(reference, current.replicated_sharding)
                batches = put_train_batches(tokens, current.train_batch_sharding)
                rngs = [jax.device_put(jax.random.fold_in(master, index + 1), current.replicated_sharding)
                        for index in range(args.warmup + args.iterations)]
                jax.block_until_ready((state, batches, rngs))
                timing, statistics = run_trial(state, current, batches, rngs, collect=collect, warmup=args.warmup,
                                               iterations=args.iterations, log_every=args.log_every)
                label = "on" if collect else "off"
                entry[label] = timing
                if statistics is not None:
                    report["statistics_sample"] = {"trial": trial, "count": len(statistics), "values": statistics}
                print(f"trial={trial} telemetry={label} median_ms={timing['median_ms']:.3f} "
                      f"p10={timing['p10_ms']:.3f} p90={timing['p90_ms']:.3f}", flush=True)
                del state, batches, rngs
            entry["overhead"] = entry["on"]["median_ms"] / entry["off"]["median_ms"] - 1
            entry["max_abs_loss_difference"] = float(np.max(np.abs(
                np.asarray(entry["on"]["losses"]) - np.asarray(entry["off"]["losses"]))))
            report["trials"].append(entry)
            write_report(args.output, report)
        off = np.concatenate([trial["off"]["samples_ms"] for trial in report["trials"]])
        on = np.concatenate([trial["on"]["samples_ms"] for trial in report["trials"]])
        report["summary"] = {
            "off_median_ms": float(np.median(off)), "on_median_ms": float(np.median(on)),
            "overhead": float(np.median(on) / np.median(off) - 1),
            "trial_overheads": [trial["overhead"] for trial in report["trials"]],
            "max_abs_loss_difference": max(trial["max_abs_loss_difference"] for trial in report["trials"]),
        }
        print(f"summary off={report['summary']['off_median_ms']:.3f}ms on={report['summary']['on_median_ms']:.3f}ms "
              f"overhead={100 * report['summary']['overhead']:+.2f}% "
              f"max_loss_diff={report['summary']['max_abs_loss_difference']:.3g}", flush=True)
        report["status"] = "complete"
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
    parser.add_argument("--config", type=Path, required=True, help="A frozen run recipe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-count", type=int, choices=(2, 4), default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 3 or args.iterations < 20 or args.trials < 2:
        parser.error("Require at least 3 warmup updates, 20 timed updates, and 2 trials")
    benchmark(args)


if __name__ == "__main__":
    main()
