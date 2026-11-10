"""Scan data batches at fixed checkpoint weights for a native gradient outlier.

The original accumulated training step computes each gradient, but its updated
state is discarded. This probes batch sensitivity, not a historical trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from experiments.transformer_pp.diagnose_gate_spikes import (
    frozen_probe, gradient_parity, serializable, sha256, utc, write_json,
)
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.loop import accumulated_train_step
from scaffolds.transformer.training.state import create_state


def native_fixed_batch(state, batches, rngs):
    """Execute the unchanged native graph without retaining an optimizer update."""
    discarded, loss, numerics = accumulated_train_step(
        state, batches, rngs, check_all_gradients=True)
    nll, norm, _ = map(float, np.asarray(jax.device_get(numerics)))
    if not all(math.isfinite(value) for value in (float(loss), nll, norm)):
        raise FloatingPointError("Nonfinite native fixed-state batch")
    del discarded
    return nll, norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=7500)
    parser.add_argument("--first-batch-update", type=int, default=7501)
    parser.add_argument("--max-batches", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=1.)
    args = parser.parse_args()
    run, output = args.parent_run.resolve(), args.output.resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("Diagnostic output must be separate from its parent")
    if args.first_batch_update < 1 or args.max_batches < 1 or not math.isfinite(args.threshold) or args.threshold <= 0:
        raise ValueError("Batch indices, batch count, and threshold must be positive")
    output.mkdir(parents=True, exist_ok=True)
    summary_path, metrics_path = output / "summary.json", output / "metrics.jsonl"
    if summary_path.exists() or metrics_path.exists():
        raise FileExistsError("Refusing to overwrite diagnostic measurements")
    config = load_config(run / "config.yaml")
    if config.model.dropout != 0 or config.optim.coefficient_penalty != 0 or config.model.remat:
        raise ValueError("Requires dropout=0, coefficient_penalty=0, and remat=False")
    devices = jax.local_devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("Use exactly one GPU")
    checkpoint = run / "checkpoints" / str(args.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    protected = [run / name for name in ("config.yaml", "record.json", "metrics.jsonl", "best_validation.json")]
    protected += [checkpoint / "_CHECKPOINT_METADATA", checkpoint / "default/_METADATA"]
    hashes = {str(path): sha256(path) for path in protected}
    original = json.loads((run / "record.json").read_text())
    identity = json.loads((run / "data_identity.json").read_text())
    if sha256(config.data.train_manifest) != identity["train_manifest_sha256"]:
        raise ValueError("Training manifest differs from the parent record")
    report = {
        "format": "frozen_batch_gradient_scan", "version": 1, "status": "running",
        "started_utc": utc(), "parent_run_id": run.name, "state_step": args.checkpoint,
        "parent_checkpoint_path": str(checkpoint), "parent_source_commit": original["source_commit"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_artifact_sha256": hashes, "device": devices[0].device_kind,
        "first_batch_update": args.first_batch_update, "max_batches": args.max_batches,
        "threshold": args.threshold, "training_updates_applied": 0,
        "seed": config.train.seed, "global_batch_size": config.data.batch_size,
        "micro_batch_size": config.train.micro_batch_size,
        "sequence_length": config.data.sequence_length,
        "limitations": [
            "Every batch uses the same retained parameters and optimizer state; no learning occurs.",
            "Batch indices identify packed data and RNGs, not the checkpoint's optimizer step.",
            "This scan does not reproduce or attribute a historical training spike.",
            "Instrumented attribution requires native-repeat and frozen-gradient parity.",
        ],
    }
    write_json(summary_path, report)
    started = time.perf_counter()
    try:
        rng = jax.random.PRNGKey(config.train.seed)
        state = create_state(config, rng)
        with ocp.StandardCheckpointer() as checkpointer:
            state = checkpointer.restore(checkpoint / "default", target={"state": state})["state"]
        if int(state.step) != args.checkpoint:
            raise ValueError("Restored step differs from the requested checkpoint")
        report["fixed_state_sha256"] = state_sha256(state)
        accumulation = config.data.batch_size // config.train.micro_batch_size
        if accumulation * config.train.micro_batch_size != config.data.batch_size:
            raise ValueError("Nonintegral gradient accumulation")
        first_sequence = (args.first_batch_update - 1) * config.data.batch_size
        iterator = PackedBatchIterator(config.data.train_manifest, batch_size=config.train.micro_batch_size,
                                       sequence_length=config.data.sequence_length, start_sequence=first_sequence)
        largest = None
        for index in range(args.max_batches):
            batch_update = args.first_batch_update + index
            host = np.stack([next(iterator) for _ in range(accumulation)])
            batches = jnp.asarray(host)
            rngs = jax.random.split(jax.random.fold_in(rng, batch_update), accumulation)
            nll, norm = native_fixed_batch(state, batches, rngs)
            cursor = (batch_update - 1) * config.data.batch_size
            row = {"event": "frozen_batch_gradient", "state_step": args.checkpoint,
                   "batch_update": batch_update, "first_sequence": cursor,
                   "next_sequence": cursor + config.data.batch_size, "train_nll": nll,
                   "gradient_norm_pre_clip": norm, "exceeds_threshold": norm > args.threshold}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            if largest is None or norm > largest["row"]["gradient_norm_pre_clip"]:
                largest = {"row": row, "batches": host.copy(), "rngs": np.asarray(jax.device_get(rngs))}
            print("FROZEN_BATCH " + json.dumps(row, sort_keys=True), flush=True)
            if row["exceeds_threshold"]:
                break
        report.update(sampled_count=index + 1, hit_threshold=largest["row"]["exceeds_threshold"],
                      largest_batch=largest["row"])
        report["fixed_state_after_scan_sha256"] = state_sha256(state)
        report["fixed_state_unchanged"] = report["fixed_state_after_scan_sha256"] == report["fixed_state_sha256"]
        if not report["fixed_state_unchanged"]:
            raise RuntimeError("The fixed checkpoint state changed during the batch scan")
        write_json(summary_path, report)
        del batches, rngs, host
        probe_batches, probe_rngs = jnp.asarray(largest["batches"]), jnp.asarray(largest["rngs"])
        repeated_nll, repeated_norm = native_fixed_batch(state, probe_batches, probe_rngs)
        report["native_repeat"] = {
            "nll": repeated_nll, "gradient_norm": repeated_norm,
            "nll_absolute_error": abs(repeated_nll - largest["row"]["train_nll"]),
            "gradient_absolute_error": abs(repeated_norm - largest["row"]["gradient_norm_pre_clip"]),
        }
        plain, _ = frozen_probe(False, config.model.num_layers)
        print("PROBE compiling plain frozen-gradient attribution", flush=True)
        plain_loss, plain_grads, plain_stats, plain_micro = plain(state, probe_batches, probe_rngs)
        jax.block_until_ready(plain_loss)
        instrumented, reductions = frozen_probe(True, config.model.num_layers)
        print("PROBE compiling instrumented frozen-gradient attribution", flush=True)
        probe_loss, probe_grads, probe_stats, probe_micro = instrumented(state, probe_batches, probe_rngs)
        parity = jax.device_get(gradient_parity(plain_grads, probe_grads))
        report["plain_probe"] = {"nll": float(plain_loss), "gradients": jax.device_get(plain_stats),
                                 "microbatches": jax.device_get(plain_micro)}
        report["instrumented_probe"] = {"nll": float(probe_loss), "gradients": jax.device_get(probe_stats),
                                        "microbatches": jax.device_get(probe_micro), "activation_reductions": reductions}
        norm = largest["row"]["gradient_norm_pre_clip"]
        report["probe_parity"] = {
            "plain_vs_native_nll_absolute_error": abs(float(plain_loss) - largest["row"]["train_nll"]),
            "plain_vs_native_gradient_norm_relative_error": abs(float(plain_stats["global_norm"]) - norm) / max(norm, 1e-30),
            "instrumented_vs_plain_nll_absolute_error": abs(float(probe_loss) - float(plain_loss)),
            "instrumented_vs_plain_gradient": parity,
        }
        report["fixed_state_final_sha256"] = state_sha256(state)
        report["fixed_state_unchanged"] &= report["fixed_state_final_sha256"] == report["fixed_state_sha256"]
        report["source_files_unchanged"] = all(sha256(path) == digest for path, digest in hashes.items())
        if not report["fixed_state_unchanged"] or not report["source_files_unchanged"]:
            raise RuntimeError("Fixed state or protected parent files changed during probing")
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = utc()
        report["wall_seconds"] = time.perf_counter() - started
        write_json(summary_path, report)
    print("DIAGNOSTIC " + json.dumps(serializable({key: report[key] for key in
          ("status", "sampled_count", "hit_threshold", "largest_batch", "probe_parity", "wall_seconds")})), flush=True)


if __name__ == "__main__":
    main()
