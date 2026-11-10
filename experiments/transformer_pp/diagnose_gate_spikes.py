"""Read-only checkpoint replay and frozen-gradient attribution of a retained interval.

The native training step is used unchanged. Instrumentation runs only after replay,
on a saved pre-update state, and must demonstrate parity before it is interpreted.
The source checkpoint and all source measurements are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from scaffolds.transformer.configs import load_config
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training import telemetry
from scaffolds.transformer.training.distributed import _loss_and_grads
from scaffolds.transformer.training.loop import accumulated_train_step
from scaffolds.transformer.training.state import create_state


EXPLICIT_GROUPS = frozenset({
    "forget_gate_kernel", "forget_gate_key_kernel", "forget_gate_log_rate",
    "forget_gate_dt_bias", "qk_norm_head_log_scale", "query_polynomial_kernel",
    "query_polynomial_amplitude", "query_polynomial_bias",
})


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return serializable(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def gradient_statistics(grads, layers):
    """Device reductions, including root-owned arrays with an explicit layer axis."""
    grouped = {}
    per_layer = jnp.zeros((layers,), jnp.float32)
    per_order = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        parts = [str(getattr(entry, "key", getattr(entry, "idx", entry))) for entry in path]
        root = parts[0]
        group = root if root in EXPLICIT_GROUPS else telemetry.parameter_group(path)
        squared = jnp.square(leaf.astype(jnp.float32))
        norm2 = jnp.sum(squared)
        grouped[group] = grouped.get(group, jnp.asarray(0., jnp.float32)) + norm2
        layer_match = next((re.fullmatch(r"layers_(\d+)", part) for part in parts
                            if re.fullmatch(r"layers_(\d+)", part)), None)
        if layer_match is not None:
            per_layer = per_layer.at[int(layer_match.group(1))].add(norm2)
        elif leaf.ndim and leaf.shape[0] == layers:
            per_layer = per_layer + jnp.sum(squared, axis=tuple(range(1, leaf.ndim)))
        if root in EXPLICIT_GROUPS and root != "qk_norm_head_log_scale":
            # Gate tensors [L,H,R,D/1]; coefficient W [L,H,D,R]; a,b [L,H,R].
            order_axis = leaf.ndim - 1 if root.startswith("query_polynomial") else 2
            axes = tuple(i for i in range(leaf.ndim) if i != order_axis)
            per_order[root] = jnp.sqrt(jnp.sum(squared, axis=axes))
    return {"global_norm": optax.tree.norm(grads),
            "group_norms": {name: jnp.sqrt(value) for name, value in grouped.items()},
            "layer_norms": jnp.sqrt(per_layer), "order_norms": per_order}


def frozen_probe(instrument, layers):
    """Build the original microbatch scan with optional forward telemetry."""
    reductions = {}

    @nnx.jit
    def probe(state, batches, rngs):
        def body(total, inputs):
            batch, rng = inputs
            result = _loss_and_grads(
                state.graphdef, state.params, state.step, batch, rng,
                **({"telemetry_reductions": reductions} if instrument else {}),
            )
            nll, grads = result[:2]
            statistics = result[2] if instrument else {}
            summed = jax.tree.map(jnp.add, total, grads)
            return summed, {"nll": nll, "gradients": gradient_statistics(grads, layers),
                            "activations": statistics}

        zero = jax.tree.map(jnp.zeros_like, state.params)
        total, micro = jax.lax.scan(body, zero, (batches, rngs))
        grads = jax.tree.map(lambda value: value / batches.shape[0], total)
        return jnp.mean(micro["nll"]), grads, gradient_statistics(grads, layers), micro

    return probe, reductions


@jax.jit
def gradient_parity(left, right):
    difference = jax.tree.map(lambda a, b: a - b, left, right)
    return {"relative_l2": optax.tree.norm(difference) / jnp.maximum(optax.tree.norm(left), 1e-30),
            "absolute_l2": optax.tree.norm(difference),
            "max_absolute": jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in jax.tree.leaves(difference)]))}


def native_log_samples(run, first, last):
    path = run / "training.log"
    if not path.exists():
        return {}
    pattern = r"step=(\d+) train_nll=([0-9.eE+\-]+).*?gradient_norm_pre_clip=([0-9.eE+\-]+)"
    return {int(m[1]): {"nll": float(m[2]), "gradient_norm": float(m[3])}
            for m in re.finditer(pattern, path.read_text(errors="replace")) if first <= int(m[1]) <= last}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=7500)
    parser.add_argument("--end-step", type=int, default=7570)
    args = parser.parse_args()
    run, output = args.parent_run.resolve(), args.output.resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("Diagnostic output must be separate from its source run")
    output.mkdir(parents=True, exist_ok=True)
    summary_path, metrics_path = output / "summary.json", output / "metrics.jsonl"
    if summary_path.exists() or metrics_path.exists():
        raise FileExistsError("Refusing to overwrite diagnostic measurements")
    config = load_config(run / "config.yaml")
    if config.model.dropout != 0 or config.optim.coefficient_penalty != 0:
        raise ValueError("This diagnostic requires dropout=0 and coefficient_penalty=0")
    if config.model.remat:
        raise ValueError("Activation telemetry requires remat=False")
    if not args.checkpoint < args.end_step <= config.optim.total_steps:
        raise ValueError("Replay interval must lie inside the original training horizon")
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
        "format": "gated-polynomial-gradient-replay", "version": 1, "status": "running",
        "started_utc": utc(), "parent_run_id": run.name, "parent_checkpoint_step": args.checkpoint,
        "parent_checkpoint_path": str(checkpoint), "parent_source_commit": original["source_commit"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_artifact_sha256": hashes, "device": devices[0].device_kind,
        "interval": [args.checkpoint + 1, args.end_step],
        "limitations": [
            "This replay addresses the retained late interval, not the earlier maximum at updates 4001-4500.",
            "Instrumented measurements are attributable only subject to native replay and gradient parity.",
            "Gradient norms are measured before global clipping; actual Adam update norms are separate quantities.",
        ],
    }
    write_json(summary_path, report)
    started = time.perf_counter()
    try:
        rng = jax.random.PRNGKey(config.train.seed)
        state = create_state(config, rng)
        with ocp.StandardCheckpointer() as checkpointer:
            state = checkpointer.restore(checkpoint / "default",
                                         target={"state": state})["state"]
        if int(state.step) != args.checkpoint:
            raise ValueError("Restored step differs from requested checkpoint")
        accumulation = config.data.batch_size // config.train.micro_batch_size
        if accumulation * config.train.micro_batch_size != config.data.batch_size:
            raise ValueError("Nonintegral gradient accumulation")
        iterator = PackedBatchIterator(config.data.train_manifest, batch_size=config.train.micro_batch_size,
                                       sequence_length=config.data.sequence_length,
                                       start_sequence=args.checkpoint * config.data.batch_size)
        history = native_log_samples(run, args.checkpoint + 1, args.end_step)
        samples, worst = [], None
        for step in range(args.checkpoint + 1, args.end_step + 1):
            batch_host = np.stack([next(iterator) for _ in range(accumulation)])
            batches = jnp.asarray(batch_host)
            micro_rngs = jax.random.split(jax.random.fold_in(rng, step), accumulation)
            before = state
            state, loss, numerics = accumulated_train_step(state, batches, micro_rngs, check_all_gradients=True)
            nll, norm, parameter_norm = map(float, np.asarray(jax.device_get(numerics)))
            if not all(math.isfinite(x) for x in (float(loss), nll, norm, parameter_norm)):
                raise FloatingPointError(f"Nonfinite native replay at update {step}")
            row = {"event": "native_replay", "step": step, "train_nll": nll,
                   "gradient_norm_pre_clip": norm, "gradient_clip_scale": min(1., config.optim.grad_clip_norm / max(norm, 1e-30))}
            if step in history:
                row["historical_nll_absolute_error"] = abs(nll - history[step]["nll"])
                row["historical_gradient_absolute_error"] = abs(norm - history[step]["gradient_norm"])
            samples.append(row)
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            if worst is None or norm > worst["row"]["gradient_norm_pre_clip"]:
                worst = {"row": row, "state": jax.device_get(before),
                         "batches": batch_host.copy(), "rngs": np.asarray(jax.device_get(micro_rngs))}
            print("REPLAY " + json.dumps(row, sort_keys=True), flush=True)
        report["native_replay"] = {
            "updates": len(samples), "max_gradient_norm": max(x["gradient_norm_pre_clip"] for x in samples),
            "max_gradient_step": worst["row"]["step"],
            "clipped_updates": sum(x["gradient_norm_pre_clip"] > config.optim.grad_clip_norm for x in samples),
            "historical_log_samples": len(history),
            "max_historical_nll_absolute_error": max((x.get("historical_nll_absolute_error", 0) for x in samples)),
            "max_historical_gradient_absolute_error": max((x.get("historical_gradient_absolute_error", 0) for x in samples)),
        }
        windows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
        matching = [row for row in windows if row.get("training_window_start_step") == args.checkpoint
                    and row.get("step") == args.end_step and "train_gradient_norm_max" in row]
        if matching:
            expected = float(matching[0]["train_gradient_norm_max"])
            report["native_replay"]["historical_window_max"] = expected
            report["native_replay"]["historical_window_max_absolute_error"] = abs(expected - report["native_replay"]["max_gradient_norm"])
        write_json(summary_path, report)
        del state, before, batches, micro_rngs
        probe_state = jax.device_put(worst["state"])
        probe_batches, probe_rngs = jnp.asarray(worst["batches"]), jnp.asarray(worst["rngs"])
        repeated, repeated_loss, repeated_metrics = accumulated_train_step(
            probe_state, probe_batches, probe_rngs, check_all_gradients=True)
        repeated_values = np.asarray(jax.device_get(repeated_metrics))
        report["native_repeat"] = {"nll": float(repeated_values[0]), "gradient_norm": float(repeated_values[1]),
                                   "nll_absolute_error": abs(float(repeated_values[0]) - worst["row"]["train_nll"]),
                                   "gradient_absolute_error": abs(float(repeated_values[1]) - worst["row"]["gradient_norm_pre_clip"])}
        update_summary = jax.jit(lambda old, new: gradient_statistics(
            jax.tree.map(lambda after, before: after - before, new, old), config.model.num_layers))
        report["native_actual_parameter_update"] = jax.device_get(update_summary(probe_state.params, repeated.params))
        del repeated, repeated_loss, repeated_metrics
        plain, _ = frozen_probe(False, config.model.num_layers)
        print("PROBE compiling plain frozen-gradient attribution", flush=True)
        plain_loss, plain_grads, plain_stats, plain_micro = plain(probe_state, probe_batches, probe_rngs)
        jax.block_until_ready(plain_loss)
        instrumented, reductions = frozen_probe(True, config.model.num_layers)
        print("PROBE compiling instrumented frozen-gradient attribution", flush=True)
        probe_loss, probe_grads, probe_stats, probe_micro = instrumented(probe_state, probe_batches, probe_rngs)
        parity = jax.device_get(gradient_parity(plain_grads, probe_grads))
        report["worst_update"] = worst["row"]
        report["plain_probe"] = {"nll": float(plain_loss), "gradients": jax.device_get(plain_stats),
                                 "microbatches": jax.device_get(plain_micro)}
        report["instrumented_probe"] = {"nll": float(probe_loss), "gradients": jax.device_get(probe_stats),
                                        "microbatches": jax.device_get(probe_micro), "activation_reductions": reductions}
        native_norm = worst["row"]["gradient_norm_pre_clip"]
        report["probe_parity"] = {
            "plain_vs_native_nll_absolute_error": abs(float(plain_loss) - worst["row"]["train_nll"]),
            "plain_vs_native_gradient_norm_relative_error": abs(float(plain_stats["global_norm"]) - native_norm) / max(native_norm, 1e-30),
            "instrumented_vs_plain_nll_absolute_error": abs(float(probe_loss) - float(plain_loss)),
            "instrumented_vs_plain_gradient": parity,
            "interpretation": "Inspect these discrepancies before attributing a native spike to instrumented statistics.",
        }
        report["source_files_unchanged"] = all(sha256(path) == digest for path, digest in hashes.items())
        if not report["source_files_unchanged"]:
            raise RuntimeError("Protected parent files changed during the diagnostic")
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = utc()
        report["wall_seconds"] = time.perf_counter() - started
        write_json(summary_path, report)
    print("DIAGNOSTIC " + json.dumps(serializable({k: report[k] for k in
          ("status", "native_replay", "probe_parity", "wall_seconds")})), flush=True)


if __name__ == "__main__":
    main()
