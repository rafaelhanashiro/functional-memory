"""Fit only free polynomial coefficients on a fixed, explicitly selected input.

Coefficient tables have [example, layer, query, head, order] axes.  Their stored
coordinates are r! c_r; the model receives ordinary monomial coefficients.
The pretrained parameter tree is an immutable argument, never an optimizer input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
import yaml

from scaffolds.records import capture_artifacts, update_record, validate_record
from scaffolds.transformer.configs import Config
from scaffolds.transformer.configs.base import validate_config
from scaffolds.transformer.configs.loader import _update_dataclass
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_config(path: Path) -> Config:
    """Read the frozen historical recipe without changing its file."""
    raw = yaml.safe_load(path.read_text())
    config = Config()
    # A retired GDN setting has no meaning for this dot-product checkpoint.
    if raw["model"].get("attention_mechanism") == "dot_product":
        names = {field.name for field in fields(type(config.model))}
        if "gdn_polynomial_degree" not in names:
            raw["model"].pop("gdn_polynomial_degree", None)
    _update_dataclass(config, raw, context=str(path))
    validate_config(config)
    return config


def restore_parent(config: Config, checkpoint: Path):
    """Restore only the pretrained parameters; never allocate parent Adam state."""
    model = build_model(config.model, nnx.Rngs(params=config.train.seed, dropout=config.train.seed))
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    graph, template = nnx.split(model, nnx.Param)
    directory = checkpoint / "default" if (checkpoint / "default").is_dir() else checkpoint
    item = {"state": {"params": template}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            directory,
            args=ocp.args.PyTreeRestore(item=item, partial_restore=True,
                restore_args=ocp.checkpoint_utils.construct_restore_args(item)),
        )
    return graph, restored["state"]["params"]


def make_functions(graph, degree: int, oracle: dict):
    factorial = jnp.asarray([math.factorial(order) for order in range(degree + 1)], jnp.float32)
    tx = optax.chain(
        optax.clip_by_global_norm(float(oracle["clip_norm"])),
        optax.scale_by_adam(b1=float(oracle["beta1"]), b2=float(oracle["beta2"]),
                           eps=float(oracle["epsilon"])),
    )

    def nll(logits, tokens):
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:],
        ).mean(axis=1)

    @jax.jit
    def capture(params, tokens):
        model = nnx.merge(graph, params)
        logits, coefficients = model(tokens[:, :-1], deterministic=True, return_coefficients=True)
        table = jnp.swapaxes(jnp.stack(coefficients), 0, 1) * factorial
        replay = model(tokens[:, :-1], deterministic=True,
                       query_coefficients=jnp.swapaxes(table / factorial, 0, 1))
        return table, nll(logits, tokens), nll(replay, tokens), jnp.max(jnp.abs(
            logits.astype(jnp.float32) - replay.astype(jnp.float32)))

    def losses(table, params, tokens):
        model = nnx.merge(graph, params)
        logits = model(tokens[:, :-1], deterministic=True,
                       query_coefficients=jnp.swapaxes(table / factorial, 0, 1))
        return nll(logits, tokens)

    def objective(table, params, tokens):
        per_example = losses(table, params, tokens)
        # Every independent example sees the same gradient as a B=1 run.
        return per_example.sum(), per_example

    @jax.jit
    def step(table, optimizer, params, tokens, learning_rate):
        (_, per_example), gradient = jax.value_and_grad(objective, has_aux=True)(table, params, tokens)
        axes = tuple(range(1, gradient.ndim))
        norms = jnp.sqrt(jnp.sum(jnp.square(gradient), axis=axes))
        updates, optimizer = jax.vmap(tx.update)(gradient, optimizer, table)
        candidate = optax.apply_updates(table, -learning_rate * updates)
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(norms))
        finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(optimizer)]))
        return candidate, optimizer, per_example, norms, finite

    return capture, jax.jit(jax.vmap(tx.init)), step, jax.jit(losses)


def plateau_action(history: list[float], step: int, reduced_at: int | None, oracle: dict):
    """Return reduce/stop/continue from best-so-far loss at evaluated steps."""
    window = int(oracle["plateau_window"])
    if len(history) <= window or step < int(oracle["initial_updates"]):
        return "continue"
    flat = history[-window - 1] - history[-1] < float(oracle["plateau_tolerance"])
    if reduced_at is None:
        return "reduce" if flat else "continue"
    if step - reduced_at >= int(oracle["confirmation_updates"]) and flat:
        return "stop"
    return "continue"


def save_state(root: Path, step: int, tables, optimizers, best, controller: dict) -> Path:
    """A committed checkpoint contains both resumable and best coefficient state."""
    path = root / str(step)
    if (path / "controller.json").exists():
        # Coefficients/moments at a given step are immutable. A plateau decision
        # can update only the controller without risking the large array save.
        _write_json(path / "controller.json", controller)
        return path
    temporary = root / f".pending-{step}-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    if path.exists():  # Only an incomplete save can reach here.
        shutil.rmtree(path)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(temporary / "default", {"tables": tables, "optimizers": optimizers, "best": best})
    _write_json(temporary / "controller.json", controller)
    temporary.rename(path)
    for previous in root.iterdir():
        if previous.is_dir() and previous.name.isdecimal() and previous != path:
            shutil.rmtree(previous)
    return path


def restore_state(path: Path, tables, optimizers):
    with ocp.StandardCheckpointer() as checkpointer:
        state = checkpointer.restore(path / "default", target={
            "tables": tables, "optimizers": optimizers, "best": tables,
        })
    return state, json.loads((path / "controller.json").read_text())


def _latest(root: Path):
    if not root.exists():
        return None
    paths = [p for p in root.iterdir() if p.name.isdecimal() and (p / "controller.json").exists()]
    return max(paths, key=lambda p: int(p.name)) if paths else None


def _retain_completed_best(run: Path, outcome: dict) -> Path:
    """Idempotently prune only after the converged coefficient save is durable."""
    root = run / "checkpoints"
    retained = root / str(outcome["best_step"])
    if not (retained / "best").is_dir():
        raise RuntimeError("Completion record has no retained best coefficient checkpoint")
    for path in root.iterdir():
        if path.is_dir() and path != retained:
            shutil.rmtree(path)
    if (retained / "default").exists():
        shutil.rmtree(retained / "default")
    (retained / "controller.json").unlink(missing_ok=True)
    return retained


def qualify(run: Path, functions, params, tokens, table, optimizer, learning_rate):
    capture, _, train_step, evaluate = functions
    _, original_nll, replay_nll, logits_error = capture(params, tokens)
    original_nll, replay_nll, logits_error = jax.device_get((original_nll, replay_nll, logits_error))
    if not np.allclose(original_nll, replay_nll, rtol=1e-5, atol=1e-5) or float(logits_error) > 0.05:
        raise RuntimeError(f"Parent replay parity failed: logits error {logits_error}")
    updated, updated_opt, before, norms, finite = train_step(table, optimizer, params, tokens, learning_rate)
    before, norms, finite = jax.device_get((before, norms, finite))
    if not finite or not np.all(np.isfinite(norms)) or not np.any(norms > 0):
        raise RuntimeError("Coefficient gradient/update acceptance failed")
    after = np.asarray(evaluate(updated, params, tokens))
    # A bad initial optimizer step must be discovered before production.
    if not np.all(np.isfinite(after)) or float(after.mean()) >= float(before.mean()):
        raise RuntimeError(f"Initial coefficient update did not improve NLL: {before.mean()} -> {after.mean()}")
    directory = run / ".runtime" / "oracle-checkpoint-qualification"
    if directory.exists():
        shutil.rmtree(directory)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(directory, {"table": updated, "optimizer": updated_opt})
        restored = checkpointer.restore(directory, target={"table": updated, "optimizer": updated_opt})
    for left, right in zip(jax.tree.leaves((updated, updated_opt)),
                           jax.tree.leaves((restored["table"], restored["optimizer"]))):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise RuntimeError("Coefficient checkpoint round-trip changed an array")
    next_a = train_step(updated, updated_opt, params, tokens, learning_rate)
    next_b = train_step(restored["table"], restored["optimizer"], params, tokens, learning_rate)
    for left, right in zip(jax.tree.leaves(next_a), jax.tree.leaves(next_b)):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise RuntimeError("Resumed coefficient update differs from uninterrupted update")
    shutil.rmtree(directory)
    return {"status": "passed", "completed_utc": _utc(), "parent_nll": float(before.mean()),
            "one_update_nll": float(after.mean()), "max_logit_replay_error": float(logits_error),
            "gradient_norm_min": float(norms.min()), "gradient_norm_max": float(norms.max()),
            "checkpoint_resume_exact": True, "frozen_parameter_sha256": state_sha256(params),
            "device": jax.devices()[0].device_kind}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    oracle = recipe["oracle"]
    print(f"Starting coefficient oracle {run.name}: {oracle['sequence_count']} fixed sequences", flush=True)
    if len(jax.devices()) != 1:
        raise ValueError("Coefficient oracle requires exactly one visible accelerator")
    if (run / "completion.json").exists() and not args.qualify_only:
        outcome = json.loads((run / "completion.json").read_text())
        checkpoint = _retain_completed_best(run, outcome)
        _write_json(run / "metadata.json", {**json.loads((run / "metadata.json").read_text()), **outcome})
        update_record(run, {**capture_artifacts(run, checkpoint_directory=checkpoint),
                            "status": "complete", "oracle_outcome": outcome})
        print("Run already complete", flush=True)
        return
    repository = Path(__file__).resolve().parents[2]
    runtime = {"source_commit": os.environ.get("COEFFICIENT_ORACLE_SOURCE_COMMIT"),
               "device_kind": jax.devices()[0].device_kind, "device_count": 1,
               "resolved_attention_backend": recipe["model"]["attention_backend"],
               "coefficient_layout": "example,layer,query,head,order",
               "optimizer_coordinates": "factorial(order)*monomial_coefficient",
               "microbatch_size": int(oracle["microbatch_size"]),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "node": os.uname().nodename,
               "python_executable": sys.executable,
               "dependency_lock_sha256": _sha256(repository / "scaffolds/transformer/uv.lock"),
               "versions": {name: version(name) for name in ("jax", "flax", "optax", "orbax-checkpoint")},
               "environment": {name: os.environ.get(name) for name in (
                   "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "XLA_FLAGS", "JAX_PLATFORMS")}}
    record = json.loads((run / "record.json").read_text())
    attempts = record.get("attempts", [])
    attempts.append({"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
                     "started_utc": _utc(), "node": runtime["node"],
                     "slurm_partition": runtime["slurm_partition"], "gpu_type": runtime["device_kind"],
                     "gpu_count": 1, "phase": "qualification" if args.qualify_only else "training",
                     "source_commit": runtime["source_commit"], "python_executable": sys.executable,
                     "dependency_lock_sha256": runtime["dependency_lock_sha256"]})
    update_record(run, {"oracle_runtime": runtime, "attempts": attempts,
                        "status": record["status"] if args.qualify_only else "running"})
    source_config = Path(oracle["parent_config"])
    if _sha256(source_config) != oracle["parent_config_sha256"]:
        raise ValueError("Parent configuration checksum changed")
    config = parent_config(source_config)
    if _sha256(Path(config.data.eval_manifest)) != oracle["eval_manifest_sha256"]:
        raise ValueError("Evaluation manifest checksum changed")
    count = int(oracle["sequence_count"])
    microbatch = min(int(oracle["microbatch_size"]), count)
    if count % microbatch:
        raise ValueError("Sequence count must be divisible by microbatch size")
    batch = next(iter(PackedBatchIterator(
        config.data.eval_manifest, batch_size=count, sequence_length=config.data.sequence_length,
        start_sequence=int(oracle["data_start"]),
    )))
    data_identity = {"manifest": str(config.data.eval_manifest), "manifest_sha256": oracle["eval_manifest_sha256"],
                     "split": "eval", "start_sequence": int(oracle["data_start"]), "sequence_count": count,
                     "batch_sha256": hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest(),
                     "sequence_length": config.data.sequence_length,
                     "interpretation": "Fixed fitting inputs; these are not held-out measurements after adaptation."}
    parent_run = source_config.parent
    parent_record = json.loads((parent_run / "record.json").read_text())
    if parent_record["selected_checkpoint"]["sha256"] != oracle["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint checksum changed")
    validate_record(parent_run, checkpoint_hashes=False)
    print("Restoring frozen parent parameters", flush=True)
    graph, params = restore_parent(config, Path(oracle["parent_checkpoint"]))
    print("Parent restored; compiling coefficient capture and replay", flush=True)
    functions = make_functions(graph, config.model.attention_degree, oracle)
    capture, initialize_optimizer, train_step, evaluate = functions
    token_chunks = [jax.device_put(batch[i:i + microbatch]) for i in range(0, count, microbatch)]
    tables, optimizers, parent_losses = [], [], []
    for index, tokens in enumerate(token_chunks):
        table, losses, replay_losses, error = capture(params, tokens)
        losses, replay_losses, error = jax.device_get((losses, replay_losses, error))
        if not np.allclose(losses, replay_losses, rtol=1e-5, atol=1e-5) or float(error) > 0.05:
            raise RuntimeError("Captured coefficients do not reproduce parent logits/loss")
        tables.append(table)
        optimizers.append(initialize_optimizer(table))
        parent_losses.extend(replay_losses.tolist())
        if index % 4 == 0 or index + 1 == len(token_chunks):
            print(f"Captured coefficients for microbatch {index + 1}/{len(token_chunks)}; NLL {np.mean(replay_losses):.6f}", flush=True)
    learning_rate = float(oracle["learning_rate"])
    runtime.update(trainable_parameter_count=sum(int(value.size) for value in tables),
                   microbatch_size=microbatch, resolved_attention_backend=config.model.attention_backend)
    print("Qualifying coefficient-only gradient, improvement, and checkpoint resume", flush=True)
    acceptance = qualify(run, functions, params, token_chunks[0], tables[0], optimizers[0], learning_rate)
    print(f"Qualification passed: NLL {acceptance['parent_nll']:.6f} -> {acceptance['one_update_nll']:.6f}", flush=True)
    _write_json(run / "oracle_acceptance.json", acceptance)
    if args.qualify_only:
        attempts[-1].update(status="passed", finished_utc=_utc(), exit_code=0)
        update_record(run, {**capture_artifacts(run), "acceptance": acceptance,
                            "oracle_runtime": runtime, "attempts": attempts})
        print(json.dumps(acceptance), flush=True)
        return
    _write_json(run / "data_identity.json", data_identity)
    checkpoint_root = run / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    for incomplete in checkpoint_root.glob(".pending-*"):
        if incomplete.is_dir():
            shutil.rmtree(incomplete)
    controller = {"step": 0, "best_step": 0, "best_loss": float(np.mean(parent_losses)),
                  "initial_loss": float(np.mean(parent_losses)), "history": [], "reduced_at": None,
                  "learning_rate": learning_rate, "started_utc": _utc()}
    best_tables = tables
    previous = _latest(checkpoint_root)
    if previous is not None:
        restored, controller = restore_state(previous, tables, optimizers)
        tables, optimizers, best_tables = restored["tables"], restored["optimizers"], restored["best"]
        learning_rate = controller["learning_rate"]
        # The durable checkpoint, not a log entry, determines resumable progress.
        metrics = run / "metrics.jsonl"
        if metrics.exists():
            lines = [line for line in metrics.read_text().splitlines()
                     if int(json.loads(line)["step"]) < int(controller["step"])]
            metrics.write_text("\n".join(lines) + ("\n" if lines else ""))
        controller["history"] = controller["history"][:int(controller["step"])]
    stop_requested = False

    def stop_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, stop_handler)
    import wandb
    wandb_run = wandb.init(
        project="functional-memory", entity="rafah-massachusetts-institute-of-technology",
        group="attention_scaling", name=run.name, id=run.name, resume="allow",
        tags=["coefficient_oracle", f"sequences-{count}", runtime["device_kind"]],
        config=recipe, dir=str(run), mode=recipe.get("train", {}).get("wandb_mode", "online"),
    )
    _write_json(run / "wandb_run.json", {"id": run.name, "url": wandb_run.url,
                                       "entity": "rafah-massachusetts-institute-of-technology", "project": "functional-memory"})
    attempts[-1]["resume_step"] = int(controller["step"])
    update_record(run, {"status": "running", "oracle_runtime": runtime, "acceptance": acceptance,
                        "started_utc": controller["started_utc"], "attempts": attempts,
                        "wandb": {"id": run.name, "url": wandb_run.url}})
    for failure in run.glob("numerical_failure_step_*.json"):
        failure.unlink()
    status = "paused"
    terminal = None
    step = int(controller["step"])
    checkpoint = previous
    with (run / "metrics.jsonl").open("a", buffering=1) as stream:
        while True:
            started = time.monotonic()
            results = [train_step(table, optimizer, params, tokens, learning_rate)
                       for table, optimizer, tokens in zip(tables, optimizers, token_chunks)]
            losses = np.concatenate([np.asarray(item[2]) for item in results])
            norms = np.concatenate([np.asarray(item[3]) for item in results])
            finite = all(bool(item[4]) for item in results) and np.all(np.isfinite(losses))
            if not finite:
                status = "numerical_failure"
                _write_json(run / f"numerical_failure_step_{step:08d}.json", {
                    "step": step, "status": status, "reason": "nonfinite coefficient loss, gradient, or update",
                    "last_valid_best_nll": controller["best_loss"], "time_utc": _utc(),
                })
                break
            loss = float(losses.mean())
            if loss <= controller["best_loss"]:
                controller.update(best_loss=loss, best_step=step)
                best_tables = tables
            controller["history"].append(controller["best_loss"])
            terminal = {"event": "train", "step": step, "nll": loss, "perplexity": math.exp(min(loss, 80)),
                        "best_nll": controller["best_loss"], "best_step": controller["best_step"],
                        "initial_nll": controller["initial_loss"], "nll_reduction": controller["initial_loss"] - loss,
                        "learning_rate": learning_rate, "sequence_count": count,
                        "first_sequence_fit_nll": float(losses[0]), "per_sequence_nll": losses.tolist(),
                        "fit_nll": loss, "fit_perplexity": math.exp(min(loss, 80)),
                        "per_sequence_nll_min": float(losses.min()), "per_sequence_nll_max": float(losses.max()),
                        "gradient_norm_min": float(norms.min()), "gradient_norm_max": float(norms.max()),
                        "seconds": time.monotonic() - started, "time_utc": _utc()}
            stream.write(json.dumps(terminal, allow_nan=False) + "\n")
            wandb_run.log({"train/nll": loss, "train/perplexity": terminal["perplexity"],
                           "train/best_nll": controller["best_loss"], "train/learning_rate": learning_rate,
                           "train/first_sequence_nll": float(losses[0]), "train/step_seconds": terminal["seconds"]}, step=step)
            action = plateau_action(controller["history"], step, controller["reduced_at"], oracle)
            if action == "reduce":
                learning_rate *= float(oracle["learning_rate_decay"])
                controller["reduced_at"] = step
                print(f"Plateau at {step}: reducing learning rate to {learning_rate}", flush=True)
                # Recompute this update with the reduced rate; retain the same gradient math.
                results = [train_step(table, optimizer, params, tokens, learning_rate)
                           for table, optimizer, tokens in zip(tables, optimizers, token_chunks)]
            controller.update(step=step, learning_rate=learning_rate)
            finished = action == "stop"
            interrupted = stop_requested or step >= int(oracle["max_updates"])
            if step % int(oracle["checkpoint_every"]) == 0 or finished or interrupted:
                checkpoint = save_state(checkpoint_root, step, tables, optimizers, best_tables, controller)
                _write_json(run / "metadata.json", {"status": "complete" if finished else "running",
                                                     "step": step, "oracle_runtime": runtime,
                                                     "best_step": controller["best_step"], "best_nll": controller["best_loss"]})
                update_record(run, {"status": "running", "step": step,
                                    "oracle_outcome": {"best_nll": controller["best_loss"], "best_step": controller["best_step"]}})
            if step % 10 == 0:
                print(f"step={step} nll={loss:.6f} best={controller['best_loss']:.6f} lr={learning_rate:g} seconds={terminal['seconds']:.3f}", flush=True)
            if finished or interrupted:
                status = "complete" if finished else "paused"
                break
            tables = [item[0] for item in results]
            optimizers = [item[1] for item in results]
            step += 1
    if state_sha256(params) != acceptance["frozen_parameter_sha256"]:
        raise RuntimeError("Frozen pretrained parameter tree changed")
    outcome = {"status": status, "step": step, "best_step": controller["best_step"],
               "initial_nll": controller["initial_loss"], "best_nll": controller["best_loss"],
               "fit_nll": terminal["nll"] if terminal else None,
               "fit_perplexity": terminal["perplexity"] if terminal else None,
               "best_fit_nll": controller["best_loss"],
               "nll_reduction": controller["initial_loss"] - controller["best_loss"],
               "learning_rate_reduced_at": controller["reduced_at"], "finished_utc": _utc(),
               "interpretation": "Fixed-input coefficient fitting, not held-out generalization.",
               "frozen_parameters_unchanged": True}
    if status == "paused":
        outcome["pause_reason"] = "preemption or interruption" if stop_requested else "Safety update budget exhausted; follow-up required."
    if status == "complete":
        # Place the coefficient-only save beside any resumable save at that step.
        # Completion is committed before pruning, so interruption is recoverable.
        checkpoint = checkpoint_root / str(controller["best_step"])
        final_path = checkpoint / "best"
        if final_path.exists():  # Incomplete finalization from an earlier attempt.
            shutil.rmtree(final_path)
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(final_path, {"coefficients": best_tables})
        _write_json(checkpoint / "coefficient_identity.json", {"step": controller["best_step"],
                    "layout": "microbatches of [example,layer,query,head,order]", "coordinates": "factorial(order)*c",
                    "parent_run_id": oracle["parent_run_id"], "parent_checkpoint": oracle["parent_checkpoint"]})
        _write_json(run / "completion.json", outcome)
    _write_json(run / "metadata.json", {**outcome, "oracle_runtime": runtime})
    wandb_run.summary.update({"terminal_nll": terminal["nll"] if terminal else None,
                             "best_nll": controller["best_loss"], "status": status})
    wandb_run.finish()
    if status == "complete":
        checkpoint = _retain_completed_best(run, outcome)
    updates = capture_artifacts(run, checkpoint_directory=checkpoint)
    update_record(run, {**updates, "status": status, "step": step, "oracle_outcome": outcome,
                        "retained_best_checkpoint_step": controller["best_step"]})
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    main()
