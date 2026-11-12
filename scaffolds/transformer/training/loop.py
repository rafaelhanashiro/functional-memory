import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import trange

from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.loader import save_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.attention.backends.selection import resolve_config_backend
from scaffolds.transformer.configs.base import OFM_SOURCE_COMMIT, validate_config
from scaffolds.transformer.data.loaders import make_eval_iterator, make_train_iterator
from scaffolds.transformer.data.packed import (
    LoadedPackedManifest,
    load_packed_manifest,
    manifest_fingerprint,
    resolve_packed_manifest,
)
from scaffolds.transformer.training.checkpointing import (
    VALIDATION_NLL_METRIC,
    cleanup_stale_checkpoint_temps,
    make_checkpoint_manager,
    restore_latest_checkpoint,
    save_checkpoint,
    state_sha256,
    validate_resume_progress,
)
from scaffolds.transformer.training import telemetry, tracking
from scaffolds.transformer.training.cadence import evaluation_due
from scaffolds.transformer.training.distributed import (
    BatchLayout,
    DataParallelFunctions,
    make_data_parallel_functions,
    put_eval_batch,
    put_replicated,
    put_train_batches,
    resolve_batch_layout,
    runtime_provenance,
    select_data_parallel_devices,
)
from scaffolds.transformer.training.diagnostics import (
    attention_diagnostics,
    attention_identity,
    fixed_diagnostic_sample,
    json_number,
)
from scaffolds.transformer.training.metrics import (
    METRICS_SCHEMA_VERSION,
    ValidationMetricsStore,
    write_json_atomic,
)
from scaffolds.transformer.training.coefficient_release import coefficient_release, mask_frozen_gradients
from scaffolds.transformer.training.optim import make_learning_rate_schedule
from scaffolds.transformer.training.numerics import checked_update_loss, requires_update_checks, update_numerics
from scaffolds.transformer.training.randomness import model_forward_rngs
from scaffolds.transformer.training.regularization import coefficient_penalty_sum
from scaffolds.transformer.training.state import TrainState, create_state


def dtype_from_name(name: str):
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"Unknown dtype: {name}")


def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    # Integer-label cross entropy avoids constructing a dense one-hot tensor
    # with shape [batch, sequence, vocab] at production vocabulary sizes.
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets
    ).mean()


def _penalty_arguments(coefficient_penalty: float, coefficient_penalty_denominator: bool = True) -> dict:
    """Forward the penalty settings only when the penalty acts; zero keeps the plain call signature."""
    if not coefficient_penalty:
        return {}
    return {"coefficient_penalty": coefficient_penalty, "coefficient_penalty_denominator": coefficient_penalty_denominator}


def loss_and_grads(state: TrainState, batch: jax.Array, rng: jax.Array, *, coefficient_penalty: float = 0.0,
                   coefficient_penalty_denominator: bool = True, telemetry_reductions: dict[str, str] | None = None):
    """The batch's NLL and the gradient of NLL + coefficient_penalty * coefficient penalty sum.

    The returned loss is the task NLL alone; the penalty enters only the gradient. With
    ``telemetry_reductions`` (a dict filled at trace time with each statistic's reduction),
    also return the forward pass's recorded activation statistics.
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    def loss_fn(params):
        model = nnx.merge(state.graphdef, params)
        statistics = None
        if telemetry_reductions is None:
            logits = model(
                inputs,
                deterministic=False,
                rngs=model_forward_rngs(rng),
            )
        else:
            with telemetry.collect() as collector:
                logits = model(
                    inputs,
                    deterministic=False,
                    rngs=model_forward_rngs(rng),
                )
            telemetry_reductions.update(collector.reductions)
            statistics = collector.values
        nll = cross_entropy_loss(logits, targets)
        if coefficient_penalty:
            penalty = coefficient_penalty_sum(params, model.config, include_denominator=coefficient_penalty_denominator)
            return nll + coefficient_penalty * penalty, (nll, statistics)
        return nll, (nll, statistics)

    # ``state.params`` is a pure NNX State pytree, so the base JAX transform also
    # traces inside the accumulation scan (an NNX transform rejects the outer trace's Params).
    (_, (nll, statistics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if telemetry_reductions is None:
        return nll, grads
    return nll, grads, statistics


def _telemetry_options(collect_telemetry: bool) -> tuple[dict[str, str] | None, dict]:
    """A step's trace-time reduction record and the keyword that enables recording."""
    reductions = {} if collect_telemetry else None
    return reductions, ({"telemetry_reductions": reductions} if collect_telemetry else {})


@partial(nnx.jit, static_argnames=("check_all_gradients", "coefficient_penalty", "coefficient_penalty_denominator",
                                   "collect_telemetry"))
def train_step(state: TrainState, batch: jax.Array, rng: jax.Array, *, check_all_gradients=False,
               coefficient_penalty: float = 0.0, coefficient_penalty_denominator: bool = True,
               collect_telemetry: bool = False):
    """One update. With ``collect_telemetry``, also return, last, the update's named statistics."""
    reductions, recording = _telemetry_options(collect_telemetry)
    loss, grads, *statistics = loss_and_grads(state, batch, rng, **recording,
                                              **_penalty_arguments(coefficient_penalty, coefficient_penalty_denominator))
    grads = mask_frozen_gradients(state, grads)
    updated = state.apply_gradients(grads=grads)
    checked_loss = checked_update_loss(state.params, grads, updated.params, loss,
                                      check_all_gradients=check_all_gradients)
    outputs = (updated, checked_loss)
    if check_all_gradients:
        outputs += (update_numerics(grads, updated.params, loss),)
    if collect_telemetry:
        outputs += (telemetry.single_device_statistics(statistics[0], reductions, state.params, grads, updated.params),)
    return outputs


def microbatch_loss_and_grads(state: TrainState, batch: jax.Array, rng: jax.Array, *, coefficient_penalty: float = 0.0,
                              coefficient_penalty_denominator: bool = True,
                              telemetry_reductions: dict[str, str] | None = None):
    """One microbatch of ``accumulated_train_step``; traced inside its scan, so not jitted itself."""
    recording = {"telemetry_reductions": telemetry_reductions} if telemetry_reductions is not None else {}
    return loss_and_grads(state, batch, rng, **recording,
                          **_penalty_arguments(coefficient_penalty, coefficient_penalty_denominator))


@nnx.jit
def apply_gradients_step(state: TrainState, grads):
    """Apply one pre-accumulated update, including any frozen-parameter mask."""
    return state.apply_gradients(grads=mask_frozen_gradients(state, grads))


@partial(nnx.jit, static_argnames=("check_all_gradients", "coefficient_penalty", "coefficient_penalty_denominator",
                                   "collect_telemetry"))
def accumulated_train_step(state: TrainState, batches: jax.Array, rngs: jax.Array, *, check_all_gradients=False,
                           coefficient_penalty: float = 0.0, coefficient_penalty_denominator: bool = True,
                           collect_telemetry: bool = False):
    """One compiled update over ``batches`` (``[microbatches, sequences, tokens]``) and their ``rngs``.

    A scan accumulates the microbatch gradients on the device, so the host dispatches one
    program per update. Dispatching one program per microbatch, plus one addition per
    parameter for every partial sum, left a single GPU idle for about 40% of each update.
    With ``collect_telemetry``, also return, last, the update's named statistics.
    """
    if len(batches) == 0 or len(batches) != len(rngs):
        raise ValueError("Accumulation requires nonempty batches and one RNG per microbatch")
    penalty = _penalty_arguments(coefficient_penalty, coefficient_penalty_denominator)
    reductions, recording = _telemetry_options(collect_telemetry)

    def accumulate(carry, values):
        loss_sum, grad_sum = carry
        batch, rng = values
        loss, grads, *statistics = microbatch_loss_and_grads(state, batch, rng, **penalty, **recording)
        return ((loss_sum + loss.astype(loss_sum.dtype), jax.tree_util.tree_map(jnp.add, grad_sum, grads)),
                statistics[0] if collect_telemetry else None)

    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    (loss_sum, grad_sum), stacked_statistics = jax.lax.scan(
        accumulate, (jnp.zeros((), dtype=jnp.float32), zero_grads), (batches, rngs))
    scale = jnp.asarray(1.0 / batches.shape[0], dtype=loss_sum.dtype)
    grads = mask_frozen_gradients(state, jax.tree_util.tree_map(lambda grad: grad * scale, grad_sum))
    loss = loss_sum * scale
    updated = state.apply_gradients(grads=grads)
    checked_loss = checked_update_loss(state.params, grads, updated.params, loss,
                                      check_all_gradients=check_all_gradients)
    outputs = (updated, checked_loss)
    if check_all_gradients:
        outputs += (update_numerics(grads, updated.params, loss),)
    if collect_telemetry:
        outputs += (telemetry.single_device_statistics(stacked_statistics, reductions, state.params, grads,
                                                       updated.params),)
    return outputs


@nnx.jit
def eval_step(
    state: TrainState,
    batch: jax.Array,
    rng: jax.Array | None = None,
):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    if rng is None:
        rng = jax.random.PRNGKey(0)
    model = nnx.merge(state.graphdef, state.params)
    logits = model(
        inputs,
        deterministic=True,
        rngs=model_forward_rngs(rng),
    )
    return cross_entropy_loss(logits, targets)


def _serializable_config(config: Config) -> dict:
    return config_to_dict(config)


def ensure_run_config(config: Config, *, resume: bool) -> None:
    recipe_path = config.train.workdir / "config.yaml"
    path = config.train.workdir / "config.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    canonical = None
    if existing is not None and not resume:
        raise ValueError(f"Run workdir is already initialized; use resume or a new workdir: {path}")
    if recipe_path.exists():
        record_path = config.train.workdir / "record.json"
        if record_path.exists():
            recipe = json.loads(record_path.read_text()).get("config", {})
            expected_hash = recipe.get("sha256") if isinstance(recipe, dict) else None
            if expected_hash is not None and expected_hash != hashlib.sha256(recipe_path.read_bytes()).hexdigest():
                raise ValueError(f"Run configuration bytes changed after allocation: {recipe_path}")
        canonical = load_config(recipe_path)
        resolve_workdir(canonical)
        if canonical.train.source_commit == "auto":
            canonical.train.source_commit = config.train.source_commit
    elif existing is None:
        # Direct smoke/CLI runs and scratch runs inside a run directory (acceptance
        # pilots) may allocate locally. Production launchers require the run's own
        # complete input recipe to be committed before any GPU allocation.
        run_root = tracking.experiment_paths(config.train.workdir)[1]
        if (os.environ.get("SHARED_TRANSFORMER_SOURCE_COMMIT") and run_root is not None
                and run_root.resolve() == Path(config.train.workdir).resolve()):
            raise ValueError(f"Production run requires a preallocated, committed configuration: {recipe_path}")
        save_config(config, recipe_path)
    if existing is not None and existing.get("optim", {}).get("weight_decay_policy") != config.optim.weight_decay_policy:
        raise ValueError(
            f"Run has a different or unrecorded weight-decay policy: {path}. "
            "Resume with its original source checkout, or start a new run."
        )
    request = config.model.attention_backend
    if existing is not None and request == "auto":
        pinned = existing.get("model", {}).get("attention_backend")
        if pinned in (None, "auto"):
            raise ValueError(f"Run has no recorded concrete attention backend; cannot safely resume: {path}")
        config.model.attention_backend = pinned
    metadata_path = config.train.workdir / "metadata.json"
    previous_request = None
    if existing is not None and metadata_path.exists():
        previous_request = json.loads(metadata_path.read_text()).get("attention_backend_requested")
    if previous_request is not None:
        # A resolved config is a valid resume input. Preserve whether this run
        # originally requested auto selection instead of relabeling provenance.
        config._attention_backend_request = previous_request
    elif not hasattr(config, "_attention_backend_request"):
        config._attention_backend_request = request
    if existing is not None:
        config._attention_backend_resume_request = request
    resolve_config_backend(config)
    current = _serializable_config(config)
    if canonical is not None:
        if canonical.model.attention_backend == "auto":
            canonical.model.attention_backend = config.model.attention_backend
        if not config_dicts_equal(_serializable_config(canonical), current):
            raise ValueError(f"Training configuration does not match the allocated run: {recipe_path}")
    if existing is not None:
        # Legacy fixed-kernel runs predate these fields. Compare their defaults
        # semantically without rewriting their persisted configuration bytes.
        if not config_dicts_equal(existing, current):
            raise ValueError(
                f"Training configuration does not match the initialized run: {path}"
            )
        return
    write_json_atomic(path, current)


def resolve_workdir(config: Config) -> None:
    workdir = config.train.workdir.expanduser()
    if not workdir.is_absolute():
        workdir = Path.cwd() / workdir
    config.train.workdir = workdir.resolve()
    if config.train.checkpoint_dir is not None:
        config.train.checkpoint_dir = Path(config.train.checkpoint_dir).expanduser().resolve()


def resolve_gradient_accumulation(config: Config) -> tuple[int, int]:
    layout = resolve_batch_layout(config)
    return layout.global_micro_batch_size, layout.grad_accum_steps


def resolve_packed_data_cursor(
    config: Config,
    start_step: int,
    *,
    validated_train_manifest: LoadedPackedManifest | None = None,
    validated_eval_manifest: LoadedPackedManifest | None = None,
) -> int:
    if config.data.source != "packed":
        return start_step * config.data.batch_size
    if not config.data.train_manifest:
        raise ValueError("data.train_manifest is required for packed data")

    train_manifest_path, train_manifest = resolve_packed_manifest(
        config.data.train_manifest,
        validated_train_manifest,
    )
    if train_manifest["sequence_length"] != config.data.sequence_length:
        raise ValueError(
            f"Configured sequence length {config.data.sequence_length} does not match packed-data "
            f"length {train_manifest['sequence_length']}"
        )
    required_sequences = config.optim.total_steps * config.data.batch_size
    if train_manifest["num_sequences"] < required_sequences:
        raise ValueError(
            f"Packed training data has {train_manifest['num_sequences']} sequences, but "
            f"{required_sequences} are required"
        )
    if config.train.eval_every > 0:
        if not config.data.eval_manifest:
            raise ValueError(
                "data.eval_manifest is required when packed-data evaluation is enabled"
            )
        _, eval_manifest = resolve_packed_manifest(
            config.data.eval_manifest,
            validated_eval_manifest,
        )
        if eval_manifest["sequence_length"] != config.data.sequence_length:
            raise ValueError(
                f"Configured sequence length {config.data.sequence_length} does not match packed "
                f"evaluation length {eval_manifest['sequence_length']}"
            )
        required_eval_sequences = config.train.eval_batches * config.data.batch_size
        if eval_manifest["num_sequences"] < required_eval_sequences:
            raise ValueError(
                f"Packed evaluation data has {eval_manifest['num_sequences']} sequences, but "
                f"{required_eval_sequences} are required"
            )

    identity = {
        "format": "ofm-packed-run-data",
        "version": 2,
        "train_manifest_sha256": manifest_fingerprint(train_manifest_path),
        "eval_manifest_sha256": (
            manifest_fingerprint(config.data.eval_manifest) if config.data.eval_manifest else None
        ),
        "batch_size": config.data.batch_size,
        "sequence_length": config.data.sequence_length,
    }
    identity_path = config.train.workdir / "data_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text())
        if existing != identity:
            raise ValueError(
                f"Packed-data identity does not match the existing run: {identity_path}"
            )
    else:
        if config.continuation.parent_checkpoint and start_step == config.continuation.parent_step:
            parent_identity_path = Path(config.continuation.parent_config).parent / "data_identity.json"
            parent_identity_bytes = parent_identity_path.read_bytes()
            if hashlib.sha256(parent_identity_bytes).hexdigest() != config.continuation.parent_data_identity_sha256:
                raise ValueError("Continuation parent data identity hash changed")
            if json.loads(parent_identity_bytes) != identity:
                raise ValueError("Continuation dataset differs from its parent")
        elif start_step != 0:
            raise ValueError(
                "Cannot resume packed data without the current scaffold's data identity"
            )
        write_json_atomic(identity_path, identity)

    return start_step * config.data.batch_size


def evaluate(
    state: TrainState,
    eval_iter,
    num_batches: int,
    *,
    data_parallel: DataParallelFunctions | None = None,
    eval_seed: int = 0,
    micro_batch_size: int | None = None,
    batch_seconds: list[float] | None = None,
) -> float:
    """Mean token NLL, with bounded evaluation memory and a fixed data prefix."""
    if num_batches < 1:
        raise ValueError("Evaluation requires at least one batch")
    nll_sum, token_count = 0.0, 0
    eval_rng = jax.random.PRNGKey(eval_seed)
    for batch_index in range(num_batches):
        batch_start = time.perf_counter() if batch_seconds is not None else None
        batch = next(eval_iter)
        micro_size = micro_batch_size or len(batch)
        for offset in range(0, len(batch), micro_size):
            micro_batch = batch[offset:offset + micro_size]
            batch_rng = jax.random.fold_in(eval_rng, batch_index)
            if data_parallel is None:
                loss = eval_step(state, jnp.asarray(micro_batch, dtype=jnp.int32), batch_rng)
            else:
                sharded = put_eval_batch(micro_batch, data_parallel.eval_batch_sharding)
                loss = data_parallel.eval_step(state, sharded, batch_rng)
            count = micro_batch.shape[0] * (micro_batch.shape[1] - 1)
            nll_sum += float(loss) * count
            token_count += count
        if batch_seconds is not None:
            # float(loss) synchronizes every microbatch. Include the global
            # batch's data fetch, transfers, and all bounded-memory forwards.
            batch_seconds.append(time.perf_counter() - batch_start)
    return nll_sum / token_count


def param_count(state: TrainState) -> int:
    return int(sum(leaf.size for leaf in jax.tree.leaves(state.params)))


def _best_validation_record(record: dict) -> dict:
    return {key: record[key] for key in (
        "step", "training_tokens", "validation_nll", "validation_perplexity"
    )} | {"selection_rule": "lowest_validation_nll"}


def _recover_pending_validation(*, metrics_store, latest_checkpoint_step, workdir, wandb_run):
    record = metrics_store.pending(latest_checkpoint_step)
    if record is not None:
        if record["is_best"]:
            write_json_atomic(workdir / "best_validation.json", _best_validation_record(record))
        metrics_store.commit(
            record, latest_checkpoint_step=latest_checkpoint_step,
            mirror=lambda event: tracking.log_validation(wandb_run, event),
        )


def _write_status(config: Config, **updates) -> None:
    path = config.train.workdir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    runtime = runtime_provenance()
    history = metadata.setdefault("runtime_history", [])
    if not history or history[-1]["runtime"] != runtime:
        history.append({"timestamp_utc": datetime.now(UTC).isoformat(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "runtime": runtime})
    metadata.update({
        "execution_runtime": runtime,
        **attention_identity(config),
        "qk_head_dim": config.model.qk_head_dim,
        "value_head_dim": config.model.value_head_dim,
        "seed": config.train.seed,
        "source_commit": config.train.source_commit or None,
        "scaffold_ofm_source_commit": OFM_SOURCE_COMMIT,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID") or metadata.get("slurm_job_id"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", str(metadata.get("slurm_restart_count", 0)))),
        "updated_utc": datetime.now(UTC).isoformat(),
        "target_steps": config.optim.total_steps,
        "target_tokens": config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1),
        **updates,
    })
    retention_path = config.train.workdir / "retention.json"
    if retention_path.exists():
        retention = json.loads(retention_path.read_text())
        metadata["checkpoint_retention"] = retention
        metadata["retained_terminal_checkpoint_step"] = (
            retention["best_checkpoint_step"]
            if retention["best_checkpoint_step"] == retention.get("terminal_step", config.optim.total_steps)
            else None
        )
    identity_path = config.train.workdir / "data_identity.json"
    if identity_path.exists():
        metadata["data_identity"] = json.loads(identity_path.read_text())
    launch_path = config.train.workdir / "launch.json"
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        if launch.get("log_path") is not None:
            metadata["log_path"] = launch["log_path"]
    write_json_atomic(path, metadata)
    try:
        _update_run_record(config, metadata)
    except Exception as error:
        if metadata.get("status") != "failed":
            raise
        # Bookkeeping must never replace the failure being recorded; metadata.json
        # above already holds it, and the caller re-raises the original error.
        print(f"run_record_update_failed={type(error).__name__}: {error}", flush=True)


def _update_run_record(config: Config, metadata: dict) -> None:
    """Summarize execution in the canonical record; keep raw files for resume checks."""
    from scaffolds.records import capture_artifacts, update_record
    from scaffolds.transformer.training.checkpointing import checkpoint_directory

    workdir = config.train.workdir
    resolved = _serializable_config(config)
    raw_path = workdir / "metadata.json"
    status = metadata.get("status", "initialized")
    if status == "failed" and metadata.get("numerical_failure_path"):
        status = "numerical_failure"
    changes = {
        "status": status,
        "source_commit": config.train.source_commit or None,
        "seed": config.train.seed,
        "updated_utc": metadata["updated_utc"],
        "execution": {
            "resolved_config": resolved,
            "resolved_config_sha256": hashlib.sha256(
                json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "raw_metadata": {"path": "metadata.json", "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
            "runtime": metadata["execution_runtime"],
            "runtime_history": metadata["runtime_history"],
            "attention_backend": config.model.attention_backend,
        },
    }
    raw_config = workdir / "config.json"
    if raw_config.exists():  # Written by ensure_run_config before training; absent for bare status writes.
        changes["execution"]["raw_config"] = {"path": "config.json",
                                              "sha256": hashlib.sha256(raw_config.read_bytes()).hexdigest()}
    for key in ("step", "training_tokens", "wandb_run_id", "retained_best_checkpoint_step",
                "retained_terminal_checkpoint_step", "data_identity", "error", "numerical_failure_path"):
        if key in metadata:
            changes[key] = metadata[key]
    if os.environ.get("TRANSFORMER_CLUSTER"):
        changes["home_cluster"] = os.environ["TRANSFORMER_CLUSTER"]
    changes["log_path"] = str(workdir / "training.log")
    if status in {"complete", "failed", "numerical_failure"}:
        selected_step = metadata.get("retained_best_checkpoint_step")
        selected = checkpoint_directory(config) / str(selected_step) if selected_step is not None else None
        artifacts = capture_artifacts(workdir, checkpoint_directory=selected)
        if "selected_checkpoint" in artifacts:
            artifacts["selected_checkpoint"]["step"] = selected_step
        changes.update(artifacts)
    update_record(workdir, changes)


def _fail_numerics(config, state, step, message, sample, provenance, *, diagnostics=None, loss=None,
                   numerical_metrics=None, telemetry_record=None):
    """Persist the offending state diagnostics before aborting without a repair."""
    failure = {
        "format": "shared-transformer-numerical-failure", "version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(), "step": step,
        "training_tokens": step * config.data.batch_size * (config.data.sequence_length - 1),
        **attention_identity(config), "error": message,
        "diagnostics_state": "after_attempted_training_update",
    }
    if loss is not None:
        failure["loss"] = json_number(loss)
        failure["checked_quantities"] = ["forward_loss", "global_gradient_norm", "updated_parameters"]
        failure["loss_note"] = "A NaN loss scalar also signals an invalid gradient norm or parameter update."
    if numerical_metrics is not None:
        failure["update_numerics"] = dict(zip(
            ("forward_loss", "global_gradient_norm", "updated_parameter_norm"),
            (json_number(value) for value in numerical_metrics), strict=True,
        ))
    if telemetry_record is not None:
        failure["training_telemetry"] = {
            name: {key: json_number(value) for key, value in values.items()} if values is not None else None
            for name, values in telemetry_record.items()
        }
    try:
        failure["attention_diagnostics"] = diagnostics if diagnostics is not None else attention_diagnostics(
            state, config, sample, provenance,
        )
    except Exception as error:
        failure["diagnostics_error"] = f"{type(error).__name__}: {error}"
        failure["diagnostic_sample"] = provenance
    failure_path = config.train.workdir / f"numerical_failure_step_{step:08d}.json"
    write_json_atomic(failure_path, failure)
    _write_status(config, status="failed", step=step, training_tokens=failure["training_tokens"],
                  error=message, numerical_failure_path=str(failure_path),
                  attention_diagnostics=failure.get("attention_diagnostics"))
    raise FloatingPointError(message)


def _perplexity(config, state, step, nll, sample, provenance, *, diagnostics=None):
    """Classify metric overflow as a numerical failure of this variant."""
    try:
        return math.exp(nll)
    except OverflowError:
        _fail_numerics(config, state, step,
                       f"Perplexity overflow at step {step} for finite NLL {nll}",
                       sample, provenance, diagnostics=diagnostics, loss=nll)


def run_training(config: Config, *, resume: bool = True, stop_after_step: int | None = None,
                 audit_checkpoint: bool = False, profile_timing: bool = False) -> dict:
    """Train with an exact packed-data cursor and transactional local validation metrics."""
    validate_config(config)
    resolve_workdir(config)
    layout = resolve_batch_layout(config)
    devices = select_data_parallel_devices(layout)
    end_step = config.optim.total_steps if stop_after_step is None else stop_after_step
    if not 1 <= end_step <= config.optim.total_steps:
        raise ValueError("stop_after_step must be between 1 and optim.total_steps")
    if config.train.eval_every < 1 or config.train.eval_batches < 1:
        raise ValueError("Training requires validation, including at the terminal step")
    config.train.workdir.mkdir(parents=True, exist_ok=True)
    ensure_run_config(config, resume=resume)
    for path in cleanup_stale_checkpoint_temps(config):
        print(f"removed_stale_checkpoint_tmp={path}", flush=True)
    manager = make_checkpoint_manager(config)
    try:
        # Check before the run can overwrite progress metadata or reuse existing
        # metric event IDs. A rejected resume leaves the scientific record intact.
        validate_resume_progress(config, manager.latest_step())
        try:
            return _run_training(config, manager, layout, devices, resume=resume, end_step=end_step,
                                 audit_checkpoint=audit_checkpoint, profile_timing=profile_timing)
        except BaseException as error:
            _write_status(config, status="failed", error=f"{type(error).__name__}: {error}")
            raise
    finally:
        manager.wait_until_finished()
        manager.close()


def _update_audit(step, batches, rng):
    return {"step": step, "batch_sha256": state_sha256(batches), "rng_sha256": state_sha256(rng)}


def _run_training(config, manager, layout, devices, *, resume, end_step, audit_checkpoint=False,
                  profile_timing=False):
    runtime_start = time.perf_counter()
    timing = {"train_step_seconds": [], "evaluation_seconds": [], "checkpoint_seconds": []}
    if profile_timing:
        timing.update(setup_seconds=0.0, diagnostic_warmup_seconds=0.0, evaluation_warmup_seconds=0.0,
                      diagnostic_seconds=[], evaluation_batch_seconds=[])
    check_all_gradients = requires_update_checks(config.model)
    completion_path = config.train.workdir / "completion.json"
    latest_before = manager.latest_step()
    validated_train_manifest = None
    validated_eval_manifest = None
    if config.data.source == "packed":
        validated_train_manifest = load_packed_manifest(config.data.train_manifest)
        validated_eval_manifest = load_packed_manifest(config.data.eval_manifest)
    # Check data identity on all entries, including an already completed run.
    start_sequence = resolve_packed_data_cursor(
        config, latest_before if latest_before is not None else config.continuation.parent_step,
        validated_train_manifest=validated_train_manifest,
        validated_eval_manifest=validated_eval_manifest,
    )
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        retained_best_complete = False
        retention_path = config.train.workdir / "retention.json"
        if retention_path.exists():
            retention = json.loads(retention_path.read_text())
            retained_best_complete = (
                retention.get("format") in (
                    "shared-transformer-checkpoint-retention", "odd-taylor-checkpoint-retention",
                    "learned-quadratic-checkpoint-retention", "key-dimension-taylor-checkpoint-retention",
                )
                and retention.get("completion_sha256") == hashlib.sha256(completion_path.read_bytes()).hexdigest()
                and retention.get("best_checkpoint_step") == completion["retained_best_checkpoint_step"] == latest_before
            )
        if (not resume or completion.get("format") not in (
                    "shared-transformer-training-completion", "polynomial-attention-training-completion",
                    "key-dimension-training-completion")
                or completion.get("final_step") != config.optim.total_steps
                or (latest_before != config.optim.total_steps and not retained_best_complete)):
            raise ValueError(f"Invalid or incompatible completed run: {completion_path}")
        _write_status(config, status="complete", step=config.optim.total_steps, **{
            key: completion[key] for key in (
                "training_tokens", "validation_nll", "validation_perplexity",
                "retained_terminal_checkpoint_step", "retained_best_checkpoint_step",
                "attention_diagnostics",
            ) if key in completion
        })
        print(f"run_already_complete_step={config.optim.total_steps}", flush=True)
        return completion
    rng = jax.random.PRNGKey(config.train.seed)
    state = create_state(config, rng)
    restored_step = None
    if resume:
        state, restored_step = restore_latest_checkpoint(manager, state)
    fresh_continuation = restored_step is None and bool(config.continuation.parent_checkpoint)
    if fresh_continuation:
        from scaffolds.transformer.training.continuation import initialize_continuation_state
        state, continuation_audit = initialize_continuation_state(config, state)
        write_json_atomic(config.train.workdir / "continuation.json", continuation_audit)
    start_step = int(state.step)
    if end_step < start_step:
        raise ValueError("stop_after_step precedes restored training step")
    if restored_step is not None and start_step != restored_step:
        raise ValueError("Checkpoint state step differs from checkpoint directory")
    if start_sequence != start_step * config.data.batch_size:
        raise ValueError("Checkpoint step and data cursor disagree")
    audit = {"start_state_sha256": state_sha256(state)} if audit_checkpoint else None
    parameter_count = param_count(state)
    tokens_per_step = config.data.batch_size * (config.data.sequence_length - 1)
    # A numerical failure ends one execution. Training the run again supersedes it, so the
    # run keeps a single outcome: the earlier failure record is removed (the log keeps it).
    for superseded in sorted(config.train.workdir.glob("numerical_failure_step_*.json")):
        superseded.unlink()
    _write_status(config, status="running", step=start_step, training_tokens=start_step * tokens_per_step,
                  parameter_count=parameter_count, error=None, numerical_failure_path=None,
                  devices=[str(d) for d in devices],
                  rng_state={"seed": config.train.seed, "completed_updates": start_step},
                  next_train_sequence=start_sequence)
    print(f"param_count={parameter_count} restored_step={restored_step} packed_data_start_sequence={start_sequence}", flush=True)
    print(f"data_parallel_devices={layout.device_count} global_batch_size={layout.global_batch_size} "
          f"global_micro_batch_size={layout.global_micro_batch_size} grad_accum_steps={layout.grad_accum_steps}", flush=True)
    release = coefficient_release(config)
    if release is not None:
        print(f"coefficient_release_schedule={release.schedule} "
              f"release_updates={','.join(map(str, release.counts))} "
              f"trainable_orders={release.trainable_orders(start_step)}", flush=True)
    penalty_weight = config.optim.coefficient_penalty
    penalty_options = _penalty_arguments(penalty_weight, config.optim.coefficient_penalty_denominator)
    penalty_sum = (jax.jit(partial(coefficient_penalty_sum, model=config.model,
                                   include_denominator=config.optim.coefficient_penalty_denominator))
                   if penalty_weight else None)
    penalty_value = None
    data_parallel = None
    # Telemetry is standard on one GPU and several; recording cannot cross remat.
    collect_telemetry = not config.model.remat
    telemetry_options = {"collect_telemetry": True} if collect_telemetry else {}
    telemetry_keys: list[str] = []
    if layout.device_count > 1:
        data_parallel = make_data_parallel_functions(state, devices, check_all_gradients=check_all_gradients,
                                                    collect_telemetry=collect_telemetry, **penalty_options)
        state = put_replicated(state, data_parallel.replicated_sharding)
        telemetry_keys = data_parallel.telemetry_keys
    train_iter = make_train_iterator(
        replace(config.data, batch_size=layout.global_micro_batch_size),
        start_sequence=start_sequence, validated_manifest=validated_train_manifest,
    )
    diagnostic_sample, diagnostic_provenance = fixed_diagnostic_sample(
        config, validated_manifest=validated_eval_manifest,
    )
    _write_status(config, diagnostic_sample=diagnostic_provenance)
    wandb_run, wandb_run_id = tracking.init_run(
        config, tracking.runtime_settings(layout.device_count, jax.devices()[0].device_kind))
    diagnostics_warmed = evaluation_warmed = False

    def collect_diagnostics():
        nonlocal diagnostics_warmed
        if not profile_timing:
            return attention_diagnostics(state, config, diagnostic_sample, diagnostic_provenance)
        if not diagnostics_warmed:
            warmup_start = time.perf_counter()
            warmup = attention_diagnostics(state, config, diagnostic_sample, diagnostic_provenance)
            timing["diagnostic_warmup_seconds"] = time.perf_counter() - warmup_start
            diagnostics_warmed = True
            # The first call includes compilation and ordinary work. An invalid
            # result must reach the existing failure path, never be discarded.
            if not warmup["valid"]:
                return warmup
        diagnostic_start = time.perf_counter()
        report = attention_diagnostics(state, config, diagnostic_sample, diagnostic_provenance)
        timing["diagnostic_seconds"].append(time.perf_counter() - diagnostic_start)
        return report

    def evaluate_prefix(num_batches, *, batch_seconds=None):
        return evaluate(
            state, make_eval_iterator(config.data, validated_manifest=validated_eval_manifest), num_batches,
            data_parallel=data_parallel, eval_seed=config.train.seed,
            micro_batch_size=layout.global_micro_batch_size,
            **({"batch_seconds": batch_seconds} if batch_seconds is not None else {}),
        )

    def collect_evaluation():
        nonlocal evaluation_warmed
        if not profile_timing:
            return evaluate_prefix(config.train.eval_batches)
        if not evaluation_warmed:
            warmup_start = time.perf_counter()
            warmup_loss = evaluate_prefix(1)
            timing["evaluation_warmup_seconds"] = time.perf_counter() - warmup_start
            evaluation_warmed = True
            if not math.isfinite(warmup_loss):
                return warmup_loss
        samples = timing["evaluation_batch_seconds"]
        eval_loss = evaluate_prefix(config.train.eval_batches, batch_seconds=samples)
        if not math.isfinite(eval_loss):
            return eval_loss
        # A pilot can validate on one global batch. Replay its fixed prefix to
        # obtain at least three warm samples without consuming any training data
        # or changing the validation average, sequence indices, or RNG seeds.
        for _ in range(max(0, 3 - len(samples))):
            profile_loss = evaluate_prefix(1, batch_seconds=samples)
            if not math.isfinite(profile_loss):
                return profile_loss
        return eval_loss

    try:
        metrics_store = ValidationMetricsStore(config.train.workdir)
        _recover_pending_validation(metrics_store=metrics_store, latest_checkpoint_step=manager.latest_step(),
                                    workdir=config.train.workdir, wandb_run=wandb_run)
        best_step = manager.best_step()
        best_metrics = manager.metrics(best_step) if best_step is not None else None
        best_nll = float(best_metrics[VALIDATION_NLL_METRIC]) if best_metrics else math.inf
        schedule = make_learning_rate_schedule(config)
        window_time = time.perf_counter()
        window_step = start_step
        train_losses = []
        gradient_norms = []
        telemetry_window = telemetry.TelemetryWindow()
        telemetry_values = previous_telemetry = None
        attempt_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.environ.get('SLURM_RESTART_COUNT', '0')}:{uuid.uuid4().hex[:8]}"
        if profile_timing:
            jax.block_until_ready(state)
            timing["setup_seconds"] = time.perf_counter() - runtime_start
        if fresh_continuation:
            # Measure the child itself before its first update. Persist the baseline
            # transactionally so it can remain the validation-best checkpoint.
            baseline_start = time.perf_counter()
            diagnostics = collect_diagnostics()
            if not diagnostics["valid"]:
                _fail_numerics(config, state, start_step, "Invalid continuation baseline diagnostics",
                               diagnostic_sample, diagnostic_provenance, diagnostics=diagnostics)
            eval_loss = collect_evaluation()
            if not math.isfinite(eval_loss):
                _fail_numerics(config, state, start_step, "Nonfinite continuation baseline validation loss",
                               diagnostic_sample, diagnostic_provenance, diagnostics=diagnostics, loss=eval_loss)
            eval_ppl = _perplexity(config, state, start_step, eval_loss, diagnostic_sample,
                                   diagnostic_provenance, diagnostics=diagnostics)
            record = {
                "schema_version": METRICS_SCHEMA_VERSION, "event": "validation",
                "event_id": f"validation:{start_step}", "attempt_id": attempt_id,
                "timestamp_utc": datetime.now(UTC).isoformat(), "step": start_step,
                "checkpoint_step": start_step, "continuation_updates": 0,
                "training_tokens": start_step * tokens_per_step,
                "training_window_start_step": start_step, "train_nll_sample_count": 0,
                "validation_nll": eval_loss, "validation_perplexity": eval_ppl,
                "validation_batches": config.train.eval_batches,
                "validation_sequences": config.train.eval_batches * config.data.batch_size,
                "validation_tokens": config.train.eval_batches * tokens_per_step,
                "learning_rate": float(schedule(start_step)), "parameter_count": parameter_count,
                "qk_head_dim": config.model.qk_head_dim, "value_head_dim": config.model.value_head_dim,
                **attention_identity(config), "attention_diagnostics": diagnostics,
                "source_commit": config.train.source_commit or None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
                "wandb_run_id": wandb_run_id, "is_best": True,
            }
            timing["evaluation_seconds"].append(time.perf_counter() - baseline_start)
            checkpoint_start = time.perf_counter() if profile_timing else None
            metrics_store.stage(record)
            if not save_checkpoint(manager, start_step, state, metrics={
                "validation_nll": eval_loss, "validation_perplexity": eval_ppl,
                "has_validation": True, "training_tokens": start_step * tokens_per_step,
            }):
                raise RuntimeError("Failed to save continuation baseline")
            manager.wait_until_finished()
            if profile_timing:
                timing["checkpoint_seconds"].append(time.perf_counter() - checkpoint_start)
            best_nll = eval_loss
            write_json_atomic(config.train.workdir / "best_validation.json", _best_validation_record(record))
            metrics_store.commit(record, latest_checkpoint_step=manager.latest_step(),
                                 mirror=lambda event: tracking.log_validation(wandb_run, event))
            print(f"continuation_baseline step={start_step} val_nll={eval_loss:.6f} val_ppl={eval_ppl:.4f}", flush=True)
            window_time = time.perf_counter()
        for step in trange(start_step + 1, end_step + 1, desc="train"):
            if release is not None and (released := release.released_at(step - 1)):
                # This step's update, optimizer count step - 1, is the first to train these orders.
                print(f"coefficient_release update={step - 1} step={step} "
                      f"orders={','.join(map(str, released))}", flush=True)
                tracking.log_coefficient_release(wandb_run, step, release.trainable_orders(step - 1))
            train_step_start = time.perf_counter()
            step_rng = jax.random.fold_in(rng, step)
            if data_parallel is not None:
                batches = np.stack([next(train_iter) for _ in range(layout.grad_accum_steps)])
                if audit is not None and step == start_step + 1:
                    audit["first_update"] = _update_audit(step, batches, step_rng)
                step_result = data_parallel.train_step(state, put_train_batches(batches, data_parallel.train_batch_sharding), step_rng)
            elif layout.grad_accum_steps == 1:
                batch = jnp.asarray(next(train_iter))
                if audit is not None and step == start_step + 1:
                    audit["first_update"] = _update_audit(step, batch[None], step_rng)
                step_result = (train_step(state, batch, step_rng, check_all_gradients=True,
                                          **penalty_options, **telemetry_options) if check_all_gradients
                               else train_step(state, batch, step_rng, **penalty_options, **telemetry_options))
            else:
                batches = jnp.asarray(np.stack([next(train_iter) for _ in range(layout.grad_accum_steps)]))
                if audit is not None and step == start_step + 1:
                    audit["first_update"] = _update_audit(step, batches, step_rng)
                micro_rngs = jax.random.split(step_rng, layout.grad_accum_steps)
                step_result = (accumulated_train_step(state, batches, micro_rngs, check_all_gradients=True,
                                                      **penalty_options, **telemetry_options)
                               if check_all_gradients
                               else accumulated_train_step(state, batches, micro_rngs, **penalty_options,
                                                           **telemetry_options))
            state, loss, *numerical_metrics = step_result
            telemetry_vector = numerical_metrics.pop() if collect_telemetry else None
            if telemetry_vector is not None and data_parallel is None:
                # The single-device step returns its statistics by name; pack them for one transfer.
                telemetry_keys = sorted(telemetry_vector)
                telemetry_vector = telemetry.pack(telemetry_vector)
            is_log = step == start_step + 1 or step % config.train.log_every == 0
            is_eval = evaluation_due(config, step)
            is_checkpoint = is_eval or step % config.train.ckpt_every == 0 or step == end_step
            # Check every update, including steps between log/checkpoint events.
            loss_value = float(loss)
            # The host scalar synchronizes the update. Keep early measurements
            # so pilots can separate compilation from steady training speed.
            if len(timing["train_step_seconds"]) < (32 if profile_timing else 8):
                timing["train_step_seconds"].append(time.perf_counter() - train_step_start)
            if telemetry_vector is not None:
                previous_telemetry = telemetry_values
                telemetry_values = dict(zip(telemetry_keys,
                                            np.asarray(jax.device_get(telemetry_vector)).tolist(), strict=True))
            if not math.isfinite(loss_value):
                _fail_numerics(config, state, step,
                               f"Nonfinite training loss or optimizer update at step {step}: {loss_value}",
                               diagnostic_sample, diagnostic_provenance, loss=loss_value,
                               numerical_metrics=numerical_metrics[0] if numerical_metrics else None,
                               telemetry_record=({"failed_update": telemetry_values,
                                                  "preceding_update": previous_telemetry}
                                                 if telemetry_values is not None else None))
            # This is the already-computed norm of the full, averaged gradient
            # before the optimizer clips it, including any coefficient penalty and
            # excluding coefficient entries a release schedule still freezes.
            gradient_norm = (float(np.asarray(jax.device_get(numerical_metrics[0]))[1])
                             if numerical_metrics else None)
            if gradient_norm is not None:
                gradient_norms.append(gradient_norm)
            clipped = gradient_norm is not None and gradient_norm > config.optim.grad_clip_norm
            clip_scale = config.optim.grad_clip_norm / gradient_norm if clipped else 1.0
            if telemetry_values is not None:
                spike = telemetry_window.add(step, telemetry_keys,
                                             list(telemetry_values.values()), gradient_norm)
                if spike is not None:
                    print(f"gradient_spike step={step} gradient_norm={gradient_norm:.6f} "
                          f"running_median={spike['running_median']:.6f}"
                          + "".join(f" {key}={value:.6g}" for key, value in telemetry.log_summary(telemetry_values).items()),
                          flush=True)
            if (is_log or is_eval) and penalty_sum is not None:
                # The penalty term as added to the loss, at the updated parameters; the
                # recorded training NLL and perplexities exclude it.
                penalty_value = penalty_weight * float(penalty_sum(state.params))
            if is_log:
                train_losses.append(loss_value)
                train_ppl = _perplexity(config, state, step, loss_value, diagnostic_sample, diagnostic_provenance)
                print(f"step={step} train_nll={loss_value:.6f} train_ppl={train_ppl:.4f}"
                      + (f" train_coefficient_penalty={penalty_value:.6f}" if penalty_value is not None else "")
                      # Retained logs keep the individual sampled points; the
                      # validation record summarizes every update in its window.
                      + (f" gradient_norm_pre_clip={gradient_norm:.6f} gradient_clip_scale={clip_scale:.6f}"
                         if gradient_norm is not None else "")
                      + ("".join(f" {key}={value:.6g}" for key, value in telemetry.log_summary(telemetry_values).items())
                         if telemetry_values is not None else ""),
                      flush=True)
                tracking.log_train(wandb_run, step, loss_value, train_ppl, float(schedule(step - 1)),
                                   coefficient_penalty=penalty_value, gradient_norm=gradient_norm,
                                   gradient_clip_norm=config.optim.grad_clip_norm, telemetry=telemetry_values)
            if is_eval:
                interval_seconds = time.perf_counter() - window_time
                evaluation_start = time.perf_counter()
                if not train_losses:
                    train_losses.append(loss_value)
                diagnostics = collect_diagnostics()
                if not diagnostics["valid"]:
                    _fail_numerics(config, state, step,
                                   f"Invalid attention diagnostics at step {step}: {'; '.join(diagnostics['issues'])}",
                                   diagnostic_sample, diagnostic_provenance, diagnostics=diagnostics)
                eval_loss = collect_evaluation()
                if not math.isfinite(eval_loss):
                    _fail_numerics(config, state, step,
                                   f"Nonfinite validation loss at step {step}: {eval_loss}",
                                   diagnostic_sample, diagnostic_provenance, diagnostics=diagnostics, loss=eval_loss)
                eval_ppl = _perplexity(config, state, step, eval_loss, diagnostic_sample, diagnostic_provenance,
                                       diagnostics=diagnostics)
                timing["evaluation_seconds"].append(time.perf_counter() - evaluation_start)
                # Orbax keeps the latest checkpoint among equal best metrics.
                # Match that tie rule in the learned family's selected record.
                # Orbax chooses the later checkpoint on exact validation ties.
                # Keep best_validation.json and committed metrics in agreement.
                is_best = eval_loss <= best_nll
                train_nll = float(np.mean(train_losses))
                train_ppl = _perplexity(config, state, step, train_nll, diagnostic_sample, diagnostic_provenance,
                                        diagnostics=diagnostics)
                throughput = (step - window_step) * tokens_per_step / interval_seconds
                record = {
                    "schema_version": METRICS_SCHEMA_VERSION, "event": "validation",
                    "event_id": f"validation:{step}", "attempt_id": attempt_id,
                    "timestamp_utc": datetime.now(UTC).isoformat(), "step": step, "checkpoint_step": step,
                    "training_tokens": step * tokens_per_step, "training_window_start_step": window_step,
                    "train_nll_sample_count": len(train_losses), "train_nll_sample_mean": train_nll,
                    "train_perplexity_from_mean_nll": train_ppl,
                    "validation_nll": eval_loss, "validation_perplexity": eval_ppl,
                    "validation_batches": config.train.eval_batches,
                    "validation_sequences": config.train.eval_batches * config.data.batch_size,
                    "validation_tokens": config.train.eval_batches * tokens_per_step,
                    **({
                        "train_gradient_norm_count": len(gradient_norms),
                        "train_gradient_norm_mean": float(np.mean(gradient_norms)),
                        "train_gradient_norm_max": max(gradient_norms),
                        "train_gradient_clipped_fraction": float(np.mean(
                            np.asarray(gradient_norms) > config.optim.grad_clip_norm)),
                        "gradient_clip_norm": config.optim.grad_clip_norm,
                    } if gradient_norms else {}),
                    **({"train_coefficient_penalty": penalty_value} if penalty_value is not None else {}),
                    **(telemetry_window.summary(telemetry_keys) if collect_telemetry else {}),
                    "learning_rate": float(schedule(step - 1)), "tokens_per_second": throughput,
                    "tokens_per_second_per_device": throughput / layout.device_count,
                    "training_interval_seconds": interval_seconds, "parameter_count": parameter_count,
                    "qk_head_dim": config.model.qk_head_dim, "value_head_dim": config.model.value_head_dim,
                    **attention_identity(config), "attention_diagnostics": diagnostics,
                    "source_commit": config.train.source_commit or None,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
                    "wandb_run_id": wandb_run_id, "is_best": is_best,
                }
                checkpoint_start = time.perf_counter()
                metrics_store.stage(record)
                if not save_checkpoint(manager, step, state, metrics={
                    "validation_nll": eval_loss, "validation_perplexity": eval_ppl,
                    "has_validation": True, "training_tokens": step * tokens_per_step,
                }):
                    raise RuntimeError(f"Failed to save validation checkpoint at step {step}")
                manager.wait_until_finished()
                timing["checkpoint_seconds"].append(time.perf_counter() - checkpoint_start)
                if is_best:
                    best_nll = eval_loss
                    write_json_atomic(config.train.workdir / "best_validation.json", _best_validation_record(record))
                metrics_store.commit(record, latest_checkpoint_step=manager.latest_step(),
                                     mirror=lambda event: tracking.log_validation(wandb_run, event))
                _write_status(config, step=step, training_tokens=step * tokens_per_step,
                              validation_nll=eval_loss, validation_perplexity=eval_ppl,
                              attention_diagnostics=diagnostics,
                              next_train_sequence=step * config.data.batch_size,
                              rng_state={"seed": config.train.seed, "completed_updates": step})
                print(f"step={step} val_nll={eval_loss:.6f} val_ppl={eval_ppl:.4f} best={is_best}", flush=True)
                window_time, window_step, train_losses = time.perf_counter(), step, []
                gradient_norms = []
                telemetry_window.reset()
            elif is_checkpoint:
                if not save_checkpoint(manager, step, state, metrics=None):
                    raise RuntimeError(f"Failed to save resume checkpoint at step {step}")
            if is_checkpoint:
                print(f"checkpoint_saved_step={step}", flush=True)
        manager.wait_until_finished()
        timing["runtime_seconds"] = time.perf_counter() - runtime_start
        final_step = int(state.step)
        result = {
            "latest_checkpoint_before": latest_before, "restored": restored_step is not None,
            "restored_step": restored_step, "start_step": start_step, "end_step": end_step,
            "data_start_sequence": start_sequence, "final_step": final_step,
            "training_tokens": final_step * tokens_per_step,
            "latest_checkpoint_after": manager.latest_step(), "best_checkpoint_after": manager.best_step(),
            "data_parallel_devices": layout.device_count,
            "timing": timing,
            **attention_identity(config),
        }
        if audit is not None:
            # Compare the live arrays to durable checkpoint bytes, independently
            # of nondeterministic GPU arithmetic in separate training executions.
            audit["final_state_sha256"] = state_sha256(state)
            restored, checkpoint_step = restore_latest_checkpoint(manager, state)
            audit["checkpoint_state_sha256"] = state_sha256(restored)
            if checkpoint_step != final_step or audit["final_state_sha256"] != audit["checkpoint_state_sha256"]:
                raise RuntimeError("Live training state differs from its durable checkpoint")
            del restored
            if final_step < config.optim.total_steps:
                # Peek at the original iterator. The fresh resumed process hashes
                # its actual first batch, so cursor bugs cannot hide behind a
                # common reconstruction helper or identical step counters.
                next_batches = np.stack([next(train_iter) for _ in range(layout.grad_accum_steps)])
                audit["next_update"] = _update_audit(final_step + 1, next_batches,
                                                     jax.random.fold_in(rng, final_step + 1))
            result["checkpoint_audit"] = audit
        if final_step == config.optim.total_steps:
            terminal_metrics = manager.metrics(final_step)
            if not terminal_metrics or manager.latest_step() != final_step:
                raise RuntimeError("Completed run is missing its terminal validation checkpoint")
            terminal_record = next(
                (record for line in reversed(metrics_store.metrics_path.read_text().splitlines())
                 if (record := json.loads(line))["step"] == final_step), None,
            )
            if terminal_record is None:
                raise RuntimeError("Completed run is missing its committed terminal diagnostics")
            completion = {
                "format": "shared-transformer-training-completion", "version": 1,
                "completed_utc": datetime.now(UTC).isoformat(), **result,
                "validation_nll": terminal_metrics["validation_nll"],
                "validation_perplexity": terminal_metrics["validation_perplexity"],
                "retained_terminal_checkpoint_step": final_step,
                "retained_best_checkpoint_step": manager.best_step(),
                "source_commit": config.train.source_commit or None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "parameter_count": parameter_count, "qk_head_dim": config.model.qk_head_dim,
                "attention_diagnostics": terminal_record["attention_diagnostics"],
            }
            write_json_atomic(completion_path, completion)
            best_path = config.train.workdir / "best_validation.json"
            tracking.summarize_completion(
                wandb_run, {**completion, "step": final_step},
                json.loads(best_path.read_text()) if best_path.exists() else None)
            _write_status(config, status="complete", step=final_step, **{
                key: completion[key] for key in (
                    "training_tokens", "validation_nll", "validation_perplexity",
                    "retained_terminal_checkpoint_step", "retained_best_checkpoint_step",
                    "attention_diagnostics",
                )
            })
            result.update(completion)
        else:
            _write_status(config, status="paused", step=final_step, training_tokens=final_step * tokens_per_step,
                          next_train_sequence=final_step * config.data.batch_size,
                          rng_state={"seed": config.train.seed, "completed_updates": final_step})
        return result
    finally:
        tracking.finish(wandb_run)
