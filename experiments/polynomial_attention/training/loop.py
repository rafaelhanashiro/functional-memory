import json
import math
import os
import time
import uuid
from dataclasses import asdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import trange

from experiments.polynomial_attention.configs import Config
from experiments.polynomial_attention.configs.base import OFM_SOURCE_COMMIT, validate_config
from experiments.polynomial_attention.data.loaders import make_eval_iterator, make_train_iterator
from experiments.polynomial_attention.data.packed import (
    LoadedPackedManifest,
    load_packed_manifest,
    manifest_fingerprint,
    resolve_packed_manifest,
)
from experiments.polynomial_attention.training.checkpointing import (
    VALIDATION_NLL_METRIC,
    cleanup_stale_checkpoint_temps,
    make_checkpoint_manager,
    restore_latest_checkpoint,
    save_checkpoint,
)
from experiments.polynomial_attention.training.distributed import (
    BatchLayout,
    DataParallelFunctions,
    make_data_parallel_functions,
    put_eval_batch,
    put_replicated,
    put_train_batches,
    resolve_batch_layout,
    select_data_parallel_devices,
)
from experiments.polynomial_attention.training.diagnostics import (
    attention_diagnostics,
    attention_identity,
    fixed_diagnostic_sample,
    json_number,
)
from experiments.polynomial_attention.training.metrics import (
    METRICS_SCHEMA_VERSION,
    ValidationMetricsStore,
    write_json_atomic,
)
from experiments.polynomial_attention.training.optim import make_learning_rate_schedule
from experiments.polynomial_attention.training.randomness import model_forward_rngs
from experiments.polynomial_attention.training.state import TrainState, create_state


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


def loss_and_grads(state: TrainState, batch: jax.Array, rng: jax.Array):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    def loss_fn(params):
        model = nnx.merge(state.graphdef, params)
        logits = model(
            inputs,
            deterministic=False,
            rngs=model_forward_rngs(rng),
        )
        return cross_entropy_loss(logits, targets)

    return nnx.value_and_grad(loss_fn)(state.params)


@nnx.jit
def train_step(state: TrainState, batch: jax.Array, rng: jax.Array):
    loss, grads = loss_and_grads(state, batch, rng)
    state = state.apply_gradients(grads=grads)
    return state, loss


@nnx.jit
def microbatch_loss_and_grads(state: TrainState, batch: jax.Array, rng: jax.Array):
    return loss_and_grads(state, batch, rng)


@nnx.jit
def apply_gradients_step(state: TrainState, grads):
    return state.apply_gradients(grads=grads)


def accumulated_train_step(state: TrainState, batches: jax.Array, rngs: jax.Array):
    grad_sum = None
    loss_sum = None
    for batch, rng in zip(batches, rngs, strict=True):
        loss, grads = microbatch_loss_and_grads(state, batch, rng)
        if grad_sum is None:
            grad_sum = grads
            loss_sum = loss
        else:
            grad_sum = jax.tree_util.tree_map(jnp.add, grad_sum, grads)
            loss_sum = loss_sum + loss

    scale = jnp.asarray(1.0 / len(batches), dtype=loss_sum.dtype)
    grads = jax.tree_util.tree_map(lambda grad: grad * scale, grad_sum)
    loss = loss_sum * scale
    return apply_gradients_step(state, grads), loss


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
    serializable = asdict(config)
    serializable["train"]["workdir"] = str(serializable["train"]["workdir"])
    return serializable


def maybe_init_wandb(config: Config) -> tuple[object | None, str | None]:
    if config.train.wandb_mode == "disabled":
        return None, None

    run_id = None
    try:
        identity_path = config.train.workdir / "wandb_run.json"
        expected_identity = {
            "entity": config.train.wandb_entity,
            "project": config.train.wandb_project,
            "run_name": config.train.wandb_run_name,
        }
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            for key, value in expected_identity.items():
                if identity.get(key) != value:
                    raise ValueError(
                        f"W&B {key} changed from {identity.get(key)!r} to {value!r}"
                    )
            run_id = str(identity["run_id"])
        else:
            run_id = uuid.uuid4().hex
            write_json_atomic(
                identity_path,
                {**expected_identity, "run_id": run_id},
            )

        import wandb

        run = wandb.init(
            project=config.train.wandb_project,
            entity=config.train.wandb_entity,
            name=config.train.wandb_run_name,
            id=run_id,
            resume="allow",
            mode=config.train.wandb_mode,
            dir=str(config.train.workdir),
            config=_serializable_config(config),
            settings=wandb.Settings(init_timeout=30),
        )
    except Exception as error:
        print(
            f"wandb_init_failed={type(error).__name__}: {error}; "
            "continuing_with_local_metrics=true",
            flush=True,
        )
        return None, run_id
    return run, run_id


def safe_wandb_log(wandb_run, record: dict) -> None:
    if wandb_run is None:
        return
    payload = {
        "train/nll_sample_mean": record["train_nll_sample_mean"],
        "train/ppl_from_mean_nll": record["train_perplexity_from_mean_nll"],
        "train/learning_rate": record["learning_rate"],
        "train/tokens": record["training_tokens"],
        "train/tokens_per_second": record["tokens_per_second"],
        "train/tokens_per_second_per_device": record[
            "tokens_per_second_per_device"
        ],
        "val/nll": record["validation_nll"],
        "val/ppl": record["validation_perplexity"],
        "val/is_best": record["is_best"],
        "model/parameter_count": record["parameter_count"],
    }
    try:
        wandb_run.log(payload, step=int(record["step"]))
    except Exception as error:
        print(
            f"wandb_log_failed_step={record['step']} "
            f"error={type(error).__name__}: {error}; "
            "local_metrics_committed=true",
            flush=True,
        )


def safe_wandb_finish(wandb_run) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except Exception as error:
        print(
            f"wandb_finish_failed={type(error).__name__}: {error}",
            flush=True,
        )


def ensure_run_config(config: Config, *, resume: bool) -> None:
    path = config.train.workdir / "config.json"
    current = _serializable_config(config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != current:
            raise ValueError(
                f"Training configuration does not match the initialized run: {path}"
            )
        if not resume and not (config.train.workdir / "completion.json").exists():
            raise ValueError(
                f"Run workdir is already initialized; use resume or a new workdir: {path}"
            )
        return
    write_json_atomic(path, current)


def resolve_workdir(config: Config) -> None:
    workdir = config.train.workdir.expanduser()
    if not workdir.is_absolute():
        workdir = Path.cwd() / workdir
    config.train.workdir = workdir.resolve()


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
        if start_step != 0:
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
) -> float:
    """Mean token NLL, with bounded evaluation memory and a fixed data prefix."""
    if num_batches < 1:
        raise ValueError("Evaluation requires at least one batch")
    nll_sum, token_count = 0.0, 0
    eval_rng = jax.random.PRNGKey(eval_seed)
    for batch_index in range(num_batches):
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
            mirror=lambda event: safe_wandb_log(wandb_run, event),
        )


def _write_status(config: Config, **updates) -> None:
    path = config.train.workdir / "metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update({
        **attention_identity(config),
        "qk_head_dim": config.model.qk_head_dim,
        "value_head_dim": config.model.value_head_dim,
        "seed": config.train.seed,
        "source_commit": config.train.source_commit or None,
        "ofm_source_commit": OFM_SOURCE_COMMIT,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID") or metadata.get("slurm_job_id"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", str(metadata.get("slurm_restart_count", 0)))),
        "updated_utc": datetime.now(UTC).isoformat(),
        "target_steps": config.optim.total_steps,
        "target_tokens": config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1),
        **updates,
    })
    identity_path = config.train.workdir / "data_identity.json"
    if identity_path.exists():
        metadata["data_identity"] = json.loads(identity_path.read_text())
    launch_path = config.train.workdir / "launch.json"
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        if launch.get("log_path") is not None:
            metadata["log_path"] = launch["log_path"]
    write_json_atomic(path, metadata)
    lines = [
        f"# Polynomial attention: {metadata['attention_kernel']}"
        + (f" degree {config.model.attention_degree}" if config.model.attention_degree is not None else " control"), "",
        f"Status: {metadata.get('status', 'initialized')}.", "",
        f"- Attention: {metadata['attention_kernel']}; degree: {config.model.attention_degree}; backend: {config.model.attention_backend}.",
        f"- Query/key width: {config.model.qk_head_dim}; value width: {config.model.value_head_dim}.",
        f"- Parameters: {metadata.get('parameter_count', 'pending')}; seed: {config.train.seed}.",
        f"- Updates: {metadata.get('step', 0):,} / {config.optim.total_steps:,}.",
        f"- Prediction tokens: {metadata.get('training_tokens', 0):,} / {metadata['target_tokens']:,}.",
        f"- Source: `{metadata['source_commit']}`; Slurm job: `{metadata['slurm_job_id']}`.",
    ]
    if metadata.get("validation_nll") is not None:
        lines += [f"- Latest validation NLL: {metadata['validation_nll']:.6f}; PPL: {metadata['validation_perplexity']:.6f}."]
    if metadata.get("m0_qk_parameter_note"):
        lines += ["", metadata["m0_qk_parameter_note"]]
    if metadata.get("attention_diagnostics"):
        diagnostic = metadata["attention_diagnostics"]
        lines += ["", f"Attention diagnostics at update {diagnostic['step']}: "
                  f"{'valid' if diagnostic['valid'] else 'failed'} on the fixed first held-out "
                  f"sequence ({diagnostic['sample']['prediction_tokens']} prediction positions)."]
    if metadata.get("status") == "complete":
        lines += ["", "Primary result: terminal PPL at the common prediction-token budget.",
                  f"Retained terminal checkpoint: `{metadata['retained_terminal_checkpoint_step']}`; validation best: `{metadata['retained_best_checkpoint_step']}`."]
    if metadata.get("error"):
        lines += ["", f"Failure: `{metadata['error']}`"]
    from experiments.polynomial_attention.training.metrics import _write_text_atomic
    _write_text_atomic(config.train.workdir / "REPORT.md", "\n".join(lines) + "\n")


def _fail_numerics(config, state, step, message, sample, provenance, *, diagnostics=None, loss=None):
    """Persist the offending state diagnostics before aborting without a repair."""
    failure = {
        "format": "polynomial-attention-numerical-failure", "version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(), "step": step,
        "training_tokens": step * config.data.batch_size * (config.data.sequence_length - 1),
        **attention_identity(config), "error": message,
        "diagnostics_state": "after_attempted_training_update",
    }
    if loss is not None:
        failure["loss"] = json_number(loss)
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


def run_training(config: Config, *, resume: bool = True, stop_after_step: int | None = None) -> dict:
    """Train with an exact packed-data cursor and transactional local validation metrics."""
    validate_config(config)
    resolve_workdir(config)
    layout = resolve_batch_layout(config)
    devices = select_data_parallel_devices(layout)
    end_step = config.optim.total_steps if stop_after_step is None else stop_after_step
    if not 1 <= end_step <= config.optim.total_steps:
        raise ValueError("stop_after_step must be between 1 and optim.total_steps")
    if config.train.eval_every < 1 or config.train.eval_batches < 1:
        raise ValueError("This experiment requires validation, including at the terminal step")
    config.train.workdir.mkdir(parents=True, exist_ok=True)
    ensure_run_config(config, resume=resume)
    for path in cleanup_stale_checkpoint_temps(config):
        print(f"removed_stale_checkpoint_tmp={path}", flush=True)
    manager = make_checkpoint_manager(config)
    try:
        return _run_training(config, manager, layout, devices, resume=resume, end_step=end_step)
    except BaseException as error:
        _write_status(config, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manager.wait_until_finished()
        manager.close()


def _run_training(config, manager, layout, devices, *, resume, end_step):
    completion_path = config.train.workdir / "completion.json"
    latest_before = manager.latest_step()
    validated_train_manifest = None
    validated_eval_manifest = None
    if config.data.source == "packed":
        validated_train_manifest = load_packed_manifest(config.data.train_manifest)
        validated_eval_manifest = load_packed_manifest(config.data.eval_manifest)
    # Check data identity on all entries, including an already completed run.
    start_sequence = resolve_packed_data_cursor(
        config, latest_before or 0,
        validated_train_manifest=validated_train_manifest,
        validated_eval_manifest=validated_eval_manifest,
    )
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if (not resume or completion.get("format") != "polynomial-attention-training-completion"
                or completion.get("final_step") != config.optim.total_steps
                or latest_before != config.optim.total_steps):
            raise ValueError(f"Invalid or incompatible completed run: {completion_path}")
        _write_status(config, status="complete", step=config.optim.total_steps, **{
            key: completion[key] for key in (
                "training_tokens", "validation_nll", "validation_perplexity",
                "retained_terminal_checkpoint_step", "retained_best_checkpoint_step",
                "attention_diagnostics",
            )
        })
        print(f"run_already_complete_step={config.optim.total_steps}", flush=True)
        return completion
    rng = jax.random.PRNGKey(config.train.seed)
    state = create_state(config, rng)
    restored_step = None
    if resume:
        state, restored_step = restore_latest_checkpoint(manager, state)
    start_step = int(state.step)
    if end_step < start_step:
        raise ValueError("stop_after_step precedes restored training step")
    if restored_step is not None and start_step != restored_step:
        raise ValueError("Checkpoint state step differs from checkpoint directory")
    if start_sequence != start_step * config.data.batch_size:
        raise ValueError("Checkpoint step and data cursor disagree")
    parameter_count = param_count(state)
    tokens_per_step = config.data.batch_size * (config.data.sequence_length - 1)
    _write_status(config, status="running", step=start_step, training_tokens=start_step * tokens_per_step,
                  parameter_count=parameter_count, error=None, devices=[str(d) for d in devices],
                  rng_state={"seed": config.train.seed, "completed_updates": start_step},
                  next_train_sequence=start_sequence)
    print(f"param_count={parameter_count} restored_step={restored_step} packed_data_start_sequence={start_sequence}", flush=True)
    print(f"data_parallel_devices={layout.device_count} global_batch_size={layout.global_batch_size} "
          f"global_micro_batch_size={layout.global_micro_batch_size} grad_accum_steps={layout.grad_accum_steps}", flush=True)
    data_parallel = None
    if layout.device_count > 1:
        data_parallel = make_data_parallel_functions(state, devices)
        state = put_replicated(state, data_parallel.replicated_sharding)
    train_iter = make_train_iterator(
        replace(config.data, batch_size=layout.global_micro_batch_size),
        start_sequence=start_sequence, validated_manifest=validated_train_manifest,
    )
    diagnostic_sample, diagnostic_provenance = fixed_diagnostic_sample(
        config, validated_manifest=validated_eval_manifest,
    )
    _write_status(config, diagnostic_sample=diagnostic_provenance)
    wandb_run, wandb_run_id = maybe_init_wandb(config)
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
        attempt_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.environ.get('SLURM_RESTART_COUNT', '0')}:{uuid.uuid4().hex[:8]}"
        for step in trange(start_step + 1, end_step + 1, desc="train"):
            step_rng = jax.random.fold_in(rng, step)
            if data_parallel is not None:
                batches = np.stack([next(train_iter) for _ in range(layout.grad_accum_steps)])
                state, loss = data_parallel.train_step(state, put_train_batches(batches, data_parallel.train_batch_sharding), step_rng)
            elif layout.grad_accum_steps == 1:
                state, loss = train_step(state, jnp.asarray(next(train_iter)), step_rng)
            else:
                batches = jnp.asarray(np.stack([next(train_iter) for _ in range(layout.grad_accum_steps)]))
                micro_rngs = jax.random.split(step_rng, layout.grad_accum_steps)
                state, loss = accumulated_train_step(state, batches, micro_rngs)
            is_log = step == start_step + 1 or step % config.train.log_every == 0
            is_eval = step % config.train.eval_every == 0 or step == config.optim.total_steps
            is_checkpoint = is_eval or step % config.train.ckpt_every == 0 or step == end_step
            # Check every update, including steps between log/checkpoint events.
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                _fail_numerics(config, state, step,
                               f"Nonfinite training loss at step {step}: {loss_value}",
                               diagnostic_sample, diagnostic_provenance, loss=loss_value)
            if is_log:
                train_losses.append(loss_value)
                print(f"step={step} train_nll={loss_value:.6f} train_ppl={math.exp(loss_value):.4f}", flush=True)
            if is_eval:
                interval_seconds = time.perf_counter() - window_time
                if not train_losses:
                    train_losses.append(loss_value)
                diagnostics = attention_diagnostics(state, config, diagnostic_sample, diagnostic_provenance)
                if not diagnostics["valid"]:
                    _fail_numerics(config, state, step,
                                   f"Invalid attention diagnostics at step {step}: {'; '.join(diagnostics['issues'])}",
                                   diagnostic_sample, diagnostic_provenance, diagnostics=diagnostics)
                eval_loss = evaluate(
                    state, make_eval_iterator(config.data, validated_manifest=validated_eval_manifest),
                    config.train.eval_batches, data_parallel=data_parallel, eval_seed=config.train.seed,
                    micro_batch_size=layout.global_micro_batch_size,
                )
                if not math.isfinite(eval_loss):
                    _fail_numerics(config, state, step,
                                   f"Nonfinite validation loss at step {step}: {eval_loss}",
                                   diagnostic_sample, diagnostic_provenance, diagnostics=diagnostics, loss=eval_loss)
                eval_ppl = math.exp(eval_loss)
                is_best = eval_loss < best_nll
                train_nll = float(np.mean(train_losses))
                throughput = (step - window_step) * tokens_per_step / interval_seconds
                record = {
                    "schema_version": METRICS_SCHEMA_VERSION, "event": "validation",
                    "event_id": f"validation:{step}", "attempt_id": attempt_id,
                    "timestamp_utc": datetime.now(UTC).isoformat(), "step": step, "checkpoint_step": step,
                    "training_tokens": step * tokens_per_step, "training_window_start_step": window_step,
                    "train_nll_sample_count": len(train_losses), "train_nll_sample_mean": train_nll,
                    "train_perplexity_from_mean_nll": math.exp(train_nll),
                    "validation_nll": eval_loss, "validation_perplexity": eval_ppl,
                    "validation_batches": config.train.eval_batches,
                    "validation_sequences": config.train.eval_batches * config.data.batch_size,
                    "validation_tokens": config.train.eval_batches * tokens_per_step,
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
                metrics_store.stage(record)
                if not save_checkpoint(manager, step, state, metrics={
                    "validation_nll": eval_loss, "validation_perplexity": eval_ppl,
                    "has_validation": True, "training_tokens": step * tokens_per_step,
                }):
                    raise RuntimeError(f"Failed to save validation checkpoint at step {step}")
                manager.wait_until_finished()
                if is_best:
                    best_nll = eval_loss
                    write_json_atomic(config.train.workdir / "best_validation.json", _best_validation_record(record))
                metrics_store.commit(record, latest_checkpoint_step=manager.latest_step(),
                                     mirror=lambda event: safe_wandb_log(wandb_run, event))
                _write_status(config, step=step, training_tokens=step * tokens_per_step,
                              validation_nll=eval_loss, validation_perplexity=eval_ppl,
                              attention_diagnostics=diagnostics,
                              next_train_sequence=step * config.data.batch_size,
                              rng_state={"seed": config.train.seed, "completed_updates": step})
                print(f"step={step} val_nll={eval_loss:.6f} val_ppl={eval_ppl:.4f} best={is_best}", flush=True)
                window_time, window_step, train_losses = time.perf_counter(), step, []
            elif is_checkpoint:
                if not save_checkpoint(manager, step, state, metrics=None):
                    raise RuntimeError(f"Failed to save resume checkpoint at step {step}")
            if is_checkpoint:
                print(f"checkpoint_saved_step={step}", flush=True)
        manager.wait_until_finished()
        final_step = int(state.step)
        result = {
            "latest_checkpoint_before": latest_before, "restored": restored_step is not None,
            "restored_step": restored_step, "start_step": start_step, "end_step": end_step,
            "data_start_sequence": start_sequence, "final_step": final_step,
            "training_tokens": final_step * tokens_per_step,
            "latest_checkpoint_after": manager.latest_step(), "best_checkpoint_after": manager.best_step(),
            "data_parallel_devices": layout.device_count,
            **attention_identity(config),
        }
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
                "format": "polynomial-attention-training-completion", "version": 1,
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
        safe_wandb_finish(wandb_run)
