"""Fit polynomial coefficient tables or generators on fixed selected inputs.

Coefficient tables have [example, layer, query, head, order] axes and store the
effective monomial c_r exactly. Adam operates in factorial coordinates r! c_r
through gradient/update preconditioning, without rounding the initialization.
The pretrained parameter tree is an immutable argument, never an optimizer input.
Generator fitting instead optimizes only the original model's query-coefficient
parameters. Each forward recomputes the full model with its frozen backbone.
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


GENERATOR_PARAMETER_NAMES = frozenset({
    "query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel",
    "query_polynomial_tanh_bias",
})


def split_generator_parameters(params):
    """Select only the model-owned query generator, excluding every backbone leaf."""
    generator, frozen = nnx.split_state(
        params, lambda path, value: str(path[-1]) in GENERATOR_PARAMETER_NAMES, ...,
    )
    names = {str(path[-1]) for path, _ in nnx.to_flat_state(generator)}
    required = GENERATOR_PARAMETER_NAMES - {"query_polynomial_tanh_bias"}
    if not required <= names or not jax.tree.leaves(frozen):
        raise ValueError("Expected a tanh query generator and a nonempty frozen backbone")
    return generator, frozen


def _generator_group_norms(state):
    squared = {}
    for path, value in nnx.to_flat_state(state):
        name = str(path[-1])
        squared[name] = squared.get(name, 0.) + jnp.sum(jnp.square(value[...].astype(jnp.float32)))
    return {name: jnp.sqrt(value) for name, value in squared.items()}


def make_generator_functions(graph, oracle: dict):
    """Adam over stored generator parameters, with all downstream paths differentiated.

    The model already applies its factorial factors to bias and amplitude.
    Applying another coordinate scaling here would change the requested optimizer.
    The immutable source tree is passed separately and never enters the optimizer.
    """
    tx = optax.chain(
        optax.clip_by_global_norm(float(oracle["clip_norm"])),
        optax.scale_by_adam(b1=float(oracle["beta1"]), b2=float(oracle["beta2"]),
                           eps=float(oracle["epsilon"])),
    )

    def logits(generator, source, tokens):
        _, frozen = split_generator_parameters(source)
        return nnx.merge(graph, frozen, generator)(tokens[:, :-1], deterministic=True)

    def losses(generator, source, tokens):
        if tokens.shape[0] != 1:
            raise ValueError("Shared query-generator fitting currently requires one sequence")
        return optax.softmax_cross_entropy_with_integer_labels(
            logits(generator, source, tokens).astype(jnp.float32), tokens[:, 1:],
        ).mean(axis=1)

    @jax.jit
    def capture(source, tokens):
        generator, _ = split_generator_parameters(source)
        original = nnx.merge(graph, source)(tokens[:, :-1], deterministic=True)
        replay = logits(generator, source, tokens)
        nll = lambda values: optax.softmax_cross_entropy_with_integer_labels(
            values.astype(jnp.float32), tokens[:, 1:],
        ).mean(axis=1)
        return generator, nll(original), nll(replay), jnp.max(jnp.abs(
            original.astype(jnp.float32) - replay.astype(jnp.float32)))

    def objective(generator, source, tokens):
        per_example = losses(generator, source, tokens)
        return per_example.mean(), per_example

    @jax.jit
    def step(generator, optimizer, source, tokens, learning_rate):
        (_, per_example), gradient = jax.value_and_grad(objective, has_aux=True)(
            generator, source, tokens)
        norm = optax.global_norm(gradient)
        updates, optimizer = tx.update(gradient, optimizer, generator)
        updates = jax.tree.map(lambda value: -learning_rate * value, updates)
        candidate = optax.apply_updates(generator, updates)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf))
                                   for leaf in jax.tree.leaves((candidate, optimizer, gradient))]))
        finite &= jnp.isfinite(norm) & jnp.all(jnp.isfinite(per_example))
        return candidate, optimizer, per_example, norm[None], finite, _generator_group_norms(gradient)

    return capture, jax.jit(tx.init), step, jax.jit(losses)


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
        table = jnp.swapaxes(jnp.stack(coefficients), 0, 1)
        replay = model(tokens[:, :-1], deterministic=True,
                       query_coefficients=jnp.swapaxes(table, 0, 1))
        return table, nll(logits, tokens), nll(replay, tokens), jnp.max(jnp.abs(
            logits.astype(jnp.float32) - replay.astype(jnp.float32)))

    def losses(table, params, tokens):
        model = nnx.merge(graph, params)
        logits = model(tokens[:, :-1], deterministic=True,
                       query_coefficients=jnp.swapaxes(table, 0, 1))
        return nll(logits, tokens)

    def objective(table, params, tokens):
        per_example = losses(table, params, tokens)
        # Every independent example sees the same gradient as a B=1 run.
        return per_example.sum(), per_example

    @jax.jit
    def step(table, optimizer, params, tokens, learning_rate):
        (_, per_example), gradient = jax.value_and_grad(objective, has_aux=True)(table, params, tokens)
        # For u_r = r! c_r, dL/du_r = (dL/dc_r)/r!. Keep c itself as the
        # stored parameter so captured FP32 values are reproduced bit-for-bit.
        gradient = gradient / factorial
        axes = tuple(range(1, gradient.ndim))
        norms = jnp.sqrt(jnp.sum(jnp.square(gradient), axis=axes))
        updates, optimizer = jax.vmap(tx.update)(gradient, optimizer, table)
        candidate = optax.apply_updates(table, -learning_rate * updates / factorial)
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(norms))
        finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(optimizer)]))
        return candidate, optimizer, per_example, norms, finite

    return capture, jax.jit(jax.vmap(tx.init)), step, jax.jit(losses)


def replay_diagnostics(graph, params, tokens, degree: int):
    """Separate override semantics from FP32 factorial coordinate roundoff."""
    factorial = jnp.asarray([math.factorial(order) for order in range(degree + 1)], jnp.float32)

    @jax.jit
    def compare(frozen, inputs):
        model = nnx.merge(graph, frozen)
        original, coefficients = model(inputs[:, :-1], deterministic=True, return_coefficients=True)
        replay_raw = model(inputs[:, :-1], deterministic=True, query_coefficients=coefficients)
        rounded = (coefficients * factorial) / factorial
        replay_factorial = model(inputs[:, :-1], deterministic=True, query_coefficients=rounded)
        reference = original.astype(jnp.float32)
        reference_rms = jnp.sqrt(jnp.mean(jnp.square(reference)))
        result = {"parent_logit_rms": reference_rms,
                  "coefficient_max_absolute_roundtrip_error": jnp.max(jnp.abs(rounded - coefficients)),
                  "coefficient_relative_roundtrip_l2_error": jnp.linalg.norm(rounded - coefficients) /
                      jnp.maximum(jnp.linalg.norm(coefficients), 1e-30)}
        for name, logits in (("original", original), ("raw_replay", replay_raw), ("factorial_replay", replay_factorial)):
            difference = logits.astype(jnp.float32) - reference
            result[name + "_nll"] = optax.softmax_cross_entropy_with_integer_labels(
                logits.astype(jnp.float32), inputs[:, 1:]).mean(axis=1)
            result[name + "_logit_max_absolute_error"] = jnp.max(jnp.abs(difference))
            result[name + "_logit_relative_rms_error"] = jnp.sqrt(jnp.mean(jnp.square(difference))) / reference_rms
        return result

    return {name: np.asarray(value).tolist() for name, value in jax.device_get(compare(params, tokens)).items()}


def external_replay_diagnostics(graph, params, tokens, table, *, generator=False):
    """Compare independently compiled forward graphs, including repeatability."""
    @jax.jit
    def original(frozen, inputs):
        return nnx.merge(graph, frozen)(inputs[:, :-1], deterministic=True)

    @jax.jit
    def external(frozen, inputs, values):
        if generator:
            _, backbone = split_generator_parameters(frozen)
            return nnx.merge(graph, backbone, values)(inputs[:, :-1], deterministic=True)
        return nnx.merge(graph, frozen)(inputs[:, :-1], deterministic=True,
                                       query_coefficients=jnp.swapaxes(values, 0, 1))

    @jax.jit
    def statistics(reference, replay, repeated, labels):
        reference, replay, repeated = (value.astype(jnp.float32) for value in (reference, replay, repeated))
        error = replay - reference
        return {"original_nll": optax.softmax_cross_entropy_with_integer_labels(reference, labels).mean(axis=1),
                "external_nll": optax.softmax_cross_entropy_with_integer_labels(replay, labels).mean(axis=1),
                "logit_max_absolute_error": jnp.max(jnp.abs(error)),
                "logit_relative_rms_error": jnp.sqrt(jnp.mean(jnp.square(error))) /
                    jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(reference))), 1e-30),
                "repeated_external_logit_max_absolute_error": jnp.max(jnp.abs(replay - repeated)),
                "all_logits_finite": jnp.all(jnp.isfinite(reference)) & jnp.all(jnp.isfinite(replay)) & jnp.all(jnp.isfinite(repeated))}

    reference = original(params, tokens)
    replay = external(params, tokens, table)
    repeated = external(params, tokens, table)
    return {name: np.asarray(value).tolist() for name, value in
            jax.device_get(statistics(reference, replay, repeated, tokens[:, 1:])).items()}


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


def qualify(run: Path, functions, params, tokens, table, optimizer, learning_rate,
            *, graph=None, external_nll_atol=1e-5, external_nll_rtol=1e-5, generator=False):
    capture, _, train_step, evaluate = functions
    frozen = split_generator_parameters(params)[1] if generator else params
    frozen_sha256 = state_sha256(frozen)
    _, original_nll, replay_nll, logits_error = capture(params, tokens)
    original_nll, replay_nll, logits_error = jax.device_get((original_nll, replay_nll, logits_error))
    if not np.allclose(original_nll, replay_nll, rtol=1e-5, atol=1e-5) or float(logits_error) > 0.05:
        raise RuntimeError(f"Parent replay parity failed: logits error {logits_error}")
    external_nll = np.asarray(evaluate(table, params, tokens))
    repeated_nll = np.asarray(evaluate(table, params, tokens))
    if not np.array_equal(external_nll, repeated_nll):
        raise RuntimeError("Independent external coefficient forward is not repeatable")
    external_diagnostics = None
    if graph is not None:
        print("Measuring independent-graph logit error and external-forward repeatability", flush=True)
        external_diagnostics = external_replay_diagnostics(graph, params, tokens, table, generator=generator)
        print("Independent replay: " + json.dumps(external_diagnostics), flush=True)
        if (not external_diagnostics["all_logits_finite"]
                or external_diagnostics["repeated_external_logit_max_absolute_error"] != 0.0):
            raise RuntimeError("External coefficient logits are nonfinite or nonrepeatable")
    result = train_step(table, optimizer, params, tokens, learning_rate)
    updated, updated_opt, before, norms, finite = result[:5]
    group_norms = ({name: float(value) for name, value in jax.device_get(result[5]).items()}
                   if generator else {})
    if generator and not all(np.isfinite(value) and value > 0 for value in group_norms.values()):
        raise RuntimeError(f"Generator parameter group has no finite nonzero gradient: {group_norms}")
    before, norms, finite = jax.device_get((before, norms, finite))
    if not np.allclose(original_nll, before, rtol=external_nll_rtol, atol=external_nll_atol):
        failure = {"status": "failed", "phase": "separate_training_forward_replay", "time_utc": _utc(),
                   "original_nll": original_nll.tolist(), "training_forward_nll": before.tolist(),
                   "nll_max_absolute_error": float(np.max(np.abs(original_nll - before))),
                   "nll_rtol": external_nll_rtol, "nll_atol": external_nll_atol}
        _write_json(run / "oracle_acceptance_failure.json", failure)
        print(json.dumps(failure), flush=True)
        raise RuntimeError("Separately compiled training forward does not reproduce original NLL")
    if not finite or not np.all(np.isfinite(norms)) or not np.any(norms > 0):
        raise RuntimeError("Coefficient gradient/update acceptance failed")
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
    # Measure improvement using the identical compiled training forward on both
    # sides; BF16 graph-optimization offsets must not count as fitting gains.
    after = np.asarray(next_a[2])
    if not np.all(np.isfinite(after)) or float(after.mean()) >= float(before.mean()):
        raise RuntimeError(f"Initial coefficient update did not improve NLL: {before.mean()} -> {after.mean()}")
    if state_sha256(split_generator_parameters(params)[1] if generator else params) != frozen_sha256:
        raise RuntimeError("Qualification changed the frozen parameter tree")
    return {"status": "passed", "completed_utc": _utc(), "parent_nll": float(original_nll.mean()),
            "separate_training_forward_nll": float(before.mean()),
            "external_evaluation_nll": float(external_nll.mean()),
            "external_evaluation_max_nll_error": float(np.max(np.abs(external_nll - original_nll))),
            "training_forward_max_nll_error": float(np.max(np.abs(before - original_nll))),
            "cross_compilation_nll_atol": external_nll_atol, "cross_compilation_nll_rtol": external_nll_rtol,
            "external_forward_repeated_exactly": True,
            "independent_graph_diagnostics": external_diagnostics,
            "one_update_nll": float(after.mean()), "max_logit_replay_error": float(logits_error),
            "gradient_norm_min": float(norms.min()), "gradient_norm_max": float(norms.max()),
            "checkpoint_resume_exact": True, "frozen_parameter_sha256": frozen_sha256,
            "generator_gradient_norms": group_norms,
            "frozen_parameter_scope": "backbone" if generator else "all_pretrained_parameters",
            "device": jax.devices()[0].device_kind}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    oracle = recipe["oracle"]
    parameterization = oracle.get("parameterization", "free_per_example_query_coefficients")
    if parameterization not in {"free_per_example_query_coefficients", "query_coefficient_generator"}:
        raise ValueError(f"Unknown coefficient parameterization: {parameterization}")
    generator = parameterization == "query_coefficient_generator"
    if generator and (int(oracle["sequence_count"]) != 1 or int(oracle["microbatch_size"]) != 1):
        raise ValueError("Generator fitting requires one fixed sequence and microbatch size one")
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
    repository = Path(__file__).resolve().parents[3]
    runtime = {"source_commit": os.environ.get("COEFFICIENT_ORACLE_SOURCE_COMMIT"),
               "device_kind": jax.devices()[0].device_kind, "device_count": 1,
               "resolved_attention_backend": recipe["model"]["attention_backend"],
               "coefficient_layout": "model_parameter_tree" if generator else "example,layer,query,head,order",
               "optimizer_coordinates": "stored_generator_parameters" if generator else "factorial(order)*monomial_coefficient",
               "coefficient_storage": "model_generator_parameters" if generator else "effective_monomial_coefficients",
               "parameterization": parameterization,
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
    if generator and (config.model.attention_coefficient_conditioning != "query"
                      or config.model.attention_coefficient_activation != "tanh"
                      or config.model.attention_coefficients != "learned"):
        raise ValueError("Generator fitting requires the original learned tanh query-only model")
    # GPU BF16 fusion/layout choices can shift separately compiled forwards by
    # several 1e-4 nats even with exact coefficients. Internal replay remains
    # strict; this independently measured bound is absolute, per example.
    bf16_gpu = config.train.dtype == "bfloat16" and jax.devices()[0].platform == "gpu"
    external_nll_atol, external_nll_rtol = (0.002, 0.0) if bf16_gpu else (1e-5, 1e-5)
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
    if oracle.get("batch_sha256") and data_identity["batch_sha256"] != oracle["batch_sha256"]:
        raise ValueError("Fixed fitting batch does not match the recipe's pinned token checksum")
    parent_run = source_config.parent
    parent_record = json.loads((parent_run / "record.json").read_text())
    if parent_record["selected_checkpoint"]["sha256"] != oracle["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint checksum changed")
    validate_record(parent_run, checkpoint_hashes=False)
    print("Restoring frozen parent parameters", flush=True)
    graph, params = restore_parent(config, Path(oracle["parent_checkpoint"]))
    print("Parent restored; compiling coefficient capture and replay", flush=True)
    functions = (make_generator_functions(graph, oracle) if generator else
                 make_functions(graph, config.model.attention_degree, oracle))
    capture, initialize_optimizer, train_step, evaluate = functions
    token_chunks = [jax.device_put(batch[i:i + microbatch]) for i in range(0, count, microbatch)]
    tables, optimizers, parent_losses = [], [], []
    for index, tokens in enumerate(token_chunks):
        table, losses, replay_losses, error = capture(params, tokens)
        losses, replay_losses, error = jax.device_get((losses, replay_losses, error))
        external_replay_losses = np.asarray(evaluate(table, params, tokens))
        if (not np.allclose(losses, replay_losses, rtol=1e-5, atol=1e-5)
                or not np.allclose(losses, external_replay_losses, rtol=external_nll_rtol, atol=external_nll_atol)
                or float(error) > 0.05):
            failure = {"status": "failed", "phase": "initial_coefficient_replay", "microbatch_index": index,
                       "original_nll": losses.tolist(), "replay_nll": replay_losses.tolist(),
                       "external_replay_nll": external_replay_losses.tolist(),
                       "external_nll_max_absolute_error": float(np.max(np.abs(losses - external_replay_losses))),
                       "nll_max_absolute_error": float(np.max(np.abs(losses - replay_losses))),
                       "logit_max_absolute_error": float(error), "time_utc": _utc(),
                       "tolerances": {"nll_rtol": 1e-5, "nll_atol": 1e-5, "logit_max_absolute_error": 0.05,
                                      "external_nll_rtol": external_nll_rtol, "external_nll_atol": external_nll_atol}}
            print("Capture replay failed: " + json.dumps(failure), flush=True)
            print("Diagnosing raw coefficient replay versus factorial coordinate roundtrip", flush=True)
            failure["diagnostics"] = (external_replay_diagnostics(graph, params, tokens, table, generator=True)
                                      if generator else replay_diagnostics(graph, params, tokens, config.model.attention_degree))
            _write_json(run / "oracle_acceptance_failure.json", failure)
            print(json.dumps(failure), flush=True)
            raise RuntimeError("Captured coefficients do not reproduce parent logits/loss; see oracle_acceptance_failure.json")
        tables.append(table)
        optimizers.append(initialize_optimizer(table))
        parent_losses.extend(external_replay_losses.tolist())
        if index % 4 == 0 or index + 1 == len(token_chunks):
            print(f"Captured coefficients for microbatch {index + 1}/{len(token_chunks)}; external replay NLL {np.mean(external_replay_losses):.6f}", flush=True)
    learning_rate = float(oracle["learning_rate"])
    runtime.update(trainable_parameter_count=sum(int(value.size) for value in jax.tree.leaves(tables)),
                   microbatch_size=microbatch, resolved_attention_backend=config.model.attention_backend)
    print("Qualifying coefficient-only gradient, improvement, and checkpoint resume", flush=True)
    acceptance = qualify(run, functions, params, token_chunks[0], tables[0], optimizers[0], learning_rate,
                         graph=graph, external_nll_atol=external_nll_atol, external_nll_rtol=external_nll_rtol,
                         generator=generator)
    print(f"Qualification passed: NLL {acceptance['parent_nll']:.6f} -> {acceptance['one_update_nll']:.6f}", flush=True)
    _write_json(run / "oracle_acceptance.json", acceptance)
    (run / "oracle_acceptance_failure.json").unlink(missing_ok=True)
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
        group=recipe["train"]["wandb_group"], name=recipe["train"].get("wandb_run_name", run.name),
        id=run.name, resume="allow",
        tags=["coefficient_fit", parameterization, f"sequences-{count}", runtime["device_kind"]],
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
            if step == 0:
                controller["initial_external_evaluation_nll"] = controller["initial_loss"]
                controller["initial_loss"] = loss
                controller["best_loss"] = loss
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
            if generator:
                terminal["generator_gradient_norms"] = {
                    name: float(value) for name, value in jax.device_get(results[0][5]).items()}
            stream.write(json.dumps(terminal, allow_nan=False) + "\n")
            wandb_metrics = {"train/nll": loss, "train/perplexity": terminal["perplexity"],
                             "train/best_nll": controller["best_loss"], "train/learning_rate": learning_rate,
                             "train/first_sequence_nll": float(losses[0]), "train/step_seconds": terminal["seconds"]}
            if generator:
                wandb_metrics["train/gradient_norm"] = float(norms[0])
                wandb_metrics.update({f"attention/query_coefficients/{name}/gradient_norm": value
                                      for name, value in terminal["generator_gradient_norms"].items()})
            wandb_run.log(wandb_metrics, step=step)
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
    if state_sha256(split_generator_parameters(params)[1] if generator else params) != acceptance["frozen_parameter_sha256"]:
        raise RuntimeError("Frozen pretrained parameter tree changed")
    outcome = {"status": status, "step": step, "best_step": controller["best_step"],
               "initial_nll": controller["initial_loss"], "best_nll": controller["best_loss"],
               "fit_nll": terminal["nll"] if terminal else None,
               "fit_perplexity": terminal["perplexity"] if terminal else None,
               "best_fit_nll": controller["best_loss"],
               "nll_reduction": controller["initial_loss"] - controller["best_loss"],
               "learning_rate_reduced_at": controller["reduced_at"], "finished_utc": _utc(),
               "interpretation": "Fixed-input coefficient fitting, not held-out generalization.",
               "frozen_parameters_unchanged": True, "parameterization": parameterization,
               "frozen_parameter_scope": "backbone" if generator else "all_pretrained_parameters"}
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
            checkpointer.save(final_path, {"generator_parameters" if generator else "coefficients": best_tables})
        _write_json(checkpoint / "coefficient_identity.json", {"step": controller["best_step"],
                    "layout": "model_parameter_tree" if generator else "microbatches of [example,layer,query,head,order]",
                    "coordinates": "stored_generator_parameters" if generator else "effective_monomial_c",
                    "optimizer_coordinates": "stored_generator_parameters" if generator else "factorial(order)*c",
                    "parameterization": parameterization,
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
