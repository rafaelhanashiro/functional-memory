"""Fit polynomial coefficient tables or generators on fixed selected inputs.

Coefficient tables have [example, layer, query, head, order] axes and store the
effective monomial c_r exactly. Adam operates in factorial coordinates r! c_r
through gradient/update preconditioning, without rounding the initialization.
The pretrained parameter tree is an immutable argument, never an optimizer input.
Generator fitting optimizes a shared query-coefficient network, optionally
embedding its diagonal readout in a dense one. Each forward recomputes the full
model with its frozen backbone; microbatch gradients are averaged before Adam.
Shared-batch fitting uses one [layer,query,head,order] table and averages every
sequence's gradient before a single clipping/Adam update. Static fitting uses
one [layer,head,order] table shared across query positions as well as sequences.
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
from dataclasses import fields, replace
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
    "query_polynomial_tanh_bias", "query_polynomial_mixing",
    "key_polynomial_kernel", "key_polynomial_amplitude",
})


def split_generator_parameters(params):
    """Select only the model-owned query generator, excluding every backbone leaf."""
    generator, frozen = nnx.split_state(
        params, lambda path, value: str(path[-1]) in GENERATOR_PARAMETER_NAMES, ...,
    )
    names = {str(path[-1]) for path, _ in nnx.to_flat_state(generator)}
    required = {"query_polynomial_bias", "query_polynomial_kernel"}
    readout = names & {"query_polynomial_amplitude", "query_polynomial_mixing"}
    if not required <= names or len(readout) != 1 or not jax.tree.leaves(frozen):
        raise ValueError("Expected a tanh query generator and a nonempty frozen backbone")
    key_names = names & {"key_polynomial_kernel", "key_polynomial_amplitude"}
    if key_names and len(key_names) != 2:
        raise ValueError("Key conditioning requires both its kernel and amplitude")
    return generator, frozen


def _generator_group_norms(state):
    squared = {}
    for path, value in nnx.to_flat_state(state):
        name = str(path[-1])
        squared[name] = squared.get(name, 0.) + jnp.sum(jnp.square(value[...].astype(jnp.float32)))
    return {name: jnp.sqrt(value) for name, value in squared.items()}


def exact_static_coefficients(captured):
    """Preserve a static checkpoint's coefficients without a rounded batch mean."""
    if captured.ndim != 5:
        raise ValueError("Captured coefficients need [sequence,layer,query,head,order] axes")
    static = captured[0, :, 0, :, :]
    if not bool(jnp.all(captured == static[None, :, None, :, :])):
        raise ValueError("Static checkpoint coefficients vary across sequences or query positions")
    return static


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

    def microbatch_losses(generator, source, tokens):
        return optax.softmax_cross_entropy_with_integer_labels(
            logits(generator, source, tokens).astype(jnp.float32), tokens[:, 1:],
        ).mean(axis=1)

    def losses(generator, source, tokens):
        if tokens.ndim == 2:
            return microbatch_losses(generator, source, tokens)
        if tokens.ndim != 3:
            raise ValueError("Generator inputs require [sequence,token] or [microbatch,sequence,token] axes")
        _, per_example = jax.lax.scan(
            lambda _, chunk: (None, microbatch_losses(generator, source, chunk)), None, tokens)
        return per_example.reshape(-1)

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
        per_example = microbatch_losses(generator, source, tokens)
        return per_example.mean(), per_example

    @jax.jit
    def step(generator, optimizer, source, tokens, learning_rate):
        if tokens.ndim == 2:
            (_, per_example), gradient = jax.value_and_grad(objective, has_aux=True)(
                generator, source, tokens)
        elif tokens.ndim == 3:
            # Every chunk uses the same generator. Average all gradients before
            # the single clipping/Adam update; no per-sequence optimizer exists.
            def accumulate(total, chunk):
                (_, losses), gradient = jax.value_and_grad(objective, has_aux=True)(
                    generator, source, chunk)
                return jax.tree.map(jnp.add, total, gradient), losses
            gradient, per_example = jax.lax.scan(
                accumulate, jax.tree.map(jnp.zeros_like, generator), tokens)
            gradient = jax.tree.map(lambda value: value / tokens.shape[0], gradient)
            per_example = per_example.reshape(-1)
        else:
            raise ValueError("Generator inputs require [sequence,token] or [microbatch,sequence,token] axes")
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


def _json_safe(value):
    """Nonfinite floats become null, so a failure record can always be written."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


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


def convert_generator_readout(config: Config, graph, params, readout: str):
    """Embed the trained diagonal generator in the dense family, without resetting it.

    Only the readout's representation changes: V=diag(a), with the same stored
    factorial coordinates. Backbone, W, and b arrays are reused verbatim. The
    diagnostic initializes fresh Adam state after conversion in either arm.
    """
    current = getattr(config.model, "attention_coefficient_readout", "diagonal")
    if readout == current:
        return graph, params
    if current != "diagonal" or readout != "dense":
        raise ValueError("Only diagonal-to-dense query generator conversion is supported")
    model = build_model(replace(config.model, attention_coefficient_readout="dense"),
                        nnx.Rngs(params=config.train.seed, dropout=config.train.seed))
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    dense_graph, template = nnx.split(model, nnx.Param)
    converted = []
    for path, value in nnx.to_flat_state(params):
        if str(path[-1]) == "query_polynomial_amplitude":
            amplitude = value[...]
            mixing = amplitude[..., :, None] * jnp.eye(amplitude.shape[-1], dtype=amplitude.dtype)
            converted.append(((*path[:-1], "query_polynomial_mixing"), value.replace(mixing)))
        else:
            converted.append((path, value))
    result = nnx.from_flat_state(converted)
    expected = {path: value.shape for path, value in nnx.to_flat_state(template)}
    actual = {path: value.shape for path, value in nnx.to_flat_state(result)}
    if actual != expected:
        raise ValueError("Converted generator state does not match the dense model")
    if state_sha256(split_generator_parameters(params)[1]) != state_sha256(split_generator_parameters(result)[1]):
        raise RuntimeError("Readout conversion changed the frozen backbone")
    return dense_graph, result


def make_conversion_diagnostics(original_graph, graph):
    """Compare the converted model against the actual original parent forward."""
    @jax.jit
    def compare(original_params, params, tokens):
        original, old_coefficients = nnx.merge(original_graph, original_params)(
            tokens[:, :-1], deterministic=True, return_coefficients=True)
        converted, coefficients = nnx.merge(graph, params)(
            tokens[:, :-1], deterministic=True, return_coefficients=True)
        nll = lambda logits: optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)
        old_coefficients, coefficients = jnp.asarray(old_coefficients), jnp.asarray(coefficients)
        return {"original_nll": nll(original), "converted_nll": nll(converted),
                "coefficient_max_absolute_error": jnp.max(jnp.abs(old_coefficients - coefficients)),
                "coefficient_relative_l2_error": jnp.linalg.norm(old_coefficients - coefficients) /
                    jnp.maximum(jnp.linalg.norm(old_coefficients), 1e-30),
                "logit_max_absolute_error": jnp.max(jnp.abs(original.astype(jnp.float32) - converted.astype(jnp.float32))),
                "all_finite": jnp.all(jnp.isfinite(original)) & jnp.all(jnp.isfinite(converted)) &
                    jnp.all(jnp.isfinite(coefficients))}
    return compare


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


def make_query_key_functions(parent_graph, graph, degree: int, oracle: dict):
    """Free per-example query coefficients c and key factors g: the pair coefficient is c_r(t) g_r(i).

    Tables are (c, g), each [example, layer, position, head, order]. Capture runs the
    parent graph; losses and updates run ``graph``, the parent with its query-key
    kernel backend. Adam treats c in factorial coordinates r! c_r, as for free query
    coefficients, and g directly; each example's (c, g) is clipped jointly.
    """
    factorial = jnp.asarray([math.factorial(order) for order in range(degree + 1)], jnp.float32)
    capture = make_functions(parent_graph, degree, oracle)[0]
    tx = optax.chain(
        optax.clip_by_global_norm(float(oracle["clip_norm"])),
        optax.scale_by_adam(b1=float(oracle["beta1"]), b2=float(oracle["beta2"]),
                           eps=float(oracle["epsilon"])),
    )

    def nll(logits, tokens):
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:],
        ).mean(axis=1)

    def losses(tables, params, tokens):
        coefficients, key_factors = tables
        model = nnx.merge(graph, params)
        logits = model(tokens[:, :-1], deterministic=True,
                       query_coefficients=jnp.swapaxes(coefficients, 0, 1),
                       key_factors=jnp.swapaxes(key_factors, 0, 1))
        return nll(logits, tokens)

    def objective(tables, params, tokens):
        per_example = losses(tables, params, tokens)
        return per_example.sum(), per_example

    @jax.jit
    def step(tables, optimizer, params, tokens, learning_rate):
        (_, per_example), (coefficient_gradient, key_gradient) = jax.value_and_grad(
            objective, has_aux=True)(tables, params, tokens)
        gradient = (coefficient_gradient / factorial, key_gradient)
        axes = tuple(range(1, coefficient_gradient.ndim))
        norms = jnp.sqrt(sum(jnp.sum(jnp.square(leaf), axis=axes) for leaf in gradient))
        updates, optimizer = jax.vmap(tx.update)(gradient, optimizer, tables)
        candidate = (tables[0] - learning_rate * updates[0] / factorial, tables[1] - learning_rate * updates[1])
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(candidate)]))
        finite &= jnp.all(jnp.isfinite(norms))
        finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(optimizer)]))
        return candidate, optimizer, per_example, norms, finite

    return capture, jax.jit(jax.vmap(tx.init)), step, jax.jit(losses)


def expand_shared_coefficients(table, tokens, *, static=False):
    """Broadcast one trainable table; no sequence/position-specific parameters exist."""
    if static:
        if table.ndim != 3:
            raise ValueError("Static coefficients require [layer,head,order] axes")
        return jnp.broadcast_to(table[:, None, None],
            (table.shape[0], tokens.shape[0], tokens.shape[1] - 1, *table.shape[1:]))
    if table.ndim != 4:
        raise ValueError("Shared coefficients require [layer,query,head,order] axes")
    return jnp.broadcast_to(table[:, None],
        (table.shape[0], tokens.shape[0], *table.shape[1:]))


def make_shared_functions(graph, degree: int, oracle: dict):
    """Fit one coefficient table against the mean loss of the complete batch.

    Inputs have [microbatch,microbatch_size,tokens] axes. The scan keeps only one
    microbatch's activations live, recomputes the full backbone, and accumulates
    gradients at the same coefficients. Clipping and Adam run once after the scan.
    """
    factorial = jnp.asarray([math.factorial(order) for order in range(degree + 1)], jnp.float32)
    tx = optax.chain(
        optax.clip_by_global_norm(float(oracle["clip_norm"])),
        optax.scale_by_adam(b1=float(oracle["beta1"]), b2=float(oracle["beta2"]),
                           eps=float(oracle["epsilon"])),
    )

    static = oracle.get("parameterization") == "static_coefficients"

    def microbatch_losses(table, params, tokens):
        coefficients = expand_shared_coefficients(table, tokens, static=static)
        logits = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True,
                                        query_coefficients=coefficients)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)

    def objective(table, params, tokens):
        per_example = microbatch_losses(table, params, tokens)
        return per_example.mean(), per_example

    @jax.jit
    def losses(table, params, token_chunks):
        def evaluate(_, tokens):
            return None, microbatch_losses(table, params, tokens)
        _, per_example = jax.lax.scan(evaluate, None, token_chunks)
        return per_example.reshape(-1)

    @jax.jit
    def step(table, optimizer, params, token_chunks, learning_rate):
        def accumulate(total, tokens):
            (_, per_example), gradient = jax.value_and_grad(objective, has_aux=True)(table, params, tokens)
            return total + gradient, per_example
        gradient, per_example = jax.lax.scan(accumulate, jnp.zeros_like(table), token_chunks)
        gradient = gradient / token_chunks.shape[0] / factorial
        norm = optax.global_norm(gradient)
        updates, optimizer = tx.update(gradient, optimizer, table)
        candidate = optax.apply_updates(table, -learning_rate * updates / factorial)
        finite = jnp.isfinite(norm) & jnp.all(jnp.isfinite(per_example))
        finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(value))
                                    for value in jax.tree.leaves((candidate, optimizer, gradient))]))
        return candidate, optimizer, per_example.reshape(-1), norm[None], finite

    # Capture operates on each original microbatch before projection; the caller
    # verifies replay and averages those tables without keeping a batch-sized one.
    return make_functions(graph, degree, oracle)[0], jax.jit(tx.init), step, losses


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


def reference_backend_agreement(graph, reference_graph, params, tokens):
    """Restored-parent forward on its production backend against the blocked reference."""
    @jax.jit
    def compare(source, inputs):
        labels = inputs[:, 1:]
        production = nnx.merge(graph, source)(inputs[:, :-1], deterministic=True).astype(jnp.float32)
        reference = nnx.merge(reference_graph, source)(inputs[:, :-1], deterministic=True).astype(jnp.float32)
        error = production - reference
        return {"production_nll": optax.softmax_cross_entropy_with_integer_labels(production, labels).mean(axis=1),
                "reference_nll": optax.softmax_cross_entropy_with_integer_labels(reference, labels).mean(axis=1),
                "logit_max_absolute_error": jnp.max(jnp.abs(error)),
                "logit_relative_rms_error": jnp.sqrt(jnp.mean(jnp.square(error))) /
                    jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(reference))), 1e-30),
                "all_logits_finite": jnp.all(jnp.isfinite(production)) & jnp.all(jnp.isfinite(reference))}

    return {name: np.asarray(value).tolist() for name, value in jax.device_get(compare(params, tokens)).items()}


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


def finalize_stopped(run: Path, reason: str) -> dict:
    """Seal a stopped free-table or generator fit's best snapshot as its result, without more updates.

    Reads the latest resumable checkpoint's best tables or generator and its controller,
    writes the same best save and completion as a converged fit, and records why it stopped.
    """
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    oracle = recipe["oracle"]
    generator = oracle["parameterization"] == "query_coefficient_generator"
    if not generator and oracle["parameterization"] not in {"free_per_example_query_coefficients",
                                                            "free_per_example_query_key_coefficients"}:
        raise ValueError("Early finalization applies to per-example free-table and generator fits")
    if (run / "completion.json").exists():
        raise FileExistsError("The fit is already complete")
    root = run / "checkpoints"
    latest = _latest(root)
    if latest is None:
        raise FileNotFoundError("No resumable checkpoint to finalize")
    controller = json.loads((latest / "controller.json").read_text())
    query_key = oracle["parameterization"] == "free_per_example_query_key_coefficients"
    if generator:
        # The generator's tree comes from the parent model, which the fit ran unchanged.
        config = parent_config(Path(oracle["parent_config"]))
        model = nnx.eval_shape(lambda: build_model(config.model, nnx.Rngs(params=0, dropout=0)))
        template = [split_generator_parameters(nnx.split(model, nnx.Param)[1])[0]]
    else:
        model = recipe["model"]
        microbatch = int(oracle["microbatch_size"])
        shape = (microbatch, model["num_layers"], recipe["data"]["sequence_length"] - 1,
                 model["num_heads"], model["attention_degree"] + 1)
        leaf = jax.ShapeDtypeStruct(shape, jnp.float32)
        template = [(leaf, leaf) if query_key else leaf for _ in range(int(oracle["sequence_count"]) // microbatch)]
    item = {"best": template}
    with ocp.PyTreeCheckpointer() as checkpointer:
        best = checkpointer.restore(latest / "default", args=ocp.args.PyTreeRestore(
            item=item, partial_restore=True,
            restore_args=ocp.checkpoint_utils.construct_restore_args(item)))["best"]
    if not all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(best)):
        raise FloatingPointError("Stopped fit's best tables are not finite")
    step, best_step = int(controller["step"]), int(controller["best_step"])
    metrics = run / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if int(row["step"]) <= step]
    metrics.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    terminal = rows[-1]
    checkpoint = root / str(best_step)
    final_path = checkpoint / "best"
    if final_path.exists():
        shutil.rmtree(final_path)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(final_path, {"generator_parameters" if generator else "coefficients": best})
    _write_json(checkpoint / "coefficient_identity.json", {"step": best_step,
                "layout": ("model_parameter_tree" if generator else
                           "microbatches of (c, g), each [example,layer,position,head,order]; pair coefficient "
                           "c_r(t) g_r(i)" if query_key else "microbatches of [example,layer,query,head,order]"),
                "coordinates": ("stored_generator_parameters" if generator else
                                "effective_monomial_c and key factor g" if query_key else "effective_monomial_c"),
                "optimizer_coordinates": "stored_generator_parameters" if generator else "factorial(order)*c",
                "parameterization": oracle["parameterization"],
                "parent_run_id": oracle["parent_run_id"], "parent_checkpoint": oracle["parent_checkpoint"]})
    outcome = {"status": "complete", "step": step, "best_step": best_step,
               "initial_nll": controller["initial_loss"], "best_nll": controller["best_loss"],
               "fit_nll": terminal["nll"], "fit_perplexity": math.exp(min(terminal["nll"], 80)),
               "best_fit_nll": controller["best_loss"],
               "nll_reduction": controller["initial_loss"] - controller["best_loss"],
               "learning_rate_reduced_at": controller["reduced_at"], "finished_utc": _utc(),
               "stopping": reason, "interpretation": "Fixed-input coefficient fitting, not held-out generalization.",
               "parameterization": oracle["parameterization"],
               "frozen_parameter_scope": "backbone" if generator else "all_pretrained_parameters"}
    if generator:
        outcome.update(generator_readout=oracle.get("generator_readout", "diagonal"), optimizer_updates_per_step=1,
                       coefficient_sharing="one input-dependent generator per layer/head shared across the fitting batch")
    _write_json(run / "completion.json", outcome)
    _write_json(run / "metadata.json", outcome)
    checkpoint = _retain_completed_best(run, outcome)
    update_record(run, {**capture_artifacts(run, checkpoint_directory=checkpoint), "status": "complete",
                        "step": step, "oracle_outcome": outcome, "retained_best_checkpoint_step": best_step})
    return outcome


def qualify(run: Path, functions, params, tokens, table, optimizer, learning_rate,
            *, graph=None, external_nll_atol=1e-5, external_nll_rtol=1e-5, generator=False,
            shared=False, parent_nll=None, capture_logit_error=None, parent_preserving=False):
    capture, _, train_step, evaluate = functions
    frozen = split_generator_parameters(params)[1] if generator else params
    frozen_sha256 = state_sha256(frozen)
    if shared or (generator and parent_nll is not None):
        # Original capture/replay was checked on every microbatch. A shared
        # table's projection changes the model; generator conversion must not.
        if parent_nll is None or capture_logit_error is None:
            raise ValueError("Batch acceptance requires checked parent capture measurements")
        original_nll = np.asarray(parent_nll)
        logits_error = float(capture_logit_error)
    else:
        _, original_nll, replay_nll, logits_error = capture(params, tokens)
        original_nll, replay_nll, logits_error = jax.device_get((original_nll, replay_nll, logits_error))
        if not np.allclose(original_nll, replay_nll, rtol=1e-5, atol=1e-5) or float(logits_error) > 0.05:
            raise RuntimeError(f"Parent replay parity failed: logits error {logits_error}")
    external_nll = np.asarray(evaluate(table, params, tokens))
    if parent_preserving and not np.allclose(original_nll, external_nll,
                                             rtol=external_nll_rtol, atol=external_nll_atol):
        raise RuntimeError("Exact static initialization does not reproduce the parent NLL")
    repeated_nll = np.asarray(evaluate(table, params, tokens))
    if not np.array_equal(external_nll, repeated_nll):
        raise RuntimeError("Independent external coefficient forward is not repeatable")
    external_diagnostics = None
    if graph is not None:
        print("Measuring independent-graph logit error and external-forward repeatability", flush=True)
        diagnostic_tokens = tokens[0] if tokens.ndim == 3 else tokens
        diagnostic_table = (jnp.swapaxes(expand_shared_coefficients(
            table, diagnostic_tokens, static=table.ndim == 3), 0, 1) if shared else table)
        external_diagnostics = external_replay_diagnostics(
            graph, params, diagnostic_tokens, diagnostic_table, generator=generator)
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
    projected = shared and not parent_preserving
    reference_nll = external_nll if projected else original_nll
    if not np.allclose(reference_nll, before, rtol=external_nll_rtol, atol=external_nll_atol):
        failure = {"status": "failed", "phase": "separate_training_forward_replay", "time_utc": _utc(),
                   "reference_nll": reference_nll.tolist(), "training_forward_nll": before.tolist(),
                   "reference": "projected_coefficients" if projected else "original_parent",
                   "nll_max_absolute_error": float(np.max(np.abs(reference_nll - before))),
                   "nll_rtol": external_nll_rtol, "nll_atol": external_nll_atol}
        _write_json(run / "oracle_acceptance_failure.json", failure)
        print(json.dumps(failure), flush=True)
        raise RuntimeError("Separately compiled training forward does not reproduce its reference NLL")
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
    next_group_norms = ({name: float(value) for name, value in jax.device_get(next_a[5]).items()}
                        if generator else {})
    if generator and not all(np.isfinite(value) and value > 0 for value in next_group_norms.values()):
        raise RuntimeError(f"Generator parameter group has no finite nonzero gradient after one update: "
                           f"{next_group_norms}")
    # Adam need not decrease loss on its first step near a trained solution.
    # Require strict improvement within a bounded, disposable optimization probe,
    # always measured by the identical compiled training forward. These candidate
    # parameters/moments are never returned to the production training loop.
    improvement_limit = 10
    nll_trace = [float(before.mean())]
    candidate = next_a
    for improving_update in range(1, improvement_limit + 1):
        after, candidate_norms, candidate_finite = jax.device_get(candidate[2:5])
        after, candidate_norms = np.asarray(after), np.asarray(candidate_norms)
        if (not bool(candidate_finite) or not np.all(np.isfinite(after))
                or not np.all(np.isfinite(candidate_norms))):
            raise RuntimeError("Nonfinite coefficient loss, gradient, or update during qualification")
        nll_trace.append(float(after.mean()))
        if nll_trace[-1] < nll_trace[0]:
            break
        if improving_update < improvement_limit:
            candidate = train_step(candidate[0], candidate[1], params, tokens, learning_rate)
    else:
        raise RuntimeError(f"Coefficient updates did not improve NLL within {improvement_limit} updates: {nll_trace}")
    if state_sha256(split_generator_parameters(params)[1] if generator else params) != frozen_sha256:
        raise RuntimeError("Qualification changed the frozen parameter tree")
    return {"status": "passed", "completed_utc": _utc(), "parent_nll": float(original_nll.mean()),
            "separate_training_forward_nll": float(before.mean()),
            "external_evaluation_nll": float(external_nll.mean()),
            "external_evaluation_max_nll_error": (None if projected else
                                                  float(np.max(np.abs(external_nll - original_nll)))),
            "training_forward_max_nll_error": float(np.max(np.abs(before - reference_nll))),
            "training_forward_reference": "projected_coefficients" if projected else "original_parent",
            "projected_initial_nll": float(external_nll.mean()) if projected else None,
            "projection_nll_change": float(external_nll.mean() - original_nll.mean()) if projected else None,
            "cross_compilation_nll_atol": external_nll_atol, "cross_compilation_nll_rtol": external_nll_rtol,
            "external_forward_repeated_exactly": True,
            "independent_graph_diagnostics": external_diagnostics,
            "one_update_nll": nll_trace[1], "improved_training_nll": nll_trace[-1],
            "improving_update": improving_update, "qualification_update_limit": improvement_limit,
            "qualification_nll_trace": nll_trace,
            "max_logit_replay_error": float(logits_error),
            "gradient_norm_min": float(norms.min()), "gradient_norm_max": float(norms.max()),
            "checkpoint_resume_exact": True, "frozen_parameter_sha256": frozen_sha256,
            "generator_gradient_norms": group_norms,
            "generator_gradient_norms_after_one_update": next_group_norms,
            **({"trainable_parameter_names": sorted(group_norms),
                "frozen_parameter_names": sorted(".".join(map(str, path))
                                                 for path, _ in nnx.to_flat_state(frozen))}
               if generator else {}),
            "frozen_parameter_scope": "backbone" if generator else "all_pretrained_parameters",
            "device": jax.devices()[0].device_kind}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--finalize-stopped", metavar="REASON",
                        help="seal a stopped free-table or generator fit's best snapshot as its result")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if args.finalize_stopped:
        print(json.dumps(finalize_stopped(run, args.finalize_stopped)), flush=True)
        return
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    oracle = recipe["oracle"]
    parameterization = oracle.get("parameterization", "free_per_example_query_coefficients")
    if parameterization not in {"free_per_example_query_coefficients", "query_coefficient_generator",
                                "shared_batch_query_coefficients", "static_coefficients",
                                "free_per_example_query_key_coefficients"}:
        raise ValueError(f"Unknown coefficient parameterization: {parameterization}")
    query_key = parameterization == "free_per_example_query_key_coefficients"
    generator = parameterization == "query_coefficient_generator"
    static = parameterization == "static_coefficients"
    shared = parameterization == "shared_batch_query_coefficients" or static
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
               "coefficient_layout": ("model_parameter_tree" if generator else
                                      "layer,head,order" if static else
                                      "layer,query,head,order" if shared else
                                      "(c, g), each example,layer,position,head,order" if query_key else
                                      "example,layer,query,head,order"),
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
    exact_static = static and oracle.get("static_initialization") == "checkpoint"
    if exact_static and config.model.attention_coefficient_conditioning != "static":
        raise ValueError("Exact static initialization requires a static parent checkpoint")
    conditioning = config.model.attention_coefficient_conditioning
    if generator and (conditioning not in ("query", "query_key")
                      or config.model.attention_coefficient_activation != "tanh"
                      or config.model.attention_coefficients != "learned"):
        raise ValueError("Generator fitting requires a learned tanh query or query-key model")
    # A query-key parent keeps its own trained key generator; nothing is added or reset.
    native_query_key = generator and conditioning == "query_key"
    if native_query_key and recipe["model"] != yaml.safe_load(source_config.read_text())["model"]:
        raise ValueError("A query-key generator fit runs the parent's model unchanged")
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
    if query_key and (config.model.attention_coefficient_conditioning != "static"
                      or config.model.attention_coefficients != "fixed"):
        raise ValueError("Free query-key fitting starts from a fixed Taylor parent")
    print("Restoring frozen parent parameters", flush=True)
    graph, params = restore_parent(config, Path(oracle["parent_checkpoint"]))
    conversion_compare, conversion, original_graph, original_params = None, None, graph, params
    if query_key:
        # The same parameters on the parent model with its query-key kernel backend;
        # capture still runs the unchanged parent graph.
        backend = oracle["fit_backend"]
        query_key_config = replace(config, model=replace(config.model, attention_backend=backend))
        query_key_graph, query_key_params = restore_parent(query_key_config, Path(oracle["parent_checkpoint"]))
        if state_sha256(query_key_params) != state_sha256(params):
            raise RuntimeError("The query-key graph's parameters differ from the parent's")
        del query_key_params
        runtime.update(fit_attention_backend=backend)
    if generator:
        readout = recipe["model"].get("attention_coefficient_readout", "diagonal")
        if oracle.get("generator_readout", readout) != readout:
            raise ValueError("Generator readout differs between oracle and model recipe")
        original_readout = getattr(config.model, "attention_coefficient_readout", "diagonal")
        graph, params = convert_generator_readout(config, graph, params, readout)
        runtime.update(generator_readout=readout, optimizer_updates_per_step=1,
                       generator_shared_across_sequences=True,
                       optimizer_initialization="fresh zero moments", generator_conditioning=conditioning)
        if native_query_key:
            trainable = sorted({str(path[-1]) for path, _ in
                                nnx.to_flat_state(split_generator_parameters(params)[0])})
            if trainable != sorted(oracle["trainable_parameters"]) or "key_polynomial_kernel" not in trainable:
                raise ValueError(f"Query-key generator parameters {trainable} differ from the recipe's")
        if readout != original_readout:
            conversion_compare = make_conversion_diagnostics(original_graph, graph)
            conversion = {"source_readout": original_readout, "target_readout": readout,
                          "initialization": "V=diag(trained amplitude); W and b unchanged",
                          "checked_sequence_count": 0,
                          "coefficient_max_absolute_error": 0., "coefficient_relative_l2_error": 0.,
                          "logit_max_absolute_error": 0., "nll_max_absolute_error": 0.}
    print("Parent restored; compiling coefficient capture and replay", flush=True)
    functions = (make_generator_functions(graph, oracle) if generator else
                 make_shared_functions(graph, config.model.attention_degree, oracle) if shared else
                 make_query_key_functions(graph, query_key_graph, config.model.attention_degree, oracle)
                 if query_key else make_functions(graph, config.model.attention_degree, oracle))
    capture, initialize_optimizer, train_step, evaluate = functions
    capture_evaluate = (make_functions(graph, config.model.attention_degree, oracle)[3]
                        if shared else evaluate)
    token_chunks = [jax.device_put(batch[i:i + microbatch]) for i in range(0, count, microbatch)]
    tables, optimizers, parent_losses = [], [], []
    original_parent_losses = []
    shared_table_sum, static_parent_table, maximum_capture_error = None, None, 0.0
    for index, tokens in enumerate(token_chunks):
        table, losses, replay_losses, error = capture(params, tokens)
        losses, replay_losses, error = jax.device_get((losses, replay_losses, error))
        if query_key:
            # Key factors start at one: the pair coefficient is the parent's c_r.
            # The external replay below then checks the query-key backend against the parent.
            table = (table, jnp.ones_like(table))
        external_replay_losses = np.asarray(capture_evaluate(table, params, tokens))
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
        if conversion_compare is not None:
            measured = {key: np.asarray(value) for key, value in jax.device_get(
                conversion_compare(original_params, params, tokens)).items()}
            conversion_error = float(np.max(np.abs(measured["original_nll"] - measured["converted_nll"])))
            if (not bool(measured["all_finite"])
                    or not np.allclose(measured["original_nll"], measured["converted_nll"],
                                       rtol=external_nll_rtol, atol=external_nll_atol)
                    or float(measured["coefficient_relative_l2_error"]) > 1e-6
                    or float(measured["logit_max_absolute_error"]) > .05):
                failure = {"status": "failed", "phase": "generator_readout_conversion",
                           "microbatch_index": index,
                           **{key: value.tolist() for key, value in measured.items()}}
                _write_json(run / "oracle_acceptance_failure.json", failure)
                raise RuntimeError("Dense generator conversion does not reproduce the original parent")
            for key in ("coefficient_max_absolute_error", "coefficient_relative_l2_error", "logit_max_absolute_error"):
                conversion[key] = max(conversion[key], float(measured[key]))
            conversion["nll_max_absolute_error"] = max(conversion["nll_max_absolute_error"], conversion_error)
            conversion["checked_sequence_count"] += tokens.shape[0]
            # Qualification compares the training graph to the true parent,
            # rather than only comparing the converted model to itself.
            losses = measured["original_nll"]
        if shared:
            if exact_static:
                captured_static = exact_static_coefficients(table)
                if static_parent_table is not None and not bool(jnp.array_equal(captured_static, static_parent_table)):
                    raise ValueError("Static checkpoint coefficients differ across captured microbatches")
                static_parent_table = captured_static
            else:
                table_sum = table.sum(axis=(0, 2) if static else 0)
                shared_table_sum = table_sum if shared_table_sum is None else shared_table_sum + table_sum
                # Capture storage never grows with the fitting batch size.
                shared_table_sum.block_until_ready()
        elif not generator or index == 0:
            tables.append(table)
            optimizers.append(initialize_optimizer(table))
        parent_losses.extend(external_replay_losses.tolist())
        original_parent_losses.extend(losses.tolist())
        maximum_capture_error = max(maximum_capture_error, float(error))
        if index % 4 == 0 or index + 1 == len(token_chunks):
            print(f"Captured coefficients for microbatch {index + 1}/{len(token_chunks)}; external replay NLL {np.mean(external_replay_losses):.6f}", flush=True)
    reference_agreement = None
    if native_query_key:
        # Capture replays the production graph against itself; this checks the restored
        # parent against an independently implemented attention backend.
        print("Comparing the restored forward with the blocked reference backend", flush=True)
        reference_config = replace(config, model=replace(config.model, attention_backend="blocked"))
        reference_model = nnx.eval_shape(lambda: build_model(
            reference_config.model, nnx.Rngs(params=config.train.seed, dropout=config.train.seed)))
        for _, module in nnx.iter_graph(reference_model):
            if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
                module.dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
        reference_agreement = reference_backend_agreement(
            graph, nnx.split(reference_model, nnx.Param)[0], params, token_chunks[0])
        reference_error = float(np.max(np.abs(np.asarray(reference_agreement["production_nll"])
                                              - np.asarray(reference_agreement["reference_nll"]))))
        reference_agreement.update(reference_backend="blocked", sequence_count=int(token_chunks[0].shape[0]),
                                   nll_max_absolute_error=reference_error, nll_atol=external_nll_atol)
        print("Reference backend agreement: " + json.dumps(reference_agreement), flush=True)
        if not reference_agreement["all_logits_finite"] or reference_error > external_nll_atol:
            _write_json(run / "oracle_acceptance_failure.json",
                        {"status": "failed", "phase": "reference_backend_agreement", "time_utc": _utc(),
                         **_json_safe(reference_agreement)})
            raise RuntimeError("The restored parent's forward disagrees with the blocked reference backend")
    if generator and count > 1:
        # Keep one generator and optimizer for the full batch. A scan holds one
        # microbatch's activations at a time and returns one mean-gradient update.
        token_chunks = [jnp.stack(token_chunks)]
    if conversion is not None:
        conversion["status"] = "passed"
        print("Generator readout conversion: " + json.dumps(conversion), flush=True)
    if shared:
        projection_count = count * (batch.shape[1] - 1) if static else count
        tables = [static_parent_table if exact_static else shared_table_sum / projection_count]
        optimizers = [initialize_optimizer(tables[0])]
        token_chunks = [jnp.stack(token_chunks)]
        projected_losses = np.asarray(evaluate(tables[0], params, token_chunks[0]))
        if not np.all(np.isfinite(projected_losses)):
            raise RuntimeError("Batch-mean coefficient projection produced nonfinite NLL")
        projection = {"method": ("exact effective static coefficients from the parent checkpoint" if exact_static else
                                 "arithmetic mean of original effective coefficients across all fitting sequences and query positions"
                                 if static else "arithmetic mean of original effective coefficients across all fitting sequences"),
                      "sequence_count": count, "coefficient_shape": list(tables[0].shape),
                      "parent_nll": float(np.mean(original_parent_losses)),
                      "parent_external_replay_nll": float(np.mean(parent_losses)),
                      "projected_initial_nll": float(projected_losses.mean()),
                      "parent_per_sequence_nll": original_parent_losses,
                      "projected_per_sequence_nll": projected_losses.tolist(),
                      "maximum_capture_logit_error": maximum_capture_error,
                      "interpretation": ("Initialization preserves the parent static coefficients exactly." if exact_static else
                                         "Projection changes the model; fitting improvement starts after projection.")}
        _write_json(run / "coefficient_projection.json", projection)
        runtime.update(original_parent_nll=projection["parent_nll"],
                       projected_initial_nll=projection["projected_initial_nll"],
                       coefficient_shape=projection["coefficient_shape"],
                       optimizer_updates_per_step=1)
        print("Shared coefficient initialization: " + json.dumps({
            key: value for key, value in projection.items() if "per_sequence" not in key}), flush=True)
        parent_losses = projected_losses.tolist()
    learning_rate = float(oracle["learning_rate"])
    runtime.update(trainable_parameter_count=sum(int(value.size) for value in jax.tree.leaves(tables)),
                   microbatch_size=microbatch, resolved_attention_backend=config.model.attention_backend)
    print("Qualifying coefficient-only gradient, improvement, and checkpoint resume", flush=True)
    acceptance = qualify(run, functions, params, token_chunks[0], tables[0], optimizers[0], learning_rate,
                         graph=None if query_key else graph,
                         external_nll_atol=external_nll_atol, external_nll_rtol=external_nll_rtol,
                         generator=generator, shared=shared,
                         parent_nll=original_parent_losses if shared or generator else None,
                         capture_logit_error=maximum_capture_error if shared or generator else None,
                         parent_preserving=exact_static)
    if generator:
        acceptance.update(generator_readout=runtime["generator_readout"],
                          generator_shared_across_sequences=True, optimizer_updates_per_step=1,
                          sequence_count=count, microbatch_size=microbatch,
                          optimizer_initialization="fresh zero moments", generator_conditioning=conditioning)
        if reference_agreement is not None:
            acceptance["reference_backend_agreement"] = reference_agreement
        if conversion is not None:
            acceptance["readout_conversion"] = conversion
    print(f"Qualification passed: training NLL {acceptance['separate_training_forward_nll']:.6f} "
          f"-> {acceptance['improved_training_nll']:.6f} after {acceptance['improving_update']} "
          "disposable qualification update(s)", flush=True)
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
            if shared:
                terminal["gradient_norm"] = float(norms[0])
            stream.write(json.dumps(terminal, allow_nan=False) + "\n")
            wandb_metrics = {"train/nll": loss, "train/perplexity": terminal["perplexity"],
                             "train/best_nll": controller["best_loss"], "train/learning_rate": learning_rate,
                             "train/first_sequence_nll": float(losses[0]), "train/step_seconds": terminal["seconds"]}
            if generator:
                wandb_metrics["train/gradient_norm"] = float(norms[0])
                wandb_metrics.update({f"attention/{name.split('_')[0]}_coefficients/{name}/gradient_norm": value
                                      for name, value in terminal["generator_gradient_norms"].items()})
            if shared:
                wandb_metrics["train/gradient_norm"] = float(norms[0])
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
    if generator:
        outcome.update(generator_readout=runtime["generator_readout"],
                       coefficient_sharing="one input-dependent generator per layer/head shared across the fitting batch",
                       optimizer_updates_per_step=1)
    if shared:
        outcome.update(original_parent_nll=projection["parent_nll"],
                       projected_initial_evaluation_nll=projection["projected_initial_nll"],
                       coefficient_sharing=("one table shared across every fitting sequence and query position" if static else
                                            "one table shared across every fitting sequence"),
                       optimizer_updates_per_step=1)
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
                    "layout": ("model_parameter_tree" if generator else
                               "one [layer,head,order] table shared across all sequences and query positions" if static else
                               "one [layer,query,head,order] table shared across the batch" if shared else
                               "microbatches of (c, g), each [example,layer,position,head,order]; pair coefficient c_r(t) g_r(i)"
                               if query_key else "microbatches of [example,layer,query,head,order]"),
                    "coordinates": ("stored_generator_parameters" if generator else
                                    "effective_monomial_c and key factor g" if query_key else "effective_monomial_c"),
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
