"""Supervised fitting of query-key coefficient generators onto free pair coefficients.

The teacher is a free query-key fit: per-sequence query coefficients c and key
factors g, whose pair coefficient for query t and key i is c_r(t) g_r(i). An example
is one causal (t, i <= t) pair of one sequence. Inputs are the recorded attention
inputs of the query and key tokens, [layer, pair, hidden] each, captured at the
attention module's entry during the teacher's own forward pass. Targets are
factorial-coordinate pair coefficients r! c_r(t) g_r(i), [layer, pair, head, order].

The student is the native query-key generator: c_r(x) = [b~_r + a_r tanh(w_r x)] / r!
(stored bias b~ = r! b) and key factors f_r(x) = 1 + a^k_r tanh(w^k_r x), with pair
coefficient c_r(x_t) f_r(x_i). Each layer/head is an independent regression with its
own gradient clipping, as for the dense query regression.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorstore as ts

from scaffolds.transformer.attention.input_dependent import project_coefficients, project_key_factors
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_distillation_model import _sequence_nll
from scaffolds.transformer.training.coefficient_distillation_regression import (
    clip_network_gradients, regression_metrics,
)
from scaffolds.transformer.training.coefficient_fit import split_generator_parameters

PACKED_NAMES = {
    "kernel": "query_polynomial_kernel",
    "bias": "query_polynomial_bias",
    "amplitude": "query_polynomial_amplitude",
    "key_kernel": "key_polynomial_kernel",
    "key_amplitude": "key_polynomial_amplitude",
}


def query_key_table_chunks(checkpoint: Path, recipe: dict):
    """Yield (first sequence, c[B,L,T,H,R], g[B,L,T,H,R]) from a saved free query-key fit."""
    directory = Path(checkpoint) / "best"
    metadata = json.loads((directory / "_METADATA").read_text())
    if not metadata.get("use_ocdbt") or metadata.get("use_zarr3"):
        raise ValueError("Expected OCDBT / Zarr-v2 free-table storage")
    arrays = set()
    for entry in metadata["tree_metadata"].values():
        keys = [item["key"] for item in entry["key_metadata"]]
        if len(keys) != 3 or keys[0] != "coefficients" or keys[2] not in ("0", "1"):
            raise ValueError("Unexpected free query-key checkpoint structure")
        arrays.add((int(keys[1]), int(keys[2])))
    count = len(arrays) // 2
    if sorted(arrays) != [(index, part) for index in range(count) for part in (0, 1)]:
        raise ValueError("Free query-key microbatches are not contiguous pairs")
    model = recipe["model"]
    tail = [model["num_layers"], recipe["data"]["sequence_length"] - 1,
            model["num_heads"], model["attention_degree"] + 1]
    offset = 0

    def read(index, part):
        array = ts.open({"driver": "zarr", "kvstore": {
            "driver": "ocdbt", "base": {"driver": "file", "path": str(directory) + "/"},
            "path": f"coefficients.{index}.{part}",
        }}, open=True, read=True).result()
        values = np.asarray(array.read().result())
        if list(values.shape[1:]) != tail or not np.isfinite(values).all():
            raise ValueError("Invalid free query-key table")
        return values

    for index in range(count):
        coefficients, key_factors = read(index, 0), read(index, 1)
        if coefficients.shape != key_factors.shape:
            raise ValueError("Query coefficients and key factors differ in shape")
        yield offset, coefficients, key_factors
        offset += coefficients.shape[0]
    if offset != recipe["oracle"]["sequence_count"]:
        raise ValueError("The checkpoint does not contain its full fitting batch")


def make_teacher_functions(graph):
    """Return (capture, native_losses) for the free query-key teacher on ``graph``.

    Both accept (parent_params, tokens[B,T+1], c[B,L,T,H,R], g[B,L,T,H,R]); ``graph``
    is the Taylor parent with a query-key kernel backend. Capture records each layer's
    attention input [B,L,T,D] during the actual teacher forward.
    """
    native_call = MultiHeadAttention.__call__

    def teacher(params, tokens, coefficients, key_factors):
        return nnx.merge(graph, params)(
            tokens[:, :-1], deterministic=True, query_coefficients=jnp.swapaxes(coefficients, 0, 1),
            key_factors=jnp.swapaxes(key_factors, 0, 1))

    @jax.jit
    def native_losses(params, tokens, coefficients, key_factors):
        return _sequence_nll(teacher(params, tokens, coefficients, key_factors), tokens)

    @jax.jit
    def capture(params, tokens, coefficients, key_factors):
        captured = []

        def observe(module, x, *args, **kwargs):
            captured.append(x)
            return native_call(module, x, *args, **kwargs)

        with patch.object(MultiHeadAttention, "__call__", observe):
            observed_logits = teacher(params, tokens, coefficients, key_factors)
        if len(captured) != coefficients.shape[1]:
            raise ValueError("Teacher capture did not visit every layer exactly once")
        native_logits = teacher(params, tokens, coefficients, key_factors)
        difference = observed_logits.astype(jnp.float32) - native_logits.astype(jnp.float32)
        return {"inputs": jnp.stack(captured, axis=1), "nll": _sequence_nll(observed_logits, tokens),
                "joint_native_nll": _sequence_nll(native_logits, tokens),
                "joint_logit_max_absolute_error": jnp.max(jnp.abs(difference))}

    return capture, native_losses


def build_student(parent_config, parent_params, *, backend: str, projection_initial_std=None):
    """Add a Taylor-initialized query-key generator to a fixed Taylor parent.

    Returns (student Config, graph definition, full parameter state, identity audit). The
    stored query bias starts at one (1/r! effective), both amplitudes at zero, and both
    projections at the 4B reference draw N(0, 1/hidden_size); every backbone leaf is the parent's.
    """
    model = parent_config.model
    if (model.attention_coefficients != "fixed" or model.attention_coefficient_conditioning != "static"
            or model.attention_fixed_coefficients is not None or model.attention_normalization
            or not model.attention_degree or model.remat):
        raise ValueError("Query-key distillation requires a fixed, unnormalized Taylor parent without remat")
    std = model.hidden_size ** -.5 if projection_initial_std is None else projection_initial_std
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Zero-amplitude factors need nonzero finite projections")
    config = deepcopy(parent_config)
    config.model = replace(model, attention_backend=backend,
        attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query_key", attention_coefficient_activation="tanh",
        attention_coefficient_readout="diagonal", attention_coefficient_amplitude_initial=0.0,
        attention_coefficient_projection_initial_std=std, attention_coefficient_tanh_bias=False,
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_initialization="taylor")
    student = build_model(config.model, nnx.Rngs(params=config.train.seed, dropout=config.train.seed))
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
    for _, module in nnx.iter_graph(student):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    graph, template = nnx.split(student, nnx.Param)
    generator, frozen_template = split_generator_parameters(template)
    expected = {path: value.shape for path, value in nnx.to_flat_state(frozen_template)}
    actual = {path: value.shape for path, value in nnx.to_flat_state(parent_params)}
    if actual != expected:
        raise ValueError("Query-key student backbone tree differs from the frozen Taylor parent")
    params = nnx.State.merge(parent_params, generator)
    packed = pack_generator(params)
    if (not bool(jnp.all(packed["bias"] == 1)) or not bool(jnp.all(packed["amplitude"] == 0))
            or not bool(jnp.all(packed["key_amplitude"] == 0))):
        raise AssertionError("New generator does not represent exact Taylor coefficients")
    parent_hash = state_sha256(parent_params)
    if parent_hash != state_sha256(split_generator_parameters(params)[1]):
        raise AssertionError("Constructing the query-key student changed its backbone")
    audit = {"parent_parameters_sha256": parent_hash, "backbone_parameters_sha256": parent_hash,
             "generator_parameter_count": sum(int(np.prod(value.shape)) for value in packed.values()),
             "projection_initial_std": float(std), "backend": backend,
             "initialization": "stored bias one (Taylor), zero query and key amplitudes, reference random projections"}
    return config, graph, params, audit


def pack_generator(params):
    """Native query-key generator arrays with leading [layer, head] axes."""
    generator, _ = split_generator_parameters(params)
    leaves = {str(path[-1]): value[...] for path, value in nnx.to_flat_state(generator)}
    if set(leaves) != set(PACKED_NAMES.values()):
        raise ValueError("Expected exactly a diagonal query-key generator")
    result = {key: leaves[name] for key, name in PACKED_NAMES.items()}
    if result["kernel"].ndim != 4 or result["key_kernel"].ndim != 4 or result["bias"].ndim != 3:
        raise ValueError("Query-key distillation requires distinct parameters per layer and head")
    return result


def replace_generator(params, packed):
    """Return a full native state with only its generator arrays replaced."""
    expected = pack_generator(params)
    if set(packed) != set(expected) or any(packed[key].shape != expected[key].shape for key in expected):
        raise ValueError("Replacement generator differs from the native model")
    replacements = {PACKED_NAMES[key]: value for key, value in packed.items()}
    return nnx.from_flat_state([
        (path, value.replace(replacements[str(path[-1])]) if str(path[-1]) in replacements else value)
        for path, value in nnx.to_flat_state(params)
    ])


def predict_pair_factorial(generator, query_inputs, key_inputs):
    """r! c_r(x_t) f_r(x_i) [L,P,H,R] with the native FP32 coefficient and key-factor projections."""
    if query_inputs.shape != key_inputs.shape or query_inputs.ndim != 3:
        raise ValueError("Pair inputs need matching [layer, pair, hidden] axes")
    factors = jnp.asarray([math.factorial(r) for r in range(generator["bias"].shape[-1])], jnp.float32)

    def layer(xq, xk, kernel, bias, amplitude, key_kernel, key_amplitude):
        # As in the model: the stored bias receives its one factorial division first.
        coefficients = project_coefficients(xq[None], kernel, bias.astype(jnp.float32) / factors,
                                            amplitude=amplitude, activation="tanh", amplitude_scale="factorial")[0]
        return coefficients * project_key_factors(xk[None], key_kernel, key_amplitude)[0] * factors

    return jax.vmap(layer)(query_inputs, key_inputs, *(generator[key] for key in PACKED_NAMES))


def sample_pairs(key, *, sequence_count: int, length: int, size: int):
    """Uniform causal pairs: query and key token indices into [sequence * length + position]."""
    sequence_key, pair_key = jax.random.split(key)
    sequence = jax.random.randint(sequence_key, (size,), 0, sequence_count)
    pair = jax.random.randint(pair_key, (size,), 0, length * (length + 1) // 2)
    # Invert pair = t (t + 1) / 2 + i with 0 <= i <= t, in integer arithmetic after a float estimate.
    query = jnp.floor((jnp.sqrt(8.0 * pair.astype(jnp.float32) + 1.0) - 1.0) / 2.0).astype(jnp.int32)
    query = jnp.where(query * (query + 1) // 2 > pair, query - 1, query)
    query = jnp.where((query + 1) * (query + 2) // 2 <= pair, query + 1, query)
    position = pair - query * (query + 1) // 2
    return sequence * length + query, sequence * length + position


def make_regression_functions(*, beta1=.9, beta2=.95, epsilon=1e-8, clip_norm=1.0):
    """Return ``init(generator), step(...), evaluate(generator, xq, xk, targets)``.

    ``step(generator, optimizer, xq, xk, targets, learning_rate)`` returns the new generator,
    optimizer, and metrics of the pre-update prediction. Targets are factorial pair coefficients.
    """
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    tx = optax.scale_by_adam(b1=beta1, b2=beta2, eps=epsilon)

    def loss(generator, query_inputs, key_inputs, targets):
        prediction = predict_pair_factorial(generator, query_inputs, key_inputs)
        if prediction.shape != targets.shape:
            raise ValueError("Targets must have shape [L,P,H,R]")
        per_network = jnp.mean(jnp.square(prediction - targets), axis=(1, 3))
        return jnp.sum(per_network), prediction

    @jax.jit
    def step(generator, optimizer, query_inputs, key_inputs, targets, learning_rate):
        (_, prediction), gradient = jax.value_and_grad(loss, has_aux=True)(generator, query_inputs, key_inputs, targets)
        clipped, norms = clip_network_gradients(gradient, clip_norm)
        updates, optimizer = tx.update(clipped, optimizer, generator)
        candidate = optax.apply_updates(generator, jax.tree.map(lambda value: -learning_rate * value, updates))
        metrics = regression_metrics(prediction, targets)
        metrics.update(gradient_norm_mean=jnp.mean(norms), gradient_norm_max=jnp.max(norms),
                       clipping_fraction=jnp.mean(norms > clip_norm),
                       finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(
                           (candidate, optimizer, gradient, prediction, targets))])))
        return candidate, optimizer, metrics

    @jax.jit
    def evaluate(generator, query_inputs, key_inputs, targets):
        return regression_metrics(predict_pair_factorial(generator, query_inputs, key_inputs), targets)

    return jax.jit(tx.init), step, evaluate
