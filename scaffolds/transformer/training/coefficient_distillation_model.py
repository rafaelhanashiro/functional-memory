"""Native model adapters for supervised polynomial-coefficient distillation.

Teacher inputs are captured at the unchanged attention module's entry, after the
block's pre-attention normalization. The capture observes the actual free-table
forward, not a separately reconstructed layer loop. Deployment uses an ordinary
query-conditioned model and never receives the teacher's activations or table.
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

from scaffolds.transformer.attention.input_dependent import project_coefficients
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import split_generator_parameters


_PACKED_NAMES = {
    "kernel": "query_polynomial_kernel",
    "bias": "query_polynomial_bias",
    "mixing": "query_polynomial_mixing",
    "tanh_bias": "query_polynomial_tanh_bias",
}


def coefficient_chunks(checkpoint: Path, recipe: dict):
    """Yield (first sequence, effective c[B,L,T,H,R]) from a saved free fit.

    The reader preserves saved microbatches and never restores teacher Adam state.
    Record/config/checkpoint hashes are checked by the experiment launcher.
    """
    directory = Path(checkpoint) / "best"
    metadata = json.loads((directory / "_METADATA").read_text())
    if not metadata.get("use_ocdbt") or metadata.get("use_zarr3"):
        raise ValueError("Expected OCDBT / Zarr-v2 free-coefficient storage")
    arrays = []
    for entry in metadata["tree_metadata"].values():
        keys = entry["key_metadata"]
        if len(keys) != 2 or keys[0]["key"] != "coefficients":
            raise ValueError("Unexpected free-table checkpoint structure")
        arrays.append((int(keys[1]["key"]), entry["value_metadata"]["write_shape"]))
    arrays.sort()
    if [index for index, _ in arrays] != list(range(len(arrays))):
        raise ValueError("Coefficient microbatch indices are not contiguous")
    model = recipe["model"]
    tail = [model["num_layers"], recipe["data"]["sequence_length"] - 1,
            model["num_heads"], model["attention_degree"] + 1]
    offset = 0
    for index, shape in arrays:
        if shape[1:] != tail:
            raise ValueError("Free-table shape differs from its frozen recipe")
        array = ts.open({"driver": "zarr", "kvstore": {
            "driver": "ocdbt", "base": {"driver": "file", "path": str(directory) + "/"},
            "path": f"coefficients.{index}",
        }}, open=True, read=True).result()
        values = np.asarray(array.read().result())
        if list(values.shape) != shape or not np.isfinite(values).all():
            raise ValueError("Invalid free coefficient table")
        yield offset, values
        offset += shape[0]
    if offset != recipe["oracle"]["sequence_count"]:
        raise ValueError("The checkpoint does not contain its full fitting batch")


def build_dense_student(parent_config, parent_params, *, feature_count=None,
                        projection_initial_std=None, readout_scale=1.0, tanh_bias=False):
    """Add a Taylor-initialized dense generator without changing any backbone leaf.

    Returns (student Config, graph definition, full parameter state, identity audit).
    The caller specifies the historically matched feature count and readout scale.
    All model fields outside the generator retain the parent's exact values.
    """
    model = parent_config.model
    if (model.attention_coefficients != "fixed"
            or model.attention_coefficient_conditioning != "static"
            or model.attention_fixed_coefficients is not None
            or model.attention_fixed_square_factors is not None
            or model.attention_coefficient_parameterization != "direct"
            or model.attention_normalization or not model.attention_degree
            or model.remat):
        raise ValueError("Distillation requires a fixed, unnormalized Taylor parent without remat")
    std = model.hidden_size ** -.5 if projection_initial_std is None else projection_initial_std
    if not math.isfinite(std) or std <= 0:
        raise ValueError("A zero-mixing dense initialization needs nonzero finite features")
    config = deepcopy(parent_config)
    config.model = replace(model,
        attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_readout="dense", attention_coefficient_feature_count=feature_count,
        attention_coefficient_readout_scale=readout_scale,
        attention_coefficient_amplitude_initial=0.0,
        attention_coefficient_projection_initial_std=std,
        attention_coefficient_tanh_bias=tanh_bias,
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_initialization="taylor")
    student = build_model(config.model,
                          nnx.Rngs(params=config.train.seed, dropout=config.train.seed))
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
    for _, module in nnx.iter_graph(student):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    graph, template = nnx.split(student, nnx.Param)
    generator, frozen_template = split_generator_parameters(template)
    expected = {path: value.shape for path, value in nnx.to_flat_state(frozen_template)}
    actual = {path: value.shape for path, value in nnx.to_flat_state(parent_params)}
    if actual != expected:
        raise ValueError("Dense student backbone tree differs from the frozen Taylor parent")
    params = nnx.State.merge(parent_params, generator)
    packed = pack_generator(params)
    if not bool(jnp.all(packed["bias"] == 1)) or not bool(jnp.all(packed["mixing"] == 0)):
        raise AssertionError("New generator does not represent exact Taylor coefficients")
    parent_hash = state_sha256(parent_params)
    frozen_hash = state_sha256(split_generator_parameters(params)[1])
    if parent_hash != frozen_hash:
        raise AssertionError("Constructing the dense student changed its backbone")
    audit = {
        "parent_parameters_sha256": parent_hash,
        "backbone_parameters_sha256": frozen_hash,
        "generator_parameter_count": sum(int(np.prod(value.shape)) for value in packed.values()),
        "feature_count": int(packed["kernel"].shape[-1]),
        "readout_scale": float(readout_scale),
        "projection_initial_std": float(std),
        "tanh_bias": bool(tanh_bias),
        "initialization": "Taylor bias in factorial coordinates; zero mixing; reference random features",
    }
    return config, graph, params, audit


def pack_generator(params):
    """Extract native dense arrays with leading [layer,head] axes, without copies."""
    generator, _ = split_generator_parameters(params)
    leaves = {str(path[-1]): value[...] for path, value in nnx.to_flat_state(generator)}
    if "query_polynomial_mixing" not in leaves:
        raise ValueError("Expected a dense generator")
    if set(leaves) - set(_PACKED_NAMES.values()):
        raise ValueError("Unexpected dense-generator parameter")
    result = {key: leaves[name] for key, name in _PACKED_NAMES.items() if name in leaves}
    if result["kernel"].ndim != 4 or result["bias"].ndim != 3 or result["mixing"].ndim != 4:
        raise ValueError("Dense distillation requires distinct parameters per layer and head")
    return result


def replace_generator(params, packed):
    """Return a full native state with only its generator arrays replaced."""
    expected = pack_generator(params)
    if set(packed) != set(expected):
        raise ValueError("Replacement generator fields differ from the native model")
    for key in expected:
        if packed[key].shape != expected[key].shape:
            raise ValueError(f"Replacement generator shape changed: {key}")
    replacements = {_PACKED_NAMES[key]: value for key, value in packed.items()}
    return nnx.from_flat_state([
        (path, value.replace(replacements[str(path[-1])])
         if str(path[-1]) in replacements else value)
        for path, value in nnx.to_flat_state(params)
    ])


def predict_coefficients(packed, inputs, *, readout_scale=1.0):
    """Native effective c[L,N,H,R] for recorded attention inputs x[L,N,D].

    The caller may multiply c by r! to obtain factorial coordinates. Using the
    production projection preserves its output-order scaling and FP32 arithmetic.
    """
    if inputs.ndim != 3 or inputs.shape[0] != packed["kernel"].shape[0]:
        raise ValueError("Recorded inputs need [layer,sample,hidden] axes")
    factors = jnp.asarray([math.factorial(r) for r in range(packed["bias"].shape[-1])], jnp.float32)

    def layer(x, kernel, bias, mixing, inner=None):
        return project_coefficients(
            x[None], kernel, bias / factors, mixing=mixing, activation="tanh",
            tanh_bias=inner, amplitude_scale="factorial", readout_scale=readout_scale)[0]

    args = (inputs, packed["kernel"], packed["bias"], packed["mixing"])
    if "tanh_bias" in packed:
        return jax.vmap(layer)(*args, packed["tanh_bias"])
    return jax.vmap(layer)(*args)


def _sequence_nll(logits, tokens):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)


def make_teacher_functions(graph):
    """Return (capture, native_losses), independently JIT-compiled.

    Both accept (parent_params, tokens[B,T+1], coefficients[B,L,T,H,R]).
    Capture returns inputs[B,L,T,D], per-sequence NLL, and within-compilation
    native/capture parity. Its NLL must also be compared with native_losses,
    because exporting activations can change compiler arithmetic on BF16 graphs.
    The temporary Python hook applies only while JAX traces the capture graph;
    callers must not trace this factory concurrently in multiple threads.
    """
    native_call = MultiHeadAttention.__call__

    def teacher(params, tokens, coefficients):
        return nnx.merge(graph, params)(
            tokens[:, :-1], deterministic=True,
            query_coefficients=jnp.swapaxes(coefficients, 0, 1))

    @jax.jit
    def native_losses(params, tokens, coefficients):
        return _sequence_nll(teacher(params, tokens, coefficients), tokens)

    @jax.jit
    def capture(params, tokens, coefficients):
        captured = []

        def observe(module, x, *args, **kwargs):
            captured.append(x)
            return native_call(module, x, *args, **kwargs)

        with patch.object(MultiHeadAttention, "__call__", observe):
            observed_logits = teacher(params, tokens, coefficients)
        if len(captured) != coefficients.shape[1]:
            raise ValueError("Teacher capture did not visit every layer exactly once")
        native_logits = teacher(params, tokens, coefficients)
        difference = observed_logits.astype(jnp.float32) - native_logits.astype(jnp.float32)
        return {
            "inputs": jnp.stack(captured, axis=1),
            "nll": _sequence_nll(observed_logits, tokens),
            "joint_native_nll": _sequence_nll(native_logits, tokens),
            "joint_logit_max_absolute_error": jnp.max(jnp.abs(difference)),
            "joint_logit_relative_l2_error": jnp.linalg.norm(difference) /
                jnp.maximum(jnp.linalg.norm(native_logits.astype(jnp.float32)), 1e-30),
        }

    return capture, native_losses


def make_student_losses(graph):
    """Deploy the native student; no teacher inputs or coefficient override exist."""
    @jax.jit
    def losses(params, tokens):
        logits = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True)
        return _sequence_nll(logits, tokens)

    return losses
