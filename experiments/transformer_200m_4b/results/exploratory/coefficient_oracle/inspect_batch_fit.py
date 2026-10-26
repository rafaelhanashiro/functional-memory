"""Replay completed coefficient fits and measure functional compression on one GPU.

Raw compact measurements are written beside the selected run. PCA is fitted on
192 of the already-fitted sequences, not on an independent LM training split.
The remaining 64 test transfer of the coefficient basis, not LM generalization.
No parameter is trained or overwritten by this analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorstore as ts
import yaml

from scaffolds.records import load_run, update_record, validate_record
from scaffolds.transformer.attention.parameters import parameters_for_layer
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training import telemetry
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import (
    parent_config, restore_parent, split_generator_parameters,
)
from experiments.transformer_200m_4b.results.exploratory.coefficient_oracle.analyze_batch_coefficients import (
    pca, sample_coefficients,
)

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = "batch_functional_probe.json"
QUANTILES = [0., .01, .1, .5, .9, .99, 1.]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_checkpoint(run, record, completion):
    descriptor = record["selected_checkpoint"]
    checkpoint = Path(descriptor["path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    expected = {entry["path"]: entry["size_bytes"] for entry in descriptor["files"]}
    actual = {path.relative_to(checkpoint).as_posix(): path.stat().st_size
              for path in checkpoint.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Retained checkpoint file names/sizes disagree with the execution record")
    identity_path = checkpoint / "coefficient_identity.json"
    identity = json.loads(identity_path.read_text())
    identity_descriptor = next(entry for entry in descriptor["files"]
                               if entry["path"] == identity_path.name)
    if sha256(identity_path) != identity_descriptor["sha256"]:
        raise ValueError("Checkpoint identity checksum changed")
    if identity["step"] != completion["best_step"]:
        raise ValueError("The checkpoint is not the fitting-best checkpoint")
    return checkpoint, identity


def coefficient_chunks(checkpoint, recipe):
    """Read the free tables one saved microbatch at a time, never all on GPU."""
    directory = checkpoint / "best"
    metadata = json.loads((directory / "_METADATA").read_text())
    if not metadata.get("use_ocdbt") or metadata.get("use_zarr3"):
        raise ValueError("Expected OCDBT / Zarr-v2 coefficient storage")
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
            raise ValueError("Free-table shape differs from the frozen recipe")
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
        raise ValueError("The checkpoint does not contain the full fitting batch")


def fit_direction_basis(sampled, train_sequences):
    """Centered PCA of u=(r!c)/||r!c||, separately for every layer/head."""
    factorial = np.asarray([math.factorial(r) for r in range(sampled.shape[-1])])
    coordinates = np.asarray(sampled[:train_sequences], np.float64) * factorial
    norms = np.linalg.norm(coordinates, axis=-1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("A coefficient vector has undefined direction")
    directions = coordinates / norms
    layers, heads, orders = directions.shape[1], directions.shape[3], directions.shape[4]
    means = np.empty((layers, heads, orders))
    bases = np.empty((layers, heads, orders, orders))
    cumulative = np.empty((layers, heads, orders))
    for layer in range(layers):
        for head in range(heads):
            result = pca(directions[:, layer, :, head].reshape(-1, orders))
            means[layer, head], bases[layer, head] = result["mean"], result["basis"]
            cumulative[layer, head] = result["cumulative"]
    return means, bases, cumulative


def transform_coefficients(table, means, bases, *, rank=None, unit=False):
    """Project coefficient directions, preserving the original positive radius.

    All coordinate arithmetic is FP64 on CPU, then rounded once to stored FP32.
    The full-rank control measures the effect of this coordinate round trip.
    """
    factorial = np.asarray([math.factorial(r) for r in range(table.shape[-1])])
    coordinates = np.asarray(table, np.float64) * factorial
    norms = np.linalg.norm(coordinates, axis=-1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("A coefficient vector has undefined direction")
    directions = coordinates / norms
    if rank is not None:
        basis = bases[..., :rank]
        projected = np.einsum("blthr,lhrk->blthk", directions - means[None, :, None],
                              basis, optimize=True)
        directions = means[None, :, None] + np.einsum(
            "blthk,lhrk->blthr", projected, basis, optimize=True)
        reconstructed_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
        if np.any(reconstructed_norm == 0):
            raise ValueError("PCA reconstruction has undefined direction")
        directions /= reconstructed_norm
    result = directions * (1. if unit else norms) / factorial
    if not np.isfinite(result).all():
        raise ValueError("Coefficient reconstruction is nonfinite")
    return result.astype(np.float32)


def first_layer_token_directions(chunks, inputs, vocabulary_size):
    """Optimistic mean direction for each input token, using all fitted examples."""
    sums = None
    counts = np.zeros(vocabulary_size, dtype=np.int64)
    for offset, table in chunks:
        factorial = np.asarray([math.factorial(r) for r in range(table.shape[-1])])
        coordinates = np.asarray(table[:, 0], np.float64) * factorial
        norms = np.linalg.norm(coordinates, axis=-1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Free coefficient vector has undefined direction")
        tokens = inputs[offset:offset + len(table)].reshape(-1)
        directions = (coordinates / norms).reshape(len(tokens), *coordinates.shape[-2:])
        if sums is None:
            sums = np.zeros((vocabulary_size, *coordinates.shape[-2:]), dtype=np.float64)
        np.add.at(sums, tokens, directions)
        counts += np.bincount(tokens, minlength=vocabulary_size)
    seen = counts > 0
    mean = sums[seen] / counts[seen, None, None]
    mean_norm = np.linalg.norm(mean, axis=-1, keepdims=True)
    if np.any(mean_norm == 0):
        raise ValueError("The same-token mean has undefined direction")
    sums[seen] = mean / mean_norm
    return sums, {
        "token_occurrences": int(counts.sum()), "unique_input_tokens": int(seen.sum()),
        "tokens_with_multiple_occurrences": int((counts > 1).sum()),
        "token_frequency_quantiles": np.quantile(counts[seen], QUANTILES).tolist(),
        "unnormalized_mean_direction_norm_quantiles": np.quantile(mean_norm, QUANTILES).tolist(),
        "mean_direction_table_sha256": hashlib.sha256(sums.tobytes()).hexdigest(),
        "limits": "Optimistic token lookup fitted to all optimized coefficients and their labels; only layer zero is replaced. The original-radius variant retains per-sequence/query radial dependence, which can matter through RMS epsilon. The fixed-radius variant uses unit factorial radius and is strictly token-only in layer zero. Neither is a refitted model, causal generalization test, or proof that optimized coefficients are identifiable.",
    }


def replace_first_layer_by_token(table, tokens, token_directions, *, unit=False):
    factorial = np.asarray([math.factorial(r) for r in range(table.shape[-1])])
    radius = np.linalg.norm(np.asarray(table[:, 0], np.float64) * factorial,
                            axis=-1, keepdims=True)
    result = table.copy()
    result[:, 0] = token_directions[tokens] * (1. if unit else radius) / factorial
    return result


def losses_summary(values, split):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite replay NLL")
    summary = {"per_sequence_nll": values.tolist()}
    for label, chunk in (("all_fitting_sequences", values),
                         ("basis_fit_sequences", values[:split]),
                         ("other_fitting_sequences", values[split:])):
        if len(chunk):
            nll = float(chunk.mean())
            summary[label] = {"sequence_count": len(chunk), "nll": nll,
                              "perplexity": math.exp(min(nll, 80.)),
                              "per_sequence_nll_quantiles": np.quantile(chunk, QUANTILES).tolist()}
    return summary


def make_native_evaluator(graph):
    @jax.jit
    def evaluate(params, tokens):
        logits = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)
    return evaluate


def make_override_evaluator(graph):
    @jax.jit
    def evaluate(params, tokens, table):
        logits = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True,
                                        query_coefficients=jnp.swapaxes(table, 0, 1))
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)
    return evaluate


def make_query_inspector(graph, positions):
    """Use existing blocks in native order; collect diagnostics beside the forward."""
    positions = jnp.asarray(positions)
    original_record_attention_output = telemetry.record_attention_output

    def record_attention_output(y, *, head_rms_norm_eps=None):
        original_record_attention_output(y, head_rms_norm_eps=head_rms_norm_eps)
        if telemetry.active() and head_rms_norm_eps is not None:
            squared = jnp.mean(jnp.square(y.astype(jnp.float32)), axis=-1)
            for multiplier in (1, 100):
                telemetry.record(f"head_mean_square_lt_{multiplier}eps_fraction_mean",
                    jnp.mean((squared < multiplier * head_rms_norm_eps).astype(jnp.float32)), "mean")

    def forward(params, inputs):
        model = nnx.merge(graph, params)
        x = model.dropout(model.token_embedding(inputs), deterministic=True)
        statistics, norm_samples, coordinate_samples = [], [], []
        # The process-local hook records two extra reductions during JIT tracing;
        # the scaffold and native mathematical forward are unchanged.
        with patch.object(telemetry, "record_attention_output", record_attention_output), telemetry.collect() as collector:
            for index, name in enumerate(model.layer_names):
                layer = getattr(model, name)
                parameters = parameters_for_layer(model, model.attention_parameter_specs, index)
                h = layer.attention_norm(x)
                kernel = parameters["query_polynomial_kernel"].astype(jnp.float32)
                bias = parameters["query_polynomial_bias"].astype(jnp.float32)
                amplitude = parameters["query_polynomial_amplitude"].astype(jnp.float32)
                z = jnp.einsum("btd,hdr->bthr", h.astype(jnp.float32), kernel,
                               precision=jax.lax.Precision.HIGHEST,
                               preferred_element_type=jnp.float32)
                if "query_polynomial_tanh_bias" in parameters:
                    z = z + parameters["query_polynomial_tanh_bias"].astype(jnp.float32)
                feature = jnp.tanh(z)
                derivative = 1. - jnp.square(feature)
                with telemetry.scope(f"layer_{index:02d}"):
                    x, coefficients = layer(x, deterministic=True, attention_parameters=parameters,
                                            return_coefficients=True)
                factorial = jnp.asarray([math.factorial(r) for r in range(coefficients.shape[-1])])
                coordinates = coefficients * factorial
                norm = jnp.linalg.norm(coordinates, axis=-1, keepdims=True)
                direction = coordinates / jnp.maximum(norm, 1e-30)
                modulation = amplitude * feature
                cancellation = jnp.abs(coordinates) / jnp.maximum(
                    jnp.abs(bias) + jnp.abs(modulation), 1e-30)
                # Geometry sensitivity, not the task-loss gradient: ||d(u/||u||)/du_r||.
                direction_jacobian = jnp.sqrt(jnp.maximum(0., 1. - direction ** 2)) / jnp.maximum(norm, 1e-12)
                h_norm = jnp.linalg.norm(h.astype(jnp.float32), axis=-1)[:, :, None, None]
                values = {
                    "preactivation_mean_square": z ** 2,
                    "saturated_099_fraction": (jnp.abs(feature) >= .99).astype(jnp.float32),
                    "tanh_derivative_mean": derivative,
                    "tanh_derivative_lt001_fraction": (derivative < .01).astype(jnp.float32),
                    "cancellation_ratio_mean": cancellation,
                    "cancellation_ratio_lt001_fraction": (cancellation < .01).astype(jnp.float32),
                    "coefficient_mean_square_factorial_coordinates": coordinates ** 2,
                    "direction_jacobian_norm_floor_fraction": jnp.broadcast_to(
                        (norm < 1e-12).astype(jnp.float32), coordinates.shape),
                    "direction_bias_jacobian_mean_square": direction_jacobian ** 2,
                    "direction_amplitude_jacobian_mean_square": (direction_jacobian * feature) ** 2,
                    "direction_projection_jacobian_frobenius_mean_square": (
                        direction_jacobian * amplitude * derivative * h_norm) ** 2,
                }
                statistics.append({key: value.mean(axis=(0, 1)) for key, value in values.items()})
                # take() preserves [sequence, sampled_query, head]; mixed advanced
                # indexing with a scalar final index would move the sampled axis.
                norm_samples.append(jnp.take(norm[..., 0], positions, axis=1))
                coordinate_samples.append(jnp.take(coordinates, positions, axis=1))
            logits = model.lm_head(model.final_norm(x))
        stacked = {key: jnp.stack([row[key] for row in statistics]) for key in statistics[0]}
        return logits, stacked, jnp.stack(norm_samples), collector.values, jnp.stack(coordinate_samples)

    @jax.jit
    def inspect(params, tokens):
        logits, stats, norms, activation, coordinates = forward(params, tokens[:, :-1])
        nll = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[:, 1:]).mean(axis=1)
        return nll, stats, norms, activation, coordinates

    @jax.jit
    def compare(params, tokens):
        native = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True)
        wrapped, _, _, _, _ = forward(params, tokens[:, :-1])
        difference = native.astype(jnp.float32) - wrapped.astype(jnp.float32)
        return {"logit_max_absolute_error": jnp.max(jnp.abs(difference)),
                "logit_relative_rms_error": jnp.linalg.norm(difference) /
                    jnp.maximum(jnp.linalg.norm(native.astype(jnp.float32)), 1e-30)}
    return inspect, compare


def query_diagnostics(graph, params, batch, microbatch, positions):
    inspect, compare = make_query_inspector(graph, positions)
    parity = {key: float(value) for key, value in jax.device_get(
        compare(params, jax.device_put(batch[:microbatch]))).items()}
    if parity["logit_max_absolute_error"] > .05:
        raise ValueError(f"Diagnostic wrapper does not reproduce native logits: {parity}")
    losses, stats, norm_samples, activation, coordinates = [], [], [], [], []
    for offset in range(0, len(batch), microbatch):
        row = jax.device_get(inspect(params, jax.device_put(batch[offset:offset + microbatch])))
        losses.extend(row[0].tolist()); stats.append(row[1]); norm_samples.append(row[2]); activation.append(row[3])
        coordinates.append(row[4])
    means = {key: np.mean([row[key] for row in stats], axis=0).tolist() for key in stats[0]}
    norms = np.concatenate(norm_samples, axis=1)  # [layer,sequence,sampled_query,head]
    telemetry_values = {}
    for key in activation[0]:
        values = np.asarray([row[key] for row in activation], np.float64)
        if key.endswith("_min"):
            value = values.min()
        elif key.endswith("_max"):
            value = values.max()
        elif key.endswith("_rms"):
            value = np.sqrt(values.mean())  # collector stores a mean square
        elif key.endswith("_mean"):
            value = values.mean()
        else:
            raise ValueError(f"Unknown activation telemetry reduction: {key}")
        telemetry_values[key] = float(value)
    return np.asarray(losses), {
        "native_wrapper_parity": parity,
        "statistics_axes": ["layer", "head", "order"], "statistics": means,
        "factorial_coordinate_norm_quantiles": np.quantile(norms, QUANTILES).tolist(),
        "per_layer_head_factorial_coordinate_norm_quantiles": np.quantile(norms, QUANTILES, axis=(1, 2)).transpose(1, 2, 0).tolist(),
        "sampled_query_positions": positions.tolist(), "quantile_probabilities": QUANTILES,
        "pre_output_rms_telemetry": telemetry_values,
        "direction_jacobian_interpretation": "Sensitivity of normalized factorial-coordinate direction to stored b/a/W; not the task-loss gradient or a score-weighted polynomial norm. Radius denominator is floored at 1e-12; the fraction affected is reported.",
    }, np.concatenate(coordinates, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pca-train-sequences", type=int, default=192)
    parser.add_argument("--query-samples", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve(); output = run / OUTPUT
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Use one GPU for model replay")
    started = time.monotonic()
    recipe = yaml.safe_load((run / "config.yaml").read_text()); oracle = recipe["oracle"]
    record = load_run(run); completion = json.loads((run / "completion.json").read_text())
    if record["status"] != "complete" or completion["status"] != "complete":
        raise ValueError("Only completed fitting runs may be inspected")
    validate_record(run, checkpoint_hashes=False)
    checkpoint, identity = selected_checkpoint(run, record, completion)
    parent_path = Path(oracle["parent_config"])
    if sha256(parent_path) != oracle["parent_config_sha256"]:
        raise ValueError("Frozen parent configuration changed")
    validate_record(parent_path.parent, checkpoint_hashes=False)
    parent_record = load_run(parent_path.parent)
    if parent_record["selected_checkpoint"]["sha256"] != oracle["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint identity changed")
    config = parent_config(parent_path)
    if (identity["parent_run_id"] != oracle["parent_run_id"] or config.model.remat
            or config.model.attention_normalization):
        raise ValueError("Unexpected coefficient diagnostic parent")
    if sha256(config.data.eval_manifest) != oracle["eval_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    count, microbatch = int(oracle["sequence_count"]), int(oracle["microbatch_size"])
    if count != 256 or count % microbatch or not 0 < args.pca_train_sequences < count:
        raise ValueError("This diagnostic requires the complete 256-sequence batch")
    batch = next(iter(PackedBatchIterator(config.data.eval_manifest, batch_size=count,
        sequence_length=config.data.sequence_length, start_sequence=oracle["data_start"])))
    if hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest() != oracle["batch_sha256"]:
        raise ValueError("Fitting tokens changed")
    positions = np.unique(np.linspace(0, batch.shape[1] - 2, args.query_samples, dtype=int))
    graph, source = restore_parent(config, Path(oracle["parent_checkpoint"]))
    source_hash = state_sha256(source)
    native = make_native_evaluator(graph); override = make_override_evaluator(graph)
    parameterization = oracle["parameterization"]
    tolerance = .002 if config.train.dtype == "bfloat16" else 1e-5
    measurements, diagnostics = {}, {}
    initial = np.concatenate([np.asarray(native(source, jax.device_put(batch[i:i + microbatch])))
                              for i in range(0, count, microbatch)])
    measurements["original_parent"] = losses_summary(initial, args.pca_train_sequences)
    if abs(float(initial.mean()) - completion["initial_nll"]) > tolerance:
        raise ValueError("Original parent replay does not match the initial fitting NLL")
    if parameterization == "free_per_example_query_coefficients":
        sampled = sample_coefficients(checkpoint, recipe, positions)
        means, bases, cumulative = fit_direction_basis(sampled, args.pca_train_sequences)
        del sampled
        token_directions, token_diagnostic = first_layer_token_directions(
            coefficient_chunks(checkpoint, recipe), batch[:, :-1], config.model.vocab_size)
        orders = config.model.attention_degree + 1
        variants = {"fitted_raw": (None, False), "unit_direction_gauge_control": (None, True)}
        variants.update({f"pca_rank_{rank}": (rank, False) for rank in range(orders + 1)})
        values = {key: [] for key in variants}
        values["first_layer_token_mean_direction"] = []
        values["first_layer_token_mean_fixed_radius"] = []
        roundtrip_error = 0.
        for offset, table in coefficient_chunks(checkpoint, recipe):
            tokens = jax.device_put(batch[offset:offset + len(table)])
            for label, (rank, unit) in variants.items():
                transformed = table if label == "fitted_raw" else transform_coefficients(
                    table, means, bases, rank=rank, unit=unit)
                if rank == orders:
                    roundtrip_error = max(roundtrip_error, float(np.max(np.abs(transformed - table))))
                loss = np.asarray(override(source, tokens, jax.device_put(transformed)))
                values[label].extend(loss.tolist())
            for label, unit in (("first_layer_token_mean_direction", False),
                                ("first_layer_token_mean_fixed_radius", True)):
                transformed = replace_first_layer_by_token(
                    table, batch[offset:offset + len(table), :-1], token_directions, unit=unit)
                values[label].extend(np.asarray(override(source, tokens, jax.device_put(transformed))).tolist())
            print(f"Free replay sequences {offset + len(table)}/{count}", flush=True)
        measurements.update({label: losses_summary(losses, args.pca_train_sequences)
                             for label, losses in values.items()})
        diagnostics["pca"] = {"coordinates": "unit direction of r! c_r; positive radius retained during rank reconstruction",
            "fit_sequence_indices": [0, args.pca_train_sequences], "query_positions": positions.tolist(),
            "per_layer_head_means": means.tolist(), "per_layer_head_bases_columns": bases.tolist(),
            "per_layer_head_cumulative_variance": cumulative.tolist(),
            "full_rank_max_coefficient_roundtrip_error": roundtrip_error,
            "rank_zero_is_unrefitted_mean_direction": True,
            "limits": "PCA target coefficients already used every fitting sequence's labels. Other sequences test basis transfer only. PCA latent coordinates still come from each sequence's optimized table, not a causal input-dependent generator. Rank zero has a fixed direction but retains the original per-sequence/query radius; it is not a strictly static coefficient table when RMS epsilon matters."}
        diagnostics["first_layer_token_mean_direction"] = token_diagnostic
    elif parameterization == "static_coefficients":
        target = jnp.zeros((config.model.num_layers, config.model.num_heads, config.model.attention_degree + 1))
        with ocp.StandardCheckpointer() as checkpointer:
            table = checkpointer.restore(checkpoint / "best", target={"coefficients": [target]})["coefficients"][0]
        losses = []
        for offset in range(0, count, microbatch):
            tokens = jax.device_put(batch[offset:offset + microbatch])
            expanded = jnp.broadcast_to(table[None, :, None],
                (len(tokens), config.model.num_layers, tokens.shape[1] - 1, *table.shape[1:]))
            losses.extend(np.asarray(override(source, tokens, expanded)).tolist())
        measurements["fitted_raw"] = losses_summary(losses, args.pca_train_sequences)
    elif parameterization == "query_coefficient_generator":
        if (config.model.attention_coefficient_readout != "diagonal"
                or config.model.attention_coefficient_scale != "factorial"
                or config.model.attention_coefficient_amplitude_scale != "factorial"):
            raise ValueError("Generator diagnostics require the original factorial-scaled diagonal generator")
        original_generator, frozen = split_generator_parameters(source)
        with ocp.StandardCheckpointer() as checkpointer:
            fitted = checkpointer.restore(checkpoint / "best", target={
                "generator_parameters": [original_generator]})["generator_parameters"][0]
        fitted_params = nnx.merge_state(frozen, fitted)
        losses = np.concatenate([np.asarray(native(fitted_params, jax.device_put(batch[i:i + microbatch])))
                                  for i in range(0, count, microbatch)])
        measurements["fitted_raw"] = losses_summary(losses, args.pca_train_sequences)
        parameter_snapshots, coefficient_snapshots = {}, {}
        for label, params in (("before", source), ("after", fitted_params)):
            replay, diagnostic, coordinates = query_diagnostics(graph, params, batch, microbatch, positions)
            expected = initial if label == "before" else losses
            error = float(np.max(np.abs(replay - expected)))
            if error > tolerance:
                raise ValueError(f"Diagnostic wrapper NLL differs from native: {error}")
            diagnostic["external_native_max_nll_error"] = error
            generator, _ = split_generator_parameters(params)
            parameter_values = {str(path[-1]): np.asarray(value[...], np.float64)
                                for path, value in nnx.to_flat_state(generator)}
            parameter_snapshots[label], coefficient_snapshots[label] = parameter_values, coordinates
            diagnostic["stored_bias"] = parameter_values["query_polynomial_bias"].tolist()
            diagnostic["stored_amplitude"] = parameter_values["query_polynomial_amplitude"].tolist()
            diagnostic["projection_rms_per_layer_head_order"] = np.sqrt(np.mean(
                parameter_values["query_polynomial_kernel"] ** 2, axis=-2)).tolist()
            diagnostic["stored_parameter_rms"] = {
                name: float(np.sqrt(np.mean(value ** 2))) for name, value in parameter_values.items()}
            diagnostics[label] = diagnostic
        parameter_changes = {}
        for name, before in parameter_snapshots["before"].items():
            after = parameter_snapshots["after"][name]
            difference = after - before
            parameter_changes[name] = {
                "delta_rms": float(np.sqrt(np.mean(difference ** 2))),
                "delta_relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(before), 1e-30)),
                "delta_rms_per_layer_head_order": (
                    np.sqrt(np.mean(difference ** 2, axis=-2)) if difference.ndim == 4 else np.abs(difference)).tolist(),
            }
        before, after = (np.asarray(coefficient_snapshots[label], np.float64) for label in ("before", "after"))
        before_norm = np.linalg.norm(before, axis=-1, keepdims=True)
        after_norm = np.linalg.norm(after, axis=-1, keepdims=True)
        cosine = np.sum(before * after, axis=-1) / np.maximum((before_norm * after_norm)[..., 0], 1e-30)
        angles = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
        diagnostics["native_generator_change"] = {
            "parameter_changes": parameter_changes,
            "sampled_direction_angle_degrees_quantiles": np.quantile(angles, QUANTILES).tolist(),
            "per_layer_head_direction_angle_degrees_quantiles": np.quantile(angles, QUANTILES, axis=(1, 2)).transpose(1, 2, 0).tolist(),
            "sampled_radius_ratio_quantiles": np.quantile(after_norm / np.maximum(before_norm, 1e-30), QUANTILES).tolist(),
            "sampled_factorial_coordinate_delta_rms": float(np.sqrt(np.mean((after - before) ** 2))),
            "limits": "Before and after coefficients are captured from each generator's own causal native forward, so later-layer inputs also change. Parameter differences are accumulated changes, not Adam update estimates from gradient norms.",
        }
    else:
        raise ValueError("Unsupported coefficient fitting parameterization")
    fitted_nll = measurements["fitted_raw"]["all_fitting_sequences"]["nll"]
    if abs(fitted_nll - completion["best_fit_nll"]) > tolerance:
        raise ValueError(f"Best checkpoint replay differs from recorded fitting NLL: {fitted_nll} vs {completion['best_fit_nll']}")
    if state_sha256(source) != source_hash:
        raise RuntimeError("Analysis changed the source parameter tree")
    result = {"schema_version": 1, "run_id": run.name, "parameterization": parameterization,
        "config_sha256": sha256(run / "config.yaml"), "completion_sha256": sha256(run / "completion.json"),
        "selected_checkpoint_step": identity["step"], "checkpoint_sha256": record["selected_checkpoint"]["sha256"],
        "parent_run_id": oracle["parent_run_id"], "parent_checkpoint_sha256": oracle["parent_checkpoint_sha256"],
        "batch_sha256": oracle["batch_sha256"], "eval_manifest_sha256": oracle["eval_manifest_sha256"],
        "sequence_count": count, "prediction_positions": batch.shape[1] - 1,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "analysis_script_sha256": sha256(__file__), "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu": jax.devices()[0].device_kind,
        "cuda_environment": os.environ.get("TRANSFORMER_CUDA_ENVIRONMENT"),
        "microbatch_size": microbatch, "nll_replay_atol": tolerance,
        "recorded_best_nll": completion["best_fit_nll"], "best_checkpoint_replay_nll_error": fitted_nll - completion["best_fit_nll"],
        "frozen_parent_parameters_unchanged": True, "measurements": measurements, "diagnostics": diagnostics,
        "elapsed_seconds": time.monotonic() - started,
        "limits": "Fixed-input fitting diagnostics with different source backbones. These are not held-out LM perplexities. Coefficient-space direction/Jacobian statistics do not directly measure attention-output or loss sensitivity; functional replay supplies the corresponding loss measurements."}
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    update_record(run, {"analyses": {"batch_functional_probe": {
        "path": str(output.relative_to(ROOT)), "sha256": sha256(output), "size_bytes": output.stat().st_size,
        "source_commit": result["source_commit"], "analysis_script_sha256": result["analysis_script_sha256"],
        "selected_checkpoint_sha256": result["checkpoint_sha256"]}}})
    print(json.dumps({"run_id": run.name, "output": str(output), "measurements": {
        label: value["all_fitting_sequences"] for label, value in measurements.items()}}), flush=True)


if __name__ == "__main__":
    main()
