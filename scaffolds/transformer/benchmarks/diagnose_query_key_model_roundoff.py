"""Replay real model attention adjoints against an independent FP64 formula.

Both full-model geometries supply exact attention operands, incoming cotangents,
and coefficient projection inputs. Each is repeated to check determinism. An
independent FP64 attention VJP and FP64 coefficient-projection contraction
separate local arithmetic error from differences already in the incoming
cotangent. The production FP32 projection VJP is checked separately. This
diagnostic does not waive a failed full-model qualification.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import partial
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends import fused_query_key_polynomial as kernel
from scaffolds.transformer.attention import input_dependent as projections
from scaffolds.transformer.benchmarks.benchmark_attention import write_report
from scaffolds.transformer.benchmarks.benchmark_train_step import parameter_digest
from scaffolds.transformer.benchmarks.benchmark_training_throughput import _execution_provenance, digest
from scaffolds.transformer.benchmarks.qualify_query_key_model import (
    compare_array, compare_gradients, full_objective, immutable_snapshot, perturbed_parameters,
    name_for_path, PROJECTION_NAMES, COEFFICIENT_NAMES, LOSS_TOLERANCE,
)
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.state import create_state


NAMES = ("q", "k", "v", "coefficients", "factors")
TOLERANCE = 0.025
GROUP_TOLERANCE = 0.05
ATTRIBUTION_TOLERANCE = 0.1


def array_hash(value):
    array = np.asarray(jax.device_get(value))
    return hashlib.sha256(memoryview(np.ascontiguousarray(array).view(np.uint8))).hexdigest()


def capture_dispatcher(layer_count, heads, layers, captures, enabled, forward, backward):
    """Bind explicit layer identity while Python traces the forward model loop."""
    counts = {"forward": 0}

    def save(layer, *arrays):
        if not enabled["capture"]:
            return
        if layer in captures:
            raise RuntimeError(f"Duplicate capture for layer {layer}")
        captures[layer] = tuple(np.array(array, copy=True) for array in arrays)

    def dispatch(q, k, v, coefficients, factors, interpret):
        if interpret:
            raise ValueError("Capture requires an allocated GPU implementation")
        layer = counts["forward"]
        counts["forward"] += 1
        if layer >= layer_count:
            raise RuntimeError("Attention traced more than once per model layer")

        @jax.custom_vjp
        def attention(*operands):
            return forward(*operands, False)[0]

        def fwd(*operands):
            return forward(*operands, False)

        def bwd(residuals, cotangent):
            if layer in layers:
                selected = tuple(jnp.take(array, jnp.asarray(heads), axis=2)
                                 for array in (*residuals, cotangent))
                jax.debug.callback(partial(save, layer), *selected, ordered=True)
            return backward(False, residuals, cotangent)

        attention.defvjp(fwd, bwd)
        return attention(q, k, v, coefficients, factors)

    return dispatch, counts


def projection_capture(original, *, kind, heads, layers, captures, enabled):
    """Capture the actual projection inputs without changing its differentiation."""
    counts = {"forward": 0}

    def save(layer, *arrays):
        if not enabled["capture"]:
            return
        if layer in captures:
            raise RuntimeError(f"Duplicate {kind} projection capture at layer {layer}")
        captures[layer] = tuple(np.array(array, copy=True) for array in arrays)

    def dispatch(x, weights, *args, **kwargs):
        layer = counts["forward"]
        counts["forward"] += 1
        result = original(x, weights, *args, **kwargs)
        if layer in layers:
            index = jnp.asarray(heads)
            selected_weights = jnp.take(weights, index, axis=0)
            if kind == "query":
                if kwargs.get("activation") != "tanh" or kwargs.get("tanh_bias") is not None:
                    raise ValueError("Diagnostic expects tanh coefficients without an inner bias")
                arrays = (x, selected_weights, jnp.take(args[0], index, axis=0),
                          jnp.take(kwargs["amplitude"], index, axis=0))
            else:
                arrays = (selected_weights, jnp.take(args[0], index, axis=0))
            jax.debug.callback(partial(save, layer), *arrays, ordered=True)
        return result

    return dispatch, counts


def fp64_gradients(q, k, v, coefficients, factors, cotangent):
    """Independent dense analytic VJP; inputs have one selected head."""
    q, k, v, coefficients, factors, cotangent = (
        x[:, :, 0].astype(jnp.float64) for x in (q, k, v, coefficients, factors, cotangent))
    dot = lambda a, b: jnp.einsum("bij,bjk->bik", a, b, precision=jax.lax.Precision.HIGHEST)
    scale = q.shape[-1] ** -0.5
    scores = dot(q, jnp.swapaxes(k, 1, 2)) * scale
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(k.shape[1])[None, :]
    scores = jnp.where(causal, scores, 0.0)
    degree = coefficients.shape[-1] - 1
    values = coefficients[..., degree, None] * factors[:, None, :, degree]
    derivative = jnp.zeros_like(scores)
    for order in range(degree - 1, -1, -1):
        derivative = derivative * scores + values
        values = values * scores + coefficients[..., order, None] * factors[:, None, :, order]
    weights = jnp.where(causal, values, 0.0)
    dp = dot(cotangent, jnp.swapaxes(v, 1, 2))
    ds = jnp.where(causal, dp * derivative * scale, 0.0)
    dq = dot(ds, k)
    dk = dot(jnp.swapaxes(ds, 1, 2), q)
    dv = dot(jnp.swapaxes(weights, 1, 2), cotangent)
    power = jnp.ones_like(scores)
    dc, df = [], []
    for order in range(degree + 1):
        contribution = jnp.where(causal, dp * power, 0.0)
        dc.append(jnp.sum(contribution * factors[:, None, :, order], axis=2))
        df.append(jnp.sum(contribution * coefficients[..., order, None], axis=1))
        power = power * scores
    return tuple(x[:, :, None] for x in (dq, dk, dv, jnp.stack(dc, -1), jnp.stack(df, -1)))


def compare_local(actual, reference):
    result = {}
    for name, observed, expected in zip(NAMES, actual, reference, strict=True):
        row = compare_array(observed, expected, tolerance=TOLERANCE)
        if name in ("coefficients", "factors"):
            row["orders"] = [dict(order=order, **compare_array(observed[..., order], expected[..., order],
                                                              tolerance=TOLERANCE))
                             for order in range(expected.shape[-1])]
            row["accepted"] &= all(item["accepted"] for item in row["orders"])
        result[name] = row
    return {"accepted": all(item["accepted"] for item in result.values()), "gradients": result}


def failed_projection_groups(report):
    return [dict(parameter=leaf["parameter"], **group)
            for leaf in report["gradients"]["leaves"]
            if any(name in leaf["parameter"] for name in PROJECTION_NAMES)
            for group in leaf.get("groups", {}).get("failed", [])]


def failed_coefficient_groups(report):
    return [dict(parameter=leaf["parameter"], **group)
            for leaf in report["gradients"]["leaves"]
            if any(name in leaf["parameter"] for name in COEFFICIENT_NAMES)
            for group in leaf.get("groups", {}).get("failed", [])]


def projection_pullback(kind, captured, head_index, amplitude_scale):
    """Use the production FP32 tanh/projection Jacobian, holding its inputs fixed."""
    if kind == "query":
        x, weights, bias, amplitude = captured
        weights, bias, amplitude = (jnp.asarray(a[head_index:head_index + 1])
                                    for a in (weights, bias, amplitude))
        x = jnp.asarray(x)
        operation = lambda w: projections.project_coefficients(
            x, w, bias, amplitude=amplitude, activation="tanh", amplitude_scale=amplitude_scale)
    else:
        x, weights, amplitude = captured
        weights, amplitude = (jnp.asarray(a[head_index:head_index + 1]) for a in (weights, amplitude))
        x = jnp.asarray(x)
        operation = lambda w: projections.project_key_factors(x, w, amplitude)

    @jax.jit
    def pullback(cotangent):
        _, vjp = jax.vjp(operation, weights)
        return vjp(cotangent.astype(jnp.float32))[0][0]

    return pullback


def compare_projection(actual, reference, tolerance=GROUP_TOLERANCE):
    orders = [dict(order=order, **compare_array(actual[:, order], reference[:, order], tolerance=tolerance))
              for order in range(reference.shape[-1])]
    return {"accepted": all(row["accepted"] for row in orders), "orders": orders}


def fp64_projection(kind, captured, head_index, cotangent, amplitude_scale):
    """Independent analytic FP64 contraction; reconstruct tanh from captured X/W.

    This intentionally reconstructs XW in FP64 instead of sharing the production
    FP32 projection Jacobian. The production VJP is measured separately.
    """
    import math
    x, weights, *rest = captured
    amplitude = rest[-1]
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        x = jnp.asarray(x, jnp.float64).reshape(-1, x.shape[-1])
        weights = jnp.asarray(weights[head_index], jnp.float64)
        amplitude = jnp.asarray(amplitude[head_index], jnp.float64)
        dc = jnp.asarray(cotangent[:, :, 0], jnp.float64).reshape(-1, weights.shape[-1])
        if kind == "query" and amplitude_scale == "factorial":
            amplitude = amplitude / jnp.asarray([math.factorial(r) for r in range(weights.shape[-1])], jnp.float64)
        activation = jnp.tanh(x @ weights)
        result = x.T @ (dc * amplitude * (1 - activation ** 2))
        return np.asarray(jax.device_get(result))


def fp64_coefficient_parameters(kind, captured, head_index, cotangent, amplitude_scale, bias_scale):
    """Independent bias/amplitude derivatives, including model parameter scales."""
    x, weights, *_ = captured
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        x = jnp.asarray(x, jnp.float64).reshape(-1, x.shape[-1])
        weights = jnp.asarray(weights[head_index], jnp.float64)
        cotangent = jnp.asarray(cotangent[:, :, 0], jnp.float64).reshape(-1, weights.shape[-1])
        factorials = jnp.asarray([math.factorial(r) for r in range(weights.shape[-1])], jnp.float64)
        amplitude = jnp.sum(cotangent * jnp.tanh(x @ weights), axis=0)
        if kind == "query":
            bias = jnp.sum(cotangent, axis=0)
            if amplitude_scale == "factorial":
                amplitude = amplitude / factorials
            if bias_scale == "factorial":
                bias = bias / factorials
            values = {"query_polynomial_bias": bias, "query_polynomial_amplitude": amplitude}
        else:
            values = {"key_polynomial_amplitude": amplitude}
        return {name: np.asarray(jax.device_get(value)) for name, value in values.items()}


def coefficient_parameter_pullback(kind, captured, head_index, amplitude_scale, bias_scale):
    """Production FP32 projection VJP with respect to its bias/amplitudes."""
    x, weights, *rest = captured
    x = jnp.asarray(x)
    weights = jnp.asarray(weights[head_index:head_index + 1])
    amplitude = jnp.asarray(rest[-1][head_index:head_index + 1])
    if kind == "query":
        effective_bias = jnp.asarray(rest[0][head_index:head_index + 1])

        @jax.jit
        def pullback(cotangent):
            operation = lambda b, a: projections.project_coefficients(
                x, weights, b, amplitude=a, activation="tanh", amplitude_scale=amplitude_scale)
            _, vjp = jax.vjp(operation, effective_bias, amplitude)
            db, da = vjp(cotangent.astype(jnp.float32))
            if bias_scale == "factorial":
                # The captured bias is after model-owned factorial_scale; apply
                # its transpose to recover the raw model parameter gradient.
                db = db / jnp.asarray([math.factorial(r) for r in range(db.shape[-1])], jnp.float32)
            return {"query_polynomial_bias": db[0], "query_polynomial_amplitude": da[0]}
    else:
        @jax.jit
        def pullback(cotangent):
            _, vjp = jax.vjp(lambda a: projections.project_key_factors(x, weights, a), amplitude)
            return {"key_polynomial_amplitude": vjp(cotangent.astype(jnp.float32))[0][0]}
    return pullback


def compare_repeated_gradients(actual, reference):
    """Quantify execution/compilation variability without hiding it in a tree hash.

    In BF16 training Embed casts its table before take, so its transpose may
    scatter BF16 gradients. Do not assume embedding noise has an FP32 bound.
    Exact repeat acceptance is preserved until measured evidence justifies a
    separate numerical criterion; the existing parity gates are also reported.
    """
    result = compare_gradients(actual, reference)
    pairs_a = jax.tree_util.tree_flatten_with_path(actual)[0]
    pairs_b = jax.tree_util.tree_flatten_with_path(reference)[0]
    for row, (_, observed), (_, expected) in zip(result["leaves"], pairs_a, pairs_b, strict=True):
        row["actual_sha256"] = array_hash(observed)
        row["reference_sha256"] = array_hash(expected)
        row["dtype"] = str(expected.dtype)
        row["is_embedding"] = "['token_embedding']['embedding']" in row["parameter"]
    result["nonembedding_array_exact"] = all(row["array_equal"] and row["all_finite"]
                                              for row in result["leaves"] if not row["is_embedding"])
    result["embedding_array_exact"] = all(row["array_equal"] and row["all_finite"]
                                           for row in result["leaves"] if row["is_embedding"])
    result["array_exact"] = result["nonembedding_array_exact"] and result["embedding_array_exact"]
    result["changed_parameters"] = [row["parameter"] for row in result["leaves"] if not row["array_equal"]]
    return result


def capture_models(state, batch, rng, heads, layers, report, output, original_query, original_key):
    captures, projection_inputs, full_grads = {}, {}, {}
    original_attention = kernel._attention
    for tile in (64, 32):
        kernel._attention = original_attention
        projections.project_coefficients, projections.project_key_factors = original_query, original_key
        kernel._backward_tile_sizes = lambda *a, tile=tile, **kw: (tile, tile)
        jax.clear_caches()
        print(f"Tile{tile} uninstrumented baseline", flush=True)
        operation = full_objective()
        (loss, logits), grads = jax.block_until_ready(operation(state, batch, rng))
        if not bool(jnp.isfinite(loss)):
            raise FloatingPointError("Nonfinite uninstrumented full-model loss")
        grad_hash, _ = parameter_digest(grads)
        baseline = {"loss": float(loss), "logits_sha256": array_hash(logits), "gradient_sha256": grad_hash}
        baseline_grads = immutable_snapshot(grads)
        del loss, logits, grads, operation
        captures[tile] = {}
        projection_inputs[tile] = {"query": {}, "key": {}}
        enabled = {"capture": True}
        kernel._attention, counts = capture_dispatcher(
            15, heads, layers, captures[tile], enabled, kernel._attention_fwd, kernel._attention_bwd)
        projections.project_coefficients, query_counts = projection_capture(
            original_query, kind="query", heads=heads, layers=layers,
            captures=projection_inputs[tile]["query"], enabled=enabled)
        projections.project_key_factors, key_counts = projection_capture(
            original_key, kind="key", heads=heads, layers=layers,
            captures=projection_inputs[tile]["key"], enabled=enabled)
        kernel._backward_tile_sizes = lambda *a, tile=tile, **kw: (tile, tile)
        jax.clear_caches()
        operation = full_objective()
        trial = {"tile": tile, "uninstrumented": baseline, "evaluations": []}
        report["full_model_trials"].append(trial)
        for repeat in range(2):
            print(f"Tile{tile} full-model capture/determinism pass {repeat + 1}", flush=True)
            result = jax.block_until_ready(operation(state, batch, rng))
            jax.effects_barrier()
            (loss, logits), grads = result
            if not bool(jnp.isfinite(loss)):
                raise FloatingPointError("Nonfinite full-model capture loss")
            grad_hash, _ = parameter_digest(grads)
            trial["evaluations"].append({"loss": float(loss), "logits_sha256": array_hash(logits), "gradient_sha256": grad_hash})
            if repeat == 0:
                full_grads[tile] = immutable_snapshot(grads)
                trial["instrumentation_gradients"] = compare_repeated_gradients(full_grads[tile], baseline_grads)
                del baseline_grads
            else:
                repeat_grads = immutable_snapshot(grads)
                trial["repeat_gradients"] = compare_repeated_gradients(repeat_grads, full_grads[tile])
                del repeat_grads
            enabled["capture"] = False
            del result, loss, logits, grads
            write_report(output, report)
        if (any(c["forward"] != 15 for c in (counts, query_counts, key_counts))
                or set(captures[tile]) != layers
                or any(set(values) != layers for values in projection_inputs[tile].values())):
            raise RuntimeError(f"Capture did not bind exactly 15 layers: {counts}, {query_counts}, {key_counts}")
        trial["deterministic"] = trial["evaluations"][0] == trial["evaluations"][1]
        trial["instrumentation_array_exact"] = baseline == trial["evaluations"][0]
        trial["repeat_forward_exact"] = all(trial["evaluations"][0][key] == trial["evaluations"][1][key]
                                             for key in ("loss", "logits_sha256"))
        trial["instrumentation_forward_exact"] = all(baseline[key] == trial["evaluations"][0][key]
                                                      for key in ("loss", "logits_sha256"))
        write_report(output, report)
        del operation
    report["gradients"] = compare_gradients(full_grads[32], full_grads[64])
    projection_grads = {
        tile: {name_for_path(p): a for p, a in jax.tree_util.tree_flatten_with_path(full_grads[tile])[0]
               if any(name in name_for_path(p) for name in (*PROJECTION_NAMES, *COEFFICIENT_NAMES))} for tile in (64, 32)}
    return captures, projection_inputs, projection_grads


def diagnose_case(layer, head, heads, captures, projection_inputs, projection_grads, failures, amplitude_scale,
                  bias_scale, coefficient_outputs, coefficient_layers, replay_operations, oracle_operation):
    print(f"Local FP64 oracle and geometry replay: layer={layer} head={head}", flush=True)
    index = heads.index(head)
    host_inputs = {tile: tuple(x[:, :, index:index + 1] for x in captures[tile][layer]) for tile in (64, 32)}
    same_operands = all(np.array_equal(a, b) for a, b in zip(host_inputs[64][:5], host_inputs[32][:5], strict=True))
    same_projection = all(np.array_equal(a, b) for kind in ("query", "key")
                          for a, b in zip(projection_inputs[64][kind][layer], projection_inputs[32][kind][layer], strict=True))
    case = {"layer": layer, "head": head, "operands_exact_between_geometries": same_operands,
            "projection_inputs_exact_between_geometries": same_projection,
            "cotangent_comparison": compare_array(host_inputs[32][5], host_inputs[64][5], tolerance=TOLERANCE),
            "input_sha256": {str(tile): [array_hash(x) for x in host_inputs[tile]] for tile in (64, 32)},
            "comparisons": {}, "projection_comparisons": {}, "attribution": []}
    if not same_operands or not same_projection:
        raise RuntimeError("Full-model primal operands changed with backward geometry")
    projection_values = {
        "query": projection_inputs[64]["query"][layer],
        "key": (projection_inputs[64]["query"][layer][0], *projection_inputs[64]["key"][layer]),
    }
    pullbacks = {kind: projection_pullback(kind, values, index, amplitude_scale)
                 for kind, values in projection_values.items()}
    coefficient_pullbacks = ({kind: coefficient_parameter_pullback(kind, values, index, amplitude_scale, bias_scale)
                             for kind, values in projection_values.items()} if layer in coefficient_layers else {})
    coefficient_data = {"fp64": {}, "production_oracles": {}, "replays": {}}
    projected_oracles, production_oracles = {}, {}
    for incoming in (64, 32):
        print(f"  incoming cotangent from tile{incoming}: FP64 oracle", flush=True)
        arrays = tuple(jax.device_put(np.array(x, copy=True), may_alias=False) for x in host_inputs[incoming])
        with jax.enable_x64(), jax.default_matmul_precision("highest"):
            reference = immutable_snapshot(jax.block_until_ready(oracle_operation(*arrays)))
        projected_oracles[incoming], production_oracles[incoming] = {}, {}
        coefficient_data["fp64"][incoming], coefficient_data["production_oracles"][incoming] = {}, {}
        for kind, offset in (("query", 3), ("key", 4)):
            projected_oracles[incoming][kind] = fp64_projection(kind, projection_values[kind], index, reference[offset], amplitude_scale)
            production_oracles[incoming][kind] = np.asarray(jax.device_get(pullbacks[kind](jnp.asarray(reference[offset], dtype=jnp.float32))))
            case["projection_comparisons"][f"incoming{incoming}_{kind}_fp32_vjp_versus_fp64_contraction"] = compare_projection(
                production_oracles[incoming][kind], projected_oracles[incoming][kind])
            if coefficient_pullbacks:
                coefficient_data["fp64"][incoming].update(fp64_coefficient_parameters(
                    kind, projection_values[kind], index, reference[offset], amplitude_scale, bias_scale))
                coefficient_data["production_oracles"][incoming].update(immutable_snapshot(
                    coefficient_pullbacks[kind](jnp.asarray(reference[offset], dtype=jnp.float32))))
        candidates = {}
        for tile in (64, 32):
            kernel._backward_tile_sizes = lambda *a, tile=tile, **kw: (tile, tile)
            if tile not in replay_operations:
                # Freeze each geometry into its own compiled executable once;
                # subsequent heads reuse it without depending on global selector.
                operation = jax.jit(lambda *xs: kernel._attention_bwd(False, tuple(xs[:5]), xs[5]))
                replay_operations[tile] = operation.lower(*arrays).compile()
            operation = replay_operations[tile]
            candidates[tile] = immutable_snapshot(jax.block_until_ready(operation(*arrays)))
            coefficient_data["replays"][(tile, incoming)] = {}
            label = f"tile{tile}_at_incoming{incoming}_versus_fp64"
            case["comparisons"][label] = compare_local(candidates[tile], reference)
            for kind, offset in (("query", 3), ("key", 4)):
                contracted = np.asarray(jax.device_get(pullbacks[kind](jnp.asarray(candidates[tile][offset]))))
                case["projection_comparisons"][f"{label}_{kind}"] = compare_projection(contracted, projected_oracles[incoming][kind])
                if coefficient_pullbacks:
                    coefficient_data["replays"][(tile, incoming)].update(immutable_snapshot(
                        coefficient_pullbacks[kind](jnp.asarray(candidates[tile][offset]))))
                if tile == incoming:
                    parameter = next(name for name in projection_grads[tile] if f"{kind}_polynomial_kernel" in name)
                    expected = projection_grads[tile][parameter][layer, head]
                    case["projection_comparisons"][f"tile{tile}_{kind}_replay_versus_full_model"] = compare_projection(contracted, expected, tolerance=0.001)
        case["comparisons"][f"tile32_versus_tile64_at_incoming{incoming}"] = compare_local(candidates[32], candidates[64])
        del arrays, reference, candidates, operation
    for failure in failures:
        if (failure["layer"], failure["head"]) != (layer, head):
            continue
        parameter, order = failure["parameter"], failure["order"]
        kind = "query" if "query_polynomial_kernel" in parameter else "key"
        observed = (projection_grads[32][parameter][layer, head, :, order].astype(np.float64)
                    - projection_grads[64][parameter][layer, head, :, order].astype(np.float64))
        predicted = (projected_oracles[32][kind][:, order] - projected_oracles[64][kind][:, order])
        predicted_production = (production_oracles[32][kind][:, order].astype(np.float64)
                                - production_oracles[64][kind][:, order].astype(np.float64))
        attribution = compare_array(predicted, observed, tolerance=ATTRIBUTION_TOLERANCE)
        attribution["explained_fraction_lower_bound"] = (None if attribution["relative_l2_error"] is None else 1 - attribution["relative_l2_error"])
        attribution["production_projection_control"] = compare_array(predicted_production, observed, tolerance=ATTRIBUTION_TOLERANCE)
        case["attribution"].append(dict(parameter=parameter, order=order, **attribution))
    case["accepted"] = (all(v["accepted"] for v in case["comparisons"].values())
                        and all(v["accepted"] for v in case["projection_comparisons"].values())
                        and all(v["accepted"] and v["production_projection_control"]["accepted"] for v in case["attribution"]))
    if coefficient_pullbacks:
        coefficient_outputs[(layer, head)] = coefficient_data
        case["coefficient_parameter_sha256"] = {
            category: {str(key): {name: array_hash(value) for name, value in values.items()}
                       for key, values in entries.items()} for category, entries in coefficient_data.items()}
    return case


def compare_coefficient_groups(layers, heads, outputs, full_grads, failures):
    """Compare model-owned coefficient gradients as vectors across all heads."""
    required = {(g["parameter"], g["layer"], g["order"]) for g in failures}
    rows = []
    for layer in sorted(layers):
        for name in COEFFICIENT_NAMES:
            parameter = next(p for p in full_grads[64] if name in p)
            count = full_grads[64][parameter].shape[-1]
            for order in range(count):
                row = {"parameter": parameter, "layer": layer, "order": order, "heads": list(heads),
                       "comparisons": {}, "required_attribution": (parameter, layer, order) in required,
                       "attribution": None}
                oracles, controls = {}, {}
                for incoming in (64, 32):
                    oracles[incoming] = np.asarray([outputs[(layer, head)]["fp64"][incoming][name][order]
                                                   for head in heads], np.float64)
                    controls[incoming] = np.asarray([outputs[(layer, head)]["production_oracles"][incoming][name][order]
                                                    for head in heads], np.float64)
                    row["comparisons"][f"incoming{incoming}_fp32_vjp_versus_fp64_contraction"] = compare_array(
                        controls[incoming], oracles[incoming], tolerance=GROUP_TOLERANCE)
                    for tile in (64, 32):
                        replay = np.asarray([outputs[(layer, head)]["replays"][(tile, incoming)][name][order]
                                             for head in heads], np.float64)
                        row["comparisons"][f"tile{tile}_at_incoming{incoming}_versus_fp64"] = compare_array(
                            replay, oracles[incoming], tolerance=GROUP_TOLERANCE)
                        if tile == incoming:
                            row["comparisons"][f"tile{tile}_replay_versus_full_model"] = compare_array(
                                replay, full_grads[tile][parameter][layer, :, order], tolerance=0.001)
                if row["required_attribution"]:
                    observed = (full_grads[32][parameter][layer, :, order].astype(np.float64)
                                - full_grads[64][parameter][layer, :, order].astype(np.float64))
                    attribution = compare_array(oracles[32] - oracles[64], observed, tolerance=ATTRIBUTION_TOLERANCE)
                    attribution["explained_fraction_lower_bound"] = (None if attribution["relative_l2_error"] is None
                                                                     else 1 - attribution["relative_l2_error"])
                    attribution["production_projection_control"] = compare_array(
                        controls[32] - controls[64], observed, tolerance=ATTRIBUTION_TOLERANCE)
                    row["attribution"] = attribution
                row["accepted"] = all(v["accepted"] for v in row["comparisons"].values())
                if row["attribution"] is not None:
                    row["accepted"] &= row["attribution"]["accepted"] and row["attribution"]["production_projection_control"]["accepted"]
                rows.append(row)
    return rows


def diagnose(args):
    path, output = args.config.resolve(), args.output.resolve()
    config = load_config(path)
    validate_config(config)
    if (config.model.num_layers != 15 or config.model.num_heads != 14 or config.model.qk_head_dim != 64
            or config.model.attention_degree not in (8, 12) or config.model.attention_coefficient_conditioning != "query_key"
            or config.model.attention_backend != "fused" or config.train.dtype != "bfloat16"
            or config.train.micro_batch_size != 16 or config.data.sequence_length != 2049 or config.model.remat
            or config.data.source != "packed" or config.optim.coefficient_penalty != 0):
        raise ValueError("Requires the unrematerialized full 4B query-key recipe")
    selected_reports = [(p, json.loads(p.read_text())) for p in args.parity_report]
    for selected_path, selected_report in selected_reports:
        if selected_report["input_recipe"]["sha256"] != digest(path):
            raise ValueError("Parity report describes a different frozen recipe")
    # Capture once before choosing cases: fresh compilations can change which
    # small cancelled projection groups fail, even at identical primal inputs.
    heads = tuple(range(config.model.num_heads))
    layers = set(range(config.model.num_layers))
    devices = jax.local_devices()
    if len(devices) != 1 or "B300" not in devices[0].device_kind:
        raise ValueError("Requires exactly one B300")
    source = _execution_provenance(devices[0], args.experiment)
    source["benchmark_source_sha256"] = digest(Path(__file__))
    report = {"format": "query-key-model-roundoff-diagnostic", "schema_version": 1, "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(), "provenance": source,
              "input_recipe": {"path": str(path), "sha256": digest(path), "configuration": config_to_dict(config)},
              "configuration": {"heads_captured": list(heads), "layers_captured": sorted(layers),
                                "case_selection": "historical/current failed projection pairs, all heads of layers with failed coefficient bias/amplitude groups, plus layer14/head2 control",
                                "repeat_acceptance": "exact full gradients; per-leaf differences and original numerical gates reported separately; no unmeasured embedding tolerance",
                                "perturbation_seed": args.perturbation_seed, "local_relative_l2_tolerance": TOLERANCE,
                                "projection_group_tolerance": GROUP_TOLERANCE, "attribution_residual_tolerance": ATTRIBUTION_TOLERANCE,
                                "reference": "independent analytic FP64 attention VJP and FP64 reconstruction of tanh/projection; actual FP32 projection VJP measured separately",
                                "attribution": "oracle(candidate32 incoming dO) - oracle(original64 incoming dO) must explain at least 90% of each failed full-model projection group delta in L2",
                                "limitation": "local replay does not override failed full-model gates; attribution locates error before this block but does not identify the upstream operation responsible"},
              "cases": [], "coefficient_groups": [], "full_model_trials": []}
    report["selected_parity_reports"] = [{"path": str(p.resolve()), "sha256": digest(p)} for p, _ in selected_reports]
    write_report(output, report)
    original_attention, original_selector = kernel._attention, kernel._backward_tile_sizes
    original_query, original_key = projections.project_coefficients, projections.project_key_factors
    started = time.monotonic()
    try:
        print("Loading and checking packed data", flush=True)
        batch_host = next(PackedBatchIterator(config.data.train_manifest, batch_size=16, sequence_length=2049))
        report["batch"] = {"shape": list(batch_host.shape), "sha256": array_hash(batch_host), "manifest_sha256": digest(Path(config.data.train_manifest))}
        print("Initializing exact recipe and deterministic qualification perturbations", flush=True)
        state = create_state(config, jax.random.PRNGKey(config.train.seed))
        jax.block_until_ready(state)
        params, report["parameter_perturbations"] = perturbed_parameters(immutable_snapshot(state.params), args.perturbation_seed)
        report["parameter_sha256"], _ = parameter_digest(params)
        state = state.replace(params=jax.tree.map(lambda x: jax.device_put(np.array(x, copy=True), may_alias=False), params))
        batch = jax.device_put(batch_host, may_alias=False)
        rng = jax.random.fold_in(jax.random.PRNGKey(config.train.seed), 1)
        captures, projection_inputs, projection_grads = capture_models(
            state, batch, rng, heads, layers, report, output, original_query, original_key)
        current_failures = failed_projection_groups(report) + failed_coefficient_groups(report)
        selected_failures = [dict(group, source_report=str(p.resolve())) for p, old in selected_reports
                             for group in failed_projection_groups(old) + failed_coefficient_groups(old)]
        identity = lambda group: (group["parameter"], group["layer"], group.get("head"), group["order"])
        failure_union = {identity(group): group for group in selected_failures}
        failure_union.update({identity(group): group for group in current_failures})
        failures = list(failure_union.values())
        projection_failures = [g for g in failures if g.get("head") is not None]
        coefficient_failures = [g for g in failures if g.get("head") is None]
        coefficient_layers = {g["layer"] for g in coefficient_failures}
        cases = sorted({(g["layer"], g["head"]) for g in projection_failures}
                       | {(layer, head) for layer in coefficient_layers for head in heads} | {(14, 2)})
        report["configuration"]["cases"] = [list(case) for case in cases]
        report["configuration"]["coefficient_layers"] = sorted(coefficient_layers)
        report["selected_report_failures_reproduced"] = {identity(group) for group in selected_failures} <= {identity(group) for group in current_failures}
        report["original_failure_reproduction"] = []
        for original in selected_failures:
            parameter, layer, head, order = identity(original)
            select = ((layer, slice(None), order) if head is None else (layer, head, slice(None), order))
            metrics = compare_array(projection_grads[32][parameter][select], projection_grads[64][parameter][select],
                                    tolerance=GROUP_TOLERANCE)
            report["original_failure_reproduction"].append({
                "parameter": parameter, "layer": layer, "head": head, "order": order,
                "source_report": original["source_report"],
                "original_relative_l2_error": original["relative_l2_error"],
                "original_reference_l2": original["reference_l2"], "current": metrics})
        uncovered = [g for g in projection_failures if (g["layer"], g["head"]) not in cases]
        uncovered += [g for g in coefficient_failures if any((g["layer"], head) not in cases for head in heads)]
        unhandled = [dict(parameter=leaf["parameter"], **group) for leaf in report["gradients"]["leaves"]
                     if not any(name in leaf["parameter"] for name in (*PROJECTION_NAMES, *COEFFICIENT_NAMES))
                     for group in leaf.get("groups", {}).get("failed", [])]
        report["failed_groups_covered"], report["uncovered_failures"] = not uncovered, uncovered
        report["unhandled_nonprojection_groups"] = unhandled
        report["full_model_failed_projection_groups"] = projection_failures
        report["full_model_failed_coefficient_groups"] = coefficient_failures
        trials = report["full_model_trials"]
        report["logits_exact"] = trials[0]["evaluations"][0]["logits_sha256"] == trials[1]["evaluations"][0]["logits_sha256"]
        report["loss_absolute_difference"] = abs(trials[0]["evaluations"][0]["loss"] - trials[1]["evaluations"][0]["loss"])
        report["captured_layers"] = sorted(layers)
        report["capture_layer_identity"] = "Explicit IDs assigned during the single forward trace; callbacks may arrive in any order"
        after_digest, _ = parameter_digest(state.params)
        if after_digest != report["parameter_sha256"]:
            raise RuntimeError("Capture modified immutable model parameters")
        report["same_parameters_and_batch_as_parity"] = all(
            report["parameter_sha256"] == old.get("compared_parameter_sha256", old.get("parameter_sha256"))
            and report["batch"]["sha256"] == old["batch"]["sha256"] for _, old in selected_reports)
        if not report["same_parameters_and_batch_as_parity"]:
            raise RuntimeError("Capture did not reproduce qualification parameters and data")
        write_report(output, report)
        kernel._attention = original_attention
        projections.project_coefficients, projections.project_key_factors = original_query, original_key
        del state, batch, rng, params
        jax.clear_caches()
        gc.collect()
        coefficient_outputs, completed_coefficient_layers, replay_operations = {}, set(), {}
        oracle_operation = jax.jit(fp64_gradients)
        for layer, head in cases:
            report["cases"].append(diagnose_case(layer, head, heads, captures, projection_inputs, projection_grads,
                                                 projection_failures, config.model.attention_coefficient_amplitude_scale,
                                                 config.model.attention_coefficient_scale, coefficient_outputs,
                                                 coefficient_layers, replay_operations, oracle_operation))
            if (layer in coefficient_layers and layer not in completed_coefficient_layers
                    and all((layer, h) in coefficient_outputs for h in heads)):
                report["coefficient_groups"].extend(compare_coefficient_groups(
                    {layer}, heads, coefficient_outputs, projection_grads, coefficient_failures))
                completed_coefficient_layers.add(layer)
            write_report(output, report)
        report["source_recipe_unchanged"] = digest(path) == report["input_recipe"]["sha256"]
        report["full_model_leaf_gates_pass"] = all(
            leaf["all_finite"] and leaf["relative_l2_error"] is not None and leaf["relative_l2_error"] <= TOLERANCE
            and leaf["max_abs_error"] <= 5 * TOLERANCE * leaf["reference_max_abs"] for leaf in report["gradients"]["leaves"])
        report["status"] = ("complete" if report["source_recipe_unchanged"] and report["failed_groups_covered"]
                            and not unhandled
                            and all(t["deterministic"] and t["instrumentation_array_exact"] for t in trials)
                            and report["full_model_leaf_gates_pass"]
                            and report["logits_exact"] and report["loss_absolute_difference"] <= LOSS_TOLERANCE
                            and all(case["accepted"] for case in report["cases"])
                            and all(group["accepted"] for group in report["coefficient_groups"])
                            and completed_coefficient_layers == coefficient_layers else "failed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        kernel._attention, kernel._backward_tile_sizes = original_attention, original_selector
        projections.project_coefficients, projections.project_key_factors = original_query, original_key
        jax.clear_caches()
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        write_report(output, report)
    print("MODEL_ROUNDOFF " + json.dumps({"status": report["status"], "cases": len(report["cases"]),
                                         "deterministic": all(t["deterministic"] for t in trials)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="transformer_200m_4b")
    parser.add_argument("--perturbation-seed", type=int, default=86421)
    parser.add_argument("--parity-report", type=Path, action="append", default=[],
                        help="Include failed groups from this recipe's qualification report; repeat for multiple reports")
    if diagnose(parser.parse_args())["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
