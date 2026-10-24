"""Compare full-model query-key gradients under production and original tiles.

Use the supplied frozen recipe, its initialization, and one real packed-data
microbatch. Deterministic nonzero coefficient amplitudes and nonuniform gains
exercise every coefficient projection. This diagnostic compares the candidate
with the original implementation; the separate kernel tests supply independent
FP64/reference correctness checks. No optimizer update or checkpoint is written.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scaffolds.transformer.attention.backends import fused_query_key_polynomial as kernel
from scaffolds.transformer.benchmarks.benchmark_attention import write_report
from scaffolds.transformer.benchmarks.benchmark_train_step import parameter_digest
from scaffolds.transformer.benchmarks.benchmark_training_throughput import _execution_provenance, digest
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.loop import cross_entropy_loss
from scaffolds.transformer.training.randomness import model_forward_rngs
from scaffolds.transformer.training.state import create_state


LEAF_TOLERANCE = 0.025
GROUP_TOLERANCE = 0.05
LOGIT_TOLERANCE = 0.005
LOSS_TOLERANCE = 0.002
GAIN_NAMES = ("qk_norm_q_gain", "qk_norm_k_gain", "output_norm_scale")
COEFFICIENT_NAMES = ("query_polynomial_bias", "query_polynomial_amplitude", "key_polynomial_amplitude")
PROJECTION_NAMES = ("query_polynomial_kernel", "key_polynomial_kernel")


def name_for_path(path):
    return jax.tree_util.keystr(path)


def immutable_snapshot(tree):
    def snapshot(value):
        array = np.array(jax.device_get(value), copy=True)
        array.setflags(write=False)
        return array
    return jax.tree.map(snapshot, tree)


def perturbed_parameters(params, seed):
    random = np.random.default_rng(seed)
    changed = []

    def perturb(path, array):
        name = name_for_path(path)
        if any(part in name for part in ("query_polynomial_amplitude", "key_polynomial_amplitude")):
            value = random.uniform(0.1, 0.2, size=array.shape).astype(np.float32)
        elif any(part in name for part in (*GAIN_NAMES, "query_polynomial_bias")):
            value = np.asarray(array) * random.uniform(0.9, 1.1, size=array.shape).astype(np.float32)
        else:
            return array
        value.setflags(write=False)
        changed.append({"parameter": name, "shape": list(array.shape),
                        "min": float(value.min()), "max": float(value.max())})
        return value

    result = jax.tree_util.tree_map_with_path(perturb, params)
    if len(changed) != 6:
        raise ValueError(f"Expected six coefficient/gain arrays, found {len(changed)}: {changed}")
    return result, changed


def full_objective():
    # A fresh function object and cleared JAX caches isolate each launch geometry.
    @nnx.jit
    def operation(state, batch, rng):
        def loss_fn(params):
            model = nnx.merge(state.graphdef, params)
            logits = model(batch[:, :-1], deterministic=False, rngs=model_forward_rngs(rng))
            return cross_entropy_loss(logits, batch[:, 1:]), logits
        # Same shared production NLL and model call as training.loss_and_grads;
        # retain full logits as auxiliary output for complete forward comparison.
        return nnx.value_and_grad(loss_fn, has_aux=True)(state.params)
    return operation


def compare_array(actual, reference, *, tolerance, require_nonzero=True):
    """Chunked host FP64 reductions keep full-vocabulary logits memory bounded."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {reference.shape}")
    error_sq = norm_sq = max_error = max_reference = 0.0
    finite = exact = True
    # ravel may copy strided mechanism slices, which are small compared with logits.
    observed, expected = actual.ravel(), reference.ravel()
    for start in range(0, expected.size, 1 << 20):
        a = observed[start:start + (1 << 20)].astype(np.float64)
        b = expected[start:start + (1 << 20)].astype(np.float64)
        finite &= bool(np.isfinite(a).all() and np.isfinite(b).all())
        if not finite:
            return {"accepted": False, "all_finite": False, "array_equal": False,
                    "reference_nonzero": False, "reference_l2": None,
                    "relative_l2_error": None, "max_abs_error": None,
                    "reference_max_abs": None, "relative_l2_tolerance": tolerance,
                    "relative_max_tolerance": 5 * tolerance}
        exact &= bool(np.array_equal(a, b))
        delta = a - b
        error_sq += float(np.dot(delta, delta))
        norm_sq += float(np.dot(b, b))
        max_error = max(max_error, float(np.max(np.abs(delta), initial=0)))
        max_reference = max(max_reference, float(np.max(np.abs(b), initial=0)))
    nonzero = norm_sq > 0
    relative = (error_sq / norm_sq) ** 0.5 if nonzero else None
    accepted = finite and (nonzero or not require_nonzero)
    accepted &= error_sq <= tolerance ** 2 * norm_sq
    # The max gate prevents a localized error from hiding in a large parameter leaf.
    accepted &= max_error <= 5 * tolerance * max_reference
    return {"accepted": bool(accepted), "all_finite": bool(finite), "array_equal": bool(exact),
            "reference_nonzero": bool(nonzero), "reference_l2": norm_sq ** 0.5,
            "relative_l2_error": relative, "max_abs_error": max_error,
            "reference_max_abs": max_reference, "relative_l2_tolerance": tolerance,
            "relative_max_tolerance": 5 * tolerance}


def compare_gradients(actual, reference):
    pairs_a = jax.tree_util.tree_flatten_with_path(actual)[0]
    pairs_b = jax.tree_util.tree_flatten_with_path(reference)[0]
    if [name_for_path(p) for p, _ in pairs_a] != [name_for_path(p) for p, _ in pairs_b]:
        raise ValueError("Gradient tree paths differ")
    leaves = []
    for (path, observed), (_, expected) in zip(pairs_a, pairs_b, strict=True):
        name = name_for_path(path)
        result = {"parameter": name, "shape": list(expected.shape),
                  **compare_array(observed, expected, tolerance=LEAF_TOLERANCE)}
        groups = []
        if any(part in name for part in PROJECTION_NAMES):
            # [layer, head, hidden, order]: inspect each layer/head/order vector.
            for layer in range(expected.shape[0]):
                for head in range(expected.shape[1]):
                    for order in range(expected.shape[-1]):
                        checked = compare_array(observed[layer, head, :, order], expected[layer, head, :, order],
                                                tolerance=GROUP_TOLERANCE)
                        groups.append({"layer": layer, "head": head, "order": order, **checked})
        elif any(part in name for part in COEFFICIENT_NAMES):
            # Scalar head gradients can cancel; compare each layer/order across heads.
            for layer in range(expected.shape[0]):
                for order in range(expected.shape[-1]):
                    checked = compare_array(observed[layer, :, order], expected[layer, :, order],
                                            tolerance=GROUP_TOLERANCE)
                    groups.append({"layer": layer, "order": order, **checked})
        elif any(part in name for part in GAIN_NAMES):
            for layer in range(expected.shape[0]):
                checked = compare_array(observed[layer], expected[layer], tolerance=GROUP_TOLERANCE)
                groups.append({"layer": layer, **checked})
        if groups:
            # Keep compact maxima and failures, not thousands of passing group rows.
            result["groups"] = {
                "count": len(groups), "nonzero_count": sum(g["reference_nonzero"] for g in groups),
                "max_relative_l2_error": max(g["relative_l2_error"] or 0.0 for g in groups),
                "failed": [g for g in groups if not g["accepted"]],
            }
            result["accepted"] &= all(g["accepted"] for g in groups)
        leaves.append(result)
    return {"accepted": all(leaf["accepted"] for leaf in leaves), "leaf_count": len(leaves),
            "parameter_count": sum(int(np.prod(leaf["shape"])) for leaf in leaves),
            "max_leaf_relative_l2_error": max(leaf["relative_l2_error"] or 0.0 for leaf in leaves),
            "leaves": leaves}


def qualify(args):
    path, output = args.config.resolve(), args.output.resolve()
    recipe_digest = digest(path)
    config = load_config(path)
    validate_config(config)
    model = config.model
    if (model.num_layers, model.hidden_size, model.num_heads, model.qk_head_dim, model.value_head_dim) != (15, 896, 14, 64, 64):
        raise ValueError("Requires the complete 15-layer, 14-head 200M model")
    if (model.attention_coefficient_conditioning != "query_key" or model.attention_degree not in (8, 12)
            or model.attention_backend != "fused" or config.train.dtype != "bfloat16"
            or config.data.source != "packed" or config.data.sequence_length != 2049
            or config.optim.coefficient_penalty != 0 or config.train.micro_batch_size != 16):
        raise ValueError("Requires the frozen BF16 query-key 4B recipe and production microbatch16")
    devices = jax.local_devices()
    if len(devices) != 1 or "B300" not in devices[0].device_kind:
        raise ValueError("Requires exactly one allocated B300")
    selected = kernel._backward_tile_sizes(64, 64, degree=model.attention_degree, time=2048,
                                           dtype=jnp.bfloat16, interpret=False)
    if selected != (32, 32):
        raise ValueError(f"Production selector did not choose the candidate: {selected}")
    report = {
        "format": "query-key-full-model-gradient-qualification", "schema_version": 1, "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": _execution_provenance(devices[0], args.experiment),
        "input_recipe": {"path": str(path), "sha256": recipe_digest, "configuration": config_to_dict(config)},
        "configuration": {"microbatch_size": 16, "sequence_length": 2049, "perturbation_seed": args.perturbation_seed,
                          "comparison": "full model at identical immutable parameters, real packed tokens, and RNG; candidate versus original backward tiles",
                          "reference_limit": "original implementation, not an independent gradient oracle",
                          "gradient_leaf_relative_l2_tolerance": LEAF_TOLERANCE,
                          "mechanism_group_relative_l2_tolerance": GROUP_TOLERANCE,
                          "logit_relative_l2_tolerance": LOGIT_TOLERANCE, "loss_absolute_tolerance": LOSS_TOLERANCE},
        "trials": [],
    }
    report["provenance"]["benchmark_source_sha256"] = digest(Path(__file__))
    write_report(output, report)
    original_selector = kernel._backward_tile_sizes
    started = time.monotonic()
    try:
        batch_host = next(PackedBatchIterator(config.data.train_manifest, batch_size=16, sequence_length=2049))
        batch_host.setflags(write=False)
        report["batch"] = {"shape": list(batch_host.shape), "sha256": hashlib.sha256(batch_host.tobytes()).hexdigest(),
                           "manifest_path": str(config.data.train_manifest), "manifest_sha256": digest(Path(config.data.train_manifest)),
                           "start_sequence": 0}
        state = create_state(config, jax.random.PRNGKey(config.train.seed))
        jax.block_until_ready(state)
        reference_params = immutable_snapshot(state.params)
        report["initial_parameter_sha256"], report["parameter_count"] = parameter_digest(reference_params)
        reference_params, report["parameter_perturbations"] = perturbed_parameters(reference_params, args.perturbation_seed)
        report["compared_parameter_sha256"], _ = parameter_digest(reference_params)
        rng_host = np.asarray(jax.random.fold_in(jax.random.PRNGKey(config.train.seed), 1))
        report["forward_rng"] = rng_host.tolist()
        outputs = {}
        for label in ("candidate32", "original64"):
            kernel._backward_tile_sizes = original_selector if label == "candidate32" else lambda *a, **kw: (64, 64)
            jax.clear_caches()
            gc.collect()
            params = jax.tree.map(lambda a: jax.device_put(np.array(a, copy=True), may_alias=False), reference_params)
            trial_state = state.replace(params=params)
            batch = jax.device_put(np.array(batch_host, copy=True), may_alias=False)
            rng = jax.device_put(np.array(rng_host, copy=True), may_alias=False)
            operation = full_objective()
            print(f"degree={model.attention_degree} full_model={label}: compiling and evaluating", flush=True)
            start = time.monotonic()
            result = jax.block_until_ready(operation(trial_state, batch, rng))
            elapsed = time.monotonic() - start
            if not bool(jnp.isfinite(result[0][0])):
                raise FloatingPointError(f"Nonfinite full-model NLL for {label}")
            outputs[label] = immutable_snapshot(result)
            after_digest, _ = parameter_digest(trial_state.params)
            if after_digest != report["compared_parameter_sha256"]:
                raise RuntimeError("Gradient diagnostic mutated input parameters")
            report["trials"].append({"id": label, "backward_tiles": [32, 32] if label == "candidate32" else [64, 64],
                                     "seconds_including_compilation": elapsed, "loss": float(result[0][0]),
                                     "parameter_sha256_after": after_digest})
            write_report(output, report)
            del result, trial_state, params, batch, rng, operation
        (candidate_loss, candidate_logits), candidate_grads = outputs["candidate32"]
        (original_loss, original_logits), original_grads = outputs["original64"]
        loss_error = abs(float(candidate_loss) - float(original_loss))
        report["loss"] = {"candidate": float(candidate_loss), "original": float(original_loss),
                          "absolute_error": loss_error, "accepted": bool(np.isfinite(loss_error) and loss_error <= LOSS_TOLERANCE)}
        report["logits"] = compare_array(candidate_logits, original_logits, tolerance=LOGIT_TOLERANCE)
        report["gradients"] = compare_gradients(candidate_grads, original_grads)
        report["source_recipe_unchanged"] = digest(path) == recipe_digest
        accepted = report["source_recipe_unchanged"] and all(report[key]["accepted"] for key in ("loss", "logits", "gradients"))
        report["status"] = "complete" if accepted else "failed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        kernel._backward_tile_sizes = original_selector
        jax.clear_caches()
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        write_report(output, report)
    print("FULL_MODEL_PARITY " + json.dumps({"status": report["status"], "degree": model.attention_degree,
          "loss": report["loss"], "logit_relative_l2_error": report["logits"]["relative_l2_error"],
          "max_gradient_leaf_relative_l2_error": report["gradients"]["max_leaf_relative_l2_error"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="transformer_200m_4b")
    parser.add_argument("--perturbation-seed", type=int, default=86421)
    report = qualify(parser.parse_args())
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
