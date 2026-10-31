"""Bounded selected-checkpoint coefficient VJP and functional-conditioning replay.

One predeclared fitting sequence, all layers, and all causal keys. Uniform query
samples and deliberately selected minimum-RMS extremes are reported separately.
No parameters or optimizer state are updated. Numerical discrepancies are measured,
not classified using a newly invented pass tolerance.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
import yaml

from scaffolds.records import load_run, update_record, validate_record
from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import parent_config, restore_parent, split_generator_parameters
from experiments.transformer_pp.results.exploratory.coefficient_oracle.inspect_batch_fit import (
    selected_checkpoint, sha256,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUTPUT = "coefficient_vjp_probe.json"
PROBABILITIES = [0., .1, .5, .9, .99, 1.]


def rms_jacobian(y, gain, eps):
    """Independent FP64 derivative of gain*y/sqrt(mean(y**2)+eps)."""
    y, gain = np.asarray(y, np.float64), np.asarray(gain, np.float64)
    denominator = np.mean(y * y) + eps
    return gain[:, None] * (
        np.eye(len(y)) / np.sqrt(denominator)
        - np.outer(y, y) / (len(y) * denominator ** 1.5))


def basis_outputs(q, k, v, query_positions, degree):
    """Independent full-causal-key T[q,value,order] for stored u_r=r!c_r.

    Inputs are the actual rounded post-QK-normalization/post-RoPE operands of one
    head. FP64 dot products deliberately supply a smooth numerical reference.
    Absolute query positions retain the full original causal mask.
    """
    q, k, v = (np.asarray(value, np.float64) for value in (q, k, v))
    positions = np.asarray(query_positions, np.int64)
    scores = q[positions] @ k.T / math.sqrt(q.shape[-1])
    causal = np.arange(len(k))[None] <= positions[:, None]
    powers = np.stack([np.where(causal, scores ** order / math.factorial(order), 0.)
                       for order in range(degree + 1)], axis=-1)
    return np.einsum("qkr,kd->qdr", powers, v, optimize=True), scores, causal


def tangent_geometry(matrix, coordinates, y, gain, eps):
    """Functional Jacobian in coefficient and angular tangent coordinates.

    SVD rank uses only NumPy's conventional FP64 numerical-rank threshold, which
    is recorded. It is not a scientific claim that small singular values vanish.
    """
    coordinates = np.asarray(coordinates, np.float64)
    radius = float(np.linalg.norm(coordinates))
    if not radius > 0:
        raise ValueError("Coefficient direction is undefined at zero radius")
    direction = coordinates / radius
    # Right singular vectors after the first span direction-perpendicular space.
    tangent = np.linalg.svd(direction[None], full_matrices=True)[2][1:].T
    jacobian = rms_jacobian(y, gain, eps) @ matrix
    singular = np.linalg.svd(jacobian @ tangent, compute_uv=False)
    threshold = float(max(matrix.shape) * np.finfo(np.float64).eps * singular[0]) if len(singular) else 0.
    rank = int(np.count_nonzero(singular > threshold))
    radial = jacobian @ coordinates
    return {
        "coefficient_radius": radius,
        "coefficient_tangent_singular_values": singular.tolist(),
        "direction_tangent_singular_values": (radius * singular).tolist(),
        "tangent_numerical_rank": rank, "tangent_rank_threshold": threshold,
        "tangent_condition_number": float(singular[0] / singular[-1]) if rank == len(singular) and singular[-1] > 0 else None,
        "log_radius_response_norm": float(np.linalg.norm(radial)),
        "log_radius_to_largest_direction_response": float(np.linalg.norm(radial) / (radius * singular[0])) if singular[0] > 0 else None,
    }


def error_metrics(actual, reference):
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite derivative or reference")
    actual_norm, reference_norm = float(np.linalg.norm(actual)), float(np.linalg.norm(reference))
    error = float(np.linalg.norm(actual - reference))
    return {"absolute_l2_error": error, "reference_l2_norm": reference_norm,
            "actual_l2_norm": actual_norm,
            "relative_l2_error": error / reference_norm if reference_norm > 0 else None,
            "cosine": float(np.dot(actual.ravel(), reference.ravel()) / (actual_norm * reference_norm))
            if actual_norm > 0 and reference_norm > 0 else None}


def make_capture(graph, model_config, *, backend="mosaic"):
    """Capture operands and instrumented-execution task cotangents without replacing coefficients.

    Zero additive probes enter after each head's output RMSNorm. Differentiating
    their values retains every downstream native block and generator-input path.
    All hooks are process-local tracing instrumentation; scaffold files are untouched.
    """
    native_core, native_scale = multihead.attention_core, multihead.scale_attention_output

    def objective(probes, params, tokens):
        captured, gains = [], []

        def core(q, k, v, **kwargs):
            if kwargs["backend"] != backend or kwargs.get("return_diagnostics"):
                raise ValueError(f"Expected ordinary {backend} query attention")
            coefficients = kwargs["query_polynomial_coefficients"]
            output = native_core(q, k, v, **kwargs)
            captured.append((q, k, v, coefficients, output))
            return output

        def scaling(y, mode, *, output_norm_scale=None, eps=1e-6):
            if mode != "head_rms_norm" or eps != model_config.rms_norm_eps:
                raise ValueError("Unexpected output normalization")
            index = len(gains)
            gains.append(output_norm_scale)
            return native_scale(y, mode, output_norm_scale=output_norm_scale, eps=eps) + probes[index]

        with patch.object(multihead, "attention_core", core), patch.object(multihead, "scale_attention_output", scaling):
            logits = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True)
        if len(captured) != model_config.num_layers or len(gains) != model_config.num_layers:
            raise ValueError("Native forward did not visit every expected attention layer")
        loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:]).mean()
        return loss, (tuple(captured), tuple(gains), logits)

    capture = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))

    @jax.jit
    def compare(params, tokens, instrumented_logits):
        native = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True)
        difference = native.astype(jnp.float32) - instrumented_logits.astype(jnp.float32)
        nll = optax.softmax_cross_entropy_with_integer_labels(native.astype(jnp.float32), tokens[:, 1:]).mean()
        return {"native_nll": nll, "logit_max_absolute_error": jnp.max(jnp.abs(difference)),
                "logit_relative_rms_error": jnp.linalg.norm(difference) / jnp.maximum(jnp.linalg.norm(native.astype(jnp.float32)), 1e-30)}
    return capture, compare


def assess_parity(parity, instrumented_nll, recorded_nll, tolerance):
    """Preserve established bounds and identify which one actually failed."""
    result = {**parity, "instrumented_nll": float(instrumented_nll),
              "recorded_sequence_nll": float(recorded_nll), "established_nll_atol": float(tolerance)}
    result["instrumented_minus_native_nll"] = float(instrumented_nll - parity["native_nll"])
    result["native_minus_recorded_nll"] = float(parity["native_nll"] - recorded_nll)
    finite = all(math.isfinite(value) for value in result.values())
    checks = {"finite_measurements": finite,
              "native_checkpoint_nll": finite and abs(result["native_minus_recorded_nll"]) <= tolerance,
              "instrumented_native_nll": finite and abs(result["instrumented_minus_native_nll"]) <= tolerance}
    result["checks"] = checks
    result["failed_checks"] = [name for name, passed in checks.items() if not passed]
    result["logit_errors_are_measured_without_an_added_acceptance_threshold"] = True
    return result


def precision_control_graph(config, source):
    """Same checkpoint arrays with FP32 module arithmetic and blocked attention.

    Mosaic requires BF16 inputs, so this is a precision/backend sensitivity
    control, not an isolated change to Mosaic's arithmetic.
    """
    control_config = replace(config.model, attention_backend="blocked")
    model = build_model(control_config, nnx.Rngs(params=config.train.seed, dropout=config.train.seed))
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = jnp.float32
    graph, template = nnx.split(model, nnx.Param)
    signature = lambda state: {path: (value.shape, value.dtype) for path, value in nnx.to_flat_state(state)}
    if signature(template) != signature(source):
        raise ValueError("FP32 control graph does not accept the identical checkpoint parameter tree")
    return graph, control_config


def precision_capture_control(capture, compare, probes, params, tokens):
    """Run the same instrumented value-and-gradient capture, keeping only parity."""
    with jax.default_matmul_precision("highest"):
        (loss, (_, _, logits)), _ = capture(probes, params, tokens)
        parity = {key: float(value) for key, value in jax.device_get(compare(params, tokens, logits)).items()}
    parity["instrumented_nll"] = float(loss)
    parity["instrumented_minus_native_nll"] = float(loss) - parity["native_nll"]
    if not all(math.isfinite(value) for value in parity.values()):
        raise ValueError(f"FP32 capture control produced nonfinite parity: {parity}")
    return {"backend": "blocked", "module_dtype": "float32", "matmul_precision": "highest",
            "same_checkpoint_parameter_arrays": True, **parity,
            "interpretation": "Same before/after checkpoint arrays and sequence. Native and instrumented graphs both use FP32 module arithmetic with blocked attention. The loss is not required to equal BF16 checkpoint NLL; changing the backend as well as arithmetic prevents attributing any difference to one precise cause."}


@jax.jit
def native_coefficient_vjp(q, k, v, coefficients, recorded_y, gain, cotangent, eps):
    # Keep the recorded forward output when differentiating RMS. This separates
    # the kernel VJP from any fresh-compilation difference in its forward output.
    _, norm_pullback = jax.vjp(
        lambda y: scale_attention_output(y, "head_rms_norm", output_norm_scale=gain, eps=eps), recorded_y)
    raw_cotangent = norm_pullback(cotangent)[0]
    replay_y, pullback = jax.vjp(lambda c: query_polynomial_attention(q, k, v, c), coefficients)
    return replay_y, raw_cotangent, pullback(raw_cotangent)[0]


def sampled_measurements(captured, gains, cotangents, *, eps, uniform_count):
    rows, replay_errors = [], []
    for layer, ((q, k, v, coefficients, y), gain, cotangent) in enumerate(zip(captured, gains, cotangents, strict=True)):
        replay, raw_cotangent, coefficient_vjp = native_coefficient_vjp(q, k, v, coefficients, y, gain, cotangent, eps)
        q, k, v, coefficients, y, gain, cotangent, replay, raw_cotangent, coefficient_vjp = (
            np.asarray(value, np.float64) for value in jax.device_get(
                (q, k, v, coefficients, y, gain, cotangent, replay, raw_cotangent, coefficient_vjp)))
        if any(not np.isfinite(value).all() for value in (q, k, v, coefficients, y, gain, cotangent, raw_cotangent, coefficient_vjp)):
            raise ValueError(f"Nonfinite native captured tensor at layer {layer}")
        replay_errors.append({"layer": layer, **error_metrics(replay, y)})
        degree = coefficients.shape[-1] - 1
        factorials = np.asarray([math.factorial(order) for order in range(degree + 1)])
        uniform = np.unique(np.linspace(0, q.shape[1] - 1, uniform_count, dtype=int))
        for head in range(q.shape[2]):
            raw_square = np.mean(y[0, :, head] ** 2, axis=-1)
            minimum = int(np.argmin(raw_square))
            positions = np.union1d(uniform, [minimum])
            matrices, _, _ = basis_outputs(q[0, :, head], k[0, :, head], v[0, :, head], positions, degree)
            for position, matrix in zip(positions, matrices, strict=True):
                coordinates = coefficients[0, position, head] * factorials
                recorded = y[0, position, head]
                smooth = matrix @ coordinates
                post_cotangent = cotangent[0, position, head]
                raw = raw_cotangent[0, position, head]
                actual = coefficient_vjp[0, position, head] / factorials
                independent_kernel = matrix.T @ raw
                independent_raw = rms_jacobian(recorded, gain[head], eps).T @ post_cotangent
                matched_output_gradient = matrix.T @ independent_raw
                smooth_gradient = matrix.T @ rms_jacobian(smooth, gain[head], eps).T @ post_cotangent
                geometry = tangent_geometry(matrix, coordinates, recorded, gain[head], eps)
                smooth_geometry = tangent_geometry(matrix, coordinates, smooth, gain[head], eps)
                rows.append({
                    "layer": layer, "head": head, "query_position": int(position),
                    "uniform_sample": bool(position in uniform), "minimum_rms_sample": bool(position == minimum),
                    "causal_key_count": int(position + 1), "raw_output_rms": float(np.sqrt(raw_square[position])),
                    "epsilon_fraction_of_rms_denominator": float(eps / (raw_square[position] + eps)),
                    "kernel_vjp_error_given_recorded_raw_cotangent": error_metrics(actual, independent_kernel),
                    "rms_backward_rounding_error": error_metrics(raw, independent_raw),
                    "forward_rounding_gradient_sensitivity": error_metrics(matched_output_gradient, smooth_gradient),
                    "composed_vjp_error_against_smooth_fp64": error_metrics(actual, smooth_gradient),
                    "forward_error_against_smooth_fp64": error_metrics(recorded, smooth),
                    "native_output_geometry": geometry,
                    "smooth_fp64_geometry": smooth_geometry,
                })
        print(f"Coefficient VJP and conditioning: layer {layer + 1}/{len(captured)}", flush=True)
    return rows, replay_errors


def summarize(rows):
    summary = {}
    for sample, flag in (("uniform", "uniform_sample"), ("minimum_rms", "minimum_rms_sample")):
        selected = [row for row in rows if row[flag]]
        values = {}
        for metric in ("kernel_vjp_error_given_recorded_raw_cotangent", "rms_backward_rounding_error",
                       "forward_rounding_gradient_sensitivity", "composed_vjp_error_against_smooth_fp64"):
            finite = [row[metric]["relative_l2_error"] for row in selected if row[metric]["relative_l2_error"] is not None]
            values[metric] = {"relative_error_quantiles": np.quantile(finite, PROBABILITIES).tolist() if finite else [],
                              "zero_reference_count": len(selected) - len(finite)}
        values["raw_output_rms_quantiles"] = np.quantile([row["raw_output_rms"] for row in selected], PROBABILITIES).tolist()
        values["epsilon_fraction_quantiles"] = np.quantile([row["epsilon_fraction_of_rms_denominator"] for row in selected], PROBABILITIES).tolist()
        conditions = [row["native_output_geometry"]["tangent_condition_number"] for row in selected
                      if row["native_output_geometry"]["tangent_condition_number"] is not None]
        values["finite_tangent_condition_quantiles"] = np.quantile(conditions, PROBABILITIES).tolist() if conditions else []
        values["rank_deficient_count"] = len(selected) - len(conditions)
        summary[sample] = {"sample_count": len(selected), **values}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--uniform-queries", type=int, default=16)
    parser.add_argument("--record-parity-discrepancies", action="store_true",
        help="Diagnostic only: retain instrumented-graph derivatives when capture/native NLL parity exceeds its established bound; native checkpoint replay must still pass")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve(); output = run / OUTPUT
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Use one qualified GPU for the bounded replay")
    started = time.monotonic()
    recipe = yaml.safe_load((run / "config.yaml").read_text()); oracle = recipe["oracle"]
    if oracle["parameterization"] != "query_coefficient_generator":
        raise ValueError("This bounded replay inspects the original query generator only")
    record = load_run(run); completion = json.loads((run / "completion.json").read_text())
    if record["status"] != "complete" or completion["status"] != "complete":
        raise ValueError("The fitting run must be complete before checkpoint analysis")
    validate_record(run, checkpoint_hashes=False)
    checkpoint, identity = selected_checkpoint(run, record, completion)
    parent_path = Path(oracle["parent_config"])
    if sha256(parent_path) != oracle["parent_config_sha256"]:
        raise ValueError("Frozen parent recipe changed")
    validate_record(parent_path.parent, checkpoint_hashes=False)
    parent_record = load_run(parent_path.parent)
    if parent_record["selected_checkpoint"]["sha256"] != oracle["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint identity changed")
    config = parent_config(parent_path)
    if (config.model.remat or config.model.attention_normalization or config.model.attention_backend != "mosaic"
            or config.model.attention_degree != 4 or not config.model.qk_norm_head_scale
            or config.model.attention_coefficient_scale != "factorial"
            or config.model.attention_coefficient_amplitude_scale != "factorial"
            or config.model.attention_coefficient_readout != "diagonal"
            or config.model.attention_output_scaling != "head_rms_norm"
            or identity["parent_run_id"] != oracle["parent_run_id"]):
        raise ValueError("Unexpected diagnostic source configuration")
    if sha256(config.data.eval_manifest) != oracle["eval_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    count = int(oracle["sequence_count"])
    if not 0 <= args.sequence_index < count or not 1 <= args.uniform_queries <= 32:
        raise ValueError("Invalid bounded sample request")
    batch = next(iter(PackedBatchIterator(config.data.eval_manifest, batch_size=count,
        sequence_length=config.data.sequence_length, start_sequence=oracle["data_start"])))
    if hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest() != oracle["batch_sha256"]:
        raise ValueError("Fitting tokens changed")
    tokens = jax.device_put(batch[args.sequence_index:args.sequence_index + 1])
    payload = (run / "metrics.jsonl").read_bytes()
    metrics = {row["step"]: row for line in payload.splitlines() if line.strip()
               for row in (json.loads(line),) if row.get("event") == "train"}
    expected = {"before": metrics[0]["per_sequence_nll"][args.sequence_index],
                "after": metrics[completion["best_step"]]["per_sequence_nll"][args.sequence_index]}
    graph, source = restore_parent(config, Path(oracle["parent_checkpoint"]))
    source_hash = state_sha256(source)
    original, frozen = split_generator_parameters(source)
    with ocp.StandardCheckpointer() as checkpointer:
        fitted = checkpointer.restore(checkpoint / "best", target={"generator_parameters": [original]})["generator_parameters"][0]
    fitted_params = nnx.merge_state(frozen, fitted)
    fitted_hash = state_sha256(fitted_params)
    capture, compare = make_capture(graph, config.model)
    fp32_graph, fp32_config = precision_control_graph(config, source)
    fp32_capture, fp32_compare = make_capture(fp32_graph, fp32_config, backend="blocked")
    probes = tuple(jnp.zeros((1, tokens.shape[1] - 1, config.model.num_heads, config.model.value_head_dim),
                             jnp.bfloat16) for _ in range(config.model.num_layers))
    fp32_probes = tuple(probe.astype(jnp.float32) for probe in probes)
    acceptance = json.loads((run / "oracle_acceptance.json").read_text())
    nll_tolerance = acceptance["cross_compilation_nll_atol"]
    snapshots = {}
    failure = None
    for label, params in (("before", source), ("after", fitted_params)):
        (loss, (captured, gains, logits)), cotangents = capture(probes, params, tokens)
        raw_parity = {key: float(value) for key, value in jax.device_get(compare(params, tokens, logits)).items()}
        parity = assess_parity(raw_parity, float(loss), expected[label], nll_tolerance)
        print(json.dumps({"snapshot": label, "bf16_capture_parity": parity}), flush=True)
        snapshot = {"native_wrapper_parity": parity, "instrumented_nll": float(loss),
                    "recorded_sequence_nll": expected[label]}
        snapshots[label] = snapshot
        del logits
        if not parity["checks"]["native_checkpoint_nll"]:
            failure = f"Hard native checkpoint replay check failed for {label}: {parity['failed_checks']}"
            break
        snapshot["fp32_capture_control"] = precision_capture_control(
            fp32_capture, fp32_compare, fp32_probes, params, tokens)
        print(json.dumps({"snapshot": label, "fp32_capture_control": snapshot["fp32_capture_control"]}), flush=True)
        passed = parity["checks"]["instrumented_native_nll"]
        snapshot["cotangent_execution"] = (
            "Instrumented compiled execution; native/capture NLL agrees within the established bound, but bitwise identity is not asserted."
            if passed else
            "Instrumented compiled execution with capture/native NLL discrepancy beyond the established bound; these are not claimed to be the native checkpoint's exact cotangents.")
        snapshot["capture_parity_discrepancy_recorded"] = not passed
        if not passed and not args.record_parity_discrepancies:
            failure = f"Instrumented/native NLL parity failed for {label}; no derivative measurements retained for this snapshot without --record-parity-discrepancies"
            break
        rows, replay_errors = sampled_measurements(captured, gains, cotangents,
            eps=config.model.rms_norm_eps, uniform_count=args.uniform_queries)
        snapshot.update({"kernel_forward_replay_errors": replay_errors, "summary": summarize(rows), "samples": rows})
        del captured, gains, cotangents
    if state_sha256(source) != source_hash or state_sha256(fitted_params) != fitted_hash:
        raise RuntimeError("Replay changed checkpoint parameter arrays")
    result = {
        "schema_version": 2, "run_id": run.name, "selected_checkpoint_step": identity["step"],
        "status": "failed_parity" if failure else "complete", "failure": failure,
        "record_parity_discrepancies_requested": args.record_parity_discrepancies,
        "checkpoint_sha256": record["selected_checkpoint"]["sha256"], "parent_run_id": oracle["parent_run_id"],
        "parent_checkpoint_sha256": oracle["parent_checkpoint_sha256"], "config_sha256": sha256(run / "config.yaml"),
        "completion_sha256": sha256(run / "completion.json"), "metrics_sha256": hashlib.sha256(payload).hexdigest(),
        "batch_sha256": oracle["batch_sha256"], "sequence_index": args.sequence_index,
        "sequence_sha256": hashlib.sha256(batch[args.sequence_index].astype(np.int32).tobytes()).hexdigest(),
        "prediction_positions": int(tokens.shape[1] - 1), "uniform_query_count": args.uniform_queries,
        "sampling": "Endpoint-inclusive uniform queries plus each layer/head's minimum raw-output-RMS query; full original causal keys in every reference.",
        "quantile_probabilities": PROBABILITIES, "output_rms_epsilon": config.model.rms_norm_eps,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "analysis_script_sha256": sha256(__file__), "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "imported_analysis_sha256": {name: sha256(HERE / name) for name in ("inspect_batch_fit.py", "analyze_batch_coefficients.py")},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu": jax.devices()[0].device_kind,
        "cuda_environment": os.environ.get("TRANSFORMER_CUDA_ENVIRONMENT"),
        "parameter_arrays_unchanged": True, "nll_replay_atol_from_fitting_acceptance": nll_tolerance,
        "snapshots": snapshots, "elapsed_seconds": time.monotonic() - started,
        "interpretation": {
            "kernel_vjp": "Compare Mosaic dL/du with T^T times the recorded BF16 raw-output cotangent from the instrumented graph; isolates polynomial-kernel derivative arithmetic from RMS amplification. FP64 uses the exact captured BF16 operands but higher-precision score accumulation.",
            "rms_backward": "Compare actual BF16 raw-output cotangent with the independent FP64 RMS Jacobian at the same native raw output.",
            "forward_rounding": "Hold the upstream task cotangent fixed and compare FP64 RMS-composed coefficient gradients at native rounded y versus smooth y=T u.",
            "conditioning": "T_r=sum_causal s^r V/r!, u=r!c. Native-output J uses the RMS Jacobian at the recorded rounded y; smooth J uses y=T u. Tangents exclude coefficient radius only. Direction singular values multiply coefficient-tangent singular values by ||u||. Early causal prefixes may be structurally rank deficient.",
            "limits": "One fitting sequence, not B256 gradient or clipping statistics. The before/after causal activations differ. Cotangents come from each instrumented compiled model's own sequence NLL; failed capture/native parity is explicitly labeled and never excused by widening its bound. Native checkpoint identity/NLL replay and unchanged parameter arrays remain hard requirements. FP64 smooth derivatives differ from mixed-precision surrogate derivatives; numerical discrepancies are reported without an invented derivative pass threshold. No trained parameter, optimizer state, or checkpoint is changed.",
        },
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    update_record(run, {"analyses": {"coefficient_vjp_probe": {
        "path": str(output.relative_to(ROOT)), "sha256": sha256(output), "size_bytes": output.stat().st_size,
        "source_commit": result["source_commit"], "analysis_script_sha256": result["analysis_script_sha256"],
        "selected_checkpoint_sha256": result["checkpoint_sha256"]}}})
    print(json.dumps({"run_id": run.name, "output": str(output), "summaries": {
        label: snapshot.get("summary") for label, snapshot in snapshots.items()}}), flush=True)
    if failure:
        raise ValueError(f"{failure}; full parity diagnostics saved to {output}")


if __name__ == "__main__":
    main()
