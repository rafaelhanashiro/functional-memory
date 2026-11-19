"""Summarize explicitly pinned query-fit numerical and generator diagnostics.

This regeneration reads compact committed measurements only. It never loads a
checkpoint, dataset, GPU runtime or optimizer. Pooled relative errors use the
norm of the concatenated error/reference vectors, not an average of row ratios.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
ROLE = "query_generator_full_batch_m4_head_scale"
PROBABILITIES = [0., .1, .5, .9, .99, 1.]
ERRORS = (
    "kernel_vjp_error_given_recorded_raw_cotangent",
    "rms_backward_rounding_error",
    "forward_rounding_gradient_sensitivity",
    "composed_vjp_error_against_smooth_fp64",
    "forward_error_against_smooth_fp64",
)


def quantiles(values):
    array = np.asarray(values, np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite summary input")
    return np.quantile(array, PROBABILITIES, axis=0).tolist() if len(array) else []


def load_inputs():
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    entries = [entry for entry in selection["runs"] if entry.get("role") == ROLE]
    if len(entries) != 1:
        raise ValueError(f"Select exactly one {ROLE} run")
    entry = entries[0]
    verify_pinned_runs(ROOT, [entry])
    relative = Path("experiments") / entry["study"] / "runs" / entry["run_id"]
    run = ROOT / relative
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    completion = json.loads((run / "completion.json").read_text())
    record = json.loads((run / "record.json").read_text())
    oracle = recipe["oracle"]
    if (record["status"] != "complete" or completion["status"] != "complete"
            or oracle["parameterization"] != "query_coefficient_generator"
            or recipe["model"]["attention_degree"] != 4 or not recipe["model"]["qk_norm_head_scale"]
            or oracle["parent_run_id"] != entry["parent_run_id"]):
        raise ValueError("Expected the completed degree-four canonical-QK query fit")
    probes, pins = {}, {}
    for name in ("batch_functional_probe", "coefficient_vjp_probe"):
        expected_path = (relative / f"{name}.json").as_posix()
        pin = entry.get("diagnostics", {}).get(name)
        if not pin or pin["path"] != expected_path:
            raise ValueError(f"Explicitly pin diagnostics.{name} before summarizing")
        payload = (ROOT / expected_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != pin["sha256"]:
            raise ValueError(f"Diagnostic checksum changed: {expected_path}")
        data = json.loads(payload)
        expected = {
            "run_id": entry["run_id"], "parent_run_id": oracle["parent_run_id"],
            "selected_checkpoint_step": completion["best_step"],
            "checkpoint_sha256": record["selected_checkpoint"]["sha256"],
            "parent_checkpoint_sha256": oracle["parent_checkpoint_sha256"],
            "config_sha256": entry["sources"]["config.yaml"]["sha256"],
            "completion_sha256": entry["sources"]["completion.json"]["sha256"],
            "batch_sha256": oracle["batch_sha256"],
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"{name} identity mismatch: {key}")
        probes[name], pins[name] = data, pin
    functional, vjp = probes["batch_functional_probe"], probes["coefficient_vjp_probe"]
    if (not functional["frozen_parent_parameters_unchanged"] or not vjp["parameter_arrays_unchanged"]
            or vjp["status"] != "complete" or vjp["failure"] is not None
            or vjp["metrics_sha256"] != entry["sources"]["metrics.jsonl"]["sha256"]
            or vjp["sequence_index"] != 0 or vjp["uniform_query_count"] != 16
            or set(vjp["snapshots"]) != {"before", "after"}):
        raise ValueError("Unexpected, incomplete or mutable numerical-probe inputs")
    for snapshot in vjp["snapshots"].values():
        checks = snapshot["native_wrapper_parity"]["checks"]
        if not checks["native_checkpoint_nll"] or not checks["finite_measurements"]:
            raise ValueError("Native VJP checkpoint replay did not pass")
        if not checks["instrumented_native_nll"] and not vjp["record_parity_discrepancies_requested"]:
            raise ValueError("Unacknowledged instrumented-forward discrepancy")
    return entry, probes, pins


def pooled_error(rows, name):
    values = [row[name] for row in rows]
    error_squared = math.fsum(value["absolute_l2_error"] ** 2 for value in values)
    reference_squared = math.fsum(value["reference_l2_norm"] ** 2 for value in values)
    actual_squared = math.fsum(value["actual_l2_norm"] ** 2 for value in values)
    reference_norm, actual_norm = math.sqrt(reference_squared), math.sqrt(actual_squared)
    ratios = [value["relative_l2_error"] for value in values if value["relative_l2_error"] is not None]
    dot = math.fsum(value["cosine"] * value["reference_l2_norm"] * value["actual_l2_norm"]
                    for value in values if value["cosine"] is not None)
    return {
        "sample_count": len(values), "zero_reference_count": sum(value["reference_l2_norm"] == 0 for value in values),
        "zero_actual_count": sum(value["actual_l2_norm"] == 0 for value in values),
        "pooled_absolute_l2_error": math.sqrt(error_squared),
        "pooled_reference_l2_norm": reference_norm, "pooled_actual_l2_norm": actual_norm,
        "pooled_relative_l2_error": math.sqrt(error_squared) / reference_norm if reference_norm else None,
        "pooled_cosine": float(np.clip(dot / (reference_norm * actual_norm), -1, 1)) if reference_norm and actual_norm else None,
        "row_relative_error_quantiles": quantiles(ratios),
    }


def geometry_summary(rows, kind):
    values = [row[kind] for row in rows]
    condition = [value["tangent_condition_number"] for value in values if value["tangent_condition_number"] is not None]
    rank_counts = Counter(value["tangent_numerical_rank"] for value in values)
    return {
        "sample_count": len(values), "numerical_rank_counts": {str(rank): rank_counts[rank] for rank in sorted(rank_counts)},
        "undefined_condition_count": len(values) - len(condition), "finite_condition_quantiles": quantiles(condition),
        "zero_spectrum_count": sum(max(value["coefficient_tangent_singular_values"]) == 0 for value in values),
        "coefficient_radius_quantiles": quantiles([value["coefficient_radius"] for value in values]),
        "coefficient_tangent_singular_value_quantiles": quantiles([value["coefficient_tangent_singular_values"] for value in values]),
        "direction_tangent_singular_value_quantiles": quantiles([value["direction_tangent_singular_values"] for value in values]),
        "log_radius_response_norm_quantiles": quantiles([value["log_radius_response_norm"] for value in values]),
        "log_radius_relative_response_quantiles": quantiles([value["log_radius_to_largest_direction_response"] for value in values
                                                             if value["log_radius_to_largest_direction_response"] is not None]),
    }


def vjp_groups(snapshot):
    rows = snapshot["samples"]
    groups = {
        "uniform_all": [row for row in rows if row["uniform_sample"]],
        "uniform_excluding_query0": [row for row in rows if row["uniform_sample"] and row["query_position"] > 0],
        "uniform_query0_only": [row for row in rows if row["uniform_sample"] and row["query_position"] == 0],
        "minimum_rms_targeted": [row for row in rows if row["minimum_rms_sample"]],
    }
    result = {}
    for name, selected in groups.items():
        result[name] = {
            "sample_count": len(selected), "query0_count": sum(row["query_position"] == 0 for row in selected),
            "errors": {metric: pooled_error(selected, metric) for metric in ERRORS},
            "raw_output_rms_quantiles": quantiles([row["raw_output_rms"] for row in selected]),
            "epsilon_fraction_quantiles": quantiles([row["epsilon_fraction_of_rms_denominator"] for row in selected]),
            "geometry": {kind: geometry_summary(selected, kind) for kind in ("native_output_geometry", "smooth_fp64_geometry")},
        }
    return result


def functional_snapshot(snapshot):
    statistics = {}
    for name, value in snapshot["statistics"].items():
        array = np.asarray(value, np.float64)
        if array.shape != (15, 14, 5) or not np.isfinite(array).all():
            raise ValueError(f"Unexpected layer/head/order statistic: {name}")
        statistics[name] = {"global_mean": float(array.mean()), "per_order_mean": array.mean(axis=(0, 1)).tolist()}
    parity = snapshot["external_native_nll_comparison"]
    deltas = np.asarray(parity["per_sequence_nll_deltas"])
    if len(deltas) != 256:
        raise ValueError("Expected all256 fitting sequences in diagnostic parity")
    telemetry = snapshot["pre_output_rms_telemetry"]
    get = lambda suffix: [value for key, value in telemetry.items() if key.endswith(suffix)]
    normalization = {
        "minimum_raw_output_head_rms": min(get("attention_output_head_rms_min")),
        "maximum_inverse_rms_denominator": max(get("head_rms_norm_gain_max")),
        "mean_head_mean_square_below_epsilon_fraction": float(np.mean(get("head_mean_square_lt_1eps_fraction_mean"))),
        "mean_head_mean_square_below_100epsilon_fraction": float(np.mean(get("head_mean_square_lt_100eps_fraction_mean"))),
    }
    return {
        "statistics_forward": snapshot["statistics_forward"], "statistics_axes": snapshot["statistics_axes"],
        "statistics": statistics,
        "preactivation_rms": math.sqrt(statistics["preactivation_mean_square"]["global_mean"]),
        "factorial_coefficient_rms": math.sqrt(statistics["coefficient_mean_square_factorial_coordinates"]["global_mean"]),
        "stored_parameter_rms": snapshot["stored_parameter_rms"],
        "sampled_coefficient_radius_quantiles": snapshot["factorial_coordinate_norm_quantiles"],
        "sampled_quantile_probabilities": snapshot["quantile_probabilities"],
        "output_normalization": normalization,
        "within_compiled_wrapper_logit_comparison": snapshot["native_wrapper_parity"],
        "independent_compiled_nll_comparison": {
            "sequence_count": len(deltas), "established_nll_atol": parity["nll_replay_atol"],
            "mean_nll_delta": float(deltas.mean()), "max_absolute_nll_delta": float(np.max(np.abs(deltas))),
            "sequences_exceeding_bound": int(np.count_nonzero(np.abs(deltas) > parity["nll_replay_atol"])),
            "original_per_sequence_bound_passed": parity["original_per_sequence_bound_passed"],
            "aggregate_nll_passed": parity["aggregate_nll_passed"],
            "signed_delta_quantiles": quantiles(deltas), "absolute_delta_quantiles": quantiles(np.abs(deltas)),
        },
    }


def repeatability_summary(control):
    anchors = {}
    for name, value in control["anchor_forwards"].items():
        deltas = np.asarray(value["nll_delta_from_ordinary_native"])
        anchors[name] = {"arrays_repeat_exactly": value["all_returned_arrays_repeat_exactly"],
                         "mean_nll": value["mean_nll"], "mean_delta_from_native": float(deltas.mean()),
                         "max_absolute_delta_from_native": float(np.max(np.abs(deltas)))}
    materialized = {
        name: {"arrays_repeat_exactly": value["all_returned_arrays_repeat_exactly"],
               "max_absolute_nll_delta_from_anchor": float(np.max(np.abs(value["delta_from_corresponding_anchor_nll"])))}
        for name, value in control["materialized_logit_forwards"].items()
    }
    return {"snapshot": control["snapshot"], "worst_sequence_index": control["worst_sequence_index"],
            "microbatch_start_index": control["microbatch_start_index"], "microbatch_sequence_count": control["microbatch_sequence_count"],
            "anchor_forwards": anchors, "materialized_logit_forwards": materialized,
            "independent_logit_comparisons": control["independent_logit_comparisons"],
            "within_compiled_native_hook_logit_parity": control["within_compiled_native_hook_logit_parity"],
            "limits": control["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "precision_summary.json")
    parser.add_argument("--csv", type=Path, default=HERE / "precision_vjp_summary.csv")
    args = parser.parse_args()
    entry, probes, pins = load_inputs()
    functional, vjp = probes["batch_functional_probe"], probes["coefficient_vjp_probe"]
    diagnostics = functional["diagnostics"]
    changes = diagnostics["native_generator_change"]
    snapshots = {}
    for label in ("before", "after"):
        numerical = vjp["snapshots"][label]
        snapshots[label] = {
            "generator": functional_snapshot(diagnostics[label]),
            "bf16_vjp_capture_parity": numerical["native_wrapper_parity"],
            "fp32_blocked_capture_parity": numerical["fp32_capture_control"],
            "cotangent_execution": numerical["cotangent_execution"],
            "vjp_sample_groups": vjp_groups(numerical),
        }
    result = {
        "schema_version": 1, "run_id": entry["run_id"], "role": ROLE,
        "selected_checkpoint_step": vjp["selected_checkpoint_step"], "sources": pins,
        "source_gpu_jobs": {name: value["slurm_job_id"] for name, value in probes.items()},
        "quantile_probabilities": PROBABILITIES, "vjp_sequence_index": vjp["sequence_index"],
        "gradient_coordinate": "u_r=r!c_r; error-vector norms are pooled across selected layer/head/query rows",
        "pooled_relative_error_formula": "sqrt(sum(row.absolute_l2_error**2))/sqrt(sum(row.reference_l2_norm**2))",
        "snapshots": snapshots,
        "parameter_changes": {name: {key: value[key] for key in ("delta_rms", "delta_relative_l2")}
                              for name, value in changes["parameter_changes"].items()},
        "generator_change": {key: changes[key] for key in ("sampled_direction_angle_degrees_quantiles",
            "sampled_radius_ratio_quantiles", "sampled_factorial_coordinate_delta_rms", "limits")},
        "generator_change_quantile_probabilities": diagnostics["before"]["quantile_probabilities"],
        "bounded_repeatability_control": repeatability_summary(diagnostics["bounded_repeatability_control"]),
        "interpretation_limits": [
            "Generator statistics average the instrumented B256 forward; its failed per-sequence parity bound remains visible. The VJP/conditioning probe covers only sequence0 and uses instrumented-execution cotangents.",
            "Uniform samples and deliberately minimum-RMS samples are separate. Query0 has one causal key and theoretical basis-output rank at most1; its nearly zero directional derivative makes relative ratios and floating-point rank estimates unreliable. Excluding it is explicitly reported, not silently selected.",
            "Native-output tangent geometry evaluates the smooth RMS Jacobian at the captured rounded output. Smooth-FP64 geometry evaluates it at T*u. Neither includes the coefficient generator Jacobian or represents a whole-model Hessian.",
            "Coefficient tangent singular values measure changes in u; angular-direction singular values multiply them by ||u||. Fixed-ray radius is excluded from tangent conditioning but its response is reported separately.",
            "FP32 controls change the attention backend to blocked as well as arithmetic. Improved parity supports precision/execution sensitivity, but does not isolate a compiler operation or prove the cause of the optimization gap.",
            "Small pooled kernel-only coefficient-VJP discrepancies are evidence against a concrete backward-kernel error on sampled tensors; they do not prove correctness at every checkpoint or determine B256 clipping causes.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    fields = ["snapshot", "sample_group", "metric", "sample_count", "zero_reference_count", "zero_actual_count",
              "pooled_relative_l2_error", "pooled_absolute_l2_error", "pooled_reference_l2_norm", "pooled_actual_l2_norm", "pooled_cosine"]
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label in ("before", "after"):
            for group, value in snapshots[label]["vjp_sample_groups"].items():
                for metric, error in value["errors"].items():
                    row = {"snapshot": label, "sample_group": group, "metric": metric,
                           **{key: error[key] for key in fields[3:]}}
                    writer.writerow({key: format(value, ".12g") if isinstance(value, float) else value for key, value in row.items()})
    print(json.dumps({label: {metric: snapshots[label]["vjp_sample_groups"]["uniform_excluding_query0"]["errors"][metric]["pooled_relative_l2_error"]
                             for metric in ERRORS} for label in ("before", "after")}, sort_keys=True))


if __name__ == "__main__":
    main()
