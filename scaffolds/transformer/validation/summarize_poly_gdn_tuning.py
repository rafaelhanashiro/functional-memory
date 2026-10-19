"""Regenerate polynomial-GDN tuning summaries from explicit, optionally pinned reports.

The default selection is runs/poly-gdn-tuning/record.json, whose reports list
contains {kind, job_id, path, sha256, role}; paths are relative to that record.
Supported kinds are polynomial, native_fla, training, native_model, and batch_split_diagnostic. A complete selection
requires every SHA256 pin. --profile-job JOB reads only that explicitly named
job's three known profile reports while they are being written. No newest-run
selection, source-report mutation, or cross-GPU pooling is performed.
Pinned diagnostic reports may declare diagnostic=true, expected_status, and
expected_failed_shapes; those exact failures stay visible without counting as
new audit errors. Changed hashes, outcomes, or failed shapes remain errors.
Optional training_comparisons explicitly name one job, report_roles, and clocks
for repeated trials. Only complete compatible named reports are pooled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent / "runs/poly-gdn-tuning"
REPOSITORY = Path(__file__).resolve().parents[3]
CHANNELS = {"output", "q", "k", "v", "g", "beta", "erase"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pool(samples, tokens=None):
    if not samples:
        return {"samples": 0}
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in samples):
        raise ValueError("Timing samples must be finite and positive")
    result = {"samples": len(samples), "mean_seconds": statistics.mean(samples),
              "median_seconds": statistics.median(samples), "min_seconds": min(samples), "max_seconds": max(samples)}
    if tokens is not None:
        result["tokens_per_second"] = tokens * len(samples) / sum(samples)
    return result


def identity(report):
    source = report.get("provenance", {})
    device = source.get("device", {})
    gpu = device.get("uuid")
    gpu = gpu.lower().removeprefix("gpu-") if isinstance(gpu, str) else None
    environments = [source.get(key, {}) for key in ("environment", "runtime", "runtime_environment")]
    job = next((values.get("SLURM_JOB_ID") for values in environments if values.get("SLURM_JOB_ID")), None)
    compiler_flags = next((values["JAX_TRITON_PTXAS_EXTRA_FLAGS"] for values in environments
                           if "JAX_TRITON_PTXAS_EXTRA_FLAGS" in values), None)
    return {"gpu_uuid": gpu, "source_commit": source.get("source_commit"), "reported_job_id": job,
            "compiler_ptxas_extra_flags": compiler_flags,
            "compiler_ptxas_device_debug": next((values["JAX_TRITON_PTXAS_DEVICE_DEBUG"] for values in environments
                                                  if "JAX_TRITON_PTXAS_DEVICE_DEBUG" in values), None),
            "xla_flags": next((values["XLA_FLAGS"] for values in environments if "XLA_FLAGS" in values), None),
            "dependency_lock_sha256": source.get("dependency_lock_sha256"),
            "frozen_execution_sha256": source.get("frozen_execution_sha256")}


def shape_fields(shape):
    return {key: shape.get(key) for key in ("batch", "length", "heads", "key_dim", "value_dim", "degree")}


def shape_key(shape):
    return tuple(shape.get(key) for key in ("batch", "length", "heads", "key_dim", "value_dim"))


def accuracy_fields(accuracy):
    channels = accuracy.get("channels", {})
    result = {"accuracy_passed": accuracy.get("passed"), "all_finite": accuracy.get("all_finite")}
    for metric in ("relative_rms", "relative_max", "max_abs"):
        candidates = [(value.get(metric), name) for name, value in channels.items()
                      if isinstance(value.get(metric), (int, float)) and math.isfinite(value[metric])]
        if candidates:
            value, name = max(candidates)
            result[f"worst_{metric}"] = value
            result[f"worst_{metric}_channel"] = name
    return result


class Summary:
    def __init__(self, selection):
        self.result = {"format": "polynomial-gdn-tuning-summary", "status": "complete",
            "selection": selection, "sources": [], "kernels": [], "stages": [], "resources": [], "training": [],
            "pending": [], "issues": [], "diagnostics": [], "fixture_pairs": [], "training_comparisons": [],
            "limitations": ["Native FLA is linear tied-gate GDN, not the cubic operator.",
                "Stage calls use resident intermediates; their sum excludes fusion, scheduling, and final gradient accumulation.",
                "Resource figures describe assembled kernels, not measured runtime occupancy.",
                "Kernel accuracy is same-input parity to tf32x3; independent oracle and stress qualification is separate.",
                "Native FLA full-model timings reuse one GPU-resident batch and exclude loading/transfers; JAX training intervals include its input pipeline."]}
        self.identities = defaultdict(list)
        self.fixtures = defaultdict(dict)
        self.expected_failed_shapes = {}
        self.training_samples = {}

    def issue(self, source, message):
        self.result["issues"].append({"source": source, "message": message})

    def require(self, condition, source, message):
        if not condition:
            self.issue(source, message)

    def fixture(self, base, shape, kind, hashes, complete):
        # Initial progress reports precede fixture creation. Do not compare an
        # absent fixture with a completed peer, but require it once complete.
        if not hashes:
            self.require(not complete, base["path"], "Completed report is missing shared input hashes")
            return
        self.require(set(hashes) == {"q", "k", "v", "do", "g", "beta", "erase"}, base["path"],
                     "Shared fixture hash set is incomplete")
        self.fixtures[base["job_id"], shape_key(shape)][kind] = hashes

    def checked_accuracy(self, accuracy, source, label, rms_limit, max_limit, *, expected_channels=CHANNELS):
        channels = accuracy.get("channels", {})
        limits = accuracy.get("limits", {})
        valid = (accuracy.get("passed") is True and accuracy.get("all_finite") is True
                 and set(channels) == expected_channels and limits.get("relative_rms") == rms_limit
                 and limits.get("relative_max") == max_limit)
        for channel in channels.values():
            valid = valid and channel.get("finite") is True and channel.get("passed") is True
            for metric, limit in (("relative_rms", rms_limit), ("relative_max", max_limit)):
                value = channel.get(metric)
                valid = valid and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= limit
        self.require(valid, source, f"{label} lacks passing finite metrics at its declared limits")

    def operation(self, operation, config, source, *, tokens):
        rounds = operation.get("rounds", [])
        samples, means = [], []
        for trial in rounds:
            values = trial.get("samples_seconds", [])
            self.require(len(values) == config["iterations"], source, "Incorrect samples per completed round")
            self.require(len(trial.get("warmup_seconds", [])) == config["warmup"], source, "Incorrect warmup count")
            self.require(trial.get("all_finite") is True, source, "Nonfinite measured outputs/gradients")
            measured = pool(values)
            samples.extend(values)
            if values:
                means.append(measured["mean_seconds"])
        complete = operation.get("status") == "complete"
        self.require(not complete or [item.get("round") for item in rounds] == list(range(1, config["rounds"] + 1)),
                     source, "Incorrect completed round IDs/count")
        result = pool(samples, tokens)
        if len(means) > 1:
            result["repeat_mean_spread_percent"] = 100 * (max(means) - min(means)) / statistics.mean(means)
        result.update(completed_rounds=len(rounds), expected_rounds=config["rounds"],
                      compilation_seconds=operation.get("compilation_seconds"))
        return result

    def polynomial(self, report, base):
        config = report["configuration"]
        if report.get("status") in ("complete", "completed_with_failures"):
            expected = {(batch, config["length"], config["heads"], width, config["value_dim"])
                        for batch in config["batches"] for width in config["key_dims"]}
            actual = {shape_key(entry["shape"]) for entry in report.get("shapes", {}).values()}
            self.require(actual == expected, base["path"], "Completed matrix has missing/unrequested shapes")
        for shape_name, entry in report.get("shapes", {}).items():
            shape = entry["shape"]
            fields = shape_fields(shape)
            split = entry.get("batch_split_reference", {})
            if base.get("required_batch_split_policy") and entry.get("status") == "complete":
                self.require(split.get("policy") == base["required_batch_split_policy"], base["path"],
                             "Completed shape is missing the required batch-split policy")
            if config.get("reference_batch_size") and entry.get("status") == "complete":
                self.require(split.get("status") == "complete" and split.get("accuracy", {}).get("passed") is True,
                             base["path"], "Completed shape lacks a passing batch-split reference check")
                self.require(set(split.get("accuracy", {}).get("channels", {})) == CHANNELS,
                             base["path"], "Batch-split reference check is missing gradient channels")
            promoted = split.get("promoted_fp32", {})
            if split.get("policy") == "production_dtype_and_promoted_fp32" and (
                    entry.get("status") == "complete" or split.get("status") == "complete"):
                self.require(split.get("status") == "complete" and split.get("passed") is True,
                             base["path"], "Dual batch-split reference did not pass both checks")
                self.checked_accuracy(split.get("accuracy", {}), base["path"], "Production batch-split reference",
                                      config["exact_relative_rms"], config["exact_relative_max"])
                self.require(promoted.get("status") == "complete" and promoted.get("promotion_preserves_values_exactly") is True,
                             base["path"], "Promoted FP32 reference did not complete with exact input preservation")
                self.checked_accuracy(promoted.get("accuracy", {}), base["path"], "Promoted FP32 batch-split reference", 1e-5, 1e-5)
                promotion = promoted.get("promotion_channels", {})
                names = {"q", "k", "v", "g", "beta", "erase", "cotangent"}
                self.require(set(promotion) == names and all(
                    value.get("target_dtype") == "float32" and value.get("values_preserved_exactly") is True
                    and value.get("source_dtype") == (entry.get("cotangent_dtype") if name == "cotangent"
                                                     else entry.get("input_dtypes", {}).get(name))
                    for name, value in promotion.items()), base["path"], "FP32 promotion identity/channel metadata differs")
            self.fixture(base, shape, "polynomial", entry.get("input_sha256"),
                         entry.get("status") in ("complete", "completed_with_failures"))
            if entry.get("status") in ("complete", "completed_with_failures"):
                if "min_ctas" in config:
                    expected = {f"{precision}_w{warps}_cta{ctas}" for precision in config["precisions"]
                                for warps in config["warps"] for ctas in config["min_ctas"]}
                else:
                    expected = {f"{precision}_w{warps}" for precision in config["precisions"] for warps in config["warps"]}
                self.require(set(entry.get("variants", {})) == expected, base["path"], "Completed shape has missing variants")
            for variant, value in entry.get("variants", {}).items():
                prefix = {**base, **fields, "implementation": "polynomial_gdn", "variant": variant,
                          "precision": value["precision"], "warps": value["num_warps"],
                          "min_ctas_per_sm": value.get("min_ctas_per_sm"), "status": value["status"],
                          "batch_split_reference_status": split.get("status"),
                          "batch_split_reference_passed": split.get("accuracy", {}).get("passed"),
                          "batch_split_reference_policy": split.get("policy", "strict_production_dtype_only" if split else None),
                          "promoted_fp32_reference_passed": promoted.get("accuracy", {}).get("passed"),
                          "promotion_preserves_values_exactly": promoted.get("promotion_preserves_values_exactly"),
                          "reference_batch_size": config.get("reference_batch_size", 0)}
                if value.get("error"):
                    prefix["error"] = value["error"]
                for direction in ("forward", "forward_backward"):
                    operation = value.get("operations", {}).get(direction)
                    row = {**prefix, "direction": direction, "timing": "synchronized_wall"}
                    if operation is not None:
                        row.update(self.operation(operation, config, base["path"], tokens=shape["batch"] * shape["length"]))
                        accuracy = operation.get("accuracy", {})
                        row.update(accuracy_fields(accuracy))
                        expected = CHANNELS if direction == "forward_backward" else {"output"}
                        if accuracy:
                            self.require(set(accuracy.get("channels", {})) == expected, base["path"],
                                         f"Incomplete accuracy channels for {variant}/{direction}")
                        if operation.get("status") == "complete":
                            limit_prefix = "exact" if value["precision"] == "tf32x3" else "max"
                            self.checked_accuracy(accuracy, base["path"], f"Completed {variant}/{direction}",
                                                  config[f"{limit_prefix}_relative_rms"],
                                                  config[f"{limit_prefix}_relative_max"], expected_channels=expected)
                    else:
                        row["samples"] = 0
                    self.result["kernels"].append(row)
                for stage, operation in value.get("stages", {}).items():
                    row = {**prefix, "stage": stage, "status": value.get("stage_status", operation.get("status")),
                           "timing": "isolated_stage_wall"}
                    row.update(self.operation(operation, config, base["path"], tokens=shape["batch"] * shape["length"]))
                    if value.get("stage_error"):
                        row["error"] = value["stage_error"]
                    self.result["stages"].append(row)
            if entry.get("status") == "failed":
                expected_failure = shape_name in self.expected_failed_shapes.get(base["path"], set())
                if expected_failure:
                    self.result["diagnostics"].append({**base, "shape": shape, "status": "failed",
                                                       "expected_failure": True, "error": entry.get("error")})
                else:
                    self.issue(base["path"], f"Failed shape {shape_name}: {entry.get('error')}")
                if not entry.get("variants"):
                    reference = report.get("accuracy_reference", {})
                    self.result["kernels"].append({**base, **fields, "implementation": "polynomial_gdn",
                        "variant": "reference", "precision": reference.get("precision"),
                        "warps": reference.get("num_warps"), "min_ctas_per_sm": reference.get("min_ctas_per_sm"),
                        "direction": "forward_backward", "timing": "synchronized_wall", "status": "failed",
                        "samples": 0, "expected_failure": expected_failure, "error": entry.get("error")})
        for resource in report.get("compiled_resources", []):
            shape = report.get("shapes", {}).get(resource.get("shape"), {}).get("shape", {})
            row = {**base, **shape_fields(shape), "variant": resource.get("variant"),
                   "stage": resource.get("operation"), "kernel": resource.get("kernel")}
            row.update({key: resource.get(key) for key in ("num_warps", "shared_bytes", "registers", "spill_store_bytes",
                "spill_load_bytes", "stack_frame_bytes", "ptx_sha256", "ptxas_returncode", "resource_inspection_error",
                "min_ctas_directive", "runtime_ptxas_extra_flags", "runtime_ptxas_device_debug",
                "inspection_ptxas_command", "effective_batch")})
            variant = report.get("shapes", {}).get(resource.get("shape"), {}).get("variants", {}).get(resource.get("variant"), {})
            row["min_ctas_per_sm"] = (report.get("accuracy_reference", {}).get("min_ctas_per_sm")
                                      if resource.get("variant") == "reference" else variant.get("min_ctas_per_sm"))
            if row["min_ctas_per_sm"] is not None and resource.get("kernel") in ("forward", "backward_queries", "backward_keys"):
                expected = [str(row["min_ctas_per_sm"])] if row["min_ctas_per_sm"] else []
                self.require(resource.get("min_ctas_directive") == expected, base["path"],
                             f"PTX occupancy directive differs for {row['variant']}/{row['kernel']}")
            self.result["resources"].append(row)

    def native(self, report, base):
        config = report["arguments"]
        shape = {key: config[key] for key in ("batch", "length", "heads", "key_dim", "value_dim")}
        self.fixture(base, shape, "native_fla", report.get("input_sha256"), report.get("status") == "complete")
        grouped = defaultdict(list)
        selected = [trial for trial in report.get("trials", []) if trial["implementation"] == "upstream_fla"]
        if report.get("status") == "complete":
            expected = {(direction, index) for direction in ("forward", "forward_backward") for index in range(config["rounds"])}
            actual = {(trial["direction"], trial["round"]) for trial in selected}
            self.require(actual == expected and len(selected) == len(expected), base["path"], "Native trial selection differs")
        all_finite = True
        for trial in report.get("trials", []):
            if trial["implementation"] != "upstream_fla":
                continue
            self.require(len(trial.get("warmup_samples", [])) == config["warmup"], base["path"], "Native warmup count differs")
            checks = trial.get("finite_outputs_and_gradients", [])
            count = 1 if trial["direction"] == "forward" else 6
            finite = len(checks) == count and all(value.get("finite") is True for value in checks)
            all_finite = all_finite and finite
            self.require(finite,
                         base["path"], "Native output/gradient finite checks failed or incomplete")
            grouped[trial["direction"], "synchronized_wall"].append((trial["round"], trial.get("samples", []), "wall_seconds"))
            grouped[trial["direction"], "cuda_events"].append((trial["round"], trial.get("samples", []), "cuda_seconds"))
            graph = trial.get("cuda_graph")
            if graph and graph.get("status") == "complete":
                grouped[trial["direction"], "graph_wall"].append((trial["round"], graph["samples"], "wall_seconds"))
                grouped[trial["direction"], "graph_cuda_events"].append((trial["round"], graph["samples"], "cuda_seconds"))
            elif graph:
                self.result["kernels"].append({**base, **shape, "implementation": "native_fla", "direction": trial["direction"],
                    "timing": "graph_wall", "status": graph.get("status"), "error": graph.get("error"), "samples": 0})
        for (direction, timing), trials in sorted(grouped.items()):
            samples, means = [], []
            for _, values, key in trials:
                self.require(len(values) == config["iterations"], base["path"], "Native sample count differs")
                values = [value[key] for value in values]
                measured = pool(values)
                samples.extend(values)
                if values:
                    means.append(measured["mean_seconds"])
            complete = sorted(index for index, _, _ in trials) == list(range(config["rounds"]))
            row = {**base, **shape, "implementation": "native_fla", "direction": direction, "timing": timing,
                   "status": "complete" if complete else "incomplete", "all_finite": all_finite,
                   "completed_rounds": len(trials), "expected_rounds": config["rounds"],
                   **pool(samples, shape["batch"] * shape["length"])}
            if len(means) > 1:
                row["repeat_mean_spread_percent"] = 100 * (max(means) - min(means)) / statistics.mean(means)
            self.result["kernels"].append(row)
            self.require(complete or report["status"] != "complete", base["path"], "Completed native report lacks rounds")

    def training(self, report, base):
        model, data, train = (report["configuration"][key] for key in ("model", "data", "train"))
        runtime = report["provenance"].get("runtime_environment", {})
        samples = report.get("samples", [])
        warmup, iterations = report["warmup_updates"], report["timed_updates"]
        if report["status"] == "complete":
            self.require(len(samples) == warmup + iterations, base["path"], "Training update count differs")
            self.require(report.get("source_recipe_unchanged") is True, base["path"], "Training source recipe changed")
        self.require([sample["step"] for sample in samples] == list(range(1, len(samples) + 1)),
                     base["path"], "Training update sequence differs")
        finite = all(sample.get("finite_loss_and_update_numerics") is True and
                     all(isinstance(value, (int, float)) and math.isfinite(value)
                         for value in [sample.get("loss"), *sample.get("update_numerics", [])]) for sample in samples)
        self.require(finite, base["path"], "Training loss or update numerics are nonfinite")
        shape = {"batch": data["batch_size"], "microbatch": train.get("micro_batch_size"),
                 "length": data["sequence_length"] - 1, "heads": model["num_heads"],
                 "key_dim": model["qk_head_dim"], "value_dim": model["value_head_dim"],
                 "degree": model.get("gdn_polynomial_degree") if model.get("attention_mechanism") == "gated_deltanet"
                           else model.get("attention_degree")}
        self.require(report["tokens_per_update"] == shape["batch"] * shape["length"], base["path"],
                     "Training token count differs from the configured batch and prediction positions")
        for key in ("compute_seconds", "interval_seconds"):
            measured = [sample[key] for sample in samples[warmup:] if sample.get(key) is not None]
            count_valid = report["status"] != "complete" or len(measured) == iterations
            self.require(count_valid, base["path"], f"Completed training clock {key} has missing timed samples")
            row = {**base, **shape, "implementation": report.get("resolved_implementation"),
                   "precision": runtime.get("POLY_GDN_VALUE_PRECISION"), "warps": runtime.get("POLY_GDN_NUM_WARPS"),
                   "min_ctas_per_sm": runtime.get("POLY_GDN_MIN_CTAS_PER_SM"),
                   "direction": "training_update", "timing": key,
                   "status": report["status"] if count_valid else "invalid_sample_count", "all_finite": finite,
                   "parameter_count": report.get("parameter_count"), "accumulation_steps": report.get("accumulation_steps"),
                   "warmup_updates": warmup, "validated_updates": len(samples), "seed": train.get("seed"),
                   "input_recipe_sha256": report.get("input_recipe", {}).get("sha256"),
                   "data_identity_sha256": report.get("data_identity_sha256"),
                   **pool(measured, report["tokens_per_update"])}
            self.result["training"].append(row)
            self.training_samples[base["path"], key] = measured

    def native_model(self, report, base):
        """Native upstream model compute; never label this an input-pipeline interval."""
        if report.get("status") == "running" and not all(
                name in report for name in ("training_settings", "model_configuration", "parameter_count")):
            return  # The initial progress report precedes model construction.
        config, settings = report["arguments"], report["training_settings"]
        model = report["model_configuration"]
        self.require(config.get("mode") == "model", base["path"], "Selected native_model report is not a model run")
        self.require(model.get("head_dim") == config["key_dim"] and model.get("num_heads") == config["heads"]
                     and model.get("head_dim", 0) * model.get("expand_v", 0) == config["value_dim"],
                     base["path"], "Native model dimensions differ from requested K/V/heads")
        self.require(settings["global_batch"] == config["global_batch"]
                     and settings["micro_batch"] == config["batch"]
                     and settings["prediction_positions"] == config["length"]
                     and settings["accumulation_steps"] * settings["micro_batch"] == settings["global_batch"],
                     base["path"], "Native model batch, sequence, or accumulation settings differ")
        warmups, samples = report.get("warmup_samples", []), report.get("samples", [])
        all_samples = warmups + samples
        if report["status"] == "complete":
            self.require(len(warmups) == config["warmup"] and len(samples) == config["iterations"],
                         base["path"], "Native model warmup/measured counts differ")
            self.require(report.get("all_finite") is True, base["path"], "Native model complete finite check missing")
        self.require([sample["step"] for sample in all_samples] == list(range(1, len(all_samples) + 1)),
                     base["path"], "Native model update sequence differs")
        finite = all(sample.get("all_finite") is True and all(
            isinstance(sample.get(key), (int, float)) and math.isfinite(sample[key])
            for key in ("loss", "gradient_norm_pre_clip", "parameter_norm", "learning_rate")) for sample in all_samples)
        self.require(finite, base["path"], "Native model has nonfinite loss, gradient, parameter norm, or learning rate")
        self.require(isinstance(report.get("parameter_count"), int) and report["parameter_count"] > 0,
                     base["path"], "Native model parameter count missing/invalid")
        data = report.get("data_identity", {})
        shape = {"batch": settings["global_batch"], "microbatch": settings["micro_batch"],
                 "length": settings["prediction_positions"], "heads": config["heads"],
                 "key_dim": config["key_dim"], "value_dim": config["value_dim"], "degree": 1}
        for key, timing in (("wall_seconds", "resident_compute_seconds"), ("cuda_seconds", "cuda_events")):
            row = {**base, **shape, "implementation": "native_fla_model", "direction": "training_update",
                   "timing": timing, "scope": "compute_only_resident_repeated_batch",
                   "status": ("invalid_sample_count" if report["status"] == "complete"
                              and (len(warmups) != config["warmup"] or len(samples) != config["iterations"])
                              else report["status"]), "all_finite": finite, "validated_updates": len(all_samples),
                   "warmup_updates": len(warmups), "parameter_count": report.get("parameter_count"),
                   "accumulation_steps": settings["accumulation_steps"],
                   "parameter_dtype": settings.get("parameter_dtype"), "autocast_dtype": settings.get("autocast_dtype"),
                   "initialization": settings.get("initialization"), "seed": config.get("seed"),
                   "model_recipe_sha256": hashlib.sha256(json.dumps(
                       {"model": model, "training": settings}, sort_keys=True).encode()).hexdigest(),
                   "data_manifest_sha256": data.get("manifest_sha256"), "resident_tokens_sha256": data.get("tokens_sha256"),
                   **pool([sample[key] for sample in samples], shape["batch"] * shape["length"])}
            self.result["training"].append(row)
            self.training_samples[base["path"], timing] = [sample[key] for sample in samples]

    def batch_split_diagnostic(self, report, base):
        complete = report.get("status") == "complete"
        if complete:
            self.require(report.get("promotion_preserves_values_exactly") is True, base["path"],
                         "FP32 promotion did not preserve quantized fixture values")
            self.require(report.get("all_finite") is True and report.get("fp32_strict_passed") is True,
                         base["path"], "Batch-split FP32 diagnostic failed strict accuracy or finite checks")
            self.require(set(report.get("cases", {})) == {"production_bf16", "same_quantized_values_fp32"},
                         base["path"], "Batch-split diagnostic is missing a precision case")
        for name, case in report.get("cases", {}).items():
            accuracy = case.get("strict_accuracy", {})
            if case.get("status") == "complete":
                self.require(set(accuracy.get("channels", {})) == CHANNELS, base["path"],
                             "Batch-split diagnostic is missing gradient channels")
                self.require(accuracy.get("all_finite") is True, base["path"], "Nonfinite batch-split diagnostic")
            self.result["diagnostics"].append({**base, "case": name, "status": case.get("status"),
                "strict_accuracy": accuracy_fields(accuracy), "chunk_sizes": case.get("chunk_sizes"),
                "channels": {channel: {key: values.get(key) for key in (
                    "elements", "mismatch_count", "max_bf16_ulp", "one_ulp_count", "multiple_ulp_count",
                    "multiple_ulp_reference_abs_max", "multiple_ulp_max_abs_difference")}
                    for channel, values in case.get("distribution", {}).items()}})

    def read(self, spec, parent, require_pins):
        path = Path(spec["path"])
        path = (parent / path).resolve() if not path.is_absolute() else path.resolve()
        label = str(spec["path"])
        source = {**spec, "path": label}
        self.result["sources"].append(source)
        if not path.exists():
            source["status"] = "missing"
            self.result["pending"].append({"path": label, "status": "missing"})
            return
        actual = digest(path)
        source["actual_sha256"] = actual
        if spec.get("sha256") and actual != spec["sha256"]:
            self.issue(label, "Report SHA256 differs from selection pin; excluded")
            return
        if require_pins and not spec.get("sha256"):
            self.issue(label, "Complete selection requires a SHA256 pin; excluded")
            return
        report = json.loads(path.read_text())
        source["status"] = report.get("status")
        if spec.get("diagnostic"):
            expected = set(spec.get("expected_failed_shapes", []))
            actual_failed = {name for name, entry in report.get("shapes", {}).items() if entry.get("status") != "complete"}
            valid = (bool(spec.get("sha256")) and spec["kind"] == "polynomial" and bool(expected)
                     and spec.get("expected_status") == "completed_with_failures"
                     and report.get("status") == spec["expected_status"] and actual_failed == expected)
            self.require(valid, label, "Diagnostic requires a pinned polynomial report with exactly the declared status and failed shapes")
            if valid:
                self.expected_failed_shapes[label] = expected
        elif spec.get("expected_status"):
            self.require(report.get("status") == spec["expected_status"], label, "Report status differs from expected status")
        info = identity(report)
        source.update(info)
        self.require(str(info["reported_job_id"]) == str(spec["job_id"]), label, "Slurm job differs from selection")
        self.require(info["gpu_uuid"] not in (None, "", "unavailable"), label, "Missing GPU UUID")
        self.identities[str(spec["job_id"])].append(info)
        if report.get("status") in ("running", "pending"):
            self.result["pending"].append({"path": label, "status": report["status"]})
        elif report.get("status") not in ("complete", "completed_with_failures"):
            self.issue(label, f"Report status={report.get('status')}: {report.get('error', '')}")
        base = {"path": label, "job_id": str(spec["job_id"]), "role": spec.get("role"),
                "diagnostic": bool(spec.get("diagnostic")),
                "required_batch_split_policy": spec.get("required_batch_split_policy"),
                "gpu_uuid": info["gpu_uuid"], "source_commit": info["source_commit"],
                "compiler_ptxas_extra_flags": info["compiler_ptxas_extra_flags"],
                "compiler_ptxas_device_debug": info["compiler_ptxas_device_debug"], "xla_flags": info["xla_flags"],
                "dependency_lock_sha256": info["dependency_lock_sha256"],
                "frozen_execution_sha256": info["frozen_execution_sha256"]}
        handler = {"polynomial": self.polynomial, "native_fla": self.native, "training": self.training,
                   "native_model": self.native_model, "batch_split_diagnostic": self.batch_split_diagnostic}.get(spec["kind"])
        if handler is None:
            raise ValueError(f"Unknown report kind: {spec['kind']}")
        try:
            handler(report, base)
        except (KeyError, TypeError, ValueError) as error:
            self.issue(label, f"Malformed or inconsistent report: {type(error).__name__}: {error}")

    def training_comparisons(self, groups):
        """Pool only explicitly selected trials, retaining clock and recipe identity."""
        names = set()
        compatible_fields = ("job_id", "source_commit", "gpu_uuid", "compiler_ptxas_extra_flags",
            "compiler_ptxas_device_debug", "xla_flags",
            "dependency_lock_sha256", "frozen_execution_sha256", "implementation", "precision", "warps",
            "min_ctas_per_sm", "batch", "microbatch", "length", "heads", "key_dim", "value_dim", "degree",
            "parameter_count", "accumulation_steps", "seed", "input_recipe_sha256", "model_recipe_sha256",
            "data_identity_sha256", "data_manifest_sha256", "resident_tokens_sha256", "parameter_dtype",
            "autocast_dtype", "initialization", "scope")
        for group in groups:
            name, job = group["name"], str(group["job_id"])
            roles, timings = group["report_roles"], group["timings"]
            valid = name not in names and len(roles) >= 2 and len(set(roles)) == len(roles)
            valid = valid and bool(timings) and len(set(timings)) == len(timings)
            self.require(valid, name, "Comparison names, report roles, and timing clocks must be explicit and unique")
            names.add(name)
            if not valid:
                continue
            sources = []
            for role in roles:
                matches = [source for source in self.result["sources"]
                           if str(source["job_id"]) == job and source.get("role") == role]
                valid = valid and len(matches) == 1 and matches[0].get("kind") in ("training", "native_model")
                if len(matches) == 1:
                    sources.append(matches[0])
            self.require(valid, name, "Comparison must name exactly one selected training report per role in one job")
            if not valid:
                continue
            for timing in timings:
                selected = []
                for role in roles:
                    matches = [row for row in self.result["training"]
                               if row["job_id"] == job and row.get("role") == role and row["timing"] == timing]
                    self.require(len(matches) <= 1, name, "Comparison contains duplicate trial/clock rows")
                    if len(matches) == 1:
                        selected.append(matches[0])
                complete = len(selected) == len(roles) and all(row["status"] == "complete" for row in selected)
                if all(source.get("status") == "complete" for source in sources):
                    self.require(complete, name, f"Completed comparison is missing the requested clock {timing}")
                result = {"comparison": name, "job_id": job, "report_roles": roles, "timing": timing,
                          "paths": [source["path"] for source in sources], "expected_trials": len(roles),
                          "completed_trials": sum(row["status"] == "complete" for row in selected),
                          "status": "complete" if complete else "incomplete", "samples": 0}
                self.result["training_comparisons"].append(result)
                if not complete:
                    continue
                compatible = all(all(row.get(key) == selected[0].get(key) for key in compatible_fields)
                                 for row in selected)
                self.require(compatible, name, "Named training trials differ in recipe, dimensions, compiler flags, source, device, or data")
                if not compatible:
                    result["status"] = "incompatible"
                    continue
                raw = [value for row in selected for value in self.training_samples[row["path"], timing]]
                result.update({key: selected[0].get(key) for key in compatible_fields})
                result.update(pool(raw, selected[0]["batch"] * selected[0]["length"]))
                means = [row["mean_seconds"] for row in selected]
                result.update(all_finite=all(row["all_finite"] for row in selected),
                              trial_samples=[row["samples"] for row in selected],
                              trial_mean_seconds=means,
                              warmup_updates=[row["warmup_updates"] for row in selected],
                              validated_updates=sum(row["validated_updates"] for row in selected),
                              repeat_mean_spread_percent=100 * (max(means) - min(means)) / statistics.mean(means))

    def finish(self):
        for job, reports in self.identities.items():
            self.require(len({value["gpu_uuid"] for value in reports}) <= 1, job, "Reports from one job have different GPUs")
            self.require(len({value["source_commit"] for value in reports}) <= 1, job, "Reports from one job have different source commits")
        for (job, shape), values in sorted(self.fixtures.items()):
            matched = len(values) == 2 and bool(values.get("polynomial")) and values["polynomial"] == values["native_fla"]
            self.result["fixture_pairs"].append({"job_id": job, "shape": list(shape),
                "status": "matched" if matched else "unpaired" if len(values) < 2 else "mismatch"})
            self.require(len(values) < 2 or matched, job, f"Native/polynomial shared fixture mismatch at {shape}")
        rejected = any(row.get("status") in ("rejected_accuracy", "failed", "failed_stage_profile") and not row.get("expected_failure")
                       for key in ("kernels", "stages", "training") for row in self.result[key])
        self.result["status"] = ("audit_failed" if self.result["issues"] else "incomplete" if self.result["pending"]
                                 else "complete_with_rejections" if rejected else "complete_with_diagnostics" if self.result["diagnostics"] else "complete")
        return self.result


def summarize(selection, *, parent, selection_name):
    summary = Summary(selection_name)
    reports = selection.get("reports", [])
    if not reports:
        raise ValueError("Selection must explicitly name at least one report")
    seen = set()
    for spec in reports:
        key = (str(spec["job_id"]), spec["kind"], spec["path"])
        if key in seen:
            raise ValueError(f"Duplicate selected report: {key}")
        seen.add(key)
        summary.read(spec, parent, selection.get("status") == "complete")
    summary.training_comparisons(selection.get("training_comparisons", []))
    return summary.finish()


def csv_text(result):
    rows = [{"record_type": scope, "scope": scope, **row}
            for scope in ("kernels", "stages", "resources", "training", "training_comparisons") for row in result[scope]]
    fields = list(dict.fromkeys(["record_type", "scope", "job_id", "role", "implementation", "variant", "precision", "warps", "min_ctas_per_sm", "batch",
        "length", "heads", "key_dim", "value_dim", "degree", "direction", "stage", "timing", "status", "samples",
        "mean_seconds", "tokens_per_second", *sorted({key for row in rows for key in row})]))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--selection", type=Path, help="Master record with an explicit reports list")
    group.add_argument("--profile-job", type=int, help="Inspect this exact running profile job, without final hash pins")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root used only by --profile-job")
    parser.add_argument("--json", action="store_true", help="Print the compact summary as JSON")
    parser.add_argument("--output", type=Path, help="Write deterministic summary JSON")
    parser.add_argument("--csv", type=Path, help="Write deterministic flattened CSV")
    args = parser.parse_args()
    if args.profile_job:
        parent = args.root.resolve()
        specs = [("polynomial", "poly-kernels.json"), ("native_fla", "fla-k16-b16.json"), ("native_fla", "fla-k64-b16.json")]
        selection = {"reports": [{"kind": kind, "job_id": str(args.profile_job), "path": f"{args.profile_job}/{name}",
                                   "role": "profile"} for kind, name in specs]}
        selection_name = f"explicit_profile_job:{args.profile_job}"
    else:
        path = (args.selection or ROOT / "record.json").resolve()
        parent = path.parent
        selection_name = str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else str(path)
        selection = json.loads(path.read_text())
    result = summarize(selection, parent=parent, selection_name=selection_name)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for path, contents in ((args.output, encoded), (args.csv, csv_text(result))):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
    if args.json:
        print(encoded, end="")
    else:
        print(f"Audit: {result['status']}; reports={len(result['sources'])}; pending={len(result['pending'])}")
        for row in result["kernels"] + result["training"]:
            if row.get("timing") in ("cuda_events", "graph_cuda_events"):
                continue
            measured = f"{row['mean_seconds'] * 1000:.3f} ms" if row.get("samples") else "no samples"
            print(f"job{row['job_id']} B{row.get('batch')} K{row.get('key_dim')} V{row.get('value_dim')} "
                  f"{row.get('variant') or row.get('implementation')} {row.get('direction')} "
                  f"{row.get('timing')}: {measured}; {row.get('status')}")
        for issue in result["issues"]:
            print(f"ISSUE {issue['source']}: {issue['message']}")
    if result["status"] == "audit_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
