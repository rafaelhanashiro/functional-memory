"""Regenerate polynomial-GDN tuning summaries from explicit, optionally pinned reports.

The default selection is runs/poly-gdn-tuning/record.json, whose reports list
contains {kind, job_id, path, sha256, role}; paths are relative to that record.
Supported kinds are polynomial, native_fla, training, and native_model. A complete selection
requires every SHA256 pin. --profile-job JOB reads only that explicitly named
job's three known profile reports while they are being written. No newest-run
selection, source-report mutation, or cross-GPU pooling is performed.
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
            "compiler_ptxas_extra_flags": compiler_flags}


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
            "pending": [], "issues": [], "fixture_pairs": [],
            "limitations": ["Native FLA is linear tied-gate GDN, not the cubic operator.",
                "Stage calls use resident intermediates; their sum excludes fusion, scheduling, and final gradient accumulation.",
                "Resource figures describe assembled kernels, not measured runtime occupancy.",
                "Kernel accuracy is same-input parity to tf32x3; independent oracle and stress qualification is separate.",
                "Native FLA full-model timings reuse one GPU-resident batch and exclude loading/transfers; JAX training intervals include its input pipeline."]}
        self.identities = defaultdict(list)
        self.fixtures = defaultdict(dict)

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
            if config.get("reference_batch_size") and entry.get("status") == "complete":
                self.require(split.get("status") == "complete" and split.get("accuracy", {}).get("passed") is True,
                             base["path"], "Completed shape lacks a passing batch-split reference check")
                self.require(set(split.get("accuracy", {}).get("channels", {})) == CHANNELS,
                             base["path"], "Batch-split reference check is missing gradient channels")
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
                self.issue(base["path"], f"Failed shape {shape_name}: {entry.get('error')}")
                if not entry.get("variants"):
                    reference = report.get("accuracy_reference", {})
                    self.result["kernels"].append({**base, **fields, "implementation": "polynomial_gdn",
                        "variant": "reference", "precision": reference.get("precision"),
                        "warps": reference.get("num_warps"), "min_ctas_per_sm": reference.get("min_ctas_per_sm"),
                        "direction": "forward_backward", "timing": "synchronized_wall", "status": "failed",
                        "samples": 0, "error": entry.get("error")})
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
        for key in ("compute_seconds", "interval_seconds"):
            measured = [sample[key] for sample in samples[warmup:] if sample.get(key) is not None]
            row = {**base, **shape, "implementation": report.get("resolved_implementation"),
                   "precision": runtime.get("POLY_GDN_VALUE_PRECISION"), "warps": runtime.get("POLY_GDN_NUM_WARPS"),
                   "min_ctas_per_sm": runtime.get("POLY_GDN_MIN_CTAS_PER_SM"),
                   "direction": "training_update", "timing": key, "status": report["status"], "all_finite": finite,
                   "parameter_count": report.get("parameter_count"), "accumulation_steps": report.get("accumulation_steps"),
                   "data_identity_sha256": report.get("data_identity_sha256"),
                   **pool(measured, report["tokens_per_update"])}
            self.result["training"].append(row)

    def native_model(self, report, base):
        """Native upstream model compute; never label this an input-pipeline interval."""
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
                   "status": report["status"], "all_finite": finite, "validated_updates": len(all_samples),
                   "warmup_updates": len(warmups), "parameter_count": report.get("parameter_count"),
                   "accumulation_steps": settings["accumulation_steps"],
                   "parameter_dtype": settings.get("parameter_dtype"), "autocast_dtype": settings.get("autocast_dtype"),
                   "initialization": settings.get("initialization"),
                   "data_manifest_sha256": data.get("manifest_sha256"), "resident_tokens_sha256": data.get("tokens_sha256"),
                   **pool([sample[key] for sample in samples], shape["batch"] * shape["length"])}
            self.result["training"].append(row)

    def read(self, spec, parent, require_pins):
        path = Path(spec["path"])
        path = (parent / path).resolve() if not path.is_absolute() else path.resolve()
        source = {**spec, "path": str(path)}
        self.result["sources"].append(source)
        if not path.exists():
            source["status"] = "missing"
            self.result["pending"].append({"path": str(path), "status": "missing"})
            return
        actual = digest(path)
        source["actual_sha256"] = actual
        if spec.get("sha256") and actual != spec["sha256"]:
            self.issue(str(path), "Report SHA256 differs from selection pin; excluded")
            return
        if require_pins and not spec.get("sha256"):
            self.issue(str(path), "Complete selection requires a SHA256 pin; excluded")
            return
        report = json.loads(path.read_text())
        source["status"] = report.get("status")
        info = identity(report)
        source.update(info)
        self.require(str(info["reported_job_id"]) == str(spec["job_id"]), str(path), "Slurm job differs from selection")
        self.require(info["gpu_uuid"] not in (None, "", "unavailable"), str(path), "Missing GPU UUID")
        self.identities[str(spec["job_id"])].append(info)
        if report.get("status") in ("running", "pending"):
            self.result["pending"].append({"path": str(path), "status": report["status"]})
        elif report.get("status") not in ("complete", "completed_with_failures"):
            self.issue(str(path), f"Report status={report.get('status')}: {report.get('error', '')}")
        base = {"path": str(path), "job_id": str(spec["job_id"]), "role": spec.get("role"),
                "gpu_uuid": info["gpu_uuid"], "source_commit": info["source_commit"],
                "compiler_ptxas_extra_flags": info["compiler_ptxas_extra_flags"]}
        handler = {"polynomial": self.polynomial, "native_fla": self.native, "training": self.training,
                   "native_model": self.native_model}.get(spec["kind"])
        if handler is None:
            raise ValueError(f"Unknown report kind: {spec['kind']}")
        try:
            handler(report, base)
        except (KeyError, TypeError, ValueError) as error:
            self.issue(str(path), f"Malformed or inconsistent report: {type(error).__name__}: {error}")

    def finish(self):
        for job, reports in self.identities.items():
            self.require(len({value["gpu_uuid"] for value in reports}) <= 1, job, "Reports from one job have different GPUs")
            self.require(len({value["source_commit"] for value in reports}) <= 1, job, "Reports from one job have different source commits")
        for (job, shape), values in sorted(self.fixtures.items()):
            matched = len(values) == 2 and bool(values.get("polynomial")) and values["polynomial"] == values["native_fla"]
            self.result["fixture_pairs"].append({"job_id": job, "shape": list(shape),
                "status": "matched" if matched else "unpaired" if len(values) < 2 else "mismatch"})
            self.require(len(values) < 2 or matched, job, f"Native/polynomial shared fixture mismatch at {shape}")
        rejected = any(row.get("status") in ("rejected_accuracy", "failed", "failed_stage_profile")
                       for key in ("kernels", "stages", "training") for row in self.result[key])
        self.result["status"] = ("audit_failed" if self.result["issues"] else "incomplete" if self.result["pending"]
                                 else "complete_with_rejections" if rejected else "complete")
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
    return summary.finish()


def csv_text(result):
    rows = [{"scope": scope, **row} for scope in ("kernels", "stages", "resources", "training") for row in result[scope]]
    fields = list(dict.fromkeys(["scope", "job_id", "role", "implementation", "variant", "precision", "warps", "min_ctas_per_sm", "batch",
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
        parent, selection_name = path.parent, str(path)
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
