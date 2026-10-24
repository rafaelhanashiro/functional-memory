"""Aggregate retained B300 evidence without changing any failed raw verdict.

This is a read-only gate over completed reports. It writes one separate decision,
never launches training, and never substitutes diagnostic pilots for production
acceptance. The roundoff protocol was committed after the initial parity rejection
and before the independent replay; its thresholds remain unchanged here.
"""
from __future__ import annotations

import argparse
import copy
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scaffolds.launch.frozen import changes, frozen_hash, is_frozen

ROOT = Path(__file__).resolve().parents[3]
STUDY = "transformer_200m_4b"
RUNS = {
    "r833fbb6eedcb": "db94207d47c1dc49a4bb7ad8f109a7d3fbe78ce9c49ad951df639eedfd23efc3",
    "rfd48f7bbe9f4": "83c4ece8ac43adcd62155eaf9d24c904e9e8f244dec0d9ae361716ce9714bc3c",
}
PROJECTIONS = ("query_polynomial_kernel", "key_polynomial_kernel")
COEFFICIENTS = ("query_polynomial_bias", "query_polynomial_amplitude", "key_polynomial_amplitude")
COEFFICIENT_PARAMETERS = {f"['{name}'].value" for name in COEFFICIENTS}
DIAGNOSTIC = "scaffolds/transformer/benchmarks/diagnose_query_key_model_roundoff.py"
PROTOCOL = "scaffolds/transformer/validation/runs/query-key-b300-qualification/roundoff-protocol.json"
PROTOCOL_COMMIT = "7293e9702c4f4b8e9dfb4001dbad6da946cd3ede"
REPEAT_PROTOCOL = "scaffolds/transformer/validation/runs/query-key-b300-qualification/repeat-protocol.json"
REPEAT_PROTOCOL_COMMIT = "c05b6ed306dd93b1d6e6c51b639790db7fc04c37"
COEFFICIENT_PROTOCOL = "scaffolds/transformer/validation/runs/query-key-b300-qualification/coefficient-protocol.json"
COEFFICIENT_PROTOCOL_COMMIT = "61a4f9139f8a113580eaecf7eac1ae4f188b670b"
EMBEDDING = "['token_embedding']['embedding'].value"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def group_key(group):
    return (group["parameter"], group["layer"], group.get("head"), group["order"])


def projection_failures(report):
    return {group_key(dict(parameter=leaf["parameter"], **group))
            for leaf in report["gradients"]["leaves"]
            if any(name in leaf["parameter"] for name in PROJECTIONS)
            for group in leaf.get("groups", {}).get("failed", [])}


def coefficient_failures(report):
    return {group_key(dict(parameter=leaf["parameter"], **group))
            for leaf in report["gradients"]["leaves"] if leaf["parameter"] in COEFFICIENT_PARAMETERS
            for group in leaf.get("groups", {}).get("failed", [])}


def attributable_failures(report):
    return projection_failures(report) | coefficient_failures(report)


def require_coverage(expected, covered):
    require(expected <= covered, f"Rejected groups lack passing attribution: {sorted(expected - covered)}")


def require_canonical_snapshot(report, canonical):
    """Group identities count only when measured at the canonical model/data snapshot."""
    actual = (report.get("compared_parameter_sha256", report.get("parameter_sha256")),
              report["batch"]["sha256"], report["batch"]["manifest_sha256"],
              tuple(report["forward_rng"]) if "forward_rng" in report else canonical[3],
              report["configuration"]["perturbation_seed"])
    require(actual == canonical, "Oracle/historical snapshot differs from canonical initial/assessment parity")


def metric_passes(row, tolerance):
    return (row.get("all_finite") is True and row.get("reference_nonzero") is True
            and finite(row.get("relative_l2_error")) and 0 <= row["relative_l2_error"] <= tolerance
            and finite(row.get("max_abs_error")) and finite(row.get("reference_max_abs"))
            and row["reference_max_abs"] > 0
            and 0 <= row["max_abs_error"] <= 5 * tolerance * row["reference_max_abs"])


def whole_gradient_gates(report):
    leaves = report["gradients"]["leaves"]
    require(bool(leaves) and len(leaves) == report["gradients"]["leaf_count"], "Missing gradient leaves")
    require(all(metric_passes(leaf, .025) for leaf in leaves), "Unchanged whole-parameter gate failed")
    require(all(not leaf.get("groups", {}).get("failed") for leaf in leaves
                if not any(name in leaf["parameter"] for name in PROJECTIONS)
                and leaf["parameter"] not in COEFFICIENT_PARAMETERS),
            "A rejected group outside the explicitly supported projection/coefficient families is a hard failure")
    require(all(g["nonzero_count"] == g["count"] for leaf in leaves if (g := leaf.get("groups"))),
            "Vacuous zero-gradient mechanism group")


def whole_model_gates(report, *, replay=False):
    whole_gradient_gates(report)
    if replay:
        require(report.get("logits_exact") is True and finite(report.get("loss_absolute_difference"))
                and 0 <= report["loss_absolute_difference"] <= .002, "Replay forward gate failed")
    else:
        require(metric_passes(report["logits"], .005), "Unchanged full-model logit gate failed")
        require(report["loss"].get("accepted") is True and finite(report["loss"].get("absolute_error"))
                and 0 <= report["loss"]["absolute_error"] <= .002, "Unchanged loss gate failed")
    require(report.get("source_recipe_unchanged") is True, "Parity recipe changed")


def repeat_gates(report):
    """Apply the prospective embedding-only exception to measured leaf reports."""
    trials = report["full_model_trials"]
    require(len(trials) == 2 and {t["tile"] for t in trials} == {32, 64}, "Missing geometry repeat trials")
    expected = {leaf["parameter"]: leaf["shape"] for leaf in report["gradients"]["leaves"]}
    require(EMBEDDING in expected, "Full-model report lacks the embedding gradient")
    outcomes = []
    for trial in trials:
        evaluations = trial["evaluations"]
        require(len(evaluations) == 2, "Both repeated evaluations are required")
        baseline = trial["uninstrumented"]
        require(all(finite(x["loss"]) for x in (baseline, *evaluations)), "Nonfinite repeat loss")
        for flag, a, b in (("repeat_forward_exact", evaluations[0], evaluations[1]),
                           ("instrumentation_forward_exact", baseline, evaluations[0])):
            require(trial.get(flag) is True and a["loss"] == b["loss"]
                    and a["logits_sha256"] == b["logits_sha256"], "Repeat/capture forward arrays differ")
        strict_repeat = evaluations[0] == evaluations[1]
        strict_capture = baseline == evaluations[0]
        require(trial.get("deterministic") is strict_repeat
                and trial.get("instrumentation_array_exact") is strict_capture,
                "Raw strict repeat verdict does not match its hashes")
        outcome = {"tile": trial["tile"], "raw_deterministic": strict_repeat,
                   "raw_instrumentation_array_exact": strict_capture, "comparisons": {}}
        for name in ("repeat_gradients", "instrumentation_gradients"):
            comparison = trial[name]
            leaves = comparison["leaves"]
            require(len(leaves) == len(expected) == comparison["leaf_count"]
                    and {leaf["parameter"]: leaf["shape"] for leaf in leaves} == expected,
                    "Repeat report omitted or changed a gradient leaf")
            embedding = []
            for leaf in leaves:
                is_embedding = leaf["parameter"] == EMBEDDING
                require(leaf.get("is_embedding") is is_embedding, "Incorrect embedding classification")
                require(metric_passes(leaf, .025), "Repeat gradient exceeds unchanged whole-leaf bounds")
                require(all(isinstance(leaf.get(key), str) and len(leaf[key]) == 64
                            for key in ("actual_sha256", "reference_sha256")), "Repeat leaf hashes missing")
                if is_embedding:
                    embedding.append(leaf)
                else:
                    require(leaf["array_equal"] is True and leaf["actual_sha256"] == leaf["reference_sha256"]
                            and leaf["relative_l2_error"] == 0 and leaf["max_abs_error"] == 0,
                            f"Nonembedding repeat gradient changed: {leaf['parameter']}")
            require(len(embedding) == 1 and comparison.get("nonembedding_array_exact") is True,
                    "Variation is not confined to exactly one embedding gradient")
            leaf = embedding[0]
            outcome["comparisons"][name] = {
                "nonembedding_gradients_array_exact": True,
                "embedding_array_equal": leaf["array_equal"],
                "embedding_relative_l2_error": leaf["relative_l2_error"],
                "embedding_max_error_over_reference_max": leaf["max_abs_error"] / leaf["reference_max_abs"],
            }
        outcomes.append(outcome)
    return outcomes


def coefficient_group_gates(report, failures):
    """Require independent all-head contractions and attribution for supported groups."""
    wanted = {key for key in failures if key[0] in COEFFICIENT_PARAMETERS}
    require(all(key[2] is None for key in wanted), "Coefficient groups must be vectors across all heads")
    layers = {key[1] for key in wanted}
    degree = report["input_recipe"]["configuration"]["model"]["attention_degree"]
    groups = report.get("coefficient_groups", [])
    expected = {(parameter, layer, None, order) for parameter in COEFFICIENT_PARAMETERS
                for layer in layers for order in range(degree + 1)}
    require(len(groups) == len(expected) and {group_key(group) for group in groups} == expected,
            "Coefficient contractions must cover all families/orders at every rejected layer")
    cases = {(case["layer"], case["head"]) for case in report["cases"] if case.get("accepted") is True}
    require({(layer, head) for layer in layers for head in range(14)} <= cases,
            "Coefficient replay lacks passing local FP64 cases for all 14 heads")
    labels = {f"tile{tile}_at_incoming{incoming}_versus_fp64" for tile in (32, 64) for incoming in (32, 64)}
    labels |= {f"incoming{incoming}_fp32_vjp_versus_fp64_contraction" for incoming in (32, 64)}
    labels |= {f"tile{tile}_replay_versus_full_model" for tile in (32, 64)}
    covered = set()
    for group in groups:
        key = group_key(group)
        require(group["heads"] == list(range(14)) and group.get("accepted") is True,
                "Coefficient contraction omitted a head or failed")
        require(set(group["comparisons"]) == labels, "Coefficient oracle/projection/replay comparisons missing")
        for label, comparison in group["comparisons"].items():
            tolerance = .001 if label.endswith("replay_versus_full_model") else .05
            require(comparison.get("accepted") is True and metric_passes(comparison, tolerance),
                    "Independent coefficient contraction gate failed")
        require(group.get("required_attribution") is (key in wanted), "Coefficient attribution selection differs from union")
        if key in wanted:
            attribution = group["attribution"]
            require(attribution.get("accepted") is True and metric_passes(attribution, .1)
                    and attribution["production_projection_control"].get("accepted") is True
                    and metric_passes(attribution["production_projection_control"], .1),
                    "Coefficient attribution residual exceeds 10%")
            covered.add(key)
    return covered


def assessment_stage_gates(summary):
    require(summary.get("finished_utc") and not summary.get("active_stage") and not summary.get("error"),
            "Assessment did not finish collecting evidence")
    require(summary.get("source_recipes_unchanged") is True and summary.get("scratch_removed") is True,
            "Assessment recipe or scratch-cleanup gate failed")
    require(summary.get("status") in ("complete", "failed"), "Assessment is not terminal")
    expected = {"hardware", "focused-gpu-tests"} | {
        f"{run}:{stage}" for run in RUNS
        for stage in ("full-model-parity", "throughput-production", "throughput-original", "pilot", "resume")}
    stages = summary["stages"]
    require(len(stages) == len(expected) and {s["name"] for s in stages} == expected,
            "Assessment stages are missing, duplicated, or unexpected")
    for stage in stages:
        parity = stage["name"].endswith(":full-model-parity")
        allowed = (("complete", 0), ("failed", 1)) if parity else (("complete", 0),)
        require((stage.get("status"), stage.get("exit_code")) in allowed
                and not stage.get("validation_error"), f"Operationally failed assessment stage: {stage['name']}")


class Assessment:
    def __init__(self, args):
        self.args = args
        self.report = {"format": "b300-query-key-qualification-decision", "schema_version": 1,
                       "assessor_source_sha256": digest(__file__),
                       "status": "failed", "production_allowed": False,
                       "created_at_utc": datetime.now(UTC).isoformat(),
                       "production_source_commit": args.production_source,
                       "evidence": [], "source_equivalence": [], "executed_benchmarks": [], "runs": {}, "oracles": [],
                       "interpretation": "Raw 5% projection/coefficient-group rejections and strict whole-gradient repeat verdicts remain unchanged. Qualification requires independent oracle/attribution gates plus the prospective repeat addendum: exact forward/nonembedding gradients and embedding variation within unchanged whole-leaf bounds.",
                       "limitation": "Attribution locates differences in the incoming cotangent; it does not identify the upstream operation responsible. Production must still pass its own allocation acceptance."}
        self.evidence = {}
        self.sources = set()

    def artifact(self, path, expected=None):
        path = Path(path).resolve()
        require(path != self.args.output.resolve(), "Decision output would overwrite input evidence")
        sha = digest(path)
        require(expected is None or sha == expected, f"Artifact hash mismatch: {path}")
        if str(path) not in self.evidence:
            item = {"path": str(path), "sha256": sha}
            self.evidence[str(path)] = item
            self.report["evidence"].append(item)
        return path

    def read(self, path, expected=None):
        return json.loads(self.artifact(path, expected).read_text())

    def linked(self, item):
        return self.read(item["path"], item["sha256"])

    def junit(self, item, count):
        path = self.artifact(item["path"], item["sha256"])
        cases = list(ET.parse(path).iter("testcase"))
        require(len(cases) == count and all(not any(c.find(tag) is not None for tag in ("failure", "error", "skipped"))
                                          for c in cases), f"Tests failed, skipped, or missing: {path}")

    def equivalent_source(self, source):
        if source in self.sources:
            return
        require(isinstance(source, str) and len(source) == 40, "Evidence lacks a full source commit")
        rows = [line.split("\t") for line in git("diff", "--no-renames", "--name-status", source,
                                                self.args.production_source).splitlines()]
        for row in rows:
            status, path = row
            require(path.startswith("scaffolds/transformer/validation/") or (status in ("A", "M") and path == DIAGNOSTIC),
                    f"Source difference exceeds the diagnostic-only allowance: {status} {path}")
            if is_frozen(path, STUDY, "transformer"):
                require(status in ("A", "M") and path == DIAGNOSTIC,
                        f"Production differs from measured execution: {status} {path}")
        self.report["source_equivalence"].append({"source_commit": source,
            "frozen_execution_sha256": frozen_hash(ROOT, source, STUDY, "transformer"),
            "changed_paths": [{"status": status, "path": path} for status, path in rows],
            "allowed_frozen_diagnostic_change": DIAGNOSTIC})
        self.sources.add(source)

    def provenance(self, report, *, historical=False):
        source = report["provenance"]
        self.equivalent_source(source["source_commit"])
        require(source["dependency_lock_sha256"] == self.lock, "Dependency locks differ")
        require(len(source["visible_devices"]) == 1 and "B300" in source["device"]["kind"],
                "Evidence was not collected on one B300")
        benchmark = {"query-key-full-model-gradient-qualification": "qualify_query_key_model.py",
                     "query-key-model-roundoff-diagnostic": "diagnose_query_key_model_roundoff.py",
                     "fresh-training-throughput": "benchmark_training_throughput.py"}[report["format"]]
        relative = f"scaffolds/transformer/benchmarks/{benchmark}"
        executed = subprocess.check_output(["git", "show", f"{source['source_commit']}:{relative}"], cwd=ROOT)
        require(hashlib.sha256(executed).hexdigest() == source["benchmark_source_sha256"],
                "Executed benchmark hash differs from its committed source")
        source_entry = {"source_commit": source["source_commit"], "path": relative,
                        "sha256": source["benchmark_source_sha256"]}
        if source_entry not in self.report["executed_benchmarks"]:
            self.report["executed_benchmarks"].append(source_entry)
        if digest(ROOT / relative) == source["benchmark_source_sha256"]:
            self.artifact(ROOT / relative, source["benchmark_source_sha256"])

    def run_id(self, report):
        sha = report["input_recipe"]["sha256"]
        matches = [run for run, expected in RUNS.items() if sha == expected]
        require(len(matches) == 1, "Evidence describes another recipe")
        return matches[0]

    def oracle(self, report, raw_by_hash, snapshots):
        require(report["status"] in ("complete", "failed") and report.get("finished_at_utc")
                and not report.get("error"), "Independent oracle crashed/incomplete")
        self.provenance(report)
        require(subprocess.run(["git", "merge-base", "--is-ancestor", REPEAT_PROTOCOL_COMMIT,
                                report["provenance"]["source_commit"]], cwd=ROOT, check=False).returncode == 0,
                "Repeat quantification predates the prospective protocol")
        run = self.run_id(report)
        require_canonical_snapshot(report, snapshots[run])
        selected_reports = report["selected_parity_reports"]
        require(bool(selected_reports), "Oracle must name its historical parity reports")
        historical_failures = set()
        for selected in selected_reports:
            raw = raw_by_hash.get(selected["sha256"])
            if raw is None:
                raw = self.linked(selected)
                self.provenance(raw, historical=True)
                whole_gradient_gates(raw)
            require_canonical_snapshot(raw, snapshots[run])
            require(self.run_id(raw) == run and report.get("same_parameters_and_batch_as_parity") is True
                    and report["parameter_sha256"] == raw.get("compared_parameter_sha256", raw.get("parameter_sha256"))
                    and report["batch"]["sha256"] == raw["batch"]["sha256"]
                    and report["batch"]["manifest_sha256"] == raw["batch"]["manifest_sha256"],
                    "Oracle snapshot differs from historical parity")
            historical_failures.update(attributable_failures(raw))
        require(report.get("failed_groups_covered") is True and not report.get("uncovered_failures")
                and not report.get("unhandled_nonprojection_groups")
                and not report.get("unhandled_noncoefficient_groups"), "Oracle coverage failed")
        whole_model_gates(report, replay=True)
        repeats = repeat_gates(report)
        config = report["configuration"]
        require(config["heads_captured"] == list(range(14)) and config["layers_captured"] == list(range(15))
                and report["captured_layers"] == list(range(15)), "Capture omitted a model head or layer")
        require((config["local_relative_l2_tolerance"], config["projection_group_tolerance"],
                 config["attribution_residual_tolerance"]) == (.025, .05, .1), "Oracle tolerances changed")
        covered = set()
        require(any((c["layer"], c["head"]) == (14, 2) for c in report["cases"]), "Missing control case")
        degree = report["input_recipe"]["configuration"]["model"]["attention_degree"]
        for case in report["cases"]:
            require(case.get("accepted") is True and case.get("operands_exact_between_geometries") is True
                    and case.get("projection_inputs_exact_between_geometries") is True,
                    "Oracle case failed or captured different forward inputs")
            comparisons = case["comparisons"]
            expected = {f"tile{tile}_at_incoming{incoming}_versus_fp64" for tile in (32, 64) for incoming in (32, 64)}
            expected |= {f"tile32_versus_tile64_at_incoming{incoming}" for incoming in (32, 64)}
            require(set(comparisons) == expected, "Local oracle comparisons missing")
            for comparison in comparisons.values():
                require(comparison.get("accepted") is True and set(comparison["gradients"]) == {"q", "k", "v", "coefficients", "factors"},
                        "Local gradient components missing/failed")
                for name, row in comparison["gradients"].items():
                    require(metric_passes(row, .025), "Local FP64 gradient gate failed")
                    if name in ("coefficients", "factors"):
                        require({o["order"] for o in row["orders"]} == set(range(degree + 1))
                                and all(metric_passes(o, .025) for o in row["orders"]), "Per-order FP64 gate failed")
            projections = case["projection_comparisons"]
            labels = {f"incoming{i}_{k}_fp32_vjp_versus_fp64_contraction" for i in (32, 64) for k in ("query", "key")}
            labels |= {f"tile{t}_at_incoming{i}_versus_fp64_{k}" for t in (32, 64) for i in (32, 64) for k in ("query", "key")}
            labels |= {f"tile{t}_{k}_replay_versus_full_model" for t in (32, 64) for k in ("query", "key")}
            require(set(projections) == labels, "Projection oracle/replay comparisons missing")
            for label, row in projections.items():
                tolerance = .001 if label.endswith("replay_versus_full_model") else .05
                require(row.get("accepted") is True and {o["order"] for o in row["orders"]} == set(range(degree + 1))
                        and all(metric_passes(o, tolerance) for o in row["orders"]), "Projection contraction gate failed")
            for row in case["attribution"]:
                require(row.get("accepted") is True and metric_passes(row, .1)
                        and metric_passes(row["production_projection_control"], .1), "Attribution residual exceeds 10%")
                covered.add(group_key(dict(layer=case["layer"], head=case["head"], **row)))
        failures = historical_failures | attributable_failures(report)
        coefficient_keys = {key for key in failures if key[0] in COEFFICIENT_PARAMETERS}
        if coefficient_keys:
            require(subprocess.run(["git", "merge-base", "--is-ancestor", COEFFICIENT_PROTOCOL_COMMIT,
                                    report["provenance"]["source_commit"]], cwd=ROOT, check=False).returncode == 0,
                    "Coefficient replay predates its prospective coverage protocol")
        covered.update(coefficient_group_gates(report, failures))
        require_coverage(failures, covered)
        require({group_key(g) for g in report["full_model_failed_projection_groups"]} == failures - coefficient_keys,
                "Oracle did not retain the complete projection failure union")
        require({group_key(g) for g in report.get("full_model_failed_coefficient_groups", [])} == coefficient_keys,
                "Oracle did not retain the complete coefficient failure union")
        strict_pass = all(t["raw_deterministic"] and t["raw_instrumentation_array_exact"] for t in repeats)
        require(report["status"] == ("complete" if strict_pass else "failed"),
                "Oracle strict status has an unexplained failure")
        self.report["oracles"].append({"run_id": run, "raw_status": report["status"],
            "selected_report_failures_reproduced": report["selected_report_failures_reproduced"],
            "historical_failure_membership_recurrence_required": False,
            "repeat_addendum": "passed", "repeat_measurements": repeats,
            "local_projection_and_attribution_gates": "passed",
            "coefficient_groups_independently_attributed": len(coefficient_keys)})
        return run, failures, covered

    def training_evidence(self, run, row):
        timings = {}
        for geometry, tiles in (("production", [32, 32]), ("original", [64, 64])):
            timing = self.linked(row["throughput"][geometry])
            self.provenance(timing)
            require(self.run_id(timing) == run and timing["status"] == "complete" and timing["all_finite"] is True
                    and timing["scratch_removed"] is True and timing["source_recipe_unchanged"] is True,
                    "Throughput did not finish with finite values and cleanup")
            require(timing["warmup_updates"] == 3 and timing["timed_updates"] == 3
                    and timing["tokens_per_update"] == 524288 and timing["accumulation_steps"] == 16,
                    "Throughput recipe or measurement window changed")
            original, resolved = timing["input_recipe"]["configuration"], timing["configuration"]
            require(all(original[key] == resolved[key] for key in ("model", "optim", "data")),
                    "Throughput changed the model, optimizer horizon, or scientific data recipe")
            allowed = {"workdir", "checkpoint_dir", "wandb_mode", "wandb_dir", "source_commit", "eval_every", "ckpt_every"}
            require({k: v for k, v in original["train"].items() if k not in allowed}
                    == {k: v for k, v in resolved["train"].items() if k not in allowed},
                    "Throughput changed training settings beyond documented diagnostic overrides")
            for split in ("train", "eval"):
                self.artifact(original["data"][f"{split}_manifest"], timing["data_identity"][f"{split}_manifest_sha256"])
            audit = timing["geometry_audit"]
            require(audit["geometry"] == geometry and audit["backward_tiles"] == tiles and audit["forward_tiles"] == [64, 64]
                    and audit["production_geometry_before_override"][run]["backward"] == [32, 32]
                    and (audit["override"] is None if geometry == "production" else bool(audit["override"])),
                    "Throughput geometry audit failed")
            require([s["step"] for s in timing["samples"]] == list(range(1, 7))
                    and all(s["finite_loss_and_update_numerics"] is True and finite(s["loss"]) for s in timing["samples"])
                    and all(finite(s["interval_seconds"]) and s["interval_seconds"] > 0 for s in timing["samples"][3:]),
                    "Throughput samples are incomplete/nonfinite")
            require(finite(timing["steady_training"]["mean_seconds"]) and timing["steady_training"]["mean_seconds"] > 0,
                    "Invalid mean training update time")
            require(timing["outcome"]["data_parallel_devices"] == 1 and timing["resolved_backend"] == "fused"
                    and timing["outcome"]["data_start_sequence"] == 0 and timing["outcome"]["final_step"] == 6,
                    "Throughput allocation/backend/start changed")
            timings[geometry] = timing
        a, b = timings["production"], timings["original"]
        require(a["provenance"]["device"].get("uuid") and a["provenance"]["device"]["uuid"] == b["provenance"]["device"].get("uuid"),
                "Throughput workers did not use the same physical B300")
        require(a["data_identity"] == b["data_identity"] and a["input_recipe"]["configuration"] == b["input_recipe"]["configuration"],
                "Throughput data or source recipes differ")
        require(a["provenance"]["package_versions"] == b["provenance"]["package_versions"],
                "Matched throughput software environments differ")
        configurations = [copy.deepcopy(x["configuration"]) for x in (a, b)]
        for config in configurations:
            for name in ("workdir", "checkpoint_dir"):
                config["train"].pop(name, None)
        require(configurations[0] == configurations[1], "Matched throughput changed settings beyond scratch paths")
        resume = row["pilot_resume"]
        require(resume["status"] == "complete" and resume["scratch_removed"] is True
                and finite(resume["validation_nll"]) and resume["attention_diagnostics_valid"] is True,
                "Pilot validation, diagnostics, or cleanup failed")
        before, after = self.linked(resume["pilot"]), self.linked(resume["resume"])
        require(before["final_step"] == 20 and after["restored_step"] == 20 and after["final_step"] == 21
                and after["data_start_sequence"] == 5120
                and all(x["data_parallel_devices"] == 1 and x["attention_backend"] == "fused" for x in (before, after)),
                "Pilot/resume step, allocation, or backend differs")
        audits = [x["checkpoint_audit"] for x in (before, after)]
        require(all(x.get("final_state_sha256") and x["final_state_sha256"] == x["checkpoint_state_sha256"] for x in audits)
                and audits[0]["final_state_sha256"] == audits[1]["start_state_sha256"]
                and audits[0].get("next_update") is not None and audits[0]["next_update"] == audits[1]["first_update"],
                "Fresh-process checkpoint/state/data/RNG restoration failed")
        return {"production_update_seconds": a["steady_training"]["mean_seconds"],
                "original_update_seconds": b["steady_training"]["mean_seconds"],
                "speedup": b["steady_training"]["mean_seconds"] / a["steady_training"]["mean_seconds"],
                "pilot_resume": "passed"}

    def assess(self):
        self.lock = digest(ROOT / "scaffolds/transformer/uv.lock")
        require(not changes(ROOT, self.args.production_source, STUDY, "transformer"), "Production frozen execution set changed")
        self.report["production_frozen_execution_sha256"] = frozen_hash(ROOT, self.args.production_source, STUDY, "transformer")
        self.report["dependency_lock_sha256"] = self.lock
        protocol = self.read(ROOT / PROTOCOL)
        require((ROOT / PROTOCOL).read_text().strip() == git("show", f"{PROTOCOL_COMMIT}:{PROTOCOL}"), "Predeclared protocol was changed")
        require(protocol["required_checks"] == {
            "captured_forward_inputs_array_exact": True, "local_attention_fp64_relative_l2": .025,
            "projection_contraction_fp64_group_relative_l2": .05, "repeat_each_geometry_array_exact": True,
            "unexplained_fraction_of_observed_failed_group_difference_max": .1}, "Protocol gates differ")
        self.report["protocol_commit"] = PROTOCOL_COMMIT
        repeat_protocol = self.read(ROOT / REPEAT_PROTOCOL)
        require((ROOT / REPEAT_PROTOCOL).read_text().strip() == git("show", f"{REPEAT_PROTOCOL_COMMIT}:{REPEAT_PROTOCOL}"),
                "Prospective repeat protocol was changed")
        require(repeat_protocol["required_checks"] == {
            "all_nonembedding_gradients_array_exact": True, "embedding_gradient_all_finite_nonzero": True,
            "embedding_gradient_max_absolute_error_over_reference_max": .125,
            "embedding_gradient_relative_l2_max": .025, "loss_and_logits_array_exact": True},
            "Prospective repeat protocol gates differ")
        self.report["repeat_protocol_commit"] = REPEAT_PROTOCOL_COMMIT
        coefficient_protocol = self.read(ROOT / COEFFICIENT_PROTOCOL)
        require((ROOT / COEFFICIENT_PROTOCOL).read_text().strip()
                == git("show", f"{COEFFICIENT_PROTOCOL_COMMIT}:{COEFFICIENT_PROTOCOL}"),
                "Prospective coefficient coverage protocol was changed")
        require(coefficient_protocol["additional_parameter_families"] == list(COEFFICIENTS)
                and coefficient_protocol["required_checks"] == {
                    "all_heads_covered": 14, "independent_fp64_coefficient_vector_relative_l2_max": .05,
                    "same_forward_operands_and_projection_inputs": True,
                    "unexplained_fraction_of_observed_group_difference_max": .1},
                "Coefficient coverage protocol gates differ")
        self.report["coefficient_protocol_commit"] = COEFFICIENT_PROTOCOL_COMMIT
        summary = self.read(self.args.assessment_dir / "summary.json")
        assessment_stage_gates(summary)
        self.equivalent_source(summary["source_commit"])
        require(summary["dependency_lock_sha256"] == self.lock and set(summary["runs"]) == set(RUNS), "Assessment scope/lock differs")
        require({r["run_id"]: r["sha256"] for r in summary["recipes"]} == RUNS, "Assessment recipes differ")
        for run, sha in RUNS.items():
            self.artifact(ROOT / "experiments" / STUDY / "runs" / run / "config.yaml", sha)
        hardware = self.linked(summary["hardware"])
        require(hardware["status"] == "complete" and all(hardware["production_geometry"][run]["backward"] == [32, 32]
                and hardware["production_geometry"][run]["forward"] == [64, 64] for run in RUNS), "Production selector gate failed")
        cpu = self.read(self.args.cpu_report)
        require(cpu["status"] == "passed" and cpu["tests_passed"] == 96 and cpu["dependency_lock_sha256"] == self.lock, "CPU checks failed")
        for path, sha in cpu["source_sha256"].items():
            self.artifact(ROOT / path, sha)
        self.junit(cpu["junit"], 96)
        gpu = next(s for s in summary["stages"] if s["name"] == "focused-gpu-tests")
        require(gpu["tests_passed"] == 12, "Focused GPU tests missing")
        self.junit(gpu["junit"], 12)
        raw_by_hash = {}
        initial = self.read(self.args.initial_parity)
        raw_by_hash[digest(self.args.initial_parity)] = initial
        for run, row in summary["runs"].items():
            raw = self.linked(row["full_model_parity"])
            require(self.run_id(raw) == run and raw["status"] == row["full_model_parity"]["status"], "Assessment parity identity/status differs")
            raw_by_hash[row["full_model_parity"]["sha256"]] = raw
        expected, covered = {run: set() for run in RUNS}, {run: set() for run in RUNS}
        snapshots = {}
        for raw in raw_by_hash.values():
            require(raw["status"] in ("complete", "failed") and not raw.get("error"), "Parity crashed/incomplete")
            self.provenance(raw)
            whole_model_gates(raw)
            config = raw["configuration"]
            require((config["gradient_leaf_relative_l2_tolerance"], config["mechanism_group_relative_l2_tolerance"],
                     config["logit_relative_l2_tolerance"], config["loss_absolute_tolerance"]) == (.025, .05, .005, .002),
                    "Original full-model thresholds changed")
            require(raw["status"] == ("failed" if attributable_failures(raw) else "complete"),
                    "Raw full-model status does not match its retained rejection")
            run = self.run_id(raw)
            snapshot = (raw["compared_parameter_sha256"], raw["batch"]["sha256"], raw["batch"]["manifest_sha256"],
                        tuple(raw["forward_rng"]), raw["configuration"]["perturbation_seed"])
            require(run not in snapshots or snapshot == snapshots[run], "Raw parity snapshots differ")
            snapshots[run] = snapshot
            expected[run].update(attributable_failures(raw))
        for path in self.args.oracle:
            run, failures, coverage = self.oracle(self.read(path), raw_by_hash, snapshots)
            expected[run].update(failures)
            covered[run].update(coverage)
        for run, row in summary["runs"].items():
            require_coverage(expected[run], covered[run])
            outcome = self.training_evidence(run, row)
            self.report["runs"][run] = {**outcome, "raw_full_model_status": row["full_model_parity"]["status"],
                "rejected_projection_groups": [list(key) for key in sorted(expected[run]) if key[0] not in COEFFICIENT_PARAMETERS],
                "attributed_projection_groups": [list(key) for key in sorted(covered[run]) if key[0] not in COEFFICIENT_PARAMETERS],
                "rejected_coefficient_groups": [list(key) for key in sorted(expected[run]) if key[0] in COEFFICIENT_PARAMETERS],
                "attributed_coefficient_groups": [list(key) for key in sorted(covered[run]) if key[0] in COEFFICIENT_PARAMETERS],
                "whole_parameter_and_forward_gates": "passed"}
        self.report.update(status="qualified", production_allowed=True, raw_statuses_preserved=True)
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-dir", type=Path, required=True)
    parser.add_argument("--initial-parity", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, nargs="+", required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-source", required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh decision output path; preserve earlier decisions")
    assessment = Assessment(args)
    try:
        assessment.assess()
    except Exception as error:
        assessment.report.update(status="failed", production_allowed=False, error=f"{type(error).__name__}: {error}")
    assessment.report["finished_at_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assessment.report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": assessment.report["status"], "output": str(args.output),
                      "error": assessment.report.get("error")}))
    return 0 if assessment.report["production_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
