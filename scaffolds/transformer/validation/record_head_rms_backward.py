"""Write runs/head-rms-backward/record.json from finished measure_head_rms_backward.sbatch jobs, and print the
README tables. Run from the checkout that submitted the jobs:

    python3 scaffolds/transformer/validation/record_head_rms_backward.py <job-id> [<job-id> ...] \
        [--checkpoints hashes.json] [--order <run-id>,...] [--print]

A job's results are logs/head-rms-backward-<job>-<list>.json with their logs beside them. Each run's record comes
from the first job listed that measured it; its results in later jobs are kept, compactly, as ``repeats`` (another
GPU, another compilation). ``hashes.json`` maps run IDs to {"sha256", "size_bytes", "file_count",
"matches_run_record"}: the checkpoint directory's hash in the run records' format. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "scaffolds/transformer/validation"
SCRIPT = "scaffolds/transformer/validation/measure_head_rms_backward.py"
LAUNCHER = "scaffolds/transformer/validation/measure_head_rms_backward.sbatch"
GROUPS = ("qk", "v", "attention_output", "head_rms_gain", "kernel_coefficients", "other", "all")
SCOPE = ("The gradient of the mean next-token NLL at run checkpoints, on GPU, with five head RMSNorms selected in one "
         "compiled program: each run's own source ('production'), a stable FP32 evaluation, a host FP64 one, and "
         "production's forward with the stable or the host FP64 backward; the whole model in FP64 on the first "
         "sequence; and the VJP of every committed head RMSNorm on synthetic FP32 rows against FP64. Each run keeps "
         "its recipe and precision with attention_backend blocked, on the first four held-out sequences of its eval "
         "manifest, one sequence per evaluation. Relative differences are ||left - right|| / ||right||.")
LIMITS = (
    "Measured on RTX A6000 and A100; the runs trained on H100, B200 and B300.",
    "The blocked reference kernels stand in for the Mosaic and fused kernels the runs trained on.",
    "One checkpoint per run and four sequences; the raw gradient, before clipping and the optimizer.",
    "Evaluations whose forwards differ in the last FP32 bit differ by the BF16 model's rounding noise "
    "(stable_vs_fp64_head); where forward_shared is false the production comparisons carry it.",
    "A gradient error is not a perplexity cost.",
)
COMPARISONS = ("production_vs_stable_backward", "production_vs_fp64_backward", "stable_backward_vs_fp64_backward",
               "production_vs_stable", "production_vs_fp64_head", "stable_vs_fp64_head")


def _artifact(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(VALIDATION)), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _rounded(value):
    """Seven significant digits, FP32's."""
    return float(f"{value:.7g}") if isinstance(value, float) else value


def _tidy(value, key=None):
    """The record's numbers at seven significant digits; cosines keep eight decimals (1 - cosine down to 1e-8)
    and the FP32 thresholds their full value."""
    if isinstance(value, dict):
        return {name: item if name == "thresholds" else _tidy(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_tidy(item, key) for item in value]
    if isinstance(value, float):
        return round(value, 8) if key == "cosine" else _rounded(value)
    return value


def _compact(metrics: dict | None, *, norms: bool = False) -> dict | None:
    """Relative L2 and cosine per group, with both norms on request; parameter counts are stated once per run."""
    if metrics is None:
        return None
    keys = ("relative_l2", "cosine", "norm_left", "norm_right") if norms else ("relative_l2", "cosine")
    return {group: None if metrics.get(group) is None else {key: metrics[group][key] for key in keys}
            for group in GROUPS}


def _run(result: dict, job: str, hashes: dict) -> dict:
    run = {key: result[key] for key in (
        "run_id", "checkpoint_step", "source_commit", "production_scaling_sha256", "recipe", "parameters",
        "variants", "loss", "nonfinite_gradient")}
    run["description"] = result["label"]
    # The run's own label, from its record in this checkout's study.
    study_record = ROOT / "experiments/transformer_pp/runs" / result["run_id"] / "record.json"
    run["label"] = json.loads(study_record.read_text())["label"] if study_record.exists() else None
    run["slurm_job_id"] = job
    run["checkpoint"] = hashes.get(result["run_id"])
    run["layers"] = [{key: layer[key] for key in (
        "layer", "rows", "fraction_subnormal", "fraction_vanishing", "fraction_mean_square_overflow",
        "rows_nonfinite", "heads_with_subnormal_rows_max", "largest_rms")} for layer in result["layers"]]
    run["gradients"] = {name: _compact(result["gradients"].get(name), norms=True) for name in COMPARISONS
                        if name in result["gradients"]}
    run["per_sequence"] = [{
        "sequence": entry["sequence"], "loss": entry["loss"], "largest_rms": entry["largest_rms"],
        **{name: {group: entry[name][group]["relative_l2"] for group in ("qk", "all")}
           for name in sorted(entry) if name.startswith("production_vs_")},
    } for entry in result["per_sequence"]]
    if "stable_backward" in result["variants"]:
        # Whether the compiled production evaluation and the two that take its forward agree in every loss: only
        # then is their gradient difference the backward's alone.
        run["forward_shared"] = all(
            entry["loss"]["production"] == entry["loss"]["stable_backward"] == entry["loss"].get(
                "fp64_backward", entry["loss"]["production"]) for entry in result["per_sequence"])
    if "direct_check" in result:
        check = result["direct_check"]
        run["direct_check"] = {"loss_equal": check["loss_equal"],
                               "gradient_bitwise_equal": check["gradient_bitwise_equal"],
                               "largest_absolute_difference": check["largest_absolute_difference"],
                               "relative_l2": check["all"]["relative_l2"]}
    return run


def _repeat(result: dict, job: str) -> dict:
    """A later job's result for a run already recorded: its losses, scales and gradient comparisons."""
    layers = result["layers"]
    rows = sum(layer["rows"] for layer in layers)
    repeat = {
        "run_id": result["run_id"], "slurm_job_id": job, "variants": result["variants"], "loss": result["loss"],
        "largest_rms": max(layer["largest_rms"] for layer in layers if not isinstance(layer["largest_rms"], str)),
        "fraction_subnormal": sum(layer["rows_subnormal"] for layer in layers) / rows,
        "fraction_vanishing": sum(layer["rows_vanishing"] for layer in layers) / rows,
        "gradients": {name: _compact(result["gradients"].get(name)) for name in COMPARISONS
                      if name in result["gradients"]},
    }
    if "direct_check" in result:
        check = result["direct_check"]
        repeat["direct_check"] = {"loss_equal": check["loss_equal"],
                                  "gradient_bitwise_equal": check["gradient_bitwise_equal"],
                                  "largest_absolute_difference": check["largest_absolute_difference"],
                                  "relative_l2": check["all"]["relative_l2"]}
    return repeat


def _probe_agreement(reference: dict, probe: dict) -> dict:
    """Largest relative difference between two probes' errors and radial shares over their common evaluations."""
    largest = {"relative_l2": 0.0, "radial_fraction": 0.0}
    for first, second in zip(reference["scales"], probe["scales"], strict=True):
        for name in set(reference["evaluations"]) & set(probe["evaluations"]):
            for key in largest:
                a, b = first[name][key], second[name][key]
                if isinstance(a, float) and isinstance(b, float) and a != b:
                    largest[key] = max(largest[key], abs(a - b) / max(abs(a), abs(b)))
                elif a != b:
                    largest[key] = float("inf")
    return largest


def _fp64_model(result: dict, job: str) -> dict:
    model = result["fp64_model"]
    if "loss" not in model:
        return {**model, "slurm_job_id": job}
    compact = {key: model[key] for key in ("sequence", "loss", "loss_by_evaluation", "largest_rms",
                                           "gradient_dtypes", "nonfinite_gradient")}
    compact["slurm_job_id"] = job
    for name in ("production", "stable", "stable_backward"):
        if f"{name}_vs_fp64_model" in model:
            compact[f"{name}_vs_fp64_model"] = _compact(model[f"{name}_vs_fp64_model"])
    # The two evaluations with a host FP64 head RMSNorm lie as close to the FP64 model as their FP32 counterparts.
    for name in ("fp64_head", "fp64_backward"):
        if f"{name}_vs_fp64_model" in model:
            compact[f"{name}_vs_fp64_model"] = {
                group: model[f"{name}_vs_fp64_model"][group]["relative_l2"] for group in ("qk", "all")}
    return compact


def _job(job: str) -> tuple[dict, list[dict]]:
    fields = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "-o",
                             "State,Elapsed,ExitCode,NodeList,Partition,Start,End,ReqTRES"],
                            capture_output=True, text=True, check=True).stdout.strip().splitlines()[0].split("|")
    state, elapsed, exit_code, node, partition, start, end, requested = fields
    hours, minutes, seconds = (int(value) for value in elapsed.split(":"))
    payloads = [json.loads(path.read_text())
                for path in sorted(VALIDATION.glob(f"logs/head-rms-backward-{job}-*.json"))]
    environment = payloads[0]["environment"]
    gpus = {}
    for payload in payloads:
        for line in payload["environment"]["gpu"]:
            name, uuid, bus, driver, capability = (field.strip() for field in line.split(","))
            gpus[uuid] = {"name": name, "uuid": uuid, "bus_id": bus, "driver_version": driver,
                          "compute_capability": capability}
    record = {
        "slurm_job_id": job, "slurm_state": state.split()[0], "exit_code": exit_code, "partition": partition,
        "node": node, "requested": requested, "started_at": start, "finished_at": end,
        "elapsed_seconds": 3600 * hours + 60 * minutes + seconds, "gpus": list(gpus.values()),
        "environment": f"scaffolds/transformer/.venv{'-cuda13' if environment['cuda_environment'] == 'cuda13' else ''}",
        "jax": environment["jax"], "source_commit": environment["source_commit"],
        "script_sha256": environment["script_sha256"],
        "host_callback_problem": payloads[0]["host_callback_problem"],
        "run_lists": [[run["run_id"] for run in payload["runs"]] for payload in payloads],
        "artifacts": [_artifact(path) for path in sorted(VALIDATION.glob(f"logs/head-rms-backward-{job}*"))
                      if path.suffix in (".json", ".log")],
    }
    return record, payloads


def build(jobs: list[str], hashes: dict, order: list[str]) -> dict:
    record = {
        "format": "head-rms-norm-backward-measurement", "schema_version": 1, "date": "2026-11-20",
        "home_cluster": "kennel", "scope": SCOPE, "limits": list(LIMITS), "script": SCRIPT, "launcher": LAUNCHER,
        "script_sha256": hashlib.sha256((ROOT / SCRIPT).read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256((ROOT / LAUNCHER).read_bytes()).hexdigest(),
        "checkpoint_hash_format": "sha256(relative-path NUL file-sha256 NEWLINE), sorted by relative path",
        "jobs": [], "runs": [], "repeats": [], "unmeasured": [],
    }
    runs: dict[str, dict] = {}
    problems: dict[str, str] = {}
    agreements: dict[tuple, dict] = {}
    for job in jobs:
        description, payloads = _job(job)
        record["jobs"].append(description)
        for payload in payloads:
            for key in ("thresholds", "groups"):
                record.setdefault(key, {}).update(payload[key])
            for label, scaling in payload["production_scalings"].items():
                known = record.setdefault("production_scalings", {}).setdefault(
                    label, {"sha256": scaling["sha256"], "source_commits": []})
                known["source_commits"] = sorted({*known["source_commits"], *scaling["source_commits"]})
            if "backward_probe" in payload:
                recorded = record.setdefault("backward_probe", {"slurm_job_id": job, **payload["backward_probe"]})
                agreement = _probe_agreement(recorded, payload["backward_probe"])
                # A process probes the committed versions of its own runs: the record takes the union.
                for name in payload["backward_probe"]["evaluations"]:
                    if name not in recorded["evaluations"]:
                        recorded["evaluations"] = sorted([*recorded["evaluations"], name])
                        for kept, entry in zip(recorded["scales"], payload["backward_probe"]["scales"], strict=True):
                            kept[name] = entry[name]
                gpu = payload["environment"]["gpu"][0].split(",")
                key = (job, gpu[1].strip())
                if key not in agreements or any(agreement[name] > agreements[key][name] for name in agreement):
                    agreements[key] = {"slurm_job_id": job, "gpu": gpu[0].strip(), "uuid": gpu[1].strip(), **{
                        name: max(agreement[name], agreements.get(key, agreement)[name]) for name in agreement}}
            for result in payload["runs"]:
                identifier = result["run_id"]
                if "sequence_sha256" in result:
                    # Every run reads the same held-out sequences when its eval manifest is the same.
                    data = {"eval_manifest": result["eval_manifest"], "sequences": result["sequences"],
                            "tokens_per_sequence": result["tokens_per_sequence"],
                            "sequence_sha256": result["sequence_sha256"]}
                    if record.setdefault("data", data) != data:
                        raise ValueError(f"{identifier} was measured on other sequences than the earlier runs")
                if "gradients" not in result:
                    problems.setdefault(identifier, result.get("error") or result.get("skipped"))
                    continue
                if identifier not in runs:
                    runs[identifier] = _run(result, job, hashes)
                elif runs[identifier]["slurm_job_id"] != job:
                    record["repeats"].append(_repeat(result, job))
                recorded = runs[identifier].get("fp64_model")
                if "fp64_model" in result and (recorded is None or "loss" not in recorded):
                    runs[identifier]["fp64_model"] = _fp64_model(result, job)
    # Every process probed its GPU: the largest relative difference of each from the recorded probe.
    record["backward_probe_agreement"] = list(agreements.values())
    record["runs"] = sorted(runs.values(), key=lambda run: (
        order.index(run["run_id"]) if run["run_id"] in order else len(order)))
    record["repeats"].sort(key=lambda repeat: (
        order.index(repeat["run_id"]) if repeat["run_id"] in order else len(order)))
    record["unmeasured"] = [{"run_id": identifier, "reason": reason} for identifier, reason in problems.items()
                            if identifier not in runs]
    return record


def _short(value, digits=2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return {"Infinity": "inf", "NaN": "nan"}.get(value, value)
    if value == 0:
        return "0"
    return f"{value:.{digits}g}" if abs(value) < 1e-3 or abs(value) >= 1e4 else f"{value:.{digits + 1}g}"


def _percent(value) -> str:
    """A fraction as a percentage with two significant digits (three from 10%)."""
    if value is None or isinstance(value, str):
        return _short(value)
    if value >= 10:
        return f"{round(100 * value, -1):,.0f}%"
    return "0" if value == 0 else f"{100 * value:.{3 if value >= 0.1 else 2}g}%"


def _relative(metrics: dict | None, group: str) -> str:
    return "n/a" if not metrics or metrics.get(group) is None else _percent(metrics[group]["relative_l2"])


def tables(record: dict) -> str:
    """Two tables: each run's output scales and losses, then its gradient differences by parameter group."""
    scales = ["| Run | Update | Largest output RMS | Rows over 1.9e25 | Rows over 8e29 | Layers with rows over 8e29 "
              "| Rows whose FP32 mean square overflows | NLL production / stable |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    errors = ["| Run | Forward shared | Q/K | V | Output projections | Coefficients | Other | All | Cosine Q/K / all "
              "| Stable against FP64 backward: Q/K / all | Production against stable: Q/K / all "
              "| Stable against FP64 head: Q/K / all |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in record["runs"]:
        layers = run["layers"]
        rows = sum(layer["rows"] for layer in layers)
        fraction = {key: sum(layer[key] * layer["rows"] for layer in layers) / rows
                    for key in ("fraction_subnormal", "fraction_vanishing", "fraction_mean_square_overflow")}
        vanishing = [layer for layer in layers if layer["fraction_vanishing"] > 0]
        worst = max(layers, key=lambda layer: layer["fraction_vanishing"])
        largest = max(float(layer["largest_rms"]) if not isinstance(layer["largest_rms"], str) else float("inf")
                      for layer in layers)
        name = f"`{run['run_id']}` {run['description']}"
        scales.append(
            f"| {name} | {run['checkpoint_step']:,} | {_short(largest)} | {_percent(fraction['fraction_subnormal'])} "
            f"| {_percent(fraction['fraction_vanishing'])} | "
            + (f"{len(vanishing)} of {len(layers)}, at most {_percent(worst['fraction_vanishing'])} (layer "
               f"{worst['layer']})" if vanishing else "none")
            + f" | {_percent(fraction['fraction_mean_square_overflow'])} "
            f"| {run['loss']['production']:.4f} / {run['loss']['stable']:.4f} |")
        gradients = run["gradients"]
        same = gradients.get("production_vs_stable_backward")
        errors.append(
            f"| {name} | {'yes' if run.get('forward_shared') else 'no'} | "
            + " | ".join(_relative(same, group) for group in (
                "qk", "v", "attention_output", "kernel_coefficients", "other", "all"))
            + (f" | {same['qk']['cosine']:.5f} / {same['all']['cosine']:.5f}" if same else " | n/a")
            + "".join(f" | {_relative(gradients.get(pair), 'qk')} / {_relative(gradients.get(pair), 'all')}"
                      for pair in ("stable_backward_vs_fp64_backward", "production_vs_stable", "stable_vs_fp64_head"))
            + " |")
    models = ["| Run | NLL: FP64 model / production | Production: Q/K / all | Stable: Q/K / all "
              "| Stable backward: Q/K / all |",
              "| --- | --- | --- | --- | --- |"]
    for run in record["runs"]:
        model = run.get("fp64_model") or {}
        if "loss" not in model:
            continue
        models.append(
            f"| `{run['run_id']}` | {model['loss']:.4f} / {model['loss_by_evaluation']['production']:.4f}"
            + "".join(f" | {_relative(model.get(f'{name}_vs_fp64_model'), 'qk')} / "
                      f"{_relative(model.get(f'{name}_vs_fp64_model'), 'all')}"
                      for name in ("production", "stable", "stable_backward")) + " |")
    return "\n\n".join("\n".join(table) for table in (scales, errors, models))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", nargs="+")
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--order", default="", help="run IDs, comma separated, in the order to record and print them")
    parser.add_argument("--print", action="store_true", help="print the tables without writing the record")
    arguments = parser.parse_args()
    hashes = json.loads(arguments.checkpoints.read_text()) if arguments.checkpoints else {}
    record = build(arguments.jobs, hashes, [name for name in arguments.order.split(",") if name])
    print(tables(record))
    if not arguments.print:
        path = VALIDATION / "runs/head-rms-backward/record.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_tidy(record), indent=1, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
