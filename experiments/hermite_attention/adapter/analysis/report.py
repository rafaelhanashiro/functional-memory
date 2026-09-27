"""Refresh the canonical execution manifest and regenerate the Hermite results table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scaffolds.launch.frozen import frozen_hash
from scaffolds.transformer.analysis import committed_records, terminal_row, update_results_readme, write_csv
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count

RECORD = Path(__file__).resolve().parents[2]
RUNS = tuple(f"200m-{variant}-m{degree:03d}-seed0" for variant in ("normalized", "unnormalized")
             for degree in (1, 2, 3, 4))
FIELDS = ["run", "attention_degree", "normalization", "status", "updates", "prediction_tokens",
          "terminal_validation_nll", "terminal_validation_perplexity", "best_step", "best_validation_perplexity",
          "last_valid_step", "last_valid_perplexity", "failure_step", "failure_error", "source_commit", "slurm_job_id"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def failure_path(directory: Path) -> Path | None:
    files = sorted(directory.glob("numerical_failure_step_*.json"))
    return files[-1] if files else None


def rows(record: Path = RECORD) -> list[dict]:
    """Scientific endpoints come only from production runs; pilot failures remain distinct."""
    manifest = read_json(record / "provenance/manifest.json")
    manifest_runs = {run["name"]: run for run in manifest.get("runs", [])}
    result = []
    for name in RUNS:
        directory = record / "runs" / name
        metadata = read_json(directory / "metadata.json")
        records = committed_records(directory)
        failure_file = failure_path(directory)
        failure = read_json(failure_file) if failure_file else {}
        config = read_json(directory / "config.json")
        row = dict.fromkeys(FIELDS, "")
        row.update(run=name, attention_degree=int(name.split("-m")[1][:3]), normalization=name.split("-")[1],
                   status="incomplete" if metadata else "not-started", source_commit=metadata.get("source_commit", ""),
                   slurm_job_id=metadata.get("slurm_job_id", ""), updates=metadata.get("step", ""),
                   prediction_tokens=metadata.get("training_tokens", ""))
        if records:
            last = records[-1]
            best = min(records, key=lambda item: (item["validation_nll"], -item["step"]))
            row.update(last_valid_step=last["step"], last_valid_perplexity=last["validation_perplexity"],
                       best_step=best["step"], best_validation_perplexity=best["validation_perplexity"])
        if (directory / "completion.json").exists():
            completion = read_json(directory / "completion.json")
            if failure or not records or completion["final_step"] != config["optim"]["total_steps"]:
                raise ValueError(f"Conflicting or incomplete terminal evidence for {name}")
            terminal = records[-1]
            if (terminal["step"] != completion["final_step"]
                    or terminal["validation_perplexity"] != completion["validation_perplexity"]
                    or terminal["source_commit"] != completion["source_commit"]):
                raise ValueError(f"Terminal validation does not match completion for {name}")
            row.update(terminal_row(directory), status="complete")
        elif failure:
            row.update(status="numerical-failure", failure_step=failure["step"], failure_error=failure["error"],
                       updates=failure["step"], prediction_tokens=failure.get("training_tokens", ""))
        elif not metadata and manifest_runs.get(name, {}).get("status") == "acceptance-numerical-failure":
            # The pilot is a qualification result. Never report its metrics as this run's endpoint.
            row["status"] = "acceptance-numerical-failure"
        result.append({key: row[key] for key in FIELDS})
    return result


def _acceptance(logs: Path, root: Path) -> dict:
    result = {}
    for directory in sorted((logs / "acceptance").glob("*")):
        if not directory.is_dir():
            continue
        success = read_json(directory / "success.json")
        failure = read_json(directory / "failure.json")
        identity = read_json(directory / "identity.json")
        entry = {"status": "passed" if success else "failed" if failure else "incomplete",
                 "identity": identity.get("identity"), "runtime": identity.get("runtime"),
                 "hardware": read_json(directory / "hardware.json"),
                 "passed_stages": success.get("checks", [])}
        for stage in ("cpu-tests", "cpu-distributed-tests", "gpu-tests"):
            tests = read_json(directory / f"{stage}.json")
            if tests:
                entry[stage] = {key: tests[key] for key in ("tests_passed", "note", "status") if key in tests}
        initialization = read_json(directory / "initialization.json")
        if initialization:
            entry["initialization"] = initialization.get("result")
        for stage in ("pilot", "resume"):
            stage_record = read_json(directory / f"{stage}.json")
            outcome = stage_record.get("outcome", {})
            if stage_record:
                entry[stage] = {key: outcome[key] for key in ("final_step", "restored_step", "data_start_sequence",
                    "attention_backend", "checkpoint_audit", "timing") if key in outcome}
                if stage == "pilot":
                    entry[stage]["projected_hours"] = stage_record.get("projected_hours")
                    entry[stage]["validation_nll"] = stage_record.get("validation", {}).get("validation_nll")
        if failure:
            entry["failure"] = {**artifact(directory / "failure.json", root), **failure}
        pilot_failure = failure_path(directory / "pilot")
        if pilot_failure:
            entry["status"] = "numerical-failure"
            evidence = read_json(pilot_failure)
            entry["numerical_failure"] = {**artifact(pilot_failure, root), "step": evidence["step"],
                                          "error": evidence["error"], "scope": "acceptance pilot"}
        result[directory.name] = entry
    return result


def refresh_provenance(record: Path = RECORD, *, launch_commit: str | None = None,
                       hash_checkpoints: bool = False) -> dict:
    """Consolidate raw outputs without copying submission or preflight files into the record."""
    root = record.parent.parent
    path = record / "provenance/manifest.json"
    manifest = read_json(path)
    manifest.setdefault("format", "functional-memory-experiment-setup")
    manifest.setdefault("version", 2)
    manifest.update(experiment="hermite_attention", home_cluster="kennel", artifact_root=str(root),
                    documentation="experiments/hermite_attention/README.md", scaffold="scaffolds/transformer")
    manifest.setdefault("branch", "codex/hermite-attention")
    manifest.setdefault("branch_base_commit", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    manifest.setdefault("requested", {"date": "2026-09-27", "from_cluster": "kennel", "degrees": [1, 2, 3, 4],
                                      "normalization": [True, False], "seeds": [0]})
    manifest["scientific_change"] = "Fixed exp(1/2) sum_(r<=m) He_r(s)/r!, raw scores s=q.k/sqrt(d_K), normalized and unnormalized"
    manifest["coefficient_convention"] = "Probabilists' Hermite projection under N(0,1), full exp(1/2) prefactor in all recipes"
    manifest["comparison_limitations"] = [
        "Legacy fixed Taylor controls use raw scores but decay all parameters; Hermite uses current selective decay.",
        "The ongoing normalized Taylor rerun uses selective decay and QK-norm; it is not a matched raw-score control.",
        "One seed per variant; a coefficient-only empirical comparison needs a matched fixed Taylor control.",
    ]
    manifest["shared_dependency_lock"] = artifact(root / "scaffolds/transformer/uv.lock", root)
    if launch_commit:
        if manifest.get("launch_commit") not in (None, launch_commit):
            raise ValueError("Cannot overwrite a different launch commit")
        manifest["launch_commit"] = launch_commit
    source = manifest.get("launch_commit")
    if source:
        manifest["frozen_execution_set_sha256"] = frozen_hash(root, source, "hermite_attention", "transformer")
    previous = {run["name"]: run for run in manifest.get("runs", [])}
    current_rows = {row["run"]: row for row in rows(record)}
    run_entries = []
    for name in RUNS:
        entry = previous.get(name, {"name": name, "slurm_job_ids": []})
        config_path = record / "configs" / f"{name}.yaml"
        config = load_config(config_path)
        resolved = config_to_dict(config)
        if source:
            resolved["train"]["source_commit"] = source
        directory = record / "runs" / name
        actual_config = read_json(directory / "config.json")
        metadata = read_json(directory / "metadata.json")
        logs = record / "logs" / name
        entry.update(attention_degree=config.model.attention_degree, attention_normalization=config.model.attention_normalization,
                     fixed_coefficients=list(config.model.attention_fixed_coefficients), seed=config.train.seed,
                     home_cluster="kennel", config=artifact(config_path, root),
                     resolved_configuration=actual_config or resolved,
                     resolved_configuration_origin="trainer output" if actual_config else "prepared recipe",
                     parameter_count=model_parameter_count(config.model),
                     source_commit=metadata.get("source_commit") or source,
                     run_record=str(directory.relative_to(root)), target_updates=config.optim.total_steps,
                     target_prediction_tokens=config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1))
        encoded = json.dumps(entry["resolved_configuration"], sort_keys=True, separators=(",", ":")).encode()
        entry["resolved_configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
        entry["resolved_configuration_hash_format"] = "UTF-8 JSON, sorted keys, compact separators"
        entry["status"] = current_rows[name]["status"]
        attempts = []
        for attempt_path in sorted((logs / "submissions").glob("*.json")):
            raw = read_json(attempt_path)
            allocation = dict(re.findall(r"(?:^|\s)([A-Za-z_]+)=([^\s]*)", raw.get("allocation", "")))
            attempt = {key: raw.get(key) for key in ("slurm_job_id", "slurm_restart_count", "slurm_partition",
                       "started_utc", "gpu_type", "gpu_count", "launch_commit")}
            attempt.update(node=allocation.get("NodeList"), time_limit=allocation.get("TimeLimit"))
            attempts.append(attempt)
        execution = entry.setdefault("execution", {})
        if attempts:
            execution["attempts"] = attempts
        if metadata:
            execution.update(runtime=metadata.get("execution_runtime"), runtime_history=metadata.get("runtime_history"),
                             attention_backend=metadata.get("attention_backend"),
                             attention_backend_requested=metadata.get("attention_backend_requested"))
            entry["data_identity"] = metadata.get("data_identity") or read_json(directory / "data_identity.json")
        entry["slurm_job_ids"] = sorted(set(map(str, entry.get("slurm_job_ids", []))) |
                                       {str(attempt["slurm_job_id"]) for attempt in attempts})
        entry["acceptance"] = _acceptance(logs, root)
        if not metadata and any(item["status"] == "numerical-failure" for item in entry["acceptance"].values()):
            entry["status"] = "acceptance-numerical-failure"
        elif not metadata and entry["slurm_job_ids"]:
            entry["status"] = "submitted"
        entry["artifacts"] = [artifact(item, root) for item in sorted(directory.glob("*")) if item.is_file()]
        entry["logs"] = [artifact(item, root) for item in sorted(logs.glob("slurm-*.out"))]
        wandb = read_json(directory / "wandb_run.json")
        if wandb:
            entry["wandb"] = wandb
        best = read_json(directory / "best_validation.json")
        if best:
            checkpoint = record / "checkpoints" / name / str(best["step"])
            selected = {**best, "path": str(checkpoint.relative_to(root)), "present": checkpoint.exists()}
            if hash_checkpoints and checkpoint.exists():
                files = {str(item.relative_to(checkpoint)): sha256(item)
                         for item in sorted(checkpoint.rglob("*")) if item.is_file()}
                selected.update(files=files, tree_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
            elif entry.get("selected_checkpoint", {}).get("step") == best["step"]:
                for key in ("files", "tree_sha256"):
                    if key in entry["selected_checkpoint"]:
                        selected[key] = entry["selected_checkpoint"][key]
            entry["selected_checkpoint"] = selected
        run_entries.append(entry)
    manifest["runs"] = run_entries
    first = load_config(record / "configs" / f"{RUNS[0]}.yaml")
    manifest["data_identity"] = {split: {"path": location, "sha256": sha256(Path(location))}
                                 for split, location in (("train", first.data.train_manifest), ("eval", first.data.eval_manifest))}
    manifest["status"] = "concluded" if all(entry["status"] in ("complete", "numerical-failure") for entry in run_entries) else "active"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def regenerate(record: Path = RECORD, output: Path | None = None) -> None:
    output = output or record
    output.mkdir(parents=True, exist_ok=True)
    table = rows(record)
    write_csv(output / "summary.csv", table, FIELDS)
    lines = ["| Variant | Degree | Status | Last update | Terminal PPL | Failure update |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in table:
        perplexity = row["terminal_validation_perplexity"]
        lines.append(f"| {row['normalization']} | {row['attention_degree']} | {row['status']} | {row['updates']} | "
                     f"{perplexity if perplexity == '' else f'{perplexity:.4f}'} | {row['failure_step']} |")
    update_results_readme(output / "README.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--refresh-provenance", action="store_true")
    parser.add_argument("--launch-commit")
    parser.add_argument("--hash-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.refresh_provenance:
        refresh_provenance(args.record, launch_commit=args.launch_commit, hash_checkpoints=args.hash_checkpoints)
    elif args.launch_commit or args.hash_checkpoints:
        parser.error("--launch-commit and --hash-checkpoints require --refresh-provenance")
    regenerate(args.record)


if __name__ == "__main__":
    main()
