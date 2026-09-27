"""Refresh the canonical execution manifest and regenerate the factored Hermite results table."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from scaffolds.launch.frozen import frozen_hash
from scaffolds.transformer.analysis import committed_records, terminal_row, update_results_readme, write_csv
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count

RECORD = Path(__file__).resolve().parents[2]
RUNS = ("200m-normalized-factored-m002-seed0",)
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
        qualification = read_json(directory / "qualification_failure.json")
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
        elif qualification:
            if metadata:
                raise ValueError(f"Qualification failure conflicts with production metadata for {name}")
            row.update(status="acceptance-numerical-failure", failure_step=qualification["failure"]["step"],
                       failure_error=qualification["failure"]["error"],
                       source_commit=qualification["source_commit"], slurm_job_id=qualification["slurm_job_id"])
        elif not metadata and manifest_runs.get(name, {}).get("status") == "acceptance-numerical-failure":
            # The pilot is a qualification result. Never report its metrics as this run's endpoint.
            row["status"] = "acceptance-numerical-failure"
        elif not metadata and manifest_runs.get(name, {}).get("slurm_job_ids"):
            row["status"] = "submitted"
        result.append({key: row[key] for key in FIELDS})
    return result


def concluded_job(job_id: str) -> dict:
    """Require terminal Slurm evidence and no live/requeueable allocation before retention."""
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Qualification failure requires one numeric Slurm job ID")
    queued = subprocess.run(["squeue", "--noheader", "--jobs", job_id, "--format=%i"],
                            text=True, capture_output=True)
    if queued.stdout.strip():
        raise ValueError(f"Job {job_id} is still live in Slurm; do not conclude its qualification")
    if queued.returncode and "Invalid job id" not in queued.stderr:
        raise RuntimeError(f"Cannot verify Slurm queue state: {queued.stderr.strip()}")
    accounting = subprocess.check_output(
        ["sacct", "--noheader", "--parsable2", "--jobs", job_id,
         "--format=JobIDRaw,State,ExitCode,End"], text=True)
    records = [line.split("|") for line in accounting.splitlines() if line.split("|")[0] == job_id]
    if not records:
        raise ValueError(f"No Slurm accounting record proves that job {job_id} concluded")
    fields = records[-1]
    state = fields[1].split(" ", 1)[0].rstrip("+")
    if state not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}:
        raise ValueError(f"Slurm state {fields[1]!r} does not prove job {job_id} concluded")
    return dict(slurm_job_id=job_id, state=fields[1], exit_code=fields[2], ended=fields[3])


def retain_qualification_failure(record: Path, name: str, job_id: str) -> Path:
    """Create one scientific failure record from a concluded pilot, without copying preflight files."""
    if name not in RUNS:
        raise ValueError(f"Unknown factored Hermite run: {name}")
    scheduler = concluded_job(job_id)
    root = record.parent.parent
    directory = record / "runs" / name
    if (directory / "metadata.json").exists() or (directory / "completion.json").exists():
        raise ValueError("Production has its own results; a pilot cannot replace them")
    logs = record / "logs" / name
    candidates = []
    for acceptance in sorted((logs / "acceptance").glob("*")):
        failure_file = failure_path(acceptance / "pilot")
        metadata = read_json(acceptance / "pilot/metadata.json")
        gate = read_json(acceptance / "failure.json")
        if (failure_file and str(metadata.get("slurm_job_id")) == job_id
                and str(gate.get("slurm_job_id")) == job_id):
            candidates.append((acceptance, failure_file, metadata, gate))
    if len(candidates) != 1:
        raise ValueError(f"Expected one numerical pilot failure for {name}, job {job_id}; found {len(candidates)}")
    acceptance, failure_file, metadata, gate = candidates[0]
    failure = read_json(failure_file)
    identity = read_json(acceptance / "identity.json")
    config_path = acceptance / "pilot/config.json"
    config = read_json(config_path)
    if (failure.get("format") != "shared-transformer-numerical-failure" or not config
            or gate.get("stage") not in {"pilot", "resume"}
            or identity.get("source_commit") != metadata.get("source_commit")
            or config["train"].get("source_commit") != metadata.get("source_commit")):
        raise ValueError("Pilot failure, executed configuration, and source identity do not agree")
    log_path = logs / f"slurm-{job_id}.out"
    if not log_path.exists():
        raise ValueError("The matching Slurm log is required to retain a concluded numerical failure")
    validations = committed_records(acceptance / "pilot")
    # Training scalars are printed rather than committed before the pilot's first
    # validation. Preserve the last finite printed measurement with its exact log hash.
    training = []
    for step, nll, perplexity in re.findall(r"^step=(\d+) train_nll=([^\s]+) train_ppl=([^\s]+)",
                                           log_path.read_text(), flags=re.MULTILINE):
        try:
            values = float(nll), float(perplexity)
        except ValueError:
            continue
        if int(step) < failure["step"] and all(math.isfinite(value) for value in values):
            training.append(dict(step=int(step), train_nll=values[0], train_perplexity=values[1],
                                 precision="as printed in the matching Slurm log"))
    evidence_paths = [failure_file, config_path, acceptance / "pilot/metadata.json",
                      acceptance / "identity.json", acceptance / "failure.json", log_path]
    if (acceptance / "pilot/metrics.jsonl").exists():
        evidence_paths.append(acceptance / "pilot/metrics.jsonl")
    retained = {
        "format": "hermite-factored-m2-qualification-numerical-failure", "version": 1, "run": name,
        "status": "acceptance-numerical-failure", "scope": "full-recipe acceptance pilot",
        "source_commit": metadata["source_commit"], "slurm_job_id": job_id, "home_cluster": "kennel",
        "scheduler": scheduler, "executed_pilot_configuration": config,
        "production_started": False, "failure": failure,
        "last_valid_training_measurement": training[-1] if training else None,
        "last_valid_validation": validations[-1] if validations else None,
        "evidence": [artifact(path, root) for path in evidence_paths],
        "limitations": ["Pilot evaluation uses one held-out batch and a shorter checkpoint/evaluation interval.",
                        "The failure is not a completed production training run or terminal perplexity.",
                        "Printed training scalars have limited precision; absent measurements remain null."],
    }
    output = directory / "qualification_failure.json"
    if output.exists() and read_json(output) != retained:
        raise ValueError(f"Refusing to replace different concluded qualification evidence: {output}")
    directory.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(retained, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


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
    manifest.update(experiment="hermite_factored_m2", home_cluster="kennel", artifact_root=str(root),
                    documentation="experiments/hermite_factored_m2/README.md", scaffold="scaffolds/transformer")
    manifest.setdefault("branch", "codex/hermite-factored-m2")
    manifest.setdefault("branch_base_commit", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    manifest.setdefault("requested", {"date": "2026-09-27", "from_cluster": "kennel", "degrees": [2],
                                      "normalization": [True], "factorization": "square", "seeds": [0]})
    manifest["scientific_change"] = "Evaluate fixed normalized degree-two Hermite p_2(s)=exp(1/2)*(s+1)^2/2 as a square of a linear factor"
    manifest["coefficient_convention"] = "Fixed square-factor coefficients [a, a, 0], a=sqrt(exp(1/2)/2), in ascending order"
    manifest["comparison_limitations"] = [
        "The original expanded degree-two run failed the first acceptance pilot step; it has no production endpoint.",
        "A successful two-step pilot does not establish stability over the full 10B-token training budget.",
        "A single seed and GPU type do not establish performance or stability across configurations.",
    ]
    manifest["shared_dependency_lock"] = artifact(root / "scaffolds/transformer/uv.lock", root)
    if launch_commit:
        if manifest.get("launch_commit") not in (None, launch_commit):
            raise ValueError("Cannot overwrite a different launch commit")
        manifest["launch_commit"] = launch_commit
    source = manifest.get("launch_commit")
    if source:
        manifest["frozen_execution_set_sha256"] = frozen_hash(root, source, "hermite_factored_m2", "transformer")
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
                     fixed_square_factors=list(config.model.attention_fixed_square_factors), seed=config.train.seed,
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
        qualification = read_json(directory / "qualification_failure.json")
        if qualification:
            entry["status"] = "acceptance-numerical-failure"
            entry["qualification_failure"] = artifact(directory / "qualification_failure.json", root)
        elif not metadata and any(item["status"] == "numerical-failure" for item in entry["acceptance"].values()):
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
    manifest["status"] = "concluded" if all(
        entry["status"] in ("complete", "numerical-failure") or bool(entry.get("qualification_failure"))
        for entry in run_entries) else "active"
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
    parser.add_argument("--retain-qualification-failure", choices=RUNS, metavar="RUN")
    parser.add_argument("--failure-job-id", help="Concluded Slurm job whose pilot numerical failure is retained")
    args = parser.parse_args()
    if bool(args.retain_qualification_failure) != bool(args.failure_job_id):
        parser.error("--retain-qualification-failure and --failure-job-id are required together")
    if args.retain_qualification_failure:
        retain_qualification_failure(args.record, args.retain_qualification_failure, args.failure_job_id)
        args.refresh_provenance = True
    if args.refresh_provenance:
        refresh_provenance(args.record, launch_commit=args.launch_commit, hash_checkpoints=args.hash_checkpoints)
    elif args.launch_commit or args.hash_checkpoints:
        parser.error("--launch-commit and --hash-checkpoints require --refresh-provenance")
    regenerate(args.record)


if __name__ == "__main__":
    main()
