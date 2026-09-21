"""Retain one validation-best checkpoint and the final successful run log."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.key_dimension_taylor.plot import TERMINAL_STEP, summarize_run
from experiments.polynomial_attention.training.metrics import write_json_atomic


def finalize_run(manifest: dict, name: str) -> dict:
    from experiments.key_dimension_taylor.sweep import active_runs

    run = next((run for run in active_runs(manifest) if run["name"] == name), None)
    if run is None:
        raise ValueError(f"Retention may only modify a new joint-sweep run: {name}")
    project_root = Path(manifest["project_root"]).resolve()
    workdir = Path(run["workdir"]).resolve()
    if workdir != project_root / "results" / "key_dimension_taylor" / name:
        raise ValueError("Run results are not in canonical retained storage")
    row, records = summarize_run(run)
    if row["status"] != "complete" or row["terminal_step"] != TERMINAL_STEP:
        raise ValueError("Cannot retain an incomplete training run")
    completion_path = workdir / "completion.json"
    completion = json.loads(completion_path.read_text())
    best = json.loads((workdir / "best_validation.json").read_text())
    selected = min(records, key=lambda record: (record["validation_nll"], -record["step"]))
    if (completion["source_commit"] != manifest["source_commit"]
            or completion["retained_best_checkpoint_step"] != selected["step"]
            or best["step"] != selected["step"]
            or best["validation_nll"] != selected["validation_nll"]):
        raise ValueError("Retained checkpoint selection does not match the best committed validation")
    expected = project_root / "experiments" / "key_dimension_taylor" / "checkpoints" / name
    if (Path(run.get("checkpoint_dir", expected)).resolve() != expected
            or (workdir / "checkpoints").resolve() != expected
            or not (expected / str(selected["step"])).is_dir()):
        raise ValueError("Validation-best checkpoint is missing from canonical storage")
    job_id = str(completion.get("slurm_job_id", ""))
    if not job_id.isdigit():
        raise ValueError("Completed production run has no valid Slurm job ID")
    log_dir = project_root / "results" / "logs" / "key_dimension_taylor" / name
    selected_log = log_dir / f"slurm-{job_id}.out"
    if not selected_log.is_file():
        raise ValueError("Completed production run's matching Slurm log is missing")
    retention_path = workdir / "retention.json"
    previous = json.loads(retention_path.read_text()) if retention_path.exists() else {}
    completion_hash = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    if previous and previous.get("completion_sha256") != completion_hash:
        raise ValueError("Completed run changed after retention")
    log_history = {entry["path"]: entry for entry in previous.get("log_history", [])}
    for log in sorted(log_dir.glob("slurm-*.out")):
        if log == selected_log:
            continue
        log_history[str(log)] = {
            "path": str(log), "slurm_job_id": log.stem.removeprefix("slurm-"),
            "sha256": hashlib.sha256(log.read_bytes()).hexdigest(), "retained": False,
        }
    # The launcher may still append to the selected log, so its final checksum
    # must be captured after Slurm has closed it, not during this command.
    log_history[str(selected_log)] = {"path": str(selected_log), "slurm_job_id": job_id, "retained": True}
    launch_path = workdir / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
    record = {
        "format": "key-dimension-taylor-checkpoint-retention", "version": 1,
        "source_commit": manifest["source_commit"], "seed": run["config"]["train"]["seed"],
        "completion_sha256": completion_hash,
        "selection_rule": "lowest_validation_nll_latest_on_tie",
        "best_checkpoint_step": selected["step"], "best_checkpoint_path": str(expected / str(selected["step"])),
        "validation_nll": selected["validation_nll"], "validation_perplexity": selected["validation_perplexity"],
        "terminal_step": completion["final_step"], "terminal_metrics_preserved": True,
        "terminal_nll": row["terminal_nll"], "terminal_perplexity": row["terminal_ppl"],
        "slurm_job_id": job_id, "log_path": str(selected_log),
        "log_history": list(log_history.values()), "launch_history": launch or previous.get("launch_history", {}),
        "segment_job_ids": sorted({str(event["slurm_job_id"]) for event in records if event.get("slurm_job_id")}),
    }
    # Persist the recovery record before pruning. A subsequent completed-run
    # invocation can recognize completion even when best predates terminal.
    write_json_atomic(retention_path, record)
    for path in expected.iterdir():
        if path.is_dir() and ((path.name.isdigit() and int(path.name) != selected["step"])
                              or path.name.endswith(".orbax-checkpoint-tmp")):
            shutil.rmtree(path)
    for path in log_dir.glob("slurm-*.out"):
        if path != selected_log:
            path.unlink()
    metadata_path = workdir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(checkpoint_retention=record, slurm_job_id=job_id, log_path=str(selected_log),
                    retained_best_checkpoint_step=selected["step"],
                    retained_terminal_checkpoint_step=(selected["step"] if selected["step"] == TERMINAL_STEP else None))
    write_json_atomic(metadata_path, metadata)
    report_path = workdir / "REPORT.md"
    if report_path.exists():
        lines = [line for line in report_path.read_text().splitlines()
                 if not line.startswith("Retained terminal checkpoint:")
                 and not line.startswith("Retained validation-best checkpoint:")]
        lines += [f"Retained validation-best checkpoint: `{selected['step']}`. Terminal metrics remain in `metrics.jsonl`."]
        report_path.write_text("\n".join(lines) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    print(json.dumps(finalize_run(json.loads(args.manifest.read_text()), args.run_name), sort_keys=True))


if __name__ == "__main__":
    main()
