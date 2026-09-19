"""Keep the validation-best checkpoint after an odd-Taylor run completes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.polynomial_attention.training.metrics import write_json_atomic


def finalize_run(manifest: dict, name: str) -> dict:
    run = next(run for run in manifest["runs"] if run["name"] == name)
    workdir = Path(run["workdir"])
    completion_path = workdir / "completion.json"
    completion = json.loads(completion_path.read_text())
    best = json.loads((workdir / "best_validation.json").read_text())
    config = run["config"]
    if (completion["source_commit"] != manifest["source_commit"]
            or completion["final_step"] != config["optim"]["total_steps"]
            or completion["training_tokens"] != run["training_tokens"]
            or completion["retained_best_checkpoint_step"] != best["step"]
            or any(completion.get(key) != config["model"][key]
                   for key in ("attention_degree",))):
        raise ValueError("Cannot finalize an incomplete or mismatched odd-Taylor run")
    checkpoint_root = (workdir / "checkpoints").resolve()
    expected = (Path(manifest["project_root"]) / "experiments" / "odd_taylor_attention"
                / "checkpoints" / name).resolve()
    if checkpoint_root != expected or not (expected / str(best["step"])).is_dir():
        raise ValueError("Validation-best checkpoint is missing from its canonical storage")
    records = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not any(row["step"] == completion["final_step"]
               and row["validation_nll"] == completion["validation_nll"] for row in records):
        raise ValueError("Terminal validation has not been durably recorded")
    candidates = [row for row in records if row.get("event") == "validation"]
    selected = min(candidates, key=lambda row: (row["validation_nll"], -row["step"]))
    if selected["step"] != best["step"] or selected["validation_nll"] != best["validation_nll"]:
        raise ValueError("Selected checkpoint does not match the best committed validation")
    record = {
        "format": "odd-taylor-checkpoint-retention", "version": 1,
        "source_commit": manifest["source_commit"], "seed": config["train"]["seed"],
        "completion_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        "selection_rule": "lowest_validation_nll_latest_on_tie", "best_checkpoint_step": best["step"],
        "best_checkpoint_path": str(expected / str(best["step"])),
        "validation_nll": best["validation_nll"],
        "terminal_step": completion["final_step"],
        "terminal_metrics_preserved": True,
    }
    # Record intended retention before pruning: a requeue after any deletion
    # recognizes that terminal metrics are complete even when best is earlier.
    write_json_atomic(workdir / "retention.json", record)
    for path in expected.iterdir():
        if path.is_dir() and ((path.name.isdigit() and int(path.name) != best["step"])
                              or path.name.endswith(".orbax-checkpoint-tmp")):
            shutil.rmtree(path)
    metadata_path = workdir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(checkpoint_retention=record,
                    retained_terminal_checkpoint_step=(best["step"] if best["step"] == completion["final_step"] else None))
    write_json_atomic(metadata_path, metadata)
    report_path = workdir / "REPORT.md"
    if report_path.exists():
        lines = report_path.read_text().splitlines()
        lines = [line for line in lines if not line.startswith("Retained terminal checkpoint:")
                 and not line.startswith("Retained validation-best checkpoint:")]
        lines += [f"Retained validation-best checkpoint: `{best['step']}`. Terminal metrics remain in `metrics.jsonl`."]
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
