"""Regenerate matched continuation measurements from this topic's explicit selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]


def collect() -> list[dict]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    start = int(selection["settings"]["parent_step"])
    rows = []
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        status = json.loads((run / "record.json").read_text())["status"]
        path = run / "metrics.jsonl"
        if not path.exists():
            rows.append({"role": entry["role"], "run_id": entry["run_id"], "status": status})
            continue
        measurements = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        validations = {int(row["step"]): row for row in measurements if row.get("event") == "validation"}
        best_path = run / "best_validation.json"
        best_step = json.loads(best_path.read_text())["step"] if best_path.exists() else None
        final_step = start + int(selection["settings"]["updates"])
        for step, row in sorted(validations.items()):
            rows.append({
                "role": entry["role"], "run_id": entry["run_id"], "status": status,
                "continuation_update": step - start, "global_step": step,
                "validation_nll": row["validation_nll"], "validation_perplexity": row["validation_perplexity"],
                "sampled_train_nll": row.get("train_nll_sample_mean", ""),
                "is_terminal": status == "complete" and step == final_step,
                "is_validation_best": step == best_step,
            })
    return rows


def main() -> None:
    fields = ["role", "run_id", "status", "continuation_update", "global_step", "validation_nll",
              "validation_perplexity", "sampled_train_nll", "is_terminal", "is_validation_best"]
    with (TOPIC / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(collect())


if __name__ == "__main__":
    main()
