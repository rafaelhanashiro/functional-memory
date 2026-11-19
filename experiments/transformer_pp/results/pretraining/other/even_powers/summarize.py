"""Regenerate the explicitly selected even-power comparison from pinned measurements."""
import csv
import json
from pathlib import Path
import sys

import yaml

DIRECTORY = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
sys.path.insert(0, str(ROOT))

from scaffolds.records.selections import verify_pinned_runs


def main():
    selection = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    rows = []
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        record = json.loads((run / "record.json").read_text())
        row = {
            "role": entry["role"], "run_id": entry["run_id"], "status": record["status"],
            "terminal_step": "", "terminal_validation_perplexity": "",
            "best_step": "", "best_validation_perplexity": "",
        }
        if record["status"] != "complete":
            rows.append(row)
            continue
        completion = json.loads((run / "completion.json").read_text())
        final_step = completion["final_step"]
        measurements = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        terminal = [row for row in measurements
                    if row.get("step") == final_step and "validation_perplexity" in row][-1]
        best = json.loads((run / "best_validation.json").read_text())
        row.update({
            "terminal_step": final_step,
            "terminal_validation_perplexity": f"{terminal['validation_perplexity']:.9f}",
            "best_step": best["step"],
            "best_validation_perplexity": f"{best['validation_perplexity']:.9f}",
        })
        rows.append(row)
    with (DIRECTORY / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print((DIRECTORY / "summary.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
