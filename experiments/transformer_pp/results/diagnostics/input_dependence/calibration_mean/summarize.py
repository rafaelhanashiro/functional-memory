"""Rebuild the native and calibration-mean PPL table for the selected checkpoints."""

import csv
import json
import math
from pathlib import Path

import yaml

from scaffolds.records import load_run
from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = next(path for path in TOPIC.parents if (path / "AGENTS.md").is_file())
FIELDS = ("model", "run_id", "parent_run_id", "native_ppl", "mean_ppl", "ppl_change_percent")


def collect():
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    settings = selection["settings"]
    rows = []
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        recipe = yaml.safe_load((run / "config.yaml").read_text())["evaluation"]
        if load_run(run)["status"] != "complete" or recipe["parent_run_id"] != entry["parent_run_id"]:
            raise ValueError(f"{run.name}: expected a completed evaluation of the selected parent")
        for setting in ("checkpoint_step", "calibration_sequences", "evaluation_sequences", "layer_scope"):
            if recipe[setting] != settings[setting]:
                raise ValueError(f"{run.name}: {setting} differs from the selection")

        conditions = {}
        for line in (run / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            condition = row["condition"]
            if condition in ("native", "calibration_mean") and row.get("layer_scope") == settings["layer_scope"]:
                if condition in conditions or len(row["sequence_nll"]) != settings["evaluation_sequences"]:
                    raise ValueError(f"{run.name}: ambiguous or incomplete {condition} measurement")
                conditions[condition] = row["nll"]
        native = math.exp(conditions["native"])
        mean = math.exp(conditions["calibration_mean"])
        rows.append(dict(model=entry["label"], run_id=run.name, parent_run_id=entry["parent_run_id"],
                         native_ppl=native, mean_ppl=mean, ppl_change_percent=100 * (mean / native - 1)))
    return rows


def write_summary(path):
    rows = collect()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    write_summary(TOPIC / "summary.csv")
