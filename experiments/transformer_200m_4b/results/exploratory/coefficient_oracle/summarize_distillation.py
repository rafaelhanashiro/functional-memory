"""Regenerate the selected teacher-distillation comparison from pinned metrics.

Only the explicitly selected distillation run is verified. Its committed
completion and evaluation measurements contain the teacher and Taylor replays;
no checkpoint, dataset, or unrelated historical run is read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from scaffolds.records.selections import verify_pinned_runs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ROLE = "dense_teacher_distillation_m4_head_scale"
COLUMNS = ("model", "step", "fit_ppl", "validation_ppl", "test_ppl")


def perplexity(measurement):
    value = float(measurement["perplexity"])
    if not math.isfinite(value) or not math.isclose(value, math.exp(float(measurement["nll"])), rel_tol=1e-10):
        raise ValueError("Invalid perplexity measurement")
    return f"{value:.6f}"


def comparison_rows():
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    entries = [entry for entry in selection["runs"] if entry.get("role") == ROLE]
    if len(entries) != 1:
        raise ValueError(f"Select exactly one {ROLE} run")
    entry = entries[0]
    verify_pinned_runs(ROOT, [entry])
    run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
    record = json.loads((run / "record.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    settings = recipe["distillation"]
    if (record["status"] != "complete" or completion["status"] != "complete"
            or recipe["model"]["attention_degree"] != 4
            or not recipe["model"]["qk_norm_head_scale"]
            or recipe["model"]["attention_coefficient_readout"] != "dense"
            or settings["parent"]["run_id"] != entry["parent_run_id"]
            or settings["teacher"]["run_id"] != entry["teacher_run_id"]):
        raise ValueError("Expected a completed degree-four dense distillation with the selected teacher and parent")
    capacity_step = completion["best_regression_step"]
    evaluations = [row for row in map(json.loads, (run / "metrics.jsonl").read_text().splitlines())
                   if row.get("event") == "evaluation" and row["step"] == capacity_step]
    if len(evaluations) != 1:
        raise ValueError("The coefficient-best snapshot needs exactly one pinned evaluation")
    capacity_evaluation = evaluations[0]
    if not math.isclose(capacity_evaluation["fit"]["nll"], completion["capacity_best_fit"]["nll"], abs_tol=1e-8):
        raise ValueError("Coefficient-best replay and recorded fitting evaluation disagree")
    baseline = completion["baseline"]
    return [
        {"model": "Frozen Taylor", "step": settings["parent"]["checkpoint_step"],
         "fit_ppl": perplexity(baseline["fit"]), "validation_ppl": perplexity(baseline["validation"]),
         "test_ppl": perplexity(baseline["test"])},
        {"model": "Fitted free teacher", "step": settings["teacher"]["checkpoint_step"],
         "fit_ppl": perplexity(completion["teacher_fit"]), "validation_ppl": "", "test_ppl": ""},
        {"model": "Dense, best coefficient MSE", "step": capacity_step,
         "fit_ppl": perplexity(completion["capacity_best_fit"]),
         "validation_ppl": perplexity(capacity_evaluation["validation"]), "test_ppl": ""},
        {"model": "Validation-selected dense", "step": completion["best_step"],
         "fit_ppl": perplexity(completion["validation_best_fit"]),
         "validation_ppl": perplexity(completion["validation_best_validation"]),
         "test_ppl": perplexity(completion["test"])},
    ]


def markdown(rows):
    lines = ["| Model / selected snapshot | Fitting PPL | Fresh validation PPL | Untouched test PPL |",
             "| --- | ---: | ---: | ---: |"]
    for index, row in enumerate(rows):
        name = row["model"]
        if index == 2:
            name += f" (update {row['step']:,})"
        elif index == 3:
            name += f" ({'initialization, ' if row['step'] == 0 else ''}update {row['step']:,})"
        missing = "Not applicable" if index == 1 else "Not evaluated"
        lines.append(f"| {name} | {row['fit_ppl']} | {row['validation_ppl'] or missing} | {row['test_ppl'] or missing} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-markdown", action="store_true", help="Also print the README table")
    arguments = parser.parse_args()
    rows = comparison_rows()
    with (HERE / "distillation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if arguments.print_markdown:
        print(markdown(rows))


if __name__ == "__main__":
    main()
