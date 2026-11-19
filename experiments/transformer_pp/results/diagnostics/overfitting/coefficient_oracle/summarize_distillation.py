"""Regenerate the selected teacher-distillation comparison (fitting and fresh batch) from pinned metrics.

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
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
ROLE = "dense_teacher_distillation_m4_head_scale"
COLUMNS = ("model", "step", "fit_ppl", "heldout_ppl")


def perplexity(measurement):
    value = float(measurement["perplexity"])
    if not math.isfinite(value) or not math.isclose(value, math.exp(float(measurement["nll"])), rel_tol=1e-10):
        raise ValueError("Invalid perplexity measurement")
    return f"{value:.6f}"


def comparison_rows():
    """Two batches: the fitting batch and one fresh held-out batch.

    The dense generator is the snapshot with the lowest coefficient-regression error on
    the fitting batch; the held-out batch only scores it and the Taylor baseline.
    """
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
    baseline = completion["baseline"]
    if "selected_heldout" in completion:
        step, fit, heldout, baseline_heldout = (completion["best_step"], completion["selected_fit"],
                                                completion["selected_heldout"], baseline["heldout"])
    else:
        # Runs that also selected on fresh data recorded the lowest-regression-error
        # snapshot's fresh-batch score in its pinned evaluation row.
        step = completion["best_regression_step"]
        evaluations = [row for row in map(json.loads, (run / "metrics.jsonl").read_text().splitlines())
                       if row.get("event") == "evaluation" and row["step"] == step]
        if len(evaluations) != 1:
            raise ValueError("The lowest-regression-error snapshot needs exactly one pinned evaluation")
        if not math.isclose(evaluations[0]["fit"]["nll"], completion["capacity_best_fit"]["nll"], abs_tol=1e-8):
            raise ValueError("Selected-snapshot replay and recorded fitting evaluation disagree")
        fit, heldout, baseline_heldout = completion["capacity_best_fit"], evaluations[0]["validation"], baseline["validation"]
    return [
        {"model": "Frozen Taylor", "step": settings["parent"]["checkpoint_step"],
         "fit_ppl": perplexity(baseline["fit"]), "heldout_ppl": perplexity(baseline_heldout)},
        {"model": "Fitted free teacher", "step": settings["teacher"]["checkpoint_step"],
         "fit_ppl": perplexity(completion["teacher_fit"]), "heldout_ppl": ""},
        {"model": "Dense, regressed onto the teacher", "step": step,
         "fit_ppl": perplexity(fit), "heldout_ppl": perplexity(heldout)},
    ]


def markdown(rows):
    lines = ["| Model | Fitting batch PPL | Fresh batch PPL |", "| --- | ---: | ---: |"]
    for index, row in enumerate(rows):
        name = row["model"] + (f" (update {row['step']:,})" if index == 2 else "")
        lines.append(f"| {name} | {row['fit_ppl']} | {row['heldout_ppl'] or 'Not applicable'} |")
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
