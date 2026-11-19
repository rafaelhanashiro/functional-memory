"""Summarize historical gradient windows from explicitly pinned scientific inputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
sys.path.insert(0, str(REPOSITORY))

import yaml

from scaffolds.records.selections import verify_pinned_runs


def read_json(path: Path):
    return json.loads(path.read_text())


def summarize(repository: Path, entry: dict, settings: dict) -> dict:
    run = repository / "experiments" / entry["study"] / "runs" / entry["run_id"]
    record = read_json(run / "record.json")
    config = yaml.safe_load((run / "config.yaml").read_text())
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("train_gradient_norm_count", 0) > 0]
    first, last = settings["first_update"], settings["last_update"]
    expected_start = 0
    for row in rows:
        start, stop = row["training_window_start_step"], row["step"]
        if start != expected_start or row["train_gradient_norm_count"] != stop - start:
            raise ValueError(f"Noncontiguous gradient windows: {run.name} at {stop}")
        if row["gradient_clip_norm"] != settings["clip_threshold"]:
            raise ValueError(f"Different clipping threshold: {run.name}")
        expected_start = stop
    windows = [row for row in rows if row["training_window_start_step"] >= first - 1 and row["step"] <= last]
    count = sum(row["train_gradient_norm_count"] for row in windows)
    if count != last - first + 1:
        raise ValueError(f"Incomplete analysis interval: {run.name}: {count} updates")
    peak = max(windows, key=lambda row: row["train_gradient_norm_max"])
    clipped = 0
    for row in windows:
        value = row["train_gradient_clipped_fraction"] * row["train_gradient_norm_count"]
        if not math.isclose(value, round(value), abs_tol=1e-7):
            raise ValueError(f"Noninteger clipped count: {run.name} at {row['step']}")
        clipped += round(value)
    best = read_json(run / "best_validation.json")
    completion = read_json(run / "completion.json")
    if record["status"] != "complete" or completion["final_step"] != settings["training_steps"]:
        raise ValueError(f"Selected historical run is not complete: {run.name}")
    model = config["model"]
    return {
        "role": entry["role"], "run_id": run.name, "status": record["status"],
        "label": entry["label"], "attention_degree": model.get("attention_degree", ""),
        "positional_encoding": model["positional_encoding"],
        "first_update": first, "last_update": last, "update_count": count,
        "max_gradient_norm": format(peak["train_gradient_norm_max"], ".9f"),
        "max_window_first_update": peak["training_window_start_step"] + 1,
        "max_window_last_update": peak["step"],
        "mean_gradient_norm": format(sum(row["train_gradient_norm_mean"] * row["train_gradient_norm_count"] for row in windows) / count, ".9f"),
        "clipped_updates": clipped, "clipped_percent": format(100 * clipped / count, ".6f"),
        "peak_window_sampled_train_nll": format(peak["train_nll_sample_mean"], ".9f"),
        "peak_window_validation_nll": format(peak["validation_nll"], ".9f"),
        "best_step": best["step"], "best_validation_ppl": format(best["validation_perplexity"], ".9f"),
        "terminal_step": completion["final_step"],
        "terminal_validation_ppl": format(completion["validation_perplexity"], ".9f"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY,
                        help="Checkout containing the pinned canonical runs; defaults to this script's repository.")
    args = parser.parse_args()
    selection = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    repository = args.repository.resolve()
    verify_pinned_runs(repository, selection["runs"])
    rows = [summarize(repository, entry, selection["settings"]) for entry in selection["runs"]]
    output = DIRECTORY / "history_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
