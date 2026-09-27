"""Regenerate `results/unnormalized_taylor_attention/summary.csv` from runs/ and the manifest's execution records.

One row per run: Taylor degree, terminal and validation-best metrics, source commit,
Slurm job, and the completing allocation's hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from functools import partial
from scaffolds.transformer.analysis import load_selected_study, selected_run_directory

from scaffolds.transformer.analysis import verify_result_selection

from scaffolds.transformer.analysis import terminal_row, write_csv

locate_run = partial(selected_run_directory, Path(__file__).resolve().parent)

OUTPUT = Path(__file__).resolve().parent
RECORD = OUTPUT.parents[1] / "experiments" / "unnormalized_taylor_attention"
FIELDS = ["run", "attention_degree", "parameters", "updates", "prediction_tokens", "terminal_validation_nll",
          "terminal_validation_perplexity", "best_step", "best_validation_nll", "best_validation_perplexity",
          "source_commit", "slurm_job_id", "hardware"]


def rows(record: Path = RECORD) -> list[dict]:
    manifest = load_selected_study(Path(__file__).resolve().parent, record)
    result = []
    for run in manifest["runs"]:
        row = terminal_row(locate_run(record, run["name"]))
        node = run["execution"]["attempts"][-1]["node"]
        row.update(run=run["name"], attention_degree=run["attention_degree"], hardware=f"{row['gpu_count']}x{row['gpu_type']} {node}")
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    verify_result_selection(Path(__file__).resolve().parent)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", rows(args.record), FIELDS)
    print(args.output / "summary.csv")


if __name__ == "__main__":
    main()
