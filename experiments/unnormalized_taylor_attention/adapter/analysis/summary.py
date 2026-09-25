"""Regenerate `experiments/unnormalized_taylor_attention/summary.csv` from runs/ and the manifest's execution records.

One row per run: Taylor degree, terminal and validation-best metrics, source commit,
Slurm job, and the completing allocation's hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scaffolds.transformer.analysis import terminal_row, write_csv

RECORD = Path(__file__).resolve().parents[2]
FIELDS = ["run", "attention_degree", "parameters", "updates", "prediction_tokens", "terminal_validation_nll",
          "terminal_validation_perplexity", "best_step", "best_validation_nll", "best_validation_perplexity",
          "source_commit", "slurm_job_id", "hardware"]


def rows(record: Path = RECORD) -> list[dict]:
    manifest = json.loads((record / "provenance/manifest.json").read_text())
    result = []
    for run in manifest["runs"]:
        row = terminal_row(record / "runs" / run["name"])
        node = run["execution"]["attempts"][-1]["node"]
        row.update(run=run["name"], attention_degree=run["attention_degree"], hardware=f"{row['gpu_count']}x{row['gpu_type']} {node}")
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args()
    write_csv(args.record / "summary.csv", rows(args.record), FIELDS)
    print(args.record / "summary.csv")


if __name__ == "__main__":
    main()
