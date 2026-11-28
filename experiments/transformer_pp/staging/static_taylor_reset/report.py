"""Regenerate the held-out checkpoint intervention tables from paired sequence losses."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import yaml

from scaffolds.records.selections import verify_pinned_runs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"
ARMS = {
    "static_native": "Learned static, trained coefficients",
    "static_reset": "Learned static, Taylor coefficients",
    "taylor_native": "Fixed Taylor, trained with Taylor coefficients",
}


def paired_loss(first: list[float], second: list[float]) -> tuple[float, float, float]:
    differences = [a - b for a, b in zip(first, second, strict=True)]
    mean = statistics.mean(differences)
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    half_width = statistics.NormalDist().inv_cdf(.975) * sem
    return mean, mean - half_width, mean + half_width


def tables(rows: list[dict], expected_sequences: int) -> str:
    arms = {row["condition"]: row for row in rows}
    if set(arms) != set(ARMS) or len(arms) != len(rows):
        raise ValueError("Expected one measurement per evaluation condition")
    lines = ["| Frozen checkpoint and coefficients | NLL (nats/token) | PPL |", "| --- | ---: | ---: |"]
    for arm, label in ARMS.items():
        row = arms[arm]
        losses = row["sequence_nll"]
        if len(losses) != expected_sequences or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"Incomplete or nonfinite sequence losses in {arm}")
        if not math.isclose(statistics.mean(losses), row["nll"], abs_tol=2e-6):
            raise ValueError(f"Sequence losses disagree with the full-set loss in {arm}")
        lines.append(f"| {label} | {row['nll']:.6f} | {row['perplexity']:.4f} |")

    lines.extend(["", "| Difference | Δ NLL (nats/token) | Paired 95% interval | Δ PPL | Packed rows with higher loss |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    contrasts = [
        ("Reset minus trained coefficients on the static backbone", "static_reset", "static_native"),
        ("Static-trained minus Taylor-trained backbone, both using Taylor", "static_reset", "taylor_native"),
    ]
    for label, first, second in contrasts:
        delta, lo, hi = paired_loss(arms[first]["sequence_nll"], arms[second]["sequence_nll"])
        delta_ppl = arms[first]["perplexity"] - arms[second]["perplexity"]
        worse = sum(a > b for a, b in zip(arms[first]["sequence_nll"], arms[second]["sequence_nll"], strict=True))
        lines.append(f"| {label} | {delta:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {delta_ppl:+.4f} "
                     f"| {worse:,}/{expected_sequences:,} |")
    return "\n".join(lines)


def generated_tables() -> str:
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    evaluation, = [entry for entry in selection["runs"] if entry["role"] == "paired_evaluation"]
    metrics = ROOT / evaluation["sources"]["metrics.jsonl"]["path"]
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    return tables(rows, selection["settings"]["packed_sequences"])


def main() -> None:
    report = HERE / "README.md"
    before, after = report.read_text().split(BEGIN)
    _, after = after.split(END)
    report.write_text(f"{before}{BEGIN}\n\n{generated_tables()}\n\n{END}{after}")


if __name__ == "__main__":
    main()
