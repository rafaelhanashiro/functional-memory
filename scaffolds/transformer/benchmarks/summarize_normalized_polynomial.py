"""Regenerate normalized learned-polynomial throughput tables from completed reports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BEGIN = "<!-- BEGIN GENERATED NORMALIZED POLYNOMIAL THROUGHPUT -->"
END = "<!-- END GENERATED NORMALIZED POLYNOMIAL THROUGHPUT -->"
FAMILIES = [("direct", degree) for degree in (1, 2, 3, 4)] + [("sum_of_squares", degree) for degree in (2, 4)]
LABELS = {"direct": "Unrestricted", "sum_of_squares": "Nonnegative"}


def completed(path):
    report = json.loads(path.read_text())
    if report.get("status") != "complete":
        raise ValueError(f"Incomplete benchmark report: {path}")
    return report


def timings(entry, operation=None):
    values = []
    for backend in ("blocked", "fused"):
        timing = entry["implementations"][backend]
        value = float((timing[operation] if operation else timing)["median_ms"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {backend} median: {value}")
        values.append(value)
    blocked, fused = values
    return f"{blocked:.3f} | {fused:.3f} | {blocked / fused:.2f}×"


def render(validation):
    runs = validation / "runs"
    lines = ["Kernel forward and backward, including all parameter gradients:", "",
             "| GPU | Family | Degree | Blocked (ms) | Fused (ms) | Speedup |",
             "|---|---|---:|---:|---:|---:|"]
    for gpu in ("h100", "b300"):
        report = completed(runs / f"normalized-polynomial-{gpu}" / "benchmark.json")
        entries = {(entry["parameterization"], entry["degree"]): entry for entry in report["results"]}
        if set(entries) != set(FAMILIES) or len(report["results"]) != len(FAMILIES):
            raise ValueError(f"Unexpected kernel benchmark grid for {gpu}")
        for family, degree in FAMILIES:
            lines.append(f"| {gpu.upper()} | {LABELS[family]} | {degree} | "
                         f"{timings(entries[family, degree], 'forward_backward')} |")
    lines += ["", "Complete Transformer updates on two B300 GPUs:", "",
              "| Family | Degree | Blocked (ms) | Fused (ms) | Speedup |", "|---|---:|---:|---:|---:|"]
    for family in LABELS:
        for degree in (2, 4):
            report = completed(runs / "normalized-polynomial-b300-updates" / f"{family}-m{degree}.json")
            entries = report["results"]
            model = report["configuration"]["model_and_training_config"]["model"]
            if len(entries) != 1 or entries[0]["degree"] != degree or model["attention_coefficient_parameterization"] != family:
                raise ValueError(f"Unexpected update benchmark configuration: {family}, degree {degree}")
            lines.append(f"| {LABELS[family]} | {degree} | {timings(entries[0])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--check", action="store_true", help="Fail if regeneration would change the README")
    args = parser.parse_args()
    validation = args.root.resolve() / "scaffolds/transformer/validation"
    path = validation / "README.md"
    original = path.read_text()
    if original.count(BEGIN) != 1 or original.count(END) != 1 or original.index(BEGIN) >= original.index(END):
        raise ValueError("README must contain exactly one ordered throughput marker pair")
    before, rest = original.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    updated = before + BEGIN + "\n" + render(validation) + "\n" + END + after
    if args.check and updated != original:
        raise SystemExit("Generated normalized-polynomial throughput tables are stale")
    if not args.check and updated != original:
        path.write_text(updated)


if __name__ == "__main__":
    main()
