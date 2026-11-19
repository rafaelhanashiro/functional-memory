"""Write polynomial.json, the Polynomial runs page's data, from the Transformer++ result summaries.

    python3 site/data/polynomial.py

Every value is a run's validation-best perplexity (the site always quotes best, never terminal),
from query_width_degree/summary.csv and degree_scaling/summary.csv. Rerun only when the user
asks for newer results.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/transformer_pp/results"
WIDTH = RESULTS / "exploratory/query_width_degree/summary.csv"
DEGREE = RESULTS / "degree_scaling/summary.csv"
FAMILIES = (  # summary family, table label, chart legend
    ("taylor", "Taylor", "Fixed Taylor"),
    ("hermite", "Hermite", "Fixed Hermite"),
    ("static", "Static", "Learned static"),
    ("query_w_decay", "Query + W decay", "Query, W decay"),
)
REFERENCES = (("softmax", "Softmax", "solid"), ("exp", "Exp", "dashed"))


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def width_table() -> dict:
    runs = [row for row in rows(WIDTH) if row["status"] == "complete"]
    widths = sorted({int(row["qk_head_dim"]) for row in runs})
    degrees = sorted({int(row["attention_degree"]) for row in runs})
    cells = {(int(row["qk_head_dim"]), int(row["attention_degree"])): row for row in runs}
    return {
        "title": "Scaling Query Attention + WD",
        "metric": "validation-best perplexity",
        "source": WIDTH.relative_to(ROOT).as_posix(),
        "degrees": degrees,
        "rows": [{"width": width, "cells": [
            {"degree": degree, "run_id": cells[width, degree]["run_id"],
             "perplexity": float(cells[width, degree]["best_validation_perplexity"])}
            if (width, degree) in cells else None for degree in degrees]} for width in widths],
    }


def degree_scaling() -> dict:
    runs = rows(DEGREE)
    primary = [row for row in runs if row["role"] == "primary" and row["status"] == "complete"]
    degrees = sorted({int(row["degree"]) for row in primary})
    references = {row["family"]: row for row in runs if row["role"] == "reference"}
    return {
        "title": "Degree scaling of unnormalized polynomial attention",
        "metric": "validation-best perplexity",
        "source": DEGREE.relative_to(ROOT).as_posix(),
        "degrees": degrees,
        "families": [{"family": family, "label": label, "legend": legend, "points": [
            {"degree": int(row["degree"]), "run_id": row["run_id"],
             "perplexity": float(row["best_validation_perplexity"])}
            for row in sorted((row for row in primary if row["family"] == family), key=lambda row: int(row["degree"]))]}
            for family, label, legend in FAMILIES],
        "references": [{"family": family, "label": label, "line": line, "run_id": references[family]["run_id"],
                        "perplexity": float(references[family]["best_validation_perplexity"])}
                       for family, label, line in REFERENCES],
    }


def main() -> None:
    data = {"width_degree": width_table(), "degree_scaling": degree_scaling()}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
