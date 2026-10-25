"""Held-out PPL at matched token counts for the dense query coefficient topic.

Reads the explicit selection and each run's metrics.jsonl; runs without measurements stay
listed as unavailable. Writes summary.csv next to this file.

    python experiments/transformer_200m_4b/results/exploratory/dense_query_coefficients/compare.py
"""
import csv
import json
import math
from pathlib import Path

import yaml

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]


def main():
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    matched = selection["comparison"]["matched_updates"]
    tokens = selection["comparison"]["tokens_per_update"]
    rows = []
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        model = yaml.safe_load((run / "config.yaml").read_text())["model"]
        row = {"run_id": entry["run_id"], "label": entry["label"], "role": entry["role"],
               "degree": model["attention_degree"],
               "features": (model.get("attention_coefficient_feature_count") or model["attention_degree"] + 1)
               if model.get("attention_coefficient_readout") == "dense" else "",
               "status": json.loads((run / "record.json").read_text()).get("status")}
        metrics = run / "metrics.jsonl"
        validation = {}
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                record = json.loads(line)
                if "validation_nll" in record:
                    validation[record["step"]] = record["validation_nll"]
        for update in matched:
            row[f"ppl_at_{update}"] = f"{math.exp(validation[update]):.4f}" if update in validation else ""
        best = run / "best_validation.json"
        completion = run / "completion.json"
        if best.exists():
            best = json.loads(best.read_text())
            row.update(best_step=best["step"], best_ppl=f"{best['validation_perplexity']:.4f}")
        if completion.exists():
            completion = json.loads(completion.read_text())
            row.update(terminal_step=completion["final_step"], terminal_ppl=f"{completion['validation_perplexity']:.4f}")
        row["tokens_at_terminal"] = tokens * 7570
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(TOPIC / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
