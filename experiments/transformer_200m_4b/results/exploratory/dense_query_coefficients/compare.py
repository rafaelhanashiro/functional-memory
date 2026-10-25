"""Held-out PPL at matched token counts for the dense query coefficient topic.

Checks the explicit selection's scientific pins, then reads each run's measurements.
Runs without measurements remain visible with their authoritative execution status.
Writes summary.csv next to this file.

    python -m experiments.transformer_200m_4b.results.exploratory.dense_query_coefficients.compare
"""
import csv
import json
import math
from pathlib import Path

import yaml

from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]


def collect_rows(root, selection):
    verify_pinned_runs(root, selection["runs"])
    matched = selection["comparison"]["matched_updates"]
    tokens = selection["comparison"]["tokens_per_update"]
    rows = []
    for entry in selection["runs"]:
        run = root / "experiments" / entry["study"] / "runs" / entry["run_id"]
        config = yaml.safe_load((run / "config.yaml").read_text())
        model = config["model"]
        record = json.loads((run / "record.json").read_text())
        row = {"run_id": entry["run_id"], "label": entry["label"], "role": entry["role"],
               "baseline_run_id": entry.get("baseline_run_id", ""),
               "degree": model["attention_degree"],
               "features": (model.get("attention_coefficient_feature_count") or model["attention_degree"] + 1)
               if model.get("attention_coefficient_readout") == "dense" else "",
               "qk_norm_type": model.get("qk_norm_type", "cosine_log_scale"),
               "qk_norm_head_scale": model.get("qk_norm_head_scale", False),
               "readout_scale": model.get("attention_coefficient_readout_scale", 1.0),
               "status": record["status"],
               "target_step": config["optim"]["total_steps"],
               "target_tokens": tokens * config["optim"]["total_steps"]}
        metrics = run / "metrics.jsonl"
        validation = {}
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                measurement = json.loads(line)
                if "validation_nll" in measurement:
                    validation[measurement["step"]] = measurement["validation_nll"]
        if validation:
            latest = max(validation)
            row.update(latest_evaluation_step=latest, latest_ppl=f"{math.exp(validation[latest]):.4f}")
        for update in matched:
            row[f"ppl_at_{update}"] = f"{math.exp(validation[update]):.4f}" if update in validation else ""
        best = run / "best_validation.json"
        completion = run / "completion.json"
        if best.exists():
            best = json.loads(best.read_text())
            row.update(best_step=best["step"], best_ppl=f"{best['validation_perplexity']:.4f}")
        if completion.exists():
            if record["status"] != "complete":
                raise ValueError(f"{entry['run_id']} has completion measurements but status {record['status']!r}")
            completion = json.loads(completion.read_text())
            row.update(terminal_step=completion["final_step"],
                       terminal_ppl=f"{completion['validation_perplexity']:.4f}",
                       tokens_at_terminal=tokens * completion["final_step"])
        rows.append(row)
    return rows


def main():
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    rows = collect_rows(ROOT, selection)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(TOPIC / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
