"""Regenerate matched continuation measurements from this topic's explicit selection."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
ORDERS = 5
FIELDS = ["role", "run_id", "status", "continuation_update", "global_step", "validation_nll",
          "validation_perplexity", "sampled_train_nll", "is_terminal", "is_validation_best",
          *[f"mean_u{r}" for r in range(ORDERS)], "query_amplitude_mean_abs", "query_amplitude_max_abs",
          "query_projection_l2_mean"]


def coefficient_diagnostics(diagnostics: dict) -> dict:
    """Factorial-coordinate coefficients u_r = r! c_r averaged over layers and heads, and the query generator's state."""
    coefficients = diagnostics.get("coefficients")
    if not coefficients:
        return {}
    factorials = np.asarray([math.factorial(order) for order in range(ORDERS)], dtype=np.float64)
    if "projection_bias" in coefficients:
        bias = np.asarray(coefficients["projection_bias"], dtype=np.float64)
        amplitude = np.asarray(coefficients["modulation_amplitude"], dtype=np.float64)
        kernel = np.asarray(coefficients["projection_kernel_l2"], dtype=np.float64)
        # Stored biases are factorial coordinates already; the mean coefficient over tokens
        # with a zero-mean tanh feature is the bias.
        mean_u = bias.reshape(-1, bias.shape[-1]).mean(axis=0)
        extra = {"query_amplitude_mean_abs": float(np.abs(amplitude).mean()),
                 "query_amplitude_max_abs": float(np.abs(amplitude).max()),
                 "query_projection_l2_mean": float(kernel.mean())}
    elif "effective" in coefficients:
        effective = np.asarray(coefficients["effective"], dtype=np.float64)
        mean_u = (effective.reshape(-1, effective.shape[-1]) * factorials[:effective.shape[-1]]).mean(axis=0)
        extra = {}
    else:
        return {}
    return {**{f"mean_u{r}": float(mean_u[r]) for r in range(min(ORDERS, mean_u.shape[0]))}, **extra}


def collect() -> list[dict]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    rows = []
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        record = json.loads((run / "record.json").read_text())
        status = record["status"]
        path = run / "metrics.jsonl"
        if not path.exists():
            rows.append({"role": entry["role"], "run_id": entry["run_id"], "status": status})
            continue
        start = int(record["continuation"]["start_step"])
        final_step = start + int(record["continuation"]["updates"])
        measurements = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        validations = {int(row["step"]): row for row in measurements if row.get("event") == "validation"}
        best_path = run / "best_validation.json"
        best_step = json.loads(best_path.read_text())["step"] if best_path.exists() else None
        for step, row in sorted(validations.items()):
            rows.append({
                "role": entry["role"], "run_id": entry["run_id"], "status": status,
                "continuation_update": step - start, "global_step": step,
                "validation_nll": row["validation_nll"], "validation_perplexity": row["validation_perplexity"],
                "sampled_train_nll": row.get("train_nll_sample_mean", ""),
                "is_terminal": status == "complete" and step == final_step,
                "is_validation_best": step == best_step,
                **coefficient_diagnostics(row.get("attention_diagnostics", {})),
            })
    return rows


def main() -> None:
    with (TOPIC / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(collect())


if __name__ == "__main__":
    main()
