"""Regenerate matched continuation measurements from this topic's explicit selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = TOPIC.parents[4]
FIELDS = ["role", "run_id", "status", "continuation_update", "global_step", "validation_nll",
          "validation_perplexity", "sampled_train_nll", "is_terminal", "is_validation_best",
          "gate_half_life_median_tokens", "gate_heads_half_life_below_2", "gate_heads_half_life_above_64",
          "gate_mean_log_alpha_median", "gate_min_log_alpha_median", "query_amplitude_mean_abs"]


def gate_diagnostics(diagnostics: dict) -> dict:
    """Forget-gate half-lives and per-token log alpha extremes on the diagnostic sequence, per head."""
    coefficients = diagnostics.get("coefficients", {})
    report = coefficients.get("forget_gate")
    if not report:
        return {}
    half_life = np.asarray(report["mean_half_life_tokens"], dtype=np.float64).ravel()
    finite = half_life[np.isfinite(half_life)]
    sample = {name: np.asarray(values, dtype=np.float64) for name, values in report["log_alpha_sample"].items()}
    amplitude = np.asarray(coefficients.get("modulation_amplitude", np.nan), dtype=np.float64)
    return {
        "gate_half_life_median_tokens": float(np.median(finite)) if finite.size else "",
        "gate_heads_half_life_below_2": float((half_life < 2).mean()),
        "gate_heads_half_life_above_64": float((half_life > 64).mean()),
        "gate_mean_log_alpha_median": float(np.median(sample["mean"])),
        "gate_min_log_alpha_median": float(np.median(sample["min"])),
        "query_amplitude_mean_abs": float(np.abs(amplitude).mean()),
    }


def collect() -> list[dict]:
    selection = yaml.safe_load((TOPIC / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    start = int(selection["settings"]["parent_step"])
    final_step = start + int(selection["settings"]["updates"])
    rows = []
    for entry in selection["runs"]:
        run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        status = json.loads((run / "record.json").read_text())["status"]
        path = run / "metrics.jsonl"
        if not path.exists():
            rows.append({"role": entry["role"], "run_id": entry["run_id"], "status": status})
            continue
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
                **gate_diagnostics(row.get("attention_diagnostics", {})),
            })
    return rows


def main() -> None:
    with (TOPIC / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(collect())


if __name__ == "__main__":
    main()
