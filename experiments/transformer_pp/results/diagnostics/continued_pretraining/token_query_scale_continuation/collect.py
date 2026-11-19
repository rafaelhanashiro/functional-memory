"""Regenerate matched continuation measurements from this topic's explicit selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

TOPIC = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
FIELDS = ["role", "run_id", "status", "continuation_update", "global_step", "validation_nll",
          "validation_perplexity", "sampled_train_nll", "is_terminal", "is_validation_best",
          "gate_amplitude_mean_abs", "gate_amplitude_max_abs", "gate_heads_above_0p1",
          "gate_log_scale_sample_std_mean", "gate_log_scale_sample_max_abs"]


def gate_diagnostics(diagnostics: dict) -> dict:
    """Amplitudes a_h and the realized per-token log scale on the diagnostic sequence."""
    report = diagnostics.get("qk_norm_token_scales")
    if not report:
        return {}
    amplitude = np.asarray(report["effective"], dtype=np.float64)[..., 0]
    sample = {name: np.asarray(values, dtype=np.float64) for name, values in report["log_delta_sample"].items()}
    return {
        "gate_amplitude_mean_abs": float(np.abs(amplitude).mean()),
        "gate_amplitude_max_abs": float(np.abs(amplitude).max()),
        "gate_heads_above_0p1": int((np.abs(amplitude) > 0.1).sum()),
        "gate_log_scale_sample_std_mean": float(sample["std"].mean()),
        "gate_log_scale_sample_max_abs": float(max(np.abs(sample["min"]).max(), np.abs(sample["max"]).max())),
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
