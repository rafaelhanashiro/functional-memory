"""Audit coefficient-fit measurements without changing fitting inputs or checkpoints.

Normal regeneration requires pinned scientific inputs. --allow-running produces
an explicitly provisional snapshot while jobs append metrics; it still uses the
same explicit run roles and records the exact measured-prefix SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ROLES = {
    "query_generator_full_batch_m4_head_scale": "query",
    "free_coefficients_full_batch_m4_head_scale": "free",
    "static_coefficients_full_batch_m4_head_scale": "static",
}
QUANTILES = [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]


def quantiles(values):
    return np.quantile(np.asarray(values, dtype=np.float64), QUANTILES).tolist()


def measurement(row):
    return {name: row[name] for name in ("step", "nll", "perplexity", "learning_rate")}


def window(rows, threshold, independent_sequences):
    lower = np.asarray([row["gradient_norm_min"] for row in rows])
    upper = np.asarray([row["gradient_norm_max"] for row in rows])
    losses = np.asarray([row["nll"] for row in rows])
    deltas = np.diff(losses)
    any_clip, all_clip = upper > threshold, lower > threshold
    result = {
        "first_step": rows[0]["step"], "last_step": rows[-1]["step"], "measurement_count": len(rows),
        "initial_nll": float(losses[0]), "final_nll": float(losses[-1]),
        "best_nll_improvement": rows[0]["best_nll"] - rows[-1]["best_nll"],
        "gradient_norm_max_quantiles": quantiles(upper),
        "gradient_norm_min_quantiles": quantiles(lower),
        "steps_with_any_gradient_clipped_fraction": float(any_clip.mean()),
        "steps_with_all_gradients_clipped_fraction": float(all_clip.mean()),
        "loss_increase_fraction": float(np.mean(deltas > 0)) if len(deltas) else None,
        "loss_step_difference_quantiles": quantiles(deltas) if len(deltas) else None,
    }
    if independent_sequences:
        count = int(rows[0]["sequence_count"])
        # Only extrema are logged: do not invent a pooled per-sequence clipping rate.
        result["per_sequence_clipping_fraction_bounds"] = [
            float(np.mean(np.where(all_clip, count, any_clip.astype(int))) / count),
            float(np.mean(np.where(all_clip, count, any_clip.astype(int) * (count - 1))) / count),
        ]
    else:
        result["global_gradient_clipping_fraction"] = float(any_clip.mean())
        result["global_clip_multiplier_quantiles"] = quantiles(np.minimum(1, threshold / np.maximum(upper, 1e-30)))
    if rows[0].get("generator_gradient_norms"):
        result["generator_groups"] = {
            name: {
                "norm_quantiles": quantiles([row["generator_gradient_norms"][name] for row in rows]),
                "squared_gradient_share_quantiles": quantiles([
                    row["generator_gradient_norms"][name] ** 2 / max(row["gradient_norm_max"] ** 2, 1e-30)
                    for row in rows]),
            }
            for name in rows[0]["generator_gradient_norms"]
        }
    return result


def collect(entry, *, allow_running):
    run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
    record = json.loads((run / "record.json").read_text())
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    payload = (run / "metrics.jsonl").read_bytes()
    if not payload.endswith(b"\n"):
        if not allow_running:
            raise ValueError(f"Incomplete metric line in {run}")
        payload = payload[:payload.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    rows = [row for row in rows if row.get("event") == "train"]
    if not rows or [row["step"] for row in rows] != list(range(rows[-1]["step"] + 1)):
        raise ValueError(f"Missing, duplicate, or reordered fitting steps in {run}")
    for row in rows:
        values = [row[name] for name in ("nll", "perplexity", "gradient_norm_min", "gradient_norm_max")]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite measurement at {run.name}:{row['step']}")
        if not math.isclose(math.exp(row["nll"]), row["perplexity"], rel_tol=1e-10):
            raise ValueError("PPL does not equal exp(mean NLL)")
        if not math.isclose(np.mean(row["per_sequence_nll"]), row["nll"], abs_tol=1e-6):
            raise ValueError("Batch NLL does not equal the mean of per-sequence NLL")
    oracle = recipe["oracle"]
    independent = oracle["parameterization"] == "free_per_example_query_coefficients"
    result = {
        "run_id": run.name, "status": record["status"], "provisional": record["status"] != "complete",
        "parent_run_id": oracle["parent_run_id"], "parameterization": oracle["parameterization"],
        "measured_through_utc": rows[-1]["time_utc"], "initial": measurement(rows[0]),
        "latest": measurement(rows[-1]), "best_fitting": measurement(min(rows, key=lambda row: row["nll"])),
        "initialization": oracle["initialization"], "sequence_count": int(oracle["sequence_count"]),
        "gradient_norm_scope": "per-step extrema over independent per-sequence gradients" if independent else "one full-batch mean gradient",
        "clip_norm": oracle["clip_norm"],
        "source_metrics": {"path": str((run / "metrics.jsonl").relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest()},
        "source_recipe": {"path": str((run / "config.yaml").relative_to(ROOT)), "sha256": hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest()},
        "windows": {},
    }
    windows = {"all": rows, "first_500": rows[:500], "last_500": rows[-500:]}
    for lr in sorted({row["learning_rate"] for row in rows}, reverse=True):
        windows[f"lr_{lr:g}"] = [row for row in rows if row["learning_rate"] == lr]
    for name, selected in windows.items():
        result["windows"][name] = window(selected, float(oracle["clip_norm"]), independent)
    initial = np.asarray(rows[0]["per_sequence_nll"])
    latest = np.asarray(rows[-1]["per_sequence_nll"])
    result["sequence_results"] = {
        "initial_nll_quantiles": quantiles(initial), "latest_nll_quantiles": quantiles(latest),
        "nll_reduction_quantiles": quantiles(initial - latest),
        "fraction_improved": float(np.mean(latest < initial)),
    }
    by_step = {row["step"]: row for row in rows}
    result["largest_gradient_events"] = [
        {"step": row["step"], "nll": row["nll"], "gradient_norm_max": row["gradient_norm_max"],
         "next_nll_change": (by_step[row["step"] + 1]["nll"] - row["nll"] if row["step"] + 1 in by_step else None)}
        for row in sorted(rows, key=lambda row: row["gradient_norm_max"], reverse=True)[:10]
    ]
    history = [row["best_nll"] for row in rows]
    span = int(oracle["plateau_window"])
    result["recent_best_nll_improvement"] = history[-span - 1] - history[-1] if len(history) > span else None
    result["plateau_tolerance"] = oracle["plateau_tolerance"]
    result["plateau_window"] = span
    if rows[0].get("generator_gradient_norms"):
        result["group_norm_squared_sum_max_relative_error"] = max(abs(
            sum(v * v for v in row["generator_gradient_norms"].values()) / row["gradient_norm_max"] ** 2 - 1) for row in rows)
    completion = run / "completion.json"
    if completion.exists():
        value = json.loads(completion.read_text())
        result["completion"] = {key: value.get(key) for key in (
            "status", "step", "best_step", "learning_rate_reduced_at", "frozen_parameters_unchanged")}
        result["source_completion"] = {"path": str(completion.relative_to(ROOT)), "sha256": hashlib.sha256(completion.read_bytes()).hexdigest()}
    acceptance = json.loads((run / "oracle_acceptance.json").read_text())
    result["qualification"] = {key: acceptance.get(key) for key in (
        "status", "checkpoint_resume_exact", "max_logit_replay_error", "training_forward_max_nll_error",
        "improving_update", "qualification_nll_trace", "frozen_parameter_scope")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "optimization_audit.json")
    args = parser.parse_args()
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    entries = {entry["role"]: entry for entry in selection["runs"] if entry["role"] in ROLES}
    if set(entries) != set(ROLES):
        raise ValueError("Selection must name the three canonical-QK degree-four batch fits")
    if not args.allow_running:
        verify_pinned_runs(ROOT, list(entries.values()))
    result = {
        "schema_version": 1, "quantile_probabilities": QUANTILES,
        "interpretation": "Fixed-batch fitting, different source backbones; not a matched-backbone or generalization comparison.",
        "measurement_convention": "Row s evaluates coefficients after s updates, before applying update s+1. A plateau LR decision at s appears in logged LR from row s+1.",
        "runs": {ROLES[role]: collect(entry, allow_running=args.allow_running) for role, entry in entries.items()},
    }
    result["provisional"] = any(run["provisional"] for run in result["runs"].values())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({name: {"status": row["status"], "step": row["latest"]["step"], "ppl": row["latest"]["perplexity"]}
                      for name, row in result["runs"].items()}))


if __name__ == "__main__":
    main()
