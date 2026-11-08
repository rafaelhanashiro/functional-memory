"""Regenerate the selected FoX/query-distance comparison from pinned run files."""
from __future__ import annotations

import csv
from copy import deepcopy
import json
import math
from pathlib import Path

import yaml

from scaffolds.records.selections import verify_pinned_runs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"
ROLES = ("control", "query_distance", "query_key_distance", "bounded_query_distance",
         "per_order_query", "per_order_query_key")
STEPS = 7570


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def scientific_recipe(recipe: dict) -> dict:
    """Ignore execution names/paths, the attention implementation (blocked reference or Mosaic
    kernels, which compute the same weights; the blocked scan or its causal strips, the same
    arithmetic), and explicit inactive historical defaults only."""
    recipe = deepcopy(recipe)
    model = recipe["model"]
    if model.get("attention_backend") not in ("blocked", "mosaic"):
        raise ValueError("Selected runs use the blocked reference or the Mosaic kernels")
    model.pop("attention_backend")
    model.pop("attention_blocked_implementation", None)
    if "attention_coefficient_key_kernel_weight_decay" in model:
        if model["attention_coefficient_key_kernel_weight_decay"] is not False:
            raise ValueError("This query-only control must have no key-projection decay")
        model.pop("attention_coefficient_key_kernel_weight_decay")
    model.setdefault("attention_forget_gate_mode", "cumulative")
    model.setdefault("qk_norm_token_scale_kernel_weight_decay", False)
    model.setdefault("attention_coefficient_orders", None)
    model.setdefault("attention_forget_gate_form", "input")
    model.setdefault("attention_forget_gate_conditioning", "query")
    model.setdefault("attention_forget_gate_rate_activation", "softplus")
    model.setdefault("attention_forget_gate_per_order", False)
    recipe["continuation"].setdefault("preserve_all_parameters", False)
    for key in ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir"):
        recipe["train"].pop(key, None)
    return recipe


def check_recipes(runs: dict) -> None:
    control = scientific_recipe(runs["control"]["config"])
    variant = scientific_recipe(runs["query_distance"]["config"])
    if control["model"]["attention_forget_gate_mode"] != "cumulative":
        raise ValueError("The control must use cumulative FoX decay")
    if variant["model"]["attention_forget_gate_mode"] != "query_distance":
        raise ValueError("The variant must use query-dependent distance decay")
    variant["model"]["attention_forget_gate_mode"] = "cumulative"
    if control != variant:
        raise ValueError("The selected runs differ beyond their declared decay rule and execution names/paths")
    for role, field, value in (
            ("query_key_distance", "attention_forget_gate_conditioning", "query_key"),
            ("bounded_query_distance", "attention_forget_gate_rate_activation", "sigmoid")):
        candidate = scientific_recipe(runs[role]["config"])
        expected = scientific_recipe(runs["query_distance"]["config"])
        expected["model"][field] = value
        if candidate != expected:
            differences = [f"{section}.{key}" for section in set(candidate) | set(expected)
                           for key in set(candidate.get(section, {})) | set(expected.get(section, {}))
                           if candidate.get(section, {}).get(key) != expected.get(section, {}).get(key)]
            raise ValueError(f"{role} differs beyond its declared gate change: {differences}")
    if (control["optim"]["total_steps"] != STEPS or control["train"]["seed"] != 42
            or control["model"]["max_seq_len"] != 2048 or control["model"]["attention_degree"] != 4
            or control["model"]["positional_encoding"] != "none"):
        raise ValueError("The selected control does not match this comparison's declared recipe")
    for role, reference in (("per_order_query", "query_distance"),
                            ("per_order_query_key", "query_key_distance")):
        candidate = scientific_recipe(runs[role]["config"])
        expected = scientific_recipe(runs[reference]["config"])
        expected["model"]["attention_forget_gate_per_order"] = True
        if candidate != expected:
            raise ValueError(f"{role} differs beyond independent per-order gating")


def measurement(row: dict) -> tuple[float, float]:
    nll = float(row["validation_nll"])
    ppl = float(row.get("validation_perplexity", math.exp(nll)))
    if not math.isfinite(nll) or not math.isfinite(ppl) or not math.isclose(ppl, math.exp(nll), rel_tol=1e-9):
        raise ValueError("Invalid or inconsistent validation NLL/PPL")
    return nll, ppl


def collect(entry: dict) -> dict:
    directory = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
    record = read_json(directory / "record.json")
    config = yaml.safe_load((directory / "config.yaml").read_text())
    row = {"role": entry["role"], "label": entry["label"], "run_id": entry["run_id"],
           "status": record["status"], "config": config}
    metrics_path = directory / "metrics.jsonl"
    metrics = ([json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
               if metrics_path.exists() else [])
    validation = [item for item in metrics if item.get("validation_nll") is not None
                  and math.isfinite(float(item["validation_nll"]))]
    best_path = directory / "best_validation.json"
    best = read_json(best_path) if best_path.exists() else (
        min(validation, key=lambda item: (float(item["validation_nll"]), -int(item["step"]))) if validation else None)
    if best is not None:
        best_nll, best_ppl = measurement(best)
        best_step = int(best["step"])
        if not 0 <= best_step <= STEPS:
            raise ValueError("Best validation step is outside the declared training budget")
        if validation:
            expected = min(validation, key=lambda item: (float(item["validation_nll"]), -int(item["step"])))
            if (int(expected["step"]) != best_step
                    or not math.isclose(float(expected["validation_nll"]), best_nll, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("The best-validation record disagrees with the canonical measurements")
        row.update(best_step=best_step, best_nll=best_nll, best_ppl=best_ppl)
    completion_path = directory / "completion.json"
    if record["status"] == "complete":
        if not completion_path.exists() or best is None:
            raise ValueError("A completed run must have completion and validation-best measurements")
        completion = read_json(completion_path)
        if int(completion["final_step"]) != STEPS or record.get("step", STEPS) != STEPS:
            raise ValueError("A completed comparison run must reach all 7,570 updates")
        terminal_nll, terminal_ppl = measurement(completion)
        terminal = [item for item in validation if int(item["step"]) == STEPS]
        if terminal and any(not math.isclose(float(item["validation_nll"]), terminal_nll,
                                             rel_tol=0, abs_tol=1e-12) for item in terminal):
            raise ValueError("Terminal completion NLL disagrees with the canonical measurements")
        row.update(terminal_step=STEPS, terminal_nll=terminal_nll, terminal_ppl=terminal_ppl)
    elif completion_path.exists():
        raise ValueError("A completion record exists for a run whose authoritative status is not complete")
    return row


def format_value(value, digits=6) -> str:
    if value is None:
        return ""
    return str(value) if isinstance(value, (str, int)) else f"{value:.{digits}f}"


def main() -> None:
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    entries = selection["runs"]
    if len(entries) != len(ROLES) or {entry["role"] for entry in entries} != set(ROLES):
        raise ValueError("Select one run for every declared decay comparison role")
    verify_pinned_runs(ROOT, entries)
    runs = {entry["role"]: collect(entry) for entry in entries}
    check_recipes(runs)
    complete = all(runs[role]["status"] == "complete" for role in ("control", "query_distance"))
    if complete:
        control, variant = runs["control"], runs["query_distance"]
        for metric in ("best_ppl", "terminal_ppl"):
            variant[f"delta_{metric}"] = variant[metric] - control[metric]
            variant[f"percent_{metric}"] = 100.0 * (variant[metric] / control[metric] - 1.0)
    for role, reference_role in (("query_key_distance", "query_distance"),
                                 ("bounded_query_distance", "query_distance"),
                                 ("per_order_query", "query_distance"),
                                 ("per_order_query_key", "query_key_distance")):
        row = runs[role]
        reference = runs[reference_role]
        if row["status"] == reference["status"] == "complete":
            for metric in ("best_ppl", "terminal_ppl"):
                row[f"delta_{metric}"] = row[metric] - reference[metric]
                row[f"percent_{metric}"] = 100.0 * (row[metric] / reference[metric] - 1.0)
    columns = ["role", "label", "run_id", "status", "best_step", "best_nll", "best_ppl",
               "terminal_step", "terminal_nll", "terminal_ppl", "delta_best_ppl", "percent_best_ppl",
               "delta_terminal_ppl", "percent_terminal_ppl"]
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for role in ROLES:
            writer.writerow({key: format_value(runs[role].get(key), 9 if key.endswith("nll") else 6)
                             for key in columns})
    lines = ["| Variant | Run | Status | Best validation PPL | Best update | Terminal PPL |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for role in ROLES:
        row = runs[role]
        lines.append(f"| {row['label']} | `{row['run_id']}` | {row['status']} | "
                     f"{format_value(row.get('best_ppl'), 4) or '—'} | {row.get('best_step', '—')} | "
                     f"{format_value(row.get('terminal_ppl'), 4) or '—'} |")
    if complete:
        variant = runs["query_distance"]
        lines.extend(["", "Query-dependent minus cumulative FoX PPL (negative favors query-dependent decay):",
                      f"validation-best **{variant['delta_best_ppl']:+.4f}** "
                      f"(**{variant['percent_best_ppl']:+.3f}%**); "
                      f"terminal **{variant['delta_terminal_ppl']:+.4f}** "
                      f"(**{variant['percent_terminal_ppl']:+.3f}%**)."])
    else:
        lines.extend(["", "The query-dependent run is not complete; a final PPL comparison is pending."])
    lines.extend(["", "Endpoint query–key and bounded-rate variants compare against the query-dependent control.",
                  "Each per-order variant compares against its corresponding shared-gate control.",
                  "Incomplete runs have no terminal PPL; their best-so-far metrics do not establish a final comparison."])
    path = HERE / "README.md"
    text = path.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("The README must contain exactly one generated-results region")
    before, remainder = text.split(BEGIN)
    _, after = remainder.split(END)
    path.write_text(before + BEGIN + "\n" + "\n".join(lines) + "\n" + END + after)


if __name__ == "__main__":
    main()
