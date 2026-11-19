"""Write other.json, the Other runs page's data, from the runs' records and result summaries.

    python3 site/data/other.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
from pathlib import Path

from polynomial import RESULTS, ROOT, RUNS, best, rows

EVEN = RESULTS / "pretraining/other/even_powers/summary.csv"
# Selected powers: columns (label, degree) and each row's summary role per column (None: not run).
POWER_COLUMNS = (("Full", 4), ("Even-only", 8), ("Odd + constant", 7), ("Full", 8))
POWER_ROWS = (
    ("Static", ("static_full_m4", "static_even_m8", None, "static_full_m8")),
    ("Query + W decay", ("query_full_m4_w_decay", "query_even_m8_w_decay", "query_odd_m7_w_decay", "query_full_m8_w_decay")),
)
GATES = ("r1a5d217f030d", "r77698776d82c", "r8c4a6ecdf4d2")
GATE_LABELS = {"gdn": "GDN-style", "sigmoid": "Sigmoid"}
# Coefficient-input L2: per variant, (total degree, (run after RMSNorm, run on the raw input)).
L2_COLUMNS = ("L2 after RMSNorm", "L2 on raw input")
L2_GROUPS = (
    ("Query", ((8, ("r25a5f78841b2", None)), (12, ("r7e34dec9dbd5", "r1f30febd7de5")))),
    ("SoS", ((8, ("r717267afcb78", None)), (12, ("r6b17c3463a60", "r2bf71d7b4831")))),
    ("SoS + softplus input + prefactor", ((9, (None, "r35805940a380")), (13, (None, "r5198d688e9d4")))),
)
# Decay and LR experiments: rows of (label, run IDs per column; None: not run).
DECAY = (("Query", ("r2b22f2ff6461", "rca97cc9db8f2", "r68cac6f03d77")), ("Query-key", ("r501f5cca0d4c", "r79d829a16029", None)))
LEARNING_RATE = (("Query", ("r2b22f2ff6461", "rb74fa2ebede6")),)
RELEASE = (
    ("All degrees at once", "rd40b46f5dc9b"),
    ("One degree at a time (0 → 8)", "rcc4f4e5d7abb"),
    ("Query + W decay, no freeze", "r25fffc16c2f3"),
    ("Exp baseline", "r6a76e7fbde45"),
)
ZERO_CONSTANT = (
    ("Query + softplus input", "rb5b604ae829a"),
    ("Query SoS + softplus input", "r7d5baf55104e"),
    ("Static + softplus input", "rc28ef4dbe26d"),
)


def cell(run_id: str | None) -> dict | None:
    return {"run_id": run_id, "perplexity": best(run_id)} if run_id else None


def powers() -> dict:
    summary = {row["role"]: row for row in rows(EVEN)}
    return {
        "title": "Selected powers",
        "columns": [{"label": label, "degree": degree} for label, degree in POWER_COLUMNS],
        "rows": [{"label": label, "cells": [
            {"run_id": summary[role]["run_id"], "perplexity": float(summary[role]["best_validation_perplexity"])}
            if role else None for role in roles]} for label, roles in POWER_ROWS],
    }


def gates() -> dict:
    entries = []
    for run_id in GATES:
        model = json.loads((RUNS / run_id / "config.json").read_text())["model"]
        entries.append({"degree": model["attention_degree"], "gate": GATE_LABELS[model["attention_coefficient_gate"]],
                        "run_id": run_id, "perplexity": best(run_id)})
    return {
        "title": "Per-degree coefficient gates",
        "rows": entries,
    }


def input_norm() -> dict:
    return {
        "title": "Coefficient input L2",
        "columns": list(L2_COLUMNS),
        "groups": [{"label": label, "rows": [{"degree": degree, "cells": [cell(run_id) for run_id in runs]}
                                             for degree, runs in entries]} for label, entries in L2_GROUPS],
    }


def zero_constant() -> dict:
    return {
        "title": "Zero constant",
        "rows": [{"label": label, **cell(run_id)} for label, run_id in ZERO_CONSTANT],
    }


def release_update(run_id: str) -> str:
    """The update at which a run's learned coefficients start training ("—" when it learns none)."""
    config = json.loads((RUNS / run_id / "config.json").read_text())
    if config["model"].get("attention_coefficients") != "learned":
        return "—"
    start, end = config["optim"].get("coefficient_release_step"), config["optim"].get("coefficient_release_end_step")
    if start is None:
        return "1"
    return f"{start + 1:,}–{end + 1:,}" if end is not None else f"{start + 1:,}"


def decay_and_lr() -> dict:
    return {
        "title": "Decay and LR experiments",
        "parts": [
            {"key": "weight-decay", "title": "Coefficient weight decay",
             "head": ["Attention", "Without <var>W</var> decay", "With <var>W</var> decay (0.1)",
                      "With <var>a</var>, <var>b</var>, <var>W</var> decay (0.1)"],
             "rows": [[label, *(cell(run_id) for run_id in runs)] for label, runs in DECAY]},
            {"key": "learning-rate", "title": "Coefficient learning rate",
             "head": ["Attention", "Coefficient LR ×1", "Coefficient LR ×3"],
             "rows": [[label, *(cell(run_id) for run_id in runs)] for label, runs in LEARNING_RATE],
             "compact": True},
            {"key": "release", "title": "Taylor warmup, then query + W decay",
             "head": ["Run", "Release update", "PPL"],
             "rows": [[label, release_update(run_id), cell(run_id)] for label, run_id in RELEASE],
             "compact": True},
        ],
    }


def main() -> None:
    data = {"powers": powers(), "gates": gates(), "input_norm": input_norm(), "zero_constant": zero_constant(),
            "decay_and_lr": decay_and_lr()}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
