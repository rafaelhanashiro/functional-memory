"""Write alternatives.json, the tables of the Alternative attention mechanisms pages, from the runs' records.

    python3 site/data/alternatives.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from cells import number, text
from polynomial import RESULTS, ROOT, RUNS, best, rows
from scaling import change

EXP = "r6a76e7fbde45"
SCORE = "<var>s</var>"
EXP_SCORE = f"<var>e</var><sup>{SCORE}</sup>"
# Standalone alternatives to exp: (attention, its score weight, run).
ALTERNATIVES = (
    ("Linear", SCORE, "rd434701f3492"),
    ("GDN", "Native FLA Gated DeltaNet", "r6b38709ed8d3"),
    ("Sigmoid", f"<var>σ</var>({SCORE})", "r88d8654a8a1d"),
    ("Softplus", f"log(1 + {EXP_SCORE})", "r266c5dd1f935"),
)
# Exp with a prefactor of the score: (score weight, run).
PREFACTORS = (
    (f"<var>σ</var>({SCORE})&#8239;{EXP_SCORE}", "rd7c9f610e9dc"),
    (f"softplus({SCORE})&#8239;{EXP_SCORE}", "rbd24b948a240"),
)
# A run that fits neither: exp of softplus(s).
MISC = ((f"<var>e</var><sup>softplus({SCORE})</sup> = 1 + {EXP_SCORE}", "r0dd96f88d9f8"),)
# The score MLP, a learned SiLU MLP of the score and the query token in place of exp: (width J, run).
SCORE_MLP = (("<var>J</var> = 4", "rea8b942672f4"), ("<var>J</var> = 8", "rfc68c6d26f40"))
# Degree-4 splines of the score z: (initial function, static run, query + W decay run), against the
# degree-4 polynomial controls in the same two columns.
Z = "<var>z</var>"
SPLINES = (
    (f"Taylor: <var>T</var><sub>4</sub>({Z})", "r7e4193fcafbd", "rf303a5f94487"),
    (f"Affine: 1 + {Z}/4", "r325f7d64b940", "r2de7f1f4c9ce"),
    (f"Quadratic: 1 + {Z} + {Z}<sup>2</sup>/2", "r3ba9a42b7abd", "r2c5883395415"),
    (f"Binomial: (1/2 + {Z}/8)<sup>4</sup>", "ra4c3ada624af", "r78fa0d682f44"),
)
SPLINE_CONTROLS = ("r4339a245820e", "rca97cc9db8f2")
# Knot calibration: the controls' scores at initialization and update 7,500, from the spline collection.
SPLINE_RESULTS = RESULTS / "pretraining/alternative_attention/splines/spline_attention"
KNOT_ROLES = (("static", "Static"), ("query_wdecay", "Query + <var>W</var> decay"))
KNOT_STATES = (("initialization", "initialization"), ("checkpoint", "update 7,500"))
KNOT_RADII = (12, 24)


def shape(run_id: str) -> str:
    """Layers × width, from the run's recipe (GDN names its layers num_hidden_layers)."""
    model = json.loads((RUNS / run_id / "config.json").read_text())["model"]
    return f"{model.get('num_layers') or model['num_hidden_layers']} × {model['hidden_size']}"


def versus_exp(rows: tuple) -> list[list[dict]]:
    table = []
    for weight, run_id in rows:
        cell = number(best(run_id), 4, run_id)
        cell["text"] += f' <span class="change">({change(cell["value"], best(EXP))})</span>'
        table.append([text(weight), cell])
    return table


def fixed_kernels() -> list[dict]:
    head = ["Score weight", "PPL (% vs exp)"]
    return [
        {"key": "alternatives", "title": "Alternatives to exp",
         "tables": [{"head": ["Attention", "Score weight", "Layers × width", "PPL"],
                     "rows": [[text(name), text(weight), text(shape(run_id)), number(best(run_id), 4, run_id)]
                              for name, weight, run_id in ALTERNATIVES],
                     "caption": f"{SCORE} is the attention score. Linear, sigmoid, and softplus use per-head output RMSNorm."}]},
        {"key": "prefactors", "title": "Exp with prefactors",
         "tables": [{"head": head, "compact": True, "rows": versus_exp(PREFACTORS)}]},
        {"key": "misc", "title": "Misc", "tables": [{"head": head, "compact": True, "rows": versus_exp(MISC)}]},
    ]


def mlp() -> list[dict]:
    return [{"key": "score-mlp", "title": "Score MLP",
             "tables": [{"head": ["Width", "PPL (% vs exp)"], "compact": True, "rows": versus_exp(SCORE_MLP)}]}]


def splines() -> list[dict]:
    controls = [best(run_id) for run_id in SPLINE_CONTROLS]
    rows = []
    for label, *runs in SPLINES:
        cells = [number(best(run_id), 4, run_id) for run_id in runs]
        for cell, control in zip(cells, controls):
            cell["text"] += f' <span class="change">({change(cell["value"], control)})</span>'
        rows.append([text(label), *cells])
    rows.append({"className": "divider",
                 "cells": [text("Polynomial baseline"), *(number(best(run_id), 4, run_id) for run_id in SPLINE_CONTROLS)]})
    return [{"key": "splines", "title": "Degree-4 splines",
             "tables": [{"head": ["Initialization", "Static", "Query + <var>W</var> decay"], "rows": rows,
                         "caption": "PPL (% vs polynomial baseline)."}]}]


def share(fraction: str) -> str:
    """A share in percent: two decimals from 1%, three significant figures below."""
    percent = 100 * float(fraction)
    return "0%" if percent == 0 else f"{percent:.2f}%" if percent >= 1 else f"{percent:.3g}%"


def knots() -> dict:
    summary = {(row["role"], row["state"]): row for row in rows(SPLINE_RESULTS / "summary.csv")}
    coverage = {(row["role"], row["state"], int(row["radius"])): row for row in rows(SPLINE_RESULTS / "coverage.csv")}
    minus = lambda value: f"{float(value):.2f}".replace("-", "−")
    table = []
    for state, when in KNOT_STATES:
        for role, label in KNOT_ROLES:
            scores = summary[(role, state)]
            table.append([text(f"{label}, {when}"), text(f"{float(scores['score_std']):.3f}"),
                          text(f"{minus(scores['score_min'])} to {minus(scores['score_max'])}"),
                          *(text(share(coverage[(role, state, radius)]["head_mean_query_absolute_weight_share_outside"]))
                            for radius in KNOT_RADII)])
    figures = Path(__file__).resolve().parents[1] / "figures" / "splines"
    figures.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SPLINE_RESULTS / "score_coverage.png", figures / "score-coverage.png")
    return {"key": "knots", "title": "Knot calibration",
            "tables": [{"head": ["Scores", "Std", "Range", *(f"Weight outside ±{radius}" for radius in KNOT_RADII)],
                        "rows": table,
                        "caption": "Attention scores of the two degree-4 controls on 16 training sequences; weight: "
                                   "the mean share of a query's absolute attention weight."}],
            "figures": [{"src": "figures/splines/score-coverage.png",
                         "alt": "Score density and the absolute-weight share outside each candidate knot range"}]}


def main() -> None:
    data = {"fixed-kernels": fixed_kernels(), "mlp": mlp(), "splines": [*splines(), knots()]}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
