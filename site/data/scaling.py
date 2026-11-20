"""Write scaling.json, the tables of the 2.5× Chinchilla, 4k sequences and 100M pages, from the runs' records.

    python3 site/data/scaling.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
from pathlib import Path

from cells import bold_lowest, number, text
from polynomial import ROOT, best

CHINCHILLA_REFERENCES = (("Softmax", "r8ccda981be2b"), ("Exp", "rfe12e5c36e94"), ("GDN", "r2ebc05346f44"))
# Per degree: (attention, run with W decay, run without; None: no run).
CHINCHILLA_POLYNOMIALS = (
    (4, (("Fixed Taylor", None, "r4f1682c1b54d"), ("Static", None, "raa81f92b6aaf"),
         ("Query", "r77b617af91db", "reb9b5be4ff1b"), ("Query-key", "rb6313357059e", "r751984d1ca00"))),
    (8, (("Fixed Taylor", None, "r0ea4c732c5ac"), ("Static", None, "r6c9bcd661078"),
         ("Query", "ra15cbb16beb7", "r77bca86ef253"), ("Query-key", "r5bd803b3e562", "rb7e83a4b653f"))),
)
# (attention, run on 2,048-token sequences, run on 4,096-token sequences)
SEQUENCES = (
    ("Exp", "r6a76e7fbde45", "r2047d29a0d9f"),
    ("Softmax", "r9040f12f3d80", "r636f433be44f"),
    ("Query + W decay, <var>m</var> = 4", "rca97cc9db8f2", "r738b35a8e314"),
    ("Fixed Taylor, <var>m</var> = 4", "r816ab0fbe119", "r339a31ad8d4b"),
    ("FoX NoPE + W decay, <var>m</var> = 4", "r4a80e28ce0a7", "r58efc973ff28"),
)

# The 100M page: the 12-layer width-640 shape, against the 15-layer width-896 one (200M) and at 1×,
# 2× and 4× its 1× Chinchilla budget. Each table groups its runs by attention, one row per degree,
# then lists the baselines without a degree; a row's runs follow the table's columns (None: no run;
# PLANNED: a planned run not yet submitted). Every value is compared with exp in the same column.
PLANNED = "planned"
SHAPES = (("100M", "12 layers × 640"), ("200M", "15 layers × 896"))
UPDATES = {1: 7551, 2: 15103, 4: 30205}  # the 100M budgets, each update 128 × 2,048 tokens
SHAPE_PAIRS = (
    ("Query tanh + W decay", ((4, ("r4cce9a985a70", "rca97cc9db8f2")), (8, ("r95a5691d293f", "r25fffc16c2f3")))),
    ("Learned static coefficients", ((4, ("r94e0f2b8cff9", "r4339a245820e")),)),
    ("Fixed Taylor", ((4, ("r2c432bd3a2a4", "r816ab0fbe119")),)),
    ("FoX query", ((4, ("r3d14b598d88f", "r4a80e28ce0a7")),)),
)
SHAPE_BASELINES = (
    ("Exp (baseline)", ("r3e2fa2532c23", "r6a76e7fbde45")),
    ("Softmax", ("rf81c96f76a1d", "r9040f12f3d80")),
    ("FoX exp", ("ra5dfd978d3bd", "r183f08346ddc")),
)
BUDGET_POLYNOMIALS = (
    ("Query tanh + W decay", ((8, ("r95a5691d293f", "r3759b2a57a54", None)),
                              (12, (None, "r0f6d929e5992", "r2f41d054e944")))),
    ("Learned static coefficients", ((8, (None, "r3cca4c13edab", None)),
                                     (12, (None, "r84f325593209", "r0f8122dd8439")))),
    ("Fixed Taylor", ((8, (None, "rd77db79142e6", None)), (12, (None, "rffab7c4eeac5", "rcace22b32f93")))),
    ("FoX query", ((8, (None, "r656e8e652b72", None)), (12, (None, "rbf05f7baab5e", PLANNED)))),
)
BUDGET_BASELINES = (
    ("Exp (baseline)", ("r3e2fa2532c23", "rd920368a922e", "r333444bd4304")),
    ("Softmax", ("rf81c96f76a1d", "r3bb5b8bd5c6a", "r93df1e48c8fd")),
    ("FoX exp", ("ra5dfd978d3bd", "re634d82ee479", PLANNED)),
)
SOS_SOFTPLUS = (
    ("Query + softplus input", ((8, ("r1d342ca6e58c", "rc6a7e50d27e1", None)),
                                (12, (None, "r53d5bbcac16f", "rb74e8b994b74")))),
    ("Query SoS", ((8, (None, "rccd2cb5e8bab", None)), (12, (None, "r08be7093202a", "r1f935f18fe21")))),
    ("Query SoS + softplus input", ((12, ("r5a18841b21b5", "r83fa09df3931", "r0623d3020268")),)),
    ("Query SoS + softplus input + prefactor", ((13, ("r3b9c8cc6fab8", "re7593f1d0a4c", "r4589172dcd8b")),)),
)
# Other runs at 1×, each changing the first, query tanh + W decay, in one way: (attention, degree, run).
OTHER = (
    ("Query tanh + W decay", 8, "r95a5691d293f"),
    ("+ static per-degree biases", 8, "rb4e5457ef2a0"),
    ("+ query-dependent per-degree biases", 8, "r8dc6e147601c"),
    ("+ |<var>s</var>| input", 8, "r17a1290cb4ac"),
    ("Exp (baseline)", None, "r3e2fa2532c23"),
)
FOX = "FoX uses shared cumulative gates and no positional encoding."
VERSUS_EXP = "PPL (% vs exp)."


def perplexity(run_id: str | None) -> dict:
    return number(best(run_id), 4, run_id) if run_id else text("—")


def polynomial_row(label: str, decayed: str | None, plain: str | None) -> list[dict]:
    cells = [perplexity(decayed), perplexity(plain)]
    if decayed and plain:  # bold the better of the pair
        bold_lowest(cells)
    return [text(label), *cells]


def chinchilla() -> list[dict]:
    return [
        {"key": "references", "title": "Softmax, exp and GDN",
         "tables": [{"head": ["Attention", "PPL"], "compact": True,
                     "rows": [[text(label), perplexity(run_id)] for label, run_id in CHINCHILLA_REFERENCES]}]},
        {"key": "polynomials", "title": "Polynomial attention",
         "layout": "pair",
         "tables": [{"title": f"<var>m</var> = {degree}",
                     "head": ["Attention", "<var>W</var> decay", "No <var>W</var> decay"],
                     "rows": [polynomial_row(*row) for row in rows]} for degree, rows in CHINCHILLA_POLYNOMIALS]},
    ]


def sequences() -> list[dict]:
    rows = [[text(label), perplexity(short), perplexity(long)] for label, short, long in SEQUENCES]
    for column in (1, 2):
        bold_lowest([row[column] for row in rows])
    return [{
        "key": "sequences",
        "tables": [{"head": ["Attention", "2k", "4k"], "compact": True, "rows": rows}],
    }]


def change(value: float, reference: float) -> str:
    """The difference from a reference value in percent, signed as the page shows it."""
    if value == reference:
        return "0.00%"
    percent = (value / reference - 1) * 100
    return f"{'−' if percent < 0 else '+'}{abs(percent):.2f}%"


def against(run_id: str | None, reference: str) -> dict:
    """A run's perplexity with its difference from the reference run in parentheses."""
    if run_id is None:
        return text("—", muted=True)
    if run_id == PLANNED:
        return text("Unsubmitted", muted=True)
    cell = number(best(run_id), 4, run_id)
    cell["text"] += f' <span class="change">({change(best(run_id), best(reference))})</span>'
    return cell


def degree(value: int | None) -> dict:
    return text(f"<var>m</var> = {value}") if value is not None else text("—", muted=True)


def grouped(polynomials: tuple, baselines: tuple, references: tuple[str, ...]) -> list[dict]:
    """One row per attention and degree, alternate attentions shaded, then the baselines below a rule."""
    rows = []
    for index, (label, degrees) in enumerate(polynomials):
        for position, (value, runs) in enumerate(degrees):
            cells = [degree(value), *(against(run_id, reference) for run_id, reference in zip(runs, references))]
            if position == 0:
                cells.insert(0, {**text(label), "rowspan": len(degrees), "label": True})
            rows.append({"cells": cells, "className": "shade" if index % 2 else ""})
    for index, (label, runs) in enumerate(baselines, len(polynomials)):
        cells = [{**text(label), "label": True}, degree(None),
                 *(against(run_id, reference) for run_id, reference in zip(runs, references))]
        classes = ("shade" if index % 2 else "", "divider" if index == len(polynomials) else "")
        rows.append({"cells": cells, "className": " ".join(name for name in classes if name)})
    return rows


def budget_head(multiple: int) -> str:
    tokens = UPDATES[multiple] * 128 * 2048 / 1e9
    return f'{multiple}×<span class="sub">{tokens:.2f}B tokens</span>'


def hundred_m() -> list[dict]:
    budgets = ["Attention variant", "Degree", *(budget_head(multiple) for multiple in UPDATES)]
    exp = BUDGET_BASELINES[0][1]
    return [
        {"key": "shape-pairs", "title": "100M vs 200M",
         "tables": [{"head": ["Attention variant", "Degree",
                              *(f'{name}<span class="sub">{shape}</span>' for name, shape in SHAPES)],
                     "grouped": True, "rows": grouped(SHAPE_PAIRS, SHAPE_BASELINES, SHAPE_BASELINES[0][1]),
                     "caption": f"{VERSUS_EXP} Each shape at its 1× Chinchilla budget. {FOX}"}]},
        {"key": "scalings", "title": "Chinchilla scalings",
         "tables": [{"head": budgets, "grouped": True, "rows": grouped(BUDGET_POLYNOMIALS, BUDGET_BASELINES, exp),
                     "caption": f"{VERSUS_EXP} —: no run. Unsubmitted: a planned FoX run. {FOX}"}]},
        {"key": "sos-softplus", "title": "SoS and softplus",
         "tables": [{"head": budgets, "grouped": True, "rows": grouped(SOS_SOFTPLUS, BUDGET_BASELINES[:1], exp),
                     "caption": f"{VERSUS_EXP} —: no run. The SoS + softplus runs use a degree-12 core; the "
                                "prefactor raises the total degree to 13."}]},
        {"key": "other", "title": "Other runs",
         "tables": [{"head": ["Attention variant", "Degree", budget_head(1), "vs control"], "compact": True,
                     "rows": other(),
                     "caption": "Each run changes the control, query tanh + W decay "
                                "at degree 8, in one way. Per-degree biases shift each order <var>r</var>'s "
                                "score, (<var>s</var> + <var>β</var><sub><var>r</var></sub>)<sup><var>r</var></sup>, "
                                "by a learned constant (static) or by an amount that depends on the query token; "
                                "|<var>s</var>| input evaluates the polynomial at |<var>s</var>| in place of "
                                "<var>s</var>."}]},
    ]


def other() -> list[list[dict]]:
    control = best(OTHER[0][2])
    rows = []
    for position, (label, value, run_id) in enumerate(OTHER):
        versus = text("—", muted=True) if position == 0 else text(change(best(run_id), control))
        rows.append([text(label), degree(value), number(best(run_id), 4, run_id), versus])
    return rows


def main() -> None:
    data = {"chinchilla": chinchilla(), "sequences": sequences(), "100m": hundred_m()}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
