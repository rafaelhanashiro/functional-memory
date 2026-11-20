"""Write parametrizations.json, the tables of the Activations, Dense and SoS / Softplus pages, from the
runs' records.

    python3 site/data/parametrizations.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
from pathlib import Path

from cells import bold_lowest, number, text
from polynomial import ROOT, best
from scaling import change


def indexed(name: str, index: str) -> str:
    return f"<var>{name}</var><sub><var>{index}</var></sub>"


FEATURE = f"{indexed('w', 'r')}<sup>⊤</sup>{indexed('x', 't')}"
# Degree-four coefficient generators: (generator, its coefficients c_r, run). The last two, without a
# generator, follow a rule.
FORMS = (
    ("Affine tanh (query)", f"({indexed('b', 'r')} + {indexed('a', 'r')} tanh({FEATURE})) / <var>r</var>!",
     "r2b22f2ff6461"),
    ("Affine sigmoid", f"({indexed('b', 'r')} + {indexed('a', 'r')} σ({FEATURE})) / <var>r</var>!", "r917f3650296e"),
    ("Pure tanh", f"tanh({FEATURE})", "r1d578d107115"),
    ("Learned static", f"{indexed('c', 'r')} learned", "r4339a245820e"),
    ("Fixed Taylor", "1 / <var>r</var>!", "r816ab0fbe119"),
)
# Per-degree pure sigmoid, c_r = σ(w_r^T x_t) or that over r!: (scaling, run without W decay, run with).
PURE_SIGMOID = (("None", "rd5550f8db5e4", "re6d4c820a132"), ("1 / <var>r</var>!", "r8b4e2a0603aa", "r343fff99fa9e"))
# Dense tanh query coefficients with RoPE: (K, degree, readout scale, run without W decay, run with; None:
# not run).
DENSE = (
    (5, 4, "1", "r0c0f3634685c", None),
    (5, 4, "1 / √5", "rcaff0dfb095f", "rdacb58b7981c"),
    (9, 4, "1 / 3", "r04a5b2a3e8dd", "r753a7e8d3212"),
    (9, 8, "1 / 3", "rfccffdd239dd", None),
)
# Dense sigmoid coefficients at degree 4: (scaling, run without W decay, run with); the per-degree ones
# are the pure sigmoid runs above.
DENSE_SIGMOID = (("None", "ra6ae537a5e37", "r331b5fd5bebc"),
                 ("1 / <var>r</var>! and 1 / √<var>K</var>", "r239ea241c978", "rf7e1a83e2e23"))


def forms() -> dict:
    rows = [{"cells": [text(label), text(coefficients), number(best(run_id), 4, run_id)],
             "className": "divider" if label == "Learned static" else ""} for label, coefficients, run_id in FORMS]
    bold_lowest([row["cells"][2] for row in rows])
    return {"key": "forms", "title": "Affine and pure generators",
            "tables": [{"head": ["Generator", f"Coefficients {indexed('c', 'r')}", "PPL"], "compact": True,
                        "rows": rows,
                        "caption": "Validation-best perplexity at degree <var>m</var> = 4. The affine forms start "
                                   "at fixed Taylor (<var>a</var> = 0); pure tanh starts from a random polynomial, "
                                   "<var>W</var> ∼ N(0, 1/<var>d</var>)."}]}


def pure_sigmoid() -> dict:
    rows = [[text(label), number(best(plain), 4, plain), number(best(decayed), 4, decayed)]
            for label, plain, decayed in PURE_SIGMOID]
    bold_lowest([cell for row in rows for cell in row[1:]])
    return {"key": "pure-sigmoid", "title": "Pure sigmoid, per degree",
            "tables": [{"head": ["Order scaling", "No <var>W</var> decay", "<var>W</var> decay 0.1"], "compact": True,
                        "rows": rows,
                        "caption": f"Validation-best perplexity at degree <var>m</var> = 4, with {indexed('c', 'r')} "
                                   f"= σ({FEATURE}), divided by <var>r</var>! or not, from <var>W</var> = 0."}]}


# SoS and softplus: per coefficient group, (variant, runs at core degree 4, 8 and 12; None: not run). A
# prefactor (*) raises the total degree to m + 1.
SOS = (
    ("Query coefficients · <var>W</var> decay", (
        ("SoS", ("r016b914d927a", "reeefbcc19abf", "r3d52b7ab43d1")),
        ("Softplus input", (None, "ra3b54e2489f8", "r9eb845aae5bb")),
        ("SoS + softplus input", (None, "r41a4c74c5f58", "rc5aac8f37a2e")),
        ("SoS + softplus input + prefactor*", (None, "r841f35a24241", "rf637de9b65b1")),
    )),
    ("Static coefficients", (
        ("SoS", (None, "r0e32dd2bd3f7", "rdc8b04b223af")),
        ("SoS + softplus input", ("r6d4217061453", "r54f771e54722", "r1b9e2e69f7f0")),
        ("SoS + softplus input + prefactor*", ("r91e0c1bb8458", "ra3bffee9a528", "r28a991faed4f")),
    )),
)
EXP = "r6a76e7fbde45"
FIXED_KERNELS = (("Exp (baseline)", EXP), ("Sigmoid", "r88d8654a8a1d"), ("Softplus", "r266c5dd1f935"))


def perplexity(run_id: str | None) -> dict:
    return number(best(run_id), 4, run_id) if run_id else text("—")


def dense() -> list[dict]:
    decay = ["Without <var>W</var> decay", "With <var>W</var> decay (0.1)"]
    sigmoid = [("Dense (<var>K</var> = 5)", *row) for row in DENSE_SIGMOID] + [("Per degree", *row) for row in PURE_SIGMOID]
    return [
        {"key": "dense", "title": "Dense polynomial parametrizations · query coefficients · RoPE",
         "tables": [{"head": ["Parametrization", "Degree", "Readout scale", *decay], "caption": "— Not run.",
                     "rows": [[text(f"Dense (<var>K</var> = {features})"), text(f"<var>m</var> = {degree}"), text(scale),
                               perplexity(plain), perplexity(decayed)]
                              for features, degree, scale, plain, decayed in DENSE]}]},
        {"key": "sigmoid", "title": "Sigmoid coefficients · <var>m</var> = 4 · RoPE",
         "tables": [{"head": ["Readout", "Scaling", *decay],
                     "caption": "Per degree: fixed <var>V</var> = <var>I</var>. Dense: learned <var>V</var>.",
                     "rows": [[text(readout), text(scaling), perplexity(plain), perplexity(decayed)]
                              for readout, scaling, plain, decayed in sigmoid]}]},
    ]


def sos() -> list[dict]:
    """Bold marks the lowest polynomial perplexity; every value shows its change from exp."""
    band = lambda label: {"className": "band", "cells": [{"text": label, "colspan": 4}]}
    rows, polynomials = [], []
    for group, variants in SOS:
        rows.append(band(group))
        for variant, runs in variants:
            cells = [perplexity(run_id) for run_id in runs]
            polynomials += cells
            rows.append([text(variant), *cells])
    bold_lowest(polynomials)
    rows.append(band("Fixed kernels"))
    rows += [[text(label), {**perplexity(run_id), "colspan": 3}] for label, run_id in FIXED_KERNELS]
    for row in rows:
        for cell in row if isinstance(row, list) else []:
            if "value" in cell:
                cell["text"] += f' <span class="change">({change(cell["value"], best(EXP))})</span>'
    return [{"key": "sos", "title": "SoS and softplus",
             "tables": [{"head": ["Variant", *(f"<var>m</var> = {degree}" for degree in (4, 8, 12))], "rows": rows,
                         "caption": "Core degree <var>m</var>, with the change from exp in parentheses. "
                                    "* Total degree <var>m</var> + 1."}]}]


def main() -> None:
    data = {"activations": [forms(), pure_sigmoid()], "dense": dense(), "sos": sos()}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
