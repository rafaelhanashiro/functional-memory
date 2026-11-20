"""Write parametrizations.json, the tables of the Activations page, from the runs' records.

    python3 site/data/parametrizations.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
from pathlib import Path

from cells import bold_lowest, number, text
from polynomial import ROOT, best


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


def main() -> None:
    data = {"activations": [forms(), pure_sigmoid()]}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
