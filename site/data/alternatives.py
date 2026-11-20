"""Write alternatives.json, the tables of the Alternative attention mechanisms pages, from the runs' records.

    python3 site/data/alternatives.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
from pathlib import Path

from cells import number, text
from polynomial import ROOT, best
from scaling import change

EXP = "r6a76e7fbde45"
SCORE = "<var>s</var>"
EXP_SCORE = f"<var>e</var><sup>{SCORE}</sup>"
# Exp of the score s with a prefactor, or of softplus(s): (score weight, run).
EXP_VARIANTS = (
    (f"<var>σ</var>({SCORE})&#8239;{EXP_SCORE}", "rd7c9f610e9dc"),
    (f"softplus({SCORE})&#8239;{EXP_SCORE}", "rbd24b948a240"),
    (f"<var>e</var><sup>softplus({SCORE})</sup> = 1 + {EXP_SCORE}", "r0dd96f88d9f8"),
)


def exp_variants() -> dict:
    rows = []
    for weight, run_id in EXP_VARIANTS:
        cell = number(best(run_id), 4, run_id)
        cell["text"] += f' <span class="change">({change(cell["value"], best(EXP))})</span>'
        rows.append([text(weight), cell])
    return {"key": "exp-variants", "title": "Exponential score variants",
            "tables": [{"head": ["Score weight", "PPL (% vs exp)"], "compact": True, "rows": rows}]}


def main() -> None:
    data = {"fixed-kernels": [exp_variants()]}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
