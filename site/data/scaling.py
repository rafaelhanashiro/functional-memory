"""Write scaling.json, the tables of the 2.5× Chinchilla and 4k sequences pages, from the runs' records.

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


def main() -> None:
    data = {"chinchilla": chinchilla(), "sequences": sequences()}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
