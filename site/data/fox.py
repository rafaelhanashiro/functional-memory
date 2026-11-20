"""Write fox.json, the tables of the FoX / ALiBi / PaTH page, from the runs' records.

    python3 site/data/fox.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
from pathlib import Path

from cells import bold_lowest, number, text
from polynomial import ROOT, best

# Degree-4 runs: (attention, run without W decay, run with W decay 0.1; None: not run).
COMPARISON = (
    ("FoX NoPE", "r82a80d3091e2", "r4a80e28ce0a7"),
    ("FoX NoPE (static coefficients)", "r17fd7f040bff", None),
    ("FoX NoPE (query–key gate)", None, "r02f984133092"),
    ("FoX–PaTH", None, "r74495b47d226"),
    ("FoX NoPE (query gate)", None, "r3e26412b2811"),
    ("FoX NoPE (query gate, sigmoid)", None, "r4f169e5370da"),
    ("FoX", "r111843b4c7cc", None),
    ("ALiBi NoPE", "r8d4fc24ef537", "r69076c8f91c1"),
    ("PaTH", "r74a50aca921e", "r1f2c24c109f9"),
)
# Gates with query + W decay coefficients: (gate, {sharing: runs at degree 4, 8, 12}, exp run, softmax run).
GATES = (
    ("Query", {"Shared": ("r3e26412b2811", "r9320e5cd3071", "r47bfe39ce9b4"),
               "Per degree": ("r507193d47450", "r20fe6fe864bc", "rfe140c073b2f")}, "r17444ad0f60e", "r208110a25271"),
    ("Query–key", {"Shared": ("r02f984133092", None, None),
                   "Per degree": ("r03be769d2069", "r227b0ed7c605", "rc2f9273c49cc")}, "rd3501a40e020", None),
    ("FoX (cumulative)", {"Shared": ("r4a80e28ce0a7", "r9353f554b917", None),
                          "Per degree": ("r319b83140040", "rf8fc9ab2cecc", "r95bfff096cd4")}, "r183f08346ddc", "r5adc29a91e3b"),
)
# Gates with static coefficients: (gate, {sharing: runs at degree 4, 8}).
STATIC_GATES = (
    ("Static", {"Shared": ("r0e76396bdee5", "rc60345dfcff7"), "Per degree": ("r527a30ff7ed1", "re253a5f1b03e")}),
    ("Query–key", {"Shared": (None, None), "Per degree": ("ra0f402ff6758", "r2e25140c75db")}),
)
# Reference runs without a polynomial, none with RoPE: (attention, softmax run, exp run).
REFERENCES = (
    ("FoX NoPE", "r5adc29a91e3b", "r183f08346ddc"),
    ("FoX–PaTH", "r14db6f0e8592", "rafcf22b799ed"),
    ("ALiBi NoPE (fixed slopes)", "ref652c257797", "r7820049184ef"),
)
# The original FoX sigmoid gate, NoPE: (attention, QK normalization, run).
SIGMOID_GATE = (
    ("Softmax", "None", "reeb92d46a95d"),
    ("Exp + RMSNorm", "Per-head RMSNorm", "rdd3e9cb3bb80"),
    ("Query (<var>m</var> = 4) + <var>W</var> decay + RMSNorm", "Per-head RMSNorm", "rfa9636279199"),
    ("Softmax", "Per-head RMSNorm", "rac200ce11221"),
)
NOT_RUN = "—: not run."


def perplexity(run_id: str | None) -> dict:
    return number(best(run_id), 4, run_id) if run_id else text("—")


def grouped(groups: tuple) -> list[dict]:
    """A row per sharing under each gate, the gate's name and any baselines spanning its rows."""
    rows = []
    for index, (gate, sharings, *baselines) in enumerate(groups):
        for position, (sharing, runs) in enumerate(sharings.items()):
            cells = [text(sharing), *(perplexity(run_id) for run_id in runs)]
            if position == 0:
                cells.insert(0, {**text(gate), "rowspan": len(sharings), "label": True})
                cells += [{**perplexity(run_id), "rowspan": len(sharings)} for run_id in baselines]
            rows.append({"className": "shade" if index % 2 else "", "cells": cells})
    return rows


def comparison() -> dict:
    rows = [[text(label), perplexity(plain), perplexity(decayed)] for label, plain, decayed in COMPARISON]
    for column in (1, 2):
        bold_lowest([row[column] for row in rows])
    return {"key": "comparison", "title": "FoX, ALiBi and PaTH",
            "tables": [{"head": ["Attention", "Without <var>W</var> decay", "With <var>W</var> decay (0.1)"], "rows": rows,
                        "caption": f"{NOT_RUN} Static coefficients have no <var>W</var>; their gate projection keeps "
                                   "weight decay 0.1."}]}


def degrees(count: int) -> list[str]:
    return [f"<var>m</var> = {degree}" for degree in (4, 8, 12)[:count]]


def gates() -> dict:
    return {"key": "query-gates", "title": "Query, query–key and FoX gates · NoPE",
            "tables": [{"head": ["Gate", "Sharing", *degrees(3), "Exp", "Softmax"], "grouped": True, "rows": grouped(GATES),
                        "caption": f"{NOT_RUN} Exp and softmax have no degree; each spans both sharing rows."}]}


def static_gates() -> dict:
    return {"key": "static-gates", "title": "Static coefficients · distance gates · NoPE",
            "tables": [{"head": ["Gate", "Sharing", *degrees(2)], "grouped": True, "rows": grouped(STATIC_GATES),
                        "caption": f"{NOT_RUN} The static gate reads no token: <var>G</var><sub><var>t</var>,<var>i</var></sub> "
                                   "= (<var>t</var> − <var>i</var>) softplus(<var>β</var>)."}]}


def references() -> dict:
    """Bold marks each table's lowest perplexity."""
    kernels = [[text(label), perplexity(softmax), perplexity(exp)] for label, softmax, exp in REFERENCES]
    bold_lowest([cell for row in kernels for cell in row[1:]])
    sigmoid = [[text(label), text(qk), perplexity(run_id)] for label, qk, run_id in SIGMOID_GATE]
    bold_lowest([row[2] for row in sigmoid])
    return {"key": "reference", "title": "Reference FoX / PaTH",
            "tables": [{"head": ["Attention", "Softmax", "Exp"], "compact": True, "rows": kernels},
                       {"title": "Original FoX sigmoid gate · NoPE", "head": ["Attention", "QK normalization", "PPL"],
                        "compact": True, "rows": sigmoid}]}


def main() -> None:
    data = {"fox-alibi": [comparison(), gates(), static_gates(), references()]}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
