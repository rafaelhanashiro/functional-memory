"""Write alternatives.json, the tables of the Alternative attention mechanisms pages, from the runs' records.

    python3 site/data/alternatives.py

Every value is a run's validation-best perplexity. Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
from pathlib import Path

from cells import number, text
from polynomial import ROOT, RUNS, best
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


def main() -> None:
    data = {"fixed-kernels": fixed_kernels(), "mlp": mlp()}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
