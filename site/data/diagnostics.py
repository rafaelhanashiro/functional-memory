"""Write diagnostics.json, the tables of the Overfitting and Continued pretraining pages, from the
runs' records, and copy the Length extrapolation figures from their result collection into figures/.

    python3 site/data/diagnostics.py

Rerun only when the user asks for newer results.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from cells import bold_lowest, number, text
from polynomial import RESULTS, ROOT, RUNS, best
from scaling import change

# (fit, batch size, start checkpoint, run)
FITS = (
    ("Free", 256, "Taylor", "rda6029c3286f"),
    ("Query-key", 256, "Query-key", "r043b150cde03"),
    ("Query-only", 256, "Query-only", "rc95f2055e96d"),
    ("Query-only", 1024, "Query-only", "r84e37a39e2a0"),
    ("Static", 256, "Static", "rceea84f19f6b"),
)
DENSE, QUERY_KEY = "rec777591a61b", "r594b3706ecd7"  # generators regressed onto free coefficients
# Continued pretraining: (attention, Taylor start, exp start, same-model start); None: not run,
# "" for models without a 1× checkpoint of their own.
CONTINUATIONS = (
    ("Static", "r2e081e09c19a", "r7e6a68545d4f", "r1522cd56901a"),
    ("Query + W decay", "rae91f24b06b4", "r673e14eab803", "rdfc42659424f"),
    ("Fixed Taylor", "r82203a85ef8e", "rddcbe6dfb1eb", "r82203a85ef8e"),
    ("Query temperature + W decay", "re563830cf150", None, ""),
    ("FoX, without W decay", "rd5ea49c1fde6", None, ""),
    ("FoX NoPE, with W decay", "r554ce7da6a8a", None, ""),
)
# Calibration-mean interventions, all layers at once, on degree-8 query + W decay checkpoints: (model,
# evaluation run). Until Codex merges its branch, the runs' records are read from that branch.
INPUT_DEPENDENCE = (
    ("Query", "r50200ac83292"),
    ("SoS", "r850466844aff"),
    ("SoS + softplus input + prefactor", "r2384d21a3134"),
)
INPUT_DEPENDENCE_BRANCH = "codex/layerwise-polynomial-swap"
FIGURES = {"length-extrapolation": (RESULTS / "diagnostics/length_extrapolation/extrapolation_ablation", (
    "long_documents_16k.png", "long_documents_16k_zoom.png", "long_documents_16k_ablations.png",
    "position_weights_long_documents.png"))}


def record(run_id: str, name: str) -> dict:
    return json.loads((RUNS / run_id / name).read_text())


def overfitting() -> list[dict]:
    rows = []
    for fit, batch, start, run_id in FITS:
        completion = record(run_id, "completion.json")
        rows.append([text(f"{fit} (<var>B</var> = {batch:,})"), text(start),
                     number(math.exp(completion["initial_nll"]), 3, run_id),
                     number(math.exp(completion["best_nll"]), 3, run_id)])
    bold_lowest([row[3] for row in rows])

    query_key = record(QUERY_KEY, "completion.json")
    dense_mse = record(DENSE, "completion.json")
    dense = next(entry for line in (RUNS / DENSE / "metrics.jsonl").read_text().splitlines()
                 if (entry := json.loads(line)).get("event") == "evaluation"
                 and entry["step"] == dense_mse["best_regression_step"])

    def mse(run_id: str, selected: float) -> dict:
        return text(f"{record(run_id, 'initial_metrics.json')['regression']['factorial_mse']:.3f} → {selected:.3f}")

    regression = [
        [text("Taylor (no fitting)"), text("—"), number(query_key["baseline"]["fit"]["perplexity"], 3),
         number(query_key["baseline"]["heldout"]["perplexity"], 3)],
        [text("Free (teacher)"), text("—"), number(math.exp(record(FITS[0][3], "completion.json")["best_nll"]), 3),
         text("—")],
        [text("Dense (<var>K</var> = 9)"), mse(DENSE, dense_mse["best_regression_mse"]),
         number(dense["fit"]["perplexity"], 3, DENSE), number(dense["validation"]["perplexity"], 3, DENSE)],
        [text("Query-key"), mse(QUERY_KEY, query_key["best_regression_mse"]),
         number(query_key["selected_fit"]["perplexity"], 3, QUERY_KEY),
         number(query_key["selected_heldout"]["perplexity"], 3, QUERY_KEY)],
    ]
    return [
        {"key": "fits",
         "tables": [{"head": ["Fit", "Start checkpoint", "Start PPL", "Best fitted PPL"], "rows": rows}]},
        {"key": "regression", "title": "Regression",
         "tables": [{"head": ["Model", "MSE", "Fitted-batch PPL", "Held-out PPL"], "rows": regression}]},
    ]


def continued_pretraining() -> list[dict]:
    def cell(run_id: str | None) -> dict:
        if run_id is None:
            return text("Not run", muted=True)
        return number(best(run_id), 4, run_id) if run_id else text("—")

    return [{
        "key": "cpt",
        "title": "2k additional steps",
        "tables": [{"head": ["Attention during CPT", "Taylor start", "Exp start", "Same-model start"],
                    "rows": [[text(label), *(cell(run_id) for run_id in runs)] for label, *runs in CONTINUATIONS]}],
    }]


def run_text(run_id: str, name: str) -> str:
    path = RUNS / run_id / name
    if path.is_file():
        return path.read_text()
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{INPUT_DEPENDENCE_BRANCH}:{relative}"],
                          check=True, capture_output=True, text=True).stdout


def input_dependence() -> list[dict]:
    rows = []
    for model, run_id in INPUT_DEPENDENCE:
        nll = {entry["condition"]: entry["nll"] for line in run_text(run_id, "metrics.jsonl").splitlines()
               if (entry := json.loads(line)).get("layer_scope", "all_layers") == "all_layers"}
        native, mean = math.exp(nll["native"]), math.exp(nll["calibration_mean"])
        cell = number(mean, 3, run_id)
        cell["text"] += f' <span class="change">({change(mean, native)})</span>'
        rows.append([text(model), cell])
    return [{"key": "calibration-means",
             "tables": [{"head": ["Model", "Calibration means (% vs native)"], "compact": True, "rows": rows,
                         "caption": "PPL on 256 validation sequences at update 7,500 of the degree-8 runs (prefactor: core "
                                    "degree 8); native: the unchanged model."}]}]


def copy_figures() -> None:
    for page, (source, names) in FIGURES.items():
        target = Path(__file__).resolve().parents[1] / "figures" / page
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(source / name, target / name)


def main() -> None:
    data = {"overfitting": overfitting(), "continued_pretraining": continued_pretraining(),
            "input_dependence": input_dependence()}
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    copy_figures()
    print(f"wrote {output.relative_to(ROOT)} and copied the figures")


if __name__ == "__main__":
    main()
