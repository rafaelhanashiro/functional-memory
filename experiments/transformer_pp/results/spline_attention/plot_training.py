"""Quartic spline pretraining: perplexity against the degree-four controls, score coverage, and the
comparison of initial spline functions.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.spline_attention.plot_training

Reads only the runs pinned in selection.yaml and writes ``training_summary.csv``, ``training.png``,
``initialization_summary.csv`` and the generated table in ``README.md``. Coverage pools every layer and
head of the fixed evaluation sample, weighted by its causal score count; "beyond ±32" is the share of
scores evaluated by the end pieces' polynomial continuation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PAIRS = (("static", "static_spline", "control_static"), ("query + W decay", "query_spline", "control_query"))
# Each initial spline function's roles for the static and the query + W decay generator.
STARTS = (
    ("taylor", "Taylor, $T_4(s)$", "static_spline", "query_spline"),
    ("affine", "Affine, $1 + s/4$", "static_spline_affine", "query_spline_affine"),
    ("quadratic", "Quadratic, $1 + s + s^2/2$", "static_spline_quadratic", "query_spline_quadratic"),
    ("binomial", "Scaled binomial, $(1 + s/4)^4/16$", "static_spline_binomial", "query_spline_binomial"),
)
INITIALIZATION_FIELDS = ("start", "generator", "run_id", "best_validation_step", "best_validation_perplexity",
                         "terminal_validation_perplexity", "best_percent_vs_taylor_spline",
                         "best_percent_vs_polynomial_control")


def _runs() -> dict[str, str]:
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    return {entry["role"]: entry["run_id"] for entry in selection["runs"]}


def _rows(run_id: str) -> list[dict]:
    path = ROOT / "experiments/transformer_pp/runs" / run_id / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _coverage(row: dict) -> dict:
    report = row["attention_diagnostics"]["spline_score_coverage"]
    count = np.asarray(report["causal_score_count"], float)
    pieces = np.asarray(report["piece_fraction"], float)
    beyond = np.asarray(report["above_range_fraction"], float) + np.asarray(report["below_range_fraction"], float)
    pooled = (pieces * count[..., None]).sum(axis=(0, 1)) / count.sum()
    return {"outer": float(pooled[0] + pooled[3]), "beyond": float((beyond * count).sum() / count.sum()),
            "worst_head_beyond": float(beyond.max()), "score_min": float(np.min(report["score_min"])),
            "score_max": float(np.max(report["score_max"]))}


def _best_and_terminal(run_id: str) -> tuple[dict, dict]:
    if json.loads((ROOT / "experiments/transformer_pp/runs" / run_id / "record.json").read_text())["status"] != "complete":
        raise ValueError(f"Selected run is not complete: {run_id}")
    rows = [row for row in _rows(run_id) if "validation_nll" in row]
    return min(rows, key=lambda row: row["validation_nll"]), max(rows, key=lambda row: row["step"])


def summary(runs: dict[str, str]) -> list[dict]:
    table = []
    for name, spline_role, control_role in PAIRS:
        (best, terminal), (best_control, terminal_control) = (_best_and_terminal(runs[spline_role]),
                                                              _best_and_terminal(runs[control_role]))
        coverage = _coverage(terminal)
        table.append({
            "variant": name, "spline_run": runs[spline_role], "control_run": runs[control_role],
            "spline_best_ppl": f"{best['validation_perplexity']:.3f}", "control_best_ppl": f"{best_control['validation_perplexity']:.3f}",
            "spline_terminal_ppl": f"{terminal['validation_perplexity']:.3f}",
            "control_terminal_ppl": f"{terminal_control['validation_perplexity']:.3f}",
            "terminal_outer_piece_share": f"{coverage['outer']:.4f}", "terminal_beyond_range_share": f"{coverage['beyond']:.5f}",
            "terminal_worst_head_beyond_range": f"{coverage['worst_head_beyond']:.4f}",
            "terminal_score_min": f"{coverage['score_min']:.1f}", "terminal_score_max": f"{coverage['score_max']:.1f}",
        })
    return table


def initializations(runs: dict[str, str]) -> list[dict]:
    """Every initial spline function under both generators, against the Taylor start and the polynomial."""
    reference = {generator: _best_and_terminal(runs[role])[0]["validation_perplexity"]
                 for generator, role in (("static", "static_spline"), ("query + W decay", "query_spline"))}
    control = {generator: _best_and_terminal(runs[role])[0]["validation_perplexity"]
               for generator, role in (("static", "control_static"), ("query + W decay", "control_query"))}
    rows = []
    for start, _, static_role, query_role in STARTS:
        for generator, role in (("static", static_role), ("query + W decay", query_role)):
            best, terminal = _best_and_terminal(runs[role])
            rows.append({
                "start": start, "generator": generator, "run_id": runs[role], "best_validation_step": best["step"],
                "best_validation_perplexity": best["validation_perplexity"],
                "terminal_validation_perplexity": terminal["validation_perplexity"],
                "best_percent_vs_taylor_spline": 100 * (best["validation_perplexity"] / reference[generator] - 1),
                "best_percent_vs_polynomial_control": 100 * (best["validation_perplexity"] / control[generator] - 1),
            })
    return rows


def _initialization_table(rows: list[dict], runs: dict[str, str]) -> str:
    cells = {(row["start"], row["generator"]): row for row in rows}
    cell = lambda row: (f"{row['best_validation_perplexity']:.3f}" if row["start"] == "taylor" else
                        f"{row['best_validation_perplexity']:.3f} ({row['best_percent_vs_taylor_spline']:+.1f}%)")
    lines = ["| Initial function | Static | Query + W decay |", "| --- | ---: | ---: |"]
    for start, label, _, _ in STARTS:
        lines.append(f"| {label} | {cell(cells[start, 'static'])} | {cell(cells[start, 'query + W decay'])} |")
    controls = [_best_and_terminal(runs[role])[0]["validation_perplexity"] for role in ("control_static", "control_query")]
    lines.append(f"| Degree-four polynomial control | {controls[0]:.3f} | {controls[1]:.3f} |")
    lines += ["", "Validation-best perplexity; in parentheses, the difference from the Taylor-started spline with the "
              "same generator. Terminal values are in `initialization_summary.csv`."]
    return "\n".join(lines)


def update_readme(table: str, output: Path) -> None:
    path = output / "README.md"
    begin, end = "<!-- BEGIN GENERATED INITIALIZATIONS -->", "<!-- END GENERATED INITIALIZATIONS -->"
    prefix, rest = path.read_text().split(begin)
    _, suffix = rest.split(end)
    path.write_text(prefix + begin + "\n\n" + table + "\n\n" + end + suffix)


def main(output: Path = HERE):
    runs = _runs()
    table = summary(runs)
    with (output / "training_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    rows = initializations(runs)
    with (output / "initialization_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INITIALIZATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    update_readme(_initialization_table(rows, runs), output)
    figure, (gap_axis, cover_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for (name, spline_role, control_role), color in zip(PAIRS, ("#2a6f97", "#c0392b")):
        spline, control = _rows(runs[spline_role]), _rows(runs[control_role])
        control_ppl = {row["step"]: row["validation_perplexity"] for row in control}
        steps = [row["step"] for row in spline if row["step"] in control_ppl and row["step"] >= 1000]
        ratio = [100 * (row["validation_perplexity"] / control_ppl[row["step"]] - 1) for row in spline if row["step"] in steps]
        gap_axis.plot(steps, ratio, color=color, linewidth=1.6, label=name)
        coverage = [(row["step"], _coverage(row)) for row in spline]
        cover_axis.plot([s for s, _ in coverage], [100 * c["outer"] for _, c in coverage], color=color, linewidth=1.6,
                        label=f"{name}: outer pieces")
        cover_axis.plot([s for s, _ in coverage], [100 * c["worst_head_beyond"] for _, c in coverage], color=color,
                        linewidth=1.2, linestyle="--", label=f"{name}: worst head beyond ±32")
    gap_axis.axhline(0, color="0.6", linewidth=0.8)
    gap_axis.set_xlabel("training step")
    gap_axis.set_ylabel("validation PPL vs degree-four control (%)")
    gap_axis.set_title("Taylor-started spline against its m = 4 control (from step 1,000)", fontsize=10)
    gap_axis.legend(frameon=False, fontsize=8)
    cover_axis.set_xlabel("training step")
    cover_axis.set_ylabel("share of causal scores (%)")
    cover_axis.set_title("Scores in the outer pieces (|s| ≥ 6) and beyond ±32", fontsize=10)
    cover_axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "training.png", dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(figure)


if __name__ == "__main__":
    main()
