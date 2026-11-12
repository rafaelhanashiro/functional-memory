"""Regenerate the rational polynomial comparison from its pinned runs.

    PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.rational_polynomial.plot

Writes ``summary.csv``, ``learning_curves.png``/``.pdf``, and the generated tables in
``README.md``. Coefficient statistics come from each completed run's terminal
validation diagnostics in ``metrics.jsonl``; failed runs contribute their
numerical-failure record.
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


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
FIELDS = (
    "run_id", "role", "kernel", "denominator_degree", "denominator_parameterization", "denominator_constant",
    "penalized_denominator", "coefficient_penalty", "weight_decay", "status", "terminal_step", "training_tokens",
    "terminal_validation_nll", "terminal_validation_perplexity", "perplexity_above_softmax",
    "best_validation_step", "best_validation_perplexity", "terminal_coefficient_penalty",
    "median_denominator_constant", "min_denominator_constant", "median_denominator_variation",
    "max_denominator_variation", "max_abs_denominator_coefficient", "median_numerator_high_order_ratio",
    "metrics_sha256", "outcome_sha256",
)
# (denominator degree, form, constant) -> (table name, color, line style); the order is the table order.
DENOMINATORS = {
    (8, "direct", "fixed"): ("signed degree 8", "#B2182B", "-"),
    (1, "direct", "learned"): ("linear, learned constant", "#276FBF", "-"),
    (2, "sum_of_squares", "learned"): ("nonnegative quadratic, learned constant", "#1B9E77", "-"),
    (8, "sum_of_squares", "fixed"): ("nonnegative degree 8", "#D65F00", "-"),
}
CONTROLS = {"softmax": ("softmax", "#000000", "--"), "taylor": ("normalized fixed Taylor, degree 8", "#707070", ":")}


def kernel_of(model) -> str:
    """Denominator key for rational runs; ``softmax`` or ``taylor`` for the controls."""
    if model.get("attention_coefficient_parameterization") == "rational":
        key = (model.get("attention_denominator_degree") or model["attention_degree"],
               model.get("attention_denominator_parameterization", "direct"),
               model.get("attention_denominator_constant", "fixed"))
        if key not in DENOMINATORS:
            raise ValueError(f"Unlisted rational denominator: {key}")
        return "/".join(map(str, key))
    if model["attention_degree"] is None:
        return "softmax"
    if model["attention_coefficients"] == "fixed" and model["attention_degree"] == 8:
        return "taylor"
    raise ValueError("A control must be softmax or fixed Taylor of degree 8")


def _style(kernel):
    if kernel in CONTROLS:
        return CONTROLS[kernel]
    degree, form, constant = kernel.split("/")
    return DENOMINATORS[(int(degree), form, constant)]


def coefficient_statistics(diagnostics, score_range):
    """Per layer and head, over |s| <= score_range: the denominator's constant, its
    variation (max - min) relative to that constant, its largest non-constant
    coefficient, and how far the numerator's terms of order >= 2 outweigh a_1 s."""
    coefficients = diagnostics["coefficients"]
    names = list(coefficients["effective_coefficient_names"])
    values = np.asarray(coefficients["effective"], dtype=np.float64).reshape(-1, len(names))
    numerator = values[:, [i for i, name in enumerate(names) if name.startswith("numerator_")]]
    denominator = values[:, [i for i, name in enumerate(names) if name.startswith("denominator_")]]
    if "denominator_constant" not in names:  # constant fixed at 1
        denominator = np.concatenate([np.ones((len(values), 1)), denominator], axis=1)
    scores = np.linspace(-score_range, score_range, 6001)
    sums = np.stack([np.polynomial.polynomial.polyval(scores, row) for row in denominator])
    high = np.stack([np.polynomial.polynomial.polyval(scores, np.r_[0.0, 0.0, row[2:]]) for row in numerator])
    variation = (sums.max(axis=1) - sums.min(axis=1)) / denominator[:, 0]
    return dict(
        median_denominator_constant=float(np.median(denominator[:, 0])),
        min_denominator_constant=float(denominator[:, 0].min()),
        median_denominator_variation=float(np.median(variation)),
        max_denominator_variation=float(variation.max()),
        max_abs_denominator_coefficient=float(np.abs(denominator[:, 1:]).max()),
        median_numerator_high_order_ratio=float(np.median(
            np.abs(high).max(axis=1) / (score_range * np.abs(numerator[:, 1])))),
    )


def load_runs(selection: dict | None = None):
    selection = selection or yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    settings = selection["settings"]
    verify_pinned_runs(ROOT, selection["runs"])
    rows, curves = [], {}
    for entry in selection["runs"]:
        path = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
        config = yaml.safe_load((path / "config.yaml").read_text())
        model, optim = config["model"], config["optim"]
        if model["initialization"] != "transformer_reference":
            raise ValueError(f"Non-reference initialization: {entry['run_id']}")
        if not (model["qk_norm"] and model["attention_normalization"]):
            raise ValueError(f"Expected QK-norm and normalized attention: {entry['run_id']}")
        if (optim["warmup_steps"], config["train"]["seed"], model["qk_head_dim"]) != (
                settings["warmup_steps"], settings["seed"], settings["qk_head_dim"]):
            raise ValueError(f"Warmup, seed, or key width differs from the selection settings: {entry['run_id']}")
        kernel = kernel_of(model)
        rational = kernel not in CONTROLS
        if rational != (entry["role"] == "primary"):
            raise ValueError(f"Rational runs are primary and controls are softmax or Taylor: {entry['run_id']}")
        if rational and (model["attention_degree"] != settings["numerator_degree"]
                         or model["attention_coefficient_initialization"] != settings["coefficient_initialization"]
                         or optim["coefficient_penalty"] != settings["coefficient_penalty"]):
            raise ValueError(f"Numerator degree, start, or penalty differs from the selection settings: {entry['run_id']}")
        status = json.loads((path / "record.json").read_text())["status"]
        row = dict(
            run_id=entry["run_id"], role=entry["role"], kernel=kernel,
            denominator_degree=kernel.split("/")[0] if rational else "",
            denominator_parameterization=kernel.split("/")[1] if rational else "",
            denominator_constant=kernel.split("/")[2] if rational else "",
            penalized_denominator=optim.get("coefficient_penalty_denominator", True) if rational else "",
            coefficient_penalty=optim["coefficient_penalty"] if rational else "",
            weight_decay=optim["weight_decay"], status=status,
        )
        if status == "numerical_failure":
            (failure_path,) = path.glob("numerical_failure_step_*.json")
            failure = json.loads(failure_path.read_text())
            row.update(terminal_step=failure["step"], training_tokens=failure["training_tokens"],
                       outcome_sha256=entry["sources"][failure_path.name]["sha256"])
            curves[entry["run_id"]] = []
        elif status == "complete":
            completion = json.loads((path / "completion.json").read_text())
            best = json.loads((path / "best_validation.json").read_text())
            if completion["final_step"] != optim["total_steps"]:
                raise ValueError(f"Selected run is incomplete: {entry['run_id']}")
            metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
            validation = [event for event in metrics if event.get("event") == "validation"]
            curves[entry["run_id"]] = validation
            row.update(
                terminal_step=completion["final_step"], training_tokens=completion["training_tokens"],
                terminal_validation_nll=completion["validation_nll"],
                terminal_validation_perplexity=completion["validation_perplexity"],
                best_validation_step=best["step"], best_validation_perplexity=best["validation_perplexity"],
                metrics_sha256=entry["sources"]["metrics.jsonl"]["sha256"],
                outcome_sha256=entry["sources"]["completion.json"]["sha256"],
            )
            if rational:
                row["terminal_coefficient_penalty"] = validation[-1]["train_coefficient_penalty"]
                row.update(coefficient_statistics(validation[-1]["attention_diagnostics"],
                                                  settings["coefficient_score_range"]))
        else:
            raise ValueError(f"Selected run is neither complete nor a numerical failure: {entry['run_id']}")
        rows.append(row)
    (softmax,) = [row for row in rows if row["kernel"] == "softmax"]
    for row in rows:
        if "terminal_validation_perplexity" in row:
            row["perplexity_above_softmax"] = (row["terminal_validation_perplexity"]
                                               - softmax["terminal_validation_perplexity"])
    order = [*("/".join(map(str, key)) for key in DENOMINATORS), *CONTROLS]
    rows.sort(key=lambda row: (order.index(row["kernel"]), -row["weight_decay"]))
    return rows, curves


def _name(row):
    return _style(row["kernel"])[0]


def render(rows, curves, output=DIRECTORY):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.hashsalt": "rational-polynomial"}):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        failures = []
        for row in rows:
            name, color, linestyle = _style(row["kernel"])
            points = curves[row["run_id"]]
            if not points:
                failures.append(f"{name[0].upper() + name[1:]}, weight decay {row['weight_decay']:g}: "
                                f"update {row['terminal_step']}")
                continue
            label = name[0].upper() + name[1:]
            if row["role"] == "control":
                label += " (control)"
            ax.plot([p["training_tokens"] / 1e9 for p in points], [p["validation_perplexity"] for p in points],
                    color=color, linestyle=linestyle, linewidth=1.8,
                    label=f"{label}: {row['terminal_validation_perplexity']:.2f}")
        ax.set_yscale("log")
        ticks = [14, 16, 18, 20, 25, 30]
        ax.set_yticks(ticks, [str(tick) for tick in ticks])
        ax.minorticks_off()
        ax.set_xlim(0, 10.25)
        ax.set_xlabel("Training tokens (billions)")
        ax.set_ylabel("Validation perplexity (lower is better)")
        ax.grid(alpha=.18)
        ax.legend(title="Terminal validation perplexity", frameon=False, loc="upper right")
        ax.text(.02, .02, "Numerical failure before the first validation:\n" + "\n".join(failures),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=_style(rows[0]["kernel"])[1])
        ax.set_title("Rational polynomial attention, degree-8 numerator, by denominator", fontsize=11)
        fig.tight_layout()
        fig.savefig(output / "learning_curves.png", dpi=180, metadata={"Software": "Matplotlib"})
        fig.savefig(output / "learning_curves.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                                "Creator": "Matplotlib"})
        plt.close(fig)


def _tables(rows, curves, steps):
    rational = [row for row in rows if row["role"] == "primary"]
    complete = [row for row in rational if row["status"] == "complete"]
    best = min(row["terminal_validation_perplexity"] for row in complete)
    lines = ["| Run | Denominator | Penalized denominator | Weight decay | Outcome |",
             "| --- | --- | --- | ---: | ---: |"]
    for row in rows:
        if row["role"] == "control":
            name, penalized = f"{_name(row)} (control)", ""
        else:
            name, penalized = _name(row), "yes" if row["penalized_denominator"] else "no"
        if row["status"] == "numerical_failure":
            outcome = f"fail @{row['terminal_step']}"
        else:
            outcome = f"{row['terminal_validation_perplexity']:.3f}"
            outcome = f"**{outcome}**" if row["role"] == "primary" and row["terminal_validation_perplexity"] == best else outcome
        lines.append(f"| `{row['run_id']}` | {name} | {penalized} | {row['weight_decay']:g} | {outcome} |")
    lines += ["", "| Validation PPL at update | " + " | ".join(f"{step:,}" for step in steps) + " |",
              "| --- |" + " ---: |" * len(steps)]
    for row in rows:
        if row["status"] != "complete":
            continue
        by_step = {point["step"]: point["validation_perplexity"] for point in curves[row["run_id"]]}
        cells = [f"{by_step[step]:.3f}" if step == steps[-1] else f"{by_step[step]:.2f}" for step in steps]
        name = _name(row)
        lines.append(f"| {name[0].upper() + name[1:]} | " + " | ".join(cells) + " |")
    lines += ["", "| Denominator | Constant, median (min) | Variation over `abs(s) <= 3`, median (max) | "
                  "Largest `abs(b_r)`, `r >= 1` | Numerator orders `>= 2` over `a_1 s`, median | Penalty term (nats) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in complete:
        constant = ("1 (fixed)" if row["denominator_constant"] == "fixed" else
                    f"{row['median_denominator_constant']:.2f} ({row['min_denominator_constant']:.2f})")
        lines.append(f"| {_name(row)} | {constant} | "
                     f"{row['median_denominator_variation']:.2f} ({row['max_denominator_variation']:.1f}) | "
                     f"{row['max_abs_denominator_coefficient']:.2g} | {row['median_numerator_high_order_ratio']:.1f} | "
                     f"{row['terminal_coefficient_penalty']:.1e} |")
    return "\n".join(lines)


def update_tables(rows, curves, steps, output=DIRECTORY):
    path = output / "README.md"
    begin, end = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"
    prefix, rest = path.read_text().split(begin)
    _, suffix = rest.split(end)
    path.write_text(prefix + begin + "\n\n" + _tables(rows, curves, steps) + "\n\n" + end + suffix)


def main(output=DIRECTORY):
    selection = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
    rows, curves = load_runs(selection)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    render(rows, curves, output)
    update_tables(rows, curves, selection["settings"]["curve_table_steps"], output)


if __name__ == "__main__":
    main()
