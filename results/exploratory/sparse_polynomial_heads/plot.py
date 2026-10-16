"""Regenerate the explicitly selected sparse-polynomial head comparison."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
VARIANTS = ("fixed", "dynamic", "random", "dense")
STYLES = {
    "fixed": ("#2878B5", "Fixed ranked"),
    "dynamic": ("#D65F00", "Query-dependent"),
    "random": ("#6C757D", "Random fixed"),
    "dense": ("#663399", "Dense polynomial"),
}
SPLITS = ("test", "test_long", "best_validation", "terminal_validation")
METRICS = (
    "output_relative_rms", "attention_relative_rms", "invalid_fraction",
    "negative_kernel_fraction", "denominator_min", "queries", "feature_count",
    "active_features",
)
FIELDS = (
    "run_id", "role", "variant", "budget", "status", "split", "checkpoint",
    "step", "layer", "head", "measurement_available", *METRICS,
    "config_sha256", "metrics_sha256", "completion_sha256",
)


def load_runs(directory: Path = DIRECTORY, root: Path = ROOT):
    """Load pinned inputs without replacing absent or unsuccessful coordinates."""
    selection = yaml.safe_load((directory / "selection.yaml").read_text())
    entries = selection["runs"]
    verify_pinned_runs(root, entries)
    if not entries:
        raise ValueError("The comparison needs an explicit, nonempty run selection")
    rows, runs, coordinates = [], [], set()
    expected_heads = None
    for entry in entries:
        path = root / "experiments" / entry["study"] / "runs" / entry["run_id"]
        config = yaml.safe_load((path / "config.yaml").read_text())
        record = json.loads((path / "record.json").read_text())
        variant, budget = config["fit"]["variant"], int(config["fit"]["budget"])
        if variant not in VARIANTS or budget <= 0:
            raise ValueError(f"Unknown approximation or invalid budget: {entry['run_id']}")
        if entry["variant"] != variant or int(entry["budget"]) != budget:
            raise ValueError(f"Selection and recipe disagree: {entry['run_id']}")
        if (variant, budget) in coordinates:
            raise ValueError(f"Duplicate selected coordinate: {variant}, {budget}")
        coordinates.add((variant, budget))
        heads = tuple(sorted((int(h["layer"]), int(h["head"])) for h in config["heads"]))
        if len(set(heads)) != len(heads) or not heads:
            raise ValueError(f"Missing or duplicate heads: {entry['run_id']}")
        if expected_heads is not None and expected_heads != heads:
            raise ValueError(f"Selected runs do not use the same heads: {entry['run_id']}")
        expected_heads = heads
        completion_path = path / "completion.json"
        completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
        status = record["status"]
        if completion and status != "complete":
            raise ValueError(f"Completion conflicts with recorded status: {entry['run_id']}")
        if status == "complete" and not completion:
            raise ValueError(f"Complete run has no completion file: {entry['run_id']}")
        if completion and (completion["variant"] != variant or int(completion["budget"]) != budget):
            raise ValueError(f"Completion and recipe disagree: {entry['run_id']}")
        run = {
            "run_id": entry["run_id"], "role": entry["role"], "variant": variant,
            "budget": budget, "status": status, "heads": heads,
            "study": entry["study"], "completion": completion,
        }
        runs.append(run)
        for split in SPLITS:
            measurements = completion.get(split, [])
            indexed = {(int(m["layer"]), int(m["head"])): m for m in measurements}
            if len(indexed) != len(measurements) or set(indexed) - set(heads):
                raise ValueError(f"Unexpected or duplicate {split} heads: {entry['run_id']}")
            for layer, head in heads:
                metric = indexed.get((layer, head))
                row = {field: "" for field in FIELDS}
                row.update(
                    run_id=entry["run_id"], role=entry["role"], variant=variant,
                    budget=budget, status=status, split=split,
                    checkpoint="terminal" if split == "terminal_validation" else "validation_best",
                    step=completion.get("step" if split == "terminal_validation" else "best_step", ""),
                    layer=layer, head=head, measurement_available=metric is not None,
                )
                for name in ("config", "metrics", "completion"):
                    filename = f"{name}.{'yaml' if name == 'config' else 'jsonl' if name == 'metrics' else 'json'}"
                    row[f"{name}_sha256"] = entry["sources"].get(filename, {}).get("sha256", "")
                if metric is not None:
                    row.update({name: metric.get(name, "") for name in METRICS})
                rows.append(row)
    order = {name: position for position, name in enumerate(VARIANTS)}
    runs.sort(key=lambda run: (order[run["variant"]], run["budget"], run["run_id"]))
    rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["layer"], row["head"],
                               order[row["variant"]], row["budget"], row["run_id"]))
    return runs, rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def save_figure(fig, directory: Path, stem: str):
    fig.savefig(directory / f"{stem}.png", dpi=180, metadata={"Software": "Matplotlib"})
    fig.savefig(directory / f"{stem}.pdf", metadata={
        "CreationDate": None, "ModDate": None, "Creator": "Matplotlib",
    })
    plt.close(fig)


def render(runs, rows, directory: Path = DIRECTORY):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
        "svg.hashsalt": "sparse-polynomial-heads",
    })
    heads = runs[0]["heads"]
    budgets = sorted({run["budget"] for run in runs})
    styles = [Line2D([0], [0], color=STYLES[variant][0], marker="o", linewidth=1.5,
                     label=STYLES[variant][1]) for variant in VARIANTS]
    plots = (
        ("heldout_output_error", "Held-out head-output reconstruction", "output_relative_rms", ("test",)),
        ("heldout_attention_error", "Held-out normalized attention reconstruction", "attention_relative_rms", ("test",)),
        ("long_context_output_error", "Longer-context head-output reconstruction", "output_relative_rms", ("test_long",)),
        ("validation_output_error", "Validation head-output reconstruction", "output_relative_rms",
         ("best_validation", "terminal_validation")),
    )
    for stem, title, metric, splits in plots:
        fig, axes = plt.subplots(math.ceil(len(heads) / 3), 3, figsize=(12.2, 7.2), squeeze=False)
        maximum = max([1.0, *[row[metric] for row in rows
                              if row["split"] in splits and finite(row[metric])]])
        for ax, (layer, head) in zip(axes.flat, heads):
            if metric == "output_relative_rms":
                ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle=":")
            available = [row for row in rows if (row["layer"], row["head"]) == (layer, head)
                         and row["split"] in splits]
            missing, invalid = set(), set()
            for split in splits:
                for variant in VARIANTS:
                    points = sorted((row for row in available if row["variant"] == variant
                                     and row["split"] == split), key=lambda row: row["budget"])
                    xs, ys = [], []
                    for row in points:
                        xs.append(row["budget"])
                        value = row[metric]
                        valid = row["measurement_available"] and finite(value)
                        ys.append(value if valid else float("nan"))
                        if not valid:
                            missing.add((variant, row["budget"], row["status"]))
                        fraction = row["invalid_fraction"]
                        if finite(fraction) and fraction > 0:
                            invalid.add((variant, row["budget"]))
                            if valid:
                                ax.scatter(row["budget"], value, s=64, facecolors="none",
                                           edgecolors="#B91C1C", linewidths=1.2, zorder=5)
                            else:
                                ax.scatter(row["budget"], .94, marker="x", color="#B91C1C",
                                           transform=ax.get_xaxis_transform(), zorder=5)
                    terminal = split == "terminal_validation"
                    ax.plot(xs, ys, color=STYLES[variant][0], linestyle="--" if terminal else "-",
                            marker="x" if terminal else "o", markersize=4.5, linewidth=1.5)
            if missing:
                labels = [f"{variant}: " + ", ".join(str(budget) for v, budget, _ in sorted(missing)
                                                     if v == variant)
                          for variant in VARIANTS if any(v == variant for v, _, _ in missing)]
                ax.text(.02, .98, "Unavailable errors\n" + "\n".join(labels), transform=ax.transAxes,
                        va="top", fontsize=6.5, color="#666666")
            if invalid:
                ax.text(.02, .02, "Red marks: invalid query fraction > 0", transform=ax.transAxes,
                        va="bottom", fontsize=7, color="#B91C1C")
            ax.set_title(f"Layer {layer}, head {head}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets, [f"{b:,}" for b in budgets], fontsize=8)
            ax.set_xlim(min(budgets) / 1.5, max(budgets) * 1.5)
            ax.set_ylim(0, maximum * 1.10)
            ax.set_xlabel("Features per query")
            ax.set_ylabel("Relative RMS error")
            ax.grid(alpha=.18)
        for ax in list(axes.flat)[len(heads):]:
            ax.set_visible(False)
        legend = list(styles)
        if metric == "output_relative_rms":
            legend.append(Line2D([0], [0], color="#444444", linestyle=":", label="Zero output"))
        if len(splits) == 2:
            legend += [Line2D([0], [0], color="black", marker="o", label="Validation best"),
                       Line2D([0], [0], color="black", linestyle="--", marker="x", label="Terminal")]
        fig.legend(handles=legend, loc="lower center", ncol=4 if len(splits) == 2 else len(legend),
                   frameon=False, bbox_to_anchor=(.5, .025))
        fig.suptitle(title + "\nFrozen Llama-3.2-1B; degree ≤ 3; dense key updates", fontsize=12)
        fig.text(.5, .005, "All panels use the same linear error scale; missing measurements remain unavailable.",
                 ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, .11 if len(splits) == 2 else .085, 1, .93))
        fig.subplots_adjust(wspace=.32)
        save_figure(fig, directory, stem)


def update_status(runs, rows, directory: Path = DIRECTORY):
    path = directory / "README.md"
    text = path.read_text()
    begin, end = "<!-- BEGIN GENERATED STATUS -->", "<!-- END GENERATED STATUS -->"
    prefix, rest = text.split(begin)
    _, suffix = rest.split(end)
    lines = [
        "| Variant | Budget | Run ID | Recorded status | Test heads reported | Maximum invalid query fraction |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for run in runs:
        test = [row for row in rows if row["run_id"] == run["run_id"] and row["split"] == "test"]
        count = sum(row["measurement_available"] for row in test)
        fractions = [row["invalid_fraction"] for row in test if finite(row["invalid_fraction"])]
        maximum = f"{max(fractions):.6g}" if fractions else "unavailable"
        link = f"../../../experiments/{run['study']}/runs/{run['run_id']}/record.json"
        lines.append(f"| {STYLES[run['variant']][1]} | {run['budget']:,} | "
                     f"[{run['run_id']}]({link}) | {run['status']} | {count}/{len(test)} | {maximum} |")
    path.write_text(prefix + begin + "\n\n" + "\n".join(lines) + "\n\n" + end + suffix)


def main():
    runs, rows = load_runs()
    with (DIRECTORY / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    render(runs, rows)
    update_status(runs, rows)


if __name__ == "__main__":
    main()
