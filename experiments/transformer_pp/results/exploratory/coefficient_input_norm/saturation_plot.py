"""Single-color saturation bars from explicitly selected, pinned measurements.

The matched-head figure includes original-input controls and both L2 placements.
The supplemental figure averages the L2 measurements over all layers and heads;
original controls have only a matched-head probe. Missing runs stay explicit.
No GPU, checkpoint, dataset, or numerical archive is needed to regenerate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scaffolds.records import load_run
from scaffolds.records.selections import verify_pinned_runs


BLUE = "#28699b"
INK = "#24323d"
MUTED = "#667582"
GRID = "#dce2e8"
VARIANTS = ("none", "post_rmsnorm", "pre_rmsnorm")
TITLES = {"none": "No L2", "post_rmsnorm": "L2 after RMSNorm", "pre_rmsnorm": "Raw L2"}
SCOPES = ("selected_head", "all_layers_heads")


def coordinate_names(degree: int, family: str) -> list[str]:
    if family == "direct":
        return [f"r{r}" for r in range(degree + 1)]
    if family == "sum_of_squares":
        half = degree // 2
        return [f"U{r}" for r in range(half + 1)] + [f"V{r}" for r in range(half)]
    raise ValueError(f"Unknown coefficient family: {family}")


def coordinate_ticks(names: list[str]) -> list[str]:
    return [rf"${name[0]}_{{{name[1:]}}}$" for name in names]


def check_sample(sample: dict, settings: dict) -> None:
    assert sample["record_sha256"] == settings["sample_record_sha256"]
    assert sample["selection"] == settings.get("sample_selection", "first_heldout_sequence")
    assert sample["num_sequences"] == settings.get("sequences", 1)
    assert sample["prediction_tokens"] == settings.get("prediction_tokens", 2048)


def check_fractions(values: np.ndarray) -> None:
    assert np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()


def load(selection: dict) -> list[dict]:
    """Read every panel from its declared role; never infer a newest/best run."""
    verify_pinned_runs(ROOT, selection["runs"])
    settings = selection["settings"]
    assert settings["saturation_absolute_tanh_threshold"] == .99
    assert settings["checkpoint_step"] == 7500
    assert settings["ppl_metric"] == "validation-best"
    entries = {entry["role"]: entry for entry in selection["runs"]}
    assert len(entries) == len(selection["runs"])
    records: dict[str, list[dict]] = {}

    def run_path(role):
        entry = entries[role]
        return ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]

    def measurements(role):
        if role not in records:
            path = run_path(role)
            assert load_run(path)["status"] == "complete"
            records[role] = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()
                             if line.strip()]
        return records[role]

    panels = []
    for panel in selection["panels"]:
        family, degree = panel["family"], panel["degree"]
        names = coordinate_names(degree, family)
        result = {**panel, "coordinates": names, "cells": {}}
        for variant in VARIANTS:
            choice = panel["variants"][variant]
            if choice.get("status") == "not_run":
                assert variant == "pre_rmsnorm" and degree == 8
                result["cells"][variant] = {"status": "not_run", "run_id": ""}
                continue
            role = choice["run_role"]
            path = run_path(role)
            record = load_run(path)
            assert record["status"] == "complete"
            config = yaml.safe_load((path / "config.yaml").read_text())
            model = config["model"]
            assert model["attention_degree"] == degree
            assert model["attention_coefficient_parameterization"] == family
            assert model["attention_coefficient_conditioning"] == "query"
            assert model["attention_coefficient_activation"] == "tanh"
            assert model["attention_coefficient_kernel_weight_decay"] is True
            assert config["optim"]["weight_decay"] == .1
            assert model["positional_encoding"] == "rope"
            best = json.loads((path / "best_validation.json").read_text())
            assert best["step"] == record["selected_checkpoint"]["step"] == settings["checkpoint_step"]
            ppl = float(best["validation_perplexity"])
            assert np.isfinite(ppl) and ppl > 0
            cell = {"status": "complete", "run_id": entries[role]["run_id"],
                    "best_validation_perplexity": ppl, "best_validation_step": best["step"]}
            if variant == "none":
                assert model.get("attention_coefficient_input_normalization", "none") == "none"
                matches = [row for row in measurements(choice["probe_role"])
                           if row["role"] == choice["measurement_role"]]
                assert len(matches) == 1
                row = matches[0]
                assert row["source_run_id"] == cell["run_id"]
                assert row["degree"] == degree
                assert row["checkpoint_step"] == settings["checkpoint_step"]
                assert row["checkpoint_sha256"] == record["selected_checkpoint"]["sha256"]
                assert row["layer"] == settings["layer_index"] and row["head"] == settings["head_index"]
                assert row["indexing"] == "zero_based"
                assert row["parent_parameters_unchanged"] and row["optimizer_updates"] == 0
                assert row["parent_parameter_sha256_before"] == row["parent_parameter_sha256_after"]
                check_sample(row["sample"], settings)
                values = row["degrees"]
                expected = ([f"r{value['degree']}" for value in values] if family == "direct"
                            else [value["coordinate_label"] for value in values])
                assert expected == names
                cell["selected_head"] = np.asarray([value["saturation_abs_tanh_ge099"]
                                                     for value in values], dtype=np.float64)
                cell["all_layers_heads"] = None
            else:
                assert model["attention_coefficient_input_normalization"] == "l2"
                assert model.get("attention_coefficient_input_source", "post_rmsnorm") == variant
                matches = [row for row in measurements(role) if row["step"] == settings["checkpoint_step"]]
                assert len(matches) == 1
                row = matches[0]
                diagnostic = row["attention_diagnostics"]
                check_sample(diagnostic["sample"], settings)
                report = diagnostic["coefficients"]["generator_input"]
                assert report["normalization"] == "l2" and report["finite"]
                assert report.get("source", "post_rmsnorm") == variant
                assert report["coordinate_labels"] == names
                assert report["saturation_absolute_tanh_threshold"] == .99
                assert report["projection_axes"] == ["layer", "head", "generator_coordinate"]
                fractions = np.asarray(report["saturation_fraction"], dtype=np.float64)
                assert fractions.shape == (model["num_layers"], model["num_heads"], degree + 1)
                check_fractions(fractions)
                cell["selected_head"] = fractions[settings["layer_index"], settings["head_index"]]
                cell["all_layers_heads"] = fractions.mean(axis=(0, 1))
                cell["layers"] = model["num_layers"]
                cell["heads"] = model["num_heads"]
            check_fractions(cell["selected_head"])
            result["cells"][variant] = cell
        panels.append(result)
    assert [(panel["family"], panel["degree"]) for panel in panels] == [
        ("direct", 8), ("sum_of_squares", 8), ("direct", 12), ("sum_of_squares", 12)]
    return panels


def _mean_label(percent: np.ndarray) -> str:
    mean = float(percent.mean())
    return "Mean: 0%" if mean == 0 else f"Mean: {mean:.2f}%" if mean >= 1 else f"Mean: {mean:.4f}%"


def plot(panels: list[dict], settings: dict, *, scope: str) -> None:
    matched = scope == "selected_head"
    variants = VARIANTS if matched else VARIANTS[1:]
    limits = settings["head_y_limits_percent" if matched else "all_heads_y_limits_percent"]
    assert limits == ([0, 100] if matched else [0, .25])
    ymax = limits[1]
    color = settings["bar_color"]
    assert color == BLUE
    fig, axes = plt.subplots(4, len(variants), figsize=(17, 14) if matched else (13, 12),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=.075 if matched else .09, right=.985, top=.84,
                        bottom=.09, hspace=.66, wspace=.16)
    for index, panel in enumerate(panels):
        for column, variant in enumerate(variants):
            ax = axes[index, column]
            cell = panel["cells"][variant]
            if column == 0:
                ax.set_title(f"{panel['label']} · $m={panel['degree']}$", loc="left",
                             fontsize=16, fontweight="bold", color=INK, pad=28)
            ax.set_ylim(0, ymax)
            ax.set_xlim(-.65, len(panel["coordinates"]) - .35)
            ax.set_yticks([0, 25, 50, 75, 100] if matched else [0, .05, .10, .15, .20, .25])
            ax.tick_params(axis="both", colors=MUTED, labelsize=10, length=3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#b4bfc8")
            ax.set_axisbelow(True)
            if cell["status"] == "not_run":
                ax.set_facecolor("#f6f8fa")
                ax.set_xticks([])
                ax.grid(False)
                ax.text(.5, .5, "Not run", transform=ax.transAxes, ha="center", va="center",
                        fontsize=16, color=MUTED)
                continue
            percent = 100 * cell[scope]
            assert np.max(percent) <= ymax
            positions = np.arange(len(percent))
            ax.bar(positions, percent, width=.72, color=color, linewidth=0, zorder=2)
            ax.set_xticks(positions, coordinate_ticks(panel["coordinates"]))
            ax.grid(axis="y", color=GRID, linewidth=.7)
            ax.text(0, 1.035, _mean_label(percent), transform=ax.transAxes,
                    color=MUTED, fontsize=11)
            ax.text(1, 1.035, f"Best validation PPL: {cell['best_validation_perplexity']:.4f}",
                    transform=ax.transAxes, ha="right", color=INK, fontsize=11, fontweight="bold")
            if not percent.any():
                ax.text(.5, .49, "0% at every coordinate", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12.5, color=MUTED)
            if panel["family"] == "sum_of_squares":
                ax.axvline(panel["degree"] // 2 + .5, color="#acb8c2", linewidth=.8,
                           linestyle=(0, (3, 3)))
            ax.set_xlabel("Polynomial order" if panel["family"] == "direct" else "Square-factor coordinate",
                          color=MUTED, fontsize=10.5, labelpad=7)
    for column, variant in enumerate(variants):
        bounds = axes[0, column].get_position()
        fig.text((bounds.x0 + bounds.x1) / 2, .885, TITLES[variant], ha="center",
                 fontsize=18, fontweight="bold", color=INK)
    fig.suptitle("Transformer++ · tanh saturation" if matched else "Transformer++ · tanh saturation across all heads",
                 x=.055, y=.975, ha="left", fontsize=25 if matched else 23,
                 fontweight="bold", color=INK)
    if matched:
        subtitle = (f"Layer {settings['layer_index'] + 1}, head {settings['head_index'] + 1}"
                    f"  ·  checkpoint {settings['checkpoint_step']:,}  ·  same 2,048-token sequence"
                    r"  ·  $|\tanh z|\geq0.99$")
        ylabel = "Saturated tokens (%)"
        footer = ("PPL: best validation evaluation. Bars combine both saturation tails; shared 0–100% scale.\n"
                  "SoS labels are coefficients of U and V in p(s) = U(s)² + V(s)². One head and one sequence; missing runs are not zeros.")
    else:
        shapes = {(cell["layers"], cell["heads"]) for panel in panels for cell in panel["cells"].values()
                  if "layers" in cell}
        assert shapes == {(15, 14)}
        subtitle = ("Mean across 15 layers × 14 heads  ·  checkpoint 7,500  ·  same 2,048-token sequence\n"
                    r"$|\tanh z|\geq0.99$  ·  zoomed 0–0.25% scale in every panel")
        ylabel = "Mean saturated tokens (%)"
        footer = ("PPL: best validation evaluation. Bars weight layers, heads and tokens equally; SoS labels are square-factor coordinates.\n"
                  "No-L2 controls have only the selected-head probe. Raw-L2 m=8 was not run. These are one-sequence diagnostics.")
    fig.text(.055, .928 if matched else .915, subtitle, ha="left", fontsize=12, color=MUTED,
             linespacing=1.7)
    fig.supylabel(ylabel, x=.018, fontsize=13, color=INK)
    fig.text(.055, .022, footer, fontsize=10.5 if matched else 10, color=MUTED, linespacing=1.7)
    stem = "saturation_by_coordinate" if matched else "saturation_all_heads"
    fig.savefig(OUT / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def save_summary(panels: list[dict]) -> None:
    fields = ["run_id", "family", "degree", "input_variant", "scope", "coordinate", "status",
              "saturation_fraction", "saturation_percent", "best_validation_perplexity", "best_validation_step"]
    with (OUT / "saturation_by_coordinate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for panel in panels:
            for variant in VARIANTS:
                cell = panel["cells"][variant]
                for scope in SCOPES:
                    values = cell.get(scope)
                    status = cell["status"] if cell["status"] == "not_run" else (
                        "not_measured" if values is None else "measured")
                    for index, coordinate in enumerate(panel["coordinates"]):
                        value = None if values is None else float(values[index])
                        writer.writerow({"run_id": cell["run_id"], "family": panel["family"],
                                         "degree": panel["degree"], "input_variant": variant,
                                         "scope": scope, "coordinate": coordinate, "status": status,
                                         "saturation_fraction": "" if value is None else format(value, ".12g"),
                                         "saturation_percent": "" if value is None else format(100 * value, ".12g"),
                                         "best_validation_perplexity": (format(cell["best_validation_perplexity"], ".12g")
                                                                        if "best_validation_perplexity" in cell else ""),
                                         "best_validation_step": cell.get("best_validation_step", "")})


def main() -> None:
    selection = yaml.safe_load((OUT / "saturation_selection.yaml").read_text())
    panels = load(selection)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        for scope in SCOPES:
            plot(panels, selection["settings"], scope=scope)
    save_summary(panels)
    for panel in panels:
        for variant in VARIANTS:
            cell = panel["cells"][variant]
            if cell["status"] == "not_run":
                print(f"{panel['label']} m={panel['degree']} / {TITLES[variant]}: not run")
            else:
                all_heads = cell.get("all_layers_heads")
                suffix = "not measured" if all_heads is None else f"{100 * all_heads.mean():.6f}%"
                print(f"{panel['label']} m={panel['degree']} / {TITLES[variant]}: "
                      f"selected head {100 * cell['selected_head'].mean():.6f}%; all heads {suffix}")
    print("Regenerated two saturation figures and coordinate CSV from pinned measurements.")


if __name__ == "__main__":
    main()
