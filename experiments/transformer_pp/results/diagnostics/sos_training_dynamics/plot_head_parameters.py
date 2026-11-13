"""One fixed middle-layer head's recorded SoS generator parameters.

The a/b/W coordinates index U and V factor powers, not the expanded polynomial
orders. No factorial or Taylor-factor rescaling is applied to these raw arrays.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import yaml

from plot_coefficients import _save


COLORS = {"U": "#28699b", "V": "#d66a32"}
INK = "#24323d"


def _load(root: Path, entry: dict, settings: dict) -> dict:
    run = root / "experiments" / entry["study"] / "runs" / entry["run_id"]
    model = yaml.safe_load((run / "config.yaml").read_text())["model"]
    degree = int(entry["degree"])
    assert degree >= 2 and degree % 2 == 0
    assert model["attention_degree"] == degree
    assert model["attention_coefficient_parameterization"] == "sum_of_squares"
    assert model["attention_coefficient_conditioning"] == "query"
    assert model["attention_coefficient_activation"] == "tanh"
    assert model["attention_coefficient_initialization"] == "taylor"
    assert model["attention_coefficient_amplitude_initial"] == 0.0
    assert model["attention_coefficient_readout"] == "diagonal"
    assert not model["attention_coefficient_tanh_bias"]
    assert model["attention_coefficient_gate"] == "none"
    assert model["attention_coefficient_scale"] == model["attention_coefficient_amplitude_scale"] == "none"
    layer, head = settings["layer_index"], settings["head_index"]
    assert layer == model["num_layers"] // 2
    assert 0 <= head < model["num_heads"]
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    steps = np.asarray([row["step"] for row in rows], dtype=int)
    assert len(steps) > 1 and steps[0] > 0 and np.all(np.diff(steps) > 0)
    parameters = {}
    shape = (len(rows), model["num_layers"], model["num_heads"], degree + 1)
    for name, field in settings["fields"].items():
        values = np.asarray([row["attention_diagnostics"]["coefficients"][field] for row in rows], dtype=float)
        assert values.shape == shape, (entry["run_id"], field, values.shape, shape)
        assert np.isfinite(values).all(), (entry["run_id"], field)
        parameters[name] = values[:, layer, head, :]
    assert set(parameters) == {"a", "b", "w_l2"}
    assert (parameters["w_l2"] >= 0).all()
    assert settings["w_norm"]["input_dimension"] == model["hidden_size"]
    half = degree // 2
    # Executed query SoS expansion splits [U_0,...,U_d,V_0,...,V_{d-1}].
    coordinates = [("U", j) for j in range(half + 1)] + [("V", j) for j in range(half)]
    return dict(entry=entry, degree=degree, steps=steps, layer=layer, head=head,
                coordinates=coordinates, input_dimension=model["hidden_size"], **parameters)


def _figure(data: dict, out: Path) -> None:
    degree, steps = data["degree"], data["steps"]
    half = degree // 2
    fig, axes = plt.subplots(2, half + 1, figsize=(max(15.2, 2.8 * (half + 1)), 7.5),
                             sharex=True, sharey="row", squeeze=False)
    for row, (name, initial, label) in enumerate((("a", 0., "Amplitude a"), ("b", 1., "Bias b"))):
        for power, ax in enumerate(axes[row]):
            for coordinate, (factor, order) in enumerate(data["coordinates"]):
                if order == power:
                    ax.plot(steps, data[name][:, coordinate], color=COLORS[factor], lw=2,
                            marker="o", markersize=2.7, markeredgewidth=0,
                            label=rf"${factor}_{power}$")
            ax.axhline(initial, color="#929ba5", lw=.8, ls=(0, (4, 3)), zorder=0)
            if initial != 0 and data[name].min() <= 0 <= data[name].max():
                ax.axhline(0, color="#cbd1d7", lw=.7, zorder=0)
            ax.scatter([0], [initial], color="#616d79", marker="D", s=23,
                       zorder=4, clip_on=False)
            ax.set_xlim(0, steps[-1])
            ax.set_xticks([0, 2500, 5000, 7500])
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="y", color="#e4e8eb", linewidth=.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#b2bac3")
            ax.tick_params(colors="#4d5965", labelsize=10)
            if row == 0:
                ax.set_title(rf"Factor power $j={power}$", fontsize=12, loc="left", color=INK, pad=10)
            else:
                ax.set_xlabel("Training update", fontsize=10)
            if power == 0:
                ax.set_ylabel(label, fontsize=13, color=INK, labelpad=12)
    axes[0, -1].text(.96, .94, "U only", transform=axes[0, -1].transAxes,
                     ha="right", va="top", color=COLORS["U"], fontsize=10)
    fig.suptitle(rf"Transformer++ · SoS $m={degree}$ · layer {data['layer'] + 1}, head {data['head'] + 1}",
                 x=.055, y=.975, ha="left", color=INK, fontsize=21, fontweight="bold")
    fig.text(.055, .916,
             rf"Learned $a,b$ by factor power   ·   $p_t(s)=U_t(s)^2+V_t(s)^2$"
             f"   ·   U degree {half}, V degree {half - 1}", fontsize=12, color="#5a6672")
    handles = [Line2D([], [], color=COLORS[factor], lw=2, marker="o", markersize=4,
                      label=f"{factor} factor") for factor in ("U", "V")]
    handles.append(Line2D([], [], color="#929ba5", ls="--", marker="D", markersize=4,
                           label="Analytic initialization: a = 0, b = 1"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .887),
               frameon=False, ncol=3, fontsize=11, columnspacing=2.8)
    fig.text(.055, .025,
             f"Layer {data['layer'] + 1} of 15; head {data['head'] + 1} of 14 "
             f"(indices {data['layer']}, {data['head']}), fixed before inspecting the curves. "
             f"{len(steps)} saved snapshots from updates {steps[0]:,}–{steps[-1]:,}.\n"
             "Raw generator parameters; no factorial rescaling. Same linear y-scale within each row. "
             "Step 0 is analytic; lines connect recorded snapshots.",
             color="#687583", fontsize=10, linespacing=1.6)
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.16, hspace=.28, wspace=.18)
    _save(fig, out, f"head_parameters_m{degree}")


def _norm_figure(data: dict, out: Path) -> None:
    degree, steps = data["degree"], data["steps"]
    half = degree // 2
    fig, axes = plt.subplots(1, half + 1, figsize=(max(15.2, 2.8 * (half + 1)), 4.9),
                             sharex=True, sharey=True, squeeze=False)
    for power, ax in enumerate(axes[0]):
        for coordinate, (factor, order) in enumerate(data["coordinates"]):
            if order == power:
                ax.plot(steps, data["w_l2"][:, coordinate], color=COLORS[factor], lw=2,
                        marker="o", markersize=2.7, markeredgewidth=0)
        ax.set_title(rf"Factor power $j={power}$", fontsize=12, loc="left", color=INK, pad=10)
        ax.set_xlim(0, steps[-1])
        ax.set_xticks([0, 2500, 5000, 7500])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("Training update", fontsize=10)
        ax.grid(axis="y", color="#e4e8eb", linewidth=.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#b2bac3")
        ax.tick_params(colors="#4d5965", labelsize=10)
    axes[0, 0].set_ylabel(r"Projection norm $\|W_j\|_2$", fontsize=13, color=INK, labelpad=12)
    axes[0, -1].text(.96, .94, "U only", transform=axes[0, -1].transAxes,
                     ha="right", va="top", color=COLORS["U"], fontsize=10)
    fig.suptitle(rf"Transformer++ · SoS $m={degree}$ · layer {data['layer'] + 1}, head {data['head'] + 1}",
                 x=.055, y=.975, ha="left", color=INK, fontsize=21, fontweight="bold")
    fig.text(.055, .884,
             rf"Learned $W$ norms by factor power   ·   $p_t(s)=U_t(s)^2+V_t(s)^2$"
             f"   ·   Euclidean norm over {data['input_dimension']} input channels",
             fontsize=12, color="#5a6672")
    handles = [Line2D([], [], color=COLORS[factor], lw=2, marker="o", markersize=4,
                      label=f"{factor} factor") for factor in ("U", "V")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .845),
               frameon=False, ncol=2, fontsize=11, columnspacing=2.8)
    fig.text(.055, .025,
             f"Same layer/head as the a,b plots (indices {data['layer']}, {data['head']}). "
             f"{len(steps)} saved snapshots from updates {steps[0]:,}–{steps[-1]:,}; no step-zero sample.\n"
             "Raw vector norms; shared linear y-axis, no rescaling. "
             "Norms alone do not determine the distribution of tanh inputs.",
             color="#687583", fontsize=10, linespacing=1.6)
    fig.subplots_adjust(left=.065, right=.985, top=.65, bottom=.24, wspace=.18)
    _save(fig, out, f"head_w_norms_m{degree}")


def plot(root: Path, out: Path, selection: dict) -> None:
    settings = selection["settings"]["single_head_parameters"]
    datasets = [_load(root, entry, settings) for entry in selection["runs"]]
    summary = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        for data in datasets:
            _figure(data, out)
            _norm_figure(data, out)
            for coordinate, (factor, power) in enumerate(data["coordinates"]):
                row = dict(run_id=data["entry"]["run_id"], degree=data["degree"],
                           layer_index=data["layer"], head_index=data["head"],
                           generator_coordinate=coordinate, factor=factor, factor_power=power,
                           snapshots=len(data["steps"]), first_step=int(data["steps"][0]),
                           terminal_step=int(data["steps"][-1]), analytic_initial_a=0., analytic_initial_b=1.)
                for name in ("a", "b", "w_l2"):
                    values = data[name][:, coordinate]
                    row.update({f"{name}_first_recorded": values[0], f"{name}_terminal": values[-1],
                                f"{name}_recorded_min": values.min(), f"{name}_recorded_max": values.max()})
                summary.append(row)
    with (out / "head_parameter_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({key: format(value, ".12g") if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})
