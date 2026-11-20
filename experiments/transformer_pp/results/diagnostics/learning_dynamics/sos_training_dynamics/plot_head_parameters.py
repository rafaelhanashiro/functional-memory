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


COLORS = {"U": "#28699b", "V": "#d66a32"}
INK = "#24323d"


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{name}.pdf", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


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


def _combined_figure(datasets: list[dict], out: Path) -> None:
    """The same a/b/W histories, grouped into one three-row section per SoS degree."""
    datasets = sorted(datasets, key=lambda data: data["degree"])
    assert [data["degree"] for data in datasets] == [8, 12]
    assert len({(data["layer"], data["head"]) for data in datasets}) == 1
    fig = plt.figure(figsize=(20, 12.5))
    row_settings = (("a", 0., "Amplitude a"), ("b", 1., "Bias b"),
                    ("w_l2", None, r"$\|W_r\|_2$"))
    for data, (bottom, top, heading) in zip(datasets, ((.545, .855, .887), (.07, .43, .462))):
        degree, steps = data["degree"], data["steps"]
        half = degree // 2
        grid = fig.add_gridspec(3, half + 1, left=.065, right=.985,
                               bottom=bottom, top=top, hspace=.22, wspace=.16)
        fig.text(.055, heading, rf"$m={degree}$", fontsize=21, fontweight="bold", color=INK)
        for row, (name, initial, label) in enumerate(row_settings):
            low, high = data[name].min(), data[name].max()
            if initial is not None:
                low, high = min(low, initial), max(high, initial)
            padding = .09 * max(high - low, .1)
            for power in range(half + 1):
                ax = fig.add_subplot(grid[row, power])
                for coordinate, (factor, order) in enumerate(data["coordinates"]):
                    if order == power:
                        ax.plot(steps, data[name][:, coordinate], color=COLORS[factor], lw=1.9,
                                marker="o", markersize=2.4, markeredgewidth=0)
                if initial is not None:
                    ax.axhline(initial, color="#929ba5", lw=.8, ls=(0, (4, 3)), zorder=0)
                    ax.scatter([0], [initial], color="#616d79", marker="D", s=18,
                               zorder=4, clip_on=False)
                ax.set_xlim(0, steps[-1])
                ax.set_ylim(low - padding, high + padding)
                ax.set_xticks([0, 2500, 5000, 7500])
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.grid(axis="y", color="#e4e8eb", linewidth=.65)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["bottom", "left"]].set_color("#b2bac3")
                ax.tick_params(colors="#4d5965", labelsize=9,
                               labelbottom=row == 2, labelleft=power == 0)
                if row == 0:
                    suffix = r" · $q_1$ only" if power == half else ""
                    ax.set_title(rf"$r={power}$" + suffix, loc="left",
                                 fontsize=12, color=INK, pad=9)
                if power == 0:
                    ax.set_ylabel(label, fontsize=13, color=INK, labelpad=12)
                if row == 2:
                    ax.set_xlabel("Training update", fontsize=10)
    fig.add_artist(Line2D([.055, .985], [.495, .495], transform=fig.transFigure,
                          color="#dbe2e8", lw=1.2))
    fig.suptitle(r"Transformer++ · SoS · learned $a$, $b$, and $W$ norms",
                 x=.055, y=.98, ha="left", color=INK, fontsize=25, fontweight="bold")
    handles = [
        Line2D([], [], color=COLORS["U"], lw=2, marker="o", markersize=4,
               label=r"$q_1$ (degree $m/2$)"),
        Line2D([], [], color=COLORS["V"], lw=2, marker="o", markersize=4,
               label=r"$q_2$ (degree $m/2-1$)"),
    ]
    handles.append(Line2D([], [], color="#929ba5", ls="--", marker="D", markersize=4,
                           label="Analytic initialization: a = 0, b = 1"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .945),
               frameon=False, ncol=3, fontsize=12, columnspacing=2.6)
    _save(fig, out, "head_parameters_combined")


def plot(root: Path, out: Path, selection: dict) -> None:
    settings = selection["settings"]["single_head_parameters"]
    datasets = [_load(root, entry, settings) for entry in selection["runs"]]
    summary = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        _combined_figure(datasets, out)
        for data in datasets:
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
