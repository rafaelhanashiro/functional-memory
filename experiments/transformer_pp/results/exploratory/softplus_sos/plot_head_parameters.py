"""Prefactor SoS factor-generator histories for explicitly selected no-L2 runs.

Regenerate with this script from the repository root. The figure matches the
original SoS head-parameter layout; it plots raw a/b and vector norms, not
expanded polynomial coefficients or norms of effective coefficients.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scaffolds.records.selections import verify_pinned_runs

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
    assert model["attention_degree"] == degree + 1
    assert model["attention_coefficient_input_power"] == 1
    assert model["attention_score_transform"] == "softplus"
    assert model.get("attention_coefficient_input_normalization", "none") == "none"
    assert model.get("attention_coefficient_input_source", "post_rmsnorm") == "post_rmsnorm"
    assert model["positional_encoding"] == "rope"
    assert model["attention_coefficient_kernel_weight_decay"]
    assert model["initialization"] != "flax_default"
    assert json.loads((run / "record.json").read_text())["status"] == "complete"
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
    assert len(steps) == 28 and steps[0] == 100 and steps[-1] == 7570
    assert np.all(np.diff(steps) > 0)
    assert all(row["attention_diagnostics"]["valid"] for row in rows)
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
                coordinates=coordinates, input_dimension=model["hidden_size"],
                best_ppl=json.loads((run / "best_validation.json").read_text())["validation_perplexity"],
                **parameters)



def _combined_figure(datasets: list[dict], out: Path) -> None:
    """The same a/b/W histories, grouped into one three-row section per SoS degree."""
    datasets = sorted(datasets, key=lambda data: data["degree"])
    assert [data["degree"] for data in datasets] == [8, 12]
    assert len({(data["layer"], data["head"]) for data in datasets}) == 1
    fig = plt.figure(figsize=(20, 13.8))
    row_settings = (("a", 0., "Amplitude a"), ("b", 1., "Bias b"),
                    ("w_l2", None, r"$\|W_j\|_2$"))
    for data, (bottom, top, heading) in zip(datasets, ((.555, .835, .865), (.13, .445, .475))):
        degree, steps = data["degree"], data["steps"]
        half = degree // 2
        grid = fig.add_gridspec(3, half + 1, left=.065, right=.985,
                               bottom=bottom, top=top, hspace=.22, wspace=.16)
        fig.text(.055, heading, rf"$m={degree}$", fontsize=21, fontweight="bold", color=INK)
        fig.text(.145, heading, f"U degree {half} · V degree {half - 1} · Total degree {degree + 1} · Best PPL {data['best_ppl']:.4f}", fontsize=13, color="#5a6672")
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
                    suffix = " · U only" if power == half else ""
                    ax.set_title(rf"Factor power $j={power}$" + suffix, loc="left",
                                 fontsize=12, color=INK, pad=9)
                if power == 0:
                    ax.set_ylabel(label, fontsize=13, color=INK, labelpad=12)
                if row == 2:
                    ax.set_xlabel("Training update", fontsize=10)
    fig.add_artist(Line2D([.055, .985], [.505, .505], transform=fig.transFigure,
                          color="#dbe2e8", lw=1.2))
    fig.suptitle(r"Transformer++ · SoS + softplus + prefactor · learned $a$, $b$, and $W$ norms",
                 x=.055, y=.98, ha="left", color=INK, fontsize=23, fontweight="bold")
    data = datasets[0]
    fig.text(.055, .94,
             rf"Layer {data['layer'] + 1}, head {data['head'] + 1} · $z=\mathrm{{softplus}}(s)$ · $w_t(s)=z[U_t(z)^2+V_t(z)^2]$"
             rf" · $F_j(x)=f_j[1+b_j+a_j\tanh(W_j^\top x)]/2$",
             fontsize=13, color="#5a6672")
    handles = [Line2D([], [], color=COLORS[factor], lw=2, marker="o", markersize=4,
                      label=f"{factor} factor") for factor in ("U", "V")]
    handles.append(Line2D([], [], color="#929ba5", ls="--", marker="D", markersize=4,
                           label="Analytic initialization: a = 0, b = 1"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .923),
               frameon=False, ncol=3, fontsize=12, columnspacing=2.6)
    fig.text(.055, .035,
             "Original coefficient input (no L2). Raw SoS factor-generator parameters; no factorial or Taylor-factor rescaling. "
             "W norms over 896 input channels.\n"
             "m is the SoS core degree; total degree is m + 1. Shared linear y-scale within each section's row. "
             "28 snapshots, updates 100–7,570.\n"
             "Step-zero a/b diamonds are analytic; "
             "no step-zero W norm was saved. Lines connect recorded snapshots.",
             color="#687583", fontsize=11, linespacing=1.7)
    _save(fig, out, "head_parameters_combined")


def main() -> None:
    selection = yaml.safe_load((OUT / "selection.yaml").read_text())
    settings = selection["settings"]["single_head_parameters"]
    roles = settings["roles"]
    entries = [dict(entry, degree=roles[entry["role"]])
               for entry in selection["runs"] if entry["role"] in roles]
    assert len(entries) == 2 and sorted(entry["degree"] for entry in entries) == [8, 12]
    verify_pinned_runs(ROOT, entries)
    datasets = [_load(ROOT, entry, settings) for entry in entries]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        _combined_figure(datasets, OUT)
    for data in datasets:
        print(f"core m={data['degree']}: {data['entry']['run_id']}, "
              f"{len(data['steps'])} snapshots, layer {data['layer']+1}, head {data['head']+1}")


if __name__ == "__main__":
    main()
