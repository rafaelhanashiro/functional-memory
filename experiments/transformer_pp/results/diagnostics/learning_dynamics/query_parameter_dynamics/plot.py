"""One middle-layer head's regular query-polynomial parameter histories.

The raw learned parameters obey c_r(x)=(b_r+a_r*tanh(W_r^T x))/r!.
They are indexed by actual polynomial order, unlike SoS factor generators.
Inferred tanh moments in the CSV use the saved, rounded coefficient moments;
they are not direct activation measurements. No checkpoint or GPU is needed.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scaffolds.records import load_run
from scaffolds.records.selections import verify_pinned_runs


BLUE = "#28699b"
ORANGE = "#d66a32"
INK = "#24323d"


def load(root: Path, entry: dict, settings: dict) -> dict:
    run = root / "experiments" / entry["study"] / "runs" / entry["run_id"]
    model = yaml.safe_load((run / "config.yaml").read_text())["model"]
    record = load_run(run)
    assert record["status"] == "complete", entry["run_id"]
    degree = int(entry["degree"])
    assert degree in (8, 12) and model["attention_degree"] == degree
    for key, value in settings["model_shape"].items():
        assert model[key] == value, (entry["run_id"], key)
    required = dict(
        attention_coefficient_parameterization="direct", attention_coefficients="learned",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_readout="diagonal", attention_coefficient_readout_scale=1.0,
        attention_coefficient_initialization="taylor", attention_coefficient_amplitude_initial=0.0,
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_kernel_weight_decay=True, attention_coefficient_tanh_bias=False,
        attention_forget_gate=False, positional_encoding="rope", attention_backend="mosaic",
        attention_normalization=False, attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
        qk_norm_initial_scale=None, qk_norm_head_scale=True,
        initialization="transformer_truncated_reference",
    )
    for key, value in required.items():
        assert model[key] == value, (entry["run_id"], key, model[key], value)
    assert model.get("attention_coefficient_gate", "none") == "none"
    assert model.get("attention_score_transform", "identity") == "identity"
    assert model.get("attention_score_bias", "none") == "none"
    config = yaml.safe_load((run / "config.yaml").read_text())
    assert config["optim"]["weight_decay"] == 0.1
    layer, head = settings["layer_index"], settings["head_index"]
    assert layer == model["num_layers"] // 2 and 0 <= head < model["num_heads"]
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    steps = np.asarray([row["step"] for row in rows], dtype=int)
    assert len(steps) == settings["snapshots"]
    assert steps[0] == settings["first_step"] and steps[-1] == settings["terminal_step"]
    assert np.all(np.diff(steps) > 0)
    diagnostics = [row["attention_diagnostics"] for row in rows]
    assert all(report["valid"] for report in diagnostics)
    samples = [report["sample"] for report in diagnostics]
    assert all(sample == samples[0] for sample in samples)
    sample = samples[0]
    assert sample["selection"] == "first_heldout_sequence"
    assert sample["num_sequences"] == 1 and sample["prediction_tokens"] == 2048
    assert sample["record_sha256"] and sample["eval_manifest_sha256"]
    reports = [report["coefficients"] for report in diagnostics]
    assert all(report["finite"] and report["sample"] == "fixed_diagnostic_sample" for report in reports)
    assert all(report["modulation_amplitude_scale"] == report["projection_bias_scale"] == "1/r!"
               for report in reports)
    shape = (len(rows), model["num_layers"], model["num_heads"], degree + 1)

    def array(field):
        values = np.asarray([report[field] for report in reports], dtype=float)
        assert values.shape == shape and np.isfinite(values).all(), (entry["run_id"], field)
        return values

    arrays = {name: array(field) for name, field in settings["parameter_fields"].items()}
    assert set(arrays) == {"a", "b", "w_l2"} and (arrays["w_l2"] >= 0).all()
    factorials = np.asarray([math.factorial(order) for order in range(degree + 1)], dtype=float)
    for name, effective in (("a", "effective_modulation_amplitude"), ("b", "effective_projection_bias")):
        np.testing.assert_allclose(array(effective), arrays[name] / factorials, rtol=1e-12, atol=1e-20)
    moments = {}
    for name in ("mean", "std"):
        values = np.asarray([report["effective_summary"][name] for report in reports], dtype=float)
        assert values.shape == shape and np.isfinite(values).all(), (entry["run_id"], name)
        if name == "std":
            assert (values >= 0).all()
        moments[name] = values[:, layer, head, :]
    return dict(entry=entry, degree=degree, steps=steps, layer=layer, head=head,
                layers=model["num_layers"], heads=model["num_heads"], input_dimension=model["hidden_size"],
                sample=sample, factorials=factorials, **moments,
                **{name: values[:, layer, head, :] for name, values in arrays.items()})


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def combined_figure(datasets: list[dict], out: Path) -> None:
    """Match the combined SoS presentation, using direct polynomial orders here."""
    datasets = sorted(datasets, key=lambda data: data["degree"])
    assert [data["degree"] for data in datasets] == [8, 12]
    assert len({(data["layer"], data["head"]) for data in datasets}) == 1
    fig = plt.figure(figsize=(26, 11.5))
    row_settings = (("a", 0., "Amplitude a", BLUE), ("b", 1., "Bias b", ORANGE),
                    ("w_l2", None, r"$\|W_r\|_2$", BLUE))
    for data, (bottom, top, heading) in zip(datasets, ((.532, .846, .887), (.054, .41, .451))):
        degree, steps = data["degree"], data["steps"]
        grid = fig.add_gridspec(3, degree + 1, left=.055, right=.985,
                               bottom=bottom, top=top, hspace=.22, wspace=.16)
        fig.text(.045, heading, rf"$m={degree}$", fontsize=21, fontweight="bold", color=INK)
        for row, (name, initial, label, color) in enumerate(row_settings):
            low, high = data[name].min(), data[name].max()
            if initial is not None:
                low, high = min(low, initial), max(high, initial)
            padding = .09 * max(high - low, .1)
            for order in range(degree + 1):
                ax = fig.add_subplot(grid[row, order])
                ax.plot(steps, data[name][:, order], color=color, lw=1.9,
                        marker="o", markersize=2.4, markeredgewidth=0)
                if initial is not None:
                    ax.axhline(initial, color="#929ba5", lw=.8, ls=(0, (4, 3)), zorder=0)
                    ax.scatter([0], [initial], color="#616d79", marker="D", s=18,
                               zorder=4, clip_on=False)
                ax.set_xlim(0, steps[-1])
                ax.set_ylim(low - padding, high + padding)
                ax.set_xticks([0, 2500, 5000, 7500], labels=["0", "2.5k", "5k", "7.5k"])
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.grid(axis="y", color="#e4e8eb", linewidth=.65)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["bottom", "left"]].set_color("#b2bac3")
                ax.tick_params(colors="#4d5965", labelsize=10,
                               labelbottom=row == 2, labelleft=order == 0)
                if row == 0:
                    ax.set_title(rf"$r={order}$", loc="left", fontsize=13, color=INK, pad=9)
                if order == 0:
                    ax.set_ylabel(label, fontsize=13, color=INK, labelpad=12)
                if row == 2:
                    ax.set_xlabel("Training update", fontsize=10)
                    ax.get_xticklabels()[0].set_ha("left")
                    ax.get_xticklabels()[-1].set_ha("right")
    fig.add_artist(Line2D([.045, .985], [.483, .483], transform=fig.transFigure,
                          color="#dbe2e8", lw=1.2))
    fig.suptitle(r"Transformer++ · Regular query · learned $a$, $b$, and $W$ norms",
                 x=.045, y=.98, ha="left", color=INK, fontsize=25, fontweight="bold")
    handles = [Line2D([], [], color=color, lw=2, marker="o", markersize=4, label=label)
               for color, label in ((BLUE, "Amplitude a and W norms"), (ORANGE, "Bias b"))]
    handles.append(Line2D([], [], color="#929ba5", ls="--", marker="D", markersize=4,
                           label="Analytic initialization: a = 0, b = 1"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.04, .945),
               frameon=False, ncol=3, fontsize=12, columnspacing=2.6)
    save(fig, out, "head_parameters_combined")


def summary_rows(data: dict, settings: dict) -> list[dict]:
    rows = []
    threshold = float(settings["derived_tanh_moments"]["minimum_absolute_amplitude"])
    assert threshold > 0
    for order in range(data["degree"] + 1):
        row = dict(run_id=data["entry"]["run_id"], degree=data["degree"],
                   layer_index=data["layer"], head_index=data["head"], polynomial_order=order,
                   snapshots=len(data["steps"]), first_step=int(data["steps"][0]),
                   terminal_step=int(data["steps"][-1]), analytic_initial_a=0., analytic_initial_b=1.)
        for name in ("a", "b", "w_l2"):
            values = data[name][:, order]
            row.update({f"{name}_first_recorded": values[0], f"{name}_terminal": values[-1],
                        f"{name}_recorded_min": values.min(), f"{name}_recorded_max": values.max()})
        mean, std = data["mean"][-1, order], data["std"][-1, order]
        scale = data["factorials"][order]
        amplitude, bias = data["a"][-1, order], data["b"][-1, order]
        infer = abs(amplitude) >= threshold
        row.update(terminal_coefficient_mean=mean, terminal_coefficient_std=std,
                   terminal_factorial_scaled_coefficient_std=scale * std,
                   terminal_tanh_moments_inferred=infer,
                   terminal_inferred_tanh_mean=(scale * mean - bias) / amplitude if infer else None,
                   terminal_inferred_tanh_std=scale * std / abs(amplitude) if infer else None)
        rows.append(row)
    return rows


def main() -> None:
    selection = yaml.safe_load((OUT / "selection.yaml").read_text())
    verify_pinned_runs(ROOT, selection["runs"])
    datasets = [load(ROOT, entry, selection["settings"]) for entry in selection["runs"]]
    assert all(data["sample"] == datasets[0]["sample"] for data in datasets)
    summary = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        combined_figure(datasets, OUT)
        for data in datasets:
            summary.extend(summary_rows(data, selection["settings"]))
    with (OUT / "parameter_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({key: format(value, ".12g") if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})
    print("Regenerated regular-query parameter diagnostics from pinned, completed runs.")


if __name__ == "__main__":
    main()
