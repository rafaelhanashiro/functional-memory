"""Render the frozen-checkpoint activation probe from compact run measurements.

All plotted distributions are measured summaries of the selected sequence.
No checkpoint, dataset, numerical archive, GPU or live cluster path is read.
The bars display quantiles, not uncertainty estimates or confidence intervals.
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
from matplotlib.patches import Patch
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
RED = "#bd4947"
QUANTILES = (.01, .25, .5, .75, .99)


def quantile(summary: dict, fraction: float) -> float:
    values = {float(key): value for key, value in summary["quantiles"].items()}
    return float(values[fraction])


def check_summary(summary: dict) -> None:
    values = [float(summary[key]) for key in ("mean", "std", "min", "max")]
    assert all(math.isfinite(value) for value in values)
    mean, std, low, high = values
    assert std >= 0 and low <= mean <= high
    q = [quantile(summary, fraction) for fraction in QUANTILES]
    assert all(math.isfinite(value) for value in q)
    assert low <= q[0] <= q[1] <= q[2] <= q[3] <= q[4] <= high


def load(selection: dict) -> list[dict]:
    verify_pinned_runs(ROOT, selection["runs"])
    probes = [entry for entry in selection["runs"] if entry["role"] == "activation_probe"]
    assert len(probes) == 1
    probe = probes[0]
    run = ROOT / "experiments" / probe["study"] / "runs" / probe["run_id"]
    if load_run(run)["status"] != "complete":
        raise RuntimeError("The activation probe has not completed; no figures generated.")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    parents = [entry for entry in selection["runs"] if entry["role"] != "activation_probe"]
    assert len(rows) == len(parents) == 2
    by_role = {row["role"]: row for row in rows}
    assert len(by_role) == len(rows)
    settings = selection["settings"]
    assert settings["plotted_quantiles"] == list(QUANTILES)
    assert settings["saturation_absolute_tanh_threshold"] == .99
    result = []
    for parent in sorted(parents, key=lambda entry: entry["degree"]):
        row = by_role[parent["role"]]
        assert row["source_run_id"] == parent["run_id"]
        assert row["checkpoint_step"] == parent["checkpoint_step"] == settings["checkpoint_step"]
        assert row["layer"] == settings["layer_index"] and row["head"] == settings["head_index"]
        assert row["dimension"] == settings["input_dimension"]
        assert row["degree"] == parent["degree"]
        assert row["indexing"] == "zero_based"
        assert row["sequences"] == settings["sequences"]
        assert row["prediction_tokens"] == settings["prediction_tokens"]
        assert row["sample"]["selection"] == settings["sample_selection"]
        assert row["sample"]["num_sequences"] == settings["sequences"]
        assert row["sample"]["prediction_tokens"] == settings["prediction_tokens"]
        assert row["sample"]["record_sha256"] == settings["sample_record_sha256"]
        assert row["optimizer_updates"] == 0 and row["parent_parameters_unchanged"]
        assert row["parent_parameter_sha256_before"] == row["parent_parameter_sha256_after"]
        assert row["reference_validation"]["checkpoint_parameters_match"]
        validation = row["native_capture_validation"]
        assert validation["rmsnorm_output_equals_generator_input"]
        assert validation["manual_coefficient_max_normalized_error"] <= validation["manual_coefficient_tolerance"]
        for name in ("x_norm_before", "x_norm_after", "gain"):
            check_summary(row[name])
        assert row["x_norm_before"]["min"] >= 0 and row["x_norm_after"]["min"] >= 0
        degrees = row["degrees"]
        assert [value["degree"] for value in degrees] == list(range(row["degree"] + 1))
        for value in degrees:
            for field in ("w_norm", "gain_weighted_w_norm"):
                assert math.isfinite(value[field]) and value[field] >= 0
            for name in ("z", "tanh", "tanh_derivative", "alignment_cos"):
                check_summary(value[name])
            fractions = [value[name] for name in ("saturation_abs_tanh_ge099",
                         "saturation_tanh_ge099", "saturation_tanh_le_neg099")]
            assert all(math.isfinite(fraction) and 0 <= fraction <= 1 for fraction in fractions)
            assert math.isclose(fractions[0], fractions[1] + fractions[2], rel_tol=0, abs_tol=1e-12)
        result.append(row)
    assert result[0]["sample"] == result[1]["sample"]
    return result


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b2bac3")
    ax.tick_params(colors="#4d5965", labelsize=9)
    ax.grid(axis="y", color="#e4e8eb", lw=.65, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def interval(ax, position: float, summary: dict, *, color: str, width: float = .19) -> None:
    low, q1, median, q3, high = [quantile(summary, fraction) for fraction in QUANTILES]
    ax.vlines(position, low, high, color=color, lw=1.2, alpha=.7, zorder=2)
    ax.hlines([low, high], position - width / 2, position + width / 2, color=color, lw=1, zorder=2)
    ax.vlines(position, q1, q3, color=color, lw=7, alpha=.55, zorder=3)
    ax.scatter([position], [median], s=23, color=color, edgecolors="white", linewidths=.5, zorder=4)


def plot(rows: list[dict], settings: dict) -> None:
    threshold = math.atanh(settings["saturation_absolute_tanh_threshold"])
    z_low = min(-threshold, *(quantile(value["z"], .01) for row in rows for value in row["degrees"]))
    z_high = max(threshold, *(quantile(value["z"], .99) for row in rows for value in row["degrees"]))
    z_pad = .08 * max(z_high - z_low, 1.)
    x_low = min(quantile(row[name], .01) for row in rows for name in ("x_norm_before", "x_norm_after"))
    x_high = max(quantile(row[name], .99) for row in rows for name in ("x_norm_before", "x_norm_after"))
    x_pad = .08 * max(x_high - x_low, 1.)
    w_high = max(value[name] for row in rows for value in row["degrees"]
                 for name in ("w_norm", "gain_weighted_w_norm"))
    fig = plt.figure(figsize=(15.5, 10.3))
    grid = fig.add_gridspec(2, 3, width_ratios=(.85, 1.22, 1.06), left=.065, right=.985,
                          top=.82, bottom=.145, hspace=.38, wspace=.29)
    for i, row in enumerate(rows):
        left = grid[i, 0].subgridspec(2, 1, hspace=.67)
        norm_ax, w_ax = fig.add_subplot(left[0]), fig.add_subplot(left[1])
        z_ax, sat_ax = fig.add_subplot(grid[i, 1]), fig.add_subplot(grid[i, 2])
        for ax in (norm_ax, w_ax, z_ax, sat_ax):
            style(ax)
        norm_ax.set_title(rf"$m={row['degree']}$ · activation and weight norms", loc="left", fontsize=12,
                          fontweight="bold", color=INK, pad=9)
        for position, name, color in ((0, "x_norm_before", "#7b8793"), (1, "x_norm_after", BLUE)):
            interval(norm_ax, position, row[name], color=color, width=.28)
        norm_ax.set_xticks([0, 1], labels=["Before RMSNorm", "After RMSNorm"])
        norm_ax.set_xlim(-.65, 1.65)
        norm_ax.set_ylim(max(0, x_low - x_pad), x_high + x_pad)
        norm_ax.set_ylabel(r"Token $\ell_2$ norm", fontsize=10)
        orders = np.asarray([value["degree"] for value in row["degrees"]])
        for name, color, label in (("w_norm", BLUE, r"$\|W_r\|_2$"),
                                   ("gain_weighted_w_norm", ORANGE, r"$\|g\odot W_r\|_2$")):
            w_ax.plot(orders, [value[name] for value in row["degrees"]], color=color, lw=1.8,
                      marker="o", markersize=3, label=label)
        w_ax.set_ylim(0, 1.15 * max(w_high, .01))
        w_ax.set_ylabel("Vector norm", fontsize=10)
        w_ax.set_xlabel("Polynomial order r", fontsize=10)
        w_ax.set_xticks(orders)
        w_ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9, handlelength=1.4,
                    columnspacing=1.0, borderaxespad=.1)

        z_ax.set_title(rf"$m={row['degree']}$ · actual $z_r=W_r^\top x$", loc="left", fontsize=12,
                       fontweight="bold", color=INK, pad=9)
        for order, value in zip(orders, row["degrees"], strict=True):
            interval(z_ax, order, value["z"], color=BLUE)
        z_ax.axhline(0, color="#9aa4ad", lw=.7, zorder=0)
        for sign in (-1, 1):
            z_ax.axhline(sign * threshold, color=RED, lw=1.1, ls=(0, (4, 3)), zorder=1)
        z_ax.set_ylim(z_low - z_pad, z_high + z_pad)
        z_ax.set_xlim(-.65, row["degree"] + .65)
        z_ax.set_xticks(orders)
        z_ax.set_xlabel("Polynomial order r", fontsize=10)
        z_ax.set_ylabel("Projection z", fontsize=10)

        positive = np.asarray([value["saturation_tanh_ge099"] for value in row["degrees"]]) * 100
        negative = np.asarray([value["saturation_tanh_le_neg099"] for value in row["degrees"]]) * 100
        sat_ax.bar(orders, 100, color="#eff2f5", width=.7, zorder=1)
        sat_ax.bar(orders, positive, color=ORANGE, width=.7, zorder=2)
        sat_ax.bar(orders, negative, bottom=positive, color=BLUE, width=.7, zorder=2)
        sat_ax.set_title(rf"$m={row['degree']}$ · saturated query tokens", loc="left", fontsize=12,
                         fontweight="bold", color=INK, pad=9)
        sat_ax.set_ylim(0, 100)
        sat_ax.set_yticks([0, 25, 50, 75, 100])
        sat_ax.set_xlim(-.65, row["degree"] + .65)
        sat_ax.set_xticks(orders)
        sat_ax.set_xlabel("Polynomial order r", fontsize=10)
        sat_ax.set_ylabel("Share of tokens (%)", fontsize=10)
    fig.suptitle("Transformer++ · query coefficients · activation and saturation diagnostic",
                 x=.055, y=.978, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(.055, .932,
             f"Layer {settings['layer_index'] + 1}, head {settings['head_index'] + 1} · checkpoint {settings['checkpoint_step']:,} "
             f"· first held-out sequence ({settings['prediction_tokens']:,} queries) · {settings['input_dimension']} input channels",
             fontsize=12, color="#5a6672")
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.2, markersize=4,
                      label="Distribution: median, thick IQR, thin 1–99%"),
               Line2D([], [], color=RED, ls="--", lw=1.1,
                      label=rf"$|\tanh z|=0.99$ at $z=\pm {threshold:.3f}$"),
               Patch(facecolor=ORANGE, label=r"$\tanh z\geq0.99$"),
               Patch(facecolor=BLUE, label=r"$\tanh z\leq-0.99$")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .901), frameon=False,
               ncol=2, fontsize=10, columnspacing=2.0)
    fig.text(.055, .025,
             "x is the gain-applied RMSNorm output used by the coefficient generator; g is its learned channel gain. "
             "The two norm axes have separate scales.\n"
             "Quantile intervals summarize tokens, not uncertainty. Saturation uses all query tokens, including tails outside the plotted 1–99% intervals.\n"
             "This is one frozen checkpoint and sequence; it does not establish behavior throughout training or on all data.",
             fontsize=9.2, color="#687583", linespacing=1.55)
    fig.savefig(OUT / "activation_probe.png", dpi=180, facecolor="white")
    fig.savefig(OUT / "activation_probe.pdf", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def save_summary(rows: list[dict], settings: dict) -> None:
    summary = []
    for row in rows:
        for value in row["degrees"]:
            result = dict(source_run_id=row["source_run_id"], model_degree=row["degree"],
                          checkpoint_step=row["checkpoint_step"], layer_index=row["layer"], head_index=row["head"],
                          polynomial_order=value["degree"], w_norm=value["w_norm"],
                          gain_weighted_w_norm=value["gain_weighted_w_norm"])
            for name in ("x_norm_before", "x_norm_after"):
                for statistic in ("mean", "std", "min", "max"):
                    result[f"{name}_{statistic}"] = row[name][statistic]
                for q in QUANTILES:
                    result[f"{name}_q{q:g}"] = quantile(row[name], q)
            for name in ("z", "tanh", "tanh_derivative", "alignment_cos"):
                for statistic in ("mean", "std", "min", "max"):
                    result[f"{name}_{statistic}"] = value[name][statistic]
                for q in QUANTILES:
                    result[f"{name}_q{q:g}"] = quantile(value[name], q)
            for name in ("saturation_abs_z_gt2", "saturation_abs_z_gt3", "saturation_abs_tanh_ge099",
                         "saturation_tanh_ge099", "saturation_tanh_le_neg099"):
                result[name] = value[name]
            result["saturation_absolute_tanh_threshold"] = settings["saturation_absolute_tanh_threshold"]
            summary.append(result)
    with (OUT / "activation_probe_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({key: format(value, ".12g") if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})


def main() -> None:
    selection = yaml.safe_load((OUT / "probe_selection.yaml").read_text())
    rows = load(selection)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        plot(rows, selection["settings"])
    save_summary(rows, selection["settings"])
    print("Regenerated activation probe from pinned compact measurements.")


if __name__ == "__main__":
    main()
