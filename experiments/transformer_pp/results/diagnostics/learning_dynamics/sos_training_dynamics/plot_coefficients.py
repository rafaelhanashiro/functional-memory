"""Actual expanded SoS coefficient histories from pinned training measurements.

The caller verifies selection pins. Each recorded summary reduces the same held-out
sequence's query positions, retaining layer, head, and polynomial-order axes. No
generator parameter is substituted for an effective polynomial coefficient.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import yaml


BLUE = "#28699b"
INK = "#24323d"


def _load(root: Path, entry: dict) -> dict:
    run = root / "experiments" / entry["study"] / "runs" / entry["run_id"]
    config = yaml.safe_load((run / "config.yaml").read_text())
    model = config["model"]
    degree = int(entry["degree"])
    assert model["attention_degree"] == degree
    assert model["attention_coefficient_parameterization"] == "sum_of_squares"
    assert model["attention_coefficient_conditioning"] == "query"
    assert model["attention_coefficient_initialization"] == "taylor"
    assert model["attention_coefficient_amplitude_initial"] == 0.0
    assert model["attention_coefficient_scale"] == "none"
    assert model["attention_coefficient_amplitude_scale"] == "none"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    assert rows, f"No measurements: {run}"
    steps = np.asarray([row["step"] for row in rows], dtype=int)
    assert np.all(np.diff(steps) > 0), f"Non-increasing coefficient steps: {run}"
    samples = [row["attention_diagnostics"]["sample"] for row in rows]
    assert all(sample == samples[0] for sample in samples), f"Diagnostic sample changed: {run}"
    sample = samples[0]
    assert sample["num_sequences"] == 1 and sample["prediction_tokens"] == 2048
    assert sample["selection"] == "first_heldout_sequence"
    assert sample["record_sha256"] and sample["eval_manifest_sha256"]
    reports = [row["attention_diagnostics"]["coefficients"] for row in rows]
    assert all(report["finite"] for report in reports), f"Nonfinite coefficient report: {run}"
    assert all(report["sample"] == "fixed_diagnostic_sample" for report in reports)
    shape = (len(rows), model["num_layers"], model["num_heads"], degree + 1)
    summaries = {}
    for name in ("mean", "std", "min", "max"):
        values = np.asarray([report["effective_summary"][name] for report in reports], dtype=float)
        assert values.shape == shape, (run, name, values.shape, shape)
        assert np.isfinite(values).all(), f"Nonfinite {name}: {run}"
        summaries[name] = values
    assert np.all(summaries["std"] >= 0)
    tolerance = 5e-6 * np.maximum(np.abs(summaries["min"]), np.abs(summaries["max"])) + 1e-30
    assert np.all(summaries["min"] <= summaries["max"])
    assert np.all(summaries["mean"] >= summaries["min"] - tolerance)
    assert np.all(summaries["mean"] <= summaries["max"] + tolerance)
    return dict(entry=entry, degree=degree, steps=steps, sample=sample, **summaries)


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{name}.pdf", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _histories(data: dict, out: Path) -> None:
    degree, steps = data["degree"], data["steps"]
    n_layers, n_heads = data["mean"].shape[1:3]
    columns = 3 if degree <= 8 else 4
    rows = math.ceil((degree + 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15.2, 2.65 * rows + 1.9), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for order, ax in enumerate(axes):
        if order > degree:
            ax.set_visible(False)
            continue
        # These are statistics ACROSS token means, not token-level quantiles.
        means = data["mean"][..., order].reshape(len(steps), -1)
        low, median, high = np.quantile(means, [0.1, 0.5, 0.9], axis=1)
        ax.plot(steps, means, color=BLUE, alpha=0.075, lw=0.6, rasterized=True)
        ax.fill_between(steps, low, high, color=BLUE, alpha=0.21, linewidth=0)
        ax.plot(steps, median, color=BLUE, lw=2.0)
        initial = 1.0 / math.factorial(order)
        ax.axhline(initial, color="#7c8188", lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.scatter([0], [initial], marker="D", s=22, color="#434950", zorder=5, clip_on=False)
        ax.axhline(0, color="#c2c7cc", lw=0.6, zorder=0)
        ax.set_title(rf"$c_{{{order}}}(x_t)$  ·  order {order}", fontsize=12, loc="left", color=INK)
        ax.set_xlim(0, steps[-1])
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        ax.grid(axis="y", color="#e4e8eb", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#b2bac3")
        ax.tick_params(colors="#4d5965", labelsize=9)
        if order // columns == rows - 1 or order + columns > degree:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Training update", fontsize=10)
    fig.suptitle(rf"Transformer++ · SoS $m={degree}$ · actual polynomial coefficients",
                 x=0.055, y=0.982, ha="left", color=INK, fontsize=19, fontweight="bold")
    fig.text(0.055, 0.944, rf"$p_t(s)=c_0(x_t)+\cdots+c_{{{degree}}}(x_t)s^{{{degree}}}$"
             f"   ·   {n_layers} layers × {n_heads} heads   ·   same 2,048 held-out queries at every measurement",
             fontsize=11, color="#5a6672")
    handles = [Line2D([], [], color=BLUE, alpha=0.25, lw=1, label="Each layer/head: token mean"),
               Patch(facecolor=BLUE, alpha=0.21, label="10–90% across layer/head means"),
               Line2D([], [], color=BLUE, lw=2, label="Median across layer/head means"),
               Line2D([], [], color="#7c8188", ls="--", marker="D", markersize=4,
                      label=r"Analytic initialization: $1/r!$")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.922),
               frameon=False, ncol=2, fontsize=10, columnspacing=2.8)
    fig.text(0.055, 0.019,
             "Raw expanded coefficients; no extra factorial rescaling. Each panel has its own linear y-axis.\n"
             f"Step 0: analytic marker only. Curves: {len(steps)} measurements from update {steps[0]:,}. Bands summarize layer/head means, not the full token range.",
             color="#687583", fontsize=9.5, linespacing=1.6)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.105, top=0.825, hspace=0.44, wspace=0.27)
    _save(fig, out, f"coefficients_m{degree}")


def _final_heatmap(data: dict, out: Path) -> None:
    degree = data["degree"]
    means = data["mean"][-1]
    layers, heads, _ = means.shape
    factorials = np.asarray([math.factorial(order) for order in range(degree + 1)], dtype=float)
    values = (means * factorials).reshape(layers * heads, degree + 1)
    limit = max(1.0, float(np.max(np.abs(values))))
    fig, ax = plt.subplots(figsize=(10.4, 9.2))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                      norm=SymLogNorm(linthresh=1.0, vmin=-limit, vmax=limit, base=10))
    for boundary in range(heads, layers * heads, heads):
        ax.axhline(boundary - 0.5, color="white", lw=0.75)
    ax.set_yticks(np.arange(layers) * heads + (heads - 1) / 2, labels=np.arange(1, layers + 1))
    ax.set_xticks(np.arange(degree + 1))
    ax.set_ylabel(f"Layer (within each layer: heads 1–{heads}, top to bottom)", fontsize=10)
    ax.set_xlabel("Polynomial order r", fontsize=11)
    ax.set_title(rf"Transformer++ · SoS $m={degree}$ · final coefficient structure",
                 fontsize=16, loc="left", pad=35, fontweight="bold", color=INK)
    ax.text(0, 1.025, rf"Taylor-relative token means: $r!\,\overline{{c}}_r$"
            f"   ·   update {data['steps'][-1]:,}   ·   initialization = 1",
            transform=ax.transAxes, fontsize=10.5, color="#5a6672")
    colorbar = fig.colorbar(image, ax=ax, pad=0.035, fraction=0.045)
    colorbar.set_label(r"$r!\,\overline{c}_r$ (signed symmetric-log color scale)", fontsize=10)
    fig.text(0.085, 0.025,
             "Each cell averages the same 2,048 query tokens. Negative coefficients are compatible with a sum-of-squares polynomial.\n"
             "This rescaled view exposes relative order structure; the coefficient-history figures show the actual unscaled coefficients.",
             fontsize=9, color="#687583", linespacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.91, bottom=0.115, top=0.88)
    _save(fig, out, f"coefficients_m{degree}_final_taylor_relative")


def _summary_rows(data: dict) -> list[dict]:
    rows = []
    for order in range(data["degree"] + 1):
        means = data["mean"][-1, ..., order].reshape(-1)
        stds = data["std"][-1, ..., order].reshape(-1)
        minima = data["min"][-1, ..., order].reshape(-1)
        maxima = data["max"][-1, ..., order].reshape(-1)
        mean = means.mean()
        # All layer/head groups summarize the same number of query tokens.
        pooled_std = np.sqrt(np.mean(stds ** 2 + (means - mean) ** 2))
        rows.append(dict(
            run_id=data["entry"]["run_id"], degree=data["degree"], order=order,
            step=int(data["steps"][-1]), layers=data["mean"].shape[1], heads_per_layer=data["mean"].shape[2],
            diagnostic_query_tokens=data["sample"]["prediction_tokens"],
            analytic_initial_coefficient=1.0 / math.factorial(order),
            pooled_mean=mean, pooled_std=pooled_std, token_min=minima.min(), token_max=maxima.max(),
            layer_head_mean_p10=np.quantile(means, 0.1), layer_head_mean_median=np.median(means),
            layer_head_mean_p90=np.quantile(means, 0.9),
            layer_head_mean_min=means.min(), layer_head_mean_max=means.max(),
            mean_within_layer_head_token_std=stds.mean(),
            negative_mean_layer_head_count=int(np.sum(means < 0)),
            any_negative_token_layer_head_count=int(np.sum(minima < 0)),
            all_negative_token_layer_head_count=int(np.sum(maxima < 0)),
            taylor_relative_pooled_mean=mean * math.factorial(order),
        ))
    return rows


def plot(root: Path, out: Path, selection: dict) -> None:
    """Generate coefficient figures and final summaries after caller verifies pins."""
    out.mkdir(parents=True, exist_ok=True)
    datasets = [_load(root, entry) for entry in selection["runs"]]
    assert len({data["degree"] for data in datasets}) == len(datasets), "Output names require distinct degrees"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.unicode_minus": True, "savefig.transparent": False}):
        for data in datasets:
            _histories(data, out)
            _final_heatmap(data, out)
    summary = [row for data in datasets for row in _summary_rows(data)]
    with (out / "coefficient_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({key: format(value, ".12g") if isinstance(value, (float, np.floating)) else value
                             for key, value in row.items()})
