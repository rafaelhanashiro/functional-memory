"""Unified saved-data views with separate heads and separate random networks.

Head figures reduce only the input axis. Depth figures average each network's
terminal metrics over inputs and heads before summarizing random seeds. No state
matrices or spectra are pooled, and head numbers are not matched across networks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.state_rank.plot_comparison import _save_figure
from experiments.state_rank.unified_metrics import RANK_METRICS, depth_curves, head_curves


PRETRAINED_COLOR = "#1f567d"
RANDOM_COLOR = "#c16828"
ARCHITECTURE_COLORS = ("#1f567d", "#c16828", "#4c7d4f", "#8060a0")
STYLE = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
         "legend.fontsize": 9, "pdf.fonttype": 42}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _metric_label(metric, label):
    return "Threshold rank (σ > 10⁻⁴ σ₁)" if metric == "threshold_rank" else label


def _format_axis(axis):
    axis.grid(alpha=.2, linewidth=.6)
    axis.spines[["top", "right"]].set_visible(False)


def _save(figure, output, relative_name):
    output = Path(output)
    (output / relative_name).parent.mkdir(parents=True, exist_ok=True)
    _save_figure(figure, output, relative_name)
    return [f"{relative_name}.{extension}" for extension in ("png", "pdf")]


def _condition_label(condition):
    return "Pretrained" if condition.weights == "pretrained" else f"Random weights, seed {condition.seed}"


def _head_limits(analysis):
    """Use the same scale in every condition and individual-input rendering."""
    maxima = {metric: 0.0 for metric, _ in RANK_METRICS}
    for condition in analysis.conditions:
        curves = head_curves(condition, analysis.layer)
        for metric in maxima:
            maxima[metric] = max(maxima[metric], float(curves[metric]["max"].max()))
    maxima["threshold_rank"] = max(maxima["threshold_rank"], analysis.width)
    return {metric: (0.0, max(1.0, maximum) * 1.06) for metric, maximum in maxima.items()}


def _layer_curves(condition):
    """Keep every token and layer; average only the input and head axes."""
    return {metric: condition.arrays[metric].mean(axis=(1, 3)) for metric, _ in RANK_METRICS}


def _plot_layer_condition(analysis, condition, output, limits):
    plt = _pyplot()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    curves = _layer_curves(condition)
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), sharex=True, constrained_layout=True)
    normalization = Normalize(vmin=analysis.layers[0], vmax=max(analysis.layers[-1], analysis.layers[0] + 1))
    color_map = plt.get_cmap("viridis")
    for axis, (metric, label) in zip(axes.flat, RANK_METRICS, strict=True):
        for layer_axis, layer in enumerate(analysis.layers):
            axis.plot(analysis.prefixes, curves[metric][layer_axis], color=color_map(normalization(layer)),
                      linewidth=1.25, label=f"Layer {layer}")
        if metric == "threshold_rank":
            axis.plot(analysis.prefixes, np.minimum(analysis.prefixes, analysis.width),
                      color="#777777", linestyle="--", linewidth=.9, label="_rank_ceiling")
        axis.set(title=_metric_label(metric, label), xlabel="Prefix length (including BOS)",
                 ylabel="Rank / effective modes", xlim=(analysis.prefixes[0], analysis.prefixes[-1]),
                 ylim=limits[metric])
        _format_axis(axis)
    color_bar = figure.colorbar(ScalarMappable(norm=normalization, cmap=color_map), ax=list(axes.flat),
                               label="Layer index (zero based)", shrink=.88, pad=.025)
    color_bar.locator = MaxNLocator(integer=True, nbins=8)
    color_bar.update_ticks()
    figure.suptitle(
        f"{analysis.label} | {_condition_label(condition)} | all {len(analysis.layers)} measured layers | width {analysis.width}\n"
        f"Each curve is one layer: mean across {analysis.input_count} inputs and {len(analysis.head_indices)} heads\n"
        "Token positions remain separate; dashed threshold line is the rank ceiling", fontsize=12,
    )
    paths = _save(figure, output, f"{analysis.key}/layers/{condition.name}")
    plt.close(figure)
    return paths


def plot_analysis(analysis, output, sequence_index=None):
    """Save separate condition figures; return relative PNG/PDF paths.

    ``sequence_index`` is zero based. Its output namespace and displayed input
    number are one based, matching the historical ``text-1`` convention. Aggregate
    calls also save one token-position figure with separate layer curves per
    condition; those curves average inputs and heads, never token positions.
    """
    if sequence_index is not None and (
        isinstance(sequence_index, bool) or not isinstance(sequence_index, (int, np.integer))
        or not 0 <= sequence_index < analysis.input_count
    ):
        raise ValueError("sequence_index must identify an existing input (zero based)")
    if sequence_index is not None:
        sequence_index = int(sequence_index)
    plt = _pyplot()
    limits = _head_limits(analysis)
    namespace = "aggregate" if sequence_index is None else f"sequence-{sequence_index + 1}"
    head_kind = "KV head" if "llama" in f"{analysis.key} {analysis.label}".lower() else "Recurrent head"
    colors = plt.get_cmap("tab10").colors
    paths = []
    layer_limits = {}
    if sequence_index is None:
        layer_values = [_layer_curves(condition) for condition in analysis.conditions]
        for metric, _ in RANK_METRICS:
            maximum = max(float(values[metric].max()) for values in layer_values)
            if metric == "threshold_rank":
                maximum = max(maximum, analysis.width)
            layer_limits[metric] = (0, max(1.0, maximum) * 1.06)
    with plt.rc_context(STYLE):
        for condition in analysis.conditions:
            curves = head_curves(condition, analysis.layer, input_index=sequence_index)
            figure, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), sharex=True, constrained_layout=True)
            for axis, (metric, label) in zip(axes.flat, RANK_METRICS, strict=True):
                values = curves[metric]
                for head_axis, head in enumerate(analysis.head_indices):
                    color = colors[head_axis % len(colors)]
                    if sequence_index is None:
                        axis.fill_between(analysis.prefixes, values["min"][:, head_axis],
                                          values["max"][:, head_axis], color=color, alpha=.12, linewidth=0)
                    axis.plot(analysis.prefixes, values["mean"][:, head_axis], color=color,
                              linewidth=1.35, label=f"{head_kind} {head}")
                if metric == "threshold_rank":
                    axis.plot(analysis.prefixes, np.minimum(analysis.prefixes, analysis.width),
                              color="#555555", linestyle="--", linewidth=.9, label="_rank_ceiling")
                axis.set(title=_metric_label(metric, label), xlabel="Prefix length (including BOS)",
                         ylabel="Rank / effective modes", xlim=(analysis.prefixes[0], analysis.prefixes[-1]),
                         ylim=limits[metric])
                _format_axis(axis)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="outside lower center", ncol=min(4, len(handles)), frameon=False)
            if sequence_index is None:
                detail = f"Mean across {analysis.input_count} inputs; shading: input min–max"
            else:
                title = analysis.inputs[sequence_index].get("title", "Untitled input")
                detail = f"Input {sequence_index + 1}: {title}"
            figure.suptitle(
                f"{analysis.label} | {_condition_label(condition)} | layer index {analysis.layer} | width {analysis.width}\n"
                f"{detail}\nHead indices are local to this network; dashed threshold line is the rank ceiling",
                fontsize=12,
            )
            relative_name = f"{analysis.key}/heads/{namespace}/{condition.name}"
            paths.extend(_save(figure, output, relative_name))
            plt.close(figure)
            if sequence_index is None:
                paths.extend(_plot_layer_condition(analysis, condition, output, layer_limits))
    return paths


def _depth_data(analysis):
    return {condition.name: depth_curves(condition) for condition in analysis.conditions}


def _family_lines(axis, analysis, values, metric, *, x, divisor=1.0, architecture_color=None):
    """Draw each random seed faintly and its mean/range, retaining one network/unit."""
    random = [values[condition.name][metric] / divisor
              for condition in analysis.conditions if condition.weights == "random"]
    random_color = architecture_color or RANDOM_COLOR
    pretrained_color = architecture_color or PRETRAINED_COLOR
    prefix = f"{analysis.label} " if architecture_color else ""
    if random:
        seeds = np.stack(random)
        for seed_curve in seeds:
            axis.plot(x, seed_curve, color=random_color, alpha=.19, linewidth=.75, label="_random_seed")
        axis.fill_between(x, seeds.min(axis=0), seeds.max(axis=0), color=random_color, alpha=.13, linewidth=0)
        axis.plot(x, seeds.mean(axis=0), color=random_color, linestyle="--", linewidth=2,
                  label=f"{prefix}random mean ({len(random)} seeds)")
    for condition in analysis.conditions:
        if condition.weights == "pretrained":
            axis.plot(x, values[condition.name][metric] / divisor,
                      color=pretrained_color, linewidth=2, label=f"{prefix}pretrained")
    _format_axis(axis)


def plot_depth_comparison(analyses, output):
    """Save absolute-count and width-fraction depth figures from completed families.

    The fractional view uses actual layer index / (model depth - 1). Division
    by state width is descriptive and does not equate the two state definitions.
    """
    analyses = list(analyses)
    if not analyses:
        raise ValueError("At least one model analysis is required")
    plt = _pyplot()
    data = [_depth_data(analysis) for analysis in analyses]
    paths = []
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(len(RANK_METRICS), len(analyses), figsize=(6.2 * len(analyses), 12.2),
                                    squeeze=False, sharey="row", constrained_layout=True)
        for row, (metric, label) in enumerate(RANK_METRICS):
            maximum = max(float(curves[metric].max()) for values in data for curves in values.values())
            if metric == "threshold_rank":
                maximum = max(maximum, max(analysis.width for analysis in analyses))
            for column, (analysis, values) in enumerate(zip(analyses, data, strict=True)):
                axis = axes[row, column]
                _family_lines(axis, analysis, values, metric, x=analysis.layers)
                if metric == "threshold_rank":
                    axis.axhline(analysis.width, color="#777777", linestyle=":", linewidth=.9)
                axis.set(xlabel="Layer index (zero based)", ylabel=_metric_label(metric, label),
                         xlim=(analysis.layers[0], analysis.layers[-1]), ylim=(0, max(1.0, maximum) * 1.06))
                if row == 0:
                    axis.set_title(f"{analysis.label} | state width {analysis.width} | token {analysis.prefixes[-1]}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        figure.suptitle("Terminal state rank in absolute counts\n"
                       "Means over inputs and heads; faint lines: individual random seeds; shading: seed min–max\n"
                       "Threshold panels include each architecture's state-width ceiling", fontsize=12)
        paths.extend(_save(figure, output, "depth_counts"))
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
        for axis, (metric, label) in zip(axes.flat, RANK_METRICS, strict=True):
            maximum = 0.0
            for architecture_index, (analysis, values) in enumerate(zip(analyses, data, strict=True)):
                _family_lines(axis, analysis, values, metric, x=analysis.relative_depth,
                              divisor=analysis.width,
                              architecture_color=ARCHITECTURE_COLORS[architecture_index % len(ARCHITECTURE_COLORS)])
                maximum = max(maximum, max(float(curves[metric].max()) / analysis.width for curves in values.values()))
            if metric == "threshold_rank":
                maximum = max(maximum, 1.0)
                axis.axhline(1, color="#777777", linestyle=":", linewidth=.9)
            axis.set(title=_metric_label(metric, label), xlabel="Relative depth: layer index / (number of layers − 1)",
                     ylabel="Rank / state width", xlim=(0, 1), ylim=(0, max(.01, maximum) * 1.06))
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        figure.suptitle("Terminal state rank as a fraction of state width\n"
                       "Solid: pretrained; dashed: random mean; faint lines: individual seeds; shading: seed min–max\n"
                       "Width normalization does not equate architectures or their state definitions", fontsize=12)
        paths.extend(_save(figure, output, "depth_fractional"))
        plt.close(figure)
    return paths
