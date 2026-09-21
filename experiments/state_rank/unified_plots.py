"""Unified saved-data views averaged over inputs and random seeds.

Head figures preserve positional head indices while averaging scalar metrics;
they do not match learned features across networks. Layer figures additionally
average heads. The established terminal depth comparison retains its original
individual seed lines. No state matrices or spectra are pooled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.state_rank.plot_comparison import _save_figure
from experiments.state_rank.unified_metrics import RANK_METRICS, depth_curves, grouped_curves


PRETRAINED_COLOR = "#1f567d"
RANDOM_COLOR = "#c16828"
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


def _group_limits(groups, width, summary):
    limits = {}
    for metric, _ in RANK_METRICS:
        maximum = max(float(group[metric][summary].max()) for group in groups.values())
        if metric == "threshold_rank":
            maximum = max(maximum, width)
        limits[metric] = (0.0, max(1.0, maximum) * 1.06)
    return limits


def _token_axis(axis, analysis, metric, label, limits):
    if metric == "threshold_rank":
        axis.plot(analysis.prefixes, np.minimum(analysis.prefixes, analysis.width),
                  color="#777777", linestyle="--", linewidth=.9, label="_rank_ceiling")
    axis.set(xlabel="Prefix length (including BOS)", ylabel=_metric_label(metric, label),
             xlim=(analysis.prefixes[0], analysis.prefixes[-1]), ylim=limits[metric])
    _format_axis(axis)


def plot_analysis(analysis, output):
    """Save paired pretrained/random head and layer figures for one architecture.

    Random head curves average input means at the same head index across seeds;
    positional grouping does not assert a correspondence of learned features.
    """
    plt = _pyplot()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    heads = grouped_curves(analysis, view="heads")
    layers = grouped_curves(analysis, view="layers")
    head_limits = _group_limits(heads, analysis.width, "max")
    layer_limits = _group_limits(layers, analysis.width, "mean")
    seeds = sum(condition.weights == "random" for condition in analysis.conditions)
    head_kind = "KV head index" if "llama" in f"{analysis.key} {analysis.label}".lower() else "Recurrent head index"
    colors = plt.get_cmap("tab10").colors
    paths = []
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(4, 2, figsize=(12.4, 12.2), sharex=True, sharey="row", constrained_layout=True)
        for row, (metric, label) in enumerate(RANK_METRICS):
            for column, group in enumerate(("pretrained", "random")):
                axis, values = axes[row, column], heads[group][metric]
                for head_axis, head in enumerate(analysis.head_indices):
                    color = colors[head_axis % len(colors)]
                    axis.fill_between(analysis.prefixes, values["min"][:, head_axis],
                                      values["max"][:, head_axis], color=color, alpha=.12, linewidth=0)
                    axis.plot(analysis.prefixes, values["mean"][:, head_axis], color=color,
                              linewidth=1.35, label=f"{head_kind} {head}")
                _token_axis(axis, analysis, metric, label, head_limits)
        axes[0, 0].set_title(f"Pretrained: mean across {analysis.input_count} inputs\nBand: input min–max")
        axes[0, 1].set_title(f"Random: mean across {seeds} seeds and {analysis.input_count} inputs\nBand: min–max of seed input means")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=min(4, len(handles)), frameon=False)
        figure.suptitle(
            f"{analysis.label} | head-index curves | layer index {analysis.layer} | width {analysis.width}\n"
            "Random curves average positional head indices across seeds, not matched learned features\n"
            "Token positions remain separate; dashed threshold line is the rank ceiling", fontsize=12,
        )
        paths.extend(_save(figure, output, f"{analysis.key}/heads"))
        plt.close(figure)

        figure, axes = plt.subplots(4, 2, figsize=(12.4, 12.2), sharex=True, sharey="row", constrained_layout=True)
        normalization = Normalize(vmin=analysis.layers[0], vmax=max(analysis.layers[-1], analysis.layers[0] + 1))
        color_map = plt.get_cmap("viridis")
        for row, (metric, label) in enumerate(RANK_METRICS):
            for column, group in enumerate(("pretrained", "random")):
                axis, values = axes[row, column], layers[group][metric]["mean"]
                for layer_axis, layer in enumerate(analysis.layers):
                    axis.plot(analysis.prefixes, values[layer_axis], color=color_map(normalization(layer)),
                              linewidth=1.25, label=f"Layer {layer}")
                _token_axis(axis, analysis, metric, label, layer_limits)
        axes[0, 0].set_title(f"Pretrained: mean across {analysis.input_count} inputs and {analysis.head_count} heads")
        axes[0, 1].set_title(f"Random: additionally averaged across {seeds} seeds")
        color_bar = figure.colorbar(ScalarMappable(norm=normalization, cmap=color_map), ax=list(axes.flat),
                                   label="Layer index (zero based)", shrink=.88, pad=.025)
        color_bar.locator = MaxNLocator(integer=True, nbins=8)
        color_bar.update_ticks()
        figure.suptitle(
            f"{analysis.label} | all {len(analysis.layers)} measured layers | width {analysis.width}\n"
            "Each curve is one layer; token positions remain separate\n"
            "Dashed threshold line is the rank ceiling", fontsize=12,
        )
        paths.extend(_save(figure, output, f"{analysis.key}/layers"))
        plt.close(figure)
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
    """Save the established absolute-count terminal depth comparison unchanged."""
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

    return paths
