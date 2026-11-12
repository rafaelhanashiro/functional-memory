"""Plot the selected last-query polynomial kernels, with no checkpoint dependency.

The compact, checksum-pinned measurements remain under their scientific run.
All panels use the effective monomial coefficients; factorial scaling is not
applied here. PDF pages cover every layer, while the PNG previews show the last.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
COLLECTION = Path(__file__).resolve().parent
COLORS = {"before": "#D27A32", "after": "#1876A3", "exp": "#646B73"}
PDF_METADATA = {
    "Title": "Degree-eight polynomial attention at the last fitted query",
    "Author": "functional-memory",
    "Creator": "plot_last_query.py",
    "CreationDate": None,
    "ModDate": None,
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_measurement(selection_path: Path) -> tuple[dict, str]:
    selection = yaml.safe_load(selection_path.read_text())
    settings = selection.get("settings", {}).get("last_query_analysis")
    if not isinstance(settings, dict):
        raise ValueError("selection.settings.last_query_analysis must explicitly select the measurement")
    run_id = settings["run_id"]
    matches = [item for item in selection["runs"] if item["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("The last-query run must occur exactly once in selection.runs")
    run = ROOT / "experiments" / matches[0]["study"] / "runs" / run_id
    descriptor = settings["measurement"]
    if isinstance(descriptor, dict):
        measurement = Path(descriptor["path"])
        expected_hash = descriptor["sha256"]
    else:
        measurement = Path(descriptor)
        expected_hash = settings["measurement_sha256"]
    if not measurement.is_absolute():
        measurement = (run if len(measurement.parts) == 1 else ROOT) / measurement
    actual_hash = _digest(measurement)
    if actual_hash != expected_hash:
        raise ValueError(f"Last-query measurement checksum mismatch: {measurement}")
    data = json.loads(measurement.read_text())
    if data["run_id"] != run_id:
        raise ValueError("Measurement and selection run IDs disagree")
    if data["selected_checkpoint_step"] != settings["selected_checkpoint_step"]:
        raise ValueError("Measurement and selection checkpoint steps disagree")
    before = np.asarray(data["coefficients"]["before"], dtype=np.float64)
    after = np.asarray(data["coefficients"]["after"], dtype=np.float64)
    if before.shape != (10, 16, 9) or after.shape != before.shape:
        raise ValueError("This comparison requires ten layers, sixteen heads, and degree eight")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Nonfinite polynomial coefficients cannot be plotted")
    if len(data["heads"]) != 160 or len({(h["layer"], h["head"]) for h in data["heads"]}) != 160:
        raise ValueError("Expected one measurement for each of the 160 layer/head pairs")
    if data["query_index"] != data["input_tokens"] - 1:
        raise ValueError("The selected query is not the last input token")
    return data, actual_hash


def _curves(data: dict, head: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    after_scores = np.asarray(head["score_quantiles_after"], dtype=np.float64)
    lower, upper = after_scores[0], after_scores[-1]
    if not np.isfinite(after_scores).all() or upper <= lower:
        raise ValueError("A finite, nonzero fitted-score range is required")
    # The fitted polynomial was optimized on this range. Earlier layers' changes
    # can move its query far from the pretrained score range; extrapolating to
    # the union can make exp dominate while hiding the learned function.
    x = np.linspace(lower, upper, 1001)
    layer, index = head["layer"] - 1, head["head"] - 1
    curves = {name: np.polynomial.polynomial.polyval(
        x, np.asarray(data["coefficients"][name][layer][index], dtype=np.float64))
        for name in ("before", "after")}
    curves["exp"] = np.exp(x)
    if not all(np.isfinite(curve).all() for curve in curves.values()):
        raise ValueError(f"Nonfinite curve for layer {layer + 1}, head {index + 1}")
    return x, curves


def _raw_scale(ax, curves: dict[str, np.ndarray], head: dict) -> bool:
    """Keep linear axes unless amplitudes would hide whole comparison curves."""
    peaks = np.asarray([np.max(np.abs(y)) for y in curves.values()])
    positive = peaks[peaks > 0]
    rms = np.asarray(list(head["rms_on_fitted_scores"].values()), dtype=np.float64)
    positive_rms = rms[rms > 0]
    if positive.size and positive_rms.size and (
        positive.max() / positive.min() > 1000
        or positive.max() / positive_rms.max() > 1000
    ):
        threshold = float(np.median(positive_rms))
        ax.set_yscale("symlog", linthresh=threshold, linscale=0.7)
        ax.text(0.02, 0.94, "signed symlog", ha="left", va="top",
                fontsize=8, color="#616873", transform=ax.transAxes)
        return True
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    return False


def figure_for_layer(data: dict, layer: int, *, shape: bool):
    figure, axes = plt.subplots(4, 4, figsize=(14.6, 12.1))
    figure.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.115,
                           wspace=0.30, hspace=0.42)
    heads = {head["head"]: head for head in data["heads"] if head["layer"] == layer}
    any_symlog = False
    for index, ax in enumerate(axes.flat, 1):
        head = heads[index]
        x, curves = _curves(data, head)
        if shape:
            for name in curves:
                scale = float(head["rms_on_fitted_scores"][name])
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"A positive RMS is required for {layer=}, {index=}, {name=}")
                curves[name] = curves[name] / scale
        ax.plot(x, curves["exp"], color=COLORS["exp"], linestyle=(0, (4, 3)),
                linewidth=1.6, zorder=2)
        ax.plot(x, curves["before"], color=COLORS["before"], linewidth=1.8, zorder=3)
        ax.plot(x, curves["after"], color=COLORS["after"], linewidth=2.1, zorder=4)
        ax.axhline(0, color="#4B5563", linewidth=0.8, alpha=0.75, zorder=1)
        ax.axvline(0, color="#A6ABB2", linewidth=0.7, alpha=0.55, zorder=0)
        values = np.asarray(head["score_quantiles_after"], dtype=np.float64)
        ax.plot([values[0], values[-1]], [0.025, 0.025], color=COLORS["after"],
                alpha=0.35, linewidth=1.0, transform=ax.get_xaxis_transform(), zorder=5)
        ax.plot(values, np.full(values.shape, 0.025), linestyle="none", marker="|",
                markersize=5, markeredgewidth=0.9, color=COLORS["after"], alpha=0.85,
                transform=ax.get_xaxis_transform(), zorder=6)
        ax.set_title(f"Head {index}", loc="left", fontsize=11, fontweight="bold", pad=7)
        fraction = float(head["negative_fraction_after_on_fitted_scores"])
        ax.text(0.98, 0.96, f"negative: {fraction:.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color="#46515D",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.3})
        ax.set_xlim(x[0], x[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        if shape:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        else:
            any_symlog |= _raw_scale(ax, curves, head)
        ax.grid(axis="y", color="#DDE1E6", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#B6BDC5")
        ax.tick_params(axis="both", labelsize=9, length=3, color="#9AA2AD")
        ax.yaxis.get_offset_text().set_fontsize(8)

    mode = "Polynomial shapes" if shape else "Polynomial attention kernels"
    figure.suptitle(f"{mode} at the last query · layer {layer} of 10",
                   x=0.075, y=0.97, ha="left", fontsize=20, fontweight="bold", color="#172231")
    figure.text(0.075, 0.933,
                f"Degree 8 · input position {data['query_position']:,} · retained fit at update "
                f"{data['selected_checkpoint_step']:,}", fontsize=12, color="#515C69")
    handles = [Line2D([], [], color=COLORS["before"], linewidth=2, label="Before fitting"),
               Line2D([], [], color=COLORS["after"], linewidth=2.3, label="After coefficient fitting"),
               Line2D([], [], color=COLORS["exp"], linewidth=1.7, linestyle=(0, (4, 3)), label=r"$\exp(s)$ reference")]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.915),
                  ncol=3, frameon=False, fontsize=11, handlelength=3, columnspacing=2.8)
    figure.supxlabel(r"Attention score $s = q^\mathsf{T}k/\sqrt{64}$", x=0.53, y=0.07, fontsize=13)
    figure.supylabel("Kernel / RMS on fitted scores" if shape else "Kernel value",
                    x=0.014, y=0.48, fontsize=13)
    if shape:
        note = "Each curve is divided by its own positive RMS on the same fitted scores; signs are preserved."
    else:
        note = "Effective monomial coefficients are plotted directly, before the head's output RMSNorm."
        if any_symlog:
            note += " Marked panels use signed symlog axes."
    figure.text(0.075, 0.035, note, fontsize=9.2, color="#515C69")
    figure.text(0.075, 0.015,
                "All curves share the fitted query's observed score range. Blue ticks mark fitted score quantiles. "
                "Exp is an unfitted reference.", fontsize=9.2, color="#515C69")
    return figure


def _stationary_points(coefficients: np.ndarray, lower: float, upper: float) -> int:
    derivative = np.polynomial.polynomial.polyder(coefficients)
    if not np.any(derivative):
        return 0
    roots = np.polynomial.polynomial.polyroots(derivative)
    return sum(abs(root.imag) < 1e-7 * max(1, abs(root.real)) and lower < root.real < upper
               for root in roots)


def write_summary(data: dict, measurement_hash: str, path: Path) -> None:
    fields = ("run_id", "checkpoint_step", "query_position", "layer", "head",
              "fitted_score_min", "fitted_score_max", "before_score_min", "before_score_max",
              "negative_fraction_before_on_fitted_scores", "negative_fraction_after_on_fitted_scores",
              "cosine_with_exp_before_on_fitted_scores", "cosine_with_exp_after_on_fitted_scores",
              "stationary_points_before_in_fitted_range", "stationary_points_after_in_fitted_range",
              "measurement_sha256")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for head in sorted(data["heads"], key=lambda item: (item["layer"], item["head"])):
            fitted = head["score_quantiles_after"]
            before = head["score_quantiles_before"]
            row = {"run_id": data["run_id"], "checkpoint_step": data["selected_checkpoint_step"],
                   "query_position": data["query_position"], "layer": head["layer"], "head": head["head"],
                   "fitted_score_min": fitted[0], "fitted_score_max": fitted[-1],
                   "before_score_min": before[0], "before_score_max": before[-1],
                   "measurement_sha256": measurement_hash}
            for name in ("before", "after"):
                for metric in ("negative_fraction", "cosine_with_exp"):
                    key = f"{metric}_{name}_on_fitted_scores"
                    row[key] = head[key]
                coefficients = np.asarray(data["coefficients"][name][head["layer"] - 1][head["head"] - 1])
                row[f"stationary_points_{name}_in_fitted_range"] = _stationary_points(coefficients, fitted[0], fitted[-1])
            writer.writerow(row)


def build(selection_path: Path, output_dir: Path | None = None) -> list[Path]:
    data, measurement_hash = load_measurement(selection_path)
    output_dir = output_dir or selection_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        for shape in (False, True):
            stem = "last_query_polynomial_shapes" if shape else "last_query_polynomials"
            pdf_path = output_dir / f"{stem}.pdf"
            png_path = output_dir / f"{stem}.png"
            with PdfPages(pdf_path, metadata=PDF_METADATA) as pdf:
                for layer in range(1, 11):
                    figure = figure_for_layer(data, layer, shape=shape)
                    pdf.savefig(figure)
                    if layer == 10:
                        figure.savefig(png_path, dpi=180, metadata={"Software": "functional-memory plot_last_query.py"})
                    plt.close(figure)
            files.extend((png_path, pdf_path))
    summary = output_dir / "last_query_polynomial_summary.csv"
    write_summary(data, measurement_hash, summary)
    files.append(summary)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=COLLECTION / "selection.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in build(args.selection.resolve(), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
