"""Measure one layer's learned polynomials from its operand capture, then plot them.

The measurement reads the capture run's numerical archive on its home cluster and writes the
plotted polynomials, free-coefficient band and score histograms as compact JSON under that
run. The plot reads only committed JSON and selection files; it needs no GPU or archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(min(4, max(1, int(os.environ.get(_name, "4")))))

import numpy as np
import yaml

from scaffolds.records import capture_artifacts, load_run, update_record


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
OUTPUT = "learned_polynomials.json"
ROLE = "learned_polynomial_operands"
FITS = ("static", "free")
RANGE_QUANTILES = [.05, .95]
BAND_QUANTILES = [.1, .9]
BAND_POINTS = 201
BINS = 60
COLORS = {"static": "#2563a6", "free": "#d55e00", "taylor": "#38434f"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded(values) -> list[float]:
    return [float(f"{value:.6g}") for value in np.ravel(values)]


def measure(run: Path) -> None:
    run = run.resolve()
    record = load_run(run)
    recipe = yaml.safe_load((run / "config.yaml").read_text())["diagnostic"]
    if record["status"] != "complete" or [fit["role"] for fit in recipe["fits"]] != list(FITS):
        raise ValueError("Requires a completed capture of the static and free fits")
    archive = [item for item in record["artifacts"] if Path(item["path"]).name == "operands.npz"]
    if len(archive) != 1 or sha256(run / "operands.npz") != archive[0]["sha256"]:
        raise ValueError("The operand archive differs from its record")
    parent = ROOT / "experiments/transformer_pp/runs" / recipe["fits"][0]["parent_run_id"] / "config.yaml"
    layers = yaml.safe_load(parent.read_text())["model"]["num_layers"]
    data = np.load(run / "operands.npz")
    positions, heads, width = data["static_q"].shape
    degree = data["static_coefficients"].shape[-1] - 1
    rows, cols = np.tril_indices(positions)
    orders = np.arange(degree + 1)
    taylor = np.asarray([1 / math.factorial(order) for order in orders])
    measured = []
    for head in range(heads):
        # Scores as the attention core forms them, over every causal query-key pair.
        scores = {fit: ((data[f"{fit}_q"][:, head] @ data[f"{fit}_k"][:, head].T)
                        / np.float32(math.sqrt(width)))[rows, cols].astype(np.float64) for fit in FITS}
        pooled = np.concatenate([scores[fit] for fit in FITS])
        power, moments = np.ones_like(pooled), []
        for order in range(2 * degree + 1):
            moments.append(power.mean())
            power *= pooled
        gram = np.asarray(moments)[np.add.outer(orders, orders)]
        static = data["static_coefficients"][:, head].astype(np.float64)
        if np.any(static != static[0]):
            raise ValueError("Static coefficients differ across query positions")
        coefficients = {"static": static[0], "free": data["free_coefficients"][:, head].astype(np.float64),
                        "taylor": taylor}
        # Each polynomial divided by its RMS over the pooled scores: sqrt(c^T E[z^(r+s)] c).
        norms = {name: np.sqrt(np.einsum("...r,rs,...s->...", c, gram, c)) for name, c in coefficients.items()}
        if not all(np.isfinite(norm).all() and np.all(norm > 1e-12) for norm in norms.values()):
            raise FloatingPointError(f"Head {head}: a polynomial has no finite positive RMS")
        free = coefficients["free"] / norms["free"][:, None]
        low, high = np.quantile(pooled, RANGE_QUANTILES)
        grid = np.linspace(low, high, BAND_POINTS)
        curves = free @ (grid[:, None] ** orders).T
        lower, upper = np.quantile(curves, BAND_QUANTILES, axis=0)
        if not np.allclose(curves.mean(axis=0), np.polynomial.polynomial.polyval(grid, free.mean(axis=0))):
            raise ArithmeticError("The free mean is not the polynomial of the mean coefficients")
        edges = np.linspace(low, high, BINS + 1)
        measured.append({
            "head": head, "score_range": [float(low), float(high)],
            "static": (coefficients["static"] / norms["static"]).tolist(),
            "taylor": (taylor / norms["taylor"]).tolist(), "free_mean": free.mean(axis=0).tolist(),
            "free_band": [rounded(lower), rounded(upper)],
            "score_counts": {fit: np.histogram(scores[fit], bins=edges)[0].tolist() for fit in FITS}})
        print(f"Measured head {head + 1}/{heads}", flush=True)
    result = {
        "schema_version": 1, "run_id": run.name, "config_sha256": record["config"]["sha256"],
        "operands_sha256": archive[0]["sha256"], "analysis_script_sha256": sha256(Path(__file__)),
        "layer": recipe["layer"], "layers": layers, "indexing": "zero_based", "positions": positions,
        "causal_pairs": len(rows), "degree": degree,
        "settings": {
            "coefficients": "effective monomial coefficients c_r, p(z) = sum_r c_r z^r",
            "scores": "z = q.k / sqrt(d_K) for every causal pair (key at or before the query) of the first sequence",
            "normalization": "each polynomial divided by its RMS over both fits' pooled scores",
            "range_quantiles": RANGE_QUANTILES, "band_quantiles_over_query_positions": BAND_QUANTILES,
            "band_points": BAND_POINTS, "histogram_bins": BINS},
        "heads": measured}
    (run / OUTPUT).write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    update_record(run, capture_artifacts(run, include_log=True))
    print(run / OUTPUT)


def plot(selection_path: Path, output: Path) -> None:
    selection = yaml.safe_load(selection_path.read_text())
    entries = [entry for entry in selection["runs"] if entry.get("role") == ROLE]
    if len(entries) != 1:
        raise ValueError(f"Selection must name exactly one {ROLE} run")
    entry = entries[0]
    run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
    descriptor = entry.get("diagnostics", {}).get("learned_polynomials")
    if not descriptor or descriptor["path"] != str((run / OUTPUT).relative_to(ROOT)):
        raise ValueError("Pin learned_polynomials.json in this run's selection diagnostics before plotting")
    if sha256(run / OUTPUT) != descriptor["sha256"]:
        raise ValueError("The selected learned-polynomial measurement checksum changed")
    data = json.loads((run / OUTPUT).read_text())
    if data["config_sha256"] != entry["sources"]["config.yaml"]["sha256"]:
        raise ValueError("The selected recipe differs from the measurement")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(16, 15))
    grid = fig.add_gridspec(4, 4, left=.06, right=.985, bottom=.045, top=.925, hspace=.38, wspace=.28)
    for measured in data["heads"]:
        head = measured["head"]
        low, high = measured["score_range"]
        x = np.linspace(low, high, 501)
        band = np.linspace(low, high, len(measured["free_band"][0]))
        lower, upper = np.asarray(measured["free_band"])
        curves = {name: np.polynomial.polynomial.polyval(x, measured[name]) for name in ("static", "taylor", "free_mean")}
        sub = grid[head // 4, head % 4].subgridspec(2, 1, height_ratios=[3.5, 1], hspace=.06)
        ax = fig.add_subplot(sub[0])
        hist = fig.add_subplot(sub[1], sharex=ax)
        ax.fill_between(band, lower, upper, color=COLORS["free"], alpha=.20, linewidth=0)
        ax.plot(x, curves["free_mean"], color=COLORS["free"], lw=2.2)
        ax.plot(x, curves["static"], color=COLORS["static"], lw=2.2)
        ax.plot(x, curves["taylor"], color=COLORS["taylor"], lw=1.5, ls="--")
        ax.axhline(0, color="#9ca3af", lw=.6)
        ax.set_title(f"Head {head + 1}", loc="left", fontweight="bold", fontsize=12)
        ax.grid(alpha=.15)
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
        bounds = np.concatenate([lower, upper, *curves.values()])
        pad = max((bounds.max() - bounds.min()) * .08, .01)
        ax.set_ylim(bounds.min() - pad, bounds.max() + pad)
        edges = np.linspace(low, high, len(measured["score_counts"]["static"]) + 1)
        for fit in FITS:
            density = np.asarray(measured["score_counts"][fit]) / (data["causal_pairs"] * np.diff(edges))
            hist.stairs(density, edges, color=COLORS[fit], lw=1.1, fill=True, alpha=.23)
        hist.set_yticks([])
        hist.set_xlim(low, high)
        hist.tick_params(labelsize=9)
        hist.set_xlabel("Attention score z", fontsize=9, labelpad=2)
        if head % 4 == 0:
            ax.set_ylabel("p(z) / RMS")
            hist.set_ylabel("Scores", fontsize=8)
    notes = fig.add_subplot(grid[3, 2:])
    notes.axis("off")
    handles = [Line2D([], [], color=COLORS["static"], lw=2.2, label="Static fit"),
               Line2D([], [], color=COLORS["free"], lw=2.2, label="Free mean"),
               Patch(facecolor=COLORS["free"], alpha=.20, label="Free 10–90%"),
               Line2D([], [], color=COLORS["taylor"], lw=1.5, ls="--", label="Fixed Taylor")]
    notes.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, .88), ncol=2,
                 frameon=False, fontsize=12, borderaxespad=0, columnspacing=2)
    notes.text(0, .40, f"One sequence · {data['positions']:,} tokens", fontsize=13, va="top")
    notes.text(0, .22, "Each polynomial: unit RMS over both models’ pooled scores.", fontsize=11.5, va="top")
    fig.suptitle(f"Learned degree-four polynomials · layer {data['layer'] + 1} of {data['layers']}",
                 y=.978, fontsize=23, fontweight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(output.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measurement = sub.add_parser("measure")
    measurement.add_argument("--run", type=Path, required=True)
    rendering = sub.add_parser("plot")
    rendering.add_argument("--selection", type=Path, default=Path(__file__).with_name("selection.yaml"))
    rendering.add_argument("--output", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.run)
    else:
        plot(args.selection, args.output)


if __name__ == "__main__":
    main()
