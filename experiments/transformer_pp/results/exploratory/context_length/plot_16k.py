"""Plot the selected 2k-trained checkpoints evaluated at 16,384 tokens.

Run from the repository root:
    python -m experiments.transformer_pp.results.exploratory.context_length.plot_16k

Every perplexity is exp(mean NLL), including the 64-position bins and 2k blocks.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
STYLE = {
    "exp": ("exp", "#1f77b4"),
    "query_wdecay_m4": ("query + W decay", "#d62728"),
    "forget_gate_no_rope_wdecay_m4": ("FoX no RoPE + W decay", "#2ca02c"),
}
BIN = 64
BLOCK = 2048
LENGTH = 16384


def binned(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if curve.size % BIN:
        raise ValueError(f"Curve length must be a multiple of {BIN}")
    centers = np.arange(curve.size // BIN) * BIN + (BIN + 1) / 2
    return centers, curve.reshape(-1, BIN).mean(axis=1)


def load_curves() -> dict:
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    selected = [entry for entry in selection["runs"] if entry["role"] == "extrapolation_16k_evaluation"]
    if len(selected) != 1:
        raise ValueError("Select exactly one extrapolation_16k_evaluation run")
    verify_pinned_runs(ROOT, selected)
    entry = selected[0]
    run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
    if json.loads((run / "record.json").read_text())["status"] != "complete":
        raise ValueError("The selected 16k evaluation is not complete")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    curves = {}
    for row in rows:
        key = row["role"], row["length"]
        if key in curves:
            raise ValueError(f"Duplicate evaluation measurement: {key}")
        curves[key] = row
    sequence_counts = set()
    for role in STYLE:
        short, long = curves[(role, "training")], curves[(role, "evaluation")]
        values = np.asarray(long["per_position_nll"])
        if short["sequence_length"] != BLOCK + 1 or long["sequence_length"] != LENGTH + 1:
            raise ValueError(f"Unexpected context lengths for {role}")
        if values.shape != (LENGTH,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid per-position measurements for {role}")
        if not np.isclose(values.mean(), long["nll"], rtol=0, atol=1e-10):
            raise ValueError(f"Aggregate NLL does not match per-position measurements for {role}")
        sequence_counts.add(long["sequences"])
    if len(sequence_counts) != 1:
        raise ValueError("All models must evaluate the same number of sequences")
    return curves


def write_summary(curves: dict) -> None:
    with (HERE / "extrapolation_16k_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "role", "source_run_id", "sequences", "prediction_tokens", "replay_absolute_error",
            "ppl_validation_2048", "nll_positions_1_16384", "ppl_positions_1_16384",
            *[f"ppl_positions_{start + 1}_{start + BLOCK}" for start in range(0, LENGTH, BLOCK)],
        ])
        for role in STYLE:
            short, long = curves[(role, "training")], curves[(role, "evaluation")]
            values = np.asarray(long["per_position_nll"])
            writer.writerow([
                role, long["source_run_id"], long["sequences"], long["prediction_tokens"],
                f"{short['replay_absolute_error']:.2e}", f"{np.exp(short['nll']):.4f}",
                f"{values.mean():.8f}", f"{np.exp(values.mean()):.4f}",
                *[f"{np.exp(block.mean()):.4f}" for block in values.reshape(-1, BLOCK)],
            ])


def main() -> None:
    curves = load_curves()
    reference = np.asarray(curves[("exp", "evaluation")]["per_position_nll"])
    sequences = curves[("exp", "evaluation")]["sequences"]
    positions = np.arange(1, LENGTH + 1)
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(13.5, 10), sharex=True,
            gridspec_kw={"height_ratios": [3, 2]}, layout="constrained",
        )
        for role, (label, color) in STYLE.items():
            values = np.asarray(curves[(role, "evaluation")]["per_position_nll"])
            centers, means = binned(values)
            top.plot(positions, np.exp(values), color=color, lw=0.4, alpha=0.18)
            top.plot(centers, np.exp(means), color=color, lw=2.4,
                     label=f"{label} (mean PPL {np.exp(values.mean()):.3f})")
            if role != "exp":
                difference = values - reference
                centers, means = binned(difference)
                bottom.plot(positions, difference, color=color, lw=0.4, alpha=0.18)
                bottom.plot(centers, means, color=color, lw=2.4, label=f"{label} − exp")
        for axis in (top, bottom):
            axis.axvline(BLOCK, color="#555555", ls="--", lw=1.1)
            axis.grid(which="both", color="#cccccc", alpha=0.55)
            axis.set_axisbelow(True)
        top.annotate("training length", xy=(BLOCK, 1), xycoords=("data", "axes fraction"),
                     xytext=(6, -7), textcoords="offset points", ha="left", va="top",
                     color="#555555", fontsize=10)
        top.set_title(
            f"Per-position validation perplexity ({sequences:,} sequences)\n"
            f"2,048-token checkpoints · bold: {BIN}-position bins · light: every position",
            fontsize=14,
        )
        top.set_yscale("log")
        top.set_ylabel("PPL(t) = exp(mean NLL at position t)")
        top.legend(loc="best", fontsize=11, framealpha=0.95)
        bottom.axhline(0, color="black", lw=0.8)
        bottom.set_ylabel("NLL difference from exp (nats)")
        bottom.set_xlabel("Token position in the 16,384-token window")
        bottom.set_xlim(1, LENGTH)
        bottom.set_xticks(np.arange(BLOCK, LENGTH + 1, BLOCK))
        bottom.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        bottom.legend(loc="best", fontsize=11, framealpha=0.95)
        fig.savefig(HERE / "extrapolation_16k_per_position.png", dpi=160, metadata={"Software": "Matplotlib"})
        fig.savefig(HERE / "extrapolation_16k_per_position.pdf",
                    metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
        plt.close(fig)
    write_summary(curves)


if __name__ == "__main__":
    main()
