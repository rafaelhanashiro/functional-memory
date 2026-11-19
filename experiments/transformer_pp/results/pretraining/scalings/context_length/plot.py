"""Figures and summary for the context_length topic from its selected per-position evaluation."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
STYLE = {  # role: (label, color)
    "forget_gate_no_rope_wdecay_m4": ("forget gate, no RoPE + W decay (m = 4)", "#2ca02c"),
    "exp": ("exp", "#1f77b4"),
    "softmax": ("softmax", "#9467bd"),
    "fixed_taylor_m4": ("fixed Taylor (m = 4)", "#ff7f0e"),
    "query_wdecay_m4": ("query + W decay (m = 4)", "#d62728"),
}
BIN = 64


def binned(curve):
    count = curve.size // BIN
    return (np.arange(count) * BIN + BIN / 2 + 1, curve[:count * BIN].reshape(count, BIN).mean(axis=1))


def main():
    selection = yaml.safe_load((HERE / "selection.yaml").read_text())
    run = next(entry for entry in selection["runs"] if entry["role"] == "extrapolation_evaluation")
    rows = [json.loads(line) for line in open(ROOT / "experiments/transformer_pp/runs" / run["run_id"] / "metrics.jsonl")]
    curves = {(row["role"], row["length"]): row for row in rows}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    reference = np.asarray(curves[("exp", "evaluation")]["per_position_nll"])
    for role, (label, color) in STYLE.items():
        long = np.asarray(curves[(role, "evaluation")]["per_position_nll"])
        x, y = binned(long)
        top.plot(x, np.exp(y), color=color, lw=2, label=label)
        if role != "exp":
            x, d = binned(long - reference)
            bottom.plot(x, d, color=color, lw=2, label=f"{label} − exp")
    for axis in (top, bottom):
        axis.axvline(2048, color="k", ls="--", lw=0.8)
        axis.grid(alpha=0.3, which="both")
    bottom.axhline(0, color="k", lw=0.8)
    top.set_yscale("log")
    top.set_ylabel("PPL(t)")
    top.set_title(f"2,048-token checkpoints scored on 4,096-token validation rows ({BIN}-position bins)")
    top.legend(fontsize=8)
    bottom.set_ylabel("NLL difference from exp (nats)")
    bottom.set_xlabel("token position t (dashed: training length)")
    bottom.set_xlim(1, reference.size)
    bottom.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "extrapolation_per_position.png", dpi=150)
    with open(HERE / "extrapolation_summary.csv", "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["role", "source_run_id", "replay_nll", "recorded_validation_nll", "replay_absolute_error",
                         "ppl_validation_2048", "ppl_positions_1_2048", "ppl_positions_2049_4096", "ppl_positions_1_4096"])
        for role in STYLE:
            short, long = curves[(role, "training")], curves[(role, "evaluation")]
            writer.writerow([role, short["source_run_id"], f"{short['nll']:.6f}", f"{short['recorded_validation_nll']:.6f}",
                             f"{short['replay_absolute_error']:.2e}", f"{math.exp(short['nll']):.4f}",
                             f"{math.exp(long['nll_within_training_length']):.4f}",
                             f"{math.exp(long['nll_beyond_training_length']):.4f}", f"{math.exp(long['nll']):.4f}"])


if __name__ == "__main__":
    main()
