"""Render pinned m=4 B=256 functional probes using only compact repository data.

Default: require all three completed query/free/static probes. --free-only uses
only the explicitly selected free run. Neither mode reads cluster checkpoints,
datasets or environments. PCA latent coordinates remain oracle-derived and are
not refitted; a preserved fitting loss would establish compression, not causal
predictability or held-out language-model generalization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scaffolds.records.selections import verify_pinned_runs


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
TOPIC = Path(__file__).resolve().parent
ROLES = {
    "free": "free_coefficients_full_batch_m4_head_scale",
    "static": "static_coefficients_full_batch_m4_head_scale",
    "query": "query_generator_full_batch_m4_head_scale",
}
PARAMETERIZATIONS = {
    "free": "free_per_example_query_coefficients", "static": "static_coefficients",
    "query": "query_coefficient_generator",
}
VARIANTS = [
    ("fitted_raw", "Raw fitted coefficients"),
    ("pca_rank_5", "PCA 5: full-rank control"),
    ("unit_direction_gauge_control", "All layers: unit radius"),
    *[(f"pca_rank_{rank}", "PCA 0: mean direction" if rank == 0 else
       f"PCA {rank}") for rank in range(5)],
    ("first_layer_token_mean_direction", "Layer 1: token mean direction\n(original radii)"),
    ("first_layer_token_mean_fixed_radius", "Layer 1: token-only vector\n(unit radius)"),
]
SUBSETS = ("all_fitting_sequences", "basis_fit_sequences", "other_fitting_sequences")
BLUE, ORANGE, GREEN = "#2779b5", "#d77b21", "#228f70"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_probe(selection, kind, root=ROOT):
    entries = [entry for entry in selection["runs"] if entry.get("role") == ROLES[kind]]
    if len(entries) != 1:
        raise ValueError(f"Select exactly one {ROLES[kind]} run")
    entry = entries[0]
    verify_pinned_runs(root, [entry])
    relative = Path("experiments") / entry["study"] / "runs" / entry["run_id"]
    run = root / relative
    expected_path = (relative / "batch_functional_probe.json").as_posix()
    pin = entry.get("diagnostics", {}).get("batch_functional_probe")
    if not pin or pin.get("path") != expected_path:
        raise ValueError(f"Pin {expected_path} as diagnostics.batch_functional_probe before plotting")
    if sha256(root / expected_path) != pin["sha256"]:
        raise ValueError(f"Diagnostic checksum differs from the selection: {expected_path}")
    data = json.loads((root / expected_path).read_text())
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    oracle, model = recipe["oracle"], recipe["model"]
    completion = json.loads((run / "completion.json").read_text())
    record = json.loads((run / "record.json").read_text())
    expected = {
        "schema_version": 1, "run_id": entry["run_id"],
        "parent_run_id": entry["parent_run_id"], "parameterization": PARAMETERIZATIONS[kind],
        "config_sha256": entry["sources"]["config.yaml"]["sha256"],
        "completion_sha256": entry["sources"]["completion.json"]["sha256"],
        "selected_checkpoint_step": completion["best_step"],
        "checkpoint_sha256": record["selected_checkpoint"]["sha256"],
        "parent_checkpoint_sha256": oracle["parent_checkpoint_sha256"],
        "batch_sha256": oracle["batch_sha256"], "eval_manifest_sha256": oracle["eval_manifest_sha256"],
        "sequence_count": 256, "prediction_positions": recipe["data"]["sequence_length"] - 1,
        "frozen_parent_parameters_unchanged": True,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"{entry['run_id']}: probe identity mismatch for {key}")
    if (record["status"] != "complete" or completion["status"] != "complete"
            or oracle["parent_run_id"] != entry["parent_run_id"]
            or model["attention_degree"] != 4 or not model["qk_norm_head_scale"]):
        raise ValueError("Expected completed degree-four canonical-QK coefficient fits")
    if kind == "free":
        required = {key for key, _ in VARIANTS} | {"original_parent"}
        if set(data["measurements"]) != required:
            raise ValueError("The free probe must include every raw, radius, PCA and token-lookup control")
    elif set(data["measurements"]) != {"original_parent", "fitted_raw"}:
        raise ValueError("Unexpected query/static probe measurements")
    for variant, measurements in data["measurements"].items():
        values = np.asarray(measurements["per_sequence_nll"], np.float64)
        if values.shape != (256,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid per-sequence NLL: {variant}")
        split = measurements["basis_fit_sequences"]["sequence_count"]
        for subset, chunk in zip(SUBSETS, (values, values[:split], values[split:]), strict=True):
            result = measurements[subset]
            if (len(chunk) != result["sequence_count"] or
                    not np.isclose(chunk.mean(), result["nll"], atol=1e-12, rtol=1e-12) or
                    not np.isclose(math.exp(min(result["nll"], 80.)), result["perplexity"], rtol=1e-12)):
                raise ValueError(f"Invalid aggregate NLL/PPL: {variant}/{subset}")
    replay = data["measurements"]["fitted_raw"]["all_fitting_sequences"]["nll"]
    if abs(replay - completion["best_fit_nll"]) > data["nll_replay_atol"]:
        raise ValueError("The fitted checkpoint does not reproduce the recorded fitting loss")
    return entry, data


def save_figure(fig, prefix):
    fig.savefig(prefix.with_suffix(".png"), dpi=180, metadata={"Software": "Matplotlib"})
    fig.savefig(prefix.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def write_summary(probes, output):
    fields = ["run_id", "role", "parent_run_id", "checkpoint_step", "variant", "subset", "evaluation_forward",
              "sequence_count", "nll", "perplexity", "delta_nll_from_fitted_raw", "fraction_fitted_gain_erased"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for kind in ("free", "static", "query"):
            if kind not in probes:
                continue
            entry, data = probes[kind]
            variants = ["original_parent"] + ([key for key, _ in VARIANTS] if kind == "free" else ["fitted_raw"])
            for variant in variants:
                for subset in SUBSETS:
                    row = data["measurements"][variant][subset]
                    reference = data["measurements"]["fitted_raw"][subset]["nll"]
                    gain = data["measurements"]["original_parent"][subset]["nll"] - reference
                    writer.writerow({"run_id": entry["run_id"], "role": entry["role"],
                        "parent_run_id": entry["parent_run_id"], "checkpoint_step": data["selected_checkpoint_step"],
                        "variant": variant, "subset": subset, "sequence_count": row["sequence_count"],
                        "evaluation_forward": ("ordinary_native_replay" if kind == "query" or variant == "original_parent"
                                               else "coefficient_override_replay"),
                        "nll": format(row["nll"], ".12g"), "perplexity": format(row["perplexity"], ".12g"),
                        "delta_nll_from_fitted_raw": format(row["nll"] - reference, ".12g"),
                        "fraction_fitted_gain_erased": format((row["nll"] - reference) / gain, ".12g") if gain > 0 else ""})


def short_number(value):
    return f"{value:.3f}" if value < 100 else f"{value:.3g}"


def query_parity_label(label, snapshot, default_tolerance):
    """Keep the original per-sequence bound visible, including legacy reports."""
    parity = snapshot.get("external_native_nll_comparison", {})
    tolerance = parity.get("nll_replay_atol", default_tolerance)
    maximum = parity.get("max_absolute_nll_delta", snapshot.get("external_native_max_nll_error"))
    mean = parity.get("mean_nll_delta")
    if maximum is None:
        return f"{label}: instrumented/native NLL parity not recorded"
    passed = parity.get("original_per_sequence_bound_passed", maximum <= tolerance)
    state = "PASSED" if passed else "FAILED"
    mean_text = "mean ΔNLL not recorded" if mean is None else f"mean ΔNLL {mean:+.3g}"
    aggregate = "" if mean is None else (
        "; aggregate PASSED" if parity.get("aggregate_nll_passed", abs(mean) <= tolerance) else "; aggregate FAILED")
    return (f"{label}: per-sequence parity {state} (bound {tolerance:g}); "
            f"{mean_text}, max |ΔNLL| {maximum:.3g}{aggregate}")


def plot_free(data, prefix):
    measurements = data["measurements"]
    fig, (all_ax, pca_ax) = plt.subplots(1, 2, figsize=(15, 8.8),
                                       gridspec_kw={"width_ratios": [1.3, 1.]}, layout="constrained")
    fig.suptitle("Free polynomial coefficients: loss after constrained reconstruction", fontsize=18, weight="bold")
    labels = [label for _, label in VARIANTS]
    nll = [measurements[key]["all_fitting_sequences"]["nll"] for key, _ in VARIANTS]
    ppl = [measurements[key]["all_fitting_sequences"]["perplexity"] for key, _ in VARIANTS]
    colors = ["#596777", "#596777", "#9a6ab4"] + [BLUE] * 5 + [GREEN, ORANGE]
    y = np.arange(len(labels))
    all_ax.barh(y, nll, color=colors, alpha=.9)
    parent_nll = measurements["original_parent"]["all_fitting_sequences"]["nll"]
    all_ax.axvline(parent_nll, color="#676767", linestyle="--", linewidth=1.3, label="Original Taylor backbone")
    width = max(max(nll), parent_nll, .01)
    for index, (loss, perplexity) in enumerate(zip(nll, ppl, strict=True)):
        all_ax.text(loss + .02 * width, index,
                    f"{short_number(loss)} / {short_number(perplexity)}", va="center", fontsize=9)
    all_ax.set(yticks=y, yticklabels=labels, xlim=(0, width * 1.55),
               xlabel="Fitting NLL (nats/token)", title="All 256 fitting sequences\nLabels: NLL / perplexity")
    all_ax.invert_yaxis(); all_ax.legend(frameon=False, fontsize=9, loc="lower right")
    all_ax.grid(axis="x", alpha=.18)
    rank = np.arange(6)
    for subset, color, label in (("basis_fit_sequences", BLUE, "192 sequences used to fit PCA"),
                                  ("other_fitting_sequences", ORANGE, "Other 64 fitting sequences")):
        loss = [measurements[f"pca_rank_{r}"][subset]["nll"] for r in rank]
        pca_ax.plot(rank, loss, marker="o", color=color, label=label)
        pca_ax.axhline(measurements["fitted_raw"][subset]["nll"], color=color, linestyle=":", linewidth=1.1)
    pca_ax.set(xlabel="Retained centered PCA components", ylabel="Fitting NLL (log scale)",
               xticks=rank, yscale="log", title="Basis transfer to other fitted sequences\nDotted lines: unmodified free coefficients")
    pca_ax.legend(frameon=False, fontsize=9); pca_ax.grid(alpha=.18, which="both")
    for ax in (all_ax, pca_ax):
        ax.set_axisbelow(True)
    raw = measurements["fitted_raw"]["all_fitting_sequences"]["nll"]
    roundtrip = measurements["pca_rank_5"]["all_fitting_sequences"]["nll"] - raw
    unit_change = measurements["unit_direction_gauge_control"]["all_fitting_sequences"]["nll"] - raw
    fig.supxlabel(
        f"m = 4 · B = 256 · frozen Taylor backbone · full-rank ΔNLL = {roundtrip:.3g}; unit-radius ΔNLL = {unit_change:.3g}\n"
        "PCA uses factorial-coordinate directions, fitted on 192 sequences × 64 uniform positions; each original positive radius is restored.\n"
        "PCA coordinates still come from optimized oracle tables (rank 3 = 630 values per prediction); this tests compression, not a learned predictor.\n"
        "Token means use all fitting positions. Original-radius controls retain query/sequence dependence; unit-radius token vectors do not.\n"
        "Later layers retain free tables in the token interventions; harm does not prove that every token-only generator must fail.\n"
        "All measurements use fitting inputs, not held-out LM data. These samples differ from the separate CPU coefficient-shape analysis.", fontsize=9)
    save_figure(fig, prefix)


def plot_query(data, prefix):
    diagnostic = data["diagnostics"]
    before, after = diagnostic["before"], diagnostic["after"]
    layer_count = len(before["statistics"]["tanh_derivative_mean"])
    layers = np.arange(1, layer_count + 1)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.5), layout="constrained")
    fig.suptitle("Query coefficient generator: instrumented before/after statistics", fontsize=17, weight="bold")
    statistic_panels = (
        (axes[0, 0], "saturated_099_fraction", 100., "Saturated tanh units", "|tanh(z)| ≥ 0.99 (%)"),
        (axes[0, 1], "tanh_derivative_mean", 1., "Tanh derivative", "Mean 1 − tanh²(z)"),
        (axes[0, 2], "cancellation_ratio_lt001_fraction", 100., "Bias/modulation cancellation",
         "|u| < 0.01 (|b| + |a tanh(z)|) (%)"),
    )
    for ax, key, multiplier, title, ylabel in statistic_panels:
        for item, color, label in ((before, BLUE, "Pretrained query parent"), (after, ORANGE, "Fitted query generator")):
            values = np.asarray(item["statistics"][key])
            ax.plot(layers, multiplier * values.mean(axis=(1, 2)), color=color, marker="o", markersize=3, label=label)
        ax.set(title=title, ylabel=ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for item, color, label in ((before, BLUE, "Before"), (after, ORANGE, "After")):
        telemetry = item["pre_output_rms_telemetry"]
        for multiple, style in ((1, "-"), (100, "--")):
            values = [100 * telemetry[f"layer_{layer:02d}/head_mean_square_lt_{multiple}eps_fraction_mean"]
                      for layer in range(layer_count)]
            axes[1, 0].plot(layers, values, style, color=color, label=f"{label}: <{multiple}ε")
        norm = np.asarray(item["per_layer_head_factorial_coordinate_norm_quantiles"])
        quantiles = item["quantile_probabilities"]
        median = norm[:, :, quantiles.index(.5)]
        axes[1, 1].plot(layers, np.median(median, axis=1), color=color, marker="o", markersize=3, label=label)
        axes[1, 1].fill_between(layers, *np.quantile(median, [.1, .9], axis=1), color=color, alpha=.14)
        projection = np.asarray(item["projection_rms_per_layer_head_order"])
        axes[1, 2].plot(layers, np.sqrt(np.mean(projection ** 2, axis=(1, 2))), color=color,
                       marker="o", markersize=3, label=label)
    axes[1, 0].set(title="Outputs near the RMSNorm epsilon", ylabel="Head-position mean-square below threshold (%)")
    axes[1, 0].legend(frameon=False, fontsize=8, ncols=2)
    axes[1, 1].set(title="Coefficient radius in factorial coordinates", ylabel="Median over head-wise median radii")
    axes[1, 2].set(title="Stored query projection parameters", ylabel="W parameter RMS")
    for ax in axes.flat:
        ax.set_xlabel("Layer"); ax.set_xticks([1, 5, 10, 15]); ax.grid(alpha=.18); ax.set_axisbelow(True)
    initial = data["measurements"]["original_parent"]["all_fitting_sequences"]["perplexity"]
    fitted = data["measurements"]["fitted_raw"]["all_fitting_sequences"]["perplexity"]
    before_parity = query_parity_label("Before", before, data.get("nll_replay_atol", .002))
    after_parity = query_parity_label("After", after, data.get("nll_replay_atol", .002))
    fig.supxlabel(
        f"m = 4 · B = 256 · ordinary native replay fitting PPL {initial:.3f} → {fitted:.3f} · backbone and QK scales frozen\n"
        f"{before_parity}\n{after_parity}\n"
        "Activation statistics use the instrumented compiled forward; the fitting PPL and CSV losses use ordinary native replay.\n"
        "Each generator uses its own causal inputs; later-layer activations can change. Top panels average all tokens, heads and orders.\n"
        "Radius uses 64 sampled positions; shading spans 10–90% of head-wise medians. RMS epsilon frequencies use every head and position.\n"
        "These are activation and accumulated parameter measurements, not effective Adam learning rates or a causal diagnosis of loss.\n"
        "Free, static and query fits start from different pretrained backbones; their fitting losses are not a matched-backbone generalization comparison.", fontsize=9)
    save_figure(fig, prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=TOPIC / "selection.yaml")
    parser.add_argument("--output-dir", type=Path, default=TOPIC)
    parser.add_argument("--free-only", action="store_true")
    args = parser.parse_args()
    selection = yaml.safe_load(args.selection.read_text())
    kinds = ("free",) if args.free_only else ("free", "static", "query")
    probes = {kind: load_probe(selection, kind) for kind in kinds}
    if len({data["batch_sha256"] for _, data in probes.values()}) != 1:
        raise ValueError("The selected probes do not share the exact fitting inputs")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "m4-batch-functional-probe"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = "batch_functional_free_summary.csv" if args.free_only else "batch_functional_summary.csv"
    write_summary(probes, args.output_dir / summary)
    plot_free(probes["free"][1], args.output_dir / "batch_functional_free")
    if not args.free_only:
        plot_query(probes["query"][1], args.output_dir / "batch_functional_query")
    print(args.output_dir / summary)


if __name__ == "__main__":
    main()
