"""Measure free-coefficient shape on CPU, then plot its compact run artifact.

The measurement reads the explicitly selected retained coefficient checkpoint.
The plot reads only committed JSON and selection files; it does not need a GPU
or the cluster's checkpoint. Positive common coefficient scale is separated
because head-output RMSNorm makes it approximately redundant (up to epsilon).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = str(min(4, max(1, int(os.environ.get(_name, "4")))))

import numpy as np
import yaml

import analyze_batch_coefficients as geometry
from scaffolds.records import capture_artifacts, load_run, update_record


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = "free_coefficient_shape.json"
ROLE = "free_coefficients_full_batch_m4_head_scale"
QUANTILES = [0., .01, .1, .5, .9, .99, 1.]
HEAD_QUANTILES = [.1, .5, .9]


def quantiles(value, probabilities=QUANTILES, axis=None):
    return np.quantile(value, probabilities, axis=axis).tolist()


def angles(left, right):
    """Angles of directions; retain sign and reject undefined reconstructions."""
    norm = np.linalg.norm(right, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("A coefficient reconstruction has undefined direction")
    cosine = np.sum(left * (right / norm), axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def analyze(sampled, train_count):
    orders = sampled.shape[-1]
    factorial = np.asarray([math.factorial(order) for order in range(orders)])
    factorial_coefficients = sampled * factorial
    norms = np.linalg.norm(factorial_coefficients, axis=-1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("A fitted coefficient vector has no finite direction")
    unit = factorial_coefficients / norms[..., None]
    taylor = np.ones(orders) / np.sqrt(orders)
    taylor_angles = angles(unit, taylor)
    mixed_sign = np.any(sampled < 0, axis=-1) & np.any(sampled > 0, axis=-1)
    per_head = []
    pooled_pca_angles, pooled_axis_angles = [[] for _ in range(orders)], [[] for _ in range(orders)]
    for layer in range(unit.shape[1]):
        for head in range(unit.shape[3]):
            values = unit[:, layer, :, head, :]
            flat = values.reshape(-1, orders)
            fitted = geometry.pca(flat)
            train = geometry.pca(values[:train_count].reshape(-1, orders))
            test = values[train_count:].reshape(-1, orders)
            centered = flat - fitted["mean"]
            covariance = centered.T @ centered / len(flat)
            std = np.sqrt(np.maximum(np.diag(covariance), 0))
            correlation = np.divide(covariance, std[:, None] * std[None, :],
                                    out=np.zeros_like(covariance), where=std[:, None] * std[None, :] > 0)
            sequence_effect = values.mean(axis=1) - fitted["mean"]
            position_effect = values.mean(axis=0) - fitted["mean"]
            interaction = values - fitted["mean"] - sequence_effect[:, None, :] - position_effect[None, :, :]
            energies = np.asarray([np.mean(np.sum(x * x, axis=-1))
                                   for x in (sequence_effect, position_effect, interaction)])
            if not np.isclose(energies.sum(), fitted["total_variance"], rtol=1e-10, atol=1e-12):
                raise ValueError("Two-way coefficient variance decomposition failed")
            fraction = energies / fitted["total_variance"]
            test_centered = test - train["mean"]
            axis_variance = np.var(values[:train_count].reshape(-1, orders), axis=0)
            axes = np.argsort(-axis_variance, kind="stable")
            pca_angles, axis_angles = [], []
            for rank in range(1, orders + 1):
                basis = train["basis"][:, :rank]
                reconstructed = train["mean"] + (test_centered @ basis) @ basis.T
                error = angles(test, reconstructed)
                pca_angles.append(quantiles(error, HEAD_QUANTILES))
                pooled_pca_angles[rank - 1].append(error)
                axis_reconstruction = np.broadcast_to(train["mean"], test.shape).copy()
                axis_reconstruction[:, axes[:rank]] = test[:, axes[:rank]]
                axis_error = angles(test, axis_reconstruction)
                axis_angles.append(quantiles(axis_error, HEAD_QUANTILES))
                pooled_axis_angles[rank - 1].append(axis_error)
            probability = fitted["eigenvalues"] / fitted["eigenvalues"].sum()
            positive_probability = probability[probability > 0]
            per_head.append({
                "layer": layer + 1, "head": head + 1,
                "angle_from_taylor_degrees_quantiles": quantiles(taylor_angles[:, layer, :, head]),
                "factorial_coefficient_norm_quantiles": quantiles(norms[:, layer, :, head]),
                "negative_coefficient_fraction_per_order": np.mean(sampled[:, layer, :, head] < 0, axis=(0, 1)).tolist(),
                "mixed_sign_fraction": float(mixed_sign[:, layer, :, head].mean()),
                "mean_direction_angle_from_taylor_degrees": float(angles(taylor, fitted["mean"])),
                "total_direction_variance": fitted["total_variance"],
                "direction_correlation": correlation.tolist(),
                "two_way_variance_fractions": fraction.tolist(),
                "pca_cumulative_variance_fraction": fitted["cumulative"].tolist(),
                "pca_effective_rank": float(np.exp(-np.sum(positive_probability * np.log(positive_probability)))),
                "test_pca_reconstruction_angle_degrees_quantiles": pca_angles,
                "test_axis_reconstruction_angle_degrees_quantiles": axis_angles,
                "axis_order_by_train_variance": axes.tolist(),
            })
    weighted_fractions = np.average(
        [head["two_way_variance_fractions"] for head in per_head], axis=0,
        weights=[head["total_direction_variance"] for head in per_head])
    per_layer = [{
        "layer": layer + 1,
        "angle_from_taylor_degrees_quantiles": quantiles(taylor_angles[:, layer]),
        "mixed_sign_fraction": float(mixed_sign[:, layer].mean()),
        "factorial_coefficient_norm_quantiles": quantiles(norms[:, layer]),
    } for layer in range(unit.shape[1])]
    summary = {
        "sampled_vector_count": int(norms.size),
        "factorial_coefficient_quantiles_per_order": quantiles(factorial_coefficients, axis=(0, 1, 2, 3)),
        "factorial_coefficient_norm_quantiles": quantiles(norms),
        "factorial_coefficient_norm_over_taylor_quantiles": quantiles(norms / np.sqrt(orders)),
        "angle_from_taylor_degrees_quantiles": quantiles(taylor_angles),
        "negative_coefficient_fraction_per_order": np.mean(sampled < 0, axis=(0, 1, 2, 3)).tolist(),
        "any_negative_coefficient_fraction": float(np.any(sampled < 0, axis=-1).mean()),
        "mixed_sign_fraction": float(mixed_sign.mean()),
        "opposite_taylor_hemisphere_fraction": float(np.mean(taylor_angles > 90)),
        "two_way_variance_fractions_mean_over_heads": np.mean([x["two_way_variance_fractions"] for x in per_head], axis=0).tolist(),
        "two_way_variance_fractions_weighted_by_head_variance": weighted_fractions.tolist(),
        "test_pca_reconstruction_angle_degrees_quantiles_by_rank": [quantiles(np.concatenate(x)) for x in pooled_pca_angles],
        "test_axis_reconstruction_angle_degrees_quantiles_by_rank": [quantiles(np.concatenate(x)) for x in pooled_axis_angles],
    }
    for key in ("pca_cumulative_variance_fraction", "pca_effective_rank", "two_way_variance_fractions",
                "mean_direction_angle_from_taylor_degrees"):
        summary[key + "_head_quantiles"] = quantiles([x[key] for x in per_head], HEAD_QUANTILES, axis=0)
    return {"summary": summary, "per_layer": per_layer, "per_head": per_head}


def token_identity_analysis(sampled, tokens, positions, train_count, minimum_position=64):
    """A token-only lower bound on reproducing the selected first-layer directions.

    Before the first attention operation the current generator receives only a
    normalized token embedding. Equal token IDs therefore produce equal inputs,
    even at different positions or in different sequences. RoPE affects Q/K later.
    The oracle's labels need not satisfy this consistency constraint.
    """
    keep = positions >= minimum_position
    if not np.any(keep):
        raise ValueError("No sampled query positions meet the token-lookup threshold")
    coefficients = sampled[:, 0, keep, :, :]
    factorial = np.asarray([math.factorial(order) for order in range(coefficients.shape[-1])])
    coefficients = coefficients * factorial
    coefficients /= np.linalg.norm(coefficients, axis=-1, keepdims=True)
    token_ids = tokens[:, positions[keep]].reshape(-1)
    values = coefficients.reshape(-1, coefficients.shape[-2], coefficients.shape[-1])
    unique, inverse, counts = np.unique(token_ids, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique), *values.shape[1:]), np.float64)
    np.add.at(sums, inverse, values)
    means = sums / counts[:, None, None]
    group_summaries = []
    for threshold in (1, 2, 8):
        selected = counts[inverse] >= threshold
        subset = values[selected]
        errors = subset - means[inverse[selected]]
        variance = np.mean(np.sum((subset - subset.mean(axis=0)) ** 2, axis=-1), axis=0)
        residual = np.mean(np.sum(errors ** 2, axis=-1), axis=0)
        r2 = 1 - residual / variance
        group_summaries.append({
            "minimum_token_occurrences": threshold,
            "observation_count": int(selected.sum()),
            "token_type_count": int(np.sum(counts >= threshold)),
            "observation_fraction": float(selected.mean()),
            "explained_direction_variance_fraction_per_head": r2.tolist(),
            "explained_direction_variance_fraction_pooled": float(1 - residual.sum() / variance.sum()),
            "within_token_direction_variance_fraction_pooled": float(residual.sum() / variance.sum()),
            "angle_to_token_mean_degrees_quantiles": quantiles(angles(subset, means[inverse[selected]])),
        })
    train_size = train_count * int(keep.sum())
    train_values, test_values = values[:train_size], values[train_size:]
    train_inverse, test_inverse = inverse[:train_size], inverse[train_size:]
    train_counts = np.bincount(train_inverse, minlength=len(unique))
    train_sum = np.zeros_like(sums)
    np.add.at(train_sum, train_inverse, train_values)
    train_mean = train_values.mean(axis=0)
    train_lookup = np.broadcast_to(train_mean, sums.shape).copy()
    seen = train_counts > 0
    train_lookup[seen] = train_sum[seen] / train_counts[seen, None, None]
    predicted = train_lookup[test_inverse]
    error = np.mean(np.sum((test_values - predicted) ** 2, axis=-1), axis=0)
    baseline = np.mean(np.sum((test_values - train_mean) ** 2, axis=-1), axis=0)
    lookup_angles = angles(test_values, predicted)
    # Rare-token averages can overfit. Choose one shrinkage strength using only
    # the designated training sequences, then evaluate it on the held sequences.
    inner_sequences = 3 * train_count // 4
    inner_size = inner_sequences * int(keep.sum())
    inner_mean = values[:inner_size].mean(axis=0)
    inner_counts = np.bincount(inverse[:inner_size], minlength=len(unique))
    inner_sum = np.zeros_like(sums)
    np.add.at(inner_sum, inverse[:inner_size], values[:inner_size])
    validation = values[inner_size:train_size]
    validation_inverse = inverse[inner_size:train_size]
    validation_baseline = float(np.mean(np.sum((validation - inner_mean) ** 2, axis=-1)))
    strengths = [0., 1., 4., 16., 64., 256., 1024., "constant_mean"]

    def shrunk_lookup(total, count, mean, strength):
        if strength == "constant_mean":
            return np.broadcast_to(mean, total.shape).copy()
        denominator = count[:, None, None] + strength
        return np.divide(total + strength * mean, denominator,
                         out=np.broadcast_to(mean, total.shape).copy(), where=denominator > 0)

    validation_errors = []
    for strength in strengths:
        candidate = shrunk_lookup(inner_sum, inner_counts, inner_mean, strength)[validation_inverse]
        validation_errors.append(float(np.mean(np.sum((validation - candidate) ** 2, axis=-1))))
    selected_strength = strengths[int(np.argmin(validation_errors))]
    regularized = shrunk_lookup(train_sum, train_counts, train_mean, selected_strength)[test_inverse]
    regularized_error = np.mean(np.sum((test_values - regularized) ** 2, axis=-1), axis=0)
    return {
        "minimum_query_position_zero_based": minimum_position,
        "query_positions_zero_based": positions[keep].tolist(),
        "sampled_token_count": int(len(token_ids)), "unique_token_count": int(len(unique)),
        "train_token_count": train_size, "test_token_count": int(len(token_ids) - train_size),
        "test_seen_token_fraction": float(np.mean(seen[test_inverse])),
        "optimistic_within_sample_lookup": group_summaries,
        "held_sequence_lookup": {
            "explained_direction_variance_fraction_per_head": (1 - error / baseline).tolist(),
            "explained_direction_variance_fraction_pooled": float(1 - error.sum() / baseline.sum()),
            "angle_to_lookup_degrees_quantiles": quantiles(lookup_angles),
            "constant_training_mean_angle_degrees_quantiles": quantiles(angles(test_values, train_mean)),
            "unseen_token_fallback": "global direction mean over the training sequences, separately per head",
        },
        "regularized_held_sequence_lookup": {
            "pseudocount_grid": strengths,
            "selection_train_sequence_count": inner_sequences,
            "selection_validation_sequence_count": train_count - inner_sequences,
            "validation_error_reduction_fraction_by_pseudocount": [1 - error / validation_baseline for error in validation_errors],
            "selected_pseudocount": selected_strength,
            "selection_objective": "mean squared coefficient-direction error across heads; held64 sequences unused for selection",
            "explained_direction_variance_fraction_per_head": (1 - regularized_error / baseline).tolist(),
            "explained_direction_variance_fraction_pooled": float(1 - regularized_error.sum() / baseline.sum()),
            "angle_to_lookup_degrees_quantiles": quantiles(angles(test_values, regularized)),
        },
        "lookup_representative": "unconstrained arithmetic mean of unit coefficient directions; its norm may be less than one, making squared-error bounds optimistic",
        "interpretation": "Equal token IDs give identical first-layer generator inputs, but free coefficients can differ across their occurrences. This bounds reproduction of these chosen oracle directions by any token-only map.",
        "limits": [
            "First layer only; later-layer causal representations contain context.",
            "The oracle is independently fitted with target access for every sequence, including those used to test lookup transfer.",
            "The result constrains reproducing these specific coefficient directions, not attainable NLL; other equally good coefficients might satisfy token consistency.",
            "Ignoring early query positions reduces a simple source of coefficient nonidentifiability but does not establish functional identifiability.",
        ],
    }


def measure(run, overwrite=False):
    run = run.resolve()
    record = load_run(run)
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    if record["status"] != "complete" or recipe.get("oracle", {}).get("parameterization") != "free_per_example_query_coefficients":
        raise ValueError("Requires a completed independent free-coefficient fit")
    checkpoint, completion = geometry.checked_checkpoint(run, record, recipe)
    previous_path = run / "batch_coefficient_geometry.json"
    previous = json.loads(previous_path.read_text())
    if (previous["config_sha256"] != record["config"]["sha256"]
            or previous["checkpoint_sha256"] != record["selected_checkpoint"]["sha256"]):
        raise ValueError("Existing geometry does not describe the retained checkpoint")
    positions = np.asarray(previous["settings"]["query_positions_zero_based"])
    train_count = len(previous["settings"]["train_sequence_indices"])
    sampled = geometry.sample_coefficients(checkpoint, recipe, positions)
    measured = analyze(sampled, train_count)
    from scaffolds.transformer.data.packed import PackedBatchIterator
    identity = json.loads((run / "data_identity.json").read_text())
    if geometry.sha256(Path(identity["manifest"])) != identity["manifest_sha256"]:
        raise ValueError("Fitting manifest checksum changed")
    tokens = next(iter(PackedBatchIterator(
        identity["manifest"], batch_size=identity["sequence_count"],
        sequence_length=identity["sequence_length"], start_sequence=identity["start_sequence"])))
    if hashlib.sha256(tokens.astype(np.int32).tobytes()).hexdigest() != identity["batch_sha256"]:
        raise ValueError("Sampled tokens differ from the pinned fitting batch")
    measured["first_layer_token_identity"] = token_identity_analysis(sampled, tokens, positions, train_count)
    if not np.allclose(measured["summary"]["pca_cumulative_variance_fraction_head_quantiles"][1],
                       previous["summary"]["pca_cumulative_variance_fraction_quantiles"][2],
                       rtol=1e-10, atol=1e-12):
        raise ValueError("PCA differs from the existing geometry on identical samples")
    result = {
        "schema_version": 1, "run_id": run.name, "parent_run_id": recipe["oracle"]["parent_run_id"],
        "config_sha256": record["config"]["sha256"],
        "checkpoint_sha256": record["selected_checkpoint"]["sha256"],
        "selected_checkpoint_step": completion["best_step"],
        "completion_sha256": geometry.sha256(run / "completion.json"),
        "geometry_sha256": geometry.sha256(previous_path),
        "fitting_batch_sha256": identity["batch_sha256"],
        "dataset_manifest_sha256": identity["manifest_sha256"],
        "analysis_script_sha256": geometry.sha256(Path(__file__)),
        "geometry_script_sha256": geometry.sha256(Path(geometry.__file__)),
        "analysis_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dependency_lock_sha256": geometry.sha256(ROOT / "scaffolds/transformer/uv.lock"),
        "checkpoint_verification": "recorded file names and sizes; coefficient_identity.json SHA256",
        "settings": {
            "coordinates": "factorial(order) times effective monomial coefficient; unit positive L2 normalization for directional statistics",
            "query_positions_zero_based": positions.tolist(), "train_sequence_count": train_count,
            "test_sequence_count": recipe["oracle"]["sequence_count"] - train_count,
            "quantile_probabilities": QUANTILES, "head_quantile_probabilities": HEAD_QUANTILES,
            "two_way_variance_order": ["sequence_mean", "query_position_mean", "sequence_by_position_remainder"],
            "pca_reconstruction": "mean plus first k components fitted on training sequences, followed by positive L2 normalization",
            "axis_reconstruction": "keep k coordinate axes ranked by training-sequence variance; other coordinates fixed at training mean",
            "angle_weighting": "equal sampled sequence/query/head weight; no score, value, gradient, or NLL weighting",
        },
        "limits": [
            "Only 64 uniformly spaced query positions are sampled; every fitting sequence is represented.",
            "A negative coefficient does not imply a negative polynomial value at an observed score.",
            "Positive common scaling is only approximately redundant because output RMSNorm has epsilon and finite precision.",
            "Angular/PCA reconstruction measures coefficient shape, not preserved attention outputs or fitting perplexity.",
            "The sequence split tests a basis on other independently fitted coefficient tables, not unseen-language-model performance.",
            "A five-coefficient polynomial direction already has four intrinsic degrees of freedom; compressible outputs need not be easy to predict from causal inputs.",
            "Oracle activations can contain target information through earlier free coefficients; predictability should first be measured from the original causal parent activations.",
        ], **measured,
    }
    geometry.atomic_write(run / OUTPUT, result, overwrite)
    update_record(run, capture_artifacts(run))
    print(json.dumps({"run_id": run.name, "output": str(run / OUTPUT), "summary": measured["summary"]}, sort_keys=True))


def plot(selection_path, output):
    output = output.with_suffix("")
    selection = yaml.safe_load(selection_path.read_text())
    entries = [x for x in selection["runs"] if x.get("role") == ROLE]
    if len(entries) != 1:
        raise ValueError(f"Selection must name exactly one {ROLE} run")
    entry = entries[0]
    run = ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"]
    descriptor = entry.get("diagnostics", {}).get("free_coefficient_shape")
    if not descriptor or descriptor["path"] != str((run / OUTPUT).relative_to(ROOT)):
        raise ValueError("Pin free_coefficient_shape.json in this run's selection diagnostics before plotting")
    if geometry.sha256(run / OUTPUT) != descriptor["sha256"]:
        raise ValueError("The selected coefficient shape measurement checksum changed")
    data = json.loads((run / OUTPUT).read_text())
    if data["config_sha256"] != entry["sources"]["config.yaml"]["sha256"]:
        raise ValueError("The selected recipe differs from the shape measurement")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "m4-free-coefficient-shape"})
    summary = data["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
    fig.suptitle("Free coefficients at m = 4: shape and variation", fontsize=18, weight="bold")
    layer = np.asarray([x["layer"] for x in data["per_layer"]])
    angle = np.asarray([x["angle_from_taylor_degrees_quantiles"] for x in data["per_layer"]])
    axes[0, 0].fill_between(layer, angle[:, 2], angle[:, 4], alpha=.18, color="#2878b5", label="10–90% of sampled vectors")
    axes[0, 0].plot(layer, angle[:, 3], color="#2878b5", marker="o", label="Median")
    axes[0, 0].set(xlabel="Layer", ylabel="Angle from Taylor direction (degrees)", title="Coefficient directions depart from Taylor")
    axes[0, 0].set_xticks([1, 5, 10, 15]); axes[0, 0].legend(frameon=False, fontsize=9)
    rank = np.arange(1, len(summary["negative_coefficient_fraction_per_order"]) + 1)
    pca_error = np.asarray(summary["test_pca_reconstruction_angle_degrees_quantiles_by_rank"])
    axis_error = np.asarray(summary["test_axis_reconstruction_angle_degrees_quantiles_by_rank"])
    axes[0, 1].plot(rank, pca_error[:, 3], marker="o", color="#2878b5", label="PCA basis")
    axes[0, 1].fill_between(rank, pca_error[:, 2], pca_error[:, 4], color="#2878b5", alpha=.18)
    axes[0, 1].plot(rank, axis_error[:, 3], marker="s", color="#dc7b21", label="Best coefficient axes")
    axes[0, 1].set(xlabel="Retained components", ylabel="Reconstruction angle (degrees)", xticks=rank,
                   title="Basis reconstruction on\nother fitted sequences")
    axes[0, 1].legend(frameon=False, fontsize=9)
    orders = rank - 1
    axes[1, 0].bar(orders, 100 * np.asarray(summary["negative_coefficient_fraction_per_order"]), color="#8b66b2")
    axes[1, 0].set(xlabel="Polynomial order", ylabel="Negative coefficients (%)", xticks=orders,
                   title="Signed coefficients occur at every order")
    fractions = np.asarray(summary["two_way_variance_fractions_mean_over_heads"])
    names = ["Sequence mean", "Query-position mean", "Sequence × position"]
    colors = ["#dc7b21", "#2878b5", "#20946f"]
    axes[1, 1].barh(np.arange(3), 100 * fractions, color=colors)
    for index, fraction in enumerate(fractions):
        axes[1, 1].text(min(100 * fraction + 1.5, 93), index, f"{100 * fraction:.1f}%", va="center")
    axes[1, 1].set(xlabel="Direction variance explained (%)", yticks=np.arange(3), yticklabels=names,
                   xlim=(0, 105), title="Most variation is specific\nto a sequence and query")
    axes[1, 1].invert_yaxis()
    for ax in axes.flat:
        ax.grid(alpha=.18, axis="y"); ax.set_axisbelow(True)
    fig.supxlabel("256 fitted sequences · 64 sampled positions · 210 layer/head pairs\n"
                  "Factorial coefficient directions; positive common scale removed. Basis fit on 192 sequences, tested on 64.\n"
                  "No attention-score or loss weighting: reconstruction does not establish preserved PPL or causal predictability.",
                  fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, metadata={"Software": "Matplotlib"})
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    lookup = data["first_layer_token_identity"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    fig.suptitle("The first-layer oracle is not a token-only coefficient map", fontsize=17, weight="bold")
    heads = np.arange(1, len(lookup["held_sequence_lookup"]["explained_direction_variance_fraction_per_head"]) + 1)
    for group, color, label in zip((lookup["optimistic_within_sample_lookup"][1], lookup["optimistic_within_sample_lookup"][2]),
                                    ("#2878b5", "#dc7b21"), ("Token occurs ≥2 times", "Token occurs ≥8 times")):
        axes[0].plot(heads, 100 * np.asarray(group["explained_direction_variance_fraction_per_head"]),
                     marker="o", color=color, label=label)
    axes[0].set(xlabel="Head", ylabel="Coefficient-direction variance explained (%)",
                title="Optimistic token lookup on the entire fitted sample")
    axes[0].legend(frameon=False, fontsize=9)
    r2 = 100 * np.asarray(lookup["regularized_held_sequence_lookup"]["explained_direction_variance_fraction_per_head"])
    raw_r2 = 100 * np.asarray(lookup["held_sequence_lookup"]["explained_direction_variance_fraction_per_head"])
    axes[1].bar(heads, r2, color="#20946f", label="Shrinkage selected on training sequences")
    axes[1].plot(heads, raw_r2, marker="o", color="#888888", label="Unregularized lookup")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set(xlabel="Head", ylabel="Error reduction versus constant mean (%)",
                title="Token lookup learned on 192 sequences, tested on 64")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks([1, 4, 7, 10, 14]); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.supxlabel("First-layer generator input = normalized token embedding; equal token IDs give identical inputs.\n"
                  "Positions ≥64 only. Labels are independently fitted oracle directions, including on the test sequences.\n"
                  "This tests coefficient consistency, not attainable loss or whether another equally good coefficient solution exists.", fontsize=9)
    token_output = output.with_name(output.name + "_token_lookup")
    fig.savefig(token_output.with_suffix(".png"), dpi=180, metadata={"Software": "Matplotlib"})
    fig.savefig(token_output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(output.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measurement = sub.add_parser("measure")
    measurement.add_argument("--run", type=Path, required=True)
    measurement.add_argument("--overwrite", action="store_true")
    rendering = sub.add_parser("plot")
    rendering.add_argument("--selection", type=Path, default=Path(__file__).with_name("selection.yaml"))
    rendering.add_argument("--output", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.run, args.overwrite)
    else:
        plot(args.selection, args.output)


if __name__ == "__main__":
    main()
