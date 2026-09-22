"""Inspect the state in Section 3.2 of overleaf/iclr2027_conference.tex.

Llama, all KV heads: uv run scaffolds/state_rank/spectrum.py --layer 8 --tokens 512 --output results/run
Select one Llama KV head: uv run scaffolds/state_rank/spectrum.py --layer 8 --kv-head 0
Gated DeltaNet: uv run scaffolds/state_rank/spectrum.py --model gated-deltanet --layer 12
Refresh saved figures/metrics: uv run scaffolds/state_rank/spectrum.py --replot results/run
For Slurm, run uv sync --frozen first, then .venv/bin/python scaffolds/state_rank/spectrum.py.
The saved eigenvalues describe the attention state before output processing.
Threshold rank counts state singular values > 1e-4 times the largest.
Participation ratio uses squared singular values; neither metric must grow monotonically.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

# Keep small matrix operations from spawning threads across the whole host.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, os.environ.get("SLURM_CPUS_PER_TASK", "4"))

import numpy as np
import torch


MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-2-raw-v1"
GDN_MODEL_ID = "m-a-p/1.3B-100B-GatedDeltaNet-pure"
GDN_REVISION = "930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa"
FLA_REVISION = "516143e31fce09925e6c39ac37148444bad176c4"
RANK_TOL = 1e-4
CACHE_RTOL = 2e-4
PSD_RTOL = 1e-10
DIRECT_RTOL = 1e-9


def iter_article_bodies(rows):
    """Split WikiText rows at top-level titles, omitting all heading levels."""
    title, paragraphs = None, []
    for row in rows:
        line = row.strip()
        if not line:
            continue
        heading = re.fullmatch(r"(?:=\s*)+(.+?)(?:\s*=)+", line)
        if heading:
            level = re.match(r"(?:=\s*)+", line).group().count("=")
            if level == 1:
                if title is not None and paragraphs:
                    yield {"title": title, "text": "\n\n".join(paragraphs)}
                title, paragraphs = heading.group(1).strip(), []
            continue
        if title is not None:
            paragraphs.append(line)
    if title is not None and paragraphs:
        yield {"title": title, "text": "\n\n".join(paragraphs)}


def select_inputs(tokenizer, rows, tokens, count=4, article_indices=None):
    """Select the first qualifying articles, saving the exact model inputs."""
    if tokens < 2 or count < 1:
        raise ValueError("Require at least two tokens and one input.")
    if tokenizer.bos_token_id is None:
        raise ValueError("The tokenizer must define a BOS token.")
    selected = []
    for article_index, article in enumerate(iter_article_bodies(rows)):
        if article_indices is not None and article_index not in article_indices:
            continue
        ids = tokenizer.encode(
            article["text"], add_special_tokens=False,
            truncation=True, max_length=tokens - 1,
        )
        if len(ids) < tokens - 1:
            continue
        selected.append({
            "title": article["title"],
            "article_index": article_index,
            "body_sha256": hashlib.sha256(article["text"].encode()).hexdigest(),
            "text": tokenizer.decode(ids, clean_up_tokenization_spaces=False),
            "token_ids": [tokenizer.bos_token_id, *ids],
        })
        if len(selected) == count:
            return selected
    raise ValueError(f"Found only {len(selected)} articles with {tokens - 1} text tokens.")


def _validate_kv(keys, values, scale):
    keys, values = np.asarray(keys, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if (keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]
            or min(*keys.shape, *values.shape) < 1):
        raise ValueError("Keys and values must be nonempty [tokens, dimension] arrays.")
    if not np.isfinite(keys).all() or not np.isfinite(values).all():
        raise ValueError("Keys and values must be finite.")
    scale = keys.shape[1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Attention scale must be positive and finite.")
    return keys, values, scale


def iter_prefix_matrices(keys, values, scale=None):
    """Yield (exp(-c_t) B_t, c_t) using the exact prefix recurrence.

    c_t is the largest scaled key squared norm in the current prefix. By
    Cauchy-Schwarz it bounds every key-key log kernel entry in that prefix.
    Each yielded matrix is independent of subsequent updates.
    """
    keys, values, scale = _validate_kv(keys, values, scale)
    diagonal = np.einsum("ij,ij->i", keys, keys) * scale
    if not np.isfinite(diagonal).all():
        raise FloatingPointError("Key squared norms exceed float64 range.")
    matrix = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    shift = -np.inf
    for index, value in enumerate(values):
        next_shift = max(shift, float(diagonal[index]))
        matrix *= np.exp(shift - next_shift)
        shift = next_shift
        logits = (keys[:index + 1] @ keys[index]) * scale
        logits[-1] = diagonal[index]
        weights = np.exp(logits - shift)
        cross = weights[:-1] @ values[:index]
        matrix += (np.outer(cross, value) + np.outer(value, cross)
                   + weights[-1] * np.outer(value, value))
        matrix = (matrix + matrix.T) * 0.5
        if not np.isfinite(matrix).all():
            raise FloatingPointError(f"Nonfinite state matrix at prefix {index + 1}.")
        yield matrix.copy(), shift


def direct_prefix_matrix(keys, values, log_scale, scale=None):
    """Reference calculation with a full key-key Gram matrix; no softmax/mask."""
    keys, values, scale = _validate_kv(keys, values, scale)
    gram = np.exp((keys @ keys.T) * scale - log_scale)
    matrix = values.T @ gram @ values
    return (matrix + matrix.T) * 0.5


def compute_spectra(keys, values, check_prefixes=(), scale=None):
    """Return every prefix spectrum, scale, energy summary, and diagnostics.

    lambda_i(B_t) = exp(log_scale[t]) * scaled_eigenvalues[t, i].
    Zero states have zero normalized spectrum/ranks and log_trace = -inf.
    Only negative eigenvalues within PSD_RTOL of the spectral scale are clipped.
    """
    keys, values, scale = _validate_kv(keys, values, scale)
    length, width = values.shape
    checks = set(check_prefixes)
    if any(not isinstance(t, (int, np.integer)) or not 1 <= t <= length for t in checks):
        raise ValueError("Check prefixes must be integers between 1 and the input length.")
    result = {
        "scaled_eigenvalues": np.zeros((length, width)),
        "log_scale": np.zeros(length),
        "key_norms": np.linalg.norm(keys, axis=1),
        "raw_min_eigenvalue": np.zeros(length),
        "direct_check_relative_error": np.full(length, np.nan),
    }
    for index, (matrix, shift) in enumerate(iter_prefix_matrices(keys, values, scale)):
        eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
        spectral_scale = max(float(np.abs(eigenvalues).max()), np.finfo(np.float64).tiny)
        result["raw_min_eigenvalue"][index] = eigenvalues[-1]
        if eigenvalues[-1] < -PSD_RTOL * spectral_scale:
            raise FloatingPointError(
                f"Materially negative eigenvalue at prefix {index + 1}: "
                f"{eigenvalues[-1]:.6g} (spectral scale {spectral_scale:.6g})."
            )
        eigenvalues = np.maximum(eigenvalues, 0)
        result["scaled_eigenvalues"][index] = eigenvalues
        result["log_scale"][index] = shift
        if index + 1 in checks:
            direct = direct_prefix_matrix(keys[:index + 1], values[:index + 1], shift, scale)
            error = np.linalg.norm(matrix - direct) / max(
                np.linalg.norm(direct), np.finfo(np.float64).tiny,
            )
            result["direct_check_relative_error"][index] = error
            if error > DIRECT_RTOL:
                raise FloatingPointError(
                    f"Direct matrix check failed at prefix {index + 1}: relative error {error:.3g}."
                )
    result.update(spectral_metrics(result["scaled_eigenvalues"], result["log_scale"]))
    return result


def threshold_rank(spectrum, tolerance):
    """Strict relative cutoff on the last axis; zero spectrum has rank zero."""
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if not np.isfinite(spectrum).all() or np.any(spectrum < 0):
        raise ValueError("Spectrum must be finite and nonnegative.")
    return np.count_nonzero(spectrum > tolerance * spectrum.max(axis=-1, keepdims=True), axis=-1)


def spectral_metrics(eigenvalues, log_scale):
    """Energy metrics from descending, nonnegative eigenvalues; last axis is modes."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    ranks = threshold_rank(eigenvalues, RANK_TOL**2)
    maximum = eigenvalues.max(axis=-1, keepdims=True)
    relative = np.divide(eigenvalues, maximum, out=np.zeros_like(eigenvalues), where=maximum > 0)
    total = relative.sum(axis=-1, keepdims=True)
    normalized = np.divide(relative, total, out=np.zeros_like(relative), where=total > 0)
    squares = np.square(normalized).sum(axis=-1)
    participation = np.divide(1., squares, out=np.zeros_like(squares), where=squares > 0)
    cumulative = normalized.cumsum(axis=-1)
    positive = maximum[..., 0] > 0
    log_trace = np.full(positive.shape, -np.inf)
    log_trace[positive] = (np.log(maximum[..., 0][positive]) + np.log(total[..., 0][positive])
                           + np.broadcast_to(log_scale, positive.shape)[positive])
    return {
        "normalized_eigenvalues": normalized,
        "participation_ratio": participation,
        "threshold_rank": ranks,
        "energy_rank90": np.where(positive, np.minimum((cumulative < .90).sum(axis=-1) + 1, eigenvalues.shape[-1]), 0),
        "energy_rank99": np.where(positive, np.minimum((cumulative < .99).sum(axis=-1) + 1, eigenvalues.shape[-1]), 0),
        "log_trace": log_trace,
    }


def iter_gdn_states(model, input_ids, layer, heads):
    """Yield copies of native [head, value, key] states, preserving all caches."""
    if model.training or input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
        raise ValueError("Require an eval model and one nonempty [1, tokens] input.")
    if not 0 <= layer < len(model.layers):
        raise ValueError("Layer outside model range.")
    if not heads or any(not 0 <= h < model.layers[layer].attn.num_v_heads for h in heads):
        raise ValueError("Head outside model range.")
    device = next(model.parameters()).device
    cache = None
    with torch.inference_mode():
        for token in input_ids.to(device).split(1, dim=1):
            outputs = model(input_ids=token, past_key_values=cache, use_cache=True, return_dict=True)
            cache = outputs.past_key_values
            state = cache[layer]["recurrent_state"]
            if state.dtype != torch.float32:
                raise RuntimeError(f"Expected float32 recurrent state, got {state.dtype}.")
            yield state[0, heads].to(device="cpu", dtype=torch.float64).numpy().copy()


def compute_gdn_spectra(model, input_ids, layer, heads, check_prefixes=()):
    """Native float32 state evolution, float64 SVD, and independent prefix checks."""
    spectra, checked_states = [], {}
    for t, state in enumerate(iter_gdn_states(model, input_ids, layer, heads), 1):
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite Gated DeltaNet state at prefix {t}.")
        spectra.append(np.linalg.svd(state, compute_uv=False))
        if t in check_prefixes:
            checked_states[t] = state
        if t % 64 == 0:
            print(f"  analyzed prefix {t}", flush=True)
    singular = np.stack(spectra)
    eigenvalues = singular**2
    result = {"singular_values": singular, "scaled_eigenvalues": eigenvalues,
              "log_scale": np.zeros(eigenvalues.shape[:-1])}
    result.update(spectral_metrics(eigenvalues, result["log_scale"]))
    result["threshold_rank"] = threshold_rank(singular, RANK_TOL)
    errors = np.full(input_ids.shape[1], np.nan)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for t, expected in checked_states.items():
            outputs = model(input_ids=input_ids[:, :t].to(device), use_cache=True, return_dict=True)
            actual = outputs.past_key_values[layer]["recurrent_state"][0, heads].double().cpu().numpy()
            per_head_error = np.linalg.norm(actual - expected, axis=(-2, -1)) / np.maximum(
                np.linalg.norm(expected, axis=(-2, -1)), np.finfo(float).tiny)
            errors[t - 1] = per_head_error.max()
            if errors[t - 1] > CACHE_RTOL:
                raise FloatingPointError(f"Prefix {t} cache mismatch: {errors[t - 1]:.3g}.")
    result["prefix_check_relative_error"] = errors
    return result


def compute_llama_spectra(keys, values, check_prefixes=(), scale=None):
    """Analyze [tokens, heads, dimension] arrays independently for every KV head."""
    keys, values = np.asarray(keys), np.asarray(values)
    if (keys.ndim != 3 or values.ndim != 3 or keys.shape[:2] != values.shape[:2]
            or min(*keys.shape, *values.shape) < 1):
        raise ValueError("Require matching nonempty [tokens, heads, dimension] arrays.")
    per_head = [compute_spectra(keys[:, head], values[:, head], check_prefixes, scale)
                for head in range(keys.shape[1])]
    return {key: np.stack([result[key] for result in per_head], axis=1) for key in per_head[0]}


def extract_kv_layers(model, input_ids, layers):
    """Read selected layers' native post-RoPE KV caches in one causal pass.

    Return an insertion-ordered dict from layer index to independent float64
    (keys, values) arrays with shape [tokens, KV heads, dimension]. These are
    the distinct KV heads, before repetition across query heads. Layer indices
    must be unique; their requested order is preserved.
    """
    try:
        layers = tuple(layers)
    except TypeError as error:
        raise ValueError("Layers must be a nonempty iterable of integer indices.") from error
    if not layers:
        raise ValueError("Layers must be nonempty.")
    if any(not isinstance(layer, (int, np.integer)) or isinstance(layer, (bool, np.bool_))
           or not 0 <= layer < model.config.num_hidden_layers for layer in layers):
        raise ValueError(f"Layers must be integer indices in [0, {model.config.num_hidden_layers - 1}].")
    layers = tuple(int(layer) for layer in layers)
    if len(set(layers)) != len(layers):
        raise ValueError("Layer indices must be unique.")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Extract one nonempty sequence at a time, with shape [1, tokens].")
    if model.training:
        raise ValueError("Call model.eval() before extracting the attention state.")
    device = next(model.parameters()).device
    with torch.inference_mode():
        outputs = model(input_ids=input_ids.to(device), use_cache=True, return_dict=True)
        return {
            layer: tuple(tensor[0].transpose(0, 1).to(device="cpu", dtype=torch.float64).numpy().copy()
                         for tensor in outputs.past_key_values[layer])
            for layer in layers
        }


def extract_kv(model, input_ids, layer, kv_head=None):
    """Read one layer's native post-RoPE KV cache in one causal pass.

    Default: [tokens, KV heads, dimension]. An integer selects [tokens, dimension].
    These are the distinct KV heads, before repetition across query heads.
    """
    if kv_head is not None and not 0 <= kv_head < model.config.num_key_value_heads:
        raise ValueError(f"KV head must be in [0, {model.config.num_key_value_heads - 1}].")
    keys, values = extract_kv_layers(model, input_ids, (layer,))[layer]
    if kv_head is None:
        return keys, values
    return keys[:, kv_head].copy(), values[:, kv_head].copy()


def plot_results(results, titles, output, context=None, layer=None, model_name=None):
    """Write publication-exportable PNG and PDF figures from saved arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output = Path(output)
    if results["scaled_eigenvalues"].ndim == 4:
        if model_name is None:
            model_name = "Gated DeltaNet" if "singular_values" in results else "Llama"
        plot_head_results(results, titles, output, layer=layer, model_name=model_name)
        return
    count, length, width = results["scaled_eigenvalues"].shape
    positions = np.arange(1, length + 1)
    colors = plt.get_cmap("tab10").colors

    def save(figure, name):
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(count, 1, figsize=(10, 2.2 * count), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    if context:
        figure.suptitle(context)
    for index, axis in enumerate(axes):
        values = results["normalized_eigenvalues"][index].T
        plotted = np.maximum(values, 1e-6)
        mesh = axis.imshow(
            plotted, origin="lower", aspect="auto", interpolation="nearest",
            extent=(0.5, length + 0.5, 0.5, width + 0.5),
            norm=LogNorm(vmin=1e-6, vmax=1), cmap="viridis",
        )
        axis.set(title=titles[index], ylabel="Sorted mode index")
    axes[-1].set_xlabel("Prefix length t (including BOS)")
    figure.colorbar(mesh, ax=list(axes), label="Normalized eigenvalue (color floor: 10⁻⁶, including zero)", pad=0.02)
    save(figure, "spectrum")

    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    figure.suptitle(f"{context + chr(10) if context else ''}Effective state dimensions (head width: {width})")
    for axis, key, label in zip(
        axes, ("participation_ratio", "threshold_rank", "energy_rank90", "energy_rank99"),
        ("Participation ratio", "Threshold rank (σ > 10⁻⁴ σ₁)", "Modes for 90% energy", "Modes for 99% energy"),
    ):
        for index, title in enumerate(titles):
            axis.plot(positions, results[key][index], color=colors[index], label=title, linewidth=1.2)
        axis.set_ylabel(label)
        axis.set_ylim(0, min(width + 1, max(2, math.ceil(results[key].max() * 1.15))))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right", ncol=4 if context else 1)
    axes[-1].set_xlabel("Prefix length t (including BOS)")
    save(figure, "effective_rank")

    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for index, title in enumerate(titles):
        axis.plot(positions, results["log_trace"][index] / np.log(10), label=title, color=colors[index])
    axis.set(xlabel="Prefix length t (including BOS)", ylabel="log₁₀ tr(Bₜ)", title=f"{context + chr(10) if context else ''}Total squared state norm")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=4 if context else 1)
    save(figure, "state_norm")


def plot_head_results(results, titles, output, layer=None, model_name="Gated DeltaNet"):
    """Per-article figures plus sequence means/ranges with all heads together."""
    # Reuse the plotting layout with heads as curves/panels within each article.
    count, length, head_count, width = results["scaled_eigenvalues"].shape
    for index, title in enumerate(titles):
        directory = output / f"text-{index + 1}"
        directory.mkdir(exist_ok=True)
        per_head = {key: np.moveaxis(results[key][index], 1, 0)
                    for key in ("scaled_eigenvalues", "normalized_eigenvalues", "participation_ratio",
                                "threshold_rank", "energy_rank90", "energy_rank99", "log_trace")}
        head_label = "KV head" if model_name == "Llama" else "Head"
        labels = [f"{head_label} {head}" for head in results["head_indices"]]
        layer_label = f" · layer {layer}" if layer is not None else ""
        plot_results(per_head, labels, directory, context=f"{model_name}{layer_label} · {title}")
    plot_aggregate(results, output / "aggregate", layer=layer, model_name=model_name)


def plot_aggregate(results, output, layer=None, model_name="Gated DeltaNet"):
    """Average each metric across sequences, preserving token and head axes.

    Shading shows the observed min–max range, not a confidence interval.
    No temporal smoothing, averaging of states, or pooling of spectra is used.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(exist_ok=True)
    count, length, _, width = results["scaled_eigenvalues"].shape
    positions = np.arange(1, length + 1)
    colors = plt.get_cmap("tab10").colors
    fields = (
        ("participation_ratio", "Participation ratio"),
        ("threshold_rank", "Threshold rank (σ > 10⁻⁴ σ₁)"),
        ("energy_rank90", "Modes for 90% energy"),
        ("energy_rank99", "Modes for 99% energy"),
    )
    summaries = {key: (results[key].mean(axis=0), results[key].min(axis=0), results[key].max(axis=0))
                 for key, _ in fields}
    layer_label = f" · layer {layer}" if layer is not None else ""
    title = (f"{model_name}{layer_label} · state width {width}\n"
             f"Mean across {count} sequences; shading: min–max range")
    head_label = "KV head" if model_name == "Llama" else "Head"

    def draw(axis, key, label):
        mean, lower, upper = summaries[key]
        for index, head in enumerate(results["head_indices"]):
            color = colors[index % len(colors)]
            axis.fill_between(positions, lower[:, index], upper[:, index],
                              color=color, alpha=0.10, linewidth=0, zorder=1)
            axis.plot(positions, mean[:, index], color=color, linewidth=1.6,
                      label=f"{head_label} {head}", zorder=2)
        limit = min(width + 1, max(2, math.ceil(upper.max() * 1.10)))
        if key == "threshold_rank":
            axis.plot(positions, np.minimum(positions, width), color="black", linestyle="--",
                      linewidth=1.1, label=f"Ceiling min(t, {width})", zorder=3)
            limit = width * 1.04
        axis.set(ylabel=label, xlim=(1, length), ylim=(0, limit))
        axis.grid(alpha=0.2)

    def save(figure, axes, name, legend_axis):
        axes[-1].set_xlabel("Prefix length t (including BOS)")
        figure.suptitle(title)
        handles, labels = legend_axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for axis, (key, label) in zip(axes, fields):
        draw(axis, key, label)
    save(figure, axes, "effective_rank", axes[1])


def replot(directory):
    """Refresh derived metrics and figures without model loading or inference."""
    with np.load(directory / "spectra.npz") as archive:
        results = dict(archive)
    results.update(spectral_metrics(results["scaled_eigenvalues"], results["log_scale"]))
    if "singular_values" in results:
        results["threshold_rank"] = threshold_rank(results["singular_values"], RANK_TOL)
    inputs = json.loads((directory / "inputs.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["threshold_rank_singular_tolerance"] = RANK_TOL
    metadata["threshold_rank_comparison"] = "strictly greater than tolerance * largest singular value"
    metadata["replotted_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["replot_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Replace the archive atomically so interrupted replotting cannot corrupt spectra.
    temporary = directory / "spectra.tmp.npz"
    np.savez_compressed(temporary, **results)
    temporary.replace(directory / "spectra.npz")
    plot_results(results, [item["title"] for item in inputs], directory, layer=metadata["layer"])
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Updated threshold ranks and plots in {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Indices are zero-based. Four WikiText-2 test articles are selected using Llama,\n"
                "then tokenized separately for each model: BOS + tokens-1 text tokens.\n"
                "Outputs: spectra.npz, inputs.json, metadata.json, and PNG/PDF figures.\n"
                "All heads are analyzed by default, with per-text plots under text-N/\n"
                "and sequence mean/min–max rank plots under aggregate/.\n"
                "Original eigenvalues = exp(log_scale) * scaled_eigenvalues.\n"
                "Gated DeltaNet requires CUDA. Output must be a new directory.\n"
                "--replot updates saved metrics/figures in place without inference."),
    )
    parser.add_argument("--model", choices=("llama", "gated-deltanet"), default="llama")
    parser.add_argument("--layer", type=int, help="Layer index (default: Llama 8, Gated DeltaNet 12)")
    parser.add_argument("--kv-head", type=int, help="Llama KV head (default: all KV heads)")
    parser.add_argument("--head", type=int, help="Gated DeltaNet head (default: all heads)")
    parser.add_argument("--tokens", type=int, default=512, help="Tokens per text including BOS (default: 512)")
    parser.add_argument("--output", type=Path, help="New results directory; default name includes model/layer/head/tokens")
    parser.add_argument("--replot", type=Path, metavar="DIRECTORY", help="Refresh a saved run without inference")
    args = parser.parse_args()
    if args.replot is not None:
        replot(args.replot)
        return
    gdn = args.model == "gated-deltanet"
    if args.layer is None:
        args.layer = 12 if gdn else 8
    if not gdn and args.head is not None:
        parser.error("--head is for Gated DeltaNet; use --kv-head for Llama")
    if gdn and args.kv_head is not None:
        parser.error("--kv-head is for Llama; use --head for Gated DeltaNet")
    if args.output is None:
        name = f"gated-deltanet-1.3b-l{args.layer}-h{args.head if args.head is not None else 'all'}" if gdn else f"llama-3.2-1b-l{args.layer}-kv{args.kv_head if args.kv_head is not None else 'all'}"
        args.output = Path(f"results/state_rank/historical/{name}-t{args.tokens}")
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}; choose a new --output.")
    if gdn and not torch.cuda.is_available():
        parser.error("Gated DeltaNet requires CUDA; launch through Slurm")

    from datasets import load_dataset
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    if gdn:
        import fla.models.gated_deltanet  # registers the architecture with HF Auto classes

    started = time.monotonic()
    model_id, revision = (GDN_MODEL_ID, GDN_REVISION) if gdn else (MODEL_ID, MODEL_REVISION)
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if not 0 <= args.layer < config.num_hidden_layers:
        parser.error(f"--layer must be in [0, {config.num_hidden_layers - 1}]")
    if gdn:
        heads = list(range(config.num_v_heads or config.num_heads)) if args.head is None else [args.head]
        if any(not 0 <= h < (config.num_v_heads or config.num_heads) for h in heads):
            parser.error("--head outside the model's head range")
        config.attn_mode = "fused_recurrent"  # float32 native kernel also for prefix checks
    else:
        heads = list(range(config.num_key_value_heads)) if args.kv_head is None else [args.kv_head]
        if any(not 0 <= head < config.num_key_value_heads for head in heads):
            parser.error(f"--kv-head must be in [0, {config.num_key_value_heads - 1}]")
    if args.tokens > config.max_position_embeddings:
        parser.error(f"--tokens exceeds the model context limit ({config.max_position_embeddings})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="test", revision=DATASET_REVISION)
    rows = dataset["text"]
    if gdn:
        llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        reference_inputs = select_inputs(llama_tokenizer, rows, args.tokens)
        indices = [item["article_index"] for item in reference_inputs]
        inputs = select_inputs(tokenizer, rows, args.tokens, article_indices=indices)
        if [item["body_sha256"] for item in inputs] != [item["body_sha256"] for item in reference_inputs]:
            raise RuntimeError("Article selection differs between models.")
    else:
        inputs = select_inputs(tokenizer, rows, args.tokens)
    args.output.mkdir(parents=True)
    (args.output / "inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Loading {model_id} in float32 on {device}; layer={args.layer}", flush=True)
    kwargs = {} if gdn else {"attn_implementation": "eager"}
    model, loading_info = AutoModel.from_pretrained(
        model_id, revision=revision, config=config, dtype=torch.float32,
        use_safetensors=True, output_loading_info=True, **kwargs,
    )
    allowed_unused = {"lm_head.weight"}  # AutoModel omits the language-model readout.
    if gdn:
        # Historical D was declared but never used in forward; removed upstream:
        # https://github.com/fla-org/flash-linear-attention/commit/834989cc5367049face0f5818659666b0b8c5f4a
        allowed_unused.update(f"layers.{i}.attn.D" for i in range(config.num_hidden_layers))
    unexpected = [key for key in loading_info["unexpected_keys"] if key not in allowed_unused]
    if loading_info["missing_keys"] or loading_info["mismatched_keys"] or unexpected or loading_info["error_msgs"]:
        raise RuntimeError(f"Checkpoint did not load exactly: {loading_info}")
    model = model.to(device).eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("All model parameters must be float32.")
    scale = None if gdn else model.layers[args.layer].self_attn.scaling
    check_prefixes = sorted({t for t in (1, 2, 8, 32, 128, args.tokens) if t <= args.tokens})
    all_results, all_keys, all_values = [], [], []
    for index, item in enumerate(inputs):
        print(f"[{index + 1}/{len(inputs)}] {item['title']}: extracting {args.tokens} tokens", flush=True)
        token_ids = torch.tensor([item["token_ids"]])
        if gdn:
            result = compute_gdn_spectra(model, token_ids, args.layer, heads, check_prefixes)
        else:
            keys, values = extract_kv(model, token_ids, args.layer)
            keys, values = keys[:, heads], values[:, heads]
            result = compute_llama_spectra(keys, values, check_prefixes=check_prefixes, scale=scale)
            all_keys.append(keys)
            all_values.append(values)
        all_results.append(result)
        print(f"  final participation={result['participation_ratio'][-1]}, threshold rank={result['threshold_rank'][-1]}", flush=True)

    results = {key: np.stack([item[key] for item in all_results]) for key in all_results[0]}
    results["token_ids"] = np.array([item["token_ids"] for item in inputs], dtype=np.int64)
    results["head_indices"] = np.array(heads, dtype=np.int64)
    if gdn:
        expected_shape = (4, args.tokens, len(heads), min(config.head_dim, int(config.head_dim * config.expand_v)))
    else:
        results.update(keys=np.stack(all_keys), values=np.stack(all_values))
        expected_shape = (4, args.tokens, len(heads), config.head_dim)
    if results["scaled_eigenvalues"].shape != expected_shape:
        raise RuntimeError(f"Unexpected spectrum shape: {results['scaled_eigenvalues'].shape}; expected {expected_shape}")
    np.savez_compressed(args.output / "spectra.npz", **results)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, check=False)
    diagnostic = "prefix_check_relative_error" if gdn else "direct_check_relative_error"
    packages = ["torch", "transformers", "datasets", "numpy", "matplotlib"]
    if gdn:
        packages += ["flash-linear-attention", "triton"]
    metadata = {
        "model": model_id, "model_revision": revision,
        "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "dataset_config": DATASET_CONFIG, "dataset_split": "test",
        "layer": args.layer, "kv_head": None if gdn else args.kv_head,
        "head_indices": heads, "head_type": "recurrent" if gdn else "kv", "tokens": args.tokens,
        "spectrum_shape": list(expected_shape), "attention_scale": scale,
        "model_dtype": "float32", "analysis_dtype": "float64",
        "attention_implementation": "fused_recurrent" if gdn else "eager",
        "fla_revision": FLA_REVISION if gdn else None,
        "spectrum_convention": "lambda_i(B_t) = exp(log_scale) * scaled_eigenvalues; descending mode axis last",
        "normalization": ("Native recurrent state before output norm/gate/projection; convolution buffers excluded from spectrum"
                          if gdn else "Unnormalized softmax numerator state; no output projection"),
        "state_layout": "value,key" if gdn else "implicit feature,value",
        "threshold_rank_singular_tolerance": RANK_TOL,
        "threshold_rank_comparison": "strictly greater than tolerance * largest singular value",
        "psd_relative_tolerance": PSD_RTOL, "direct_relative_tolerance": DIRECT_RTOL,
        "cache_relative_tolerance": CACHE_RTOL, "checked_prefixes": check_prefixes,
        "validation_metric": diagnostic, "max_validation_relative_error": float(np.nanmax(results[diagnostic])),
        "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "package_versions": {package: importlib.metadata.version(package) for package in packages},
        "unused_checkpoint_keys": loading_info["unexpected_keys"],
        "model_config": config.to_dict(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not gdn:
        metadata["max_direct_relative_error"] = metadata["max_validation_relative_error"]
    plot_results(results, [item["title"] for item in inputs], args.output, layer=args.layer)
    metadata["elapsed_seconds"] = time.monotonic() - started
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved spectra {expected_shape} and figures to {args.output.resolve()} in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
