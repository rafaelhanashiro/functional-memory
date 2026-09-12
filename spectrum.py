"""Inspect the state in Section 3.2 of overleaf/iclr/iclr2027_conference.tex.

Run with: uv run spectrum.py --layer 8 --kv-head 0 --tokens 512 --output results/run
For Slurm, run uv sync --frozen first, then launch .venv/bin/python directly.
The saved eigenvalues describe the unnormalized numerator state, before W_O.
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


def select_inputs(tokenizer, rows, tokens, count=4):
    """Select the first qualifying articles, saving the exact model inputs."""
    if tokens < 2 or count < 1:
        raise ValueError("Require at least two tokens and one input.")
    if tokenizer.bos_token_id is None:
        raise ValueError("The tokenizer must define a BOS token.")
    selected = []
    for article_index, article in enumerate(iter_article_bodies(rows)):
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
        "normalized_eigenvalues": np.zeros((length, width)),
        "participation_ratio": np.zeros(length),
        "energy_rank90": np.zeros(length, dtype=np.int64),
        "energy_rank99": np.zeros(length, dtype=np.int64),
        "log_trace": np.full(length, -np.inf),
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
        energy = eigenvalues.sum()
        if energy > 0:
            normalized = eigenvalues / energy
            result["normalized_eigenvalues"][index] = normalized
            result["participation_ratio"][index] = 1 / np.square(normalized).sum()
            cumulative = normalized.cumsum()
            result["energy_rank90"][index] = min(width, np.searchsorted(cumulative, 0.90) + 1)
            result["energy_rank99"][index] = min(width, np.searchsorted(cumulative, 0.99) + 1)
            result["log_trace"][index] = np.log(energy) + shift
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
    return result


def extract_kv(model, input_ids, layer, kv_head):
    """Read a single head from the model's native post-RoPE KV cache."""
    if not 0 <= layer < model.config.num_hidden_layers:
        raise ValueError(f"Layer must be in [0, {model.config.num_hidden_layers - 1}].")
    if not 0 <= kv_head < model.config.num_key_value_heads:
        raise ValueError(f"KV head must be in [0, {model.config.num_key_value_heads - 1}].")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Extract one nonempty sequence at a time, with shape [1, tokens].")
    if model.training:
        raise ValueError("Call model.eval() before extracting the attention state.")
    device = next(model.parameters()).device
    with torch.inference_mode():
        outputs = model(input_ids=input_ids.to(device), use_cache=True, return_dict=True)
        keys, values = outputs.past_key_values[layer]
        return (
            keys[0, kv_head].to(device="cpu", dtype=torch.float64).numpy().copy(),
            values[0, kv_head].to(device="cpu", dtype=torch.float64).numpy().copy(),
        )


def plot_results(results, titles, output):
    """Write publication-exportable PNG and PDF figures from saved arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output = Path(output)
    count, length, width = results["scaled_eigenvalues"].shape
    positions = np.arange(1, length + 1)
    colors = plt.get_cmap("tab10").colors

    def save(figure, name):
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(count, 1, figsize=(10, 2.2 * count), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
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

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    figure.suptitle(f"Effective state dimensions (head width: {width})")
    for axis, key, label in zip(
        axes, ("participation_ratio", "energy_rank90", "energy_rank99"),
        ("Participation ratio", "Modes for 90% energy", "Modes for 99% energy"),
    ):
        for index, title in enumerate(titles):
            axis.plot(positions, results[key][index], color=colors[index], label=title, linewidth=1.2)
        axis.set_ylabel(label)
        axis.set_ylim(0, min(width + 1, max(2, math.ceil(results[key].max() * 1.15))))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Prefix length t (including BOS)")
    save(figure, "effective_rank")

    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for index, title in enumerate(titles):
        axis.plot(positions, results["log_trace"][index] / np.log(10), label=title, color=colors[index])
    axis.set(xlabel="Prefix length t (including BOS)", ylabel="log₁₀ tr(Bₜ)", title="Total squared state norm")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    save(figure, "state_norm")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Indices are zero-based. Four WikiText-2 test articles are selected deterministically.\n"
                "Outputs: spectra.npz, inputs.json, metadata.json, and PNG/PDF figures.\n"
                "Original eigenvalues = exp(log_scale) * scaled_eigenvalues.\n"
                "CUDA is used when available; CPU remains supported. Output must be a new directory."),
    )
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--kv-head", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=512, help="Total tokens per text including BOS (default: 512)")
    parser.add_argument("--output", type=Path, help="New results directory; defaults to a name based on layer/head/tokens")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"results/llama-3.2-1b-l{args.layer}-kv{args.kv_head}-t{args.tokens}")
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}; choose a new --output.")

    from datasets import load_dataset
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    started = time.monotonic()
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if not 0 <= args.layer < config.num_hidden_layers:
        parser.error(f"--layer must be in [0, {config.num_hidden_layers - 1}]")
    if not 0 <= args.kv_head < config.num_key_value_heads:
        parser.error(f"--kv-head must be in [0, {config.num_key_value_heads - 1}]")
    if args.tokens > config.max_position_embeddings:
        parser.error(f"--tokens exceeds the model context limit ({config.max_position_embeddings})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="test", revision=DATASET_REVISION)
    inputs = select_inputs(tokenizer, (row["text"] for row in dataset), args.tokens)
    args.output.mkdir(parents=True)
    (args.output / "inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Loading {MODEL_ID} in float32 on {device}; layer={args.layer}, KV head={args.kv_head}", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
        attn_implementation="eager", use_safetensors=True,
    ).to(device).eval()
    scale = model.layers[args.layer].self_attn.scaling
    check_prefixes = sorted({t for t in (1, 2, 8, 32, 128, args.tokens) if t <= args.tokens})
    all_results, all_keys, all_values = [], [], []
    for index, item in enumerate(inputs):
        print(f"[{index + 1}/{len(inputs)}] {item['title']}: extracting {args.tokens} tokens", flush=True)
        keys, values = extract_kv(model, torch.tensor([item["token_ids"]]), args.layer, args.kv_head)
        result = compute_spectra(keys, values, check_prefixes=check_prefixes, scale=scale)
        all_results.append(result)
        all_keys.append(keys)
        all_values.append(values)
        print(
            f"  final participation={result['participation_ratio'][-1]:.3f}, "
            f"r90={result['energy_rank90'][-1]}, r99={result['energy_rank99'][-1]}, "
            f"max direct relative error={np.nanmax(result['direct_check_relative_error']):.3g}",
            flush=True,
        )

    results = {key: np.stack([item[key] for item in all_results]) for key in all_results[0]}
    results.update(keys=np.stack(all_keys), values=np.stack(all_values),
                   token_ids=np.array([item["token_ids"] for item in inputs], dtype=np.int64))
    expected_shape = (4, args.tokens, config.hidden_size // config.num_attention_heads)
    if results["scaled_eigenvalues"].shape != expected_shape:
        raise RuntimeError(f"Unexpected spectrum shape: {results['scaled_eigenvalues'].shape}; expected {expected_shape}")
    np.savez_compressed(args.output / "spectra.npz", **results)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
        capture_output=True, text=True, check=False,
    )
    metadata = {
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "dataset_config": DATASET_CONFIG, "dataset_split": "test",
        "layer": args.layer, "kv_head": args.kv_head, "tokens": args.tokens,
        "spectrum_shape": list(expected_shape), "attention_scale": scale,
        "model_dtype": "float32", "analysis_dtype": "float64", "attention_implementation": "eager",
        "spectrum_convention": "lambda_i(B_t) = exp(log_scale[t]) * scaled_eigenvalues[t,i]",
        "normalization": "Unnormalized softmax numerator state; no output projection",
        "psd_relative_tolerance": PSD_RTOL, "direct_relative_tolerance": DIRECT_RTOL,
        "checked_prefixes": check_prefixes,
        "max_direct_relative_error": float(np.nanmax(results["direct_check_relative_error"])),
        "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "package_versions": {package: importlib.metadata.version(package)
                             for package in ("torch", "transformers", "datasets", "numpy", "matplotlib")},
        "model_config": config.to_dict(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    plot_results(results, [item["title"] for item in inputs], args.output)
    metadata["elapsed_seconds"] = time.monotonic() - started
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved spectra {expected_shape} and figures to {args.output.resolve()} in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
