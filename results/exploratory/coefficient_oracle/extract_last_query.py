"""Extract the last fitted query's polynomial and score distribution, read-only.

Run within a one-GPU allocation. The compact JSON is the source for portable
figure regeneration; full score samples stay in an ignored archive beside it.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
import yaml

from scaffolds.records import validate_record
from scaffolds.transformer.configs import Config
from scaffolds.transformer.configs.base import validate_config
from scaffolds.transformer.configs.loader import _update_dataclass
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.models import build_model
import scaffolds.transformer.attention.multihead as attention_module


ROOT = Path(__file__).resolve().parents[3]
QUANTILES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    config = Config()
    if raw["model"].get("attention_mechanism") == "dot_product":
        names = {field.name for field in fields(type(config.model))}
        if "gdn_polynomial_degree" not in names:
            raw["model"].pop("gdn_polynomial_degree", None)
    _update_dataclass(config, raw, context=str(path))
    validate_config(config)
    return config


def restore_parameters(config: Config, checkpoint: Path):
    model = build_model(config.model, nnx.Rngs(params=config.train.seed, dropout=config.train.seed))
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    graph, template = nnx.split(model, nnx.Param)
    item = {"state": {"params": template}}
    directory = checkpoint / "default" if (checkpoint / "default").is_dir() else checkpoint
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(directory, args=ocp.args.PyTreeRestore(
            item=item, partial_restore=True,
            restore_args=ocp.checkpoint_utils.construct_restore_args(item),
        ))
    return graph, restored["state"]["params"]


def instrumented_forward(graph, params, tokens, override=None):
    """Capture operands passed to the unchanged attention core at tracing time."""
    rows = []
    original = attention_module.attention_core

    def record_scores(q, k, v, *args, **kwargs):
        if q.shape[2] != k.shape[2]:
            raise ValueError("This diagnostic requires one key head per query head")
        scores = jnp.einsum(
            "bhd,bshd->bhs", q[:, -1].astype(jnp.float32), k.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ) / math.sqrt(q.shape[-1])
        rows.append(scores)
        return original(q, k, v, *args, **kwargs)

    @jax.jit
    def forward(frozen, inputs, coefficients):
        model = nnx.merge(graph, frozen)
        logits, effective = model(
            inputs[:, :-1], deterministic=True, query_coefficients=coefficients,
            return_coefficients=True,
        )
        per_token = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), inputs[:, 1:],
        )
        return per_token.mean(), per_token[0, -1], effective[:, 0, -1], jnp.stack(rows)[:, 0]

    with patch.object(attention_module, "attention_core", record_scores):
        result = forward(params, tokens, override)
        result = jax.device_get(result)
    if len(rows) != result[2].shape[0]:
        raise RuntimeError("The forward did not expose every attention layer")
    return result


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def cosine(left, right) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def compact_heads(before, after, scores_before, scores_after):
    heads = []
    for layer in range(after.shape[0]):
        for head in range(after.shape[1]):
            s = scores_after[layer, head].astype(np.float64)
            p0 = np.polynomial.polynomial.polyval(s, before[layer, head].astype(np.float64))
            p1 = np.polynomial.polynomial.polyval(s, after[layer, head].astype(np.float64))
            exponential = np.exp(s)
            if not all(np.isfinite(value).all() for value in (p0, p1, exponential)):
                raise RuntimeError("Nonfinite polynomial or exponential on observed scores")
            heads.append({
                "layer": layer + 1, "head": head + 1,
                "score_quantiles_before": np.quantile(scores_before[layer, head], QUANTILES).tolist(),
                "score_quantiles_after": np.quantile(s, QUANTILES).tolist(),
                "rms_on_fitted_scores": {"before": rms(p0), "after": rms(p1), "exp": rms(exponential)},
                "negative_fraction_before_on_fitted_scores": float(np.mean(p0 < 0)),
                "negative_fraction_after_on_fitted_scores": float(np.mean(p1 < 0)),
                "cosine_with_exp_before_on_fitted_scores": cosine(p0, exponential),
                "cosine_with_exp_after_on_fitted_scores": cosine(p1, exponential),
                "cosine_before_after_on_fitted_scores": cosine(p0, p1),
                "minimum_before_on_fitted_scores": float(p0.min()),
                "maximum_before_on_fitted_scores": float(p0.max()),
                "minimum_after_on_fitted_scores": float(p1.min()),
                "maximum_after_on_fitted_scores": float(p1.max()),
            })
    return heads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    output = run / "last_query_polynomials.json"
    archive = run / "last_query_polynomials.npz"
    if not args.overwrite and (output.exists() or archive.exists()):
        raise FileExistsError("Diagnostic already exists; pass --overwrite to replace it explicitly")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Run this extraction in an allocation with one visible GPU")
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    oracle = recipe["oracle"]
    if int(oracle["sequence_count"]) != 1:
        raise ValueError("This extraction selects the single-sequence fit")
    record = json.loads((run / "record.json").read_text())
    complete = json.loads((run / "completion.json").read_text())
    if record["status"] != "complete":
        raise ValueError("The selected fit is not complete")
    selected = record["selected_checkpoint"]
    checkpoint = Path(selected["path"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    identity = json.loads((checkpoint / "coefficient_identity.json").read_text())
    step = int(identity["step"])
    if (step != int(checkpoint.name) or step != int(complete["best_step"])
            or ("step" in selected and int(selected["step"]) != step)):
        raise ValueError("Selected checkpoint is not the recorded best fitting checkpoint")
    if identity["coordinates"] != "effective_monomial_c":
        raise ValueError("Unexpected coefficient checkpoint coordinates or step")
    validate_record(run, checkpoint_hashes=False)
    parent_config = Path(oracle["parent_config"])
    if sha256(parent_config) != oracle["parent_config_sha256"]:
        raise ValueError("Parent configuration checksum changed")
    parent_record = json.loads((parent_config.parent / "record.json").read_text())
    if parent_record["selected_checkpoint"]["sha256"] != oracle["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint identity changed")
    validate_record(parent_config.parent, checkpoint_hashes=False)
    config = read_config(parent_config)
    if sha256(Path(config.data.eval_manifest)) != oracle["eval_manifest_sha256"]:
        raise ValueError("Dataset manifest identity changed")
    batch = next(iter(PackedBatchIterator(
        config.data.eval_manifest, batch_size=1, sequence_length=config.data.sequence_length,
        start_sequence=int(oracle["data_start"]),
    )))
    batch_hash = hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest()
    data_identity = json.loads((run / "data_identity.json").read_text())
    if batch_hash != data_identity["batch_sha256"]:
        raise ValueError("The reconstructed input differs from the fitted sequence")
    print("Restoring frozen backbone and retained fitted coefficients", flush=True)
    graph, params = restore_parameters(config, Path(oracle["parent_checkpoint"]))
    shape = (1, config.model.num_layers, batch.shape[1] - 1,
             config.model.num_heads, config.model.attention_degree + 1)
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(checkpoint / "best", target={
            "coefficients": [jnp.zeros(shape, dtype=jnp.float32)],
        })
    table = restored["coefficients"][0]
    if table.shape != shape:
        raise ValueError("Unexpected retained coefficient shape")
    tokens = jax.device_put(batch)
    print("Capturing the original and fitted last-query kernels and observed scores", flush=True)
    original = instrumented_forward(graph, params, tokens)
    fitted = instrumented_forward(graph, params, tokens, jnp.swapaxes(table, 0, 1))
    parent_nll, parent_last, before, scores_before = original
    fitted_nll, fitted_last, after, scores_after = fitted
    if before.shape != (config.model.num_layers, config.model.num_heads, config.model.attention_degree + 1):
        raise ValueError("Missing layer/head coefficients")
    if scores_after.shape != (config.model.num_layers, config.model.num_heads, batch.shape[1] - 1):
        raise ValueError("Missing layer/head score samples")
    np.testing.assert_array_equal(after, np.asarray(table)[0, :, -1])
    tolerance = 0.002 if config.train.dtype == "bfloat16" else 1e-5
    if abs(float(parent_nll) - complete["initial_nll"]) > tolerance:
        raise RuntimeError(f"Original NLL differs from fitting initialization: {parent_nll}")
    if abs(float(fitted_nll) - complete["best_fit_nll"]) > tolerance:
        raise RuntimeError(f"Fitted NLL differs from selected checkpoint: {fitted_nll}")
    heads = compact_heads(before, after, scores_before, scores_after)
    np.savez_compressed(archive, scores_before=scores_before, scores_after=scores_after)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "schema_version": 1, "run_id": run.name, "parent_run_id": oracle["parent_run_id"],
        "selected_checkpoint_step": step, "query_index": batch.shape[1] - 2,
        "query_position": batch.shape[1] - 1, "input_tokens": batch.shape[1] - 1,
        "target_index": batch.shape[1] - 1, "input_token_id": int(batch[0, -2]),
        "target_token_id": int(batch[0, -1]), "polynomial_degree": config.model.attention_degree,
        "num_layers": config.model.num_layers, "num_heads": config.model.num_heads,
        "polynomial_coordinates": "effective monomial coefficients; p(s) = sum_r c_r s**r",
        "score_definition": "post-QK-norm, post-RoPE, kernel-input q dot k / sqrt(d_K); FP32 HIGHEST dot product",
        "attention_normalization": config.model.attention_normalization,
        "attention_output_scaling": config.model.attention_output_scaling,
        "parent_nll": float(parent_nll), "fitted_nll": float(fitted_nll),
        "parent_last_token_nll": float(parent_last), "fitted_last_token_nll": float(fitted_last),
        "reference_initial_nll": complete["initial_nll"],
        "reference_best_fit_nll": complete["best_fit_nll"], "nll_acceptance_atol": tolerance,
        "coefficients": {"before": before.tolist(), "after": after.tolist()},
        "quantile_levels": QUANTILES, "heads": heads,
        "provenance": {
            "source_commit": source_commit, "extraction_code_sha256": sha256(Path(__file__)),
            "dependency_lock_sha256": sha256(ROOT / "scaffolds/transformer/uv.lock"),
            "run_config_sha256": sha256(run / "config.yaml"),
            "parent_config_sha256": sha256(parent_config),
            "parent_checkpoint_sha256": oracle["parent_checkpoint_sha256"],
            "fitted_checkpoint_sha256": selected["sha256"],
            "batch_sha256": batch_hash, "eval_manifest_sha256": oracle["eval_manifest_sha256"],
            "raw_scores_archive": {"path": archive.name, "sha256": sha256(archive)},
            "device": jax.devices()[0].device_kind, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "node": os.uname().nodename,
        },
        "interpretation": "A fixed-input fitted kernel for one query, not a claim of general superiority over exp. Polynomial values use FP64 Horner evaluation of the stored FP32 coefficients. Score ranges and empirical RMS factors are measured separately from the original and fitted forwards. Instrumentation may change BF16 compilation rounding within the recorded NLL tolerance.",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "parent_nll": float(parent_nll),
                      "fitted_nll": float(fitted_nll), "layers": config.model.num_layers,
                      "heads": config.model.num_heads, "query_index": result["query_index"]}), flush=True)


if __name__ == "__main__":
    main()
