"""Read-only probe of degree-2 Hermite attention before the first update.

The first training batch is deterministic. Causal position zero depends only on
its own token, so running the batch's first tokens through the initialized model
reproduces those rows at every layer without allocating a full sequence. A
longer prefix can be requested to inspect later rows; the default probes only
position zero and cannot rule out a failure elsewhere in the sequence.

This script writes analysis evidence under ignored logs and never changes a run
record, checkpoint, recipe, or the frozen execution set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scaffolds.transformer.attention.backends.fused_normalized_polynomial import _forward
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.acceptance import initialization_facts
from scaffolds.transformer.configs.loader import load_config
from scaffolds.transformer.data.loaders import make_train_iterator
from scaffolds.transformer.data.packed import load_packed_manifest
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.state import create_state


def _host(value):
    return np.asarray(jax.device_get(value))


def _finite_float(value):
    number = float(value)
    return number if math.isfinite(number) else None


def _first_parameter_mismatch(actual, expected_model, rng):
    """Explain a same-device hash mismatch without making that mismatch valid."""
    expected = nnx.state(build_model(
        expected_model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0))
    ), nnx.Param)
    left = jax.tree_util.tree_flatten_with_path(actual)[0]
    right = jax.tree_util.tree_flatten_with_path(expected)[0]
    if len(left) != len(right):
        return {"reason": "parameter_leaf_count", "fused": len(left), "blocked": len(right)}
    for (left_path, left_value), (right_path, right_value) in zip(left, right, strict=True):
        if str(left_path) != str(right_path):
            return {"reason": "parameter_path", "fused": str(left_path), "blocked": str(right_path)}
        left_array, right_array = _host(left_value), _host(right_value)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return {
                "reason": "parameter_shape_or_dtype", "path": str(left_path),
                "fused_shape": list(left_array.shape), "blocked_shape": list(right_array.shape),
                "fused_dtype": str(left_array.dtype), "blocked_dtype": str(right_array.dtype),
            }
        unequal = np.argwhere(left_array != right_array)
        if len(unequal):
            index = tuple(map(int, unequal[0]))
            return {
                "reason": "parameter_value", "path": str(left_path),
                "index": list(index),
                "fused": _finite_float(left_array[index]),
                "blocked": _finite_float(right_array[index]),
            }
    return {"reason": "hash_differs_despite_equal_parameter_arrays"}


def _project_qkv(layer, x, positions):
    attention = layer.attention
    h = layer.attention_norm(x)
    shapes = (
        (attention.num_heads, attention.qk_head_dim),
        (attention.num_kv_heads, attention.qk_head_dim),
        (attention.num_kv_heads, attention.value_head_dim),
    )
    projected = []
    for projection, (heads, width) in zip(
        (attention.q_proj, attention.k_proj, attention.v_proj), shapes, strict=True
    ):
        value = projection(h).astype(attention.dtype)
        projected.append(value.reshape(x.shape[0], x.shape[1], heads, width))
    q, k, v = projected
    q = apply_rope(q, positions, attention.rope_theta).astype(attention.dtype)
    k = apply_rope(k, positions, attention.rope_theta).astype(attention.dtype)
    return q, k, v


def _score_summary(q, k, coefficients):
    # For short prefixes this independently locates scores near the double root.
    # Fused normalizers below are the decisive implementation measurement.
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32
    ) * (1.0 / math.sqrt(q.shape[-1]))
    c0, c1, c2 = coefficients
    horner = (c2 * scores + c1) * scores + c0
    factored = (jnp.float32(math.exp(0.5) / 2.0) * (scores + 1.0)) * (scores + 1.0)
    causal = np.tri(scores.shape[-2], scores.shape[-1], dtype=bool)[None, None]
    scores_h, horner_h, factored_h = map(_host, (scores, horner, factored))
    distance = np.where(causal, np.abs(scores_h + 1.0), np.inf)
    nearest = np.unravel_index(int(np.argmin(distance)), distance.shape)
    values = horner_h[np.broadcast_to(causal, horner_h.shape)]
    factor_values = factored_h[np.broadcast_to(causal, factored_h.shape)]
    return {
        "nearest_root": {
            "batch_in_chunk": int(nearest[0]), "head": int(nearest[1]),
            "query": int(nearest[2]), "key": int(nearest[3]),
            "score": _finite_float(scores_h[nearest]),
            "distance_to_minus_one": _finite_float(distance[nearest]),
            "horner": _finite_float(horner_h[nearest]),
            "factored_square": _finite_float(factored_h[nearest]),
        },
        "horner_zero_count": int(np.count_nonzero(values == 0)),
        "horner_negative_count": int(np.count_nonzero(values < 0)),
        "factored_zero_count": int(np.count_nonzero(factor_values == 0)),
        "minimum_absolute_horner": _finite_float(np.min(np.abs(values))),
        "minimum_absolute_factored": _finite_float(np.min(np.abs(factor_values))),
    }


def _skipped_score_summary():
    # Retain the short-prefix report's fields without allocating B*H*T*T
    # scores for a full training sequence.
    return {
        "nearest_root": None,
        "horner_zero_count": None,
        "horner_negative_count": None,
        "factored_zero_count": None,
        "minimum_absolute_horner": None,
        "minimum_absolute_factored": None,
        "score_summary_scope": "skipped_full_matrix",
    }


def _first_invalid_row_scores(q, k, coefficients, batch_index, query, head):
    """Inspect at most one causal row with an independent JAX dot/Horner.

    This is an O(sequence length) reference calculation. Its reduction order
    can differ from the fused Pallas tile kernel, so it cannot alone establish
    which fused arithmetic condition poisoned a row.
    """
    scores = jnp.einsum(
        "d,kd->k", q[batch_index, query, head], k[batch_index, :query + 1, head],
        preferred_element_type=jnp.float32,
    ) * (1.0 / math.sqrt(q.shape[-1]))
    c0, c1, c2 = (jnp.float32(value) for value in coefficients)
    horner = (c2 * scores + c1) * scores + c0
    factored = (jnp.float32(math.exp(0.5) / 2.0) * (scores + 1.0)) * (scores + 1.0)
    scores_h, horner_h, factored_h = map(_host, (scores, horner, factored))
    nearest = int(np.argmin(np.abs(scores_h + 1.0)))
    return {
        "scope": "one_invalid_causal_row_independent_jax_reference",
        "query": query,
        "head": head,
        "key_count": query + 1,
        "nearest_root_key": nearest,
        "nearest_root_score": _finite_float(scores_h[nearest]),
        "nearest_root_distance": _finite_float(abs(scores_h[nearest] + 1.0)),
        "nearest_root_horner": _finite_float(horner_h[nearest]),
        "nearest_root_factored_square": _finite_float(factored_h[nearest]),
        "horner_zero_count": int(np.count_nonzero(horner_h == 0)),
        "horner_negative_count": int(np.count_nonzero(horner_h < 0)),
        "horner_nonfinite_count": int(np.count_nonzero(~np.isfinite(horner_h))),
        "horner_sum_fp32": _finite_float(np.sum(horner_h, dtype=np.float32)),
        "factored_sum_fp32": _finite_float(np.sum(factored_h, dtype=np.float32)),
        "note": "JAX dot and reduction may differ from the fused Pallas tile arithmetic.",
    }


def _probe_chunk(model, tokens, coefficients):
    x = model.token_embedding(jnp.asarray(tokens, dtype=jnp.int32))
    x = model.dropout(x, deterministic=True)
    positions = jnp.arange(tokens.shape[1])
    parameters = jnp.broadcast_to(
        jnp.asarray(coefficients, dtype=jnp.float32),
        (model.config.num_heads, len(coefficients)),
    )
    layers = []
    for layer_index, layer_name in enumerate(model.layer_names):
        layer = getattr(model, layer_name)
        q, k, v = _project_qkv(layer, x, positions)
        score_report = (
            _score_summary(q, k, coefficients) if tokens.shape[1] <= 128
            else _skipped_score_summary()
        )
        output, normalizer = _forward(
            q, k, v, parameters, parameterization="direct", interpret=False
        )
        norms = _host(normalizer)
        finite = np.isfinite(norms)
        # Transfer only an output count and one bit per row, not the full
        # B*T*H*D output tensor. The fused kernel poisons both output and
        # normalizer when it detects an invalid row.
        output_nonfinite = ~jnp.isfinite(output)
        output_nonfinite_count, output_nonfinite_rows = jax.device_get((
            jnp.count_nonzero(output_nonfinite), jnp.any(output_nonfinite, axis=-1),
        ))
        invalid_indices = np.argwhere(~finite | np.asarray(output_nonfinite_rows))
        report = {
            "layer": layer_index,
            **score_report,
            "fused_row_count": int(norms.size),
            "fused_nonfinite_normalizer_count": int(np.count_nonzero(~finite)),
            "fused_zero_normalizer_count": int(np.count_nonzero(norms == 0)),
            "fused_negative_normalizer_count": int(np.count_nonzero(norms < 0)),
            "fused_minimum_absolute_finite_normalizer": (
                _finite_float(np.min(np.abs(norms[finite]))) if np.any(finite) else None
            ),
            "fused_nonfinite_output_count": int(output_nonfinite_count),
        }
        if len(invalid_indices):
            batch_index, query, head = map(int, invalid_indices[0])
            report["first_invalid_row"] = {
                "batch_in_chunk": batch_index, "query": query, "head": head,
            }
            report["first_invalid_row_scores"] = _first_invalid_row_scores(
                q, k, coefficients, batch_index, query, head
            )
        layers.append(report)
        if len(invalid_indices) or report["fused_nonfinite_output_count"]:
            break
        h = layer.attention.o_proj(
            output.astype(v.dtype).reshape(x.shape[0], x.shape[1], -1)
        )
        x = x + h
        x = x + layer.feed_forward(layer.ffn_norm(x))
    return layers


def probe(config_path: Path, *, batch_limit: int, prefix_length: int, chunk_size: int):
    if jax.default_backend() != "gpu":
        raise RuntimeError("This probe requires a GPU allocation and the fused GPU backend")
    config = load_config(config_path)
    model_cfg = config.model
    if not (
        model_cfg.attention_degree == 2
        and model_cfg.attention_normalization
        and model_cfg.attention_backend == "fused"
        and model_cfg.attention_fixed_coefficients is not None
    ):
        raise ValueError("Probe requires the normalized, fixed degree-2 fused recipe")
    if not 1 <= batch_limit <= config.data.batch_size:
        raise ValueError("batch-limit must be between 1 and the global batch size")
    if not 1 <= prefix_length <= model_cfg.max_seq_len:
        raise ValueError(f"prefix-length must be between 1 and {model_cfg.max_seq_len}")
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")

    validated_manifest = load_packed_manifest(config.data.train_manifest)
    first_batch = next(make_train_iterator(
        replace(config.data, batch_size=config.data.batch_size),
        validated_manifest=validated_manifest,
    ))
    micro_batch_size = config.train.micro_batch_size
    if config.data.batch_size % micro_batch_size:
        raise ValueError("global batch must be divisible by the configured microbatch size")
    micro_iterator = make_train_iterator(
        replace(config.data, batch_size=micro_batch_size), validated_manifest=validated_manifest,
    )
    micro_batches = np.concatenate(
        [next(micro_iterator) for _ in range(config.data.batch_size // micro_batch_size)], axis=0
    )
    if not np.array_equal(first_batch, micro_batches):
        raise RuntimeError("First global batch differs from trainer microbatch sequence order")
    rng = jax.random.PRNGKey(config.train.seed)
    state = create_state(config, rng)
    model = nnx.merge(state.graphdef, state.params)
    coefficients = tuple(float(c) for c in model_cfg.attention_fixed_coefficients)
    gpu_fused_hash = state_sha256(state.params)
    blocked_config = copy.deepcopy(config)
    gpu_blocked_facts = initialization_facts(blocked_config)
    gpu_blocked_hash = gpu_blocked_facts["parameter_sha256"]
    acceptance_files = sorted(
        (config_path.resolve().parents[1] / "logs" / config_path.stem / "acceptance")
        .glob("*/initialization-result.json")
    )
    acceptance_hashes = [
        json.loads(path.read_text())["parameter_sha256"] for path in acceptance_files
    ]
    if not acceptance_hashes:
        raise FileNotFoundError("No acceptance initialization hash found for this run")
    report = {
        "format": "hermite-degree-2-initial-normalizer-probe",
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": config.train.seed,
        "backend": model_cfg.attention_backend,
        "device": jax.devices()[0].device_kind,
        "global_batch_size": config.data.batch_size,
        "probed_sequences": batch_limit,
        "prefix_length": prefix_length,
        "first_batch_sha256": hashlib.sha256(first_batch.astype("<i4").tobytes()).hexdigest(),
        "trainer_microbatch_order_matches_global_batch": True,
        "coefficients": list(coefficients),
        "gpu_fused_parameter_sha256": gpu_fused_hash,
        "gpu_blocked_parameter_sha256": gpu_blocked_hash,
        "gpu_blocked_parameter_count": gpu_blocked_facts["parameter_count"],
        "same_gpu_parameter_hash_match": gpu_fused_hash == gpu_blocked_hash,
        "acceptance_initialization_files": [str(path) for path in acceptance_files],
        "cpu_acceptance_parameter_sha256": acceptance_hashes,
        "cpu_gpu_parameter_hash_match": all(value == gpu_fused_hash for value in acceptance_hashes),
        "chunks": [],
        "limitation": (
            "Only the specified causal prefix was probed; finite results cannot rule out "
            "a failure at a later sequence position or during backward/update arithmetic."
        ),
    }
    if gpu_fused_hash != gpu_blocked_hash:
        report["first_parameter_mismatch"] = _first_parameter_mismatch(
            state.params, blocked_config.model, rng
        )
        report["all_probed_rows_finite"] = None
        return report
    for start in range(0, batch_limit, chunk_size):
        stop = min(start + chunk_size, batch_limit)
        layers = _probe_chunk(model, first_batch[start:stop, :prefix_length], coefficients)
        report["chunks"].append({"sequence_start": start, "sequence_stop": stop, "layers": layers})
        if (len(layers) != model_cfg.num_layers
                or layers[-1]["fused_nonfinite_normalizer_count"]
                or layers[-1]["fused_nonfinite_output_count"]):
            report["first_invalid_chunk"] = [start, stop]
            break
    report["all_probed_rows_finite"] = "first_invalid_chunk" not in report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-limit", type=int, default=256)
    parser.add_argument("--prefix-length", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    result = probe(args.config, batch_limit=args.batch_limit,
                   prefix_length=args.prefix_length, chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["same_gpu_parameter_hash_match"]:
        raise RuntimeError(
            "Same-GPU blocked and fused initialization differ: "
            + json.dumps({key: result[key] for key in (
                "gpu_fused_parameter_sha256", "gpu_blocked_parameter_sha256",
                "first_parameter_mismatch")}, sort_keys=True)
        )
    print(json.dumps({
        "output": str(args.output),
        "all_probed_rows_finite": result["all_probed_rows_finite"],
        "first_invalid_chunk": result.get("first_invalid_chunk"),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
