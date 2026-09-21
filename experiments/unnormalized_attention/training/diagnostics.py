"""Fixed-sample attention checks, separate from held-out loss evaluation."""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from experiments.unnormalized_attention.data.loaders import make_eval_iterator
from experiments.unnormalized_attention.data.packed import manifest_fingerprint
from experiments.unnormalized_attention.models.components.attention import QUERY_BLOCK_SIZE


def attention_identity(config) -> dict:
    normalized = config.model.attention_normalization
    identity = {
        "attention_normalization": normalized,
        "attention_kernel": "softmax" if normalized else "exp_unnormalized",
        "attention_formula": (
            "softmax_causal(QK^T / sqrt(d_k)) V" if normalized
            else "causal_exp(QK^T / sqrt(d_k)) V"
        ),
        "attention_backend": config.model.attention_backend,
        "query_block_size": QUERY_BLOCK_SIZE,
        "attention_score_scale": "1/sqrt(d_k)",
        "attention_output_normalization": False,
    }
    if config.model.attention_backend == "fused":
        from experiments.unnormalized_attention.models.components.fused_attention import _tile_sizes
        block_q, block_k = ((None, None) if normalized else
                           _tile_sizes(config.model.qk_head_dim, config.model.value_head_dim))
        identity.update({
            "attention_implementation": "cudnn" if normalized else "pallas_triton",
            "query_block_size": block_q,
            "key_block_size": block_k,
            "diagnostic_backend": "blocked",
            "diagnostic_query_block_size": QUERY_BLOCK_SIZE,
        })
    return identity


def fixed_diagnostic_sample(config, *, validated_manifest=None):
    """Read only the first held-out record, with an independent iterator."""
    sample = next(make_eval_iterator(
        replace(config.data, batch_size=1), validated_manifest=validated_manifest,
    ))
    sample = np.asarray(sample, dtype=np.int32)
    provenance = {
        "selection": "first_heldout_sequence",
        "source": config.data.source,
        "sequence_index": 0,
        "num_sequences": 1,
        "prediction_tokens": config.data.sequence_length - 1,
        "record_sha256": hashlib.sha256(sample.astype("<i4").tobytes()).hexdigest(),
        "diagnostic_pass_contributes_to_validation_loss": False,
    }
    if config.data.source == "packed":
        provenance.update({
            "eval_manifest": config.data.eval_manifest,
            "eval_manifest_sha256": manifest_fingerprint(config.data.eval_manifest),
        })
    else:
        provenance["synthetic_shuffle_seed"] = config.data.shuffle_seed + 1
    return sample, provenance


@nnx.jit
def _diagnostic_step(graphdef, params, inputs):
    model = nnx.merge(graphdef, params)
    _, diagnostics = model(inputs, deterministic=True, return_diagnostics=True)
    return diagnostics


def json_number(value):
    """Keep failure reports valid JSON even when device results are nonfinite."""
    number = float(value)
    if math.isfinite(number):
        return number
    if math.isnan(number):
        return "NaN"
    return "Infinity" if number > 0 else "-Infinity"


def attention_diagnostics(state, config, sample, provenance) -> dict:
    # Training parameters may have replicated NamedSharding. Explicitly place
    # this one-sequence check on one device to avoid ambiguous output sharding.
    device = jax.local_devices()[0]
    params = jax.tree.map(lambda value: jax.device_put(value, device), state.params)
    inputs = jax.device_put(jnp.asarray(sample[:, :-1], dtype=jnp.int32), device)
    raw_layers = jax.device_get(_diagnostic_step(state.graphdef, params, inputs))
    layers, issues = [], []
    for index, raw in enumerate(raw_layers):
        layer = {"layer": index}
        for name, value in raw.items():
            number = float(value)
            layer[name] = int(number) if name.endswith("_count") and math.isfinite(number) else json_number(number)
            # In the unnormalized operator the sum is measured only; it is
            # never used as a divisor. It may overflow even if the weighted
            # value output remains finite, so preserve it without failing.
            diagnostic_only_sum = (not config.model.attention_normalization
                                   and name in ("weight_sum_min", "weight_sum_max"))
            if not math.isfinite(number) and not diagnostic_only_sum:
                issues.append(f"layer {index}: nonfinite {name}")
            elif name.startswith("invalid_") and number != 0:
                issues.append(f"layer {index}: {name}={int(number)}")
        for name in ("kernel_min", "weight_sum_min"):
            value = float(raw[name])
            # Exponentials can underflow to zero. Only the normalized
            # control requires a strictly positive weight sum for division.
            invalid = (value <= 0 if name == "weight_sum_min" and config.model.attention_normalization
                       else value < 0)
            if math.isfinite(value) and invalid:
                issues.append(f"layer {index}: invalid {name}={value}")
        layers.append(layer)
    if len(layers) != config.model.num_layers:
        issues.append(f"Expected {config.model.num_layers} layers, got {len(layers)}")
    return {
        "format": "unnormalized-attention-diagnostics",
        "version": 1,
        "step": int(state.step),
        **attention_identity(config),
        "sample": provenance,
        "kernel_values": "row_max_shifted_exp" if config.model.attention_normalization else "unshifted_exp",
        "layers": layers,
        "valid": not issues,
        "issues": issues,
    }
