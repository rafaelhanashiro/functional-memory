"""Fixed-sample attention checks, separate from held-out loss evaluation."""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scaffolds.transformer.data.loaders import make_eval_iterator
from scaffolds.transformer.data.packed import manifest_fingerprint
from scaffolds.transformer.models.components.attention import QUERY_BLOCK_SIZE
from scaffolds.transformer.training.numerics import is_odd_taylor


M0_PARAMETER_NOTE = (
    "Degree zero uses uniform causal-prefix averaging. Q/K parameters remain in the "
    "model and parameter count; their task-loss gradients are zero, but AdamW weight "
    "decay still updates them."
)
M0_UNNORMALIZED_PARAMETER_NOTE = M0_PARAMETER_NOTE.replace("averaging", "sums")
# Row sums of unnormalized kernels, and ratios formed from them, are measurements
# only. They may be zero or infinite while the attention output stays finite.
UNNORMALIZED_MEASUREMENTS = frozenset((
    "weight_sum_min", "weight_sum_max", "min_abs_weight_sum",
    "cancellation_ratio_max", "cancellation_ratio_mean",
))


def attention_identity(config) -> dict:
    if config.model.attention_mechanism == "gla":
        model = config.model
        identity = {
            "attention_mechanism": "gla",
            "attention_kernel": "gla",
            "attention_backend": model.attention_backend,
            "attention_backend_requested": getattr(config, "_attention_backend_request", model.attention_backend),
            "attention_implementation": ("chunked_gla_triton" if model.attention_backend == "fused"
                                         else "chunkwise_associative_scan"),
            "attention_formula": "S_t=diag(sigmoid(g_t)^(1/tau)) S_{t-1}+k_t v_t^T; y_t=q_t^T S_t",
            "attention_score_scale": "1",
            "attention_output_normalization": "per_head_rmsnorm_then_silu_output_gate",
            "gla_gate_rank": model.gla_gate_rank,
            "gla_gate_tau": model.gla_gate_tau,
            "gla_chunk_size": model.gla_chunk_size,
            "gla_inner_chunk_size": model.gla_inner_chunk_size,
            "gla_use_rope": model.gla_use_rope,
            "gla_state_elements_per_layer": model.num_heads * model.qk_head_dim * model.value_head_dim,
            "gla_initial_state": "zero_each_sequence",
            "gla_state_dtype": "float32",
            "gla_kernel_dot_precision": "ieee" if model.attention_backend == "fused" else None,
            "diagnostic_backend": model.attention_backend,
        }
        if hasattr(config, "_attention_backend_resume_request"):
            identity["attention_backend_resume_requested"] = config._attention_backend_resume_request
        return identity
    degree = config.model.attention_degree
    mode = config.model.quadratic_mode
    normalized = config.model.attention_normalization
    identity = {
        "attention_degree": degree,
        "attention_normalization": normalized,
        "attention_kernel": (("exp_unnormalized" if degree is None else "taylor_unnormalized") if not normalized else
                             "learned_quadratic" if mode != "fixed" else "softmax" if degree is None else "taylor"),
        "attention_backend": config.model.attention_backend,
        "attention_backend_requested": getattr(config, "_attention_backend_request", config.model.attention_backend),
        "attention_implementation": config.model.attention_backend,
        "query_block_size": QUERY_BLOCK_SIZE,
        "m0_qk_parameter_note": (None if degree != 0 else M0_PARAMETER_NOTE if normalized
                                 else M0_UNNORMALIZED_PARAMETER_NOTE),
    }
    if not normalized:
        formula = ("causal_exp(QK^T / sqrt(d_k)) V" if degree is None else
                   f"causal_p_{degree}(QK^T / sqrt(d_k)) V, p_m(x) = sum_(n<=m) x^n / n!")
        identity.update(attention_formula=formula,
                        attention_score_scale="1/sqrt(d_k)", attention_output_normalization=False)
    if hasattr(config, "_attention_backend_resume_request"):
        identity["attention_backend_resume_requested"] = config._attention_backend_resume_request
    if is_odd_taylor(config.model) or mode == "unrestricted":
        identity["attention_signed"] = True
    if mode != "fixed":
        identity.update({
            "quadratic_mode": mode,
            "quadratic_sharing": config.model.quadratic_sharing,
            "quadratic_parameter_count": (2 if config.model.quadratic_sharing == "global" else
                                          2 * config.model.num_layers * config.model.num_heads),
        })
    if config.model.attention_backend == "fused":
        if not normalized:
            from scaffolds.transformer.attention.backends.fused_unnormalized import _tile_sizes
        else:
            from scaffolds.transformer.models.components.fused_attention import _tile_sizes
        block_q, block_k = ((None, None) if degree is None and normalized else
                           _tile_sizes(config.model.qk_head_dim, config.model.value_head_dim))
        identity.update({
            "attention_implementation": "cudnn" if degree is None and normalized else "pallas_triton",
            "query_block_size": block_q,
            "key_block_size": block_k,
            "diagnostic_backend": "blocked",
            "diagnostic_query_block_size": QUERY_BLOCK_SIZE,
        })
    elif config.model.attention_backend in ("manual", "xla", "cudnn"):
        identity.update({
            "query_block_size": None,
            "diagnostic_backend": "blocked",
            "diagnostic_query_block_size": QUERY_BLOCK_SIZE,
        })
    return identity


def quadratic_coefficient_report(params, config) -> dict:
    """Record both optimizer parameters and the polynomial they currently define."""
    raw = np.asarray(jax.device_get(params["quadratic_coefficients"][...]), dtype=np.float64)
    first, second = raw[..., 0], raw[..., 1]
    if config.model.quadratic_mode == "positive":
        effective = np.stack((0.5 * (1.0 + second ** 2), first * second, 0.5 * first ** 2), axis=-1)
        names = ["a", "b"]
    else:
        effective = np.stack((np.ones_like(first), first, second), axis=-1)
        names = ["b", "c"]

    def serializable(array):
        if array.ndim == 0:
            return json_number(array)
        return [serializable(value) for value in array]

    return {
        "raw_parameter_names": names,
        "raw": serializable(raw),
        "effective_coefficient_names": ["constant", "linear", "quadratic"],
        "effective": serializable(effective),
        "finite": bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(effective))),
    }


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


@partial(nnx.jit, static_argnums=(3,))
def _mapped_diagnostic_step(graphdef, params, inputs, device):
    from scaffolds.transformer.training.distributed import single_device_call

    return single_device_call(partial(_diagnostic_step, graphdef), params, inputs, device=device)


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
    raw_layers = jax.device_get(
        _mapped_diagnostic_step(state.graphdef, params, inputs, device)
        if jax.local_device_count() > 1 else _diagnostic_step(state.graphdef, params, inputs)
    )
    layers, issues = [], []
    for index, raw in enumerate(raw_layers):
        layer = {"layer": index}
        for name, value in raw.items():
            number = float(value)
            layer[name] = int(number) if name.endswith("_count") and math.isfinite(number) else json_number(number)
            diagnostic_only_sum = (not config.model.attention_normalization
                                   and name in UNNORMALIZED_MEASUREMENTS)
            if not math.isfinite(number) and not diagnostic_only_sum:
                issues.append(f"layer {index}: nonfinite {name}")
            elif name.startswith("invalid_") and number != 0:
                issues.append(f"layer {index}: {name}={int(number)}")
        sum_key = "normalizer_min" if config.model.attention_normalization else "weight_sum_min"
        positivity_fields = () if config.model.attention_mechanism == "gla" else ("kernel_min", sum_key)
        for name in positivity_fields:
            value = float(raw[name])
            if is_odd_taylor(config.model) or config.model.quadratic_mode == "unrestricted":
                # Signed kernels and negative nonzero sums are permitted. Exact
                # zeros are detected in invalid_normalizer_count, not extrema.
                continue
            # Stable exact softmax may underflow individual weights to zero.
            invalid = (value < 0 if (not config.model.attention_normalization or
                                     name == "kernel_min" and config.model.attention_degree is None)
                       else value <= 0)
            if math.isfinite(value) and invalid:
                issues.append(f"layer {index}: invalid {name}={value}")
        layers.append(layer)
    if len(layers) != config.model.num_layers:
        issues.append(f"Expected {config.model.num_layers} layers, got {len(layers)}")
    coefficient_report = None
    if config.model.quadratic_mode != "fixed":
        coefficient_report = quadratic_coefficient_report(params, config)
        if not coefficient_report["finite"]:
            issues.append("Nonfinite quadratic coefficients")
    report = {
        "format": "shared-transformer-attention-diagnostics",
        "version": 1,
        "step": int(state.step),
        **attention_identity(config),
        "sample": provenance,
        "kernel_values": ("vector_gated_recurrence" if config.model.attention_mechanism == "gla" else
                          ("unshifted_exp" if config.model.attention_degree is None else "unshifted_taylor")
                          if not config.model.attention_normalization else
                          "unshifted_learned_quadratic" if config.model.quadratic_mode != "fixed" else
                          "row_max_shifted_exp" if config.model.attention_degree is None else "unshifted_taylor"),
        "layers": layers,
        "valid": not issues,
        "issues": issues,
    }
    if coefficient_report is not None:
        report["coefficients"] = coefficient_report
    return report
