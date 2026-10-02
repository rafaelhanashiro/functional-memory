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
from scaffolds.transformer.attention.backends.reference import QUERY_BLOCK_SIZE
from scaffolds.transformer.training.numerics import (
    has_fixed_polynomial, has_fixed_square_factors, has_learned_polynomial, has_sum_of_squares_kernel,
    is_odd_taylor, is_signed_kernel,
)


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
    if config.model.attention_mechanism == "gated_deltanet":
        model = config.model
        return {
            "attention_mechanism": "gated_deltanet",
            "attention_kernel": "gated_deltanet",
            "attention_backend": model.attention_backend,
            "attention_backend_requested": getattr(config, "_attention_backend_request", model.attention_backend),
            "attention_implementation": ("fla_chunk_gated_delta_rule_triton" if model.attention_backend == "fused"
                                         else "token_recurrence_reference"),
            "attention_formula": ("S_t=a_t S_{t-1}+b_t k_t (v_t-a_t S_{t-1}^T k_t)^T; o_t=S_t^T q_t/sqrt(d_k); "
                                  "q,k L2-normalized after short conv+SiLU; a_t=exp(-exp(A_log) softplus(W_a x+dt_bias)); "
                                  "b_t=sigmoid(W_b x)"),
            "attention_score_scale": "1/sqrt(d_k)",
            "attention_output_normalization": "per_head_rmsnorm_times_silu_output_gate",
            "gdn_conv_size": model.gdn_conv_size,
            "gdn_state_elements_per_layer": model.num_heads * model.qk_head_dim * model.value_head_dim,
            "gdn_initial_state": "zero_each_sequence",
            "gdn_kernel_source": "flash-linear-attention chunk_gated_delta_rule (pinned in the scaffold lock)",
            "gdn_chunk_size": 64 if model.attention_backend == "fused" else None,
            "diagnostic_backend": model.attention_backend,
        }
    degree = config.model.attention_degree
    mode = config.model.quadratic_mode
    normalized = config.model.attention_normalization
    learned_polynomial = has_learned_polynomial(config.model)
    fixed_polynomial = has_fixed_polynomial(config.model)
    fixed_square_factors = has_fixed_square_factors(config.model)
    identity = {
        "attention_degree": degree,
        "attention_normalization": normalized,
        "attention_kernel": (("exp_unnormalized" if degree is None else
                              "learned_polynomial_unnormalized" if learned_polynomial else "taylor_unnormalized")
                             if not normalized else
                             "learned_quadratic" if mode != "fixed" else
                             "fixed_polynomial_sum_of_squares" if fixed_square_factors else
                             "learned_polynomial_sum_of_squares" if has_sum_of_squares_kernel(config.model) else
                             "learned_polynomial" if learned_polynomial else
                             "softmax" if degree is None else "taylor"),
        "attention_backend": config.model.attention_backend,
        "attention_backend_requested": getattr(config, "_attention_backend_request", config.model.attention_backend),
        "attention_implementation": config.model.attention_backend,
        "query_block_size": QUERY_BLOCK_SIZE,
        "m0_qk_parameter_note": (None if degree != 0 else M0_PARAMETER_NOTE if normalized
                                 else M0_UNNORMALIZED_PARAMETER_NOTE),
    }
    if not normalized:
        formula = ("causal_exp(QK^T / sqrt(d_k)) V" if degree is None else
                   f"causal_p_{degree}(QK^T / sqrt(d_k)) V, p(x) = sum_(r<={degree}) c_r x^r, "
                   f"learned c ({config.model.attention_coefficient_sharing}) initialized at 1/r!"
                   if learned_polynomial else
                   f"causal_p_{degree}(QK^T / sqrt(d_k)) V, p_m(x) = sum_(n<=m) x^n / n!")
        identity.update(attention_formula=formula,
                        attention_score_scale="1/sqrt(d_k)", attention_output_normalization=False)
    elif fixed_square_factors:
        identity.update(
            attention_formula=(f"causal_p_{degree}(QK^T / sqrt(d_k)) V / rowsum, "
                               f"p(x) = q1(x)^2 + q2(x)^2, fixed factors, "
                               f"deg q1 = {degree // 2}, deg q2 = {degree // 2 - 1}"),
            attention_score_scale="1/sqrt(d_k)", attention_output_normalization="positive_row_sum",
            attention_coefficients="fixed", attention_fixed_square_factors=list(config.model.attention_fixed_square_factors),
            attention_coefficient_parameterization="sum_of_squares",
            attention_coefficient_sharing="global", attention_coefficient_count=0,
        )
    elif has_sum_of_squares_kernel(config.model):
        identity.update(
            attention_formula=(f"causal_p_{degree}(QK^T / sqrt(d_k)) V / rowsum, p(x) = q1(x)^2 + q2(x)^2, "
                               f"deg q1 = {degree // 2}, deg q2 = {degree // 2 - 1}, learned "
                               f"({config.model.attention_coefficient_sharing}) and initialized so p = T_{degree}"),
            attention_score_scale="1/sqrt(d_k)", attention_output_normalization="positive_row_sum")
    elif learned_polynomial:
        identity.update(
            attention_formula=(f"causal_p_{degree}(QK^T / sqrt(d_k)) V / rowsum, p(x) = sum_(r<={degree}) c_r x^r, "
                               f"learned c ({config.model.attention_coefficient_sharing}) initialized at 1/r!"),
            attention_score_scale="1/sqrt(d_k)", attention_output_normalization="signed_row_sum")
    if fixed_polynomial:
        identity.update(
            attention_kernel="fixed_polynomial" if normalized else "fixed_polynomial_unnormalized",
            attention_formula=(f"causal_p_{degree}(QK^T / sqrt(d_k)) V"
                               + (" / rowsum" if normalized else "")
                               + f", p(x) = sum_(r<={degree}) c_r x^r, fixed c"),
            attention_score_scale="1/sqrt(d_k)",
            attention_output_normalization="signed_row_sum" if normalized else False,
            attention_coefficients="fixed",
            attention_fixed_coefficients=list(config.model.attention_fixed_coefficients),
            attention_coefficient_parameterization="direct",
            attention_coefficient_sharing="global",
            attention_coefficient_count=0,
        )
    if hasattr(config, "_attention_backend_resume_request"):
        identity["attention_backend_resume_requested"] = config._attention_backend_resume_request
    if is_signed_kernel(config.model):
        identity["attention_signed"] = True
    if learned_polynomial:
        identity.update({
            "attention_coefficients": "learned",
            "attention_coefficient_parameterization": config.model.attention_coefficient_parameterization,
            "attention_coefficient_sharing": config.model.attention_coefficient_sharing,
            "attention_coefficient_count": (degree + 1) * (
                1 if config.model.attention_coefficient_sharing == "global"
                else config.model.num_layers * config.model.num_heads),
        })
    if mode != "fixed":
        identity.update({
            "quadratic_mode": mode,
            "quadratic_sharing": config.model.quadratic_sharing,
            "quadratic_parameter_count": (2 if config.model.quadratic_sharing == "global" else
                                          2 * config.model.num_layers * config.model.num_heads),
        })
    if config.model.qk_norm:
        identity.update(qk_norm_identity(config.model))
    scaling = config.model.attention_output_scaling
    if scaling != "none":
        identity.update(
            attention_output_scaling=scaling, attention_output_normalization=scaling,
            attention_formula=identity["attention_formula"] + (
                ", divided by the number of visible keys t" if scaling == "inverse_position"
                else ", then per-head RMSNorm over value channels with a learned gain"))
    if config.model.attention_backend == "fused":
        if not normalized:
            from scaffolds.transformer.attention.backends.fused_unnormalized import _tile_sizes
        else:
            from scaffolds.transformer.attention.backends.fused_taylor import _tile_sizes
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


def qk_norm_identity(model) -> dict:
    """Cosine-score facts recorded beside the mechanism's own identity."""
    from scaffolds.transformer.attention.qk_norm import initial_scale

    return {
        "qk_norm": True,
        "qk_norm_sharing": model.qk_norm_sharing,
        "qk_norm_initial_scale": initial_scale(model),
        "qk_norm_eps": model.qk_norm_eps,
        "qk_norm_parameter_count": 1 if model.qk_norm_sharing == "global" else model.num_layers * model.num_heads,
        "attention_score_formula": "exp(gamma) * <q, k> / (||q|| ||k||), gamma learned, |score| <= exp(gamma)",
    }


def _serializable(array):
    if array.ndim == 0:
        return json_number(array)
    return [_serializable(value) for value in array]


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
    return {
        "raw_parameter_names": names,
        "raw": _serializable(raw),
        "effective_coefficient_names": ["constant", "linear", "quadratic"],
        "effective": _serializable(effective),
        "finite": bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(effective))),
    }


COEFFICIENT_NAMES = ("constant", "linear", "quadratic", "cubic", "quartic")


def polynomial_coefficient_report(params, config) -> dict:
    """Record the learned coefficients c_0..c_m, shared ([m+1]) or per layer and head ([L, H, m+1]).

    A sum-of-squares kernel records its square factors as the raw parameters and the
    coefficients of q1^2 + q2^2 as the effective ones.
    """
    if has_sum_of_squares_kernel(config.model):
        from scaffolds.transformer.attention.nonnegative_polynomial import effective_coefficients_numpy, factor_names

        raw = np.asarray(jax.device_get(params["polynomial_square_factors"][...]), dtype=np.float64)
        raw_names = factor_names(config.model.attention_degree)
        effective = effective_coefficients_numpy(raw, config.model.attention_degree)
    else:
        raw = np.asarray(jax.device_get(params["polynomial_coefficients"][...]), dtype=np.float64)
        raw_names = [f"c{order}" for order in range(raw.shape[-1])]
        effective = raw
    names = [COEFFICIENT_NAMES[order] if order < len(COEFFICIENT_NAMES) else f"degree_{order}"
             for order in range(effective.shape[-1])]
    return {
        "raw_parameter_names": raw_names,
        "raw": _serializable(raw),
        "effective_coefficient_names": names,
        "effective": _serializable(effective),
        "finite": bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(effective))),
    }


def qk_norm_scale_report(params, config) -> dict:
    """Record the log scales gamma and the score bounds exp(gamma), shared ([1]) or per layer and head ([L, H, 1])."""
    raw = np.asarray(jax.device_get(params["qk_norm_log_scale"][...]), dtype=np.float64)
    with np.errstate(over="ignore"):
        effective = np.exp(raw)
    return {
        "raw_parameter_names": ["gamma"],
        "raw": _serializable(raw),
        "effective_coefficient_names": ["alpha"],
        "effective": _serializable(effective),
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
        positivity_fields = (() if config.model.attention_mechanism in ("gla", "gated_deltanet")
                             else ("kernel_min", sum_key))
        for name in positivity_fields:
            value = float(raw[name])
            if is_signed_kernel(config.model):
                # Signed kernels and negative nonzero sums are permitted. Exact
                # zeros are detected in invalid_normalizer_count, not extrema.
                continue
            # Stable exact softmax may underflow individual weights to zero, and a sum of
            # squares may be exactly zero at a common root; neither may be negative.
            invalid = (value < 0 if (not config.model.attention_normalization or name == "kernel_min" and (
                                         config.model.attention_degree is None
                                         or has_sum_of_squares_kernel(config.model)))
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
    elif has_learned_polynomial(config.model):
        coefficient_report = polynomial_coefficient_report(params, config)
        if not coefficient_report["finite"]:
            issues.append("Nonfinite polynomial coefficients")
    elif has_fixed_polynomial(config.model):
        values = np.asarray(config.model.attention_fixed_coefficients, dtype=np.float32)
        coefficient_report = {
            "effective_coefficient_names": [COEFFICIENT_NAMES[order] if order < len(COEFFICIENT_NAMES)
                                             else f"degree_{order}" for order in range(len(values))],
            "effective": _serializable(values),
            "finite": bool(np.all(np.isfinite(values))),
            "trainable": False,
        }
        if not coefficient_report["finite"]:
            issues.append("Nonfinite fixed polynomial coefficients")
    elif has_fixed_square_factors(config.model):
        from scaffolds.transformer.attention.nonnegative_polynomial import effective_coefficients_numpy, factor_names

        factors = np.asarray(config.model.attention_fixed_square_factors, dtype=np.float32)
        effective = effective_coefficients_numpy(factors, config.model.attention_degree)
        coefficient_report = {
            "raw_parameter_names": factor_names(config.model.attention_degree),
            "raw": _serializable(factors),
            "effective_coefficient_names": [COEFFICIENT_NAMES[order] if order < len(COEFFICIENT_NAMES)
                                            else f"degree_{order}" for order in range(len(effective))],
            "effective": _serializable(effective),
            "finite": bool(np.all(np.isfinite(factors)) and np.all(np.isfinite(effective))),
            "trainable": False,
        }
        if not coefficient_report["finite"]:
            issues.append("Nonfinite fixed square factors")
    scale_report = None
    if config.model.qk_norm:
        scale_report = qk_norm_scale_report(params, config)
        if not scale_report["finite"]:
            issues.append("Nonfinite QK-norm scales")
    report = {
        "format": "shared-transformer-attention-diagnostics",
        "version": 1,
        "step": int(state.step),
        **attention_identity(config),
        "sample": provenance,
        "kernel_values": ("vector_gated_recurrence" if config.model.attention_mechanism == "gla" else
                          "gated_delta_rule_recurrence" if config.model.attention_mechanism == "gated_deltanet" else
                          ("unshifted_exp" if config.model.attention_degree is None else
                           "unshifted_learned_polynomial" if has_learned_polynomial(config.model) else "unshifted_taylor")
                          if not config.model.attention_normalization else
                          "unshifted_learned_quadratic" if config.model.quadratic_mode != "fixed" else
                          "unshifted_learned_polynomial" if has_learned_polynomial(config.model) else
                          "row_max_shifted_exp" if config.model.attention_degree is None else "unshifted_taylor"),
        "layers": layers,
        "valid": not issues,
        "issues": issues,
    }
    if coefficient_report is not None:
        report["coefficients"] = coefficient_report
    if has_fixed_polynomial(config.model):
        report["kernel_values"] = "unshifted_fixed_polynomial"
    elif has_fixed_square_factors(config.model):
        report["kernel_values"] = "unshifted_fixed_sum_of_squares"
    if scale_report is not None:
        report["qk_norm_scale"] = scale_report
    return report
