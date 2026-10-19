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
    has_fixed_polynomial, has_fixed_square_factors, has_learned_polynomial, has_rational_kernel,
    has_sum_of_squares_kernel, is_odd_taylor, is_signed_kernel,
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


def _initialization_text(model) -> str:
    """How the learned coefficients start, for formula strings."""
    initialization = getattr(model, "attention_coefficient_initialization", "taylor")
    return "1/r!" if initialization == "taylor" else "linear attention (a_1 = 1, every other coefficient 0)"


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
        independent = model.gdn_erase_mode == "independent"
        return {
            "attention_mechanism": "gated_deltanet",
            "attention_kernel": "gated_deltanet",
            "attention_backend": model.attention_backend,
            "attention_backend_requested": getattr(config, "_attention_backend_request", model.attention_backend),
            "attention_implementation": (("fla_chunk_independent_gated_delta_rule_triton" if independent
                                           else "fla_chunk_gated_delta_rule_triton") if model.attention_backend == "fused"
                                         else "token_recurrence_reference"),
            "attention_formula": (("S_t=a_t S_{t-1}+k_t (b_t v_t-lambda_t a_t S_{t-1}^T k_t)^T; "
                                   "lambda_t=sigmoid(W_lambda x); " if independent else
                                   "S_t=a_t S_{t-1}+b_t k_t (v_t-a_t S_{t-1}^T k_t)^T; ") +
                                  "o_t=S_t^T q_t/sqrt(d_k); "
                                  "q,k L2-normalized after short conv+SiLU; a_t=exp(-exp(A_log) softplus(W_a x+dt_bias)); "
                                  f"b_t={model.gdn_write_activation}(W_b x)"),
            "attention_score_scale": "1/sqrt(d_k)",
            "attention_output_normalization": "per_head_rmsnorm_times_silu_output_gate",
            "gdn_conv_size": model.gdn_conv_size,
            "gdn_erase_mode": model.gdn_erase_mode,
            "gdn_linear_write_activation": model.gdn_write_activation,
            "gdn_write_initialization": f"{model.gdn_write_activation}_write_projection",
            "gdn_erase_initialization": "copy_write_projection" if independent else "shared_write_gate",
            "gdn_state_elements_per_layer": model.num_heads * model.qk_head_dim * model.value_head_dim,
            "gdn_initial_state": "zero_each_sequence",
            "gdn_kernel_source": ("flash-linear-attention chunk_gated_delta_rule with local independent-gate WY kernels"
                                  if independent else "flash-linear-attention chunk_gated_delta_rule (pinned in the scaffold lock)"),
            "gdn_chunk_size": 64 if model.attention_backend == "fused" else None,
            "diagnostic_backend": model.attention_backend,
        }
    degree = config.model.attention_degree
    mode = config.model.quadratic_mode
    normalized = config.model.attention_normalization
    learned_polynomial = has_learned_polynomial(config.model)
    conditioning = config.model.attention_coefficient_conditioning
    query_conditioned = conditioning in ("query", "query_key")
    key_conditioned = conditioning == "query_key"
    query_tanh = query_conditioned and config.model.attention_coefficient_activation == "tanh"
    rational = has_rational_kernel(config.model)
    fixed_polynomial = has_fixed_polynomial(config.model)
    fixed_square_factors = has_fixed_square_factors(config.model)
    sharing = config.model.attention_coefficient_sharing
    initialized = _initialization_text(config.model)
    identity = {
        "attention_degree": degree,
        "attention_normalization": normalized,
        "attention_kernel": (("exp_unnormalized" if degree is None else
                              "learned_polynomial_unnormalized" if learned_polynomial else "taylor_unnormalized")
                             if not normalized else
                             "learned_quadratic" if mode != "fixed" else
                             "fixed_polynomial_sum_of_squares" if fixed_square_factors else
                             "learned_polynomial_sum_of_squares" if has_sum_of_squares_kernel(config.model) else
                             "learned_polynomial_rational" if rational else
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
                   f"learned c ({sharing}) initialized at {initialized}"
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
    elif rational:
        from scaffolds.transformer.attention.rational_polynomial import (
            LINEAR_START_SQUARE_FACTOR, denominator_constant, denominator_degree, denominator_parameterization,
        )

        denominator, form = denominator_degree(config.model), denominator_parameterization(config.model)
        constant = denominator_constant(config.model)
        half = denominator // 2
        if form == "direct" and constant == "fixed":
            denominator_formula = f"q(x) = 1 + sum_(1<=r<={denominator}) b_r x^r, learned a, b"
        elif form == "direct":
            denominator_formula = f"q(x) = sum_(r<={denominator}) b_r x^r, learned a, b including b_0"
        elif constant == "fixed":
            denominator_formula = (f"q(x) = (1 + sum_(1<=r<={half}) u_r x^r)^2 "
                                   f"+ (sum_(1<=r<={half}) w_r x^r)^2, learned a, u, w")
        else:
            denominator_formula = (f"q(x) = (sum_(r<={half}) q1_r x^r)^2 + (sum_(r<={half - 1}) q2_r x^r)^2, "
                                   f"learned a, q1, q2")
        start = initialized
        if config.model.attention_coefficient_initialization == "linear" and form == "sum_of_squares":
            start += (f", except w_1 = {LINEAR_START_SQUARE_FACTOR:g}" if constant == "fixed"
                      else ", denominator q1_0 = q2_0 = 1/sqrt(2)")
        identity.update(
            attention_formula=(f"causal_p_{degree}(s) V / sum_j q_{denominator}(s_j), s = QK^T / sqrt(d_k), "
                               f"p(x) = sum_(r<={degree}) a_r x^r, {denominator_formula} ({sharing}) "
                               f"initialized at {start}"),
            attention_score_scale="1/sqrt(d_k)",
            attention_output_normalization=("signed_denominator_row_sum" if form == "direct"
                                            else "nonnegative_denominator_row_sum"),
            attention_denominator_degree=denominator, attention_denominator_parameterization=form,
            attention_denominator_constant=constant)
    elif learned_polynomial:
        identity.update(
            attention_formula=(f"causal_p_{degree}(QK^T / sqrt(d_k)) V / rowsum, p(x) = sum_(r<={degree}) c_r x^r, "
                               f"learned c ({sharing}) initialized at {initialized}"),
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
        optim = getattr(config, "optim", None)
        penalty = getattr(optim, "coefficient_penalty", 0.0)
        penalize_denominator = rational and getattr(optim, "coefficient_penalty_denominator", True)
        if rational:
            from scaffolds.transformer.attention.rational_polynomial import denominator_constant, denominator_degree

            per_polynomial = degree + 1 + denominator_degree(config.model) + (denominator_constant(config.model) == "learned")
        else:
            per_polynomial = (degree + 1) * (
                config.model.hidden_size + 1 + int(query_tanh)
                + int(query_tanh and config.model.attention_coefficient_tanh_bias)
                + (config.model.hidden_size + 1 if key_conditioned else 0)
                if query_conditioned else 1)
        identity.update({
            "attention_coefficients": "learned",
            "attention_coefficient_parameterization": config.model.attention_coefficient_parameterization,
            "attention_coefficient_initialization": getattr(config.model, "attention_coefficient_initialization", "taylor"),
            "attention_coefficient_sharing": sharing,
            "attention_coefficient_count": per_polynomial * (
                1 if sharing == "global" else config.model.num_layers * config.model.num_heads),
            "coefficient_penalty": penalty,
        })
        if rational:
            identity["coefficient_penalty_denominator"] = penalize_denominator
        if penalty:
            identity["coefficient_penalty_formula"] = (
                "training loss = NLL + coefficient_penalty * sum over polynomials of "
                "a_0^2 + sum_(r>=2) r^2 a_r^2" + (" + sum_(r>=1) r^2 b_r^2" if penalize_denominator else "")
                + "; a_1 unpenalized" + ("; denominator unpenalized" if rational and not penalize_denominator else "")
                + "; validation loss and recorded NLL exclude it")
    if query_conditioned:
        amplitude = config.model.attention_coefficient_amplitude_initial
        spread = config.model.attention_coefficient_projection_initial_std
        inner_bias = query_tanh and config.model.attention_coefficient_tanh_bias
        tanh_start = (f"W initially N(0, {spread:g}^2) per entry" if spread else "W initially zero") + (
            f", a initially {amplitude:g}" if amplitude != 1.0 else ", a initially 1") + (
            ", beta initially 0" if inner_bias else "")
        tanh_formula = "c(x_t) = b + a*tanh(x_t W + beta)" if inner_bias else "c(x_t) = b + a*tanh(x_t W)"
        identity.update(
            attention_kernel="query_key_polynomial_unnormalized" if key_conditioned else "query_polynomial_unnormalized",
            attention_coefficient_conditioning=conditioning,
            attention_coefficient_activation=config.model.attention_coefficient_activation,
            attention_coefficient_input="pre_attention_rmsnorm_activation",
            attention_coefficient_projection_dtype="float32",
            **({"attention_coefficient_amplitude_initial": amplitude,
                "attention_coefficient_projection_initial_std": spread} if (amplitude, spread) != (1.0, 0.0) else {}),
            **({"attention_coefficient_tanh_bias": True} if inner_bias else {}),
            attention_formula=((f"sum_(i<=t) sum_(r<={degree}) c_r(x_t) f_r(x_i) s_ti^r v_i" if key_conditioned
                                else f"sum_(i<=t) sum_(r<={degree}) c_r(x_t) s_ti^r v_i")
                               + ", s_ti = Q_t K_i^T / sqrt(d_k); "
                               + (f"{tanh_formula} per layer/head, {tanh_start}, b_r initially 1/r!; "
                                  if query_tanh else "c(x_t) = x_t W + b per layer/head, W initially zero, b initially (1,1); ")
                               + ("f(x_i) = 1 + a_k*tanh(x_i W_k) per layer/head, W_k drawn like W and a_k starting "
                                  "like a; x_t and x_i are the key and query tokens' activations"
                                  if key_conditioned else "x_t is the activation")
                               + " after the block input RMSNorm"),
        )
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
            attention_formula=identity.get("attention_formula", "causal_softmax(QK^T / sqrt(d_k)) V") + (
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
            "attention_implementation": ("pallas_triton_query_polynomial_per_degree_values" if key_conditioned else
                                         "pallas_triton_query_polynomial" if query_tanh else
                                         "pallas_triton_linear_plus_fp32_prefix_sum" if query_conditioned else
                                         "cudnn" if degree is None and normalized else "pallas_triton"),
            "query_block_size": block_q,
            "key_block_size": block_k,
            "diagnostic_backend": "blocked",
            "diagnostic_query_block_size": QUERY_BLOCK_SIZE,
        })
    elif config.model.attention_backend == "mosaic":
        from scaffolds.transformer.attention.backends.mosaic.api import default_precision

        # Tiles differ between the Hopper and Blackwell kernels; the GPU is recorded with the hardware.
        identity.update({
            "attention_implementation": ("pallas_mosaic_gpu_query_polynomial_per_degree_values" if key_conditioned
                                         else "pallas_mosaic_gpu_query_polynomial" if query_tanh
                                         else "pallas_mosaic_gpu"),
            "attention_weight_precision": default_precision("query" if query_conditioned else "taylor", degree),
            "query_block_size": None,
            "key_block_size": None,
            "diagnostic_backend": "blocked",
            "diagnostic_query_block_size": QUERY_BLOCK_SIZE,
        })
    elif config.model.attention_backend in ("manual", "xla", "cudnn"):
        identity.update({
            "query_block_size": None,
            "diagnostic_backend": "blocked",
            "diagnostic_query_block_size": QUERY_BLOCK_SIZE,
        })
    if config.model.positional_encoding == "path":
        from scaffolds.transformer.attention.path import CHUNK_SIZE

        model = config.model
        identity.update({
            "positional_encoding": "path",
            "path_w_rank": model.path_w_rank,
            "path_conv_size": model.path_conv_size,
            "path_parameter_count": model.num_layers * (model.hidden_size * model.path_w_rank + model.num_heads * (
                (model.path_w_rank + model.path_conv_size) * model.qk_head_dim + model.hidden_size + 1)),
            "attention_implementation": "jax_path_chunked_ut_transform",
            "path_chunk_size": CHUNK_SIZE,
            "query_block_size": None,
            "diagnostic_backend": "jax_path_chunked_ut_transform",
            "attention_score_position_encoding": (
                "PaTH: s_ts = q_t^T H_t ... H_(s+1) k_s / sqrt(d_k), H_t = I - beta_t w_t w_t^T; "
                "w_t = L2norm(SiLU(causal_conv(x_t W_down W_up))) per head, beta_t = 2 sigmoid(x_t W_beta + b); no RoPE"),
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


def _coefficient_name(order: int) -> str:
    return COEFFICIENT_NAMES[order] if order < len(COEFFICIENT_NAMES) else f"degree_{order}"


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
    elif has_rational_kernel(config.model):
        from scaffolds.transformer.attention.rational_polynomial import (
            denominator_argument, denominator_constant, denominator_degree, denominator_names,
            denominator_parameterization, expanded_denominator,
        )

        # Numerator a_0..a_m, then the denominator's parameters; effective values are
        # a_0..a_m followed by the expanded denominator's b_1..b_d, from b_0 when learned.
        form, constant = denominator_parameterization(config.model), denominator_constant(config.model)
        numerator = np.asarray(jax.device_get(params["polynomial_coefficients"][...]), dtype=np.float64)
        stored = jax.device_get(params[denominator_argument(config.model)][...])
        denominator = np.asarray(stored, dtype=np.float64)
        raw = np.concatenate([numerator, denominator], axis=-1)
        raw_names = ([f"a{order}" for order in range(numerator.shape[-1])]
                     + denominator_names(denominator_degree(config.model), form, constant))
        expanded = np.asarray(expanded_denominator(jnp.asarray(stored), form, constant), dtype=np.float64)
        effective = np.concatenate([numerator, expanded if constant == "learned" else expanded[..., 1:]], axis=-1)
    else:
        raw = np.asarray(jax.device_get(params["polynomial_coefficients"][...]), dtype=np.float64)
        raw_names = [f"c{order}" for order in range(raw.shape[-1])]
        effective = raw
    if has_rational_kernel(config.model):
        from scaffolds.transformer.attention.rational_polynomial import denominator_constant, denominator_degree

        degree = config.model.attention_degree
        first = 0 if denominator_constant(config.model) == "learned" else 1
        names = ([f"numerator_{_coefficient_name(order)}" for order in range(degree + 1)]
                 + [f"denominator_{_coefficient_name(order)}" for order in range(first, denominator_degree(config.model) + 1)])
    else:
        names = [_coefficient_name(order) for order in range(effective.shape[-1])]
    return {
        "raw_parameter_names": raw_names,
        "raw": _serializable(raw),
        "effective_coefficient_names": names,
        "effective": _serializable(effective),
        "finite": bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(effective))),
    }



def query_coefficient_report(params, raw_layers) -> dict:
    """Summarize actual input-dependent coefficients on the pinned diagnostic sample.

    Projection biases are explicitly named; they are never presented as the
    effective coefficients. Summary arrays have shape [layers,heads,degree+1].
    """
    bias = np.asarray(jax.device_get(params["query_polynomial_bias"][...]), dtype=np.float64)
    kernel = np.asarray(jax.device_get(params["query_polynomial_kernel"][...]), dtype=np.float64)
    amplitude = (np.asarray(jax.device_get(params["query_polynomial_amplitude"][...]), dtype=np.float64)
                 if "query_polynomial_amplitude" in params else None)
    inner_bias = (np.asarray(jax.device_get(params["query_polynomial_tanh_bias"][...]), dtype=np.float64)
                  if "query_polynomial_tanh_bias" in params else None)
    key_amplitude = (np.asarray(jax.device_get(params["key_polynomial_amplitude"][...]), dtype=np.float64)
                     if "key_polynomial_amplitude" in params else None)
    summaries = {
        statistic: np.stack([layer[f"query_coefficient_{statistic}"] for layer in raw_layers])
        for statistic in ("mean", "std", "min", "max")
    }
    report = {
        "conditioning": "query",
        "sample": "fixed_diagnostic_sample",
        "effective_coefficient_names": [_coefficient_name(r) for r in range(bias.shape[-1])],
        "effective_summary": {name: _serializable(values) for name, values in summaries.items()},
        "projection_bias": _serializable(bias),
        "projection_kernel_l2": _serializable(np.linalg.norm(kernel, axis=-2)),
        "finite": bool(np.isfinite(bias).all() and np.isfinite(kernel).all()
                       and (amplitude is None or np.isfinite(amplitude).all())
                       and (inner_bias is None or np.isfinite(inner_bias).all())
                       and all(np.isfinite(values).all() for values in summaries.values())),
    }
    if amplitude is not None:
        report.update(activation="tanh", modulation_amplitude=_serializable(amplitude))
    if inner_bias is not None:
        report["tanh_bias"] = _serializable(inner_bias)
    if key_amplitude is not None:
        key_kernel = np.asarray(jax.device_get(params["key_polynomial_kernel"][...]), dtype=np.float64)
        key_summaries = {
            statistic: np.stack([layer[f"key_factor_{statistic}"] for layer in raw_layers])
            for statistic in ("mean", "std", "min", "max")
        }
        report["key_factors"] = {
            "formula": "1 + a_k*tanh(x_i W_k)",
            "effective_summary": {name: _serializable(values) for name, values in key_summaries.items()},
            "amplitude": _serializable(key_amplitude),
            "projection_kernel_l2": _serializable(np.linalg.norm(key_kernel, axis=-2)),
        }
        report["finite"] = bool(report["finite"] and np.isfinite(key_amplitude).all() and np.isfinite(key_kernel).all()
                                and all(np.isfinite(values).all() for values in key_summaries.values()))
    return report

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
            if name.startswith(("query_coefficient_", "key_factor_")):
                continue  # Per-head/order arrays are summarized in the coefficient report.
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
    elif config.model.attention_coefficient_conditioning in ("query", "query_key"):
        coefficient_report = query_coefficient_report(params, raw_layers)
        if not coefficient_report["finite"]:
            issues.append("Nonfinite query-dependent polynomial coefficients or projection parameters")
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
    if config.model.attention_coefficient_conditioning in ("query", "query_key"):
        report["kernel_values"] = "unshifted_query_polynomial"
    if scale_report is not None:
        report["qk_norm_scale"] = scale_report
    return report
