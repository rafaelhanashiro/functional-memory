"""A trained-width attention head: actual gate projections and fused weights against dense FP64."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.parameters import parameters_for_layer
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.attention.qk_norm import rms_score_operands, scale_rms_queries, unit_cosine_score_operands
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_gated_even_coefficients import _model

ORDERS = tuple(range(2, 17, 2))
GATE_PARAMETERS = ("query_polynomial_kernel", "query_polynomial_bias", "query_exp_gate_kernel", "query_exp_gate_bias")
VARIANTS = (("rms_norm_cosine", "zero"), ("rms_norm", "zero"), ("rms_norm", "logit_taylor"))


def _check_head(score, bias, backend, tokens, dtype, record_property):
    heads, hidden, dk, dv = 2, 64, 4, 64
    config = replace(_model(bias), hidden_size=hidden, intermediate_size=128, max_seq_len=tokens,
                     value_head_dim=dv, attention_degree=16, attention_coefficient_orders=ORDERS,
                     attention_coefficient_exp_gate=True, attention_backend=backend,
                     attention_coefficient_projection_initial_std=hidden ** -0.5,
                     qk_norm_type=score, qk_norm_head_scale=(score == "rms_norm"))
    model = build_model(config, 42)
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    attn = model.layers_0.attention
    params = parameters_for_layer(model, model.attention_parameter_specs, 0)
    rng = np.random.default_rng(64)
    raw = jnp.asarray(rng.normal(size=(1, tokens, hidden)), jnp.float32)
    # Distinct raw and post-RMS streams expose using the wrong input for any of the nine gates.
    x = jnp.asarray(rng.normal(size=raw.shape), dtype)
    cotangent = jnp.asarray(rng.normal(size=x.shape), dtype)
    gates = tuple(params[k] for k in GATE_PARAMETERS)

    def actual(raw, *state):
        return attn(x, coefficient_source_input=raw, attention_parameters={**params, **dict(zip(GATE_PARAMETERS, state))})

    def output_vjp(fn, inputs, cot):
        value, pullback = jax.vjp(fn, *inputs)
        return value, pullback(cot.astype(value.dtype))

    got, grads = jax.jit(lambda *inputs: output_vjp(actual, inputs, cotangent))(raw, *gates)
    # Use exactly the rounded score operands and values that enter the trained kernel, including positional rotations.
    q, k = [proj(x).reshape(1, tokens, heads, dk) for proj in (attn.q_proj, attn.k_proj)]
    q, k = rms_score_operands(q, k, params["qk_norm_q_gain"], params["qk_norm_k_gain"], eps=config.qk_norm_eps)
    if config.qk_norm_head_scale:
        q = scale_rms_queries(q, params["qk_norm_head_log_scale"])
    q, k = (apply_rope(t, jnp.arange(tokens), config.rope_theta) for t in (q, k))
    if score == "rms_norm_cosine":
        q, k = unit_cosine_score_operands(q, k, eps=config.qk_norm_eps)
    v = attn.v_proj(x).reshape(1, tokens, heads, dv)
    data = (q.astype(dtype), k.astype(dtype), v, attn.o_proj.kernel[...].astype(dtype), params["output_norm_scale"],
            cotangent, raw, *gates)
    host = [np.asarray(t, np.float64) for t in data]
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        q64, k64, v64, out64, gamma64, cot64, raw64, *gates64 = map(jnp.asarray, host)
        scores = jnp.einsum("bthd,bihd->bthi", q64, k64) / dk**.5
        causal = jnp.arange(tokens)[None, :] <= jnp.arange(tokens)[:, None]

        def dense_head(raw, wp, bp, we, be):
            unit = raw / jnp.maximum(jnp.linalg.norm(raw, axis=-1, keepdims=True), 1e-6)
            poly = jax.nn.sigmoid(jnp.einsum("btd,hdr->bthr", unit, wp) + bp)
            exp = jax.nn.sigmoid(jnp.einsum("btd,hdr->bthr", unit, we) + be)
            weights = exp * jnp.exp(scores)
            for j, r in enumerate(ORDERS):
                weights += poly[..., j, None] * scores ** r
            sums = jnp.einsum("bthi,bihd->bthd", jnp.where(causal[None, :, None, :], weights, 0), v64)
            y = sums * jax.lax.rsqrt(jnp.mean(sums**2, axis=-1, keepdims=True) + config.rms_norm_eps)
            return (y * gamma64).reshape(1, tokens, heads * dv)

        def reference(*args):
            return dense_head(*args) @ out64

        def matched(*args):
            # The corrected kernel passes FP32 through head RMSNorm. The O projection's activations,
            # weights, result and corresponding cotangents are BF16 in the production recipe.
            head = dense_head(*args).astype(dtype).astype(jnp.float64)
            return (head @ out64).astype(dtype)

        expected, refs = output_vjp(reference, (raw64, *gates64), cot64)
        matched_out, matched_grads = output_vjp(matched, (raw64, *gates64), cot64)

    def relative_error(actual, expected, label):
        a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        scale = np.linalg.norm(e.ravel())
        assert np.isfinite(a).all() and np.isfinite(e).all() and scale > 0, label
        return float(np.linalg.norm((a - e).ravel()) / scale)

    errors, matched_errors = {}, {}

    def compare(label, actual, full, matched):
        errors[label] = relative_error(actual, full, label)
        matched_errors[label] = relative_error(actual, matched, label)

    # Each Taylor gate receives a relative test, including gradients at the scale of 1/16!.
    compare("output", got, expected, matched_out)
    compare("raw input VJP", grads[0], refs[0], matched_grads[0])
    for name, grad, ref, match in zip(GATE_PARAMETERS, grads[1:], refs[1:], matched_grads[1:]):
        for index in range(grad.shape[-1]):
            compare(f"{name}[{index}] VJP", grad[..., index], ref[..., index], match[..., index])
    record_property("full_fp64_relative_l2", errors)
    record_property("matched_precision_relative_l2", matched_errors)
    for label in errors:
        full_limit = .045 if dtype == jnp.bfloat16 else .001
        matched_limit = .015 if dtype == jnp.bfloat16 else .001
        assert errors[label] < full_limit, f"{label} relative L2 vs FP64: {errors[label]:.6g}"
        assert matched_errors[label] < matched_limit, f"{label} relative L2 at matched precision: {matched_errors[label]:.6g}"


@pytest.mark.parametrize(("score", "bias"), VARIANTS)
def test_blocked_hybrid_head_matches_dense_fp64(score, bias, record_property):
    _check_head(score, bias, "blocked", 64, jnp.float32, record_property)


@pytest.mark.gpu
@pytest.mark.parametrize(("score", "bias"), VARIANTS)
def test_fused_bf16_hybrid_head_matches_dense_fp64(score, bias, record_property):
    assert jax.default_backend() == "gpu"
    _check_head(score, bias, "fused", 256, jnp.bfloat16, record_property)
