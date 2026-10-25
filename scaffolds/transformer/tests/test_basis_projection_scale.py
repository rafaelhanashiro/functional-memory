"""Relative RBF coefficients, their chain rule, and production-shape QK RMSNorm."""
from dataclasses import replace
from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.triton_scalar import basis_kernel, scalar_attention
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.attention.qk_norm import rms_score_operands
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_learned_basis import dense, model_config, projection
from scaffolds.transformer.tests.test_qk_rms_attention_backends import _inputs, _assert_relative, _vjp


@pytest.mark.parametrize("conditioning", ("static", "query"))
def test_relative_basis_matches_raw_coefficients_and_chain_rule(conditioning):
    m = replace(model_config(conditioning), qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
                initialization="transformer_truncated_reference", rms_norm_fp32=True,
                attention_coefficient_scale="projection",
                attention_coefficient_amplitude_scale="projection" if conditioning == "query" else "none")
    relative = build_model(m, rngs=nnx.Rngs(42))
    raw = build_model(replace(m, attention_coefficient_scale="none", attention_coefficient_amplitude_scale="none"),
                      rngs=nnx.Rngs(42))
    if conditioning == "query":
        # Raw historical bases have their own draw; compare the same generator here.
        raw.query_basis_kernel[...] = relative.query_basis_kernel[...]
    tokens = jnp.arange(14).reshape(2, 7) % 16
    # The normalized start represents precisely the same fitted kernel.
    np.testing.assert_allclose(relative(tokens), raw(tokens), rtol=3e-5, atol=3e-5)
    scale = jnp.asarray(projection().coefficients, jnp.float32)
    names = ("basis_coefficients",) if conditioning == "static" else ("query_basis_bias", "query_basis_amplitude")
    for i, name in enumerate(names):
        target = getattr(relative, name)
        value = jnp.broadcast_to(0.2 + jnp.linspace(0.1, 0.8, 9) + i, target.shape)
        target[...] = value
        getattr(raw, name)[...] = value * scale
    np.testing.assert_allclose(relative(tokens), raw(tokens), rtol=3e-5, atol=3e-5)

    def loss(model):
        logits = model(tokens)
        return jnp.sum(logits * jnp.arange(logits.size).reshape(logits.shape) / logits.size)

    _, relative_grad = nnx.value_and_grad(loss)(relative)
    _, raw_grad = nnx.value_and_grad(loss)(raw)
    for name in names:
        np.testing.assert_allclose(relative_grad[name][...], raw_grad[name][...] * scale, rtol=2e-4, atol=2e-5)
        assert np.linalg.norm(np.asarray(relative_grad[name][...])) > 0
    if conditioning == "query":
        np.testing.assert_allclose(relative_grad["query_basis_kernel"][...], raw_grad["query_basis_kernel"][...],
                                   rtol=2e-4, atol=2e-5)
        assert np.linalg.norm(np.asarray(relative_grad["query_basis_kernel"][...])) > 0


@pytest.mark.parametrize("field", ("attention_coefficient_scale", "attention_coefficient_amplitude_scale"))
def test_factorial_scaling_cannot_silently_treat_rbf_indices_as_polynomial_orders(field):
    with pytest.raises(ValueError, match="Factorial scaling"):
        replace(model_config("query"), **{field: "factorial"})


def _rbf_operation(mode, native, *operands):
    q, k, v, q_gain, k_gain, output_gain, *remaining = operands
    q, k = rms_score_operands(q, k, q_gain, k_gain, eps=1e-6)
    positions = jnp.arange(q.shape[1])
    q, k = (apply_rope(x, positions, 10000.).astype(jnp.bfloat16) for x in (q, k))
    fit = projection()
    scale = jnp.asarray(fit.coefficients, jnp.float32)
    coefficients = scale if mode == "fixed" else remaining[0] * scale
    if native:
        return scalar_attention(q, k, v, kernel=basis_kernel(fit), output_norm_scale=output_gain,
                                basis_coefficients=None if mode == "fixed" else coefficients)
    return dense(q, k, v, output_gain, coefficients, count=9).astype(v.dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ("fixed", "static", "query"))
def test_full_context_rbf_qk_rms(mode):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    projection()
    operands, cotangent = _inputs(mode, 8)
    actual = jax.jit(partial(_vjp, partial(_rbf_operation, mode, True)))(operands, cotangent)
    expected = jax.jit(partial(_vjp, partial(_rbf_operation, mode, False)))(operands, cotangent)
    _assert_relative(actual[0], expected[0], "output", tolerance=.045)
    names = ["q", "k", "v", "q_gain", "k_gain", "output_gain"] + ([] if mode == "fixed" else ["coefficients"])
    for name, got, want in zip(names, actual[1], expected[1], strict=True):
        if name == "coefficients":
            for i in range(9):
                _assert_relative(got[..., i], want[..., i], f"coefficient_{i}", tolerance=.005)
        else:
            _assert_relative(got, want, name, tolerance=.045)
