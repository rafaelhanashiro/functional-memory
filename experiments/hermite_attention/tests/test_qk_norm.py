"""Fixed Hermite/Taylor kernels compose with cosine normalization and learned log scales."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from experiments.hermite_attention.acceptance import EXPECTED_FACTORS, HERMITE_COEFFICIENTS, HERMITE_SCALE
from scaffolds.transformer.acceptance import matches_softmax_backbone_initialization
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.models import build_model

CASES = [("hermite", degree, normalized) for degree in (1, 2, 3, 4) for normalized in (False, True)]
CASES += [("taylor", degree, False) for degree in (1, 2, 3, 4)]


def _kernel_arguments(degree, normalized):
    if degree == 2 and normalized:
        return {"polynomial_square_factors": jnp.asarray(EXPECTED_FACTORS, jnp.float32)}
    return {"polynomial_coefficients": jnp.asarray(
        [HERMITE_SCALE * x for x in HERMITE_COEFFICIENTS[degree]], jnp.float32)}


def _inputs(time=17, value_dim=5, dtype=jnp.float32):
    rng = np.random.default_rng(638)
    q, k = (rng.normal(size=(1, time, 2, 64)) for _ in range(2))
    # Exercise the initial production scales away from singular row sums.
    # Separate shared fixed-polynomial tests cover signed and exactly zero sums.
    q[..., 0] += 8
    k[..., 0] += 4
    v, cotangent = (rng.normal(size=(1, time, 2, value_dim)) for _ in range(2))
    operands = tuple(jnp.asarray(x, dtype) for x in (q, k, v))
    scales = jnp.asarray([[math.log(8.0)], [math.log(5.0)]], jnp.float32)
    return (*operands, scales), jnp.asarray(cotangent, dtype)


def _attention(degree, normalized, backend, family="hermite"):
    def attend(q, k, v, log_scale):
        q, k = cosine_score_operands(q, k, log_scale, eps=1e-6)
        extra = _kernel_arguments(degree, normalized) if family == "hermite" else {}
        return attention_core(q, k, v, backend=backend, attention_degree=degree,
                              attention_normalization=normalized, **extra)
    return attend


def _reference(degree, normalized, family="hermite"):
    def attend(q, k, v, log_scale):
        unit = lambda x: x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)
        cosine = jnp.einsum("bthd,bshd->bhts", unit(q), unit(k))
        scores = jnp.exp(log_scale)[None, :, :, None] * cosine
        if family == "hermite":
            # The recurrence is independent of the production monomial/factor code.
            previous, current = jnp.ones_like(scores), scores
            weights = previous + current
            for order in range(2, degree + 1):
                previous, current = current, scores * current - (order - 1) * previous
                weights = weights + current / math.factorial(order)
            weights *= HERMITE_SCALE
        else:
            weights = sum(scores ** order / math.factorial(order) for order in range(degree + 1))
        weights = jnp.where(jnp.tril(jnp.ones(scores.shape[-2:], bool)), weights, 0)
        if normalized:
            weights /= weights.sum(axis=-1, keepdims=True)
        return jnp.einsum("bhts,bshd->bthd", weights, v)
    return attend


def _evaluate(function, operands, cotangent):
    @jax.jit
    def value_and_backward(*arrays):
        output, pullback = jax.vjp(function, *arrays)
        return output, pullback(cotangent)
    return value_and_backward(*operands)


def _assert_close(actual, expected, tolerances):
    for name, got, want, tolerance in zip(
            ("output", "dq", "dk", "dv", "dlog_scale"), (actual[0], *actual[1]),
            (expected[0], *expected[1]), tolerances, strict=True):
        got, want = (np.asarray(x, np.float64) for x in (got, want))
        assert got.shape == want.shape, name
        assert np.isfinite(got).all() and np.isfinite(want).all(), name
        relative_error = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert relative_error < tolerance, f"{name}: relative L2 error {relative_error}"


@pytest.mark.parametrize("family,degree,normalized", CASES)
def test_cosine_output_and_all_gradients_match_independent_fp64(family, degree, normalized):
    operands, cotangent = _inputs()
    with jax.default_matmul_precision("highest"):
        actual = _evaluate(_attention(degree, normalized, "blocked", family), operands, cotangent)
    with jax.enable_x64():
        expected = _evaluate(_reference(degree, normalized, family), tuple(x.astype(jnp.float64) for x in operands),
                             cotangent.astype(jnp.float64))
    _assert_close(actual, expected, (3e-5, 1e-4, 1e-4, 3e-5, 1e-4))


@pytest.mark.parametrize("degree,normalized", ((4, False), (4, True), (2, True)))
def test_fixed_kernel_keeps_backbone_and_adds_only_the_qk_scale(degree, normalized):
    fixed = {"attention_fixed_square_factors": EXPECTED_FACTORS,
             "attention_coefficient_parameterization": "sum_of_squares"} if degree == 2 else {
        "attention_fixed_coefficients": tuple(HERMITE_SCALE * x for x in HERMITE_COEFFICIENTS[degree])}
    config = Config(model=ModelConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_heads=2, num_kv_heads=2,
        num_layers=2, max_seq_len=8, qk_head_dim=64, value_head_dim=8, qk_norm=True,
        attention_degree=degree, attention_normalization=normalized, attention_backend="blocked", **fixed))
    facts = matches_softmax_backbone_initialization(config)
    assert set(facts["mechanism_parameters"]) == {"qk_norm_log_scale"}
    assert facts["mechanism_parameters"]["qk_norm_log_scale"]["weight_decay"] is False
    model = build_model(config.model, 0)
    tokens = jnp.asarray([[1, 2, 3, 4]])
    output = model(tokens)
    for name in model.layer_names:
        attention = getattr(model, name).attention
        attention.q_proj.kernel[...] *= 3.0
        attention.k_proj.kernel[...] *= 0.25
    np.testing.assert_allclose(model(tokens), output, rtol=2e-4, atol=2e-5)
    graph, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda p: jnp.square(nnx.merge(graph, p)(tokens)).mean())(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads["qk_norm_log_scale"][...])) > 0
    assert not any("polynomial" in str(path) for path, _ in nnx.to_flat_state(params))


@pytest.mark.gpu
@pytest.mark.parametrize("family,degree,normalized", CASES)
def test_compiled_bf16_cosine_and_scale_gradients(family, degree, normalized):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires the allocated GPU to compile the fused kernel")
    operands, cotangent = _inputs(time=2048, value_dim=64, dtype=jnp.bfloat16)
    fused = _evaluate(_attention(degree, normalized, "fused", family), operands, cotangent)
    blocked = _evaluate(_attention(degree, normalized, "blocked", family), operands, cotangent)
    _assert_close(fused, blocked, (0.006, 0.03, 0.03, 0.02, 0.02))
    assert fused[0].dtype == jnp.bfloat16
    assert fused[1][-1].dtype == jnp.float32
