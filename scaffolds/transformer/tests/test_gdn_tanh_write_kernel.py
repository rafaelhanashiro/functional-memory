"""GPU qualification of signed linear GDN writes and their tanh chain rule."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.tests.test_gdn_tanh_write import signed_operands, signed_state_oracle

pytestmark = pytest.mark.gpu


def _check(args, cotangent, tolerance):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import fla_independent_gated_delta_rule

    scale = args[0].shape[-1] ** -.5
    expected, expected_gradients = signed_state_oracle(args, cotangent, scale)
    function = lambda *a: fla_independent_gated_delta_rule(*a, scale)
    output = function(*args)
    gradients = jax.grad(lambda *a: jnp.sum(function(*a).astype(jnp.float32) * cotangent.astype(jnp.float32)),
                         argnums=range(6))(*args)
    assert output.dtype == args[2].dtype
    assert tuple(a.dtype for a in gradients) == tuple(a.dtype for a in args)
    failures = []
    for name, actual, reference in zip(("output", "dq", "dk", "dv", "dg", "dbeta", "derase"),
                                        (output, *gradients), (expected, *expected_gradients), strict=True):
        actual = np.asarray(actual, np.float64)
        error, norm = np.linalg.norm(actual - reference), np.linalg.norm(reference)
        if not np.all(np.isfinite(actual)) or error > tolerance * norm + 2e-5:
            failures.append((name, float(error), float(norm), tolerance))
    assert not failures, failures
    return output, gradients


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 5e-3), (jnp.bfloat16, 2e-2)])
@pytest.mark.parametrize("erase_value", [None, 0., 1e-6])
def test_signed_fused_writes_match_fp64_matrix_oracle_and_all_six_gradients(dtype, tolerance, erase_value):
    args, cotangent = signed_operands(dtype, erase_value=erase_value)
    beta = np.asarray(args[4], np.float32)
    assert beta.min() < 0 < beta.max() and np.any(beta == 0)
    output, gradients = _check(args, cotangent, tolerance)
    assert float(jnp.linalg.norm(output.astype(jnp.float32))) > 0
    assert float(jnp.linalg.norm(gradients[4].astype(jnp.float32))) > 0
    assert float(jnp.linalg.norm(gradients[5].astype(jnp.float32))) > 0


@pytest.mark.parametrize("strong_decay", [False, True])
def test_production_length_dense_k64_v64_signed_writes_and_gradients(strong_decay):
    args, cotangent = signed_operands(jnp.bfloat16, length=2048, heads=1, strong_decay=strong_decay)
    _check(args, cotangent, 2e-2)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 5e-3), (jnp.bfloat16, 2e-2)])
def test_fused_tanh_logit_chain_rule_including_zero_and_saturated_writes(dtype, tolerance):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import fla_independent_gated_delta_rule

    args, cotangent = signed_operands(dtype)
    q, k, v, g, _, erase = args
    logits = jnp.asarray(np.random.default_rng(105).normal(size=g.shape), jnp.float32)
    logits = logits.at[:, :5].set(jnp.asarray([0., -1e-6, 1e-6, -5., 5.])[:, None])
    beta32 = jnp.tanh(logits)
    beta = beta32.astype(dtype)
    _, expected_gradients = signed_state_oracle((q, k, v, g, beta, erase), cotangent, 64 ** -.5)

    def loss(z):
        output = fla_independent_gated_delta_rule(q, k, v, g, jnp.tanh(z).astype(dtype), erase, 64 ** -.5)
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual = np.asarray(jax.grad(loss)(logits), np.float64)
    expected = expected_gradients[4] * (1 - np.asarray(beta32, np.float64) ** 2)
    assert np.all(np.isfinite(actual))
    assert np.linalg.norm(actual - expected) <= tolerance * np.linalg.norm(expected) + 2e-5
    assert np.linalg.norm(actual[:, 0]) > 0
