"""The row-power shifted kernel at Q/K width 64, the reference shape's width. Its tiles were tuned and qualified at
Q/K width 4 (test_shifted_polynomial_attention.py); these cases run the same programs at width 64 and value width 64:
the kernel against FP64 at degrees 8 and 16 with signed tanh coefficients, and the fused pure tanh and tanh query
models against the blocked reference's."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.output_scaling import max_abs_head_rms
from scaffolds.transformer.attention.shifted_polynomial import shifted_polynomial_attention
from scaffolds.transformer.tests.test_shifted_polynomial_attention import (
    LOGIT_ATOL, _assert_models_agree, _oracle, _pure_tanh_config, _source_config,
)

WIDTH = 64
DEGREES = (8, 16)


def _inputs(time, degree):
    """Scores with standard deviation 6 (up to about 20), as QK RMSNorm's learned scale reaches, and coefficients
    tanh(g + atanh(1/2))/r! of either sign, as pure tanh coefficients take."""
    rng = np.random.default_rng(640 + degree)
    q = 6 * rng.normal(size=(1, time, 2, WIDTH)).astype(np.float32)
    k = rng.normal(size=(1, time, 2, WIDTH)).astype(np.float32)
    v = rng.normal(size=(1, time, 2, 64)).astype(np.float32)
    g = rng.normal(size=(1, time, 2, degree + 1))
    c = (np.tanh(g + math.atanh(0.5)) / np.array([math.factorial(r) for r in range(degree + 1)])).astype(np.float32)
    cot = rng.normal(size=v.shape).astype(np.float32)
    return (q, k, v, c), cot


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ["blocked", pytest.param("fused", marks=pytest.mark.gpu)])
def test_shifted_composition_and_all_gradients_at_width_64(backend, degree):
    # 257 tokens: five query tiles of 64 and nine key tiles of 32, the last masked.
    if backend == "fused" and jax.default_backend() != "gpu":
        pytest.skip("The fused Triton program requires a GPU")
    args, cot = _inputs(257, degree)
    with jax.enable_x64(True):
        oracle, pull = jax.vjp(lambda *a: _oracle(*a, "identity"), *(jnp.asarray(x, jnp.float64) for x in args))
        oracle_grads = pull(jnp.asarray(cot, jnp.float64))

    def normalized(*args):
        result, shift = shifted_polynomial_attention(*args, transform="identity", backend=backend)
        return max_abs_head_rms(result, 1e-6, shift[..., None])

    y, pull = jax.vjp(jax.jit(normalized), *map(jnp.asarray, args))
    grads = pull(jnp.asarray(cot))
    # At scores to about 20, a degree-16 row whose signed terms cancel resolves in FP32 (the fused kernel's TF32x3
    # products) to about 1e-5 of the output's largest entry (H200, 2026-11-23): bound the error by 1e-4 of it.
    np.testing.assert_allclose(y, oracle, rtol=2e-4, atol=1e-4 * float(jnp.max(jnp.abs(oracle))))
    for name, actual, expected in zip("qkvc", grads, oracle_grads, strict=True):
        scale = float(jnp.max(jnp.abs(expected)))
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-4 * max(1.0, scale), err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_fused_pure_tanh_models_match_the_blocked_shifted_models_at_width_64(degree):
    assert jax.default_backend() == "gpu"
    blocked, fused = (_pure_tanh_config("none", attention_backend=backend, attention_polynomial_shift="row_power",
                                        attention_degree=degree, qk_head_dim=WIDTH, value_head_dim=WIDTH)
                      for backend in ("blocked", "fused"))
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3, logit_atol=LOGIT_ATOL)


@pytest.mark.gpu
def test_fused_tanh_query_model_matches_the_blocked_shifted_model_at_width_64():
    """The degree-sixteen tanh query polynomial (b + a tanh), the pure tanh runs' affine counterpart."""
    assert jax.default_backend() == "gpu"
    blocked, fused = (_source_config("query", attention_backend=backend, attention_polynomial_shift="row_power",
                                     qk_head_dim=WIDTH, value_head_dim=WIDTH)
                      for backend in ("blocked", "fused"))
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3, logit_atol=LOGIT_ATOL)
