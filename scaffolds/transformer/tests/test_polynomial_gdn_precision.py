"""Runtime precision selection and opt-in polynomial GDN gradient checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.polynomial_gdn import (
    polynomial_gated_delta_rule,
    resolve_polynomial_gdn_options,
)


def test_runtime_options_default_and_explicit_precedence(monkeypatch):
    monkeypatch.delenv("POLY_GDN_VALUE_PRECISION", raising=False)
    monkeypatch.delenv("POLY_GDN_NUM_WARPS", raising=False)
    assert resolve_polynomial_gdn_options() == ("tf32x3", 4)
    monkeypatch.setenv("POLY_GDN_VALUE_PRECISION", "bf16_read")
    monkeypatch.setenv("POLY_GDN_NUM_WARPS", "8")
    assert resolve_polynomial_gdn_options() == ("bf16_read", 8)
    assert resolve_polynomial_gdn_options("tf32x3", 4) == ("tf32x3", 4)


@pytest.mark.parametrize("name,value", [
    ("POLY_GDN_VALUE_PRECISION", "bf16"),
    ("POLY_GDN_NUM_WARPS", "not-an-integer"),
    ("POLY_GDN_NUM_WARPS", "3"),
])
def test_invalid_runtime_options_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        resolve_polynomial_gdn_options()
    # An explicitly pinned benchmark remains independent of ambient settings.
    assert resolve_polynomial_gdn_options("tf32x3", 4) == ("tf32x3", 4)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", [2, 3])
def test_opt_in_modes_keep_finite_bf16_outputs_and_all_six_gradients(degree, monkeypatch):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")
    monkeypatch.delenv("POLY_GDN_VALUE_PRECISION", raising=False)
    monkeypatch.delenv("POLY_GDN_NUM_WARPS", raising=False)
    rng = np.random.default_rng(73)
    shape = (1, 128, 1, 16)
    q, k = (rng.standard_normal(shape, dtype=np.float32) for _ in range(2))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q, k, v, do = (jnp.asarray(x, jnp.bfloat16) for x in (
        q, k, rng.standard_normal(shape, dtype=np.float32),
        rng.standard_normal(shape, dtype=np.float32)))
    gates = shape[:3]
    args = (q, k, v, jnp.full(gates, -.02, jnp.float32),
            jnp.asarray(rng.normal(0, .2, (*gates, degree + 1)), jnp.float32),
            jnp.asarray(rng.uniform(.1, .9, gates), jnp.float32))

    def evaluate(precision):
        def objective(*inputs):
            output = polynomial_gated_delta_rule(*inputs, .25, precision=precision)
            return jnp.sum(output.astype(jnp.float32) * do.astype(jnp.float32)), output
        (_, output), gradients = jax.jit(jax.value_and_grad(
            objective, argnums=tuple(range(6)), has_aux=True))(*args)
        return (output, *gradients)

    expected = evaluate("tf32x3")
    for actual, reference in zip(evaluate(None), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    for precision in ("bf16_read", "bf16_values"):
        for actual, reference in zip(evaluate(precision), expected, strict=True):
            assert actual.dtype == reference.dtype
            actual, reference = (np.asarray(x, np.float32) for x in (actual, reference))
            assert np.all(np.isfinite(actual))
            relative_error = np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-8)
            assert relative_error < .02, (precision, degree, relative_error)
