"""Runtime precision selection and opt-in polynomial GDN gradient checks."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends import polynomial_gdn as backend
from scaffolds.transformer.attention.backends.polynomial_gdn import (
    polynomial_gated_delta_rule,
    resolve_polynomial_gdn_min_ctas,
    resolve_polynomial_gdn_options,
)


def test_runtime_options_default_and_explicit_precedence(monkeypatch):
    monkeypatch.setattr(backend, "_default_num_warps", lambda: 4)
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
    monkeypatch.setattr(backend, "_default_num_warps", lambda: 4)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        resolve_polynomial_gdn_options()
    # An explicitly pinned benchmark remains independent of ambient settings.
    assert resolve_polynomial_gdn_options("tf32x3", 4) == ("tf32x3", 4)


def test_minimum_cta_default_and_explicit_disable(monkeypatch):
    monkeypatch.delenv("POLY_GDN_MIN_CTAS_PER_SM", raising=False)
    assert resolve_polynomial_gdn_min_ctas() == 0
    monkeypatch.setenv("POLY_GDN_MIN_CTAS_PER_SM", "1")
    assert resolve_polynomial_gdn_min_ctas() == 1
    assert resolve_polynomial_gdn_min_ctas(0) == 0
    assert resolve_polynomial_gdn_min_ctas(2) == 2
    monkeypatch.setenv("POLY_GDN_MIN_CTAS_PER_SM", "0")
    assert resolve_polynomial_gdn_min_ctas() == 0


@pytest.mark.parametrize("value", ["bad", "-1", "3", "1.0"])
def test_invalid_minimum_cta_environment_is_rejected(monkeypatch, value):
    monkeypatch.setenv("POLY_GDN_MIN_CTAS_PER_SM", value)
    with pytest.raises(ValueError, match="MIN_CTAS_PER_SM|min_ctas_per_sm"):
        resolve_polynomial_gdn_min_ctas()
    assert resolve_polynomial_gdn_min_ctas(0) == 0


@pytest.mark.parametrize("value", [-1, 3, 1.0, True, "1"])
def test_invalid_explicit_minimum_cta_is_rejected_before_arrays(monkeypatch, value):
    monkeypatch.setattr(backend, "_default_num_warps", lambda: 4)
    with pytest.raises(ValueError, match="min_ctas_per_sm"):
        polynomial_gated_delta_rule(*([None] * 7), min_ctas_per_sm=value)
    with pytest.raises(ValueError, match="min_ctas_per_sm"):
        backend._polynomial_call(None, out=None, q=None, v=None, name="guard",
                                 min_ctas_per_sm=value)


@pytest.mark.parametrize("requested,expected", [(0, None), (1, 1), (2, 2), (None, 1)])
def test_minimum_cta_reaches_forward_and_all_backward_kernels(monkeypatch, requested, expected):
    """Trace the public custom VJP with a fake GPU bridge on the CPU."""
    from scaffolds.transformer.attention.backends import ejkernel_bridge

    monkeypatch.setattr(backend, "_default_num_warps", lambda: 4)
    monkeypatch.setenv("POLY_GDN_MIN_CTAS_PER_SM", "1")
    calls = []

    def fake_triton_call(*args, **kwargs):
        calls.append((kwargs["name"], kwargs["min_ctas_per_sm"]))
        return [jnp.zeros(shape.shape, shape.dtype) for shape in kwargs["out_shape"]]

    monkeypatch.setattr(ejkernel_bridge, "triton_call", fake_triton_call)
    monkeypatch.setattr(backend, "_inverse_forward", lambda k, erase:
                        jnp.zeros((*k.shape[:3], backend.CHUNK_SIZE), jnp.float32))
    q = jnp.ones((1, 64, 1, 16), jnp.float32)
    gate = jnp.zeros(q.shape[:3], jnp.float32)
    beta = jnp.zeros((*q.shape[:3], 4), jnp.float32)

    def objective(*inputs):
        output = polynomial_gated_delta_rule(*inputs, .25, precision="tf32x3", num_warps=4,
                                            min_ctas_per_sm=requested)
        return jnp.sum(output)

    jax.jit(jax.value_and_grad(objective, argnums=tuple(range(6))))(q, q, q, gate, beta, gate)
    assert calls == [("polynomial_gdn::" + name, expected) for name in (
        "forward", "recompute_queries", "backward_queries", "backward_key_values",
        "backward_key_geometry_diagonal", "backward_key_geometry_cross")]


@pytest.mark.parametrize("platform,capability,expected", [
    ("cpu", None, 4), ("gpu", "8.0", 4), ("gpu", "9.0", 4),
    ("gpu", "10.3", 2), ("gpu", "12.0", 2),
])
def test_device_warp_defaults_and_blackwell_launch_guard(monkeypatch, platform, capability, expected):
    monkeypatch.delenv("POLY_GDN_VALUE_PRECISION", raising=False)
    monkeypatch.delenv("POLY_GDN_NUM_WARPS", raising=False)
    monkeypatch.setattr(jax, "local_devices", lambda: [
        SimpleNamespace(platform=platform, compute_capability=capability)])
    assert resolve_polynomial_gdn_options() == ("tf32x3", expected)
    if expected == 2:
        for warps in (1, 2):
            assert resolve_polynomial_gdn_options(num_warps=warps) == ("tf32x3", warps)
        for warps in (4, 8):
            with pytest.raises(ValueError, match="Blackwell"):
                # Reject before inspecting arrays or importing/launching kernels.
                polynomial_gated_delta_rule(*([None] * 7), num_warps=warps)
            with pytest.raises(ValueError, match="Blackwell"):
                backend._polynomial_call(None, out=None, q=None, v=None, name="guard", num_warps=warps)
        monkeypatch.setenv("POLY_GDN_NUM_WARPS", "8")
        with pytest.raises(ValueError, match="Blackwell"):
            resolve_polynomial_gdn_options()
    else:
        assert resolve_polynomial_gdn_options(num_warps=8) == ("tf32x3", 8)


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

    def evaluate(precision, num_warps=None):
        def objective(*inputs):
            output = polynomial_gated_delta_rule(*inputs, .25, precision=precision, num_warps=num_warps)
            return jnp.sum(output.astype(jnp.float32) * do.astype(jnp.float32)), output
        (_, output), gradients = jax.jit(jax.value_and_grad(
            objective, argnums=tuple(range(6)), has_aux=True))(*args)
        return (output, *gradients)

    reference_warps = backend._default_num_warps()
    expected = evaluate("tf32x3", reference_warps)
    for actual, reference in zip(evaluate(None), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    tested_warps = (1, 2) if reference_warps == 2 else (reference_warps,)
    for warps in tested_warps:
        for precision in ("tf32x3", "bf16_read", "bf16_values"):
            for actual, reference in zip(evaluate(precision, warps), expected, strict=True):
                assert actual.dtype == reference.dtype
                actual, reference = (np.asarray(x, np.float32) for x in (actual, reference))
                assert np.all(np.isfinite(actual))
                relative_error = np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-8)
                assert relative_error < .02, (precision, warps, degree, relative_error)
