"""Independent FP64 output/VJP qualification of the scalar Triton engine."""
from pathlib import Path
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.basis import ExpProjection, sine, cosine, projection_from_config
from scaffolds.transformer.attention.backends.triton_scalar import (
    basis_kernel, exponential_kernel, polynomial_kernel, scalar_attention, uniform_rbf_kernel,
)
from scaffolds.transformer.configs import load_config


RUNS = {3: "r10b2cb22687c", 5: "r1b486c356e1a", 9: "r27f0d0f59eb1", 20: "r873f76c0a542"}
ROOT = Path(__file__).resolve().parents[3]


def kernel_and_oracle(name):
    recurrence = name.endswith("_recurrence")
    name = name.removesuffix("_recurrence")
    if name.startswith("rbf"):
        cfg = load_config(ROOT / "experiments/attention_scaling/runs" / RUNS[int(name[3:])] / "config.yaml")
        fit = projection_from_config(cfg.model.attention_basis)
    elif name == "sine":
        fit = ExpProjection((sine(.7, .2), cosine(.3)), (1., -.2), 0., 1., 16, 32, 1e-12, {})
    else:
        fit = None
    if fit is not None:
        kernel = uniform_rbf_kernel(fit) if recurrence else basis_kernel(fit)
        def evaluate(s):
            w, dw = np.zeros_like(s), np.zeros_like(s)
            for function, c in zip(fit.basis, fit.coefficients, strict=True):
                p = dict(function.parameters)
                if function.kind == "rbf":
                    z = (s - p["center"]) / p["width"]
                    b = np.exp(-.5 * z*z)
                    db = -z / p["width"] * b
                elif function.kind == "sin":
                    b = np.sin(p["frequency"] * s + p["phase"])
                    db = p["frequency"] * np.cos(p["frequency"] * s + p["phase"])
                else:
                    b = np.cos(p["frequency"] * s + p["phase"])
                    db = -p["frequency"] * np.sin(p["frequency"] * s + p["phase"])
                w += c*b
                dw += c*db
            return w, dw
        return kernel, evaluate
    if name == "exp":
        return exponential_kernel(), lambda s: (np.exp(s), np.exp(s))
    return polynomial_kernel((1., 1., .5)), lambda s: (1+s+.5*s*s, 1+s)


def oracle(q, k, v, gain, go, evaluate, mode, eps):
    q, k, v, gain, go = (np.asarray(x, np.float64) for x in (q, k, v, gain, go))
    scale = q.shape[-1] ** -.5
    s = np.einsum("bihd,bjhd->bhij", q, k) * scale
    w, dw = evaluate(s)
    mask = np.tril(np.ones(s.shape[-2:], bool))
    w, dw = np.where(mask, w, 0.), np.where(mask, dw, 0.)
    z = w.sum(-1, keepdims=True) if mode == "sum" else 1.
    a = w/z
    raw = np.einsum("bhij,bjhd->bihd", a, v)
    if mode == "rms":
        r = 1/np.sqrt(np.mean(raw*raw, -1, keepdims=True) + eps)
        n = raw*r
        y = n*gain
        t = go*gain
        g = (t-n*np.mean(t*n, -1, keepdims=True))*r
        dg = np.sum(go*n, axis=(0, 1))
    else:
        y, g, dg = raw, go, np.zeros_like(gain)
    dp = np.einsum("bihd,bjhd->bhij", g, v)
    if mode == "sum":
        dp = (dp - np.einsum("bihd,bihd->bhi", g, y)[..., None])/z
    ds = dw*dp
    dq = scale*np.einsum("bhij,bjhd->bihd", ds, k)
    dk = scale*np.einsum("bhij,bihd->bjhd", ds, q)
    dv = np.einsum("bhij,bihd->bjhd", a, g)
    return y, dq, dk, dv, dg


@pytest.mark.gpu
@pytest.mark.parametrize("name,mode", [
    (name, mode)
    for name in ("rbf3", "rbf5", "rbf9", "rbf20", "sine", "poly", "exp",
                 "rbf3_recurrence", "rbf5_recurrence", "rbf9_recurrence", "rbf20_recurrence")
    for mode in ("none", "rms", "sum")
    # Signed normalization has explicit negative/zero-sum checks below.
    if (name, mode) != ("sine", "sum")
])
@pytest.mark.parametrize("dtype", (jnp.bfloat16, jnp.float32))
def test_scalar_output_and_all_gradients(name, mode, dtype):
    if jax.default_backend() != "gpu":
        pytest.skip("requires H100")
    kernel, evaluate = kernel_and_oracle(name)
    rng = np.random.default_rng(31)
    shape = (1, 137, 2, 32)  # non-divisible length, multiple query/key tiles
    q, k, v, go = [jnp.asarray(rng.normal(size=shape), dtype) for _ in range(4)]
    gain = jnp.asarray(rng.uniform(.6, 1.4, size=shape[-2:]), jnp.float32)
    def operation(q, k, v, gain):
        return scalar_attention(q, k, v, kernel=kernel, attention_normalization=mode == "sum",
                                output_norm_scale=gain if mode == "rms" else None)
    def joint(q, k, v, gain, go):
        y, pullback = jax.vjp(operation, q, k, v, gain)
        return (y, *pullback(go))
    actual = jax.block_until_ready(jax.jit(joint)(q, k, v, gain, go))
    expected = oracle(q, k, v, gain, go, evaluate, mode, 1e-6)
    for label, observed, reference in zip(("output", "dq", "dk", "dv", "dgain"), actual, expected, strict=True):
        observed = np.asarray(observed, np.float64)
        error = np.linalg.norm(observed-reference)/max(np.linalg.norm(reference), 1e-12)
        assert np.isfinite(observed).all(), (name, mode, label)
        assert error < (4e-2 if dtype == jnp.bfloat16 else 1e-3), (name, mode, dtype, label, error)
    altered = jax.jit(operation)(q, k, v.at[:, -1].add(9.), gain)
    np.testing.assert_array_equal(np.asarray(altered[:, :-1]), np.asarray(actual[0][:, :-1]))


@pytest.mark.gpu
def test_extreme_exp_and_zero_sine_with_exact_epsilon():
    if jax.default_backend() != "gpu":
        pytest.skip("requires H100")
    shape = (1, 128, 1, 32)
    q = jnp.full(shape, math.sqrt(100/math.sqrt(32)), jnp.float32)
    k = q
    v = jnp.asarray(np.random.default_rng(5).normal(size=shape), jnp.float32)
    gain = jnp.ones((1, 32), jnp.float32)
    kernel, evaluate = kernel_and_oracle("exp")
    f = jax.jit(lambda q, k, v, g: scalar_attention(q, k, v, kernel=kernel, output_norm_scale=g))
    y, pullback = jax.vjp(f, q, k, v, gain)
    actual = (y, *pullback(jnp.ones_like(v)))
    expected = oracle(q, k, v, gain, jnp.ones_like(v), evaluate, "rms", 1e-6)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(observed), reference, rtol=2e-3, atol=2e-3)
    tiny_v = v * 1e-30
    y, pullback = jax.vjp(f, q, k, tiny_v, gain)
    actual = (y, *pullback(jnp.ones_like(v)))
    expected = oracle(q, k, tiny_v, gain, jnp.ones_like(v), evaluate, "rms", 1e-6)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(observed), reference, rtol=2e-3, atol=2e-3)
    zero = f(q, k, jnp.zeros_like(v), gain)
    np.testing.assert_array_equal(zero, np.zeros(shape))
    normalized = jax.jit(lambda q, k, v, g: scalar_attention(
        q, k, v, kernel=kernel, attention_normalization=True))
    y, pullback = jax.vjp(normalized, q, -10*k, v, gain)
    actual = (y, *pullback(jnp.ones_like(v)))
    # A fixed global shift leaves normalized exp and its derivative unchanged.
    shifted = lambda s: (np.exp(s+1000), np.exp(s+1000))
    expected = oracle(q, -10*k, v, gain, jnp.ones_like(v), shifted, "sum", 1e-6)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(observed), reference, rtol=2e-3, atol=2e-3)
    fit = ExpProjection((sine(1.),), (1.,), 0., 1., 16, 32, 1e-12, {})
    zero_fn = lambda q: scalar_attention(q, k, v, kernel=basis_kernel(fit), output_norm_scale=gain, eps=.1)
    y, pullback = jax.vjp(jax.jit(zero_fn), jnp.zeros_like(q))
    np.testing.assert_array_equal(np.asarray(y), np.zeros(shape))
    assert np.isfinite(np.asarray(pullback(jnp.ones_like(y))[0])).all()


@pytest.mark.gpu
def test_zero_signed_normalizer_is_explicit_failure():
    if jax.default_backend() != "gpu":
        pytest.skip("requires H100")
    shape = (1, 17, 1, 32)
    x = jnp.zeros(shape, jnp.float32)
    fit = ExpProjection((sine(1.),), (1.,), 0., 1., 16, 32, 1e-12, {})
    y = jax.jit(lambda q: scalar_attention(q, x, jnp.ones_like(x), kernel=basis_kernel(fit),
                                         attention_normalization=True))(x)
    assert np.isnan(np.asarray(y)).all()


def test_model_backend_is_explicit_and_reference_remains_default():
    from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
    cfg = load_config(ROOT / "experiments/attention_scaling/runs" / RUNS[3] / "config.yaml")
    assert resolve_attention_backend(cfg.model, platform="gpu") == "blocked"
    cfg.model.attention_backend = "triton_scalar"
    cfg.model.__post_init__()
    assert resolve_attention_backend(cfg.model, platform="gpu") == "triton_scalar"
    with pytest.raises(ValueError, match="H100"):
        resolve_attention_backend(cfg.model, platform="cpu")


def test_model_connects_gain_to_fused_operation_and_projects_once(monkeypatch):
    from flax import nnx
    from scaffolds.transformer.attention.multihead import MultiHeadAttention
    from scaffolds.transformer.attention.backends import triton_scalar
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
    from scaffolds.transformer.attention.output_scaling import scale_attention_output
    cfg = load_config(ROOT / "experiments/attention_scaling/runs" / RUNS[3] / "config.yaml")
    fit = projection_from_config(cfg.model.attention_basis)
    calls = []
    def fused(q, k, v, *, kernel, output_norm_scale, eps):
        calls.append(kernel)
        raw = blocked_dot_product_attention(q, k, v, score_kernel=fit, attention_normalization=False)
        return scale_attention_output(raw, "head_rms_norm", output_norm_scale=output_norm_scale, eps=eps)
    monkeypatch.setattr(triton_scalar, "scalar_attention", fused)
    options = dict(hidden_size=64, num_heads=2, num_kv_heads=2, qk_head_dim=32, value_head_dim=32,
                   score_kernel=fit, attention_normalization=False, attention_output_scaling="head_rms_norm")
    native = MultiHeadAttention(**options, backend="triton_scalar", rngs=nnx.Rngs(9))
    reference = MultiHeadAttention(**options, backend="blocked", rngs=nnx.Rngs(9))
    x = jnp.asarray(np.random.default_rng(13).normal(size=(1, 7, 64)), jnp.float32)
    parameters = {"output_norm_scale": jnp.arange(64, dtype=jnp.float32).reshape(2, 32)/64 + .5}
    actual, diagnostics = native(x, attention_parameters=parameters, return_diagnostics=True)
    expected = reference(x, attention_parameters=parameters)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert len(calls) == 1
    assert diagnostics is not None
    with pytest.raises(ValueError, match="polynomial"):
        native(x, attention_parameters=parameters, coefficients=jnp.ones((2,)))


def test_native_scalar_dispatch_rejects_conflicting_mathematics():
    from scaffolds.transformer.attention.core import attention_core
    x = jnp.ones((1, 1, 1, 32))
    with pytest.raises(ValueError, match="polynomial options"):
        attention_core(x, x, x, backend="triton_scalar", score_kernel=lambda s: s,
                       attention_degree=2)


def test_model_basis_selection_preserves_fits_and_mixed_functions():
    for count in (3, 20):
        cfg = load_config(ROOT / "experiments/attention_scaling/runs" / RUNS[count] / "config.yaml")
        fit = projection_from_config(cfg.model.attention_basis)
        selected = basis_kernel(fit, implementation="auto")
        expected = basis_kernel(fit) if count == 3 else uniform_rbf_kernel(fit)
        assert selected == expected
        assert basis_kernel(fit).name == "basis"
    mixed = ExpProjection((sine(.7), cosine(.3), sine(.5), cosine(.1), sine(.2)),
                          (1., -.2, .3, .4, .5), 0., 1., 16, 32, 1e-12, {})
    assert basis_kernel(mixed, implementation="auto") == basis_kernel(mixed)
    with pytest.raises(ValueError, match="Gaussian"):
        basis_kernel(mixed, implementation="recurrence")
    with pytest.raises(ValueError, match="implementation"):
        basis_kernel(mixed, implementation="unknown")


@pytest.mark.gpu
def test_negative_signed_normalizer_and_value_gradient():
    if jax.default_backend() != "gpu":
        pytest.skip("requires H100")
    shape = (1, 17, 1, 32)
    x = jnp.ones(shape, jnp.float32)
    v = jnp.asarray(np.random.default_rng(42).normal(size=shape), jnp.float32)
    fit = ExpProjection((cosine(0.),), (-1.,), 0., 1., 16, 32, 1e-12, {})
    f = jax.jit(lambda q, k, v: scalar_attention(q, k, v, kernel=basis_kernel(fit),
                                                attention_normalization=True))
    y, vjp = jax.vjp(f, x, x, v)
    dq, dk, dv = vjp(jnp.ones_like(y))
    positions = np.arange(1, 18)
    expected = np.cumsum(np.asarray(v), axis=1)/positions[None, :, None, None]
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(dq, np.zeros(shape))
    np.testing.assert_array_equal(dk, np.zeros(shape))
    expected_dv = np.broadcast_to(np.cumsum((1/positions)[::-1])[::-1][None, :, None, None], shape)
    np.testing.assert_allclose(dv, expected_dv, rtol=1e-5, atol=1e-5)
