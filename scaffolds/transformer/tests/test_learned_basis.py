"""Learned scalar bases: independent values/gradients, model ownership and initialization."""
from dataclasses import replace
from functools import lru_cache, partial

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.basis import fit_exp, rbf
from scaffolds.transformer.attention.backends.triton_scalar import basis_kernel, scalar_attention
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.input_dependent import project_coefficients
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.attention.parameters import attention_parameter_specs, initial_attention_parameter_values
from scaffolds.transformer.configs import ModelConfig, Config, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer import acceptance


@lru_cache
def projection(count=9):
    return fit_exp([rbf(float(c), 1.) for c in np.linspace(-6., 6., count)], quadrature_order=128, check_order=256)


def dense(q, k, v, gain, c, *, count):
    # Independent Gaussian evaluation; do not use production basis or normalization helpers.
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    c = jnp.broadcast_to(c, (*q.shape[:3], count))
    scores = jnp.einsum("bihd,bjhd->bhij", q, k, precision=jax.lax.Precision.HIGHEST) / np.sqrt(q.shape[-1])
    weights = jnp.zeros_like(scores)
    for i, center in enumerate(np.linspace(-6., 6., count)):
        weights += jnp.swapaxes(c[..., i], 1, 2)[..., None] * jnp.exp(-0.5 * (scores-center)**2)
    weights = jnp.where(jnp.tril(jnp.ones(scores.shape[-2:], bool)), weights, 0.)
    y = jnp.einsum("bhij,bjhd->bihd", weights, v, precision=jax.lax.Precision.HIGHEST)
    return y * jax.lax.rsqrt(jnp.mean(y*y, -1, keepdims=True) + 1e-6) * gain


def operation(q, k, v, gain, c, *, count, backend):
    fit = projection(count)
    if backend == "triton_scalar":
        return scalar_attention(q, k, v, kernel=basis_kernel(fit), basis_coefficients=c, output_norm_scale=gain)
    y = attention_core(q, k, v, score_kernel=fit, basis_coefficients=c,
                       attention_normalization=False, backend="blocked")
    return scale_attention_output(y, "head_rms_norm", output_norm_scale=gain)


def check_operator(backend, dtype, count, sharing, *, zero=False):
    projection(count)  # Fit outside JAX tracing.
    rng = np.random.default_rng(352)
    shape = (2, 137, 2, 32) if backend == "triton_scalar" else (2, 17, 2, 8)
    q, k, v, go = [jnp.asarray(rng.normal(size=shape), dtype) for _ in range(4)]
    gain = jnp.asarray(rng.uniform(.5, 1.5, shape[-2:]), jnp.float32)
    cshape = {"global": (count,), "head": (shape[2], count), "query": (*shape[:3], count)}[sharing]
    c = jnp.asarray(rng.normal(size=cshape), jnp.float32)
    if zero:
        c = jnp.zeros_like(c)
    inputs = q, k, v, gain, c
    def joint(fn, *args):
        out, pullback = jax.vjp(fn, *args)
        return out, pullback(go.astype(out.dtype))
    expected = jax.jit(partial(joint, partial(dense, count=count)))(*inputs)
    actual = jax.jit(partial(joint, partial(operation, count=count, backend=backend)))(*inputs)
    tolerance = 0.045 if dtype == jnp.bfloat16 else 0.002
    for label, a, b in zip(("out", "dq", "dk", "dv", "gain", "coefficients"),
                          jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        assert np.isfinite(a).all(), label
        error = np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-8)
        if label == "coefficients":
            tolerance = .002
        assert error < tolerance, (label, sharing, count, error)
        if label == "coefficients":
            # A large central coefficient must not hide a missing outer-basis gradient.
            for i in range(count):
                error = np.linalg.norm(a[..., i]-b[..., i]) / max(np.linalg.norm(b[..., i]), 1e-8)
                assert error < 2*tolerance, (label, i, error)


@pytest.mark.parametrize("sharing", ["global", "head", "query"])
def test_blocked_learned_basis_against_dense(sharing):
    check_operator("blocked", jnp.float32, 9, sharing)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("count", [3, 9])
@pytest.mark.parametrize("sharing", ["global", "head", "query"])
def test_triton_learned_basis_all_gradients(dtype, count, sharing):
    if jax.default_backend() != "gpu":
        pytest.skip("requires H100")
    check_operator("triton_scalar", dtype, count, sharing)


@pytest.mark.gpu
def test_zero_basis_coefficients_have_nonzero_coefficient_gradient():
    if jax.default_backend() != "gpu":
        pytest.skip("requires H100")
    check_operator("triton_scalar", jnp.float32, 9, "query", zero=True)


def model_config(conditioning):
    m = ModelConfig(vocab_size=16, max_seq_len=17, num_layers=2, hidden_size=16, num_heads=2,
                    num_kv_heads=2, qk_head_dim=8, value_head_dim=8, intermediate_size=24,
                    attention_basis=projection().to_config(), attention_normalization=False,
                    attention_coefficients="learned", attention_coefficient_sharing="per_head",
                    attention_coefficient_initialization="projection", attention_backend="blocked",
                    initialization="transformer_reference", attention_output_scaling="head_rms_norm", qk_norm=True)
    if conditioning == "query":
        m = replace(m, attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                    attention_coefficient_amplitude_initial=0., attention_coefficient_projection_initial_std=.25)
    return m


@pytest.mark.parametrize("conditioning", ["static", "query"])
def test_learned_basis_model_starts_at_fixed_fit(conditioning):
    m = model_config(conditioning)
    fixed = replace(m, attention_coefficients="fixed", attention_coefficient_conditioning="static",
                    attention_coefficient_activation="identity", attention_coefficient_initialization="taylor",
                    attention_coefficient_amplitude_initial=1., attention_coefficient_projection_initial_std=0.)
    learned_model = build_model(m, rngs=nnx.Rngs(9))
    fixed_model = build_model(fixed, rngs=nnx.Rngs(9))
    tokens = jnp.arange(14).reshape(2, 7) % 16
    np.testing.assert_allclose(learned_model(tokens), fixed_model(tokens), rtol=3e-5, atol=3e-5)
    identity = attention_identity(Config(model=m))
    assert identity["attention_coefficients"] == "learned"
    assert identity["attention_kernel"] == ("query_basis_unnormalized" if conditioning == "query" else "learned_basis_unnormalized")
    specs = attention_parameter_specs(m)
    assert all(not spec.weight_decay for spec in specs)
    paths = acceptance.test_paths(acceptance.AcceptanceSpec(entry_module="test", cpu_tests=()), config_to_dict(Config(model=m)),
                                  resolved_backend="triton_scalar", cpu=False)
    assert "scaffolds/transformer/tests/test_learned_basis.py" in paths


def test_input_dependent_basis_bias_amplitude_and_projection_gradients():
    m = model_config("query")
    values = initial_attention_parameter_values(attention_parameter_specs(m), num_layers=2, num_heads=2,
                                                key=jax.random.PRNGKey(8))
    b, w, a = (values[f"query_basis_{name}"][0] for name in ("bias", "kernel", "amplitude"))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 16))
    def loss(b, w, a):
        c = project_coefficients(x, w, b, amplitude=a, activation="tanh")
        return jnp.sum(c * jnp.arange(1, 10))
    db, dw, da = jax.grad(loss, (0, 1, 2))(b, w, a)
    assert jnp.linalg.norm(db) > 0 and jnp.linalg.norm(da) > 0
    np.testing.assert_array_equal(dw, jnp.zeros_like(dw))
    _, dw, _ = jax.grad(loss, (0, 1, 2))(b, w, a-.01*da)
    assert jnp.linalg.norm(dw) > 0


@pytest.mark.gpu
def test_query_basis_end_to_end_projection_gradients():
    if jax.default_backend() != "gpu":
        pytest.skip("requires H100")
    projection(9)
    rng = np.random.default_rng(224)
    q, k, v, upstream = [jnp.asarray(rng.normal(size=(1, 137, 2, 32)), jnp.bfloat16) for _ in range(4)]
    x = jnp.asarray(rng.normal(size=(1, 137, 16)), jnp.float32)
    w = jnp.asarray(rng.normal(size=(2, 16, 9)) * .25, jnp.float32)
    b = jnp.asarray(rng.normal(size=(2, 9)), jnp.float32)
    a = jnp.asarray(rng.uniform(.1, .5, size=(2, 9)), jnp.float32)
    gain = jnp.ones((2, 32), jnp.float32)
    def loss(x, w, b, a, native):
        c = project_coefficients(x, w, b, amplitude=a, activation="tanh")
        fn = partial(operation, count=9, backend="triton_scalar") if native else partial(dense, count=9)
        y = fn(q, k, v, gain, c)
        return jnp.sum(y.astype(jnp.float32) * upstream.astype(jnp.float32))
    actual = jax.jit(jax.grad(partial(loss, native=True), (0, 1, 2, 3)))(x, w, b, a)
    expected = jax.jit(jax.grad(partial(loss, native=False), (0, 1, 2, 3)))(x, w, b, a)
    for name, got, want in zip(("input", "projection", "bias", "amplitude"), actual, expected, strict=True):
        assert jnp.isfinite(got).all() and jnp.linalg.norm(got) > 0, name
        assert jnp.linalg.norm(got-want) / jnp.linalg.norm(want) < .045, name
