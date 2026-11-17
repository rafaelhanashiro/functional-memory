"""Static relative-Taylor SoS: independent square-factor algebra and model ownership."""
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.nonnegative_polynomial import (
    query_square_coefficients, static_square_coefficients, static_square_factors, taylor_square_factors,
)
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_softplus_sos import _config as query_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state

VARIANTS = (("identity", 0), ("softplus", 0), ("softplus", 1))


def _config(core_degree=8, transform="identity", power=0, **changes):
    return query_config(power, core_degree=core_degree,
                        attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
                        attention_coefficient_amplitude_initial=1.,
                        attention_coefficient_projection_initial_std=0., attention_coefficient_kernel_weight_decay=False,
                        attention_static_sos_parameterization="relative_taylor", attention_score_transform=transform,
                        **changes)


def _operation(core_degree, transform, power, backend="blocked"):
    def operation(q, k, v, b):
        factors = static_square_factors(b, core_degree, parameterization="relative_taylor")
        return attention_core(q, k, v, backend=backend, attention_degree=core_degree + power,
                              attention_normalization=False, polynomial_square_factors=factors,
                              square_factor_input_power=power, score_transform=transform,
                              blocked_implementation="strips")
    return operation


def _inputs(core_degree, time=9, batch=2, heads=2, width=8, dtype=jnp.float32):
    rng = np.random.default_rng(412 + core_degree)
    shape = (batch, time, heads, width)
    q, k, v, cotangent = (jnp.asarray(rng.normal(size=shape), dtype) for _ in range(4))
    b = jnp.asarray(1 + .3 * rng.normal(size=(heads, core_degree + 1)), jnp.float32)
    return (q, k, v, b), cotangent


def _numpy_fp64(core_degree, transform, power, operands, cotangent):
    """Independent factors, powers and analytic derivatives, including static gradient reduction."""
    q, k, v, b = (np.asarray(x.astype(jnp.float32), np.float64) for x in operands)
    do = np.asarray(cotangent.astype(jnp.float32), np.float64)
    fixed = np.asarray(taylor_square_factors(core_degree), np.float32).astype(np.float64)
    factors = np.broadcast_to(fixed * (1 + b) / 2, (q.shape[2], core_degree + 1))
    score = np.einsum("bthd,bshd->bhts", q, k, optimize=True) / math.sqrt(q.shape[-1])
    z = np.logaddexp(0, score) if transform == "softplus" else score
    dz = np.exp(-np.logaddexp(0, -score)) if transform == "softplus" else np.ones_like(score)
    split = core_degree // 2 + 1
    parts = (factors[:, :split], factors[:, split:])
    values = [sum(part[:, r][None, :, None, None] * z ** r for r in range(part.shape[-1])) for part in parts]
    slopes = [sum(r * part[:, r][None, :, None, None] * z ** (r - 1)
                  for r in range(1, part.shape[-1])) for part in parts]
    square = values[0] ** 2 + values[1] ** 2
    prefactor = z if power else np.ones_like(z)
    derivative = 2 * prefactor * (values[0] * slopes[0] + values[1] * slopes[1])
    if power:
        derivative += square
    causal = np.tril(np.ones(score.shape[-2:], bool))
    weights = np.where(causal, prefactor * square, 0.)
    dp = np.where(causal, np.einsum("bthd,bshd->bhts", do, v, optimize=True), 0.)
    ds = dp * derivative * dz
    output = np.einsum("bhts,bshd->bthd", weights, v, optimize=True)
    dq = np.einsum("bhts,bshd->bthd", ds, k, optimize=True) / math.sqrt(q.shape[-1])
    dk = np.einsum("bhts,bthd->bshd", ds, q, optimize=True) / math.sqrt(q.shape[-1])
    dv = np.einsum("bhts,bthd->bshd", weights, do, optimize=True)
    df = np.stack([np.sum(dp * prefactor * 2 * value * z ** r, axis=(0, 2, 3))
                   for part, value in zip(parts, values, strict=True) for r in range(part.shape[-1])], axis=-1)
    db = df * fixed / 2
    if b.ndim == 1:
        db = db.sum(0)
    return output, (dq, dk, dv, db)


@pytest.mark.parametrize("core_degree", (8, 12))
@pytest.mark.parametrize("transform,power", VARIANTS)
def test_static_forward_and_all_gradients_match_explicit_squares(core_degree, transform, power):
    operands, cotangent = _inputs(core_degree)
    actual, pullback = jax.vjp(_operation(core_degree, transform, power), *operands)
    expected, expected_grads = _numpy_fp64(core_degree, transform, power, operands, cotangent)
    np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=2e-5)
    for name, got, want in zip("qkvb", pullback(cotangent), expected_grads, strict=True):
        np.testing.assert_allclose(got, want, rtol=8e-5, atol=5e-5, err_msg=name)


@pytest.mark.parametrize("core_degree", (8, 12))
@pytest.mark.parametrize("transform,power", VARIANTS)
def test_static_initial_model_backbone_and_bias_gradient_match_query(core_degree, transform, power):
    static = _config(core_degree, transform, power)
    query = query_config(power, core_degree=core_degree, attention_score_transform=transform)
    first, second = build_model(static.model, 42), build_model(query.model, 42)
    flat = lambda m: {p: np.asarray(v[...]) for p, v in nnx.to_flat_state(nnx.state(m, nnx.Param))}
    a, b = flat(first), flat(second)
    assert a[("polynomial_square_factors",)].shape == (2, 2, core_degree + 1)
    np.testing.assert_array_equal(a[("polynomial_square_factors",)], 1.)
    assert not any(p[0].startswith("query_polynomial_") for p in a)
    for path in set(a) & set(b):
        np.testing.assert_array_equal(a[path], b[path], err_msg=str(path))
    assert model_parameter_count(query.model) - model_parameter_count(static.model) == 2 * 2 * (16 + 1) * (core_degree + 1)
    tokens = jnp.arange(16).reshape(2, 8) % 32
    np.testing.assert_allclose(first(tokens), second(tokens), rtol=1e-4, atol=2e-5)
    # The b gradients have the same f_j / 2 factor as the query generator at a=0.
    operands, cotangent = _inputs(core_degree)
    q, k, v, stored = operands
    def query_op(stored):
        coefficients = query_square_coefficients(jnp.broadcast_to(stored, (*q.shape[:3], core_degree + 1)),
                                                 core_degree + power, input_power=power)
        return attention_core(q, k, v, backend="blocked", attention_degree=core_degree + power,
                              attention_normalization=False, query_polynomial_coefficients=coefficients,
                              attention_coefficient_activation="tanh", score_transform=transform,
                              blocked_implementation="strips")
    static_db = jax.grad(lambda b: jnp.sum(_operation(core_degree, transform, power)(q, k, v, b) * cotangent))(stored)
    query_db = jax.grad(lambda b: jnp.sum(query_op(b) * cotangent))(stored)
    np.testing.assert_allclose(static_db, query_db, rtol=8e-5, atol=5e-5)


@pytest.mark.parametrize("core_degree", (8, 12))
def test_all_three_variants_share_initial_parameters_and_no_factor_decay(core_degree):
    params = []
    for transform, power in VARIANTS:
        config = _config(core_degree, transform, power)
        specs = attention_parameter_specs(config.model)
        names = {s.name for s in specs}
        assert "polynomial_square_factors" in names
        assert not any(n.startswith("query_polynomial_") for n in names)
        mask = attention_weight_decay_mask({n: jnp.ones((2, 2)) for n in names},
                                           excluded_names={s.name for s in specs if not s.weight_decay})
        assert not mask["polynomial_square_factors"]
        model = build_model(config.model, 42)
        params.append({p: np.asarray(v[...]) for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))})
    for variant in params[1:]:
        assert variant.keys() == params[0].keys()
        for path in variant:
            np.testing.assert_array_equal(variant[path], params[0][path])


@pytest.mark.parametrize("core_degree", (8, 12))
@pytest.mark.parametrize("transform,power", VARIANTS)
def test_static_training_updates_factors_and_reports_effective_coefficients(core_degree, transform, power):
    config = _config(core_degree, transform, power)
    state = create_state(config, jax.random.PRNGKey(1))
    tokens = jnp.asarray(np.random.default_rng(9).integers(0, 32, (2, 9)), jnp.int32)
    after, *_ = loop.train_step(state, tokens, jax.random.PRNGKey(0))
    assert np.any(np.asarray(after.params["polynomial_square_factors"][...]) != 1.)
    report = attention_diagnostics(after, config, np.asarray(tokens[:1]), {"selection": "test"})
    assert report["valid"], report["issues"]
    stored = np.asarray(after.params["polynomial_square_factors"][...])
    factors = np.asarray(taylor_square_factors(core_degree), np.float32) * (1 + stored) / 2
    expected = np.zeros((2, 2, core_degree + power + 1))
    split = core_degree // 2 + 1
    for layer in range(2):
        for head in range(2):
            first, second = factors[layer, head, :split], factors[layer, head, split:]
            expected[layer, head, power:] += np.convolve(first, first)
            expected[layer, head, power:-2] += np.convolve(second, second)
    np.testing.assert_allclose(report["coefficients"]["effective"], expected, rtol=2e-6, atol=1e-8)
    identity = attention_identity(config)
    assert identity["attention_coefficient_count"] == 2 * 2 * (core_degree + 1)
    assert identity["attention_coefficient_core_degree"] == core_degree
    assert identity["attention_coefficient_input_power"] == power


@pytest.mark.parametrize("core_degree", (8, 12))
def test_legacy_direct_factors_are_unchanged(core_degree):
    values = jnp.asarray(taylor_square_factors(core_degree))
    np.testing.assert_array_equal(static_square_factors(values, core_degree), values)
    config = _config(core_degree)
    config.model.attention_static_sos_parameterization = "direct"
    validate_config(config)
    spec = next(s for s in attention_parameter_specs(config.model) if s.name == "polynomial_square_factors")
    np.testing.assert_array_equal(spec.initial, taylor_square_factors(core_degree))


@pytest.mark.parametrize("core_degree", (8, 12))
def test_relative_start_and_prefactor_structural_zero(core_degree):
    b = jnp.ones((2, core_degree + 1))
    f = static_square_factors(b, core_degree, parameterization="relative_taylor")
    actual = static_square_coefficients(f, core_degree + 1, input_power=1)
    target = [0.] + [1 / math.factorial(r) for r in range(core_degree + 1)]
    np.testing.assert_allclose(actual, np.broadcast_to(target, actual.shape), rtol=3e-6, atol=0)
    gradient = jax.grad(lambda x: static_square_coefficients(
        static_square_factors(x, core_degree, parameterization="relative_taylor"),
        core_degree + 1, input_power=1)[..., 0].sum())(b)
    np.testing.assert_array_equal(gradient, 0.)


@pytest.mark.parametrize("dtype,time,expected", (("float32", 2048, "blocked"),
                                               ("bfloat16", 8, "blocked"),
                                               ("bfloat16", 2048, "mosaic")))
def test_auto_static_sos_never_falls_back_to_unsupported_fused(dtype, time, expected):
    from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
    config = _config(8, attention_backend="auto", hidden_size=128, qk_head_dim=64, value_head_dim=64,
                     max_seq_len=time)
    assert resolve_attention_backend(config.model, dtype=dtype, platform="gpu", sequence_length=time) == expected
