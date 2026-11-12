"""Compact polynomial bases preserve missing powers through forward/backward and updates."""

from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.parameters import attention_parameter_specs, expand_polynomial_coefficients, factorial_scale
from scaffolds.transformer.attention.input_dependent import project_coefficients
from scaffolds.transformer.attention.unnormalized import unnormalized_attention
from scaffolds.transformer.configs import Config, config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_coefficient_scale import _model, _small_config, TOKENS
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.regularization import coefficient_penalty_sum
from scaffolds.transformer.training.state import create_state


# Even powers through degree eight, and the intercept with odd powers through degree seven.
ORDER_SETS = {"even": (0, 2, 4, 6, 8), "odd": (0, 1, 3, 5, 7)}
ORDERS = ORDER_SETS["even"]
each_order_set = pytest.mark.parametrize("orders", list(ORDER_SETS.values()), ids=list(ORDER_SETS))


def _factorials(orders):
    return np.asarray([math.factorial(order) for order in orders], np.float32)


def _omitted(orders):
    """Powers below the highest retained one, which the compact basis fixes at zero."""
    return [order for order in range(orders[-1] + 1) if order not in orders]


def _parity(orders):
    """1 if p(z) - p(0) is even (only even powers), -1 if it is odd (an intercept plus odd powers)."""
    nonconstant = {order % 2 for order in orders if order}
    assert len(nonconstant) == 1
    return -1 if nonconstant.pop() else 1


def _config(orders=ORDERS, **updates):
    return _model(orders[-1], attention_coefficient_orders=orders, **updates)


@each_order_set
@pytest.mark.parametrize("scale", ["none", "factorial"])
@pytest.mark.parametrize("sharing", ["global", "per_head"])
def test_compact_initialization_and_model_gradients_match_zeroed_full_model(orders, scale, sharing):
    config = _config(orders, attention_coefficient_scale=scale, attention_coefficient_sharing=sharing)
    compact = build_model(config, 13)
    full_config = replace(config, attention_coefficient_orders=None)
    full = build_model(full_config, 13)
    expected = np.ones(len(orders), np.float32) if scale == "factorial" else 1 / _factorials(orders)
    np.testing.assert_allclose(compact.polynomial_coefficients[...],
                               np.broadcast_to(expected, compact.polynomial_coefficients.shape))
    assert compact.polynomial_coefficients.shape == ((len(orders),) if sharing == "global" else (2, 2, len(orders)))
    omitted = _omitted(orders)
    assert (model_parameter_count(full_config) - model_parameter_count(config)
            == len(omitted) * (1 if sharing == "global" else 4))
    full.polynomial_coefficients[...] = full.polynomial_coefficients[...].at[..., jnp.asarray(omitted)].set(0)
    # Same backbone and selected stored coefficients must give the same model and derivatives.
    np.testing.assert_allclose(compact(TOKENS), full(TOKENS), rtol=1e-6, atol=1e-7)
    grads = []
    for model in (compact, full):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        grads.append(jax.grad(lambda p: jnp.mean(nnx.merge(graphdef, p, rest)(TOKENS) ** 2))(params))
    for (path, got), (_, want) in zip(nnx.to_flat_state(grads[0]), nnx.to_flat_state(grads[1])):
        expected = want[...][..., list(orders)] if path == ("polynomial_coefficients",) else want[...]
        np.testing.assert_allclose(got[...], expected, rtol=3e-5, atol=1e-8)


@each_order_set
def test_optimizer_updates_only_selected_orders_and_polynomial_keeps_parity(orders):
    degree, factorials = orders[-1], _factorials(orders)
    values = jnp.ones((2, len(orders)), jnp.float32)
    optimizer = optax.adam(1e-2)
    state = optimizer.init(values)
    scores = jnp.linspace(-2.0, 2.0, 37)
    # The target has both parities, so every retained power gets a gradient and neither basis can fit it.
    target = 1 + scores + scores ** 2

    def polynomial(stored, x):
        c = factorial_scale(expand_polynomial_coefficients(stored, degree, orders))
        return jnp.sum(c[..., None, :] * x[:, None] ** jnp.arange(degree + 1), axis=-1)

    def loss(stored):
        return jnp.mean((polynomial(stored, scores) - target) ** 2)

    grad = jax.grad(loss)(values)
    # Explicit derivative uses the actual degree's factorial, not compact-index factorial.
    error = np.asarray(polynomial(values, scores) - target)
    expected = 2 * np.mean(error[..., None] * np.asarray(scores)[None, :, None] ** np.asarray(orders)
                           / factorials, axis=1) / values.shape[0]
    np.testing.assert_allclose(grad, expected, rtol=2e-6, atol=2e-6)
    for _ in range(3):
        updates, state = optimizer.update(jax.grad(loss)(values), state, values)
        values = optax.apply_updates(values, updates)
    assert np.all(np.asarray(values) != 1)
    expanded = factorial_scale(expand_polynomial_coefficients(values, degree, orders))
    np.testing.assert_array_equal(expanded[..., _omitted(orders)], 0)
    zero = polynomial(values, jnp.zeros_like(scores))
    np.testing.assert_allclose(polynomial(values, scores) - zero,
                               _parity(orders) * (polynomial(values, -scores) - zero), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("orders", [[], [0, 2, 2, 8], [2, 0, 8], [-2, 0, 8], [0, 9], [0, 2, 4],
                                    [False, 2, 8], [0.0, 2, 8], "0,2,8"])
def test_invalid_order_sets_rejected(orders):
    with pytest.raises(ValueError, match="attention_coefficient_orders"):
        replace(_config(), attention_coefficient_orders=orders)


@pytest.mark.parametrize("updates", [
    {"attention_coefficients": "fixed"},
    {"attention_coefficient_conditioning": "query_key", "attention_coefficient_activation": "tanh"},
    {"attention_coefficient_parameterization": "sum_of_squares", "attention_normalization": True},
])
def test_order_selection_restricted_to_supported_direct_generators(updates):
    with pytest.raises(ValueError, match="Selected coefficient orders"):
        _config(**updates)


@each_order_set
def test_identity_diagnostics_acceptance_and_legacy_config_compatibility(orders):
    config = _small_config(_config(orders, attention_coefficient_scale="factorial"))
    identity = attention_identity(config)
    assert identity["attention_coefficient_orders"] == list(orders)
    assert identity["attention_coefficient_count"] == 2 * 2 * len(orders)
    assert "omitted orders fixed at zero" in identity["attention_formula"]
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    coeff = report["coefficients"]
    assert coeff["raw_parameter_names"] == [f"c{r}" for r in orders]
    np.testing.assert_array_equal(np.asarray(coeff["effective"])[..., _omitted(orders)], 0)
    np.testing.assert_allclose(np.asarray(coeff["effective"])[..., list(orders)],
                               np.broadcast_to(1 / _factorials(orders), (2, 2, len(orders))))
    facts = acceptance.matches_softmax_backbone_initialization(config)
    assert facts["mechanism_parameters"]["polynomial_coefficients"]["initial"] == [1.0] * len(orders)
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (True, False):
        assert "scaffolds/transformer/tests/test_coefficient_orders.py" in acceptance.test_paths(
            spec, config_to_dict(config), resolved_backend="mosaic", cpu=cpu)
    current = config_to_dict(Config(model=_model()))
    legacy = config_to_dict(Config(model=_model()))
    del legacy["model"]["attention_coefficient_orders"]
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(config_to_dict(config), current)


@each_order_set
def test_unscaled_penalty_uses_actual_orders(orders):
    config = _config(orders)
    model = build_model(config, 13)
    params = nnx.state(model, nnx.Param)
    # a_0^2 + sum_(r>=2) r^2 a_r^2: the linear coefficient is not penalized.
    weights = np.asarray([1.0 if order == 0 else 0.0 if order == 1 else order ** 2 for order in orders])
    expected = 4 * np.sum(weights / _factorials(orders) ** 2)
    np.testing.assert_allclose(coefficient_penalty_sum(params, config), expected, rtol=1e-6)


@each_order_set
def test_query_compact_model_and_all_gradients_match_zeroed_full_model(orders):
    degree = orders[-1]
    config = _model(degree, "query", attention_coefficient_orders=orders,
                    attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
                    attention_coefficient_amplitude_initial=0.7, attention_coefficient_kernel_weight_decay=True)
    compact = build_model(config, 13)
    full = build_model(replace(config, attention_coefficient_orders=None), 13)
    query_names = ("query_polynomial_kernel", "query_polynomial_amplitude", "query_polynomial_bias")
    for name in query_names:
        values = getattr(compact, name)[...]
        assert values.shape[-1] == len(orders)
        getattr(full, name)[...] = expand_polynomial_coefficients(values, degree, orders)
    assert compact.query_polynomial_kernel.shape == (2, 2, 8, len(orders))
    np.testing.assert_array_equal(compact.query_polynomial_bias[...], 1)
    specs = {s.name: s for s in attention_parameter_specs(config)}
    assert specs["query_polynomial_kernel"].weight_decay
    assert not specs["query_polynomial_amplitude"].weight_decay
    assert not specs["query_polynomial_bias"].weight_decay
    np.testing.assert_allclose(compact(TOKENS), full(TOKENS), rtol=2e-5, atol=2e-6)
    grads = []
    for model in (compact, full):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        grads.append(jax.grad(lambda p: jnp.mean(nnx.merge(graphdef, p, rest)(TOKENS) ** 2))(params))
    for (path, got), (_, want) in zip(nnx.to_flat_state(grads[0]), nnx.to_flat_state(grads[1])):
        expected = want[...][..., list(orders)] if path[0] in query_names else want[...]
        np.testing.assert_allclose(got[...], expected, rtol=3e-4, atol=1e-8)
    identity = attention_identity(Config(model=config))
    assert identity["attention_coefficient_count"] == 2 * 2 * len(orders) * (8 + 2)
    assert "omitted orders fixed at zero" in identity["attention_formula"]


@each_order_set
def test_query_coefficients_use_actual_factorials_and_keep_omitted_orders_zero_after_adamw(orders):
    factorials = _factorials(orders)
    x = jnp.asarray(np.random.default_rng(4).normal(size=(2, 7, 3)), jnp.float32)
    values = {"w": jnp.asarray(np.random.default_rng(5).normal(size=(2, 3, len(orders))), jnp.float32),
              "b": jnp.ones((2, len(orders))), "a": jnp.full((2, len(orders)), 0.7)}
    def coefficients(p):
        return project_coefficients(x, p["w"], factorial_scale(p["b"], orders), amplitude=p["a"],
                                    activation="tanh", amplitude_scale="factorial", orders=orders)
    def independent(p):
        return (p["b"] + p["a"] * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, p["w"]))) / factorials
    np.testing.assert_allclose(coefficients(values)[..., list(orders)], independent(values), rtol=2e-6, atol=1e-7)
    got = jax.grad(lambda p: jnp.mean(coefficients(p) ** 2) * (orders[-1] + 1) / len(orders))(values)
    want = jax.grad(lambda p: jnp.mean(independent(p) ** 2))(values)
    for key in values:
        np.testing.assert_allclose(got[key], want[key], rtol=2e-5, atol=1e-8)
    optimizer = optax.adamw(1e-2, weight_decay=0.1, mask={"w": True, "a": False, "b": False})
    state = optimizer.init(values)
    for _ in range(3):
        updates, state = optimizer.update(jax.grad(lambda p: jnp.mean(coefficients(p) ** 2))(values), state, values)
        values = optax.apply_updates(values, updates)
    np.testing.assert_array_equal(coefficients(values)[..., _omitted(orders)], 0)
    assert np.any(np.asarray(jnp.std(coefficients(values), axis=1)) > 0)


@each_order_set
def test_query_diagnostics_report_retained_orders_and_actual_factorials(orders):
    factorials = _factorials(orders)
    config = _small_config(_model(orders[-1], "query", attention_coefficient_orders=orders,
                           attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
                           attention_coefficient_amplitude_initial=0.7))
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    coeff = report["coefficients"]
    assert coeff["stored_parameter_orders"] == list(orders)
    np.testing.assert_allclose(coeff["effective_projection_bias"], np.broadcast_to(1 / factorials, (2, 2, len(orders))))
    np.testing.assert_allclose(coeff["effective_modulation_amplitude"],
                               np.broadcast_to(0.7 / factorials, (2, 2, len(orders))))
    np.testing.assert_array_equal(np.asarray(coeff["effective_summary"]["mean"])[..., _omitted(orders)], 0)


@pytest.mark.gpu
@each_order_set
def test_compact_coefficients_mosaic_forward_and_backward_match_fp64(orders):
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")
    from scaffolds.transformer.tests.test_mosaic_unnormalized import _inputs, _fp64, _assert_all

    degree, factorials = orders[-1], _factorials(orders)
    operands, full, cotangent = _inputs(degree, 64, "per_head", time=2048, heads=2)
    stored = full[..., list(orders)] * factorials

    def attend(q, k, v, values):
        coeff = factorial_scale(expand_polynomial_coefficients(values, degree, orders))
        return unnormalized_attention(q, k, v, attention_degree=degree, backend="mosaic", polynomial_coefficients=coeff)

    output, backward = jax.vjp(jax.jit(attend), *operands, stored)
    actual = output, backward(cotangent)
    coeff = factorial_scale(expand_polynomial_coefficients(stored, degree, orders))
    expected_output, expected_grad = _fp64(operands, coeff, cotangent, degree)
    expected = expected_output, (*expected_grad[:3], expected_grad[3][..., list(orders)] / factorials)
    _assert_all(actual, expected, f"orders {list(orders)} compact Mosaic vs FP64")


@pytest.mark.gpu
@each_order_set
def test_compact_query_coefficients_mosaic_forward_and_backward_match_fp64(orders):
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")
    from scaffolds.transformer.tests.test_mosaic_query import _inputs, _numpy_fp64, _assert_all
    from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention

    degree, factorials = orders[-1], _factorials(orders)
    operands, cotangent = _inputs(degree, 64, time=2048, heads=2)
    stored = operands[3][..., list(orders)] * factorials
    def attend(q, k, v, values):
        coeff = factorial_scale(expand_polynomial_coefficients(values, degree, orders))
        return query_polynomial_attention(q, k, v, coeff)
    output, backward = jax.vjp(jax.jit(attend), *operands[:3], stored)
    actual = output, backward(cotangent)
    coeff = factorial_scale(expand_polynomial_coefficients(stored, degree, orders))
    expected_output, expected_grad = _numpy_fp64((*operands[:3], coeff), cotangent)
    expected = expected_output, (*expected_grad[:3], expected_grad[3][..., list(orders)] / factorials)
    _assert_all(actual, expected, f"orders {list(orders)} compact query Mosaic vs FP64")
