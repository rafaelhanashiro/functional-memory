"""Structural zero constants: compact parameters, independent algebra, and AdamW.

The reference evaluates positive powers directly, and evaluates SoS as two
explicit square factors. It never uses the production coefficient expansion.
"""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.input_dependent import project_coefficients
from scaffolds.transformer.attention.nonnegative_polynomial import query_square_coefficients, taylor_square_factors
from scaffolds.transformer.attention.parameters import attention_parameter_specs, expand_polynomial_coefficients, factorial_scale
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity
from scaffolds.transformer.training.optim import make_optimizer
from scaffolds.transformer.training.state import create_state

FAMILIES = ("query", "static", "sos")
ORDERS = tuple(range(1, 9))
SOS_INDICES = (1, 2, 3, 4, 6, 7, 8)


def _config(family, **changes):
    query = family != "static"
    sos = family == "sos"
    options = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16,
                   num_heads=2, num_kv_heads=2, qk_head_dim=8, value_head_dim=8,
                   intermediate_size=32, positional_encoding="rope", attention_backend="blocked",
                   attention_blocked_implementation="strips", attention_degree=8,
                   attention_normalization=False, attention_coefficients="learned",
                   attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query" if query else "static",
                   attention_coefficient_activation="tanh" if query else "identity",
                   attention_coefficient_parameterization="sum_of_squares" if sos else "direct",
                   attention_coefficient_zero_constant=True, attention_coefficient_input_power=0,
                   attention_coefficient_amplitude_initial=0. if query else 1., attention_coefficient_projection_initial_std=.25 if query else 0.,
                   attention_coefficient_kernel_weight_decay=query, attention_score_transform="softplus",
                   attention_coefficient_scale="none" if sos else "factorial",
                   attention_coefficient_amplitude_scale="factorial" if family == "query" else "none",
                   attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                   qk_norm_head_scale=True, qk_norm_sharing="per_layer", rms_norm_fp32=True,
                   initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**(options | changes)))
    config.data.batch_size = config.train.micro_batch_size = 2
    config.data.sequence_length, config.data.synthetic_vocab_size = 9, 32
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = .001
    config.optim.weight_decay = .1
    if config.model.attention_backend == "mosaic":
        config.train.dtype = "bfloat16"
    validate_config(config)
    return config


def _coefficients(family, operands):
    q, _, _ = operands[:3]
    if family == "static":
        c = factorial_scale(expand_polynomial_coefficients(operands[3], 8, ORDERS))
        return jnp.broadcast_to(c, (*q.shape[:3], 9))
    x, w, a, b = operands[3:]
    if family == "query":
        return project_coefficients(x, w, factorial_scale(b, ORDERS), amplitude=a,
                                    activation="tanh", amplitude_scale="factorial", orders=ORDERS)
    generated = project_coefficients(x, w, b, amplitude=a, activation="tanh")
    return query_square_coefficients(generated, 8, zero_constant=True)


def _attend(family, backend="blocked"):
    def attend(*operands):
        return attention_core(*operands[:3], backend=backend, blocked_implementation="strips",
                              attention_degree=8, attention_normalization=False,
                              query_polynomial_coefficients=_coefficients(family, operands),
                              attention_coefficient_activation="tanh", score_transform="softplus")
    return attend


def _inputs(family, *, time=9, width=8, dtype=jnp.float32, spread=1., seed=481):
    rng = np.random.default_rng(seed)
    shape = (1, time, 2, width)
    q, k, v = (jnp.asarray(rng.normal(size=shape) * scale, dtype)
               for scale in (spread, 1., 1.))
    cotangent = jnp.asarray(rng.normal(size=shape), dtype)
    count = 7 if family == "sos" else 8
    if family == "static":
        return (q, k, v, jnp.asarray(rng.uniform(-.5, 1.5, (2, count)), jnp.float32)), cotangent
    x = jnp.asarray(rng.normal(size=(1, time, 16)), jnp.float32)
    w = jnp.asarray(rng.normal(size=(2, 16, count)) * .25, jnp.float32)
    a = jnp.asarray(rng.uniform(-.7, .9, (2, count)), jnp.float32)
    b = jnp.asarray(rng.uniform(.6, 1.3, (2, count)), jnp.float32)
    return (q, k, v, x, w, a, b), cotangent


def _numpy_fp64(family, operands, cotangent):
    """Independent dense weights and analytic VJP, with no polynomial convolution."""
    arrays = tuple(np.asarray(value.astype(jnp.float32), np.float64) for value in operands)
    q, k, v = arrays[:3]
    do = np.asarray(cotangent.astype(jnp.float32), np.float64)
    scores = np.einsum("bthd,bshd->bhts", q, k, optimize=True) / math.sqrt(q.shape[-1])
    z = np.logaddexp(0., scores)
    sigmoid = np.exp(-np.logaddexp(0., -scores))
    causal = np.tril(np.ones(scores.shape[-2:], bool))
    dp = np.where(causal, np.einsum("bthd,bshd->bhts", do, v, optimize=True), 0.)
    if family == "static":
        generated = np.broadcast_to(arrays[3], (*q.shape[:3], 8))
    else:
        x, w, a, b = arrays[3:]
        features = np.tanh(np.einsum("btd,hdr->bthr", x, w, optimize=True))
        generated = b + a * features
    generated = generated.transpose(0, 2, 1, 3)
    if family == "sos":
        fixed = np.asarray(taylor_square_factors(8), np.float32).astype(np.float64)[list(SOS_INDICES)]
        factors = fixed * (1. + generated) / 2.
        u = sum(factors[..., r - 1, None] * z ** r for r in range(1, 5))
        vv = sum(factors[..., 3 + r, None] * z ** r for r in range(1, 4))
        du = sum(r * factors[..., r - 1, None] * z ** (r - 1) for r in range(1, 5))
        dv = sum(r * factors[..., 3 + r, None] * z ** (r - 1) for r in range(1, 4))
        weights, derivative = u * u + vv * vv, 2 * (u * du + vv * dv)
        assert np.all(weights >= 0.)
        dg = np.stack([np.sum(dp * value * z ** order, axis=-1)
                       for value, orders in ((u, range(1, 5)), (vv, range(1, 4)))
                       for order in orders], axis=-1) * fixed
    else:
        factorials = np.asarray([math.factorial(r) for r in ORDERS])
        coefficients = generated / factorials
        weights = sum(coefficients[..., r - 1, None] * z ** r for r in ORDERS)
        derivative = sum(r * coefficients[..., r - 1, None] * z ** (r - 1) for r in ORDERS)
        dg = np.stack([np.sum(dp * z ** r, axis=-1) / math.factorial(r) for r in ORDERS], axis=-1)
    weights = np.where(causal, weights, 0.)
    ds = dp * derivative * sigmoid
    output = np.einsum("bhts,bshd->bthd", weights, v, optimize=True)
    dq = np.einsum("bhts,bshd->bthd", ds, k, optimize=True) / math.sqrt(q.shape[-1])
    dk = np.einsum("bhts,bthd->bshd", ds, q, optimize=True) / math.sqrt(q.shape[-1])
    dv = np.einsum("bhts,bthd->bshd", weights, do, optimize=True)
    dg = dg.transpose(0, 2, 1, 3)
    if family == "static":
        return output, (dq, dk, dv, dg.sum(axis=(0, 1)))
    dpre = dg * a * (1. - features ** 2)
    dx = np.einsum("bthr,hdr->btd", dpre, w, optimize=True)
    dw = np.einsum("btd,bthr->hdr", x, dpre, optimize=True)
    da, db = np.sum(dg * features, axis=(0, 1)), np.sum(dg, axis=(0, 1))
    return output, (dq, dk, dv, dx, dw, da, db)


@pytest.mark.parametrize("family", FAMILIES)
def test_compact_attention_and_every_gradient_match_independent_positive_power_reference(family):
    operands, cotangent = _inputs(family)
    actual, backward = jax.vjp(jax.jit(_attend(family)), *operands)
    gradients = backward(cotangent)
    expected, expected_gradients = _numpy_fp64(family, operands, cotangent)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-6)
    for got, want in zip(gradients, expected_gradients, strict=True):
        assert np.linalg.norm(want) > 0
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)
    coefficients = _coefficients(family, operands)
    np.testing.assert_array_equal(coefficients[..., :2 if family == "sos" else 1], 0.)


@pytest.mark.parametrize("family", FAMILIES)
def test_compact_parameter_count_and_initialization_preserve_the_backbone(family):
    config = _config(family)
    parent = replace(config.model, attention_coefficient_zero_constant=False,
                     attention_score_transform="identity")
    model, full = build_model(config.model, 42), build_model(parent, 42)
    mechanism = {s.name for s in attention_parameter_specs(config.model)
                 if s.name.startswith(("query_polynomial_", "polynomial_coefficients"))}
    trainable = dict(nnx.to_flat_state(nnx.state(model, nnx.Param)))
    baseline = dict(nnx.to_flat_state(nnx.state(full, nnx.Param)))
    for path, value in trainable.items():
        if path[0] not in mechanism:
            np.testing.assert_array_equal(value[...], baseline[path][...])
    count = 7 if family == "sos" else 8
    removed = (2 if family == "sos" else 1) * 4 * (18 if family != "static" else 1)
    assert model_parameter_count(parent) - model_parameter_count(config.model) == removed
    selected = SOS_INDICES if family == "sos" else ORDERS
    for name in mechanism:
        assert getattr(model, name).shape[-1] == count
        np.testing.assert_array_equal(getattr(model, name)[...], getattr(full, name)[...][..., list(selected)])
    bias = model.polynomial_coefficients if family == "static" else model.query_polynomial_bias
    np.testing.assert_array_equal(bias[...], 1.)
    if family != "static":
        np.testing.assert_array_equal(model.query_polynomial_amplitude[...], 0.)
    if family == "sos":
        fixed = np.asarray(taylor_square_factors(8), np.float32)
        fixed[[0, 5]] = 0.
        expected = np.convolve(fixed[:5], fixed[:5]) + np.pad(np.convolve(fixed[5:], fixed[5:]), (0, 2))
        got = query_square_coefficients(jnp.ones((2, 7)), 8, zero_constant=True)
        np.testing.assert_allclose(got, np.broadcast_to(expected, got.shape), rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("family", FAMILIES)
def test_model_training_keeps_structural_zeros_and_decays_only_applicable_parameters(family, monkeypatch):
    config = _config(family)
    state = create_state(config, jax.random.PRNGKey(42))
    params = state.params
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    decayed = optax.apply_updates(params, updates)
    for name in ("query_polynomial_bias", "query_polynomial_amplitude", "polynomial_coefficients"):
        if name in params:
            np.testing.assert_array_equal(params[name][...], decayed[name][...])
    if family != "static":
        np.testing.assert_allclose(decayed["query_polynomial_kernel"][...],
                                   params["query_polynomial_kernel"][...] * (1 - .001 * .1), rtol=2e-7)
    batch = jnp.asarray(np.random.default_rng(617).integers(0, 32, (2, 9)), jnp.int32)
    before = {name: np.asarray(value[...]).copy() for name, value in params.items()
              if name.startswith(("query_polynomial_", "polynomial_coefficients"))}
    for step in range(3):
        state, *_ = loop.train_step(state, batch, jax.random.PRNGKey(step))
    for name in before:
        changed = np.asarray(state.params[name][...]) != before[name]
        assert np.all(np.any(changed, axis=tuple(range(changed.ndim - 1)))), name
    report = attention_diagnostics(state, config, np.asarray(batch[:1]), {"selection": "test"})
    assert report["valid"], report["issues"]
    captured = []
    original = multihead.attention_core
    def capture(q, k, v, **kwargs):
        assert kwargs["score_transform"] == "softplus"
        coefficients = kwargs["query_polynomial_coefficients"]
        np.testing.assert_array_equal(coefficients[..., :2 if family == "sos" else 1], 0.)
        assert coefficients.shape[-1] == 9
        if family == "static":
            np.testing.assert_array_equal(coefficients, np.broadcast_to(coefficients[:, :1], coefficients.shape))
        captured.append(coefficients)
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(multihead, "attention_core", capture)
    nnx.merge(state.graphdef, state.params)(batch[:, :-1])
    assert len(captured) == config.model.num_layers
    identity = attention_identity(config)
    assert identity["attention_coefficient_zero_constant"] is True


@pytest.mark.parametrize("family", FAMILIES)
def test_negative_tail_has_first_or_second_order_zero(family):
    operands, _ = _inputs(family, time=2)
    c = np.asarray(_coefficients(family, operands)[0, 0, 0], np.float64)
    leading = 2 if family == "sos" else 1
    scores = np.array([-20., -24.])
    z = np.logaddexp(0., scores)
    values = sum(c[r] * z ** r for r in range(9))
    np.testing.assert_allclose(values / np.exp(leading * scores), c[leading], rtol=2e-7)


@pytest.mark.parametrize("changes", (
    {"attention_coefficient_zero_constant": 1},
    {"attention_coefficient_zero_constant": "true"},
    {"attention_coefficient_input_power": 1, "attention_degree": 9},
    {"attention_score_transform": "sigmoid"},
    {"attention_coefficient_orders": (1, 2, 3, 4, 5, 6, 7, 8)},
    {"attention_coefficient_input_normalization": "l2"},
    {"attention_forget_gate": True},
    {"attention_normalization": True},
))
def test_unsupported_combinations_do_not_silently_ignore_zero_constraint(changes):
    with pytest.raises(ValueError):
        _config("sos", **changes)


@pytest.mark.parametrize("family", FAMILIES)
def test_mosaic_recipe_admits_exact_requested_math(family):
    config = _config(family, attention_backend="mosaic", max_seq_len=2048,
                     hidden_size=128, qk_head_dim=64, value_head_dim=64)
    assert config.model.mosaic_score_transform_supported()
    assert config.model.attention_degree == 8
    assert config.model.attention_coefficient_input_power == 0


def test_legacy_checkpoint_config_keeps_the_full_constant_term():
    from copy import deepcopy
    from scaffolds.transformer.configs import config_to_dict
    from scaffolds.transformer.configs.loader import config_dicts_equal

    current = config_to_dict(_config("query", attention_coefficient_zero_constant=False))
    legacy = deepcopy(current)
    del legacy["model"]["attention_coefficient_zero_constant"]
    assert config_dicts_equal(current, legacy)
    current["model"]["attention_coefficient_zero_constant"] = True
    assert not config_dicts_equal(current, legacy)
