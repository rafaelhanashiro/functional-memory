"""Per-order endpoint gates: explicit mathematics, gradients, and matched starts."""
from dataclasses import replace
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import forget_gate
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs.base import validate_config
from scaffolds.transformer.configs import config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_endpoint_decay import _config, _params
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.diagnostics import attention_identity


def inputs(time=9, degree=4, *, width=3, value_width=2):
    rng = np.random.default_rng(12)
    q, k = (rng.normal(size=(2, time, 2, width)) * .4 for _ in range(2))
    v = rng.normal(size=(2, time, 2, value_width))
    c = rng.normal(size=(2, time, 2, degree + 1))
    x = rng.normal(size=(2, time, 7))
    wq, wk = (rng.normal(size=(2, degree + 1, 7)) * .2 for _ in range(2))
    rate = rng.uniform(-1, 1, size=(2, degree + 1, 1))
    bias = rng.uniform(-4, -1, size=rate.shape)
    return tuple(jnp.asarray(a, jnp.float32) for a in (q, k, v, c, x, wq, wk, rate, bias))


def operation(q, k, v, c, x, wq, wk, rate, bias, *, endpoint, diagnostics=False):
    if endpoint:
        query, key = forget_gate.project_gate_inputs(x, wq, bias, wk)
        gate = dict(query_gate_input=query, key_gate_input=key, gate_log_rate=rate)
    else:
        gate = dict(query_log_decay=forget_gate.log_decay(x, wq, rate, bias))
    return blocked_dot_product_attention(q, k, v, attention_degree=c.shape[-1]-1,
        attention_normalization=False, query_polynomial_coefficients=c,
        query_decay_mode="query_distance", query_block_size=128 if q.shape[1] > 128 else 4,
        return_diagnostics=diagnostics, **gate)


def dense(q, k, v, c, x, wq, wk, rate, bias, *, endpoint):
    # Explicit powers and all pairs, independently from the blocked Horner implementation.
    score = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    query = jnp.einsum("btd,hrd->bhtr", x, wq) + bias[..., 0][None, :, None, :]
    z = query[:, :, :, None, :]
    if endpoint:
        key = jnp.einsum("bid,hrd->bhir", x, wk)
        z = z + key[:, :, None, :, :]
    lam = jnp.exp(rate[..., 0])[None, :, None, None, :] * jnp.logaddexp(0., z)
    distance = jnp.arange(q.shape[1])[:, None] - jnp.arange(q.shape[1])[None, :]
    weights = sum(c[:, :, :, r].transpose(0, 2, 1)[..., None] * score**r
                  * jnp.exp(-jnp.maximum(distance, 0) * lam[..., r]) for r in range(c.shape[-1]))
    weights = jnp.where(distance >= 0, weights, 0.)
    return jnp.einsum("bhti,bihd->bthd", weights, v)


@pytest.mark.parametrize("endpoint", (False, True))
@pytest.mark.parametrize("time,degree", ((1, 1), (9, 4), (13, 6)))
def test_explicit_output_and_every_parameter_gradient(endpoint, time, degree):
    arrays = inputs(time, degree)
    cotangent = jnp.arange(arrays[2].size).reshape(arrays[2].shape) / arrays[2].size
    def evaluate(fn, args):
        out, backward = jax.vjp(fn, *args)
        return out, backward(cotangent.astype(out.dtype))
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = evaluate(partial(dense, endpoint=endpoint), tuple(a.astype(jnp.float64) for a in arrays))
        expected = jax.tree.map(np.asarray, expected)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: evaluate(partial(operation, endpoint=endpoint), a))(*arrays)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, rtol=8e-5, atol=2e-6)
    if time > 1:
        grad_wq = actual[1][5]
        assert np.all(np.linalg.norm(grad_wq, axis=-1) > 0)
        assert not np.allclose(grad_wq[:, 0], grad_wq[:, 1])
        if endpoint:
            assert np.all(np.linalg.norm(actual[1][6], axis=-1) > 0)


@pytest.mark.parametrize("conditioning", ("query", "query_key"))
def test_model_initialization_matches_control_and_decay_masks(conditioning):
    base = _config(attention_forget_gate_conditioning=conditioning)
    config = replace(base, model=replace(base.model, attention_forget_gate_per_order=True))
    validate_config(config)
    control, model = build_model(base.model, 42), build_model(config.model, 42)
    old, new = _params(control), _params(model)
    for path, value in old.items():
        if path[0].startswith("forget_gate_"):
            expected = np.broadcast_to(np.asarray(value)[:, :, None, :], new[path].shape)
            np.testing.assert_array_equal(new[path], expected)
        else:
            np.testing.assert_array_equal(new[path], value)
    tokens = jnp.arange(16).reshape(2, 8) % 32
    np.testing.assert_allclose(model(tokens), control(tokens), rtol=2e-5, atol=2e-6)
    excluded = frozenset(s.name for s in attention_parameter_specs(config.model) if not s.weight_decay)
    mask = {p: bool(v.get_value()) for p, v in nnx.to_flat_state(
        attention_weight_decay_mask(nnx.state(model, nnx.Param), excluded_names=excluded))}
    assert mask[("forget_gate_kernel",)] and mask[("query_polynomial_kernel",)]
    if conditioning == "query_key":
        assert mask[("forget_gate_key_kernel",)]
    for name in ("forget_gate_log_rate", "forget_gate_dt_bias", "query_polynomial_amplitude", "query_polynomial_bias"):
        assert not mask[(name,)]
    identity = attention_identity(config)
    assert identity["attention_forget_gate_per_order"] is True
    assert identity["forget_gate_parameter_count"] == sum(new[p].size for p in new if p[0].startswith("forget_gate_"))


def test_endpoint_diagnostics_keep_each_order_and_exclude_padding():
    args = inputs()
    _, stats = operation(*args, endpoint=True, diagnostics=True)
    q, k, v, c, x, wq, wk, rate, bias = map(np.asarray, args)
    z = np.einsum("btd,hrd->bthr", x, wq)[:, :, None] + np.einsum("bid,hrd->bihr", x, wk)[:, None]
    alpha = -np.exp(rate[..., 0]) * np.logaddexp(0., z + bias[..., 0])
    causal = np.arange(9)[:, None] >= np.arange(9)[None, :]
    for name, reduce in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        np.testing.assert_allclose(stats[f"forget_gate_log_alpha_{name}"], reduce(alpha[:, causal], axis=(0, 1)), rtol=3e-6)


@pytest.mark.parametrize("endpoint", (False, True))
def test_degree_zero_decays_and_diagonal_is_unchanged(endpoint):
    args = list(inputs())
    args[0] = jnp.zeros_like(args[0])
    args[1] = jnp.zeros_like(args[1])
    args[2] = jnp.broadcast_to(jnp.eye(9)[None, :, None, :], (2, 9, 2, 9))
    args[3] = jnp.zeros_like(args[3]).at[..., 0].set(1.)
    weights = operation(*args, endpoint=endpoint)[0, :, 0]
    np.testing.assert_array_equal(np.diag(weights), np.ones(9))
    np.testing.assert_array_equal(np.triu(weights, 1), np.zeros((9, 9)))
    assert 0 < weights[8, 0] < 1


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_invalid_order_gate_poisoning_is_causal(bad):
    q, k, v, c, x, wq, wk, rate, bias = inputs()
    query, key = forget_gate.project_gate_inputs(x, wq, bias, wk)
    key = key.at[0, 3, 0, 2].set(bad)
    out = blocked_dot_product_attention(q, k, v, attention_degree=4, attention_normalization=False,
        query_polynomial_coefficients=c, query_gate_input=query, key_gate_input=key, gate_log_rate=rate,
        query_decay_mode="query_distance", query_block_size=4)
    invalid = ~np.isfinite(out).all(axis=-1)
    expected = np.zeros(invalid.shape, bool)
    expected[0, 3:, 0] = True
    np.testing.assert_array_equal(invalid, expected)


@pytest.mark.parametrize("changes", ({"attention_backend": "mosaic"}, {"attention_forget_gate_mode": "cumulative"},
    {"attention_forget_gate": False}, {"attention_forget_gate_rate_activation": "sigmoid"}))
def test_unsupported_per_order_recipes_rejected(changes):
    with pytest.raises(ValueError):
        validate_config(_config(attention_forget_gate_per_order=True, **changes))


def test_historical_checkpoint_defaults_keep_shared_gates():
    current = config_to_dict(_config())
    old = config_to_dict(_config())
    old["model"].pop("attention_forget_gate_per_order")
    assert config_dicts_equal(current, old)
    current["model"]["attention_forget_gate_per_order"] = True
    assert not config_dicts_equal(current, old)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_query_projection_is_not_hidden_by_softplus(bad):
    _, _, _, _, x, wq, _, rate, bias = inputs()
    alpha = forget_gate.log_decay(x, wq, rate, bias.at[0, 2, 0].set(bad))
    assert np.isnan(alpha[:, :, 0, 2]).all()


@pytest.mark.gpu
@pytest.mark.parametrize("endpoint", (False, True), ids=("query", "query_key"))
def test_gpu_per_order_forward_and_backward(endpoint):
    if jax.default_backend() != "gpu":
        pytest.skip("GPU qualification")
    arrays = list(inputs(2048, 4, width=64, value_width=64))
    arrays[:3] = [a.astype(jnp.bfloat16) for a in arrays[:3]]
    cotangent = jnp.asarray(np.random.default_rng(5).normal(size=arrays[2].shape), jnp.bfloat16)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        out, backward = jax.vjp(partial(dense, endpoint=endpoint), *[a.astype(jnp.float64) for a in arrays])
        expected = jax.tree.map(np.asarray, (out, backward(cotangent.astype(jnp.float64))))
    def evaluate(*args):
        out, backward = jax.vjp(partial(operation, endpoint=endpoint), *args)
        return out, backward(cotangent)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(evaluate)(*arrays)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        got = np.asarray(got, np.float64)
        assert np.isfinite(got).all()
        error = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
        assert error < .01, error
    test_model_initialization_matches_control_and_decay_masks("query_key" if endpoint else "query")
