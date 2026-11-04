"""Endpoint-only decay: independent pairwise mathematics, gradients, and initialization."""
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention import forget_gate
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.optim import attention_weight_decay_mask, make_optimizer


def _inputs(time=19, *, width=5, value_width=3, batch=2, heads=2, seed=91):
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(batch, time, heads, width)) for _ in range(2))
    v = rng.normal(size=(batch, time, heads, value_width))
    c = rng.normal(size=(batch, time, heads, 5)) / np.array([math.factorial(r) for r in range(5)])
    query = rng.uniform(-5, -1, size=(batch, time, heads))
    key = rng.normal(size=query.shape) * .5
    rate = rng.uniform(-1, 1, size=(heads, 1))
    return (q, k, v, c, query, key, rate), rng.normal(size=v.shape)


def _operation(q, k, v, c, query, key, rate, *, activation="softplus", block=8, diagnostics=False):
    gate = (dict(query_gate_input=query, key_gate_input=key, gate_log_rate=rate)
            if activation == "softplus" else
            dict(query_log_decay=-jnp.exp(rate[:, 0]) * jax.nn.sigmoid(query)))
    return blocked_dot_product_attention(
        q, k, v, attention_degree=4, attention_normalization=False,
        query_polynomial_coefficients=c, query_decay_mode="query_distance",
        query_block_size=block, return_diagnostics=diagnostics, **gate)


def _dense(q, k, v, c, query, key, rate, *, activation="softplus"):
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    poly = sum(jnp.swapaxes(c[..., r], 1, 2)[..., None] * scores ** r for r in range(c.shape[-1]))
    u = jnp.swapaxes(query, 1, 2)[..., None]
    if activation == "softplus":
        u = u + jnp.swapaxes(key, 1, 2)[:, :, None, :]
        lam = jnp.exp(rate[:, 0])[None, :, None, None] * jnp.logaddexp(0., u)
    else:
        lam = jnp.exp(rate[:, 0])[None, :, None, None] / (1 + jnp.exp(-u))
    positions = jnp.arange(q.shape[1])
    distance = positions[:, None] - positions[None, :]
    weights = jnp.where(distance >= 0, poly * jnp.exp(-lam * jnp.maximum(distance, 0)), 0.)
    return jnp.einsum("bhti,bihd->bthd", weights, v)


def _numpy_vjp(operands, cotangent, *, activation="softplus"):
    """Host FP64 oracle and analytic derivatives, independent of JAX attention code."""
    q, k, v, c, query, key, rate = (np.asarray(x, np.float64) for x in operands)
    qh, kh, vh, ch = (x.transpose(0, 2, 1, 3) for x in (q, k, v, c))
    dh = np.asarray(cotangent, np.float64).transpose(0, 2, 1, 3)
    scale = math.sqrt(q.shape[-1])
    scores = qh @ kh.swapaxes(-1, -2) / scale
    positions = np.arange(q.shape[1])
    distance = np.maximum(positions[:, None] - positions[None, :], 0)
    causal = positions[:, None] >= positions[None, :]
    u = query.transpose(0, 2, 1)[..., None]
    a = np.exp(rate[:, 0])[None, :, None, None]
    if activation == "softplus":
        u = u + key.transpose(0, 2, 1)[:, :, None, :]
        lam = a * np.logaddexp(0., u)
        slope_rate = a / (1 + np.exp(-u))
    else:
        sigmoid = 1 / (1 + np.exp(-u))
        lam, slope_rate = a * sigmoid, a * sigmoid * (1 - sigmoid)
    decay = np.where(causal, np.exp(-lam * distance), 0.)
    poly, slope = np.zeros_like(scores), np.zeros_like(scores)
    for r in range(c.shape[-1]):
        poly += ch[..., r, None] * scores ** r
        if r:
            slope += r * ch[..., r, None] * scores ** (r - 1)
    weights = decay * poly
    dw = dh @ vh.swapaxes(-1, -2)
    dpoly = dw * decay
    ds = dpoly * slope
    dc = np.stack([np.sum(dpoly * scores ** r, axis=-1) for r in range(c.shape[-1])], axis=-1)
    du = -dw * weights * distance * slope_rate
    gradients = (
        (ds @ kh / scale).transpose(0, 2, 1, 3),
        (ds.swapaxes(-1, -2) @ qh / scale).transpose(0, 2, 1, 3),
        (weights.swapaxes(-1, -2) @ dh).transpose(0, 2, 1, 3),
        dc.transpose(0, 2, 1, 3),
        du.sum(axis=-1).transpose(0, 2, 1),
        (du.sum(axis=-2).transpose(0, 2, 1) if activation == "softplus" else np.zeros_like(key)),
        (-dw * weights * distance * lam).sum(axis=(0, 2, 3))[:, None],
    )
    return (weights @ vh).transpose(0, 2, 1, 3), gradients


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _compare(actual, expected, tolerance):
    for i, (observed, wanted) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert observed.shape == wanted.shape and np.isfinite(observed).all(), i
        norm = np.linalg.norm(wanted)
        if norm == 0:
            np.testing.assert_array_equal(observed, wanted)
        else:
            error = np.linalg.norm(observed - wanted) / norm
            assert error < tolerance, (i, error, tolerance)


@pytest.mark.parametrize("activation", ("softplus", "sigmoid"))
def test_analytic_oracle_matches_independent_float64_autodiff(activation):
    operands, cotangent = _inputs(7)
    expected = _numpy_vjp(operands, cotangent, activation=activation)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        actual = _vjp(lambda *x: _dense(*x, activation=activation),
                      tuple(jnp.asarray(x, jnp.float64) for x in operands), jnp.asarray(cotangent))
    _compare(actual, expected, 2e-12)


@pytest.mark.parametrize("time", (1, 19, 37))
@pytest.mark.parametrize("activation", ("softplus", "sigmoid"))
def test_blocked_outputs_and_all_gradients_match_float64_oracle(time, activation):
    operands, cotangent = _inputs(time)
    expected = _numpy_vjp(operands, cotangent, activation=activation)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(lambda *args: _operation(*args, activation=activation), x,
                                        jnp.asarray(cotangent, jnp.float32)))(
            *[jnp.asarray(x, jnp.float32) for x in operands])
    _compare(actual, expected, 4e-5)


@pytest.mark.parametrize("activation", ("softplus", "sigmoid"))
def test_full_projection_chain_rule_matches_float64(activation):
    arrays, cotangent = _inputs(13)
    rng = np.random.default_rng(102)
    x = rng.normal(size=(2, 13, 7))
    wq, wk = rng.normal(size=(2, 2, 7)) * .2
    A, b = arrays[-1], np.array([[-2.], [-1.3]])
    operands = (*arrays[:4], x, wq, wk, A, b)

    def reference(q, k, v, c, x, wq, wk, A, b):
        query = jnp.einsum("btd,hd->bth", x, wq) + b[:, 0]
        key = jnp.einsum("btd,hd->bth", x, wk)
        return _dense(q, k, v, c, query, key, A, activation=activation)

    def production(q, k, v, c, x, wq, wk, A, b):
        if activation == "sigmoid":
            decay = forget_gate.log_decay(x, wq, A, b, rate_activation="sigmoid")
            return blocked_dot_product_attention(q, k, v, attention_degree=4, attention_normalization=False,
                query_polynomial_coefficients=c, query_log_decay=decay, query_decay_mode="query_distance",
                query_block_size=8)
        query, key = forget_gate.project_gate_inputs(x, wq, b, wk)
        return _operation(q, k, v, c, query, key, A)

    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(reference, tuple(jnp.asarray(x, jnp.float64) for x in operands), jnp.asarray(cotangent))
        expected = jax.tree.map(np.asarray, expected)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent, jnp.float32)))(
            *[jnp.asarray(x, jnp.float32) for x in operands])
    _compare(actual, expected, 5e-5)


def test_query_key_gate_uses_only_endpoints_and_retains_unit_diagonal():
    # Identity V makes each returned scalar exactly one causal attention weight.
    q = k = jnp.zeros((1, 9, 1, 2))
    v = jnp.eye(9)[None, :, None, :]
    c = jnp.zeros((1, 9, 1, 5)).at[..., 0].set(1.)
    query = jnp.linspace(-4, -1, 9)[None, :, None]
    key = jnp.linspace(-1, 1, 9)[None, :, None]
    rate = jnp.array([[.3]])
    output = _operation(q, k, v, c, query, key, rate, block=4)[0, :, 0, :]
    np.testing.assert_allclose(output, _dense(q, k, v, c, query, key, rate)[0, :, 0], rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(np.diag(output), np.ones(9))
    np.testing.assert_array_equal(np.triu(output, 1), np.zeros((9, 9)))
    # Position 3 lies between (8,0), yet changes to both its inputs cannot change that weight.
    changed = _operation(q, k, v, c, query.at[0, 3, 0].set(50), key.at[0, 3, 0].set(50), rate)
    np.testing.assert_array_equal(changed[0, 8, 0, 0], output[8, 0])
    dq, dk = jax.grad(lambda a, b: _operation(q, k, v, c, a, b, rate)[0, 8, 0, 0], argnums=(0, 1))(query, key)
    assert dq[0, 8, 0] < 0 and dk[0, 0, 0] < 0
    np.testing.assert_array_equal(dq[0, :8, 0], np.zeros(8))
    np.testing.assert_array_equal(dk[0, 1:, 0], np.zeros(8))


def test_endpoint_diagnostics_measure_only_real_causal_pairs():
    arrays, _ = _inputs(19)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    _, diagnostics = _operation(*operands, block=8, diagnostics=True)
    query, key, log_rate = arrays[-3:]
    log_alpha = -np.exp(log_rate[:, 0])[None, None, None, :] * np.logaddexp(
        0., query[:, :, None, :] + key[:, None, :, :])
    causal = np.arange(19)[:, None] >= np.arange(19)[None, :]
    values = log_alpha[:, causal, :]
    for name, function in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        np.testing.assert_allclose(diagnostics[f"forget_gate_log_alpha_{name}"],
                                   function(values, axis=(0, 1)), rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("operand", ("query", "key"))
def test_invalid_endpoint_inputs_poison_only_affected_causal_rows(bad, operand):
    arrays, _ = _inputs(19)
    operands = [jnp.asarray(x, jnp.float32) for x in arrays]
    index = 4 if operand == "query" else 5
    operands[index] = operands[index].at[0, 3, 0].set(bad)
    output = _operation(*operands)
    actual = ~np.isfinite(np.asarray(output)).all(axis=-1)
    expected = np.zeros(actual.shape, dtype=bool)
    if operand == "query":
        expected[0, 3, 0] = True
    else:
        expected[0, 3:, 0] = True
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("activation", ("softplus", "sigmoid"))
@pytest.mark.parametrize("input_value", (-1000., 1000., 1e20))
def test_saturated_rates_and_masked_future_have_finite_outputs_and_gradients(activation, input_value):
    arrays, cotangent = _inputs(19)
    operands = [jnp.asarray(x, jnp.float32) for x in arrays]
    operands[4] = jnp.full_like(operands[4], input_value)
    operands[5] = jnp.zeros_like(operands[5])
    with jax.default_matmul_precision("highest"):
        result = jax.jit(lambda *x: _vjp(lambda *a: _operation(*a, activation=activation), x,
                                        jnp.asarray(cotangent, jnp.float32)))(*operands)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(result))
    if input_value > 0 and activation == "softplus":
        # For these rates only the unattenuated self term remains.
        q, k, v, c = operands[:4]
        score = (q * k).sum(axis=-1) / math.sqrt(q.shape[-1])
        poly = sum(c[..., r] * score ** r for r in range(5))
        np.testing.assert_allclose(result[0], v * poly[..., None], rtol=3e-5, atol=3e-5)


def _config(**changes):
    model = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2,
                 num_kv_heads=2, qk_head_dim=8, value_head_dim=8, intermediate_size=32,
                 positional_encoding="none", attention_backend="blocked", attention_degree=4,
                 attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.,
                 attention_coefficient_projection_initial_std=16 ** -.5,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_coefficient_kernel_weight_decay=True, attention_forget_gate=True,
                 attention_forget_gate_mode="query_distance", attention_output_scaling="head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length, config.data.synthetic_vocab_size = 9, 32
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = .001
    config.optim.weight_decay = .1
    return config


def _params(model):
    return {path: value[...] for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def test_zero_key_initialization_exactly_recovers_control_and_weight_decay_policy():
    config = _config(attention_forget_gate_conditioning="query_key")
    model, control = build_model(config.model, 42), build_model(_config().model, 42)
    actual, baseline = _params(model), _params(control)
    assert set(actual) - set(baseline) == {("forget_gate_key_kernel",)}
    for path, value in baseline.items():
        np.testing.assert_array_equal(actual[path], value, err_msg=str(path))
    np.testing.assert_array_equal(actual[("forget_gate_key_kernel",)], np.zeros((2, 2, 16)))
    tokens = jnp.arange(16).reshape(2, 8) % 32
    np.testing.assert_array_equal(model(tokens), control(tokens))
    # Verify actual optimizer updates, including decay on a nonzero key projection.
    model.forget_gate_key_kernel[...] = jnp.full((2, 2, 16), .25)
    params = nnx.state(model, nnx.Param)
    excluded = frozenset(s.name for s in attention_parameter_specs(config.model) if not s.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(params, excluded_names=excluded))}
    assert mask[("forget_gate_key_kernel",)] and mask[("forget_gate_kernel",)]
    assert not mask[("forget_gate_log_rate",)] and not mask[("forget_gate_dt_bias",)]
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    values = {path: value[...] for path, value in nnx.to_flat_state(updates)}
    np.testing.assert_allclose(values[("forget_gate_key_kernel",)], -.001 * .1 * .25, rtol=2e-6, atol=1e-10)


def test_bounded_initialization_matches_zero_input_rates_and_only_changes_bias():
    bounded = build_model(_config(attention_forget_gate_rate_activation="sigmoid").model, 42)
    control = build_model(_config().model, 42)
    actual, baseline = _params(bounded), _params(control)
    assert actual.keys() == baseline.keys()
    for path, value in baseline.items():
        if path != ("forget_gate_dt_bias",):
            np.testing.assert_array_equal(actual[path], value, err_msg=str(path))
    x = jnp.zeros((2, 8, 16), jnp.float32)
    for layer in range(2):
        args = lambda m: tuple(getattr(m, name)[...][layer] for name in (
            "forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias"))
        expected = forget_gate.log_decay(x, *args(control))
        observed = forget_gate.log_decay(x, *args(bounded), rate_activation="sigmoid")
        np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=1e-8)
        random_x = jnp.asarray(np.random.default_rng(layer).normal(size=x.shape), jnp.float32) * 100
        decay = forget_gate.log_decay(random_x, *args(bounded), rate_activation="sigmoid")
        limit = np.exp(np.asarray(bounded.forget_gate_log_rate[...][layer, :, 0]))
        assert np.isfinite(decay).all() and (np.asarray(decay) <= 0).all()
        assert (np.asarray(-decay) <= limit[None, None, :] * (1 + 2e-7)).all()
        gradient = jax.grad(lambda inputs: forget_gate.log_decay(
            inputs, *args(bounded), rate_activation="sigmoid").sum())(x)
        assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_bounded_saturation_does_not_hide_nonfinite_gate_inputs(bad):
    x, w = jnp.ones((1, 3, 4)), jnp.ones((2, 4))
    A, b = jnp.zeros((2, 1)), jnp.full((2, 1), -3.)
    for arguments in ((x.at[0, 1, 0].set(bad), w, A, b),
                      (x, w, A.at[0, 0].set(bad), b), (x, w, A, b.at[0, 0].set(bad))):
        decay = forget_gate.log_decay(*arguments, rate_activation="sigmoid")
        expected = np.zeros((1, 3, 2), dtype=bool)
        if arguments[0] is not x:
            expected[0, 1, :] = True
        else:
            expected[0, :, 0] = True
        np.testing.assert_array_equal(~np.isfinite(np.asarray(decay)), expected)


@pytest.mark.parametrize("change", (
    {"attention_forget_gate_conditioning": "unknown"},
    {"attention_forget_gate_rate_activation": "tanh"},
    {"attention_forget_gate_conditioning": "query_key", "attention_forget_gate": False},
    {"attention_forget_gate_conditioning": "query_key", "attention_forget_gate_mode": "cumulative"},
    {"attention_forget_gate_conditioning": "query_key", "attention_backend": "mosaic"},
    {"attention_forget_gate_conditioning": "query_key", "attention_forget_gate_form": "constant"},
    {"attention_forget_gate_rate_activation": "sigmoid", "attention_forget_gate_mode": "cumulative"},
    {"attention_forget_gate_rate_activation": "sigmoid", "attention_forget_gate_form": "constant"},
))
def test_unsupported_gate_recipes_are_rejected(change):
    with pytest.raises(ValueError):
        _config(**change)


def test_legacy_recipe_defaults_preserve_identity():
    current = config_to_dict(_config())
    legacy = {**current, "model": {k: v for k, v in current["model"].items() if k not in (
        "attention_forget_gate_conditioning", "attention_forget_gate_rate_activation")}}
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(config_to_dict(_config(attention_forget_gate_conditioning="query_key")), legacy)
    assert not config_dicts_equal(config_to_dict(_config(attention_forget_gate_rate_activation="sigmoid")), legacy)


def test_public_core_routes_endpoint_gate_and_rejects_incomplete_operands():
    operands, _ = _inputs(7)
    q, k, v, c, query, key, rate = [jnp.asarray(x, jnp.float32) for x in operands]
    kwargs = dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
                  attention_coefficient_activation="tanh", query_decay_mode="query_distance",
                  query_gate_input=query, key_gate_input=key, gate_log_rate=rate)
    np.testing.assert_allclose(attention_core(q, k, v, **kwargs), _operation(q, k, v, c, query, key, rate),
                               rtol=2e-6, atol=1e-6)
    for missing in ("query_gate_input", "key_gate_input", "gate_log_rate"):
        with pytest.raises(ValueError):
            attention_core(q, k, v, **{key: value for key, value in kwargs.items() if key != missing})
    with pytest.raises(ValueError):
        attention_core(q, k, v, **kwargs, query_log_decay=-jnp.ones_like(query))


@pytest.mark.gpu
@pytest.mark.parametrize("activation", ("softplus", "sigmoid"))
def test_full_context_bfloat16_output_and_gradients_match_host_fp64(activation):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires the experiment's GPU allocation")
    operands, cotangent = _inputs(2048, width=64, value_width=64, batch=1, heads=2, seed=103)
    operands = tuple(jnp.asarray(x, jnp.bfloat16 if i < 3 else jnp.float32) for i, x in enumerate(operands))
    cotangent = jnp.asarray(cotangent, jnp.bfloat16)
    expected = _numpy_vjp(operands, cotangent, activation=activation)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(lambda *a: _operation(*a, activation=activation, block=128),
                                        x, cotangent))(*operands)
    _compare(actual, expected, .015)
