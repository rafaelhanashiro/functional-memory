"""Query-selected distance ranges: independent mathematics, gradients, and recipe invariants."""
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.forget_gate import log_decay
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.optim import attention_weight_decay_mask, make_optimizer


def _dense(q, k, v, coefficients, log_alpha):
    """Expanded polynomial and explicit distance matrix, without production gate/attention helpers."""
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    polynomial = sum(jnp.swapaxes(coefficients[..., order], 1, 2)[..., None] * scores ** order
                     for order in range(coefficients.shape[-1]))
    positions = jnp.arange(q.shape[1])
    distance = positions[:, None] - positions[None, :]
    attenuation = jnp.exp(jnp.swapaxes(log_alpha, 1, 2)[..., None] * jnp.maximum(distance, 0))
    weights = jnp.where(distance >= 0, polynomial * attenuation, 0.0)
    return jnp.einsum("bhti,bihd->bthd", weights, v)


def _numpy_dense_vjp(operands, cotangent):
    """Independent FP64 algebra and analytic VJP, with no GPU reference compilation."""
    q, k, v, coefficients, log_alpha = (np.asarray(x, np.float64) for x in operands)
    qh, kh, vh = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
    dh = np.asarray(cotangent, np.float64).transpose(0, 2, 1, 3)
    ch = coefficients.transpose(0, 2, 1, 3)
    scale = math.sqrt(q.shape[-1])
    scores = qh @ kh.swapaxes(-1, -2) / scale
    positions = np.arange(q.shape[1])
    distance = np.maximum(positions[:, None] - positions[None, :], 0)
    causal = positions[:, None] >= positions[None, :]
    decay = np.where(causal, np.exp(log_alpha.transpose(0, 2, 1)[..., None] * distance), 0.)
    polynomial, slope = np.zeros_like(scores), np.zeros_like(scores)
    for order in range(coefficients.shape[-1]):
        polynomial += ch[..., order, None] * scores ** order
        if order:
            slope += order * ch[..., order, None] * scores ** (order - 1)
    weights = decay * polynomial
    output = weights @ vh
    dweight = dh @ vh.swapaxes(-1, -2)
    dpolynomial = dweight * decay
    dscore = dpolynomial * slope
    dcoefficients = np.stack([
        np.sum(dpolynomial * scores ** order, axis=-1)
        for order in range(coefficients.shape[-1])
    ], axis=-1)
    gradients = (
        (dscore @ kh / scale).transpose(0, 2, 1, 3),
        (dscore.swapaxes(-1, -2) @ qh / scale).transpose(0, 2, 1, 3),
        (weights.swapaxes(-1, -2) @ dh).transpose(0, 2, 1, 3),
        dcoefficients.transpose(0, 2, 1, 3),
        np.sum(dweight * weights * distance, axis=-1).transpose(0, 2, 1),
    )
    return output.transpose(0, 2, 1, 3), gradients


def _operation(q, k, v, coefficients, log_alpha, *, block=8):
    return blocked_dot_product_attention(
        q, k, v, attention_degree=coefficients.shape[-1] - 1, attention_normalization=False,
        query_polynomial_coefficients=coefficients, query_log_decay=log_alpha,
        query_decay_mode="query_distance", query_block_size=block)


def _inputs(time=19, *, width=5, value_width=3, batch=2, heads=2, seed=31):
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(batch, time, heads, width)) for _ in range(2))
    v = rng.normal(size=(batch, time, heads, value_width))
    coefficients = rng.normal(size=(batch, time, heads, 5)) / np.asarray([math.factorial(r) for r in range(5)])
    log_alpha = -rng.uniform(0.01, 0.5, size=(batch, time, heads))
    cotangent = rng.normal(size=v.shape)
    return (q, k, v, coefficients, log_alpha), cotangent


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _compare(actual, expected, tolerance):
    for name, observed, wanted in zip(
            ["output", *[f"gradient {i}" for i in range(len(actual[1]))]],
            [actual[0], *actual[1]], [expected[0], *expected[1]], strict=True):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert observed.shape == wanted.shape and np.isfinite(observed).all(), name
        scale = np.linalg.norm(wanted)
        assert scale > 0, name
        error = np.linalg.norm(observed - wanted) / scale
        assert error < tolerance, f"{name}: relative L2 error {error} >= {tolerance}"


@pytest.mark.parametrize("time", (1, 7, 19))
def test_numpy_forward_and_analytic_vjp_match_float64_autodiff(time):
    arrays, cotangent = _inputs(time)
    actual = _numpy_dense_vjp(arrays, cotangent)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(_dense, tuple(jnp.asarray(x, jnp.float64) for x in arrays),
                        jnp.asarray(cotangent, jnp.float64))
    for observed, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(observed, np.asarray(wanted), rtol=2e-12, atol=2e-12)
    if time == 1:
        # Distance zero makes both the attenuation and its gate derivative exact.
        np.testing.assert_array_equal(actual[1][-1], np.zeros_like(arrays[-1]))
        np.testing.assert_array_equal(expected[1][-1], np.zeros_like(arrays[-1]))


@pytest.mark.parametrize("time", (1, 19, 37))
def test_forward_and_all_operand_gradients_match_independent_float64(time):
    arrays, cotangent = _inputs(time)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(_dense, tuple(jnp.asarray(x, jnp.float64) for x in arrays), jnp.asarray(cotangent))
        expected = jax.tree.map(lambda x: np.asarray(x, np.float64), expected)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(_operation, x, jnp.asarray(cotangent, jnp.float32)))(*operands)
    if time == 1:
        # A query always retains its own value with attenuation one; the gate cannot affect it.
        np.testing.assert_array_equal(actual[1][-1], np.zeros_like(arrays[-1]))
        _compare((actual[0], actual[1][:-1]), (expected[0], expected[1][:-1]), 3e-5)
    else:
        _compare(actual, expected, 3e-5)


def test_gate_projection_chain_rule_matches_independent_float64():
    arrays, cotangent = _inputs(13)
    rng = np.random.default_rng(43)
    q, k, v, coefficients, _ = arrays
    x = rng.normal(size=(2, 13, 7))
    kernel = rng.normal(size=(2, 7)) * .2
    log_rate, dt_bias = np.asarray([[.3], [1.1]]), np.asarray([[-2.0], [-1.3]])
    operands = (q, k, v, coefficients, x, kernel, log_rate, dt_bias)

    def reference(q, k, v, c, x, w, A, b):
        projected = jnp.einsum("btd,hd->bth", x, w) + b[:, 0]
        log_alpha = -jnp.exp(A[:, 0]) * jnp.logaddexp(0.0, projected)
        return _dense(q, k, v, c, log_alpha)

    def production(q, k, v, c, x, w, A, b):
        return _operation(q, k, v, c, log_decay(x, w, A, b))

    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(reference, tuple(jnp.asarray(x, jnp.float64) for x in operands), jnp.asarray(cotangent))
        expected = jax.tree.map(lambda x: np.asarray(x, np.float64), expected)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(production, x, jnp.asarray(cotangent, jnp.float32)))(
            *[jnp.asarray(x, jnp.float32) for x in operands])
    _compare(actual, expected, 4e-5)
    assert log_decay(*(jnp.asarray(x, jnp.bfloat16) for x in operands[4:])).dtype == jnp.float32


def test_constant_gate_matches_cumulative_but_varying_gate_is_query_local():
    q, k = jnp.zeros((1, 9, 1, 2)), jnp.zeros((1, 9, 1, 2))
    v = jnp.eye(9, dtype=jnp.float32)[None, :, None, :]
    c = jnp.zeros((1, 9, 1, 5), jnp.float32).at[..., 0].set(1.0)
    constant = jnp.full((1, 9, 1), -.2)
    cumulative = lambda a: blocked_dot_product_attention(
        q, k, v, attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
        query_log_decay=jnp.cumsum(a, axis=1), query_block_size=4)
    np.testing.assert_allclose(_operation(q, k, v, c, constant, block=4), cumulative(constant), atol=3e-7)
    varying = -jnp.arange(1, 10, dtype=jnp.float32)[None, :, None] * .03
    output = _operation(q, k, v, c, varying, block=4)[0, :, 0, :]
    expected = np.zeros((9, 9))
    for t in range(9):
        for i in range(t + 1):
            expected[t, i] = math.exp((t - i) * float(varying[0, t, 0]))
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(np.diag(output), np.ones(9))
    np.testing.assert_array_equal(np.triu(output, 1), np.zeros((9, 9)))
    assert np.max(np.abs(output - np.asarray(cumulative(varying)[0, :, 0, :]))) > .05

    changed = varying.at[0, 3, 0].set(-20.0)
    changed_output = _operation(q, k, v, c, changed, block=4)
    np.testing.assert_array_equal(changed_output[0, 8], output[8][None, :])
    gradient = jax.grad(lambda a: _operation(q, k, v, c, a, block=4)[0, 8, 0, 0])(varying)
    assert gradient[0, 8, 0] > 0
    np.testing.assert_array_equal(gradient[0, :8, 0], np.zeros(8))


@pytest.mark.parametrize("log_alpha_value", (0.0, -1000.0, -1e30))
def test_open_and_strong_gates_mask_future_without_overflow(log_alpha_value):
    arrays, cotangent = _inputs(19)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays[:-1]) + (
        jnp.full(arrays[-1].shape, log_alpha_value, jnp.float32),)
    with jax.default_matmul_precision("highest"):
        result = jax.jit(lambda *x: _vjp(_operation, x, jnp.asarray(cotangent, jnp.float32)))(*operands)
        expected = _dense(*operands)
    for value in jax.tree.leaves(result):
        assert np.isfinite(value).all()
    np.testing.assert_allclose(result[0], expected, rtol=3e-5, atol=2e-5)
    if log_alpha_value == 0:
        ordinary = blocked_dot_product_attention(*operands[:3], attention_degree=4, attention_normalization=False,
                                                 query_polynomial_coefficients=operands[3], query_block_size=8)
        np.testing.assert_allclose(result[0], ordinary, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf"), .01))
def test_invalid_gate_poisoning_is_local_to_its_query(bad):
    arrays, _ = _inputs(19)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    decay = operands[-1].at[0, 3, 0].set(bad)
    output = _operation(*operands[:-1], decay)
    invalid = ~np.isfinite(np.asarray(output)).all(axis=-1)
    expected = np.zeros(invalid.shape, dtype=bool)
    expected[0, 3, 0] = True
    np.testing.assert_array_equal(invalid, expected)


def _config(mode="query_distance", **changes):
    model = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2,
                 num_kv_heads=2, qk_head_dim=8, value_head_dim=8, intermediate_size=32,
                 positional_encoding="none", attention_backend="blocked", attention_degree=4,
                 attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.0,
                 attention_coefficient_projection_initial_std=16 ** -.5,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_coefficient_kernel_weight_decay=True,
                 attention_forget_gate=True, attention_forget_gate_mode=mode,
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length, config.data.synthetic_vocab_size = 9, 32
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = .001
    config.optim.weight_decay = .1
    return config


def test_mode_validation_and_legacy_recipe_equality(tmp_path):
    cumulative = config_to_dict(_config("cumulative"))
    legacy = {**cumulative, "model": {k: v for k, v in cumulative["model"].items()
                                     if k != "attention_forget_gate_mode"}}
    assert config_dicts_equal(cumulative, legacy)
    assert not config_dicts_equal(config_to_dict(_config()), legacy)
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy))
    assert load_config(path).model.attention_forget_gate_mode == "cumulative"
    for values in ({"mode": "unknown"}, {"attention_forget_gate": False}, {"attention_backend": "fused"}):
        with pytest.raises(ValueError):
            _config(**values)
    operands, _ = _inputs(5)
    q, k, v, c, _ = (jnp.asarray(x, jnp.float32) for x in operands)
    with pytest.raises(ValueError):
        attention_core(q, k, v, attention_degree=4, attention_normalization=False,
                       query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                       query_decay_mode="query_distance")


def test_initialization_factorial_coefficients_and_weight_decay_match_control():
    config = _config()
    control = build_model(_config("cumulative").model, 42)
    model = build_model(config.model, 42)
    params, control_params = nnx.state(model, nnx.Param), nnx.state(control, nnx.Param)
    actual, expected = nnx.to_flat_state(params), nnx.to_flat_state(control_params)
    assert [path for path, _ in actual] == [path for path, _ in expected]
    for (path, value), (_, baseline) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value[...], baseline[...], err_msg=str(path))
    np.testing.assert_array_equal(params["query_polynomial_bias"][...], np.ones((2, 2, 5)))
    np.testing.assert_array_equal(params["query_polynomial_amplitude"][...], np.zeros((2, 2, 5)))
    assert params["forget_gate_kernel"].shape == (2, 2, 16)
    assert params["forget_gate_log_rate"].shape == params["forget_gate_dt_bias"].shape == (2, 2, 1)
    tokens = jnp.arange(16).reshape(2, 8) % 32
    logits, coefficients = model(tokens, return_coefficients=True)
    np.testing.assert_allclose(coefficients, np.broadcast_to(1 / np.asarray([1, 1, 2, 6, 24]), coefficients.shape),
                               rtol=1e-7, atol=0)
    assert not np.allclose(logits, control(tokens), rtol=1e-7, atol=1e-7)
    for network in (model, control):
        network.forget_gate_dt_bias[...] = jnp.full_like(network.forget_gate_dt_bias[...], -200.0)
    np.testing.assert_array_equal(model(tokens), control(tokens))

    # Exercise the optimizer's actual zero-gradient update, as well as its mask.
    # This verifies the coefficient/gate exemptions without relying on array rank.
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(params, excluded_names=excluded))}
    assert mask[("forget_gate_kernel",)] and mask[("query_polynomial_kernel",)]
    for name in ("query_polynomial_bias", "query_polynomial_amplitude", "forget_gate_log_rate", "forget_gate_dt_bias"):
        assert not mask[(name,)]
    exempt_paths = [path for path, _ in actual if any("norm" in str(key) or "embedding" in str(key) for key in path)]
    assert exempt_paths and all(not mask[path] for path in exempt_paths)
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    for path, update in nnx.to_flat_state(updates):
        parameter = dict(actual)[path][...]
        wanted = -.001 * .1 * parameter if mask[path] else jnp.zeros_like(parameter)
        np.testing.assert_allclose(update[...], wanted, rtol=2e-6, atol=1e-10, err_msg=str(path))


def test_acceptance_and_diagnostics_identify_the_new_decay_rule():
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.training.diagnostics import attention_identity

    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (True, False):
        paths = acceptance.test_paths(spec, config_to_dict(_config()), resolved_backend="blocked", cpu=cpu)
        assert acceptance.QUERY_DISTANCE_DECAY_TESTS in paths
        old_paths = acceptance.test_paths(spec, config_to_dict(_config("cumulative")),
                                         resolved_backend="blocked", cpu=cpu)
        assert acceptance.QUERY_DISTANCE_DECAY_TESTS not in old_paths
    identity = attention_identity(_config())
    assert identity["attention_forget_gate_mode"] == "query_distance"
    assert "exp((t-i) log alpha_t)" in identity["attention_formula"]
    assert "prod_(j=i+1..t)" not in identity["attention_formula"]
    old = attention_identity(_config("cumulative"))
    assert old["attention_forget_gate_mode"] == "cumulative"
    assert "prod_(j=i+1..t)" in old["attention_formula"]


@pytest.mark.gpu
def test_full_context_bfloat16_forward_and_gradients_match_dense_reference():
    """Qualify the production 2,048-token/64-channel blocked computation on its allocated GPU."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires the experiment's GPU allocation")
    arrays, cotangent = _inputs(2048, width=64, value_width=64, batch=1, heads=2, seed=71)
    # Include long and short context ranges within the same batch/head layout.
    arrays = (*arrays[:4], -np.exp(np.random.default_rng(72).uniform(-8, 1, size=arrays[-1].shape)))
    operands = tuple(jnp.asarray(x, jnp.bfloat16 if i < 3 else jnp.float32) for i, x in enumerate(arrays))
    cotangent = jnp.asarray(cotangent, jnp.bfloat16)
    production = lambda *x: _operation(*x, block=128)

    # Preserve the BF16 input values, but compute the oracle and all derivatives
    # in host FP64. This keeps qualification independent of GPU/XLA compilation.
    expected = _numpy_dense_vjp(operands, cotangent)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(production, x, cotangent))(*operands)
    _compare(actual, expected, .012)
    for order in range(5):
        wanted, observed = np.asarray(expected[1][3][..., order]), np.asarray(actual[1][3][..., order])
        assert np.linalg.norm(observed - wanted) / np.linalg.norm(wanted) < .012
