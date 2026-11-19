"""Learned static coefficients under the input-free distance gate (attention_forget_gate_conditioning: none).

The rate lambda = exp(A_log) softplus(dt_bias) reads no token: one per layer and head, or per layer, head
and order, so the weight from key i to query t is sum_r c_r s_ti^r exp(-(t-i) lambda(_r)). The model runs
it as the query-coefficient decay operation with every query's coefficients and rates equal. These tests
check it against explicit FP64 mathematics, the model's dispatch, its parameters, optimizer masks,
diagnostics and recipe rules, and (GPU) the Mosaic decay kernels.
"""
from dataclasses import replace
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import forget_gate, multihead
from scaffolds.transformer.attention.backends.reference import (
    blocked_dot_product_attention, strip_dot_product_attention,
)
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, forget_gate_report
from scaffolds.transformer.training.optim import attention_weight_decay_mask, make_optimizer
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.tests.composition_checks import refused_unless_composed

GATE = ("forget_gate_log_rate", "forget_gate_dt_bias")


def inputs(time=19, degree=4, *, per_order=False, width=5, value_width=3, batch=2, heads=2, seed=61, scale=1.0):
    """q, k, v; static coefficients c [H, m+1] near 1/r!-sized; A_log, dt_bias [H,1] or [H,m+1,1]."""
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(batch, time, heads, width)) * scale for _ in range(2))
    v = rng.normal(size=(batch, time, heads, value_width))
    c = rng.normal(size=(heads, degree + 1)) / np.asarray([math.factorial(r) for r in range(degree + 1)])
    shape = (heads, degree + 1, 1) if per_order else (heads, 1)
    # Rates exp(A) softplus(b) from about 0.007 to 0.85: half-lives from under a token to about 100.
    log_rate, dt_bias = rng.uniform(-1, 1, size=shape), rng.uniform(-4, -1, size=shape)
    return (q, k, v, c, log_rate, dt_bias), rng.normal(size=v.shape)


def dense(q, k, v, c, log_rate, dt_bias):
    """Explicit all-pairs weights sum_r c_hr s_ti^r exp(-(t-i) lambda_h(r)), independent of the blocked code."""
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    rate = jnp.exp(log_rate[..., 0]) * jnp.logaddexp(0.0, dt_bias[..., 0])
    rate = jnp.broadcast_to(rate[:, None] if rate.ndim == 1 else rate, c.shape)  # [H, m+1]
    distance = jnp.arange(q.shape[1])[:, None] - jnp.arange(q.shape[1])[None, :]
    weights = sum(c[None, :, None, None, r] * scores ** r
                  * jnp.exp(-jnp.maximum(distance, 0) * rate[None, :, None, None, r]) for r in range(c.shape[-1]))
    weights = jnp.where(distance >= 0, weights, 0.0)
    return jnp.einsum("bhti,bihd->bthd", weights, v)


def operation(q, k, v, c, log_rate, dt_bias, *, implementation="scan", block=None, backend="blocked"):
    """The model's kernel operands: c broadcast to every query, and the input-free log decay at every token
    (test_attention_receives_broadcast_coefficients_and_rates checks that the model passes exactly these)."""
    batch, time, heads = q.shape[:3]
    arguments = dict(
        attention_degree=c.shape[-1] - 1, attention_normalization=False, query_decay_mode="query_distance",
        query_polynomial_coefficients=jnp.broadcast_to(c.astype(jnp.float32), (batch, time, heads, c.shape[-1])),
        query_log_decay=forget_gate.input_free_log_decay(jnp.zeros((batch, time, 1)), log_rate, dt_bias))
    if block is None:
        return attention_core(q, k, v, backend=backend, attention_coefficient_activation="tanh",
                              blocked_implementation=implementation, **arguments)
    # Several query blocks or strips at a short length, so their boundaries are exercised.
    if implementation == "scan":
        return blocked_dot_product_attention(q, k, v, query_block_size=block, **arguments)
    return strip_dot_product_attention(q, k, v, strip_size=block, **arguments)


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _compare(actual, expected, tolerances, *, order_axes=None):
    """Relative L2 error of the output and each gradient; ``order_axes`` compares those leaves order by order."""
    leaves = list(zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True))
    tolerances = tolerances if isinstance(tolerances, tuple) else (tolerances,) * len(leaves)
    for index, (observed, wanted) in enumerate(leaves):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert observed.shape == wanted.shape and np.isfinite(observed).all(), index
        axis = (order_axes or {}).get(index)
        parts = ([(observed, wanted)] if axis is None else
                 [(np.take(observed, r, axis), np.take(wanted, r, axis)) for r in range(wanted.shape[axis])])
        for order, (part, reference) in enumerate(parts):
            norm = np.linalg.norm(reference)
            if norm == 0:
                np.testing.assert_array_equal(part, reference)
                continue
            error = np.linalg.norm(part - reference) / norm
            assert error < tolerances[index], (index, order, error, tolerances[index])


# Coefficient gradients [H, m+1] (leaf 4) per order; per-order gate gradients [H, m+1, 1] (leaves 5, 6) too.
ORDER_AXES = {False: {4: -1}, True: {4: -1, 5: 1, 6: 1}}


@pytest.mark.parametrize("time", (1, 19))
@pytest.mark.parametrize("implementation,block", (("scan", None), ("strips", None), ("scan", 4), ("strips", 8)))
@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
@pytest.mark.parametrize("degree", (4, 8))
def test_output_and_every_gradient_match_float64(degree, per_order, implementation, block, time):
    """Output and the gradients of q, k, v, c, A_log and dt_bias through the blocked scan and the strips."""
    operands, cotangent = inputs(time, degree, per_order=per_order)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(dense, tuple(jnp.asarray(x, jnp.float64) for x in operands),
                        jnp.asarray(cotangent, jnp.float64))
        expected = jax.tree.map(np.asarray, expected)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(partial(operation, implementation=implementation, block=block), x,
                                        jnp.asarray(cotangent, jnp.float32)))(
            *[jnp.asarray(x, jnp.float32) for x in operands])
    _compare(actual, expected, 5e-5, order_axes=ORDER_AXES[per_order])
    gate_gradients = [np.asarray(gradient) for gradient in actual[1][4:]]
    if time == 1:
        # A query's own key is never attenuated, so the gate cannot change a one-token sequence.
        assert all(not np.any(gradient) for gradient in gate_gradients)
    else:
        assert all(np.all(np.abs(gradient) > 0) for gradient in gate_gradients)


@pytest.mark.parametrize("implementation", ("scan", "strips"))
@pytest.mark.parametrize("degree", (4, 8))
def test_duplicated_order_gates_reproduce_the_shared_gate(degree, implementation):
    """Every order a copy of one gate (the per-order runs' start): the shared gate's output, and each shared
    parameter's gradient equals the sum of its copies' gradients."""
    operands, cotangent = inputs(19, degree)
    q, k, v, c, log_rate, dt_bias = (jnp.asarray(x, jnp.float32) for x in operands)
    copies = [jnp.broadcast_to(x[:, None], (x.shape[0], degree + 1, 1)) for x in (log_rate, dt_bias)]
    run = jax.jit(lambda *x: _vjp(partial(operation, implementation=implementation), x,
                                  jnp.asarray(cotangent, jnp.float32)))
    with jax.default_matmul_precision("highest"):
        per_order, shared = run(q, k, v, c, *copies), run(q, k, v, c, log_rate, dt_bias)
    observed = (per_order[0], *per_order[1][:4], *(gradient.sum(axis=1) for gradient in per_order[1][4:]))
    # The two arithmetic orders differ only in FP32 rounding (the gate gradients are cancelling sums).
    _compare(observed, (shared[0], *shared[1]), 2e-5)


def test_input_free_log_decay_is_the_rate_at_every_token_and_nan_for_invalid_parameters():
    x = jnp.zeros((2, 5, 7))
    log_rate, dt_bias = jnp.asarray([[0.3], [-1.2]]), jnp.asarray([[-2.0], [0.5]])
    decay = np.asarray(forget_gate.input_free_log_decay(x, log_rate, dt_bias))
    expected = -np.exp([0.3, -1.2]) * np.logaddexp(0.0, [-2.0, 0.5])
    assert decay.shape == (2, 5, 2) and decay.dtype == np.float32
    np.testing.assert_allclose(decay, np.broadcast_to(expected, (2, 5, 2)), rtol=1e-6)
    per_order = forget_gate.input_free_log_decay(x, jnp.zeros((2, 3, 1)), jnp.zeros((2, 3, 1)))
    assert per_order.shape == (2, 5, 2, 3)
    for bad in (float("nan"), float("inf"), float("-inf")):
        for arguments in ((log_rate.at[0, 0].set(bad), dt_bias), (log_rate, dt_bias.at[0, 0].set(bad))):
            decay = np.asarray(forget_gate.input_free_log_decay(x, *arguments))
            assert np.isnan(decay[..., 0]).all() and np.isfinite(decay[..., 1]).all()
    with pytest.raises(ValueError):
        forget_gate.input_free_log_decay(x, log_rate[:, 0], dt_bias[:, 0])


def test_an_invalid_gate_poisons_exactly_its_head():
    operands, _ = inputs(19)
    q, k, v, c, log_rate, dt_bias = (jnp.asarray(x, jnp.float32) for x in operands)
    output = np.asarray(operation(q, k, v, c, log_rate.at[1, 0].set(jnp.nan), dt_bias))
    invalid = ~np.isfinite(output).all(axis=-1)
    expected = np.zeros(invalid.shape, bool)
    expected[:, :, 1] = True
    np.testing.assert_array_equal(invalid, expected)


def _config(**changes):
    """A small recipe of the runs' kind: learned static per-head coefficients stored factorial-scaled, the
    input-free query-distance gate, no positional encoding, QK RMSNorm with head scales, head RMSNorm."""
    model = dict(vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                 qk_head_dim=8, value_head_dim=8, intermediate_size=32, positional_encoding="none",
                 attention_backend="blocked", attention_degree=4, attention_normalization=False,
                 attention_coefficients="learned", attention_coefficient_sharing="per_head",
                 attention_coefficient_scale="factorial", attention_forget_gate=True,
                 attention_forget_gate_mode="query_distance", attention_forget_gate_conditioning="none",
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = .001
    config.optim.weight_decay = .1
    if config.model.attention_backend == "mosaic":
        config.train.dtype = "bfloat16"
    validate_config(config)
    return config


def _ungated(model):
    return replace(model, attention_forget_gate=False, attention_forget_gate_mode="cumulative",
                   attention_forget_gate_conditioning="query", attention_forget_gate_per_order=False)


def _query_gate(model):
    """The query-distance gate's recipe at the same shape: tanh query coefficients, gate projection w."""
    return replace(model, attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                   attention_coefficient_amplitude_initial=0.0,
                   attention_coefficient_projection_initial_std=model.hidden_size ** -.5,
                   attention_coefficient_amplitude_scale="factorial", attention_coefficient_kernel_weight_decay=True,
                   attention_forget_gate_conditioning="query")


def _params(model):
    return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


TOKENS = jnp.arange(16).reshape(2, 8) % 32


def test_parameters_are_the_query_gates_rate_draws_without_its_projection():
    assert forget_gate.parameter_names("input", "none") == GATE
    with pytest.raises(ValueError):
        forget_gate.parameter_names("constant", "none")
    specs = {spec.name: spec for spec in forget_gate.parameter_specs(hidden_size=16, conditioning="none")}
    assert tuple(specs) == GATE and not any(spec.weight_decay for spec in specs.values())
    assert (specs[GATE[0]].initializer, specs[GATE[1]].initializer) == ("mamba2_log_rate", "mamba2_dt_bias")
    with pytest.raises(ValueError):
        forget_gate.parameter_specs(hidden_size=16, conditioning="none", rate_activation="sigmoid")
    config = _config()
    gated, ungated = _params(build_model(config.model, 42)), _params(build_model(_ungated(config.model), 42))
    query_gate = _params(build_model(_query_gate(config.model), 42))
    assert set(gated) - set(ungated) == {(name,) for name in GATE}
    for path, value in ungated.items():
        np.testing.assert_array_equal(gated[path], value, err_msg=str(path))
    for name in GATE:  # the same Gated DeltaNet draws as the query gate's A_log and dt_bias
        assert gated[(name,)].shape == (2, 2, 1)
        np.testing.assert_array_equal(gated[(name,)], query_gate[(name,)])
    rates = np.exp(gated[(GATE[0],)]) * np.logaddexp(0.0, gated[(GATE[1],)])
    assert np.all((rates > 0) & (rates < 1.6))
    assert model_parameter_count(config.model) == model_parameter_count(_ungated(config.model)) + 2 * 2 * 2
    assert model_parameter_count(config.model) == sum(value.size for value in gated.values())


@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
def test_attention_receives_broadcast_coefficients_and_rates(per_order, monkeypatch):
    """The model's dispatch: the represented static coefficients (stored c~ / r!) for every query, and
    -exp(A_log) softplus(dt_bias) for every token, through the query-coefficient decay operation."""
    config = _config(attention_forget_gate_per_order=per_order)
    model = build_model(config.model, 42)
    params = _params(model)
    calls, original = [], multihead.attention_core
    monkeypatch.setattr(multihead, "attention_core", lambda *a, **kw: calls.append(kw) or original(*a, **kw))
    model(TOKENS)
    assert len(calls) == 2
    factorials = np.asarray([math.factorial(r) for r in range(5)], np.float32)
    for layer, call in enumerate(calls):
        assert "polynomial_coefficients" not in call and call["attention_coefficient_activation"] == "tanh"
        assert call["query_decay_mode"] == "query_distance" and call["blocked_implementation"] == "scan"
        coefficients = params[("polynomial_coefficients",)][layer] / factorials
        np.testing.assert_array_equal(call["query_polynomial_coefficients"],
                                      np.broadcast_to(coefficients, (2, 8, 2, 5)))
        rate = (np.exp(params[(GATE[0],)][layer, ..., 0].astype(np.float64))
                * np.logaddexp(0.0, params[(GATE[1],)][layer, ..., 0].astype(np.float64)))
        decay = np.asarray(call["query_log_decay"])
        assert decay.shape == ((2, 8, 2, 5) if per_order else (2, 8, 2)) and decay.dtype == np.float32
        np.testing.assert_allclose(decay, np.broadcast_to(-rate, decay.shape), rtol=1e-6)


def test_zero_rates_reproduce_the_ungated_static_model_and_the_gate_acts_otherwise():
    config = _config()
    gated, ungated = build_model(config.model, 42), build_model(_ungated(config.model), 42)
    assert not np.allclose(gated(TOKENS), ungated(TOKENS), rtol=1e-4, atol=1e-5)
    gated.forget_gate_dt_bias[...] = jnp.full_like(gated.forget_gate_dt_bias[...], -200.0)  # softplus underflows
    np.testing.assert_allclose(gated(TOKENS), ungated(TOKENS), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("degree", (4, 8))
def test_per_order_gates_start_as_copies_of_the_shared_gate(degree):
    shared_config = _config(attention_degree=degree)
    config = _config(attention_degree=degree, attention_forget_gate_per_order=True)
    shared, model = build_model(shared_config.model, 42), build_model(config.model, 42)
    old, new = _params(shared), _params(model)
    assert old.keys() == new.keys()
    for path, value in old.items():
        if path[0] in GATE:
            assert new[path].shape == (2, 2, degree + 1, 1)
            np.testing.assert_array_equal(new[path], np.broadcast_to(value[:, :, None, :], new[path].shape))
        else:
            np.testing.assert_array_equal(new[path], value, err_msg=str(path))
    np.testing.assert_allclose(model(TOKENS), shared(TOKENS), rtol=2e-5, atol=2e-6)
    assert model_parameter_count(config.model) == model_parameter_count(shared_config.model) + 2 * 2 * 2 * degree


@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
def test_weight_decay_exempts_the_coefficients_and_the_gate(per_order):
    config = _config(attention_forget_gate_per_order=per_order)
    params = nnx.state(build_model(config.model, 42), nnx.Param)
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    assert {"polynomial_coefficients", *GATE} <= excluded
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(params, excluded_names=excluded))}
    for name in ("polynomial_coefficients", *GATE, "output_norm_scale", "qk_norm_head_log_scale"):
        assert not mask[(name,)], name
    assert any(mask.values())
    # The optimizer's zero-gradient update: weight decay alone moves only the decayed arrays.
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    values = dict(nnx.to_flat_state(params))
    for path, update in nnx.to_flat_state(updates):
        parameter = values[path][...]
        wanted = -.001 * .1 * parameter if mask[path] else jnp.zeros_like(parameter)
        np.testing.assert_allclose(update[...], wanted, rtol=2e-6, atol=1e-10, err_msg=str(path))


@pytest.mark.parametrize("implementation", ("scan", "strips"))
@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
def test_training_learns_the_coefficients_and_every_gate(per_order, implementation):
    config = _config(attention_forget_gate_per_order=per_order, attention_blocked_implementation=implementation)
    state = create_state(config, jax.random.PRNGKey(1))
    batch = jnp.asarray(np.random.default_rng(0).integers(0, 32, (2, 9)), jnp.int32)
    loss, gradients = jax.jit(lambda *x: loop.loss_and_grads(*x))(state, batch, jax.random.PRNGKey(0))
    assert np.isfinite(float(loss))
    for name in ("polynomial_coefficients", *GATE):
        assert np.all(np.abs(np.asarray(gradients[name][...])) > 0), name
    before = {name: np.asarray(state.params[name][...]) for name in ("polynomial_coefficients", *GATE)}
    after, *_ = loop.train_step(state, batch, jax.random.PRNGKey(0))
    for name, value in before.items():
        assert np.all(np.asarray(after.params[name][...]) != value), name


@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
def test_diagnostics_report_rates_half_lives_and_the_static_coefficients(per_order):
    config = _config(attention_forget_gate_per_order=per_order)
    state = create_state(config, jax.random.PRNGKey(1))
    _, raw = nnx.merge(state.graphdef, state.params)(TOKENS, deterministic=True, return_diagnostics=True)
    rate = (np.exp(np.asarray(state.params[GATE[0]][...], np.float64)[..., 0])
            * np.logaddexp(0.0, np.asarray(state.params[GATE[1]][...], np.float64)[..., 0]))
    for layer, values in enumerate(raw):
        for statistic in ("mean", "min", "max"):
            np.testing.assert_allclose(values[f"forget_gate_log_alpha_{statistic}"], -rate[layer], rtol=2e-6)
    report = forget_gate_report(state.params, raw, conditioning="none")
    assert report["finite"] and ("axes" in report) == per_order
    np.testing.assert_allclose(report["decay_rate"], rate, rtol=1e-12)
    np.testing.assert_allclose(report["half_life_tokens"], math.log(2) / rate, rtol=1e-12)
    np.testing.assert_allclose(report["mean_half_life_tokens"], math.log(2) / rate, rtol=2e-6)
    sample = np.asarray(TOKENS[:1, :1].repeat(9, axis=1) + jnp.arange(9), np.int32) % 32
    diagnostics = attention_diagnostics(state, config, sample, {"selection": "test"})
    assert diagnostics["valid"], diagnostics["issues"]
    coefficients = diagnostics["coefficients"]
    assert coefficients["raw_parameter_names"] == [f"c{r}" for r in range(5)]
    np.testing.assert_allclose(coefficients["effective"], np.broadcast_to(1 / np.asarray(
        [math.factorial(r) for r in range(5)]), (2, 2, 5)), rtol=1e-7)
    np.testing.assert_allclose(coefficients["forget_gate"]["half_life_tokens"], math.log(2) / rate, rtol=1e-12)
    assert diagnostics["attention_forget_gate_conditioning"] == "none"


@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
def test_identity_names_the_gate_and_the_mosaic_decay_kernels(per_order):
    identity = attention_identity(_config(attention_forget_gate_per_order=per_order))
    assert identity["attention_kernel"] == "learned_polynomial_unnormalized"
    assert identity["attention_forget_gate_mode"] == "query_distance"
    assert identity["forget_gate_parameter_count"] == 2 * 2 * 2 * (5 if per_order else 1)
    assert "read from no token" in identity["attention_formula"] and "x_t w" not in identity["attention_formula"]
    assert ("lambda_r" in identity["attention_formula"]) == per_order
    mosaic = attention_identity(_mosaic_config(per_order=per_order))
    assert mosaic["attention_implementation"] == ("pallas_mosaic_gpu_query_polynomial_per_order_query_decay"
                                                  if per_order else "pallas_mosaic_gpu_query_polynomial_query_decay")


def _mosaic_config(degree=4, per_order=False, **changes):
    return _config(attention_backend="mosaic", attention_degree=degree, attention_forget_gate_per_order=per_order,
                   max_seq_len=2048, hidden_size=128, qk_head_dim=64, value_head_dim=64, intermediate_size=256,
                   **changes)


def test_mosaic_accepts_the_gate_on_its_decay_kernels_only():
    """The runs' degrees with shared and per-order gates; per-order gates need only fit H100's kernels."""
    for degree, per_order in ((4, False), (8, False), (4, True), (8, True), (12, True)):
        model = _mosaic_config(degree, per_order).model
        assert model.mosaic_decay_supported() and not model.mosaic_supported()
    assert _mosaic_config(positional_encoding="rope").model.mosaic_decay_supported()
    with pytest.raises(ValueError, match="mosaic"):
        _mosaic_config(13)


@pytest.mark.parametrize("changes", (
    {"attention_forget_gate": False},
    {"attention_forget_gate_mode": "cumulative"},
    {"attention_forget_gate_form": "constant"},
    {"attention_forget_gate_form": "sigmoid"},
    {"attention_forget_gate_rate_activation": "sigmoid"},
    {"attention_backend": "fused"},
    {"attention_backend": "auto"},
    {"positional_encoding": "path"},
    {"attention_coefficients": "fixed", "attention_coefficient_scale": "none"},
    {"attention_coefficient_sharing": "global"},
    {"attention_coefficient_orders": (0, 2, 4)},
    {"attention_coefficient_initialization": "linear"},
    {"attention_normalization": True, "attention_output_scaling": "none"},
    {"attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh"},
    {"attention_kernel_basis": "bspline", "attention_spline_knots": (-4.0, 0.0, 4.0),
     "attention_coefficient_scale": "none"},
))
def test_unqualified_input_free_recipes_are_rejected(changes):
    refused_unless_composed(_config, changes)


@pytest.mark.parametrize("conditioning", ("query", "query_key"))
def test_static_coefficients_reject_gates_that_read_tokens(conditioning):
    # On the blocked backend static coefficients take them per query; the Mosaic decay kernels do not.
    refused_unless_composed(_config, {"attention_forget_gate_conditioning": conditioning})


@pytest.mark.parametrize("changes", (
    {"attention_forget_gate_rate_activation": "sigmoid"},
    {"attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh"},
    {"attention_forget_gate_mode": "cumulative"},
    {"attention_backend": "fused"},
))
def test_the_attention_module_rejects_them_too(changes):
    model = _config().model
    for name, value in changes.items():
        setattr(model, name, value)  # bypass the recipe's validation
    with pytest.raises(ValueError):
        multihead.attention_from_config(model, rngs=nnx.Rngs(0))


def test_historical_recipes_and_mechanisms_are_unchanged(monkeypatch):
    # A recipe without the field is a query gate: the default stays query.
    config = _config()
    current = config_to_dict(replace(config, model=_query_gate(config.model)))
    legacy = {**current, "model": {k: v for k, v in current["model"].items()
                                   if k != "attention_forget_gate_conditioning"}}
    assert config_dicts_equal(current, legacy)
    gated = config_to_dict(config)
    assert not config_dicts_equal(gated, {**gated, "model": {
        k: v for k, v in gated["model"].items() if k != "attention_forget_gate_conditioning"}})
    # The other gates keep their arrays, in order.
    names = lambda **kw: tuple(spec.name for spec in forget_gate.parameter_specs(hidden_size=8, **kw))
    assert names() == forget_gate.PARAMETERS
    assert names(conditioning="query_key") == (*forget_gate.PARAMETERS, forget_gate.KEY_PARAMETER)
    assert names(form="constant") == forget_gate.CONSTANT_PARAMETERS
    assert names(form="sigmoid") == forget_gate.SIGMOID_PARAMETERS
    # Ungated static coefficients keep the gate-free Mosaic kernel and the static attention call.
    static = _ungated(_mosaic_config(positional_encoding="rope").model)
    assert static.mosaic_supported() and not static.mosaic_decay_supported()
    assert not any(spec.name.startswith("forget_gate") for spec in attention_parameter_specs(static))
    calls, original = [], multihead.attention_core
    monkeypatch.setattr(multihead, "attention_core", lambda *a, **kw: calls.append(kw) or original(*a, **kw))
    build_model(_ungated(config.model), 42)(TOKENS)
    assert all("polynomial_coefficients" in call and "query_polynomial_coefficients" not in call
               and "query_log_decay" not in call for call in calls) and len(calls) == 2


@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
def test_acceptance_runs_the_gate_tests_on_both_stages(per_order):
    from scaffolds.transformer import acceptance

    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    config = config_to_dict(_mosaic_config(per_order=per_order))
    for cpu in (True, False):
        paths = acceptance.test_paths(spec, config, resolved_backend="mosaic", cpu=cpu)
        assert acceptance.INPUT_FREE_GATE_TESTS in paths
        assert "scaffolds/transformer/tests/test_endpoint_decay.py" in paths  # the Mosaic decay kernels
        assert ("scaffolds/transformer/tests/test_per_order_decay.py" in paths) == per_order
    base = _mosaic_config(positional_encoding="rope")  # Mosaic's gate-free static kernel needs RoPE
    ungated = config_to_dict(replace(base, model=_ungated(base.model)))
    for cpu in (True, False):
        assert acceptance.INPUT_FREE_GATE_TESTS not in acceptance.test_paths(spec, ungated, resolved_backend="mosaic",
                                                                             cpu=cpu)


def test_initialization_matches_the_softmax_backbone_and_declared_draws():
    from scaffolds.transformer import acceptance

    for per_order in (False, True):
        facts = acceptance.matches_softmax_backbone_initialization(_config(attention_forget_gate_per_order=per_order))
        assert facts["mechanism_parameters"][GATE[0]]["initializer"] == "mamba2_log_rate"
        assert facts["mechanism_parameters"][GATE[1]]["initializer"] == "mamba2_dt_bias"
        assert ("repeat_initial_axis" in facts["mechanism_parameters"][GATE[0]]) == per_order


# Mosaic GPU decay kernels: the operation the runs train with (model.attention_backend: mosaic).

def _lowers_for_cuda(function, *operands):
    """Trace and lower ``function`` for the cuda platform on any host, building every Mosaic GPU kernel."""
    jax.jit(lambda *arguments: function(*arguments)).trace(*operands).lower(lowering_platforms=("cuda",))


def _abstract_operands(degree, per_order, time=2048):
    q = jax.ShapeDtypeStruct((1, time, 2, 64), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((2, degree + 1, 1) if per_order else (2, 1), jnp.float32)
    return q, q, q, jax.ShapeDtypeStruct((2, degree + 1), jnp.float32), gate, gate


def _gradients(*operands):
    return _vjp(partial(operation, backend="mosaic"), operands, jnp.ones(operands[2].shape, operands[2].dtype))


@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
@pytest.mark.parametrize("degree", (4, 8))
def test_mosaic_dispatch_lowers_the_h100_decay_kernels_at_the_run_shape(monkeypatch, degree, per_order):
    """The production dispatch reaches the query-decay (or per-order query-decay) kernels with valid FP32
    operands, and their forward, dQ and dK/dV lower for H100 at 2,048 tokens and width 64."""
    from scaffolds.transformer.attention.backends.mosaic import api, hopper

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    traced = {}
    for name in ("query_decay_attention", "per_order_query_decay_attention"):
        function = getattr(api, name)
        monkeypatch.setattr(api, name, lambda *a, _f=function, _n=name, **kw: traced.setdefault(_n, True) and _f(*a, **kw))
    _lowers_for_cuda(_gradients, *_abstract_operands(degree, per_order))
    assert set(traced) == {"per_order_query_decay_attention" if per_order else "query_decay_attention"}


def test_b300_refuses_the_degree_eight_per_order_gate_before_building_a_kernel(monkeypatch):
    from scaffolds.transformer.attention.backends.mosaic import api

    monkeypatch.setattr(api, "_kernels", lambda: ("blackwell", None))  # no kernel may be reached
    with pytest.raises(ValueError, match="degree 8 .* exceeds B300 shared memory"):
        _lowers_for_cuda(_gradients, *_abstract_operands(8, True))


def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def _blackwell_refuses(degree, time):
    """Whether this GPU is a B300 that cannot hold the per-order shape (its entry points raise)."""
    from scaffolds.transformer.attention.backends.mosaic.api import DECAY_FAMILY, blackwell_supported

    return (jax.devices()[0].compute_capability != "9.0"
            and not blackwell_supported(DECAY_FAMILY["query", True], degree, 64, time))


def _mosaic_operands(time, degree, *, per_order):
    # Unit-scale scores, as QK RMSNorm gives, so that every order contributes at every degree.
    operands, cotangent = inputs(time, degree, per_order=per_order, width=64, value_width=64, batch=1,
                                 seed=67, scale=1.0)
    operands = tuple(jnp.asarray(x, jnp.bfloat16 if i < 3 else jnp.float32) for i, x in enumerate(operands))
    return operands, jnp.asarray(cotangent, jnp.bfloat16)


@pytest.mark.gpu
@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
@pytest.mark.parametrize("degree", (4, 8))
def test_mosaic_output_and_every_gradient_match_float64(degree, per_order):
    _requires_mosaic_gpu()
    operands, cotangent = _mosaic_operands(2048, degree, per_order=per_order)
    run = jax.jit(lambda *x: _vjp(partial(operation, backend="mosaic"), x, cotangent))
    if per_order and _blackwell_refuses(degree, 2048):
        with pytest.raises(ValueError, match="exceeds B300 shared memory"):
            run(*operands)
        return
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(dense, tuple(x.astype(jnp.float64) for x in operands), cotangent.astype(jnp.float64))
        expected = jax.tree.map(np.asarray, expected)
    actual = run(*operands)
    # Output, dq, dk, dv at the Mosaic BF16 bounds; c, A_log and dt_bias through FP32 sums, each order's
    # coefficient (and per-order gate) gradient compared on its own.
    tolerances = (.006, .025, .025, .015, .01, .01, .01)
    _compare(actual, expected, tolerances, order_axes=ORDER_AXES[per_order])


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (4, 8))
def test_mosaic_duplicated_orders_reproduce_the_shared_gate_kernel(degree):
    """Every order a copy of one gate (the per-order runs' start): the per-order kernels equal the shared-gate
    kernels, and each shared gate parameter's gradient is the sum over its copies."""
    _requires_mosaic_gpu()
    if _blackwell_refuses(degree, 1024):
        pytest.skip("B300 cannot hold this per-order shape")
    (q, k, v, c, log_rate, dt_bias), cotangent = _mosaic_operands(1024, degree, per_order=False)
    copies = [jnp.broadcast_to(x[:, None], (x.shape[0], degree + 1, 1)) for x in (log_rate, dt_bias)]
    run = jax.jit(lambda *x: _vjp(partial(operation, backend="mosaic"), x, cotangent))
    per_order, shared = run(q, k, v, c, *copies), run(q, k, v, c, log_rate, dt_bias)
    observed = (per_order[0], *per_order[1][:4], *(gradient.sum(axis=1) for gradient in per_order[1][4:]))
    _compare(observed, (shared[0], *shared[1]), 2e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("per_order", (False, True), ids=("shared", "per_order"))
def test_mosaic_model_trains_like_the_float32_blocked_model(per_order, monkeypatch):
    """End-to-end model on Mosaic in BF16 against the FP32 blocked model: its loss and gradient direction (as
    test_endpoint_decay's model check, whose tolerances these share)."""
    _requires_mosaic_gpu()
    from scaffolds.transformer.attention.backends.mosaic import api

    rng = np.random.default_rng(59)
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    coefficients = jnp.asarray(rng.normal(size=(2, 2, 5)) * .3, jnp.float32)  # move c off the Taylor start
    rates = jnp.asarray(rng.normal(size=(2, 2, 5, 1) if per_order else (2, 2, 1)) * .5, jnp.float32)  # orders differ
    results = {}
    for backend, dtype, precision in (("blocked", "float32", None), ("blocked", "bfloat16", None),
                                      ("mosaic", "bfloat16", "bf16"), ("mosaic", "bfloat16", "bf16x3")):
        monkeypatch.setattr(api, "default_precision", lambda family, degree, precision=precision: precision)
        config = _config(vocab_size=64, max_seq_len=256, hidden_size=128, qk_head_dim=64, value_head_dim=64,
                         intermediate_size=256, attention_backend=backend, attention_forget_gate_per_order=per_order)
        config.train.dtype = dtype
        state = create_state(config, jax.random.PRNGKey(3))
        state.params["polynomial_coefficients"][...] = state.params["polynomial_coefficients"][...] + coefficients
        state.params[GATE[0]][...] = state.params[GATE[0]][...] + rates
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            # A fresh function per case: the precision is read while tracing.
            loss, gradients = jax.jit(lambda *x: loop.loss_and_grads(*x))(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype, precision] = float(loss), {path: np.asarray(value[...], np.float64)
                                                          for path, value in nnx.to_flat_state(gradients)}
    reference_loss, reference = results["blocked", "float32", None]
    bf16 = results["blocked", "bfloat16", None][1]
    loss, actual = results["mosaic", "bfloat16", "bf16"]
    assert math.isfinite(loss) and abs(loss - reference_loss) < .02
    assert all(np.isfinite(value).all() for value in actual.values())
    for name in ("polynomial_coefficients", *GATE):
        assert np.linalg.norm(actual[(name,)]) > 0, name

    def cosine(gradients):
        a, b = (np.concatenate([g[path].ravel() for path in sorted(reference, key=str)]) for g in (gradients, reference))
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    exact = results["mosaic", "bfloat16", "bf16x3"][1]
    assert cosine(exact) >= min(.995, cosine(bf16) - .01), (cosine(exact), cosine(bf16))
    assert cosine(actual) >= min(.995, cosine(bf16) - .03), (cosine(actual), cosine(bf16))
