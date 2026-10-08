"""Query-conditioned m=1: independent math, canonical initialization, and CUDA gates."""

from dataclasses import replace
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends import fused_learned_polynomial as fused
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.input_dependent import project_coefficients, query_linear_attention
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import attention_metrics


def _model(**updates):
    settings = dict(vocab_size=16, max_seq_len=13, num_layers=2, hidden_size=8, num_heads=2,
                    num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
                    attention_degree=1, attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                    attention_backend="blocked", initialization="transformer_reference", qk_norm=True,
                    attention_output_scaling="head_rms_norm")
    return ModelConfig(**{**settings, **updates})


def _config():
    config = Config()
    config.model = _model()
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 14, 2
    config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    return config


def _inputs(time=13, dtype=jnp.float32, qk_dim=7, value_dim=5, heads=2):
    random = np.random.default_rng(231)
    q, k = (jnp.asarray(random.normal(size=(2, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(random.normal(size=(2, time, heads, value_dim)), dtype)
    x = jnp.asarray(random.normal(size=(2, time, 6)), dtype)
    kernel = jnp.asarray(random.normal(size=(heads, 6, 2)) * 0.2, jnp.float32)
    bias = jnp.asarray(random.normal(size=(heads, 2)), jnp.float32)
    cotangent = jnp.asarray(random.normal(size=v.shape), dtype)
    return (q, k, v, x, kernel, bias), cotangent


def _dense(q, k, v, x, kernel, bias):
    """Independent dense oracle with no production projection or attention helpers."""
    coefficients = jnp.einsum("btd,hdr->bthr", x, kernel) + bias
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    c0 = jnp.swapaxes(coefficients[..., 0], 1, 2)[..., None]
    c1 = jnp.swapaxes(coefficients[..., 1], 1, 2)[..., None]
    weights = c0 + c1 * scores
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, weights, 0.0), v)


def _operation(q, k, v, x, kernel, bias, *, backend="blocked"):
    return query_linear_attention(q, k, v, project_coefficients(x, kernel, bias), backend=backend)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _oracle(operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        return _vjp(_dense, tuple(x.astype(jnp.float64) for x in operands), cotangent.astype(jnp.float64))


def _assert_close(actual, expected, tolerance):
    actual, expected = (np.asarray(x, np.float64) for x in (actual, expected))
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    scale = np.linalg.norm(expected)
    if scale == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert np.linalg.norm(actual - expected) / scale < tolerance
        assert np.max(np.abs(actual - expected)) <= 4 * tolerance * np.max(np.abs(expected))


def _compare(actual, expected, tolerance=8e-5):
    _assert_close(actual[0], expected[0], tolerance)
    for observed, reference in zip(actual[1], expected[1], strict=True):
        _assert_close(observed, reference, tolerance)


def _interpret(monkeypatch):
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST))

    def load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)

    monkeypatch.setattr(fused, "_load", load)

    def interpreted(q, k, v, coefficients, *, keep_output_float32=False, query_kernel_guard=None):
        assert keep_output_float32 and query_kernel_guard is not None
        return fused._attention_guarded_fp32(q, k, v, coefficients, query_kernel_guard, True)

    monkeypatch.setattr(fused, "fused_learned_polynomial_attention", interpreted)


@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_forward_and_all_six_gradients_match_independent_fp64(monkeypatch, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    operands, cotangent = _inputs()
    with jax.default_matmul_precision("highest"):
        actual = _vjp(partial(_operation, backend=backend), operands, cotangent)
    _compare(actual, _oracle(operands, cotangent))
    assert actual[1][4].dtype == actual[1][5].dtype == jnp.float32
    assert np.linalg.norm(actual[1][4]) > 0 and np.linalg.norm(actual[1][5]) > 0


def test_blocked_query_coefficients_cross_padding_boundaries():
    operands, _ = _inputs(time=13)
    q, k, v, x, kernel, bias = operands
    actual, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=1, attention_normalization=False, query_block_size=4,
        query_polynomial_coefficients=project_coefficients(x, kernel, bias), return_diagnostics=True)
    expected = _oracle(operands, jnp.ones_like(v))[0]
    _assert_close(actual, expected, 3e-5)
    assert int(diagnostics["row_count"]) == 2 * 13 * 2
    assert int(diagnostics["causal_entry_count"]) == 2 * 2 * 13 * 14 // 2


@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_outputs_and_gradients_are_causal(monkeypatch, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    operands, cotangent = _inputs(time=7)
    prefix = 3
    cotangent = cotangent.at[:, prefix:].set(0)
    operation = partial(_operation, backend=backend)
    actual = _vjp(operation, operands, cotangent)
    changed = tuple(x.at[:, prefix:].add(50) if index < 4 else x for index, x in enumerate(operands))
    altered = _vjp(operation, changed, cotangent)
    np.testing.assert_allclose(actual[0][:, :prefix], altered[0][:, :prefix], rtol=1e-5, atol=1e-6)
    for index, (left, right) in enumerate(zip(actual[1], altered[1], strict=True)):
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=2e-5)
        if index < 4:
            np.testing.assert_array_equal(left[:, prefix:], 0)


@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_actual_kernel_overflow_cannot_hide_behind_zero_values(monkeypatch, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    q = jnp.ones((1, 3, 1, 1), jnp.float32) * 10
    c = jnp.asarray([[[[1., 1.]], [[0., 1e38]], [[1., -1.]]]], jnp.float32)
    output = query_linear_attention(q, q, jnp.zeros_like(q), c, backend=backend)
    assert bool(jnp.isfinite(output[:, 0]).all())
    assert bool(jnp.isnan(output[:, 1]).all())
    assert bool(jnp.isfinite(output[:, 2]).all())


@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_nonfinite_future_key_does_not_poison_earlier_queries(monkeypatch, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    k = q.at[:, 2].set(jnp.inf)
    c = jnp.ones((1, 3, 1, 2), jnp.float32)
    output = query_linear_attention(q, k, q, c, backend=backend)
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [2, 4])
    assert bool(jnp.isnan(output[:, 2]).all())


@pytest.mark.parametrize("updates", [
    {"attention_normalization": True}, {"attention_degree": 2},
    {"attention_coefficients": "fixed"}, {"attention_coefficient_sharing": "global"},
    {"attention_coefficient_parameterization": "rational"}, {"attention_coefficient_initialization": "linear"},
    {"attention_coefficient_conditioning": "key"},
])
def test_rejects_unsupported_query_coefficient_configurations(updates):
    with pytest.raises(ValueError, match="conditioning|Query-conditioned"):
        _model(**updates)


def test_penalty_and_legacy_config_semantics():
    config = _config()
    config.optim.coefficient_penalty = 0.1
    with pytest.raises(ValueError, match="static polynomial"):
        validate_config(config)
    config = _config()
    config.model = replace(config.model, attention_coefficient_conditioning="static")
    current = config_to_dict(config)
    legacy = config_to_dict(config)
    del legacy["model"]["attention_coefficient_conditioning"]
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(current, config_to_dict(_config()))


def test_canonical_backbone_and_initial_function_match_static_control():
    dynamic_config = _model()
    static_config = replace(dynamic_config, attention_coefficient_conditioning="static")
    dynamic, static = build_model(dynamic_config, 13), build_model(static_config, 13)
    dynamic_params = dict(nnx.to_flat_state(nnx.state(dynamic, nnx.Param)))
    static_params = dict(nnx.to_flat_state(nnx.state(static, nnx.Param)))
    for path, parameter in static_params.items():
        if path != ("polynomial_coefficients",):
            np.testing.assert_array_equal(parameter[...], dynamic_params[path][...])
    np.testing.assert_array_equal(dynamic.query_polynomial_kernel[...], 0)
    np.testing.assert_array_equal(dynamic.query_polynomial_bias[...], 1)
    assert dynamic.query_polynomial_kernel.shape == (2, 2, 8, 2)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(dynamic_params))
    assert model_parameter_count(dynamic_config) - model_parameter_count(static_config) == 2 * 2 * 8 * 2
    tokens = jnp.arange(26).reshape(2, 13) % 16
    np.testing.assert_allclose(dynamic(tokens), static(tokens), rtol=2e-5, atol=2e-6)


def test_projection_parameters_remain_fp32_without_weight_decay():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(0))
    state.params["query_polynomial_kernel"][...] += 0.125
    after = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    for name in ("query_polynomial_kernel", "query_polynomial_bias"):
        np.testing.assert_array_equal(state.params[name][...], after.params[name][...])
    x = jnp.asarray(np.random.default_rng(1).normal(size=(1, 13, 8)), jnp.bfloat16)
    values = project_coefficients(x, state.params["query_polynomial_kernel"][0],
                                 state.params["query_polynomial_bias"][0])
    assert values.dtype == jnp.float32


def test_diagnostics_report_actual_token_distributions_separately_from_biases():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(0))
    state.params["query_polynomial_kernel"][...] = jnp.asarray(
        np.random.default_rng(2).normal(size=(2, 2, 8, 2)) * 0.2, jnp.float32)
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["attention_kernel"] == "query_polynomial_unnormalized"
    assert report["attention_coefficient_count"] == 2 * 2 * (8 + 1) * 2
    coefficients = report["coefficients"]
    assert "effective" not in coefficients
    assert np.asarray(coefficients["effective_summary"]["mean"]).shape == (2, 2, 2)
    assert np.min(coefficients["effective_summary"]["std"]) > 0
    np.testing.assert_array_equal(coefficients["projection_bias"], 1)
    model = nnx.merge(state.graphdef, state.params)
    x = model.layers_0.attention_norm(model.token_embedding(jnp.asarray(sample[:, :-1])))
    expected = project_coefficients(x, model.query_polynomial_kernel[0], model.query_polynomial_bias[0])
    np.testing.assert_allclose(coefficients["effective_summary"]["mean"][0], jnp.mean(expected, axis=(0, 1)),
                               rtol=1e-5, atol=1e-6)
    metrics = attention_metrics(report)
    assert metrics["attention/query_polynomial_unnormalized/coefficients/layer00/head00/constant/std"] > 0


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("blocked", "fused"))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_gpu_forward_and_all_gradients_match_fp64(backend, dtype):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    operands, cotangent = _inputs(time=131, dtype=dtype, qk_dim=32, value_dim=17)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(partial(_vjp, partial(_operation, backend=backend)))(operands, cotangent)
    _compare(actual, _oracle(operands, cotangent), tolerance=0.025 if dtype == jnp.bfloat16 else 1e-4)
    assert actual[1][4].dtype == actual[1][5].dtype == jnp.float32


@pytest.mark.gpu
def test_gpu_bf16_production_context_and_nonfinite_kernel_guard():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    operands, cotangent = _inputs(time=2048, dtype=jnp.bfloat16, qk_dim=64, value_dim=64, heads=4)
    operation = jax.jit(partial(_vjp, partial(_operation, backend="fused")))
    actual = operation(operands, cotangent)
    with jax.default_matmul_precision("highest"):
        reference = jax.jit(partial(_vjp, _operation))(operands, cotangent)
    _compare(actual, reference, tolerance=0.025)
    q = jnp.ones((1, 3, 1, 1), jnp.float32) * 10
    c = jnp.asarray([[[[1., 1.]], [[0., 1e38]], [[1., -1.]]]], jnp.float32)
    result = jax.jit(partial(query_linear_attention, backend="fused"))(q, q, jnp.zeros_like(q), c)
    assert bool(jnp.isfinite(result[:, 0]).all()) and bool(jnp.isnan(result[:, 1]).all())
    assert bool(jnp.isfinite(result[:, 2]).all())
