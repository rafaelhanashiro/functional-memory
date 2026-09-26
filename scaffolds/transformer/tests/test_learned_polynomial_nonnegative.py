"""Row-normalized attention with a learned nonnegative kernel p = q1^2 + q2^2.

The factors reproduce the Taylor polynomial at initialization, the blocked
reference matches an independent FP64 evaluation of the sum of squares in value
and every gradient, the kernel is never negative even where the expanded
coefficients would cancel, exact zeros are valid while zero row sums are not, and
configuration, parameter ownership, diagnostics, and training with resume follow
the direct learned coefficients. GPU cases exercise the production BF16 context.
"""

from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import matches_softmax_backbone_initialization
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.nonnegative_polynomial import (
    effective_coefficients_numpy, factor_names, square_sum_polynomial, taylor_square_factors,
)
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.attention.taylor import taylor_attention
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import make_optimizer
from scaffolds.transformer.training.state import create_state

DEGREES = (2, 4)
SHARINGS = ("global", "per_head")
EPS = 1e-6
BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def _taylor(degree):
    return np.array([1 / math.factorial(order) for order in range(degree + 1)])


def _model_config(degree=4, sharing="per_head", qk_norm=False, **updates):
    values = dict(vocab_size=16, max_seq_len=6, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                  qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=degree,
                  attention_normalization=True, attention_coefficients="learned",
                  attention_coefficient_sharing=sharing, attention_coefficient_parameterization="sum_of_squares",
                  attention_backend="blocked", qk_norm=qk_norm)
    return ModelConfig(**{**values, **updates})


def _config(path, degree=4, sharing="per_head", qk_norm=False):
    config = Config()
    config.model = _model_config(degree, sharing, qk_norm)
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 7, 2
    config.train.micro_batch_size = 2
    config.train.workdir = path
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def _operands(time=13, heads=3, qk_dim=8, value_dim=5, seed=11, scale=0.9, dtype=jnp.float32):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(scale * rng.normal(size=(2, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(2, time, heads, value_dim)), dtype)
    cotangent = jnp.asarray(rng.normal(size=v.shape), dtype)
    return (q, k, v), cotangent


def _perturbed_factors(degree, heads, sharing, seed=5):
    rng = np.random.default_rng(seed + degree)
    base = np.asarray(taylor_square_factors(degree))
    values = base * (1 + 0.3 * rng.normal(size=(heads, degree + 1)))
    return jnp.asarray(values[0] if sharing == "global" else values, jnp.float32)


def _dense_reference(q, k, v, factors, degree, log_scale=None):
    """Independent dense (q1^2 + q2^2)(s) V / rowsum over the causal matrix; callers run it in FP64."""
    if log_scale is not None:
        unit = lambda x: x / jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + EPS)
        alpha = jnp.exp(jnp.broadcast_to(log_scale, (q.shape[2], 1)))[None, :, :, None]
        scores = alpha * jnp.einsum("bthd,bshd->bhts", unit(q), unit(k))
    else:
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    factors = jnp.broadcast_to(factors, (q.shape[2], degree + 1))
    half = degree // 2
    q1 = sum(factors[:, order][None, :, None, None] * scores ** order for order in range(half + 1))
    q2 = sum(factors[:, half + 1 + order][None, :, None, None] * scores ** order for order in range(half))
    kernel = jnp.square(q1) + jnp.square(q2)
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    kernel = jnp.where(causal, kernel, 0.0)
    return jnp.einsum("bhts,bshd->bthd", kernel / jnp.sum(kernel, axis=-1, keepdims=True), v)


def _assert_relative(observed, expected, tolerance, label):
    observed, expected = (np.asarray(x, dtype=np.float64) for x in (observed, expected))
    assert observed.shape == expected.shape, label
    assert np.isfinite(observed).all() and np.isfinite(expected).all(), label
    error = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    assert error <= tolerance, f"{label}: relative L2 error {error:.3g} > {tolerance}"


def _fp64_vjp(function, operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(function, *(x.astype(jnp.float64) for x in operands))
        return output, pullback(cotangent.astype(jnp.float64))


@pytest.mark.parametrize("degree", (2, 4, 6))
def test_taylor_factors_reproduce_the_taylor_polynomial(degree):
    factors = np.asarray(taylor_square_factors(degree))
    assert factors.shape == (degree + 1,)
    np.testing.assert_allclose(effective_coefficients_numpy(factors), _taylor(degree), rtol=0, atol=1e-12)
    assert all(value > 0 for value in factors[[0, degree // 2 + 1]])
    assert factor_names(degree)[0] == "q1_0" and factor_names(degree)[-1] == f"q2_{degree // 2 - 1}"
    if degree == 2:
        np.testing.assert_allclose(factors, [1 / math.sqrt(2)] * 3, rtol=0, atol=1e-15)
    for bad in (1, 3, 0, True, 2.0):
        with pytest.raises(ValueError, match="even integer degree"):
            taylor_square_factors(bad)


@pytest.mark.parametrize("degree", DEGREES)
def test_kernel_equals_expanded_coefficients_where_well_conditioned(degree):
    factors = _perturbed_factors(degree, 3, "per_head")
    scores = jnp.asarray(np.random.default_rng(3).uniform(-3, 3, size=(2, 3, 5, 7)), jnp.float32)
    coefficients = effective_coefficients_numpy(np.asarray(factors), degree)
    expanded = sum(coefficients[:, order][None, :, None, None] * np.asarray(scores, np.float64) ** order
                   for order in range(degree + 1))
    np.testing.assert_allclose(square_sum_polynomial(scores, factors, degree), expanded, rtol=2e-5, atol=1e-5)


def test_kernel_stays_nonnegative_where_expanded_coefficients_cancel():
    # p(s) = (s - 20)^2 (s + 3)^2 / 1e4 + (1e-4 (s - 20))^2 is zero-free but nearly zero at s = 20.
    q1 = np.polynomial.polynomial.polymul([-20.0, 1.0], [3.0, 1.0]) / 100.0  # ascending
    factors = jnp.asarray(np.concatenate([q1, [-20e-4, 1e-4]]), jnp.float32)
    scores = jnp.asarray(np.linspace(19.9, 20.1, 4001).reshape(1, 1, 1, -1), jnp.float32)
    kernel = np.asarray(square_sum_polynomial(scores, factors, 4))
    assert np.all(kernel >= 0)
    coefficients = effective_coefficients_numpy(np.asarray(factors, np.float64), 4).astype(np.float32)
    expanded = np.zeros_like(np.asarray(scores))
    for order in range(4, -1, -1):  # FP32 Horner on the expanded coefficients, as a direct kernel would
        expanded = expanded * np.asarray(scores) + coefficients[order]
    assert np.min(expanded) < 0, "the expanded form is expected to cancel below zero here"


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("sharing", SHARINGS)
def test_reference_outputs_and_all_gradients_match_independent_fp64(degree, sharing):
    (q, k, v), cotangent = _operands(seed=20 + degree)
    factors = _perturbed_factors(degree, q.shape[2], sharing)
    blocked = partial(attention_core, backend="blocked", attention_degree=degree, attention_normalization=True)
    function = lambda q, k, v, f: blocked(q, k, v, polynomial_square_factors=f)
    output, pullback = jax.vjp(function, q, k, v, factors)
    grads = pullback(cotangent)
    expected, expected_grads = _fp64_vjp(lambda q, k, v, f: _dense_reference(q, k, v, f, degree),
                                         (q, k, v, factors), cotangent)
    _assert_relative(output, expected, 2e-5, "output")
    for name, observed, reference in zip("qkvf", grads, expected_grads, strict=True):
        _assert_relative(observed, reference, 5e-5, f"gradient {name}")


@pytest.mark.parametrize("degree", DEGREES)
def test_composed_with_qk_norm_matches_independent_fp64(degree):
    (q, k, v), cotangent = _operands(seed=40 + degree)
    factors = _perturbed_factors(degree, q.shape[2], "per_head")
    log_scale = jnp.asarray([[math.log(3.0)], [math.log(1.5)], [math.log(6.0)]], jnp.float32)

    def function(q, k, v, f, log_scale):
        scaled_q, unit_k = cosine_score_operands(q, k, log_scale, eps=EPS)
        return attention_core(scaled_q, unit_k, v, backend="blocked", attention_degree=degree,
                              attention_normalization=True, polynomial_square_factors=f)

    output, pullback = jax.vjp(function, q, k, v, factors, log_scale)
    grads = pullback(cotangent)
    expected, expected_grads = _fp64_vjp(lambda q, k, v, f, s: _dense_reference(q, k, v, f, degree, s),
                                         (q, k, v, factors, log_scale), cotangent)
    _assert_relative(output, expected, 2e-5, "output")
    for name, observed, reference in zip(("q", "k", "v", "factors", "log_scale"), grads, expected_grads, strict=True):
        _assert_relative(observed, reference, 5e-5, f"gradient {name}")


@pytest.mark.parametrize("degree", DEGREES)
def test_taylor_initialized_factors_match_fixed_normalized_taylor(degree):
    (q, k, v), cotangent = _operands(seed=12)
    factors = jnp.broadcast_to(jnp.asarray(taylor_square_factors(degree), jnp.float32), (q.shape[2], degree + 1))
    fixed = partial(taylor_attention, attention_degree=degree, backend="blocked")
    learned = partial(attention_core, backend="blocked", attention_degree=degree, attention_normalization=True,
                      polynomial_square_factors=factors)
    np.testing.assert_allclose(learned(q, k, v), fixed(q, k, v), rtol=2e-5, atol=2e-6)
    expected = jax.vjp(fixed, q, k, v)[1](cotangent)
    actual = jax.vjp(learned, q, k, v)[1](cotangent)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-4, atol=2e-5)


def test_zero_kernel_values_are_valid_and_only_zero_row_sums_poison_their_rows():
    # q1 = s - 1, q2 = 0: p(s) = (s - 1)^2 is zero at s = 1.
    factors = jnp.asarray([[-1.0, 1.0, 0.0]], jnp.float32)
    q = jnp.ones((1, 4, 1, 1), jnp.float32)
    k = jnp.asarray([1.0, 3.0, 1.0, -1.0], jnp.float32).reshape(1, 4, 1, 1)  # kernels 0, 4, 0, 4
    v = jnp.asarray([1.0, 2.0, 3.0, 4.0], jnp.float32).reshape(1, 4, 1, 1)
    output, diagnostics = blocked_dot_product_attention(q, k, v, attention_degree=2, attention_normalization=True,
                                                        polynomial_square_factors=factors, return_diagnostics=True)
    values = np.asarray(output).reshape(-1)
    # Row 0's only causal kernel value is zero: its row sum is zero and the row is poisoned.
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1:], [2.0, 2.0, (2 * 4 + 4 * 4) / 8], rtol=1e-6)
    assert int(diagnostics["invalid_kernel_count"]) == 0
    assert int(diagnostics["invalid_normalizer_count"]) == 1
    assert int(diagnostics["negative_kernel_count"]) == 0 and int(diagnostics["negative_weight_count"]) == 0
    assert float(diagnostics["kernel_min"]) == 0.0


def test_configuration_validation_backend_selection_and_legacy_equality():
    config = _model_config(4)
    assert config.attention_coefficient_parameterization == "sum_of_squares"
    for updates, message in ((dict(attention_degree=3), "even attention_degree"),
                             (dict(attention_degree=1), "even attention_degree"),
                             (dict(attention_normalization=False), "row-normalized"),
                             (dict(attention_coefficients="fixed"), "learned polynomial"),
                             (dict(attention_coefficient_parameterization="softplus"), "Unknown coefficient"),
                             (dict(attention_backend="fused"), "no fused kernel")):
        with pytest.raises(ValueError, match=message):
            _model_config(4, **updates)
    auto = _model_config(4, attention_backend="auto")
    assert resolve_attention_backend(auto, dtype="bfloat16", platform="gpu") == "blocked"
    assert ModelConfig().attention_coefficient_parameterization == "direct"
    # A run recorded before the option existed used direct coefficients.
    direct = Config()
    legacy = config_to_dict(direct)
    legacy["model"].pop("attention_coefficient_parameterization")
    assert config_dicts_equal(legacy, config_to_dict(direct))
    squares = Config()
    squares.model = _model_config(4)
    legacy_squares = config_to_dict(squares)
    legacy_squares["model"].pop("attention_coefficient_parameterization")
    assert not config_dicts_equal(legacy_squares, config_to_dict(squares))
    with pytest.raises(ValueError, match="exactly one"):
        attention_core(*_operands()[0], backend="blocked", attention_degree=2, attention_normalization=True,
                       polynomial_coefficients=jnp.ones((3,)), polynomial_square_factors=jnp.ones((3,)))
    with pytest.raises(ValueError, match="row-normalized"):
        attention_core(*_operands()[0], backend="blocked", attention_degree=2, attention_normalization=False,
                       polynomial_square_factors=jnp.ones((3,)))


@pytest.mark.parametrize("qk_norm", (False, True))
def test_identity_diagnostics_and_mirror(tmp_path, qk_norm):
    config = _config(tmp_path, 4, "per_head", qk_norm)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "learned_polynomial_sum_of_squares"
    assert "attention_signed" not in identity
    assert identity["attention_output_normalization"] == "positive_row_sum"
    assert identity["attention_coefficient_parameterization"] == "sum_of_squares"
    assert identity["attention_coefficient_count"] == 2 * 2 * 5
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert all(layer["negative_kernel_count"] == 0 and "min_abs_normalizer" in layer for layer in report["layers"])
    assert report["coefficients"]["raw_parameter_names"] == ["q1_0", "q1_1", "q1_2", "q2_0", "q2_1"]
    np.testing.assert_allclose(np.asarray(report["coefficients"]["effective"])[1, 1], _taylor(4), rtol=1e-6)
    metrics = tracking.attention_metrics(report)
    assert metrics["attention/learned_polynomial_sum_of_squares/coefficients/constant/mean"] == pytest.approx(1.0, rel=1e-6)
    assert tracking.run_tags(config, {})[:3] == ["learned_polynomial_sum_of_squares", "dk2", "m4"]


@pytest.mark.parametrize("degree", DEGREES)
def test_production_parameter_counts_softmax_backbone_and_no_decay(degree):
    base = load_config(BASE_CONFIG).model
    production = replace(base, attention_degree=degree, attention_normalization=True, attention_backend="blocked",
                         attention_coefficients="learned", attention_coefficient_sharing="per_head",
                         attention_coefficient_parameterization="sum_of_squares", qk_norm=True)
    params = jax.eval_shape(lambda: nnx.state(build_model(production, 0), nnx.Param))
    heads = production.num_layers * production.num_heads
    expected = 199_906_304 + heads * (degree + 1) + heads
    assert sum(x.size for x in jax.tree.leaves(params)) == expected == model_parameter_count(production)
    config = Config()
    config.model = production
    facts = matches_softmax_backbone_initialization(config)
    assert set(facts["mechanism_parameters"]) == {"polynomial_square_factors", "qk_norm_log_scale"}
    assert facts["mechanism_parameters"]["polynomial_square_factors"]["initial"] == list(taylor_square_factors(degree))
    assert facts["mechanism_parameters"]["polynomial_square_factors"]["weight_decay"] is False


def test_factors_receive_no_weight_decay(tmp_path):
    config = _config(tmp_path, 4, "per_head", qk_norm=True)
    state = create_state(config, jax.random.PRNGKey(0))
    tx = make_optimizer(config)
    zeros = jax.tree.map(jnp.zeros_like, state.params)
    updates, _ = tx.update(zeros, tx.init(state.params), state.params)
    updated = jax.tree.map(lambda p, u: p + u, state.params, updates)
    np.testing.assert_array_equal(updated["polynomial_square_factors"][...], state.params["polynomial_square_factors"][...])


def test_model_forward_backward_is_finite_and_causal_with_qk_norm():
    config = _model_config(4, "per_head", qk_norm=True, vocab_size=128, max_seq_len=16, hidden_size=32,
                           num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8, intermediate_size=64)
    model = build_model(config, 0)
    tokens = jnp.arange(32).reshape(2, 16)
    logits, diagnostics = model(tokens, return_diagnostics=True)
    assert bool(jnp.all(jnp.isfinite(logits)))
    for layer in diagnostics:
        assert int(layer["invalid_normalizer_count"]) == 0 and int(layer["negative_kernel_count"]) == 0
    graph, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda candidate: jnp.square(nnx.merge(graph, candidate)(tokens)).mean())(params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads["polynomial_square_factors"][...])) > 0
    altered = tokens.at[:, 9:].set(127)
    np.testing.assert_array_equal(model(altered)[:, :9], logits[:, :9])


@pytest.mark.parametrize("degree", DEGREES)
def test_training_resume_and_identity(tmp_path, degree):
    config = tiny_config(tmp_path / "full")
    config.model.attention_backend = "blocked"
    config.model.attention_degree = degree
    config.model.attention_normalization = True
    config.model.attention_coefficients = "learned"
    config.model.attention_coefficient_sharing = "per_head"
    config.model.attention_coefficient_parameterization = "sum_of_squares"
    config.model.qk_norm = True
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
    assert resumed["restored_step"] == 1
    assert full["attention_kernel"] == resumed["attention_kernel"] == "learned_polynomial_sum_of_squares"
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
            assert step == config.optim.total_steps
            states.append(state)
        finally:
            manager.close()
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    factors = np.asarray(states[0].params["polynomial_square_factors"][...]).reshape(-1, degree + 1)
    assert np.all(factors != np.asarray(taylor_square_factors(degree), np.float32))
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validations = [record for record in records if record.get("event") == "validation"]
    assert validations and all(record["attention_diagnostics"]["valid"] for record in validations)
    json.dumps(records, allow_nan=False)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_production_context_bf16_with_cosine_operands(degree):
    """The blocked reference at the production context: finite, nonnegative, and matching FP64."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")
    (q, k, v), cotangent = _operands(time=2048, heads=2, qk_dim=64, value_dim=64, seed=92, scale=1.0,
                                     dtype=jnp.bfloat16)
    factors = jnp.broadcast_to(jnp.asarray(taylor_square_factors(degree), jnp.float32), (2, degree + 1))
    log_scale = jnp.asarray([[math.log(8.0)], [math.log(3.0)]], jnp.float32)

    def function(q, k, v, f, log_scale):
        scaled_q, unit_k = cosine_score_operands(q, k, log_scale, eps=EPS)
        return attention_core(scaled_q, unit_k, v, backend="blocked", attention_degree=degree,
                              attention_normalization=True, polynomial_square_factors=f)

    @jax.jit
    def forward_and_vjp(operands, cotangent):
        output, pullback = jax.vjp(function, *operands)
        return output, pullback(cotangent)

    output, grads = forward_and_vjp((q, k, v, factors, log_scale), cotangent)
    assert output.dtype == jnp.bfloat16 and bool(jnp.all(jnp.isfinite(output)))
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in grads)
    assert all(float(jnp.linalg.norm(grad.astype(jnp.float32))) > 0 for grad in grads)
    # FP64 on a causal prefix: the positive kernel keeps row sums well conditioned at scale 8.
    prefix = 256
    sliced = (q[:, :prefix], k[:, :prefix], v[:, :prefix], factors, log_scale)
    observed = jax.jit(function)(*sliced).astype(jnp.float32)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _dense_reference(*(x.astype(jnp.float64) for x in sliced[:3]),
                                    sliced[3].astype(jnp.float64), degree, sliced[4].astype(jnp.float64))
    _assert_relative(observed, expected, 2e-2, "BF16 output prefix")
