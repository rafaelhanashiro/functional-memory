"""Normalized attention with separate learned numerator and denominator polynomials.

Tests compare the blocked reference with an independent FP64 dense evaluation of
p_a(s) V / sum p_b(s) for degrees 1-4 and 8, alone and under QK-norm, including
denominators of their own degree, direct or as sums of squares; check the
linear-attention and Taylor initializations, signed and zero denominator sums,
configuration, parameter ownership and decay, the coefficient penalty (training
loss and gradient only, with or without the denominator), diagnostics and the W&B
mirror, and training with resume.
"""

import csv
import json
import math
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import attention_test_paths, matches_softmax_backbone_initialization
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.attention.qk_norm import cosine_score_operands
from scaffolds.transformer.attention.rational_polynomial import (
    LINEAR_START_SQUARE_FACTOR, expand_square_factors, full_denominator, initial_denominator,
    initial_denominator_square_factors,
)
from scaffolds.transformer.attention.taylor import taylor_attention
from scaffolds.transformer.attention.unnormalized import initial_coefficients
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.distributed import make_data_parallel_functions, put_replicated, put_train_batches
from scaffolds.transformer.training.optim import make_optimizer
from scaffolds.transformer.training.regularization import coefficient_penalty_sum, penalty_weights
from scaffolds.transformer.training.state import create_state

DEGREES = (1, 2, 3, 4, 8)
SHARINGS = ("global", "per_head")
EPS = 1e-6
BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/base.yaml"
ARRAYS = ("polynomial_coefficients", "polynomial_denominator_coefficients")


def _model_config(degree=2, sharing="per_head", qk_norm=False, initialization="linear", **updates):
    """A tiny rational model; ``updates`` may set the denominator degree and parameterization."""
    values = dict(vocab_size=16, max_seq_len=6, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                  qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_degree=degree,
                  attention_normalization=True, attention_coefficients="learned",
                  attention_coefficient_parameterization="rational",
                  attention_coefficient_initialization=initialization,
                  attention_coefficient_sharing=sharing, attention_backend="blocked", qk_norm=qk_norm)
    return ModelConfig(**{**values, **updates})


def _config(path, degree=2, sharing="per_head", qk_norm=False, penalty=0.0, initialization="linear"):
    config = Config()
    config.model = _model_config(degree, sharing, qk_norm, initialization)
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 7, 2
    config.train.micro_batch_size = 2
    config.train.workdir = path
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.coefficient_penalty = penalty
    validate_config(config)
    return config


def _operands(time=13, heads=3, qk_dim=8, value_dim=5, seed=11, scale=0.45, dtype=jnp.float32):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(scale * rng.normal(size=(2, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(2, time, heads, value_dim)), dtype)
    cotangent = jnp.asarray(rng.normal(size=v.shape), dtype)
    return (q, k, v), cotangent


def _parameters(degree, heads, sharing, seed=5):
    """A signed numerator and a denominator 1 + small signed terms (well-conditioned sums)."""
    rng = np.random.default_rng(seed + degree)
    factorials = np.array([math.factorial(order) for order in range(degree + 1)])
    numerator = rng.uniform(-0.5, 0.5, size=(heads, degree + 1)) / factorials
    numerator[:, 1] = np.linspace(0.8, 1.2, heads)
    denominator = rng.uniform(-0.05, 0.05, size=(heads, degree)) / factorials[1:]
    if sharing == "global":
        numerator, denominator = numerator[0], denominator[0]
    return jnp.asarray(numerator, jnp.float32), jnp.asarray(denominator, jnp.float32)


def _dense_reference(q, k, v, numerator, denominator, log_scale=None, form="direct"):
    """Independent dense p_a(s) V / sum p_b(s) over the causal matrix; callers run it in FP64.

    ``denominator`` holds b_1..b_d (``form="direct"``) or u_1..u_k, w_1..w_k
    (``form="sum_of_squares"``), evaluated as unexpanded powers.
    """
    if log_scale is not None:
        unit = lambda x: x / jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + EPS)
        alpha = jnp.exp(jnp.broadcast_to(log_scale, (q.shape[2], 1)))[None, :, :, None]
        scores = alpha * jnp.einsum("bthd,bshd->bhts", unit(q), unit(k))
    else:
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    degree, count = numerator.shape[-1] - 1, denominator.shape[-1]
    numerator = jnp.broadcast_to(numerator, (q.shape[2], degree + 1))
    denominator = jnp.broadcast_to(denominator, (q.shape[2], count))
    values = sum(numerator[:, order][None, :, None, None] * scores ** order for order in range(degree + 1))
    if form == "direct":
        sums = 1 + sum(denominator[:, order - 1][None, :, None, None] * scores ** order for order in range(1, count + 1))
    else:
        half = count // 2
        first = 1 + sum(denominator[:, order - 1][None, :, None, None] * scores ** order for order in range(1, half + 1))
        second = sum(denominator[:, half + order - 1][None, :, None, None] * scores ** order
                     for order in range(1, half + 1))
        sums = first ** 2 + second ** 2
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    values, sums = jnp.where(causal, values, 0.0), jnp.where(causal, sums, 0.0)
    return jnp.einsum("bhts,bshd->bthd", values / jnp.sum(sums, axis=-1, keepdims=True), v)


def _assert_relative(observed, expected, tolerance, label):
    observed, expected = (np.asarray(x, dtype=np.float64) for x in (observed, expected))
    assert observed.shape == expected.shape, label
    assert np.isfinite(observed).all() and np.isfinite(expected).all(), label
    error = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    assert error <= tolerance, f"{label}: relative L2 error {error:.3g} > {tolerance}"
    assert np.max(np.abs(observed - expected)) <= 3 * tolerance * np.max(np.abs(expected)), label


def _fp64_vjp(function, operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(function, *(x.astype(jnp.float64) for x in operands))
        return output, pullback(cotangent.astype(jnp.float64))


def _blocked(degree, **options):
    def attend(q, k, v, numerator, denominator):
        return blocked_dot_product_attention(q, k, v, attention_degree=degree, attention_normalization=True,
                                             polynomial_coefficients=numerator,
                                             polynomial_denominator_coefficients=denominator, **options)
    return attend


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("sharing", SHARINGS)
def test_reference_outputs_and_all_gradients_match_independent_fp64(degree, sharing):
    (q, k, v), cotangent = _operands(seed=degree)
    numerator, denominator = _parameters(degree, q.shape[2], sharing)
    operands = (q, k, v, numerator, denominator)
    attend = _blocked(degree, query_block_size=4)
    output, pullback = jax.vjp(attend, *operands)
    grads = pullback(cotangent)
    expected_output, expected_grads = _fp64_vjp(_dense_reference, operands, cotangent)
    _assert_relative(output, expected_output, 2e-6, "output")
    for name, observed, expected in zip(("q", "k", "v", "numerator", "denominator"), grads, expected_grads, strict=True):
        assert observed.shape == expected.shape
        _assert_relative(observed, expected, 5e-6, f"gradient {name}")


@pytest.mark.parametrize("degree", (2, 3, 8))
def test_composed_with_qk_norm_matches_independent_fp64(degree):
    (q, k, v), cotangent = _operands(seed=degree + 20, scale=1.0)
    numerator, denominator = _parameters(degree, q.shape[2], "per_head")
    log_scale = jnp.asarray([[math.log(1.5)], [math.log(0.7)], [0.0]], jnp.float32)

    def composed(q, k, v, numerator, denominator, log_scale):
        q, k = cosine_score_operands(q, k, log_scale, eps=EPS)
        return _blocked(degree)(q, k, v, numerator, denominator)

    operands = (q, k, v, numerator, denominator, log_scale)
    output, pullback = jax.vjp(composed, *operands)
    grads = pullback(cotangent)
    expected_output, expected_grads = _fp64_vjp(_dense_reference, operands, cotangent)
    _assert_relative(output, expected_output, 2e-6, "output")
    for index, (observed, expected) in enumerate(zip(grads, expected_grads, strict=True)):
        _assert_relative(observed, expected, 5e-6, f"gradient {index}")


@pytest.mark.parametrize("degree", (1, 4, 8))
def test_linear_initialization_is_linear_attention_divided_by_visible_keys(degree):
    (q, k, v), _ = _operands(seed=3)
    heads = q.shape[2]
    numerator = jnp.asarray(initial_coefficients(degree, "linear"), jnp.float32)
    denominator = jnp.asarray(initial_denominator(degree, "linear"), jnp.float32)
    np.testing.assert_array_equal(numerator, [0.0, 1.0, *(0.0,) * (degree - 1)])
    np.testing.assert_array_equal(denominator, np.zeros(degree))
    output = attention_core(q, k, v, backend="blocked", attention_degree=degree, attention_normalization=True,
                            polynomial_coefficients=jnp.broadcast_to(numerator, (heads, degree + 1)),
                            polynomial_denominator_coefficients=jnp.broadcast_to(denominator, (heads, degree)))
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    visible = jnp.arange(1, q.shape[1] + 1, dtype=jnp.float32)[None, None, :, None]
    expected = jnp.einsum("bhts,bshd->bthd", jnp.where(causal, scores, 0.0) / visible, v)
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("degree", (1, 2, 4, 8))
def test_taylor_initialization_matches_fixed_normalized_taylor(degree):
    (q, k, v), cotangent = _operands(seed=12)
    heads = q.shape[2]
    numerator = jnp.broadcast_to(jnp.asarray(initial_coefficients(degree, "taylor"), jnp.float32), (heads, degree + 1))
    denominator = jnp.broadcast_to(jnp.asarray(initial_denominator(degree, "taylor"), jnp.float32), (heads, degree))
    np.testing.assert_array_equal(full_denominator(denominator), numerator)
    fixed = partial(taylor_attention, attention_degree=degree, backend="blocked")
    learned = partial(attention_core, backend="blocked", attention_degree=degree, attention_normalization=True,
                      polynomial_coefficients=numerator, polynomial_denominator_coefficients=denominator)
    np.testing.assert_allclose(learned(q, k, v), fixed(q, k, v), rtol=2e-6, atol=5e-7)
    expected = jax.vjp(fixed, q, k, v)[1](cotangent)
    actual = jax.vjp(learned, q, k, v)[1](cotangent)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=3e-6)


def test_zero_denominator_sums_poison_only_their_rows_and_signed_sums_are_valid():
    # p_a(s) = s, p_b(s) = 1 - s on key scores 1, 3, 5, -2: denominators 0, -2, -4, 3.
    numerator = jnp.asarray([[0.0, 1.0]], jnp.float32)
    denominator = jnp.asarray([[-1.0]], jnp.float32)
    q = jnp.ones((1, 4, 1, 1), jnp.float32)
    k = jnp.asarray([1.0, 3.0, 5.0, -2.0], jnp.float32).reshape(1, 4, 1, 1)
    v = jnp.asarray([1.0, 2.0, 3.0, 4.0], jnp.float32).reshape(1, 4, 1, 1)
    output, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=1, attention_normalization=True, polynomial_coefficients=numerator,
        polynomial_denominator_coefficients=denominator, return_diagnostics=True)
    values = np.asarray(output).reshape(-1)
    # Denominator sums 0, -2, -6, -3: row 0 is poisoned; the negative sums are valid.
    assert np.isnan(values[0])
    np.testing.assert_allclose(values[1:], [(1 + 6) / -2, (1 + 6 + 15) / -6, (1 + 6 + 15 - 8) / -3], rtol=1e-6)
    assert int(diagnostics["invalid_normalizer_count"]) == 1
    assert float(diagnostics["normalizer_min"]) == -6.0
    assert float(diagnostics["denominator_min"]) == -4.0 and float(diagnostics["denominator_max"]) == 3.0
    assert float(diagnostics["kernel_min"]) == -2.0 and float(diagnostics["kernel_max"]) == 5.0
    assert int(diagnostics["negative_weight_count"]) > 0
    # Without the zero sum (key scores 2, 3, 5, -2: denominators -1, -2, -4, 3) every row is
    # valid, and the cancellation ratio describes the denominator sums: row 3 has
    # (1 + 2 + 4 + 3) / |-4|.
    shifted = k.at[0, 0].set(2.0)
    output, diagnostics = blocked_dot_product_attention(
        q, shifted, v, attention_degree=1, attention_normalization=True, polynomial_coefficients=numerator,
        polynomial_denominator_coefficients=denominator, return_diagnostics=True)
    np.testing.assert_allclose(np.asarray(output).reshape(-1),
                               [2 / -1, (2 + 6) / -3, (2 + 6 + 15) / -7, (2 + 6 + 15 - 8) / -4], rtol=1e-6)
    assert int(diagnostics["invalid_normalizer_count"]) == 0
    assert float(diagnostics["cancellation_ratio_max"]) == pytest.approx(2.5)
    # A zero numerator with valid denominators is a valid zero output.
    zeros = blocked_dot_product_attention(q, k, v, attention_degree=1, attention_normalization=True,
                                          polynomial_coefficients=jnp.zeros((1, 2), jnp.float32),
                                          polynomial_denominator_coefficients=denominator)
    assert np.isnan(np.asarray(zeros).reshape(-1)[0])
    np.testing.assert_array_equal(np.asarray(zeros).reshape(-1)[1:], [0.0, 0.0, 0.0])


def test_configuration_validation_round_trip_and_legacy_equality(tmp_path):
    config = _config(tmp_path, 2, "per_head", penalty=0.01)
    assert config.model.attention_coefficient_parameterization == "rational"
    with pytest.raises(ValueError, match="normalized"):
        _model_config(attention_normalization=False)
    with pytest.raises(ValueError, match="learned"):
        _model_config(attention_coefficients="fixed", initialization="taylor")
    with pytest.raises(ValueError, match="learned polynomial coefficients only"):
        _model_config(attention_coefficients="fixed", attention_coefficient_parameterization="direct")
    with pytest.raises(ValueError, match="sum-of-squares"):
        _model_config(attention_coefficient_parameterization="sum_of_squares", initialization="linear")
    with pytest.raises(ValueError, match="Unknown coefficient initialization"):
        _model_config(initialization="zeros")
    for bad in (-0.01, float("nan")):
        bad_config = _config(tmp_path)
        bad_config.optim.coefficient_penalty = bad
        with pytest.raises(ValueError, match="coefficient_penalty"):
            validate_config(bad_config)
    unsupported = _config(tmp_path)
    unsupported.model = _model_config(attention_coefficient_parameterization="sum_of_squares", initialization="taylor")
    unsupported.optim.coefficient_penalty = 0.01
    with pytest.raises(ValueError, match="direct or rational"):
        validate_config(unsupported)
    fixed = Config()
    fixed.optim.coefficient_penalty = 0.01
    with pytest.raises(ValueError, match="direct or rational"):
        validate_config(fixed)
    with pytest.raises(ValueError, match="normalized"):
        attention_core(*_operands()[0], backend="blocked", attention_degree=2, attention_normalization=False,
                       polynomial_coefficients=jnp.ones((3,)), polynomial_denominator_coefficients=jnp.zeros((2,)))
    with pytest.raises(ValueError, match="square factors"):
        attention_core(*_operands()[0], backend="blocked", attention_degree=2, attention_normalization=True,
                       polynomial_square_factors=jnp.ones((3,)), polynomial_denominator_coefficients=jnp.zeros((2,)))
    with pytest.raises(ValueError, match="shape"):
        attention_core(*_operands()[0], backend="blocked", attention_degree=2, attention_normalization=True,
                       polynomial_coefficients=jnp.ones((3,)), polynomial_denominator_coefficients=jnp.zeros((2, 3)))
    save_config(config, tmp_path / "recipe.yaml")
    loaded = load_config(tmp_path / "recipe.yaml")
    assert loaded.model.attention_coefficient_initialization == "linear"
    assert loaded.optim.coefficient_penalty == 0.01
    assert config_to_dict(loaded) == config_to_dict(config)
    # Recipes recorded before these fields existed compare equal to their resolved defaults.
    legacy = config_to_dict(_config(tmp_path, initialization="taylor"))
    legacy["model"]["attention_coefficient_parameterization"] = "direct"
    resolved = json.loads(json.dumps(legacy))
    del legacy["model"]["attention_coefficient_initialization"]
    del legacy["optim"]["coefficient_penalty"]
    assert config_dicts_equal(legacy, resolved)
    assert not config_dicts_equal(legacy, {**resolved, "optim": {**resolved["optim"], "coefficient_penalty": 0.5}})


@pytest.mark.parametrize("initialization", ("linear", "taylor"))
def test_parameter_specs_counts_softmax_backbone_and_no_decay(tmp_path, initialization):
    degree = 8
    base = load_config(BASE_CONFIG).model
    production = replace(base, attention_degree=degree, attention_normalization=True, attention_backend="blocked",
                         attention_coefficients="learned", attention_coefficient_sharing="per_head",
                         attention_coefficient_parameterization="rational",
                         attention_coefficient_initialization=initialization, qk_norm=True,
                         initialization="transformer_reference")
    specs = attention_parameter_specs(production)
    assert [spec.name for spec in specs] == [*ARRAYS, "qk_norm_log_scale"]
    assert all(spec.weight_decay is False for spec in specs)
    assert len(specs[0].initial) == degree + 1 and len(specs[1].initial) == degree
    params = jax.eval_shape(lambda: nnx.state(build_model(production, 0), nnx.Param))
    heads = production.num_layers * production.num_heads
    expected = 199_906_304 + heads * (2 * degree + 1) + heads
    assert sum(x.size for x in jax.tree.leaves(params)) == expected == model_parameter_count(production)
    config = Config()
    config.model = production
    facts = matches_softmax_backbone_initialization(config)
    assert set(facts["mechanism_parameters"]) == {*ARRAYS, "qk_norm_log_scale"}
    expected_numerator = ([0.0, 1.0] + [0.0] * (degree - 1) if initialization == "linear"
                          else [1 / math.factorial(order) for order in range(degree + 1)])
    assert facts["mechanism_parameters"]["polynomial_coefficients"]["initial"] == expected_numerator
    assert facts["mechanism_parameters"]["polynomial_denominator_coefficients"]["initial"] == (
        [0.0] * degree if initialization == "linear" else expected_numerator[1:])
    # AdamW leaves both arrays alone under zero gradients: no decoupled decay.
    tiny = _config(tmp_path, 4, "per_head", qk_norm=True, initialization=initialization)
    state = create_state(tiny, jax.random.PRNGKey(0))
    tx = make_optimizer(tiny)
    zeros = jax.tree.map(jnp.zeros_like, state.params)
    updates, _ = tx.update(zeros, tx.init(state.params), state.params)
    updated = jax.tree.map(lambda p, u: p + u, state.params, updates)
    for name in ARRAYS:
        np.testing.assert_array_equal(updated[name][...], state.params[name][...])
    assert "scaffolds/transformer/tests/test_learned_polynomial_rational.py" in attention_test_paths(
        config_to_dict(tiny), "blocked", True)


def test_penalty_weights_and_sum_follow_the_definition():
    assert penalty_weights(1) == ((1.0, 0.0), (1.0,))
    assert penalty_weights(3) == ((1.0, 0.0, 4.0, 9.0), (1.0, 4.0, 9.0))
    with pytest.raises(ValueError):
        penalty_weights(0)
    model = _model_config(3, "per_head", initialization="taylor")
    rng = np.random.default_rng(1)
    numerator = rng.normal(size=(2, 2, 4))
    denominator = rng.normal(size=(2, 2, 3))
    params = {"polynomial_coefficients": jnp.asarray(numerator, jnp.float32),
              "polynomial_denominator_coefficients": jnp.asarray(denominator, jnp.float32)}
    expected = (np.sum(numerator[..., 0] ** 2) + np.sum(4 * numerator[..., 2] ** 2 + 9 * numerator[..., 3] ** 2)
                + np.sum(denominator[..., 0] ** 2 + 4 * denominator[..., 1] ** 2 + 9 * denominator[..., 2] ** 2))
    assert float(coefficient_penalty_sum(params, model)) == pytest.approx(expected, rel=1e-5)
    direct = _model_config(3, "global", initialization="taylor", attention_coefficient_parameterization="direct")
    shared = {"polynomial_coefficients": jnp.asarray(numerator[0, 0], jnp.float32)}
    assert float(coefficient_penalty_sum(shared, direct)) == pytest.approx(
        numerator[0, 0, 0] ** 2 + 4 * numerator[0, 0, 2] ** 2 + 9 * numerator[0, 0, 3] ** 2, rel=1e-5)
    # The linear-attention initialization is the penalty's minimum apart from a_1.
    linear = create_state(_config(Path("unused"), 3, "per_head"), jax.random.PRNGKey(0)).params
    assert float(coefficient_penalty_sum(linear, model)) == 0.0
    with pytest.raises(ValueError, match="direct or rational"):
        coefficient_penalty_sum(params, _model_config(2, attention_coefficient_parameterization="sum_of_squares",
                                                      initialization="taylor"))


def test_loss_reports_the_nll_while_the_gradient_includes_the_penalty(tmp_path):
    config = _config(tmp_path, 2, "per_head", qk_norm=True, initialization="taylor")
    state = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.asarray(np.arange(14).reshape(2, 7) % 16)
    rng = jax.random.PRNGKey(2)
    nll, grads = loop.loss_and_grads(state, batch, rng)
    penalized_nll, penalized_grads = loop.loss_and_grads(state, batch, rng, coefficient_penalty=0.25)
    assert float(nll) == float(penalized_nll)
    penalty_grads = jax.grad(lambda params: coefficient_penalty_sum(params, config.model))(state.params)
    for (path, observed), (_, expected), (_, penalty) in zip(
            jax.tree_util.tree_leaves_with_path(penalized_grads), jax.tree_util.tree_leaves_with_path(grads),
            jax.tree_util.tree_leaves_with_path(penalty_grads), strict=True):
        if any(name in str(path) for name in ARRAYS):
            np.testing.assert_allclose(np.asarray(observed) - np.asarray(expected), 0.25 * np.asarray(penalty),
                                       rtol=1e-5, atol=1e-7)
            assert np.linalg.norm(np.asarray(penalty)) > 0
        else:
            np.testing.assert_array_equal(observed, expected)
    # The checked step returns the NLL too, and its update follows the penalized gradient.
    updated, loss = loop.train_step(state, batch, rng, coefficient_penalty=0.25)
    assert float(loss) == float(nll)
    plain, _ = loop.train_step(state, batch, rng)
    assert not np.array_equal(np.asarray(updated.params["polynomial_denominator_coefficients"][...]),
                              np.asarray(plain.params["polynomial_denominator_coefficients"][...]))
    accumulated, accumulated_loss = loop.accumulated_train_step(
        state, batch.reshape(2, 1, 7), jax.random.split(rng, 2), coefficient_penalty=0.25)
    assert np.isfinite(float(accumulated_loss))
    # The evaluation loss is the cross-entropy alone.
    assert float(loop.eval_step(state, batch)) == pytest.approx(float(loop.cross_entropy_loss(
        nnx.merge(state.graphdef, state.params)(batch[:, :-1]), batch[:, 1:])), rel=1e-6)


@pytest.mark.skipif(jax.local_device_count() < 2, reason="requires two JAX devices")
def test_distributed_step_applies_the_same_penalty(tmp_path):
    config = _config(tmp_path, 2, "per_head", qk_norm=True, initialization="taylor")
    config.data.batch_size, config.train.micro_batch_size = 4, 2
    validate_config(config)
    state = create_state(config, jax.random.PRNGKey(0))
    batch = np.arange(28, dtype=np.int32).reshape(4, 7) % 16
    expected, expected_loss = loop.train_step(state, jnp.asarray(batch), jax.random.PRNGKey(1), coefficient_penalty=0.3)
    functions = make_data_parallel_functions(state, tuple(jax.local_devices()), coefficient_penalty=0.3)
    replicated = put_replicated(state, functions.replicated_sharding)
    batches = put_train_batches(batch.reshape(1, 4, 7), functions.train_batch_sharding)
    actual, actual_loss = functions.train_step(replicated, batches, jax.random.PRNGKey(1))
    np.testing.assert_allclose(float(actual_loss), float(expected_loss), rtol=1e-6)
    for name in ARRAYS:
        np.testing.assert_allclose(np.asarray(actual.params[name][...]), np.asarray(expected.params[name][...]),
                                   rtol=1e-6, atol=1e-7)
    # The compiled step donates the replicated buffers, which share device memory with
    # the original state, so the control starts from a freshly created identical state.
    control = create_state(config, jax.random.PRNGKey(0))
    unpenalized = make_data_parallel_functions(control, tuple(jax.local_devices()))
    plain, _ = unpenalized.train_step(put_replicated(control, unpenalized.replicated_sharding),
                                      put_train_batches(batch.reshape(1, 4, 7), unpenalized.train_batch_sharding),
                                      jax.random.PRNGKey(1))
    assert not np.allclose(np.asarray(plain.params["polynomial_denominator_coefficients"][...]),
                           np.asarray(actual.params["polynomial_denominator_coefficients"][...]))


@pytest.mark.parametrize("qk_norm", (False, True))
def test_identity_diagnostics_and_mirror(tmp_path, qk_norm):
    config = _config(tmp_path, 3, "per_head", qk_norm, penalty=0.01)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "learned_polynomial_rational"
    assert identity["attention_signed"] is True
    assert identity["attention_output_normalization"] == "signed_denominator_row_sum"
    assert identity["attention_coefficient_count"] == 2 * 2 * 7
    assert identity["attention_coefficient_initialization"] == "linear"
    assert identity["coefficient_penalty"] == 0.01 and "a_1 unpenalized" in identity["coefficient_penalty_formula"]
    assert "q(x) = 1 + sum_(1<=r<=3) b_r x^r" in identity["attention_formula"]
    assert identity.get("qk_norm", False) is qk_norm
    state = create_state(config, jax.random.PRNGKey(0))
    # A negative linear denominator term makes some denominators negative: valid here.
    state.params["polynomial_denominator_coefficients"][...] = (
        state.params["polynomial_denominator_coefficients"][...].at[..., 0].set(-3.0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == "unshifted_learned_polynomial"
    assert all("denominator_min" in layer and "min_abs_normalizer" in layer for layer in report["layers"])
    coefficients = report["coefficients"]
    assert coefficients["raw_parameter_names"] == ["a0", "a1", "a2", "a3", "b1", "b2", "b3"]
    assert coefficients["effective_coefficient_names"][:2] == ["numerator_constant", "numerator_linear"]
    assert coefficients["effective_coefficient_names"][4:] == ["denominator_linear", "denominator_quadratic", "denominator_cubic"]
    assert coefficients["effective"][1][1][1] == 1.0 and coefficients["effective"][1][1][4] == -3.0
    assert ("qk_norm_scale" in report) is qk_norm
    metrics = tracking.attention_metrics(report)
    assert metrics["attention/learned_polynomial_rational/coefficients/numerator_linear/mean"] == 1.0
    assert metrics["attention/learned_polynomial_rational/coefficients/denominator_linear/min"] == -3.0
    assert tracking.run_tags(config, {}) == (["learned_polynomial_rational", "dk2", "m3"] + (["qk_norm"] if qk_norm else []))


def test_fused_diagnostics_preserve_output_and_parameter_gradients(monkeypatch):
    from scaffolds.transformer.attention.backends import fused_normalized_polynomial

    (q, k, v), _ = _operands(time=5)
    numerator = jnp.asarray([1.0, 0.5, 0.25], jnp.float32)
    denominator = jnp.asarray([0.1, 0.2], jnp.float32)

    def fused(q, k, v, parameters, *, parameterization, denominator_degree, denominator_form):
        assert (parameterization, denominator_degree, denominator_form) == ("rational", 2, "direct")
        assert parameters.shape == (q.shape[2], 5) and parameters.dtype == jnp.float32
        return jnp.broadcast_to(jnp.sum(parameters, axis=-1)[None, None, :, None], v.shape)

    monkeypatch.setattr(fused_normalized_polynomial, "fused_normalized_polynomial_attention", fused)
    apply = lambda a, b: attention_core(q, k, v, attention_degree=2, attention_normalization=True, backend="fused",
                                        return_diagnostics=True, polynomial_coefficients=a,
                                        polynomial_denominator_coefficients=b)
    output, diagnostics = apply(numerator, denominator)
    _, expected_diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=2, attention_normalization=True, return_diagnostics=True,
        polynomial_coefficients=numerator, polynomial_denominator_coefficients=denominator)
    np.testing.assert_array_equal(output, jnp.full_like(v, jnp.sum(numerator) + jnp.sum(denominator)))
    for name in expected_diagnostics:
        np.testing.assert_array_equal(diagnostics[name], expected_diagnostics[name])
    gradients = jax.grad(lambda a, b: jnp.sum(apply(a, b)[0]), argnums=(0, 1))(numerator, denominator)
    np.testing.assert_array_equal(gradients[0], jnp.full_like(numerator, v.size))
    np.testing.assert_array_equal(gradients[1], jnp.full_like(denominator, v.size))


def test_training_resume_identity_and_recorded_penalty(tmp_path):
    config = tiny_config(tmp_path / "full")
    config.model.attention_backend = "blocked"
    config.model.attention_degree = 2
    config.model.attention_normalization = True
    config.model.attention_coefficients = "learned"
    config.model.attention_coefficient_parameterization = "rational"
    config.model.attention_coefficient_initialization = "linear"
    config.model.attention_coefficient_sharing = "per_head"
    config.model.qk_norm = True
    config.optim.coefficient_penalty = 0.01
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
    assert resumed["restored_step"] == 1
    assert full["attention_kernel"] == resumed["attention_kernel"] == "learned_polynomial_rational"
    assert full["coefficient_penalty"] == 0.01
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
    numerator = np.asarray(states[0].params["polynomial_coefficients"][...])
    denominator = np.asarray(states[0].params["polynomial_denominator_coefficients"][...])
    assert numerator.shape == (1, 2, 3) and denominator.shape == (1, 2, 2)
    assert np.all(numerator != np.array([0.0, 1.0, 0.0], np.float32)) and np.all(denominator != 0)
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    validations = [record for record in records if record.get("event") == "validation"]
    assert validations and all(record["attention_diagnostics"]["valid"] for record in validations)
    assert all(record["attention_kernel"] == "learned_polynomial_rational" for record in validations)
    # The recorded penalty is the term added to the loss at that step's parameters;
    # the validation NLL is the plain cross-entropy of the checkpointed model.
    terminal = validations[-1]
    assert terminal["train_coefficient_penalty"] == pytest.approx(
        0.01 * float(coefficient_penalty_sum(states[0].params, config.model)), rel=1e-5)
    assert terminal["train_coefficient_penalty"] > 0
    json.dumps(records, allow_nan=False)


def test_run_index_exposes_initialization_and_penalty(tmp_path):
    from scaffolds.records import create_run, regenerate_index

    study = tmp_path / "experiments/demo"
    study.mkdir(parents=True)
    (study / "sweep.yaml").write_text(json.dumps({"study_id": "demo", "run_ids": []}))
    recipe = config_to_dict(_config(tmp_path / "placeholder", 8, "per_head", qk_norm=True, penalty=0.01))
    create_run(study, recipe, label="rational")
    regenerate_index(study)
    with (study / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["attention_family"] == "learned_polynomial"
    assert row["coefficient_parameterization"] == "rational"
    assert row["coefficient_initialization"] == "linear"
    assert row["coefficient_penalty"] == "0.01"
    assert row["coefficient_sharing"] == "per_head"


DENOMINATORS = [(8, 1, "direct"), (8, 2, "sum_of_squares"), (3, 4, "sum_of_squares"), (2, 5, "direct")]
DENOMINATOR_ARRAYS = {"direct": "polynomial_denominator_coefficients",
                      "sum_of_squares": "polynomial_denominator_square_factors"}


def _denominator_parameters(count, heads, form, sharing, seed=9):
    """Small signed b_r, or u_r near 0 and w_r of order 0.2: well-conditioned sums."""
    rng = np.random.default_rng(seed + count)
    if form == "direct":
        values = rng.uniform(-0.05, 0.05, size=(heads, count)) / np.array(
            [math.factorial(order) for order in range(1, count + 1)])
    else:
        half = count // 2
        values = np.concatenate([rng.uniform(-0.1, 0.1, size=(heads, half)),
                                 rng.uniform(0.1, 0.3, size=(heads, half))], axis=-1)
    return jnp.asarray(values[0] if sharing == "global" else values, jnp.float32)


@pytest.mark.parametrize("degree,count,form", DENOMINATORS)
@pytest.mark.parametrize("sharing", SHARINGS)
def test_denominators_of_their_own_degree_match_independent_fp64(degree, count, form, sharing):
    (q, k, v), cotangent = _operands(seed=degree + count)
    numerator, _ = _parameters(degree, q.shape[2], sharing)
    denominator = _denominator_parameters(count, q.shape[2], form, sharing)

    def attend(q, k, v, numerator, denominator):
        return blocked_dot_product_attention(q, k, v, attention_degree=degree, attention_normalization=True,
                                             polynomial_coefficients=numerator, query_block_size=4,
                                             **{DENOMINATOR_ARRAYS[form]: denominator})

    operands = (q, k, v, numerator, denominator)
    output, pullback = jax.vjp(attend, *operands)
    grads = pullback(cotangent)
    expected_output, expected_grads = _fp64_vjp(partial(_dense_reference, form=form), operands, cotangent)
    _assert_relative(output, expected_output, 2e-6, "output")
    for name, observed, expected in zip(("q", "k", "v", "numerator", "denominator"), grads, expected_grads, strict=True):
        _assert_relative(observed, expected, 5e-6, f"gradient {name}")


@pytest.mark.parametrize("degree", (2, 4, 8))
def test_square_factor_initializations(degree):
    taylor = initial_denominator_square_factors(degree, "taylor")
    np.testing.assert_allclose(expand_square_factors(taylor),
                               [1 / math.factorial(order) for order in range(1, degree + 1)], rtol=0, atol=1e-12)
    linear = initial_denominator_square_factors(degree, "linear")
    half = degree // 2
    assert linear == (0.0,) * half + (LINEAR_START_SQUARE_FACTOR,) + (0.0,) * (half - 1)
    expected = np.zeros(degree)
    expected[1] = LINEAR_START_SQUARE_FACTOR ** 2
    np.testing.assert_allclose(expand_square_factors(linear), expected, rtol=0, atol=1e-18)
    with pytest.raises(ValueError, match="even"):
        initial_denominator_square_factors(3)


def test_linear_start_of_a_square_factor_denominator_can_move_every_factor():
    """With every w_r = 0 their gradient vanishes; the linear start keeps w_1 nonzero."""
    (q, k, v), cotangent = _operands(seed=4, scale=1.0)
    heads = q.shape[2]
    numerator = jnp.broadcast_to(jnp.asarray(initial_coefficients(8, "linear"), jnp.float32), (heads, 9))

    def loss(factors):
        output = blocked_dot_product_attention(q, k, v, attention_degree=8, attention_normalization=True,
                                               polynomial_coefficients=numerator,
                                               polynomial_denominator_square_factors=factors)
        return jnp.sum(output * cotangent)

    start = jnp.broadcast_to(jnp.asarray(initial_denominator_square_factors(2, "linear"), jnp.float32), (heads, 2))
    assert np.all(np.asarray(jax.grad(loss)(start)) != 0)
    assert not np.any(np.asarray(jax.grad(loss)(jnp.zeros((heads, 2), jnp.float32)))[:, 1])
    # The start is linear attention divided by the visible keys, up to 1e-6 s^2 in the denominator.
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    visible = jnp.arange(1, q.shape[1] + 1, dtype=jnp.float32)[None, None, :, None]
    expected = jnp.einsum("bhts,bshd->bthd", jnp.where(causal, scores, 0.0) / visible, v)
    output = blocked_dot_product_attention(q, k, v, attention_degree=8, attention_normalization=True,
                                           polynomial_coefficients=numerator, polynomial_denominator_square_factors=start)
    np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-5)


def test_denominator_configuration(tmp_path):
    for degree, form in ((1, "direct"), (2, "sum_of_squares"), (None, "sum_of_squares")):
        config = _model_config(8 if degree else 4, attention_denominator_degree=degree,
                               attention_denominator_parameterization=form)
        expected = degree or 4
        spec = attention_parameter_specs(config)[1]
        assert spec.name == DENOMINATOR_ARRAYS[form] and len(spec.initial) == expected
    with pytest.raises(ValueError, match="rational kernels only"):
        _model_config(initialization="taylor", attention_coefficient_parameterization="direct",
                      attention_denominator_degree=1)
    with pytest.raises(ValueError, match="rational kernels only"):
        _model_config(initialization="taylor", attention_coefficient_parameterization="direct",
                      attention_denominator_parameterization="sum_of_squares")
    with pytest.raises(ValueError, match="even denominator degree"):
        _model_config(8, attention_denominator_degree=3, attention_denominator_parameterization="sum_of_squares")
    with pytest.raises(ValueError, match="even denominator degree"):
        _model_config(3, attention_denominator_parameterization="sum_of_squares")
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="attention_denominator_degree"):
            _model_config(attention_denominator_degree=bad)
    with pytest.raises(ValueError, match="Unknown denominator parameterization"):
        _model_config(attention_denominator_parameterization="positive")
    config = _config(tmp_path, 8, penalty=0.01)
    config.model = _model_config(8, attention_denominator_degree=2, attention_denominator_parameterization="sum_of_squares")
    with pytest.raises(ValueError, match="coefficient_penalty_denominator: false"):
        validate_config(config)
    config.optim.coefficient_penalty_denominator = False
    validate_config(config)
    save_config(config, tmp_path / "recipe.yaml")
    loaded = load_config(tmp_path / "recipe.yaml")
    assert (loaded.model.attention_denominator_degree, loaded.model.attention_denominator_parameterization,
            loaded.optim.coefficient_penalty_denominator) == (2, "sum_of_squares", False)
    plain = Config()
    plain.optim.coefficient_penalty_denominator = False
    with pytest.raises(ValueError, match="rational kernels only"):
        validate_config(plain)
    plain.optim.coefficient_penalty_denominator = "no"
    with pytest.raises(ValueError, match="boolean"):
        validate_config(plain)
    # Recipes recorded before the denominator options compare equal to their defaults.
    resolved = config_to_dict(_config(tmp_path, 3))
    legacy = json.loads(json.dumps(resolved))
    for key in ("attention_denominator_degree", "attention_denominator_parameterization"):
        del legacy["model"][key]
    del legacy["optim"]["coefficient_penalty_denominator"]
    assert config_dicts_equal(legacy, resolved)


def test_penalty_can_leave_the_denominator_out(tmp_path):
    model = _model_config(3, initialization="taylor", attention_denominator_degree=1)
    assert penalty_weights(3, 1) == ((1.0, 0.0, 4.0, 9.0), (1.0,))
    rng = np.random.default_rng(3)
    numerator, denominator = rng.normal(size=(2, 2, 4)), rng.normal(size=(2, 2, 1))
    params = {"polynomial_coefficients": jnp.asarray(numerator, jnp.float32),
              "polynomial_denominator_coefficients": jnp.asarray(denominator, jnp.float32)}
    numerator_only = np.sum(numerator[..., 0] ** 2 + 4 * numerator[..., 2] ** 2 + 9 * numerator[..., 3] ** 2)
    assert float(coefficient_penalty_sum(params, model, include_denominator=False)) == pytest.approx(numerator_only, rel=1e-5)
    assert float(coefficient_penalty_sum(params, model)) == pytest.approx(numerator_only + np.sum(denominator ** 2), rel=1e-5)
    square = _model_config(3, initialization="taylor", attention_denominator_degree=2,
                           attention_denominator_parameterization="sum_of_squares")
    square_params = {"polynomial_coefficients": params["polynomial_coefficients"],
                     "polynomial_denominator_square_factors": jnp.ones((2, 2, 2), jnp.float32)}
    assert float(coefficient_penalty_sum(square_params, square, include_denominator=False)) == pytest.approx(
        numerator_only, rel=1e-5)
    with pytest.raises(ValueError, match="direct denominator"):
        coefficient_penalty_sum(square_params, square)
    # In training, the excluded denominator receives exactly the unpenalized gradient.
    config = _config(tmp_path, 2, "per_head", qk_norm=True, initialization="taylor")
    config.model = _model_config(2, qk_norm=True, initialization="taylor", attention_denominator_degree=2,
                                 attention_denominator_parameterization="sum_of_squares")
    state = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.asarray(np.arange(14).reshape(2, 7) % 16)
    rng_key = jax.random.PRNGKey(2)
    nll, plain = loop.loss_and_grads(state, batch, rng_key)
    penalized_nll, penalized = loop.loss_and_grads(state, batch, rng_key, coefficient_penalty=0.5,
                                                   coefficient_penalty_denominator=False)
    assert float(nll) == float(penalized_nll)
    np.testing.assert_array_equal(penalized["polynomial_denominator_square_factors"][...],
                                  plain["polynomial_denominator_square_factors"][...])
    assert not np.array_equal(np.asarray(penalized["polynomial_coefficients"][...]),
                              np.asarray(plain["polynomial_coefficients"][...]))


@pytest.mark.parametrize("count,form", [(1, "direct"), (2, "sum_of_squares")])
def test_denominator_parameter_counts_and_softmax_backbone(count, form):
    base = load_config(BASE_CONFIG).model
    production = replace(base, attention_degree=8, attention_normalization=True, attention_backend="blocked",
                         attention_coefficients="learned", attention_coefficient_sharing="per_head",
                         attention_coefficient_parameterization="rational", attention_coefficient_initialization="linear",
                         attention_denominator_degree=count, attention_denominator_parameterization=form,
                         qk_norm=True, initialization="transformer_reference")
    params = jax.eval_shape(lambda: nnx.state(build_model(production, 0), nnx.Param))
    heads = production.num_layers * production.num_heads
    expected = 199_906_304 + heads * (9 + count) + heads
    assert sum(x.size for x in jax.tree.leaves(params)) == expected == model_parameter_count(production)
    config = Config()
    config.model = production
    facts = matches_softmax_backbone_initialization(config)
    assert set(facts["mechanism_parameters"]) == {"polynomial_coefficients", DENOMINATOR_ARRAYS[form], "qk_norm_log_scale"}
    assert facts["mechanism_parameters"][DENOMINATOR_ARRAYS[form]]["initial"] == (
        [0.0] if form == "direct" else [0.0, LINEAR_START_SQUARE_FACTOR])
    assert facts["mechanism_parameters"][DENOMINATOR_ARRAYS[form]]["weight_decay"] is False


def test_square_factor_denominator_identity_diagnostics_and_mirror(tmp_path):
    config = _config(tmp_path, 3, "per_head", qk_norm=True, penalty=0.01)
    config.model = _model_config(3, qk_norm=True, attention_denominator_degree=2,
                                 attention_denominator_parameterization="sum_of_squares")
    config.optim.coefficient_penalty_denominator = False
    validate_config(config)
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "learned_polynomial_rational"
    assert identity["attention_output_normalization"] == "nonnegative_denominator_row_sum"
    assert (identity["attention_denominator_degree"], identity["attention_denominator_parameterization"]) == (
        2, "sum_of_squares")
    assert identity["attention_coefficient_count"] == 2 * 2 * 6
    assert identity["coefficient_penalty_denominator"] is False
    assert "denominator unpenalized" in identity["coefficient_penalty_formula"]
    assert "b_r" not in identity["coefficient_penalty_formula"]
    assert f"except w_1 = {LINEAR_START_SQUARE_FACTOR:g}" in identity["attention_formula"]
    state = create_state(config, jax.random.PRNGKey(0))
    state.params["polynomial_denominator_square_factors"][...] = jnp.broadcast_to(
        jnp.asarray([0.5, 0.25], jnp.float32), (2, 2, 2))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert all(layer["denominator_min"] >= 0 for layer in report["layers"])
    coefficients = report["coefficients"]
    assert coefficients["raw_parameter_names"] == ["a0", "a1", "a2", "a3", "u1", "w1"]
    assert coefficients["effective_coefficient_names"][4:] == ["denominator_linear", "denominator_quadratic"]
    # (1 + 0.5 s)^2 + (0.25 s)^2 = 1 + s + 0.3125 s^2.
    np.testing.assert_allclose(coefficients["effective"][1][1][4:], [1.0, 0.3125])
    metrics = tracking.attention_metrics(report)
    assert metrics["attention/learned_polynomial_rational/coefficients/denominator_quadratic/mean"] == pytest.approx(0.3125)


@pytest.mark.parametrize("count,form", [(1, "direct"), (2, "sum_of_squares")])
def test_fused_wrapper_passes_the_denominator_layout(monkeypatch, count, form):
    from scaffolds.transformer.attention.backends import fused_normalized_polynomial

    (q, k, v), _ = _operands(time=5)
    numerator = jnp.asarray([1.0, 0.5, 0.25], jnp.float32)
    denominator = jnp.full((count,), 0.1, jnp.float32)

    def fused(q, k, v, parameters, *, parameterization, denominator_degree, denominator_form):
        assert (parameterization, denominator_degree, denominator_form) == ("rational", count, form)
        assert parameters.shape == (q.shape[2], 3 + count) and parameters.dtype == jnp.float32
        return jnp.broadcast_to(jnp.sum(parameters, axis=-1)[None, None, :, None], v.shape)

    monkeypatch.setattr(fused_normalized_polynomial, "fused_normalized_polynomial_attention", fused)
    apply = lambda a, b: attention_core(q, k, v, attention_degree=2, attention_normalization=True, backend="fused",
                                        polynomial_coefficients=a, **{DENOMINATOR_ARRAYS[form]: b})
    np.testing.assert_array_equal(apply(numerator, denominator), jnp.full_like(v, jnp.sum(numerator) + jnp.sum(denominator)))
    gradients = jax.grad(lambda a, b: jnp.sum(apply(a, b)), argnums=(0, 1))(numerator, denominator)
    np.testing.assert_array_equal(gradients[1], jnp.full_like(denominator, v.size))


@pytest.mark.parametrize("count,form", [(1, "direct"), (2, "sum_of_squares")])
def test_training_resume_with_a_denominator_of_its_own_degree(tmp_path, count, form):
    config = tiny_config(tmp_path / "full")
    model = config.model
    model.attention_backend, model.attention_degree, model.attention_normalization = "blocked", 8, True
    model.attention_coefficients, model.attention_coefficient_parameterization = "learned", "rational"
    model.attention_coefficient_initialization, model.attention_coefficient_sharing = "linear", "per_head"
    model.attention_denominator_degree, model.attention_denominator_parameterization = count, form
    model.qk_norm = True
    config.optim.coefficient_penalty, config.optim.coefficient_penalty_denominator = 0.01, False
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
    assert full["attention_denominator_degree"] == count and full["coefficient_penalty_denominator"] is False
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            states.append(restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))[0])
        finally:
            manager.close()
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    denominator = np.asarray(states[0].params[DENOMINATOR_ARRAYS[form]][...])
    assert denominator.shape == (1, 2, count) and np.all(np.isfinite(denominator))
    start = np.asarray(initial_denominator(count, "linear") if form == "direct"
                       else initial_denominator_square_factors(count, "linear"))
    assert np.all(denominator != start)
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert all(record["attention_diagnostics"]["valid"] for record in records if record.get("event") == "validation")
    terminal = [record for record in records if record.get("event") == "validation"][-1]
    assert terminal["train_coefficient_penalty"] == pytest.approx(
        0.01 * float(coefficient_penalty_sum(states[0].params, config.model, include_denominator=False)), rel=1e-5)


def test_run_index_exposes_the_denominator(tmp_path):
    from scaffolds.records import create_run, regenerate_index

    study = tmp_path / "experiments/demo"
    study.mkdir(parents=True)
    (study / "sweep.yaml").write_text(json.dumps({"study_id": "demo", "run_ids": []}))
    config = _config(tmp_path / "placeholder", 8, "per_head", qk_norm=True, penalty=0.01)
    config.model = _model_config(8, qk_norm=True, attention_denominator_degree=2,
                                 attention_denominator_parameterization="sum_of_squares")
    config.optim.coefficient_penalty_denominator = False
    validate_config(config)
    create_run(study, config_to_dict(config), label="square")
    regenerate_index(study)
    with (study / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert (row["denominator_degree"], row["denominator_parameterization"], row["coefficient_penalty_denominator"]) == (
        "2", "sum_of_squares", "False")
