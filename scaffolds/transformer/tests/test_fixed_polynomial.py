"""Independent fixed-polynomial mathematics, parameter, and resume contracts."""

from dataclasses import replace
from functools import partial
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.configs import config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import (
    make_checkpoint_manager, restore_latest_checkpoint, save_checkpoint,
)
from scaffolds.transformer.training.state import create_state


DEGREES = (1, 2, 3, 4)
# Explicit monomial coefficients; the oracle below never evaluates this table.
HERMITE_COEFFICIENTS = {
    1: (1.0, 1.0),
    2: (0.5, 1.0, 0.5),
    3: (0.5, 0.5, 0.5, 1 / 6),
    4: (5 / 8, 0.5, 0.25, 1 / 6, 1 / 24),
}


def _coefficients(degree, prefactor=1.0):
    return tuple(prefactor * value for value in HERMITE_COEFFICIENTS[degree])


def _hermite_kernel(scores, degree, prefactor=1.0):
    """Probabilists' recurrence, independent of monomial/Horner evaluation."""
    previous, current = jnp.ones_like(scores), scores
    result = previous + current
    for order in range(2, degree + 1):
        previous, current = current, scores * current - (order - 1) * previous
        result = result + current / math.factorial(order)
    return prefactor * result


def _dense_reference(q, k, v, *, degree, normalized, prefactor=1.0):
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    kernel = _hermite_kernel(scores, degree, prefactor)
    kernel = jnp.where(jnp.tril(jnp.ones(scores.shape[-2:], bool)), kernel, 0.0)
    if normalized:
        kernel = kernel / jnp.sum(kernel, axis=-1, keepdims=True)
    return jnp.einsum("bhts,bshd->bthd", kernel, v)


def _operands(degree, shape=(2, 17, 3, 8, 5), dtype=jnp.float32, seed=902):
    batch, time, heads, dq, dv = shape
    rng = np.random.default_rng(seed + degree)
    q, k = (0.6 * rng.normal(size=(batch, time, heads, dq)) for _ in range(2))
    # Isolate two exact signed-score rows from the random suffix. Their first
    # denominator is negative for degrees 1, 3, 4, and the second mixes signs.
    # Degree two is nonnegative; its zero denominator is checked separately.
    score = {1: -2.0, 2: -0.5, 3: -3.0, 4: -2.0}[degree]
    q[..., 0] = k[..., 0] = 0.0
    q[:, :2] = k[:, :2] = 0.0
    q[:, :2, :, 0] = 1.0
    k[:, 0, :, 0] = score * math.sqrt(dq)
    k[:, 1, :, 0] = math.sqrt(dq)
    v, cotangent = (rng.normal(size=(batch, time, heads, dv)) for _ in range(2))
    return tuple(jnp.asarray(x, dtype) for x in (q, k, v)), jnp.asarray(cotangent, dtype)


def _attention(degree, normalized, *, prefactor=1.0, backend="blocked"):
    return partial(attention_core, attention_degree=degree, attention_normalization=normalized,
                   polynomial_coefficients=jnp.asarray(_coefficients(degree, prefactor), jnp.float32),
                   backend=backend)


def _evaluate(attend, operands, cotangent):
    def output_and_grads(*arrays):
        output, pullback = jax.vjp(attend, *arrays)
        return output, pullback(cotangent)

    return jax.jit(output_and_grads)(*operands)


def _fp64(operands, cotangent, degree, normalized, prefactor=1.0):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _evaluate(partial(_dense_reference, degree=degree, normalized=normalized, prefactor=prefactor),
                           tuple(x.astype(jnp.float64) for x in operands), cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, np.float64), result)


def _assert_result(actual, expected, output_tolerance=3e-5, gradient_tolerances=(2e-4,) * 3):
    """Per-head relative norms avoid mistaking cancellation for implementation error."""
    pairs = [("output", actual[0], expected[0], output_tolerance)]
    pairs.extend((f"d{name}", got, want, tolerance) for name, got, want, tolerance in
                 zip("qkv", actual[1], expected[1], gradient_tolerances, strict=True))
    for name, actual_array, expected_array, tolerance in pairs:
        actual_array, expected_array = (np.asarray(x, np.float64) for x in (actual_array, expected_array))
        assert actual_array.shape == expected_array.shape, name
        assert np.isfinite(actual_array).all() and np.isfinite(expected_array).all(), name
        for batch in range(expected_array.shape[0]):
            for head in range(expected_array.shape[2]):
                got, want = actual_array[batch, :, head], expected_array[batch, :, head]
                norm = np.linalg.norm(want)
                if norm == 0:
                    np.testing.assert_array_equal(got, want, err_msg=name)
                    continue
                error = np.linalg.norm(got - want) / norm
                assert error <= tolerance, f"{name}, batch {batch}, head {head}: relative L2 {error}"
                assert np.max(np.abs(got - want)) <= 3 * tolerance * np.max(np.abs(want)), name


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("normalized", (False, True))
@pytest.mark.parametrize("prefactor", (1.0, math.exp(0.5)))
def test_fixed_outputs_and_qkv_gradients_match_independent_fp64(degree, normalized, prefactor):
    operands, cotangent = _operands(degree)
    with jax.default_matmul_precision("highest"):
        actual = _evaluate(_attention(degree, normalized, prefactor=prefactor), operands, cotangent)
    _assert_result(actual, _fp64(operands, cotangent, degree, normalized, prefactor))
    if degree != 2:
        _, diagnostics = _attention(degree, normalized, prefactor=prefactor)(*operands, return_diagnostics=True)
        assert float(diagnostics["kernel_min"]) < 0
        assert int(diagnostics["invalid_normalizer_count" if normalized else "invalid_weight_sum_count"]) == 0
        if normalized:
            assert float(diagnostics["normalizer_min"]) < 0


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("normalized", (False, True))
def test_common_prefactor_cancels_only_when_normalized(degree, normalized):
    operands, cotangent = _operands(degree)
    with jax.default_matmul_precision("highest"):
        original = _evaluate(_attention(degree, normalized), operands, cotangent)
        scaled = _evaluate(_attention(degree, normalized, prefactor=math.exp(0.5)), operands, cotangent)
    expected = original if normalized else jax.tree.map(lambda x: math.exp(0.5) * x, original)
    _assert_result(scaled, expected)


@pytest.mark.parametrize("prefactor", (1.0, math.exp(0.5)))
def test_zero_denominator_is_not_floored_and_later_rows_recover(prefactor):
    q = jnp.asarray([0., 1., 0.], jnp.float32).reshape(1, 3, 1, 1)
    k = jnp.asarray([-1., -1., 0.], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3., 7., 11.], jnp.float32).reshape(q.shape)
    output, diagnostics = _attention(2, True, prefactor=prefactor)(q, k, v, return_diagnostics=True)
    np.testing.assert_allclose(np.asarray(output).reshape(-1)[[0, 2]], [3., 7.], rtol=2e-6)
    assert bool(jnp.isnan(output[0, 1, 0, 0]))
    assert int(diagnostics["invalid_normalizer_count"]) == 1
    unnormalized, diagnostics = _attention(2, False, prefactor=prefactor)(q, k, v, return_diagnostics=True)
    np.testing.assert_allclose(np.asarray(unnormalized).reshape(-1),
                               prefactor * np.asarray([1.5, 0., 10.5]), rtol=2e-6)
    assert int(diagnostics["invalid_weight_sum_count"]) == 0


def test_signed_cancellation_is_invalid_but_small_nonzero_denominator_is_valid():
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    k = jnp.asarray([0., -2., 1.], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3., 7., 11.], jnp.float32).reshape(q.shape)
    output, diagnostics = _attention(1, True)(q, k, v, return_diagnostics=True)
    np.testing.assert_array_equal(np.asarray(output).reshape(-1)[[0, 2]], [3., 9.])
    assert bool(jnp.isnan(output[0, 1, 0, 0]))
    assert int(diagnostics["invalid_normalizer_count"]) == 1
    output, diagnostics = _attention(1, True)(q[:, :1], q[:, :1] * (-1 + 2 ** -20), v[:, :1],
                                            return_diagnostics=True)
    np.testing.assert_array_equal(output, v[:, :1])
    assert float(diagnostics["normalizer_min"]) == 2 ** -20
    assert int(diagnostics["invalid_normalizer_count"]) == 0


@pytest.mark.parametrize("normalized", (False, True))
def test_model_routes_static_coefficients_to_the_requested_kernel(monkeypatch, tmp_path, normalized):
    from scaffolds.transformer.attention import multihead

    config = tiny_config(tmp_path)
    coefficients = _coefficients(4, math.exp(0.5))
    config.model = replace(config.model, attention_degree=4, attention_normalization=normalized,
                           attention_backend="blocked", attention_fixed_coefficients=coefficients)
    model = build_model(config.model, 0)
    tokens = jnp.asarray([[1, 2, 3, 4]])
    expected = model(tokens)
    calls = []

    def recurrence(q, k, v, **kwargs):
        np.testing.assert_array_equal(kwargs["polynomial_coefficients"], np.asarray(coefficients, np.float32))
        assert kwargs["attention_degree"] == 4
        assert kwargs["attention_normalization"] == normalized
        calls.append(True)
        return _dense_reference(q, k, v, degree=4, normalized=normalized, prefactor=math.exp(0.5))

    monkeypatch.setattr(multihead, "attention_core", recurrence)
    np.testing.assert_allclose(model(tokens), expected, rtol=2e-5, atol=2e-6)
    assert len(calls) == config.model.num_layers


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("normalized", (False, True))
def test_fixed_coefficients_preserve_backbone_initialization_and_parameter_count(tmp_path, degree, normalized):
    config = tiny_config(tmp_path)
    config.model = replace(config.model, attention_degree=degree, attention_normalization=normalized,
                           attention_backend="blocked", attention_coefficients="fixed",
                           attention_fixed_coefficients=_coefficients(degree, math.exp(0.5)), qk_norm=False)
    models = [build_model(model_config, 13) for model_config in (
        config.model, replace(config.model, attention_fixed_coefficients=None),
        replace(config.model, attention_fixed_coefficients=None, attention_degree=None))]
    flat_states = [dict(nnx.to_flat_state(nnx.state(model, nnx.Param))) for model in models]
    assert all(set(state) == set(flat_states[0]) for state in flat_states)
    assert not any("polynomial" in str(path) for path in flat_states[0])
    for path, value in flat_states[0].items():
        for state in flat_states[1:]:
            np.testing.assert_array_equal(value[...], state[path][...])
    assert model_parameter_count(config.model) == sum(x.size for x in jax.tree.leaves(nnx.state(models[0], nnx.Param)))
    for _, module in nnx.iter_graph(models[0]):
        if isinstance(module, MultiHeadAttention):
            assert module.attention_fixed_coefficients == config.model.attention_fixed_coefficients


@pytest.mark.parametrize("degree", (2, 4))
@pytest.mark.parametrize("normalized", (False, True))
def test_fixed_coefficients_survive_training_and_checkpoint_resume(tmp_path, degree, normalized):
    config = tiny_config(tmp_path)
    coefficients = _coefficients(degree, math.exp(0.5))
    config.model = replace(config.model, attention_degree=degree, attention_normalization=normalized,
                           attention_backend="blocked", attention_coefficients="fixed",
                           attention_fixed_coefficients=coefficients, qk_norm=False)
    loop.ensure_run_config(config, resume=False)
    rng = jax.random.PRNGKey(0)
    initial = create_state(config, rng)
    batch = jnp.asarray(np.arange(20).reshape(4, 5) % 16)
    first, loss = loop.train_step(initial, batch, rng)
    assert np.isfinite(loss)
    assert any(not np.array_equal(a, b) for a, b in
               zip(jax.tree.leaves(initial.params), jax.tree.leaves(first.params), strict=True))
    manager = make_checkpoint_manager(config)
    try:
        assert save_checkpoint(manager, 1, first, metrics={"validation_nll": float(loss)})
        manager.wait_until_finished()
        restored, step = restore_latest_checkpoint(manager, create_state(config, rng))
        assert step == 1
        next_rng = jax.random.fold_in(rng, 1)
        uninterrupted, expected_loss = loop.train_step(first, batch, next_rng)
        resumed, actual_loss = loop.train_step(restored, batch, next_rng)
    finally:
        manager.close()
    np.testing.assert_array_equal(actual_loss, expected_loss)
    for got, want in zip(jax.tree.leaves(resumed), jax.tree.leaves(uninterrupted), strict=True):
        np.testing.assert_array_equal(got, want)
    for state in (initial, first, restored, resumed):
        model = nnx.merge(state.graphdef, state.params)
        modules = [module for _, module in nnx.iter_graph(model) if isinstance(module, MultiHeadAttention)]
        assert modules and all(module.attention_fixed_coefficients == coefficients for module in modules)
        assert not any("polynomial" in str(path) for path, _ in nnx.to_flat_state(state.params))
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["model"]["attention_fixed_coefficients"] == list(coefficients)
    assert config_dicts_equal(saved, config_to_dict(config))
    loop.ensure_run_config(config, resume=True)
    config.model = replace(config.model, attention_fixed_coefficients=_coefficients(degree))
    with pytest.raises(ValueError, match="does not match"):
        loop.ensure_run_config(config, resume=True)
