"""Coefficient-only fitting is independent per example and resumes exactly."""

import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.attention_scaling.coefficient_oracle import (
    make_functions, plateau_action, restore_state, save_state,
)
from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.models import build_model


def _oracle():
    return dict(clip_norm=1.0, beta1=0.9, beta2=0.95, epsilon=1e-8,
                initial_updates=2000, plateau_window=200, plateau_tolerance=1e-3,
                confirmation_updates=500)


@pytest.fixture(scope="module")
def fitting():
    config = ModelConfig(
        vocab_size=16, max_seq_len=7, num_layers=2, hidden_size=8, num_heads=2,
        num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
        attention_degree=8, attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
        attention_coefficient_activation="tanh", attention_coefficient_scale="factorial",
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_tanh_bias=True,
        attention_coefficient_projection_initial_std=0.05, attention_backend="blocked",
        initialization="transformer_reference", qk_norm=True,
        attention_output_scaling="head_rms_norm", remat=False,
    )
    model = build_model(config, 7)
    graph, params = nnx.split(model, nnx.Param)
    functions = make_functions(graph, config.attention_degree, _oracle())
    tokens = jnp.arange(16).reshape(2, 8) % config.vocab_size
    return functions, params, tokens, graph


def _assert_trees_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def _assert_trees_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-7)


def test_single_sequence_batch_and_microbatch_updates_agree_and_freeze_parent(fitting):
    (capture, initialize, step, evaluate), params, tokens, _ = fitting
    frozen_before = jax.tree.map(lambda x: np.array(x, copy=True), params)
    table, original, replay, logits_error = capture(params, tokens)
    assert table.shape == (2, 2, 7, 2, 9)
    assert float(logits_error) == 0.0
    np.testing.assert_array_equal(original, replay)
    np.testing.assert_array_equal(evaluate(table, params, tokens), original)
    optimizer = initialize(table)
    split_tokens = [tokens[index:index + 1] for index in range(2)]
    split_tables = [capture(params, inputs)[0] for inputs in split_tokens]
    split_optimizers = [initialize(value) for value in split_tables]
    learning_rate = jnp.asarray(1e-3, jnp.float32)
    for _ in range(3):
        batch_result = step(table, optimizer, params, tokens, learning_rate)
        split_results = [step(value, state, params, inputs, learning_rate)
                         for value, state, inputs in zip(split_tables, split_optimizers, split_tokens, strict=True)]
        concatenated = jax.tree.map(lambda *xs: jnp.concatenate(xs),
                                    *[result[:4] for result in split_results])
        _assert_trees_close(batch_result[:4], concatenated)
        assert bool(batch_result[4]) and all(bool(result[4]) for result in split_results)
        assert np.isfinite(batch_result[3]).all() and np.all(np.asarray(batch_result[3]) > 0)
        table, optimizer = batch_result[:2]
        split_tables = [result[0] for result in split_results]
        split_optimizers = [result[1] for result in split_results]
    assert np.all(np.asarray(evaluate(table, params, tokens)) < np.asarray(original))
    _assert_trees_equal(params, frozen_before)


def test_effective_storage_preserves_capture_and_factorial_coordinate_chain_rule(fitting):
    (capture, _, _, evaluate), params, tokens, graph = fitting
    table, _, _, _ = capture(params, tokens)
    factorial = jnp.asarray([math.factorial(order) for order in range(9)], jnp.float32)

    @jax.jit
    def capture_effective(params, tokens):
        _, coefficients = nnx.merge(graph, params)(tokens[:, :-1], return_coefficients=True)
        return jnp.swapaxes(coefficients, 0, 1)

    np.testing.assert_array_equal(table, capture_effective(params, tokens))

    # An independent objective in factorial-scaled increments gives dL/du
    # without multiplying/dividing the initial effective coefficient values.
    def coordinate_objective(increment):
        return evaluate(table + increment / factorial, params, tokens).sum()

    coordinate_gradient = jax.jit(jax.grad(coordinate_objective))(jnp.zeros_like(table))
    axes = tuple(range(1, table.ndim))
    expected_norms = jnp.sqrt(jnp.square(coordinate_gradient).sum(axis=axes))
    oracle = {**_oracle(), "clip_norm": float(expected_norms.min()) / 4}
    _, initialize, step, _ = make_functions(graph, 8, oracle)
    initial_optimizer = initialize(table)
    learning_rate = 1e-3
    updated, optimizer, _, actual_norms, finite = step(
        table, initial_optimizer, params, tokens, learning_rate)
    assert bool(finite)
    np.testing.assert_allclose(actual_norms, expected_norms, rtol=2e-5, atol=1e-8)
    shape = (table.shape[0],) + (1,) * (table.ndim - 1)
    clipped = coordinate_gradient * (oracle["clip_norm"] / expected_norms).reshape(shape)
    # Adam's first bias-corrected update, in u coordinates, then transformed to c.
    expected_update_u = clipped / (jnp.abs(clipped) + oracle["epsilon"])
    expected_coefficients = table - learning_rate * expected_update_u / factorial
    np.testing.assert_allclose(updated, expected_coefficients, rtol=2e-5, atol=2e-8)
    adam = optimizer[1]
    np.testing.assert_array_equal(adam.count, np.ones(table.shape[0], dtype=np.int32))
    np.testing.assert_allclose(adam.mu, (1 - oracle["beta1"]) * clipped, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(adam.nu, (1 - oracle["beta2"]) * jnp.square(clipped), rtol=3e-5, atol=1e-12)
    zero_rate, _, _, _, _ = step(table, initial_optimizer, params, tokens, 0.0)
    np.testing.assert_array_equal(zero_rate, table)


def test_plateau_requires_initial_budget_window_and_confirmation():
    oracle = _oracle()
    flat = [2.0] * 2601
    assert plateau_action(flat[:2000], 1999, None, oracle) == "continue"
    assert plateau_action(flat[:2001], 2000, None, oracle) == "reduce"
    assert plateau_action(flat[:2500], 2499, 2000, oracle) == "continue"
    assert plateau_action(flat[:2501], 2500, 2000, oracle) == "stop"
    assert plateau_action([2.0] * 200, 2000, None, oracle) == "continue"
    improving = [3.0 - 0.01 * index for index in range(201)]
    assert plateau_action(improving, 2000, None, oracle) == "continue"
    assert plateau_action(improving, 2500, 2000, oracle) == "continue"
    just_small_enough = [2.0] * 200 + [1.9995]
    assert plateau_action(just_small_enough, 2000, None, oracle) == "reduce"
    still_improving = [2.0] * 200 + [1.998]
    assert plateau_action(still_improving, 2000, None, oracle) == "continue"


def test_checkpoint_restores_controller_optimizer_and_exact_next_step(fitting, tmp_path):
    (capture, initialize, step, _), params, tokens, _ = fitting
    initial, _, _, _ = capture(params, tokens)
    optimizer = initialize(initial)
    learning_rate = 1e-3
    table, optimizer, _, _, _ = step(initial, optimizer, params, tokens, learning_rate)
    tables, optimizers, best = [table], [optimizer], [initial]
    controller = dict(step=1, best_step=0, best_loss=2.0, initial_loss=2.1,
                      history=[2.1, 2.0], reduced_at=None, learning_rate=learning_rate,
                      started_utc="2026-10-23T00:00:00+00:00")
    root = tmp_path / "checkpoints"
    root.mkdir()
    saved = save_state(root, 1, tables, optimizers, best, controller)
    restored, restored_controller = restore_state(saved, tables, optimizers)
    assert restored_controller == controller
    _assert_trees_equal(restored, {"tables": tables, "optimizers": optimizers, "best": best})
    expected_next = step(table, optimizer, params, tokens, learning_rate)
    actual_next = step(restored["tables"][0], restored["optimizers"][0], params, tokens,
                       restored_controller["learning_rate"])
    _assert_trees_equal(actual_next, expected_next)

    # A plateau decision can change the durable controller at the same table step.
    updated_controller = {**controller, "reduced_at": 1, "learning_rate": learning_rate / 10}
    rewritten = save_state(root, 1, tables, optimizers, best, updated_controller)
    rewritten_state, rewritten_controller = restore_state(rewritten, tables, optimizers)
    assert rewritten_controller == updated_controller
    _assert_trees_equal(rewritten_state, restored)
    expected_reduced_next = step(table, optimizer, params, tokens, learning_rate / 10)
    actual_reduced_next = step(rewritten_state["tables"][0], rewritten_state["optimizers"][0], params,
                               tokens, rewritten_controller["learning_rate"])
    _assert_trees_equal(actual_reduced_next, expected_reduced_next)

    newer = save_state(root, 2, [expected_reduced_next[0]], [expected_reduced_next[1]], best,
                       {**updated_controller, "step": 2})
    assert newer.exists() and not saved.exists()
    assert [path.name for path in root.iterdir()] == ["2"]
