"""Static coefficient fitting ties both positions and sequences with one optimizer."""
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer.tests.test_shared_coefficient_fit import fitting, _equal
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import (
    make_shared_functions, qualify, restore_state, save_state,
)


@pytest.fixture(scope="module")
def static_fit(fitting):
    graph, source, tokens, options, positional_functions = fitting
    options = {**options, "parameterization": "static_coefficients"}
    return graph, source, tokens, options, make_shared_functions(graph, 8, options), positional_functions


@pytest.mark.parametrize("sequence_count", [1, 4])
def test_static_projection_and_constant_positional_equivalence(static_fit, sequence_count):
    graph, source, tokens, options, functions, positional_functions = static_fit
    capture, initialize, _, evaluate = functions
    tokens = tokens[:sequence_count]
    captured, original, replay, error = capture(source, tokens)
    np.testing.assert_array_equal(original, replay)
    assert float(error) == 0
    table = captured.mean(axis=(0, 2))
    assert table.shape == (2, 2, 9)
    assert table.size == 2 * 2 * 9
    assert initialize(table)[1].count.shape == ()
    streaming = sum(capture(source, chunk[None])[0].sum(axis=(0, 2))
                    for chunk in tokens) / (sequence_count * 7)
    np.testing.assert_allclose(streaming, table, rtol=1e-6, atol=1e-7)
    positional = jnp.repeat(table[:, None], 7, axis=1)
    np.testing.assert_allclose(evaluate(table, source, tokens[None]),
                               positional_functions[3](positional, source, tokens[None]), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="Static coefficients"):
        evaluate(positional, source, tokens[None])


def test_static_gradient_ties_positions_and_accumulates_batch_before_adam(static_fit):
    graph, source, tokens, options, functions, positional_functions = static_fit
    capture, initialize, step, evaluate = functions
    table = capture(source, tokens)[0].mean(axis=(0, 2))
    source_hash = state_sha256(source)
    optimizer = initialize(table)
    factorial = jnp.asarray([math.factorial(r) for r in range(9)], jnp.float32)
    # Express the full model independently, with a single coefficient per layer/head/order.
    def full_objective(value):
        coefficients = jnp.broadcast_to(value[:, None, None], (2, 4, 7, 2, 9))
        logits = nnx.merge(graph, source)(tokens[:, :-1], query_coefficients=coefficients)
        return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:]).mean()
    gradient = jax.jit(jax.grad(full_objective))(table) / factorial
    # Tying positions sums the positional gradients; sequence loss remains a mean.
    positional = jnp.repeat(table[:, None], 7, axis=1)
    positional_gradient = jax.grad(lambda value: positional_functions[3](value, source, tokens[None]).mean())(positional)
    np.testing.assert_allclose(gradient, positional_gradient.sum(axis=1) / factorial, rtol=4e-5, atol=1e-8)
    norm = optax.global_norm(gradient)
    clipped = gradient * jnp.minimum(1., options["clip_norm"] / norm)
    expected = table - 1e-3 * clipped / (jnp.abs(clipped) + options["epsilon"]) / factorial
    result = step(table, optimizer, source, tokens.reshape(2, 2, 8), 1e-3)
    assert bool(result[4])
    assert int(result[1][1].count) == 1
    np.testing.assert_allclose(result[3][0], norm, rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(result[0], expected, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(result[1][1].mu, (1-options["beta1"]) * clipped, rtol=3e-5, atol=1e-9)
    for microbatch in (1, 4):
        other = step(table, optimizer, source, tokens.reshape(4//microbatch, microbatch, 8), 1e-3)
        for left, right in zip(jax.tree.leaves(result), jax.tree.leaves(other), strict=True):
            np.testing.assert_allclose(left, right, rtol=4e-5, atol=2e-7)
    assert state_sha256(source) == source_hash


@pytest.mark.parametrize("sequence_count", [1, 4])
def test_static_checkpoint_and_qualification(static_fit, tmp_path, sequence_count):
    graph, source, tokens, _, functions, _ = static_fit
    tokens = tokens[:sequence_count]
    capture, initialize, step, evaluate = functions
    captured, parent_nll, _, error = capture(source, tokens)
    table = captured.mean(axis=(0, 2))
    optimizer = initialize(table)
    chunks = tokens.reshape(sequence_count, 1, 8)
    updated, moments = step(table, optimizer, source, chunks, 1e-3)[:2]
    controller = dict(step=1, best_step=1, best_loss=2., history=[2.1, 2.], learning_rate=1e-3)
    root = tmp_path / "checkpoints"
    root.mkdir()
    path = save_state(root, 1, [updated], [moments], [updated], controller)
    restored, restored_controller = restore_state(path, [updated], [moments])
    assert restored_controller == controller
    _equal(step(restored["tables"][0], restored["optimizers"][0], source, chunks, 1e-3),
           step(updated, moments, source, chunks, 1e-3))
    report = qualify(tmp_path, functions, source, chunks, table, optimizer, 1e-3,
                     graph=graph, shared=True, parent_nll=parent_nll, capture_logit_error=error)
    assert report["checkpoint_resume_exact"]
    assert report["training_forward_reference"] == "projected_coefficients"
    assert report["projected_initial_nll"] == float(evaluate(table, source, chunks).mean())
    assert report["one_update_nll"] < report["separate_training_forward_nll"]
    assert report["frozen_parameter_scope"] == "all_pretrained_parameters"
