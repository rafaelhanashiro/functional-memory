"""A single table learns from the complete batch, with one mean-gradient Adam step."""
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import (
    make_functions, make_shared_functions, qualify, restore_state, save_state,
)


@pytest.fixture(scope="module")
def fitting():
    config = ModelConfig(
        vocab_size=16, max_seq_len=7, num_layers=2, hidden_size=8, num_heads=2,
        num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
        attention_degree=8, attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
        attention_coefficient_activation="tanh", attention_coefficient_scale="factorial",
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_tanh_bias=True,
        attention_coefficient_projection_initial_std=.2, attention_coefficient_amplitude_initial=.2,
        attention_backend="blocked", attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
        rms_norm_fp32=True, initialization="transformer_truncated_reference", remat=False,
    )
    graph, source = nnx.split(build_model(config, 7), nnx.Param)
    options = dict(clip_norm=.01, beta1=.9, beta2=.95, epsilon=1e-8)
    functions = make_shared_functions(graph, 8, options)
    tokens = (jnp.arange(32).reshape(4, 8) + jnp.arange(4)[:, None]) % 16
    return graph, source, tokens, options, functions


def _equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_mean_projection_has_no_sequence_axis_and_is_reused(fitting):
    graph, source, tokens, options, (capture, initialize, _, evaluate) = fitting
    captured, original, replay, error = capture(source, tokens)
    np.testing.assert_array_equal(original, replay)
    assert float(error) == 0
    table = captured.mean(axis=0)
    assert table.shape == (2, 7, 2, 9)
    assert table.size == 2 * 7 * 2 * 9
    assert initialize(table)[1].count.shape == ()
    # Streaming projection across microbatches equals the full-batch arithmetic mean.
    streaming = sum(capture(source, chunk)[0].sum(axis=0) for chunk in tokens.reshape(2, 2, 8)) / 4
    np.testing.assert_allclose(streaming, table, rtol=1e-6, atol=1e-7)
    independent = make_functions(graph, 8, options)[3]
    repeated = jnp.broadcast_to(table, (4, *table.shape))
    np.testing.assert_allclose(evaluate(table, source, tokens.reshape(2, 2, 8)),
                               independent(repeated, source, tokens), rtol=1e-5, atol=1e-6)
    # Projection usually changes loss: it must not be called original-forward replay.
    assert not np.allclose(evaluate(table, source, tokens[None]), original, rtol=1e-7, atol=1e-7)
    with pytest.raises(ValueError, match="Shared coefficients"):
        evaluate(repeated, source, tokens[None])


def test_accumulation_matches_full_batch_and_adam_updates_once(fitting):
    graph, source, tokens, options, (capture, initialize, step, evaluate) = fitting
    table = capture(source, tokens)[0].mean(axis=0)
    original_hash = state_sha256(source)
    optimizer = initialize(table)
    factorial = jnp.asarray([math.factorial(r) for r in range(9)], jnp.float32)
    # Differentiate an independently expressed, direct full-batch forward.
    def full_objective(value):
        coefficients = jnp.broadcast_to(value[:, None], (2, 4, 7, 2, 9))
        logits = nnx.merge(graph, source)(tokens[:, :-1], query_coefficients=coefficients)
        return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), tokens[:, 1:]).mean()
    gradient = jax.jit(jax.grad(full_objective))(table) / factorial
    norm = optax.global_norm(gradient)
    clipped = gradient * jnp.minimum(1., options["clip_norm"] / norm)
    expected = table - 1e-3 * clipped / (jnp.abs(clipped) + options["epsilon"]) / factorial
    result = step(table, optimizer, source, tokens.reshape(2, 2, 8), 1e-3)
    assert bool(result[4])
    assert int(result[1][1].count) == 1
    np.testing.assert_allclose(result[3][0], norm, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(result[0], expected, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(result[1][1].mu, (1-options["beta1"]) * clipped, rtol=2e-5, atol=1e-9)
    for microbatch in (1, 4):
        other = step(table, optimizer, source, tokens.reshape(4//microbatch, microbatch, 8), 1e-3)
        for left, right in zip(jax.tree.leaves(result), jax.tree.leaves(other), strict=True):
            np.testing.assert_allclose(left, right, rtol=4e-5, atol=2e-7)
    np.testing.assert_array_equal(step(table, optimizer, source, tokens[None], 0.)[0], table)
    assert state_sha256(source) == original_hash


def test_shared_checkpoint_resume_and_projection_qualification(fitting, tmp_path):
    graph, source, tokens, _, functions = fitting
    capture, initialize, step, evaluate = functions
    captured, parent_nll, _, error = capture(source, tokens)
    table = captured.mean(axis=0)
    optimizer = initialize(table)
    chunks = tokens.reshape(2, 2, 8)
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
