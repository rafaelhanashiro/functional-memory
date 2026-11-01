"""FoX without RoPE restores the backbone and decays only its projection matrices."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scaffolds.transformer.tests.test_continuation import assert_equal_tree
from scaffolds.transformer.tests.test_token_scale_continuation import (
    forget_gate_config, pretrained,
)
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.continuation import initialize_continuation_state
from scaffolds.transformer.training.optim import split_continuation_parameters


def test_removing_rope_for_fox_keeps_backbone_and_adam_history(pretrained, tmp_path):
    _, parent, _, batch = pretrained
    config = forget_gate_config(pretrained, tmp_path / "fox")
    config.model = replace(config.model, attention_coefficient_kernel_weight_decay=True)
    with_rope, _ = initialize_continuation_state(config)
    config.model = replace(config.model, positional_encoding="none")
    without_rope, audit = initialize_continuation_state(config)

    # Positional encoding is an intentional functional change, not a new or reset
    # projection: both continuations have exactly the same initialized arrays.
    assert_equal_tree(without_rope.params, with_rope.params)
    introduced, backbone = split_continuation_parameters(without_rope.params)
    assert_equal_tree(backbone, parent.params)
    assert_equal_tree(without_rope.opt_state[1].backbone, parent.opt_state[1])
    assert audit["parent_parameters_sha256"] == audit["backbone_parameters_sha256"]
    assert audit["parent_optimizer_sha256"] == audit["backbone_optimizer_sha256"]
    assert audit["new_coefficient_optimizer_count"] == 0
    assert audit["backbone_optimizer_count"] == 3
    assert audit["data_start_sequence"] == 12
    np.testing.assert_array_equal(introduced["query_polynomial_bias"][...], 1.0)
    np.testing.assert_array_equal(introduced["query_polynomial_amplitude"][...], 0.0)
    inputs = batch[:, :-1]
    rope_logits = nnx.merge(with_rope.graphdef, with_rope.params)(inputs, deterministic=True)
    no_rope_logits = nnx.merge(without_rope.graphdef, without_rope.params)(inputs, deterministic=True)
    assert not np.allclose(no_rope_logits, rope_logits, rtol=1e-5, atol=1e-6)

    after, loss, *_ = loop.train_step(without_rope, batch, jax.random.PRNGKey(17))
    assert np.isfinite(float(loss))
    assert int(after.opt_state[1].coefficients[0].count) == 1
    assert int(after.opt_state[1].backbone[0].count) == 4


def test_fox_continuation_decays_both_w_matrices_but_not_amplitudes_or_biases(pretrained, tmp_path):
    config = forget_gate_config(pretrained, tmp_path / "fox", new_rate=1e-2)
    config.model = replace(config.model, positional_encoding="none",
                           attention_coefficient_kernel_weight_decay=True)
    state, _ = initialize_continuation_state(config)
    zero_grads = jax.tree.map(jnp.zeros_like, state.params)
    updates, _ = state.tx.update(zero_grads, state.opt_state, state.params)
    introduced, _ = split_continuation_parameters(state.params)
    introduced_updates, _ = split_continuation_parameters(updates)
    for name, value in introduced.items():
        if name in {"query_polynomial_kernel", "forget_gate_kernel"}:
            assert np.any(np.asarray(value[...]) != 0)
            expected = -config.continuation.new_parameter_learning_rate * config.optim.weight_decay * value[...]
            np.testing.assert_allclose(introduced_updates[name][...], expected, rtol=1e-6, atol=1e-8)
        else:
            np.testing.assert_array_equal(introduced_updates[name][...], 0.0)
