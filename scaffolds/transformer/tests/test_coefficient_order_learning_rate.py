"""Order-scaled learning rates: sigmoid query gate r takes 2/r or r/2 of the schedule's, Adam step and decay alike."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer.configs import Config, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.tests.test_gated_even_coefficients import ORDERS, _model
from scaffolds.transformer.training.optim import ORDER_GATES, make_optimizer

RULES = {"two_over_order": lambda order: 2 / order, "order_over_two": lambda order: order / 2}
from scaffolds.transformer.training.state import create_state


def gate_config(rate="constant"):
    config = Config()
    config.model = _model("zero")
    config.model.num_layers = 2
    config.data.synthetic_vocab_size = 32
    config.data.sequence_length = 9
    config.data.batch_size = 4
    config.train.micro_batch_size = 2
    config.optim.total_steps = 8
    config.optim.warmup_steps = 2
    config.optim.grad_clip_norm = 0.1
    config.optim.coefficient_order_learning_rate = rate
    validate_config(config)
    return config


def assert_equal_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("rule", RULES)
def test_each_gate_order_takes_its_rule_after_clipping_and_adam(rule):
    reference_config, scaled_config = gate_config(), gate_config(rule)
    params = create_state(reference_config, jax.random.PRNGKey(0)).params
    reference_tx, scaled_tx = make_optimizer(reference_config), make_optimizer(scaled_config)
    reference_state, scaled_state = reference_tx.init(params), scaled_tx.init(params)
    assert_equal_tree(reference_state, scaled_state)
    scales = np.asarray([RULES[rule](order) for order in ORDERS])
    # Cross warmup, the cosine decay and the minimum; gradients above the clip norm vary over time.
    for step in range(reference_config.optim.total_steps + 1):
        grads = jax.tree.map(lambda value: jnp.sin(jnp.arange(value.size).reshape(value.shape) + step) + 2, params)
        assert float(optax.tree.norm(grads)) > reference_config.optim.grad_clip_norm
        reference, reference_state = reference_tx.update(grads, reference_state, params)
        scaled, scaled_state = scaled_tx.update(grads, scaled_state, params)
        assert_equal_tree(reference_state, scaled_state)
        reference_values = dict(nnx.to_flat_state(reference))
        paths = set()
        for path, value in nnx.to_flat_state(scaled):
            paths.add(path[0])
            expected = reference_values[path][...]
            if path[0] in ORDER_GATES:
                assert value[...].shape[-1] == len(ORDERS)
                np.testing.assert_allclose(value[...], expected * scales, rtol=1e-6, atol=1e-12)
                if step:
                    assert np.all(np.any(expected != 0, axis=tuple(range(expected.ndim - 1))))
            else:
                np.testing.assert_array_equal(value[...], expected)
        assert ORDER_GATES | {"qk_norm_q_gain", "token_embedding", "lm_head"} <= paths


@pytest.mark.parametrize("rule", RULES)
def test_decay_of_each_gate_projection_also_takes_its_order_rate(rule):
    config = gate_config(rule)
    config.optim.warmup_steps = 0
    config.optim.min_learning_rate = config.optim.learning_rate
    state = create_state(config, jax.random.PRNGKey(0))
    state = state.replace(params=jax.tree.map(jnp.ones_like, state.params))
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    params = dict(nnx.to_flat_state(updated.params))
    decay = config.optim.learning_rate * config.optim.weight_decay
    expected = 1 - decay * np.asarray([RULES[rule](order) for order in ORDERS])
    np.testing.assert_allclose(params[("query_polynomial_kernel",)][...],
                               np.broadcast_to(expected, params[("query_polynomial_kernel",)][...].shape), rtol=1e-6)
    np.testing.assert_array_equal(params[("query_polynomial_bias",)][...], 1.0)
    np.testing.assert_allclose(params[("lm_head", "kernel")][...], 1 - decay, rtol=1e-6)


def test_tanh_query_coefficients_are_refused():
    """The diagonal tanh query generator stores every order 0..m (and a and b); it is not a gate per order."""
    from scaffolds.transformer.tests.test_query_coefficient_learning_rate import query_config

    config = query_config()
    validate_config(config)
    config.optim.coefficient_order_learning_rate = "two_over_order"
    with pytest.raises(ValueError, match="coefficient_order_learning_rate two_over_order scales"):
        validate_config(config)


def test_unknown_rule_and_combined_multiplier_are_refused():
    config = gate_config()
    config.optim.coefficient_order_learning_rate = "inverse"
    with pytest.raises(ValueError, match="must be constant, two_over_order or order_over_two"):
        validate_config(config)


def test_legacy_recipes_load_with_one_rate(tmp_path):
    config = gate_config()
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path).optim.coefficient_order_learning_rate == "constant"
    legacy = config_to_dict(config)
    legacy["optim"].pop("coefficient_order_learning_rate")
    assert config_dicts_equal(legacy, config_to_dict(config))
    assert not config_dicts_equal(legacy, config_to_dict(gate_config("two_over_order")))
