"""Optional decay affects only the query-temperature direction, including after continuation restore."""

import copy
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from scaffolds.records import _tree_descriptor
from scaffolds.transformer.configs import ContinuationConfig, save_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.tests.test_token_query_scale import _batch, _config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.continuation import initialize_continuation_state
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.state import create_state


def test_token_scale_decay_configuration_preserves_legacy_semantics():
    config = _config()
    current = config_to_dict(config)
    legacy = copy.deepcopy(current)
    del legacy["model"]["qk_norm_token_scale_kernel_weight_decay"]
    assert config_dicts_equal(legacy, current)
    current["model"]["qk_norm_token_scale_kernel_weight_decay"] = True
    assert not config_dicts_equal(legacy, current)
    for invalid in (0, 1, "true", None):
        with pytest.raises(ValueError, match="must be a boolean"):
            _config(qk_norm_token_scale_kernel_weight_decay=invalid)
    with pytest.raises(ValueError, match="requires an enabled"):
        _config("none", qk_norm_token_scale_kernel_weight_decay=True)


def test_decay_does_not_change_initial_parameters_or_model_function():
    states = []
    for decay in (False, True):
        config = _config(qk_norm_token_scale_kernel_weight_decay=decay)
        states.append(create_state(config, jax.random.PRNGKey(4)))
        identity = attention_identity(config)
        assert identity["qk_norm_token_scale_kernel_weight_decay"] is decay
        assert identity["qk_norm_token_scale_amplitude_weight_decay"] is False
    for left, right in zip(jax.tree.leaves(states[0].params), jax.tree.leaves(states[1].params), strict=True):
        np.testing.assert_array_equal(left, right)
    batch = _batch(config)
    np.testing.assert_array_equal(
        loop.eval_step(states[0], batch, jax.random.PRNGKey(0)),
        loop.eval_step(states[1], batch, jax.random.PRNGKey(0)),
    )


@pytest.fixture(scope="module")
def token_scale_parent(tmp_path_factory):
    directory = tmp_path_factory.mktemp("token-scale-decay-parent")
    config = _config("none", num_layers=1)
    config.train.workdir = directory
    config.optim.learning_rate = config.optim.min_learning_rate = 1e-4
    config_path = directory / "config.yaml"
    save_config(config, config_path)
    state = create_state(config, jax.random.PRNGKey(config.train.seed))
    state, *_ = loop.train_step(state, _batch(config), jax.random.PRNGKey(1))
    checkpoint = directory / "checkpoint" / "1"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(checkpoint / "default", {"state": state})
    continuation = ContinuationConfig(
        parent_config=str(config_path),
        parent_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        parent_checkpoint=str(checkpoint),
        parent_checkpoint_sha256=_tree_descriptor(checkpoint)["sha256"],
        parent_step=1,
        new_parameter_learning_rate=2e-3,
    )
    return config, continuation


@pytest.mark.parametrize("decay", (False, True))
def test_continuation_zero_gradient_update_decays_only_direction(token_scale_parent, tmp_path, decay):
    parent, continuation = token_scale_parent
    config = copy.deepcopy(parent)
    config.train.workdir = tmp_path
    config.continuation = continuation
    config.model.qk_norm_token_scale = "cosine"
    config.model.qk_norm_token_scale_kernel_weight_decay = decay
    state, audit = initialize_continuation_state(config)
    assert audit["parent_parameters_sha256"] == audit["backbone_parameters_sha256"]
    assert audit["parent_optimizer_sha256"] == audit["backbone_optimizer_sha256"]
    assert audit["backbone_optimizer_count"] == 1
    assert audit["new_coefficient_optimizer_count"] == 0
    # A nonzero amplitude detects accidental amplitude decay, which its zero
    # initialization would hide. New parameters still have zero Adam moments.
    state.params["qk_norm_token_scale_amplitude"][...] = jnp.full_like(
        state.params["qk_norm_token_scale_amplitude"][...], 2.0)
    initial_kernel = np.asarray(state.params["qk_norm_token_scale_kernel"][...]).copy()
    initial_amplitude = np.asarray(state.params["qk_norm_token_scale_amplitude"][...]).copy()
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    actual = np.asarray(updated.params["qk_norm_token_scale_kernel"][...])
    if decay:
        factor = 1 - continuation.new_parameter_learning_rate * config.optim.weight_decay
        np.testing.assert_allclose(actual, initial_kernel * factor, rtol=1e-7, atol=0)
        assert np.any(actual != initial_kernel)
    else:
        np.testing.assert_array_equal(actual, initial_kernel)
    np.testing.assert_array_equal(updated.params["qk_norm_token_scale_amplitude"][...], initial_amplitude)
    assert int(updated.opt_state[1].coefficients[0].count) == 1
    assert int(updated.opt_state[1].backbone[0].count) == 2
    # Exemption does not freeze either parameter: gradient-driven updates still
    # train both the direction and the amplitude under either decay policy.
    learned = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    assert np.any(np.asarray(learned.params["qk_norm_token_scale_kernel"][...]) != actual)
    assert np.any(np.asarray(learned.params["qk_norm_token_scale_amplitude"][...]) != initial_amplitude)
