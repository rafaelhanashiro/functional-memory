"""Query readout LR changes affect a/b updates, not W, clipping, or Adam history."""

import copy
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.training.loop import ensure_run_config
from scaffolds.transformer.training.optim import (
    attention_weight_decay_mask, make_learning_rate_schedule, make_optimizer,
)
from scaffolds.transformer.training.state import create_state


READOUTS = {"query_polynomial_bias", "query_polynomial_amplitude"}


def query_config():
    config = Config()
    config.model = ModelConfig(
        vocab_size=16, max_seq_len=4, num_layers=2, hidden_size=8,
        num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
        intermediate_size=12, attention_backend="blocked", attention_degree=2,
        attention_normalization=False, attention_output_scaling="head_rms_norm",
        attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=0.1,
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
    )
    config.data.synthetic_vocab_size = 16
    config.data.sequence_length = 5
    config.data.batch_size = 4
    config.train.micro_batch_size = 2
    config.optim.total_steps = 8
    config.optim.warmup_steps = 2
    config.optim.learning_rate = 1e-3
    config.optim.min_learning_rate = 1e-4
    config.optim.grad_clip_norm = 0.1
    return config


def assert_equal_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("scale", [0.3, 3.0])
def test_only_a_b_updates_scale_after_clipping_and_adam(scale):
    config = query_config()
    params = create_state(config, jax.random.PRNGKey(0)).params
    reference_tx = make_optimizer(config)
    config.optim.query_coefficient_learning_rate_scale = scale
    validate_config(config)
    scaled_tx = make_optimizer(config)
    reference_state, scaled_state = reference_tx.init(params), scaled_tx.init(params)
    assert_equal_tree(reference_state, scaled_state)
    # Keep parameters and supplied gradients matched to isolate the optimizer's
    # response. Vary gradients over time to exercise moment normalization, and
    # cross warmup, peak, cosine decay, and the minimum schedule value.
    for step in range(config.optim.total_steps + 1):
        grads = jax.tree.map(
            lambda value: jnp.sin(jnp.arange(value.size).reshape(value.shape) + step) + 2,
            params,
        )
        assert float(optax.tree.norm(grads)) > config.optim.grad_clip_norm
        reference, reference_state = reference_tx.update(grads, reference_state, params)
        scaled, scaled_state = scaled_tx.update(grads, scaled_state, params)
        assert_equal_tree(reference_state, scaled_state)
        reference_values = dict(nnx.to_flat_state(reference))
        paths = set()
        for path, value in nnx.to_flat_state(scaled):
            paths.add(path[0])
            expected = reference_values[path][...]
            if path[0] in READOUTS:
                np.testing.assert_allclose(value[...], expected * scale, rtol=1e-6, atol=1e-9)
                if step:
                    assert np.any(expected != 0)
            else:
                np.testing.assert_array_equal(value[...], expected)
        assert READOUTS | {"query_polynomial_kernel", "token_embedding", "lm_head"} <= paths


@pytest.mark.parametrize("decay_projection", [False, True])
def test_projection_decay_does_not_decay_a_b_or_acquire_their_lr(decay_projection):
    config = query_config()
    config.model.attention_coefficient_kernel_weight_decay = decay_projection
    config.optim.query_coefficient_learning_rate_scale = 3.0
    config.optim.warmup_steps = 0
    config.optim.min_learning_rate = config.optim.learning_rate
    validate_config(config)
    state = create_state(config, jax.random.PRNGKey(0))
    # The actual initialization has a=0; nonzero values expose accidental decay.
    state = state.replace(params=jax.tree.map(jnp.ones_like, state.params))
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    params = dict(nnx.to_flat_state(updated.params))
    for name in READOUTS:
        np.testing.assert_array_equal(params[(name,)][...], 1.0)
    factor = 1 - config.optim.learning_rate * config.optim.weight_decay
    np.testing.assert_allclose(params[("query_polynomial_kernel",)][...],
                               factor if decay_projection else 1.0, rtol=1e-6)
    np.testing.assert_allclose(params[("lm_head", "kernel")][...], factor, rtol=1e-6)
    np.testing.assert_array_equal(params[("token_embedding", "embedding")][...], 1.0)


def test_default_preserves_existing_adamw_state_and_updates():
    config = query_config()
    params = create_state(config, jax.random.PRNGKey(0)).params
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    legacy_tx = optax.chain(
        optax.clip_by_global_norm(config.optim.grad_clip_norm),
        optax.adamw(make_learning_rate_schedule(config), b1=config.optim.beta1,
                    b2=config.optim.beta2, weight_decay=config.optim.weight_decay,
                    mask=partial(attention_weight_decay_mask, excluded_names=excluded)),
    )
    current_tx = make_optimizer(config)
    old_state, current_state = legacy_tx.init(params), current_tx.init(params)
    assert_equal_tree(old_state, current_state)
    for _ in range(3):
        grads = jax.tree.map(jnp.ones_like, params)
        old_updates, old_state = legacy_tx.update(grads, old_state, params)
        current_updates, current_state = current_tx.update(grads, current_state, params)
        assert_equal_tree(old_updates, current_updates)
        assert_equal_tree(old_state, current_state)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "3"])
def test_invalid_multiplier_is_rejected(scale):
    config = query_config()
    config.optim.query_coefficient_learning_rate_scale = scale
    with pytest.raises(ValueError, match="query_coefficient_learning_rate_scale must be finite and positive"):
        validate_config(config)


@pytest.mark.parametrize("overrides", [
    {"attention_coefficient_conditioning": "static", "attention_coefficient_activation": "identity",
     "attention_coefficient_amplitude_scale": "none", "attention_coefficient_amplitude_initial": 1.0,
     "attention_coefficient_projection_initial_std": 0.0},
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_activation": "affine_sigmoid"},
    {"attention_coefficient_readout": "dense"},
])
def test_nondefault_multiplier_rejects_other_parameterizations(overrides):
    config = query_config()
    for key, value in overrides.items():
        setattr(config.model, key, value)
    config.optim.query_coefficient_learning_rate_scale = 3.0
    with pytest.raises(ValueError, match="query_coefficient_learning_rate_scale requires"):
        validate_config(config)


def test_nondefault_multiplier_does_not_reinterpret_continuation_lr():
    config = query_config()
    config.optim.warmup_steps = 0
    config.optim.min_learning_rate = config.optim.learning_rate
    config.continuation.parent_config = "parent/config.yaml"
    config.continuation.parent_checkpoint = "parent/checkpoints/1"
    config.continuation.parent_config_sha256 = "0" * 64
    config.continuation.parent_checkpoint_sha256 = "0" * 64
    config.continuation.parent_step = 1
    config.continuation.new_parameter_learning_rate = 3e-3
    validate_config(config)
    config.optim.query_coefficient_learning_rate_scale = 3.0
    with pytest.raises(ValueError, match="not supported for checkpoint continuations"):
        validate_config(config)


def test_legacy_default_and_frozen_resume_recipe(tmp_path):
    config = query_config()
    config.train.workdir = tmp_path
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path).optim.query_coefficient_learning_rate_scale == 1.0
    ensure_run_config(config, resume=False)
    saved = json.loads((tmp_path / "config.json").read_text())
    saved["optim"].pop("query_coefficient_learning_rate_scale")
    old_bytes = json.dumps(saved).encode()
    (tmp_path / "config.json").write_bytes(old_bytes)
    assert config_dicts_equal(saved, config_to_dict(config))
    ensure_run_config(config, resume=True)
    config.optim.query_coefficient_learning_rate_scale = 3.0
    assert not config_dicts_equal(saved, config_to_dict(config))
    with pytest.raises(ValueError, match="config"):
        ensure_run_config(config, resume=True)
    assert (tmp_path / "config.json").read_bytes() == old_bytes
    # A newly recorded 3x run may resume its own unchanged recipe.
    config.train.workdir = tmp_path / "scaled"
    config.train.workdir.mkdir()
    ensure_run_config(config, resume=False)
    ensure_run_config(copy.deepcopy(config), resume=True)
