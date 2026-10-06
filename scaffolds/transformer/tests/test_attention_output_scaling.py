"""Contracts for rescaling attention outputs: 1/t and per-head RMSNorm.

``inverse_position`` divides query t's output by its t visible keys and requires
unnormalized attention. ``head_rms_norm`` normalizes each head's value vector and
multiplies a learned gain; for a positive kernel it equals the RMSNorm of the
normalized output, so unnormalized exp and softmax coincide after it, and it also
applies to softmax itself.
"""

import copy
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer.acceptance import attention_test_paths, matches_softmax_backbone_initialization
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.attention.parameters import attention_parameter_count
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.optim import make_optimizer
from scaffolds.transformer.training.state import create_state

TEST_PATH = "scaffolds/transformer/tests/test_attention_output_scaling.py"


def _model_config(scaling="none", **updates):
    values = dict(vocab_size=128, max_seq_len=16, num_layers=2, hidden_size=32, num_heads=4, num_kv_heads=4,
                  qk_head_dim=8, value_head_dim=8, intermediate_size=64, tie_embeddings=False,
                  attention_backend="blocked", attention_degree=None, attention_normalization=False,
                  qk_norm=True, attention_output_scaling=scaling)
    return ModelConfig(**{**values, **updates})


def _config_for(model_config):
    config = Config()
    config.model = model_config
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 128, 9, 2
    config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


def _operands(seed=3, time=13, heads=3, qk_dim=8, value_dim=5):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(0.7 * rng.normal(size=(2, time, heads, qk_dim)), jnp.float32) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(2, time, heads, value_dim)), jnp.float32)
    return q, k, v


def test_inverse_position_divides_each_query_by_its_visible_keys():
    q, k, v = _operands()
    positions = np.arange(1, q.shape[1] + 1)[None, :, None, None]
    with jax.default_matmul_precision("highest"):
        y = blocked_dot_product_attention(q, k, v, attention_normalization=False)
        np.testing.assert_allclose(scale_attention_output(y, "inverse_position"), np.asarray(y) / positions,
                                   rtol=1e-6)
        # With the constant kernel p_0 = 1 the scaled output is the running mean of the values.
        constant = blocked_dot_product_attention(q, k, v, attention_normalization=False, attention_degree=0)
        np.testing.assert_allclose(scale_attention_output(constant, "inverse_position"),
                                   np.cumsum(np.asarray(v), axis=1) / positions, rtol=1e-5, atol=1e-6)


def test_head_rms_norm_removes_the_row_normalizer():
    q, k, v = _operands()
    gain = jnp.asarray(np.random.default_rng(5).uniform(0.5, 2.0, size=(3, 5)), jnp.float32)
    with jax.default_matmul_precision("highest"):
        unnormalized = blocked_dot_product_attention(q, k, v, attention_normalization=False)
        softmax = blocked_dot_product_attention(q, k, v, attention_normalization=True)
        actual = scale_attention_output(unnormalized, "head_rms_norm", output_norm_scale=gain, eps=0.0)
        np.testing.assert_allclose(
            actual, scale_attention_output(softmax, "head_rms_norm", output_norm_scale=gain, eps=0.0),
            rtol=1e-5, atol=1e-6)
    y = np.asarray(unnormalized, np.float64)
    expected = y / np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True) + 1e-6) * np.asarray(gain)
    np.testing.assert_allclose(scale_attention_output(unnormalized, "head_rms_norm", output_norm_scale=gain),
                               expected, rtol=1e-5, atol=1e-6)


def test_scaling_rejects_inconsistent_arguments():
    y = _operands()[2]
    assert scale_attention_output(y, "none") is y
    with pytest.raises(ValueError, match="Unknown attention output scaling"):
        scale_attention_output(y, "mean")
    with pytest.raises(ValueError, match="output_norm_scale parameter"):
        scale_attention_output(y, "head_rms_norm")
    with pytest.raises(ValueError, match="output_norm_scale parameter"):
        scale_attention_output(y, "inverse_position", output_norm_scale=jnp.ones((3, 5)))
    with pytest.raises(ValueError, match="shape"):
        scale_attention_output(y, "head_rms_norm", output_norm_scale=jnp.ones((5,)))


def test_module_applies_the_scaling_before_the_output_projection():
    kwargs = dict(hidden_size=32, num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                  attention_normalization=False, backend="blocked")
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 11, 32)), jnp.float32)
    base = MultiHeadAttention(**kwargs, rngs=nnx.Rngs(8))
    inverse = MultiHeadAttention(**kwargs, attention_output_scaling="inverse_position", rngs=nnx.Rngs(8))
    rms = MultiHeadAttention(**kwargs, attention_output_scaling="head_rms_norm", rngs=nnx.Rngs(8))
    gain = {"output_norm_scale": jnp.ones((4, 8), jnp.float32)}
    with jax.default_matmul_precision("highest"):
        # The output projection has no bias, so dividing its input by t divides its output by t.
        np.testing.assert_allclose(inverse(x), np.asarray(base(x)) / np.arange(1, 12)[None, :, None],
                                   rtol=1e-5, atol=1e-6)
        before = rms(x, attention_parameters=gain)
        # Scaling every value by 3 triples the unnormalized output but leaves the RMSNorm output unchanged.
        rms.v_proj.kernel[...] = rms.v_proj.kernel[...] * 3.0
        np.testing.assert_allclose(rms(x, attention_parameters=gain), before, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="output_norm_scale"):
        rms(x)
    with pytest.raises(ValueError, match="output_norm_scale"):
        inverse(x, attention_parameters=gain)
    with pytest.raises(ValueError, match="only to unnormalized"):
        MultiHeadAttention(**{**kwargs, "attention_normalization": True}, attention_output_scaling="inverse_position",
                           rngs=nnx.Rngs(8))


def test_softmax_with_head_rms_norm_matches_unnormalized_exp_with_head_rms_norm():
    kwargs = dict(hidden_size=32, num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                  backend="blocked", attention_output_scaling="head_rms_norm")
    x = jnp.asarray(np.random.default_rng(2).normal(size=(2, 11, 32)), jnp.float32)
    softmax = MultiHeadAttention(**kwargs, attention_normalization=True, rngs=nnx.Rngs(8))
    unnormalized = MultiHeadAttention(**kwargs, attention_normalization=False, rngs=nnx.Rngs(8))
    gain = {"output_norm_scale": jnp.asarray(np.random.default_rng(4).uniform(0.5, 2.0, size=(4, 8)), jnp.float32)}
    with jax.default_matmul_precision("highest"):
        # Same weights; the two differ only through rms_norm_eps relative to each row's kernel sum.
        np.testing.assert_allclose(softmax(x, attention_parameters=dict(gain)),
                                   unnormalized(x, attention_parameters=dict(gain)), rtol=1e-4, atol=1e-5)
    config = _config_for(_model_config("head_rms_norm", attention_normalization=True))
    assert TEST_PATH in attention_test_paths(config_to_dict(config), "fused", True)
    assert "RMSNorm" in attention_identity(config)["attention_formula"]
    matches_softmax_backbone_initialization(config)


@pytest.mark.parametrize("scaling,updates,error", [
    ("inverse_position", dict(attention_normalization=True), "inverse_position applies only to unnormalized"),
    ("head_rms_norm", dict(attention_mechanism="gla", attention_normalization=True, qk_norm=False),
     "only to dot-product attention"),
    ("inverse_position", dict(attention_output_scaling="mean"), "Unknown attention output scaling"),
])
def test_invalid_configurations_are_rejected(scaling, updates, error):
    with pytest.raises(ValueError, match=error):
        _model_config(scaling, **updates)


def test_default_and_recipes_without_the_field_mean_no_scaling():
    assert ModelConfig().attention_output_scaling == "none"
    current = config_to_dict(_config_for(_model_config()))
    legacy = copy.deepcopy(current)
    del legacy["model"]["attention_output_scaling"]
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(current, config_to_dict(_config_for(_model_config("inverse_position"))))


def test_gain_is_a_model_parameter_without_weight_decay_and_backbone_is_unchanged():
    config = _config_for(_model_config("head_rms_norm"))
    state = create_state(config, jax.random.PRNGKey(0))
    scale = np.asarray(state.params["output_norm_scale"][...])
    assert scale.shape == (2, 4, 8) and np.all(scale == 1)
    assert attention_parameter_count(config.model) == attention_parameter_count(_model_config()) + 2 * 4 * 8
    tx = make_optimizer(config)
    grads = jax.tree.map(jnp.zeros_like, state.params)
    updates, _ = tx.update(grads, tx.init(state.params), state.params)
    updated = optax.apply_updates(state.params, updates)
    np.testing.assert_array_equal(updated["output_norm_scale"][...], scale)
    inverse = _config_for(_model_config("inverse_position"))
    assert "output_norm_scale" not in create_state(inverse, jax.random.PRNGKey(0)).params
    for current in (config, inverse):
        matches_softmax_backbone_initialization(current)


def test_identity_tags_and_acceptance_name_the_scaling():
    for scaling, phrase in (("inverse_position", "visible keys t"), ("head_rms_norm", "RMSNorm")):
        config = _config_for(_model_config(scaling))
        identity = attention_identity(config)
        assert identity["attention_output_scaling"] == identity["attention_output_normalization"] == scaling
        assert phrase in identity["attention_formula"]
        assert f"output_{scaling}" in tracking.run_tags(config, {})
        assert TEST_PATH in attention_test_paths(config_to_dict(config), "fused", True)
    plain = _config_for(_model_config())
    assert "attention_output_scaling" not in attention_identity(plain)
    assert not any(tag.startswith("output_") for tag in tracking.run_tags(plain, {}))
    assert TEST_PATH not in attention_test_paths(config_to_dict(plain), "fused", True)


@pytest.mark.parametrize("scaling,degree,coefficients,normalization", [
    ("inverse_position", None, "fixed", False),
    ("head_rms_norm", 2, "learned", False),
    ("head_rms_norm", None, "fixed", True),
])
def test_training_resumes_exactly(tmp_path, scaling, degree, coefficients, normalization):
    config = tiny_config(tmp_path / "full")
    config.model.attention_backend = "blocked"
    config.model.attention_degree = degree
    config.model.attention_normalization = normalization
    config.model.attention_coefficients = coefficients
    config.model.attention_coefficient_sharing = "per_head"
    config.model.qk_norm = True
    config.model.attention_output_scaling = scaling
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
    assert resumed["restored_step"] == 1
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
    if scaling == "head_rms_norm":
        assert np.any(np.asarray(states[0].params["output_norm_scale"][...]) != 1)
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["attention_output_scaling"] == scaling
