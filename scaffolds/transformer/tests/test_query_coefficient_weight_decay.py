"""Independent AdamW decay for the affine query generator's W, amplitude a, and baseline b."""
import copy
from dataclasses import replace
import hashlib
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.tests.test_query_coefficient_learning_rate import query_config
from scaffolds.transformer.training.continuation import _parent_config
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.loop import ensure_run_config
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import run_tags


FLAGS = ("attention_coefficient_amplitude_weight_decay", "attention_coefficient_bias_weight_decay")


def _config(activation="tanh"):
    config = query_config()
    config.model.attention_degree = 4
    config.model.attention_coefficient_activation = activation
    config.optim.warmup_steps = 0
    config.optim.min_learning_rate = config.optim.learning_rate
    config.optim.weight_decay = 0.1
    return config


@pytest.mark.parametrize("activation", ["tanh", "affine_sigmoid"])
@pytest.mark.parametrize("kernel,amplitude,bias", [
    (False, False, False), (True, False, False), (True, True, False),
    (True, False, True), (True, True, True), (False, True, True),
])
def test_actual_adamw_update_decays_only_requested_arrays(activation, kernel, amplitude, bias):
    config = _config(activation)
    config.model.attention_coefficient_kernel_weight_decay = kernel
    config.model.attention_coefficient_amplitude_weight_decay = amplitude
    config.model.attention_coefficient_bias_weight_decay = bias
    config.model.attention_coefficient_tanh_bias = True
    validate_config(config)
    state = create_state(config, jax.random.PRNGKey(0))
    # a and the inner bias normally start at zero. Nonzero arrays expose decay
    # errors, independently of gradients and Adam moments.
    state = state.replace(params=jax.tree.map(jnp.ones_like, state.params))
    updated = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    learned = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    values = dict(nnx.to_flat_state(updated.params))
    learned_values = dict(nnx.to_flat_state(learned.params))
    factor = 1 - config.optim.learning_rate * config.optim.weight_decay
    for name, enabled in (("query_polynomial_kernel", kernel),
                          ("query_polynomial_amplitude", amplitude),
                          ("query_polynomial_bias", bias)):
        np.testing.assert_allclose(values[(name,)][...], factor if enabled else 1.0, rtol=1e-7)
        # A unit gradient still gives Adam's first bias-corrected learning-rate
        # step (up to epsilon), in addition to the independently selected decay.
        expected = (factor if enabled else 1.0) - config.optim.learning_rate
        assert np.isfinite(learned_values[(name,)][...]).all()
        np.testing.assert_allclose(learned_values[(name,)][...], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(values[("query_polynomial_tanh_bias",)][...], 1.0)
    np.testing.assert_array_equal(values[("token_embedding", "embedding")][...], 1.0)
    np.testing.assert_array_equal(values[("final_norm", "scale")][...], 1.0)
    np.testing.assert_allclose(values[("lm_head", "kernel")][...], factor, rtol=1e-7)


@pytest.mark.parametrize("name", FLAGS)
@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_decay_flags_require_booleans(name, value):
    config = _config()
    setattr(config.model, name, value)
    with pytest.raises(ValueError, match=f"{name} must be a boolean"):
        validate_config(config)


@pytest.mark.parametrize("name", FLAGS)
@pytest.mark.parametrize("overrides", [
    {"attention_coefficient_conditioning": "static", "attention_coefficient_activation": "identity",
     "attention_coefficient_amplitude_scale": "none", "attention_coefficient_amplitude_initial": 1.0,
     "attention_coefficient_projection_initial_std": 0.0},
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_readout": "dense"},
    {"attention_coefficient_activation": "identity", "attention_degree": 1,
     "attention_coefficient_amplitude_scale": "none", "attention_coefficient_amplitude_initial": 1.0,
     "attention_coefficient_projection_initial_std": 0.0},
    {"attention_coefficient_activation": "pure_tanh", "attention_coefficient_amplitude_initial": 1.0,
     "attention_coefficient_scale": "none"},
    {"attention_coefficient_activation": "sigmoid", "attention_coefficient_amplitude_initial": 1.0,
     "attention_coefficient_scale": "none"},
])
def test_decay_flags_reject_other_parameterizations(name, overrides):
    config = _config()
    for key, value in overrides.items():
        setattr(config.model, key, value)
    setattr(config.model, name, True)
    with pytest.raises(ValueError, match=f"{name} requires learned diagonal query-only affine coefficients"):
        validate_config(config)


def test_legacy_config_and_resume_preserve_exemption(tmp_path):
    config = _config()
    config.train.workdir = tmp_path
    path = tmp_path / "config.yaml"
    legacy = config_to_dict(config)
    for name in FLAGS:
        legacy["model"].pop(name)
    path.write_text(yaml.safe_dump(legacy))
    loaded = load_config(path)
    assert all(getattr(loaded.model, name) is False for name in FLAGS)
    assert config_dicts_equal(legacy, config_to_dict(loaded))
    ensure_run_config(loaded, resume=False)
    saved_path = tmp_path / "config.json"
    saved = json.loads(saved_path.read_text())
    for name in FLAGS:
        saved["model"].pop(name)
    original = json.dumps(saved).encode()
    saved_path.write_bytes(original)
    ensure_run_config(loaded, resume=True)
    assert saved_path.read_bytes() == original
    for name in FLAGS:
        changed = copy.deepcopy(loaded)
        setattr(changed.model, name, True)
        assert not config_dicts_equal(saved, config_to_dict(changed))
        with pytest.raises(ValueError, match="config"):
            ensure_run_config(changed, resume=True)
        assert saved_path.read_bytes() == original
    config.model.attention_coefficient_amplitude_weight_decay = True
    config.model.attention_coefficient_bias_weight_decay = True
    save_config(config, path)
    assert config_to_dict(load_config(path)) == config_to_dict(config)


@pytest.mark.parametrize("activation", ["tanh", "affine_sigmoid"])
def test_acceptance_initialization_and_report_include_decay_policy(activation):
    config = _config(activation)
    before = attention_identity(config)
    assert not any(name in before for name in FLAGS)
    config.model.attention_coefficient_amplitude_weight_decay = True
    config.model.attention_coefficient_bias_weight_decay = True
    validate_config(config)
    report = attention_identity(config)
    assert all(report[name] is True for name in FLAGS)
    assert {"coefficient_amplitude_weight_decay", "coefficient_bias_weight_decay"} <= set(run_tags(config, {}))
    spec = acceptance.attention_spec(config, entry_module="scaffolds.transformer.acceptance")
    paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=True)
    assert "scaffolds/transformer/tests/test_query_coefficient_weight_decay.py" in paths
    assert acceptance.matches_softmax_backbone_initialization(config)


def test_continuations_preserve_existing_policy_but_can_set_new_readout_policy(tmp_path):
    parent = _config()
    path = tmp_path / "parent.yaml"
    save_config(parent, path)
    child = copy.deepcopy(parent)
    child.continuation.parent_config = str(path)
    child.continuation.parent_config_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    child.continuation.parent_checkpoint = "unused"
    child.continuation.parent_step = 1
    child.continuation.preserve_all_parameters = True
    child.model.attention_coefficient_amplitude_weight_decay = True
    child.model.attention_coefficient_bias_weight_decay = True
    with pytest.raises(ValueError, match="Continuation changes parent model.attention_coefficient_amplitude_weight_decay"):
        _parent_config(child)
    parent.model = replace(parent.model, attention_coefficients="fixed", attention_coefficient_conditioning="static",
                           attention_coefficient_activation="identity", attention_coefficient_amplitude_initial=1.0,
                           attention_coefficient_projection_initial_std=0.0,
                           attention_coefficient_amplitude_scale="none", attention_coefficient_scale="none")
    save_config(parent, path)
    child.continuation.parent_config_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    child.continuation.preserve_all_parameters = False
    restored = _parent_config(child)
    assert all(getattr(restored.model, name) is False for name in FLAGS)
