"""The optimizer consumes mechanism-owned parameter policy generically."""
from types import SimpleNamespace
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax import nnx

from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import Config, ModelConfig, load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.training import optim
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.loop import ensure_run_config
from scaffolds.transformer.training.tracking import run_tags


def test_new_parameter_name_can_disable_decay_without_changing_trainer(monkeypatch):
    monkeypatch.setattr(optim, "attention_parameter_specs", lambda _: (
        SimpleNamespace(name="temperature", weight_decay=False),
        SimpleNamespace(name="scale", weight_decay=True),
    ))
    config = Config()
    config.optim.warmup_steps = 0
    params = {"temperature": jnp.array([2.0]), "scale": jnp.array([3.0]),
              "projection": {"kernel": jnp.array([4.0])}}
    tx = optim.make_optimizer(config)
    grads = jax.tree.map(jnp.zeros_like, params)
    updates, _ = tx.update(grads, tx.init(params), params)
    updated = optax.apply_updates(params, updates)
    np.testing.assert_array_equal(updated["temperature"], params["temperature"])
    factor = 1 - config.optim.learning_rate * config.optim.weight_decay
    np.testing.assert_allclose(updated["scale"], params["scale"] * factor)
    np.testing.assert_allclose(updated["projection"]["kernel"], params["projection"]["kernel"] * factor)


@pytest.mark.parametrize("mechanism", ("dot_product", "gla"))
def test_canonical_decay_groups_remain_trainable(mechanism):
    config = Config()
    config.model = ModelConfig(
        vocab_size=16, max_seq_len=4, num_layers=1, hidden_size=8,
        num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
        intermediate_size=12, attention_backend="blocked", attention_mechanism=mechanism,
    )
    config.data.synthetic_vocab_size = 16
    config.train.dtype = "bfloat16"
    config.optim.warmup_steps = 0
    config.optim.grad_clip_norm = 1e6
    state = create_state(config, jax.random.PRNGKey(0))
    # A nonzero bias makes accidental bias decay visible; initialized zero biases
    # would pass a zero-gradient check even with an incorrect decay mask.
    state = state.replace(params=jax.tree.map(jnp.ones_like, state.params))
    zero_update = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    learned_update = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    zeros = dict(nnx.to_flat_state(zero_update.params))
    learned = dict(nnx.to_flat_state(learned_update.params))
    paths = set()
    for path, variable in nnx.to_flat_state(state.params):
        paths.add(path)
        no_decay = path[-1] in ("embedding", "scale", "bias", "qk_norm_log_scale")
        decay = 0.0 if no_decay else config.optim.weight_decay
        np.testing.assert_allclose(zeros[path][...], variable[...] * (1 - config.optim.learning_rate * decay),
                                   rtol=1e-6, atol=1e-7)
        # Removing decay must not freeze the parameter or bypass Adam's gradient
        # update. Its first bias-corrected update for a unit gradient is one.
        np.testing.assert_allclose(learned[path][...],
                                   variable[...] * (1 - config.optim.learning_rate * decay) - config.optim.learning_rate,
                                   rtol=1e-6, atol=1e-7)
    assert ("token_embedding", "embedding") in paths
    assert ("lm_head", "kernel") in paths
    assert ("final_norm", "scale") in paths
    if mechanism == "gla":
        assert ("layers_0", "attention", "gate_proj", "bias") in paths
        assert zeros[("layers_0", "attention", "head_norm", "scale")].shape == (2, 4)


def test_old_run_cannot_resume_with_new_decay_policy(tmp_path):
    config = Config()
    config.model.attention_backend = "blocked"
    config.train.workdir = tmp_path
    ensure_run_config(config, resume=False)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    assert saved["optim"]["weight_decay_policy"] == config.optim.weight_decay_policy
    ensure_run_config(config, resume=True)
    del saved["optim"]["weight_decay_policy"]
    old_bytes = json.dumps(saved).encode()
    path.write_bytes(old_bytes)
    with pytest.raises(ValueError, match="weight-decay policy"):
        ensure_run_config(config, resume=True)
    assert path.read_bytes() == old_bytes


def query_key_policy_config(*, query_decay=False, key_decay=False):
    config = Config(model=ModelConfig(
        vocab_size=16, max_seq_len=4, num_layers=2, hidden_size=8,
        num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
        intermediate_size=12, attention_backend="blocked", attention_degree=4,
        attention_normalization=False, attention_output_scaling="head_rms_norm",
        attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query_key", attention_coefficient_activation="tanh",
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=0.1,
        attention_coefficient_kernel_weight_decay=query_decay,
        attention_coefficient_key_kernel_weight_decay=key_decay,
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
    ))
    config.data.synthetic_vocab_size = 16
    config.data.sequence_length = 5
    config.data.batch_size = 4
    config.train.micro_batch_size = 2
    config.optim.warmup_steps = 0
    config.optim.min_learning_rate = config.optim.learning_rate
    config.optim.grad_clip_norm = 1e6
    validate_config(config)
    return config


@pytest.mark.parametrize("query_decay,key_decay", [(False, False), (True, False), (False, True), (True, True)])
def test_query_key_projection_decay_flags_are_independent(query_decay, key_decay):
    config = query_key_policy_config(query_decay=query_decay, key_decay=key_decay)
    state = create_state(config, jax.random.PRNGKey(0))
    # Nonzero a and b expose accidental decay, including key a which initializes at zero.
    state = state.replace(params=jax.tree.map(jnp.ones_like, state.params))
    zero_update = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    gradient_update = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    zeros = dict(nnx.to_flat_state(zero_update.params))
    learned = dict(nnx.to_flat_state(gradient_update.params))
    readouts = {"query_polynomial_bias", "query_polynomial_amplitude", "key_polynomial_amplitude"}
    projections = {"query_polynomial_kernel": query_decay, "key_polynomial_kernel": key_decay}
    paths = set()
    for path, value in nnx.to_flat_state(state.params):
        paths.add(path)
        if path[0] in projections:
            decays = projections[path[0]]
        else:
            decays = (path[0] not in readouts | {"output_norm_scale"} and not path[0].startswith("qk_norm_")
                      and path[-1] not in ("embedding", "scale", "bias"))
        expected = value[...] * (1 - config.optim.learning_rate * config.optim.weight_decay * decays)
        np.testing.assert_allclose(zeros[path][...], expected, rtol=1e-6, atol=1e-7, err_msg=str(path))
        np.testing.assert_allclose(learned[path][...], expected - config.optim.learning_rate,
                                   rtol=1e-6, atol=1e-7, err_msg=str(path))
    assert {(name,) for name in readouts | projections.keys()} <= paths
    assert {("token_embedding", "embedding"), ("final_norm", "scale"), ("lm_head", "kernel")} <= paths
    identity = attention_identity(config)
    tags = run_tags(config, {})
    assert identity.get("attention_coefficient_kernel_weight_decay", False) is query_decay
    assert identity.get("attention_coefficient_key_kernel_weight_decay", False) is key_decay
    assert ("coefficient_kernel_weight_decay" in tags) is query_decay
    assert ("coefficient_key_kernel_weight_decay" in tags) is key_decay


def test_key_projection_decay_preserves_initialization_acceptance():
    config = query_key_policy_config(query_decay=True, key_decay=True)
    facts = acceptance.matches_softmax_backbone_initialization(config)
    parameters = facts["mechanism_parameters"]
    for name in ("query_polynomial_kernel", "key_polynomial_kernel"):
        assert parameters[name]["weight_decay"] is True
    for name in ("query_polynomial_bias", "query_polynomial_amplitude", "key_polynomial_amplitude"):
        assert parameters[name]["weight_decay"] is False
    spec = acceptance.attention_spec(config, entry_module="scaffolds.transformer.train")
    assert "scaffolds/transformer/tests/test_parameter_policy.py" in spec.cpu_tests


@pytest.mark.parametrize("value", [True, 1, None, "false"])
def test_key_projection_decay_requires_boolean_and_query_key_conditioning(value):
    model = query_key_policy_config().model
    if value is True:
        with pytest.raises(ValueError, match="requires query-key-conditioned"):
            replace(model, attention_coefficient_conditioning="query", attention_coefficient_key_kernel_weight_decay=value)
    else:
        with pytest.raises(ValueError, match="key_kernel_weight_decay must be a boolean"):
            replace(model, attention_coefficient_key_kernel_weight_decay=value)


def test_legacy_query_projection_decay_never_enables_key_decay_on_resume(tmp_path):
    config = query_key_policy_config(query_decay=True)
    config.train.workdir = tmp_path
    recipe = config_to_dict(config)
    recipe["model"].pop("attention_coefficient_key_kernel_weight_decay")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(recipe))
    legacy = load_config(path)
    assert legacy.model.attention_coefficient_key_kernel_weight_decay is False
    assert config_dicts_equal(recipe, config_to_dict(config))
    ensure_run_config(config, resume=False)
    runtime_path = tmp_path / "config.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["model"].pop("attention_coefficient_key_kernel_weight_decay")
    original = json.dumps(runtime).encode()
    runtime_path.write_bytes(original)
    ensure_run_config(config, resume=True)
    config.model.attention_coefficient_key_kernel_weight_decay = True
    assert not config_dicts_equal(recipe, config_to_dict(config))
    with pytest.raises(ValueError, match="config"):
        ensure_run_config(config, resume=True)
    assert runtime_path.read_bytes() == original
