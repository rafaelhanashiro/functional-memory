"""The optimizer consumes mechanism-owned parameter policy generically."""
from types import SimpleNamespace
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training import optim
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.loop import ensure_run_config


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


@pytest.mark.parametrize("variant", ("path", "dense_k5", "dense_k9"))
def test_coefficient_projection_decay_preserves_initialization_and_readout_policy(variant):
    """Decay only coefficient W; dense readouts and PaTH's existing groups keep their policy."""
    features = int(variant.removeprefix("dense_k")) if variant.startswith("dense") else None
    states, configs = [], []
    for decay_projection in (False, True):
        config = Config(model=ModelConfig(
            vocab_size=16, max_seq_len=4, num_layers=2, hidden_size=8,
            num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
            intermediate_size=12, attention_backend="blocked", attention_degree=4,
            attention_normalization=False, attention_output_scaling="head_rms_norm",
            attention_coefficients="learned", attention_coefficient_sharing="per_head",
            attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
            attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
            attention_coefficient_amplitude_initial=0.0, attention_coefficient_projection_initial_std=8 ** -.5,
            attention_coefficient_kernel_weight_decay=decay_projection,
            attention_coefficient_readout="dense" if features else "diagonal",
            attention_coefficient_feature_count=features,
            attention_coefficient_readout_scale=features ** -.5 if features else 1.0,
            positional_encoding="path" if variant == "path" else "rope",
            qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
            initialization="transformer_truncated_reference",
        ))
        config.data.synthetic_vocab_size = 16
        config.data.sequence_length = 5
        config.optim.warmup_steps = 0
        config.optim.min_learning_rate = config.optim.learning_rate
        config.optim.grad_clip_norm = 1e6
        configs.append(config)
        states.append(create_state(config, jax.random.PRNGKey(42)))

    initial = [dict(nnx.to_flat_state(state.params)) for state in states]
    assert initial[0].keys() == initial[1].keys()
    for path in initial[0]:
        np.testing.assert_array_equal(initial[0][path][...], initial[1][path][...])
    counts = [sum(leaf.size for leaf in jax.tree.leaves(state.params)) for state in states]
    assert counts[0] == counts[1] == model_parameter_count(configs[0].model)

    for config, state in zip(configs, states, strict=True):
        # Nonzero values expose accidental decay of a/V or biases initialized to zero.
        state = state.replace(params=jax.tree.map(jnp.ones_like, state.params))
        zero_update = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
        learned_update = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
        zeros = dict(nnx.to_flat_state(zero_update.params))
        learned = dict(nnx.to_flat_state(learned_update.params))
        decayed = {("lm_head", "kernel")}
        exempt = {(name,) for name in (
            "query_polynomial_bias", "qk_norm_q_gain", "qk_norm_k_gain",
            "qk_norm_head_log_scale", "output_norm_scale",
            "query_polynomial_mixing" if features else "query_polynomial_amplitude",
        )} | {("token_embedding", "embedding"), ("final_norm", "scale")}
        (decayed if config.model.attention_coefficient_kernel_weight_decay else exempt).add(
            ("query_polynomial_kernel",))
        if variant == "path":
            decayed |= {(name,) for name in (
                "path_w_down", "path_w_up", "path_w_conv", "path_beta_kernel")}
            exempt.add(("path_beta_bias",))
        for paths, decay in ((decayed, config.optim.weight_decay), (exempt, 0.0)):
            expected = 1 - config.optim.learning_rate * decay
            for path in paths:
                np.testing.assert_allclose(zeros[path][...], expected, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(learned[path][...], expected - config.optim.learning_rate,
                                           rtol=1e-6, atol=1e-7)
