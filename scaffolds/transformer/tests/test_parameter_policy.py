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
