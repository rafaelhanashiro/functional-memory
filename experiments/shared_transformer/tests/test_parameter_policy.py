"""The optimizer consumes mechanism-owned parameter policy generically."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.shared_transformer.configs import Config
from experiments.shared_transformer.training import optim


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
