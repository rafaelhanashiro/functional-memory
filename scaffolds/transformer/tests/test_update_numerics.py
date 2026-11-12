"""Reject invalid updates even when their forward loss remains finite."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.training.numerics import (
    checked_update_loss,
    requires_update_checks,
    update_numerics,
)
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.state import create_state


@pytest.mark.parametrize("settings", [
    {},
    {"attention_degree": 0},
    {"attention_degree": 2},
    {"attention_degree": 4},
    {"attention_degree": 3},
    {"attention_degree": 2, "quadratic_mode": "positive"},
    {"attention_degree": 2, "quadratic_mode": "unrestricted"},
    {"attention_normalization": False},
    {"attention_degree": 2, "attention_normalization": False,
     "attention_coefficients": "learned"},
    {"attention_mechanism": "gla"},
])
def test_every_mechanism_requires_update_checks(settings):
    assert requires_update_checks(ModelConfig(**settings))


@pytest.mark.parametrize("check_all_gradients", [False, True])
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("invalid", [jnp.nan, jnp.inf, -jnp.inf])
def test_finite_loss_cannot_hide_invalid_gradient(check_all_gradients, learned, invalid):
    params = {"projection": {"kernel": jnp.ones((2,), dtype=jnp.float32)}}
    grads = {"projection": {"kernel": jnp.array([1.0, invalid], dtype=jnp.float32)}}
    if learned:
        params["quadratic_coefficients"] = jnp.ones((2,), dtype=jnp.float32)
        grads["quadratic_coefficients"] = jnp.zeros((2,), dtype=jnp.float32)

    # Use finite candidate weights so the gradient check itself must detect it.
    checked = jax.jit(lambda candidate: checked_update_loss(
        params, candidate, params, jnp.asarray(1.25),
        check_all_gradients=check_all_gradients,
    ))(grads)
    assert bool(jnp.isnan(checked))


@pytest.mark.parametrize("check_all_gradients", [False, True])
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("invalid", [jnp.nan, jnp.inf, -jnp.inf])
def test_finite_gradients_cannot_hide_corrupted_noncoefficient_weight(
    check_all_gradients, learned, invalid,
):
    params = {"projection": {"kernel": jnp.ones((2,), dtype=jnp.float32)}}
    updated = {"projection": {"kernel": jnp.array([1.0, invalid], dtype=jnp.float32)}}
    if learned:
        # The former learned-quadratic guard checked only these updated weights.
        params["quadratic_coefficients"] = jnp.ones((2,), dtype=jnp.float32)
        updated["quadratic_coefficients"] = jnp.ones((2,), dtype=jnp.float32)
    grads = jax.tree.map(jnp.zeros_like, params)
    checked = checked_update_loss(
        params, grads, updated, jnp.asarray(1.25),
        check_all_gradients=check_all_gradients,
    )
    assert bool(jnp.isnan(checked))


@pytest.mark.parametrize("check_all_gradients", [False, True])
def test_finite_gradients_with_overflowing_clipping_norm_are_rejected(check_all_gradients):
    params = {"kernel": jnp.ones((1,), dtype=jnp.float32)}
    grads = {"kernel": jnp.array([1e20], dtype=jnp.float32)}
    assert bool(jnp.all(jnp.isfinite(grads["kernel"])))
    clip = optax.clip_by_global_norm(1.0)
    clipped, _ = clip.update(grads, clip.init(params))
    np.testing.assert_array_equal(clipped["kernel"], jnp.zeros((1,)))
    checked = checked_update_loss(
        params, grads, params, jnp.asarray(1.25),
        check_all_gradients=check_all_gradients,
    )
    assert bool(jnp.isnan(checked))


@pytest.mark.parametrize("check_all_gradients", [False, True])
def test_finite_updates_preserve_loss_and_diagnostic_values(check_all_gradients):
    params = {"kernel": jnp.array([3.0, 4.0], dtype=jnp.float32)}
    grads = {"kernel": jnp.array([0.0, 2.0], dtype=jnp.float32)}
    loss = jnp.asarray(1.25, dtype=jnp.float32)
    checked = checked_update_loss(
        params, grads, params, loss, check_all_gradients=check_all_gradients,
    )
    assert checked.dtype == loss.dtype
    np.testing.assert_array_equal(checked, loss)
    np.testing.assert_array_equal(update_numerics(grads, params, loss), [1.25, 2.0, 5.0])


@pytest.mark.parametrize("degree,mode", [(None, "fixed"), (2, "fixed"), (2, "positive")])
@pytest.mark.parametrize("micro_batch_size", [2, 4])
def test_invalid_update_stops_training_before_checkpoint(tmp_path, monkeypatch, degree, mode, micro_batch_size):
    config = tiny_config(tmp_path)
    config.model.attention_degree = degree
    config.model.quadratic_mode = mode
    config.train.micro_batch_size = micro_batch_size

    def finite_loss_invalid_gradient(actual):
        # Keep the step's telemetry statistics; replace only the loss and gradient.
        def step(state, *args, **kwargs):
            _, _, *statistics = actual(state, *args, **kwargs)
            return jnp.array(2.0), jax.tree.map(lambda value: jnp.full_like(value, jnp.inf), state.params), *statistics
        return step

    monkeypatch.setattr(loop, "loss_and_grads", finite_loss_invalid_gradient(loop.loss_and_grads))
    monkeypatch.setattr(loop, "microbatch_loss_and_grads", finite_loss_invalid_gradient(loop.microbatch_loss_and_grads))
    with pytest.raises(FloatingPointError, match="optimizer update at step 1"):
        loop.run_training(config)
    failure = json.loads((tmp_path / "numerical_failure_step_00000001.json").read_text())
    assert failure["update_numerics"]["forward_loss"] == 2.0
    assert failure["update_numerics"]["global_gradient_norm"] == "Infinity"
    assert "updated_parameters" in failure["checked_quantities"]
    assert not (tmp_path / "completion.json").exists()
    assert not any(path.name.isdigit() for path in (tmp_path / "checkpoints").iterdir())


@pytest.mark.skipif(jax.local_device_count() < 2, reason="requires two JAX devices")
def test_distributed_default_step_cannot_bypass_update_checks(tmp_path, monkeypatch):
    from scaffolds.transformer.training import distributed

    config = tiny_config(tmp_path)
    state = create_state(config, jax.random.PRNGKey(0))

    def finite_loss_invalid_gradient(graphdef, params, *args):
        return jnp.array(2.0), jax.tree.map(lambda value: jnp.full_like(value, jnp.inf), params)

    monkeypatch.setattr(distributed, "_loss_and_grads", finite_loss_invalid_gradient)
    layout = distributed.resolve_batch_layout(config)
    functions = distributed.make_data_parallel_functions(state, tuple(jax.local_devices()))
    state = distributed.put_replicated(state, functions.replicated_sharding)
    batches = np.arange(20, dtype=np.int32).reshape(layout.grad_accum_steps, layout.global_micro_batch_size, 5) % 16
    _, loss = functions.train_step(
        state, distributed.put_train_batches(batches, functions.train_batch_sharding), jax.random.PRNGKey(1),
    )
    assert bool(jnp.isnan(loss))
