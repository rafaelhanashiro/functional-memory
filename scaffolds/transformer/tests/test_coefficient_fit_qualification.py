"""Qualification permits transient Adam overshoot without accepting a bad optimizer."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer.training.coefficient_fit import qualify


def _quadratic_probe(*, ascent=False, nonfinite_after=None):
    table = jnp.asarray([.003], jnp.float32)
    source = {"frozen": jnp.asarray([7.], jnp.float32)}
    tx = optax.scale_by_adam(b1=.9, b2=.95, eps=1e-8)
    optimizer = tx.init(table)

    @jax.jit
    def evaluate(value, source, tokens):
        return jnp.square(value)

    def capture(source, tokens):
        loss = evaluate(table, source, tokens)
        return table, loss, loss, jnp.asarray(0.)

    @jax.jit
    def step(value, state, source, tokens, lr):
        gradient = 2 * value
        update, state = tx.update(gradient, state, value)
        candidate = value + (1 if ascent else -1) * lr * update
        finite = jnp.all(jnp.isfinite(candidate))
        if nonfinite_after is not None:
            candidate = jnp.where(state.count >= nonfinite_after, jnp.nan, candidate)
            finite &= jnp.all(jnp.isfinite(candidate))
        return candidate, state, evaluate(value, source, tokens), jnp.abs(gradient), finite

    return (capture, tx.init, step, evaluate), source, table, optimizer


def test_adam_overshoot_can_qualify_without_changing_initialization(tmp_path):
    functions, source, table, optimizer = _quadratic_probe()
    original_table = np.asarray(table).copy()
    report = qualify(tmp_path, functions, source, jnp.zeros((1, 1)), table, optimizer, .01)
    assert report["one_update_nll"] > report["separate_training_forward_nll"]
    assert 1 < report["improving_update"] <= report["qualification_update_limit"] == 10
    assert report["improved_training_nll"] < report["separate_training_forward_nll"]
    assert report["qualification_nll_trace"][1] == report["one_update_nll"]
    assert len(report["qualification_nll_trace"]) == report["improving_update"] + 1
    assert report["checkpoint_resume_exact"]
    np.testing.assert_array_equal(table, original_table)
    assert int(optimizer.count) == 0


def test_qualification_rejects_updates_that_never_improve(tmp_path):
    functions, source, table, optimizer = _quadratic_probe(ascent=True)
    with pytest.raises(RuntimeError, match="did not improve NLL within 10 updates"):
        qualify(tmp_path, functions, source, jnp.zeros((1, 1)), table, optimizer, .01)


def test_qualification_rejects_nonfinite_update_after_first_overshoot(tmp_path):
    functions, source, table, optimizer = _quadratic_probe(nonfinite_after=3)
    with pytest.raises(RuntimeError, match="Nonfinite coefficient loss, gradient, or update"):
        qualify(tmp_path, functions, source, jnp.zeros((1, 1)), table, optimizer, .01)
