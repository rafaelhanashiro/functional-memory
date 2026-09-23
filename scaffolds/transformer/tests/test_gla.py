"""Independent recurrence checks for the OFM-derived GLA mechanism."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.gla import GatedLinearAttention, chunkwise_gla


def sequential_gla(q, k, v, log_alpha):
    """Token-by-token fp32 oracle independent of associative/chunk kernels."""
    initial = jnp.zeros((q.shape[0], q.shape[2], q.shape[3], v.shape[3]), jnp.float32)

    def step(state, inputs):
        q_t, k_t, v_t, g_t = (value.astype(jnp.float32) for value in inputs)
        state = jnp.exp(g_t)[..., None] * state + k_t[..., :, None] * v_t[..., None, :]
        read = jnp.einsum("bhk,bhkv->bhv", q_t, state)
        return state, read

    state, output = jax.lax.scan(step, initial, tuple(jnp.swapaxes(x, 0, 1) for x in (q, k, v, log_alpha)))
    return jnp.swapaxes(output, 0, 1).astype(q.dtype), state


def operands(key_dim, *, seq_len=11, value_dim=5, dtype=jnp.float32, seed=82):
    keys = jax.random.split(jax.random.key(seed), 4)
    qk_shape = (1, seq_len, 2, key_dim)
    q, k = [(jax.random.normal(key, qk_shape) * 0.15).astype(dtype) for key in keys[:2]]
    v = (jax.random.normal(keys[2], (1, seq_len, 2, value_dim)) * 0.2).astype(dtype)
    log_alpha = -jax.nn.softplus(jax.random.normal(keys[3], qk_shape)) / 16
    return q, k, v, log_alpha


def recurrence_loss(fn, *args):
    output, state = fn(*args)
    return jnp.square(output.astype(jnp.float32)).mean() + 0.03 * jnp.square(state).mean()


@pytest.mark.parametrize("key_dim,chunk_size", [(2, 4), (8, 1), (8, 4), (32, 16)])
def test_associative_outputs_final_state_and_all_gradients_match_sequential(key_dim, chunk_size):
    args = operands(key_dim)
    actual_fn = partial(chunkwise_gla, chunk_size=chunk_size)
    actual, expected = jax.jit(actual_fn)(*args), jax.jit(sequential_gla)(*args)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=2e-7)
    actual_grads = jax.jit(jax.grad(partial(recurrence_loss, actual_fn), argnums=(0, 1, 2, 3)))(*args)
    expected_grads = jax.jit(jax.grad(partial(recurrence_loss, sequential_gla), argnums=(0, 1, 2, 3)))(*args)
    for observed, reference in zip(actual_grads, expected_grads, strict=True):
        assert bool(jnp.all(jnp.isfinite(observed)))
        np.testing.assert_allclose(observed, reference, rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize("log_decay", [0., -3., -100.])
def test_zero_decay_and_strong_decay_are_finite(log_decay):
    q, k, v, log_alpha = operands(2, seq_len=65)
    args = q, k, v, jnp.full_like(log_alpha, log_decay)
    actual = jax.jit(chunkwise_gla)(*args)
    expected = jax.jit(sequential_gla)(*args)
    for observed, reference in zip(actual, expected, strict=True):
        assert bool(jnp.all(jnp.isfinite(observed)))
        np.testing.assert_allclose(observed, reference, atol=3e-7, rtol=2e-5)


def test_read_is_inclusive_and_no_softmax_scale_or_denominator():
    q = jnp.asarray([[[[2., 3.]], [[5., 7.]]]])
    k = jnp.asarray([[[[11., 13.]], [[17., 19.]]]])
    v = jnp.asarray([[[[23.]], [[29.]]]])
    output, state = chunkwise_gla(q, k, v, jnp.zeros_like(q))
    np.testing.assert_array_equal(output, [[[[23 * (2 * 11 + 3 * 13)]],
                                            [[23 * (5 * 11 + 7 * 13) + 29 * (5 * 17 + 7 * 19)]]]])
    np.testing.assert_array_equal(state, [[[[11 * 23 + 17 * 29], [13 * 23 + 19 * 29]]]])


def test_module_retains_ofm_constructor_order_and_reports_invalid_gate():
    module = GatedLinearAttention(16, 2, 8, 12, chunk_size=4, rngs=nnx.Rngs(5))
    x = jax.random.normal(jax.random.key(8), (1, 9, 16))
    output, diagnostics = module(x, return_diagnostics=True)
    np.testing.assert_array_equal(output, module(x))
    assert output.shape == x.shape
    assert float(diagnostics["log_decay_max"]) <= 0
    assert not any(int(value) for name, value in diagnostics.items() if name.startswith("invalid_"))
    altered = x.at[:, 6:].set(100)
    np.testing.assert_array_equal(module(altered)[:, :6], output[:, :6])
    module.gate_proj.bias[...] = jnp.full_like(module.gate_proj.bias[...], jnp.nan)
    invalid_output, invalid = module(x, return_diagnostics=True)
    assert int(invalid["invalid_gate_count"]) > 0
    assert int(invalid["invalid_output_count"]) > 0
    assert not bool(jnp.all(jnp.isfinite(invalid_output)))


@pytest.mark.parametrize("overrides", [{"gate_tau": 0}, {"gate_tau": float("nan")},
                                       {"gate_rank": 0}, {"chunk_size": 0},
                                       {"key_dim": 6}, {"value_dim": 7}])
def test_invalid_module_dimensions_and_decay_fail_explicitly(overrides):
    settings = dict(hidden_size=16, num_heads=2, key_dim=8, value_dim=12)
    with pytest.raises(ValueError):
        GatedLinearAttention(**(settings | overrides), rngs=nnx.Rngs(0))
