"""Production GLA kernel correctness across the complete d_K sweep."""
from functools import partial
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.tests.test_gla import operands, recurrence_loss, sequential_gla

GPU_AVAILABLE = importlib.util.find_spec("triton") is not None and any(
    device.platform == "gpu" for device in jax.devices()
)
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not GPU_AVAILABLE, reason="Requires GPU and Triton")]


def relative_rms(actual, expected):
    actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
    return float(jnp.sqrt(jnp.mean(jnp.square(actual - expected))) /
                 jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(expected))), 1e-12))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("key_dim,value_dim", [(2, 64), (4, 64), (8, 64), (16, 64),
                                               (32, 64), (64, 64), (128, 64), (76, 152)])
def test_fused_outputs_state_and_all_gradients_match_recurrence(key_dim, value_dim, dtype):
    from scaffolds.transformer.attention.backends.chunked_gla_triton import chunked_gla, chunked_gla_forward_intermediates

    args = operands(key_dim, seq_len=128, value_dim=value_dim, dtype=dtype)
    expected = jax.jit(sequential_gla)(*args)
    actual = jax.jit(chunked_gla)(*args)
    assert relative_rms(actual[0], expected[0]) < 1e-3
    assert relative_rms(actual[1], expected[1]) < 1e-5
    _, _, boundaries, local = jax.jit(chunked_gla_forward_intermediates)(*args)
    assert boundaries.shape == (1, 3, 2, key_dim, value_dim)
    assert local.shape == (1, 2, 2, 64, 64)
    actual_grad = jax.jit(jax.grad(partial(recurrence_loss, chunked_gla), argnums=(0, 1, 2, 3)))(*args)
    expected_grad = jax.jit(jax.grad(partial(recurrence_loss, sequential_gla), argnums=(0, 1, 2, 3)))(*args)
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        assert bool(jnp.all(jnp.isfinite(observed)))
        assert bool(jnp.any(observed != 0))
        assert relative_rms(observed, reference) < 5e-3


@pytest.mark.parametrize("log_decay", [0., -3., -100.])
def test_fused_strong_and_zero_decay_preserve_finite_outputs_and_gradients(log_decay):
    from scaffolds.transformer.attention.backends.chunked_gla_triton import chunked_gla

    q, k, v, log_alpha = operands(8, seq_len=64, value_dim=64, dtype=jnp.bfloat16)
    args = q, k, v, jnp.full_like(log_alpha, log_decay)
    actual, expected = jax.jit(chunked_gla)(*args), jax.jit(sequential_gla)(*args)
    for observed, reference in zip(actual, expected, strict=True):
        assert bool(jnp.all(jnp.isfinite(observed)))
        assert relative_rms(observed, reference) < 1e-3
    actual_grad = jax.jit(jax.grad(partial(recurrence_loss, chunked_gla), argnums=(0, 1, 2, 3)))(*args)
    expected_grad = jax.jit(jax.grad(partial(recurrence_loss, sequential_gla), argnums=(0, 1, 2, 3)))(*args)
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        assert bool(jnp.all(jnp.isfinite(observed)))
        assert relative_rms(observed, reference) < 5e-3


def test_fused_rejects_partial_outer_chunk():
    from scaffolds.transformer.attention.backends.chunked_gla_triton import chunked_gla

    with pytest.raises(ValueError, match="divisible by chunk_size"):
        chunked_gla(*operands(8, seq_len=65, value_dim=64))


def test_fused_mixed_rope_operand_dtypes_match_outputs_and_gradients():
    from scaffolds.transformer.attention.backends.chunked_gla_triton import chunked_gla
    from scaffolds.transformer.attention.positional import apply_rope

    q, k, v, log_alpha = operands(8, seq_len=128, value_dim=64, dtype=jnp.bfloat16)
    positions = jnp.arange(q.shape[1])
    args = apply_rope(q, positions), apply_rope(k, positions), v, log_alpha
    assert args[0].dtype == args[1].dtype == jnp.float32
    assert args[2].dtype == jnp.bfloat16
    actual, expected = jax.jit(chunked_gla)(*args), jax.jit(sequential_gla)(*args)
    for observed, reference in zip(actual, expected, strict=True):
        assert relative_rms(observed, reference) < 1e-5
    actual_grad = jax.jit(jax.grad(partial(recurrence_loss, chunked_gla), argnums=(0, 1, 2, 3)))(*args)
    expected_grad = jax.jit(jax.grad(partial(recurrence_loss, sequential_gla), argnums=(0, 1, 2, 3)))(*args)
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        assert bool(jnp.all(jnp.isfinite(observed)))
        assert relative_rms(observed, reference) < 5e-3
