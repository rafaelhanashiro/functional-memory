"""GPU qualification for transported-score quadratic GDN, including gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from scaffolds.transformer.attention.polynomial_gdn_reference import polynomial_gated_delta_rule_reference
from scaffolds.transformer.tests.test_polynomial_gdn import (
    assert_outputs_and_gradients, polynomial_operands, tensor_oracle, value_and_input_gradients,
)

pytestmark = pytest.mark.gpu


def _fused():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")
    from scaffolds.transformer.attention.backends.polynomial_gdn import polynomial_gated_delta_rule
    return polynomial_gated_delta_rule


def _check_against_tensor_reference(args, cotangent, *, tolerance):
    fused = _fused()
    scale = args[0].shape[-1] ** -.5
    # Both objectives receive the same output cotangent after the public BF16
    # read cast. The reference otherwise retains FP32 intermediate arithmetic.
    cotangent = cotangent.astype(args[2].dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, scale)
    with jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(polynomial_gated_delta_rule_reference,
                                             tuple(a.astype(jnp.float32) for a in args), cotangent, scale)
    assert actual[0].dtype == args[2].dtype
    for gradient, argument in zip(actual[1:], args, strict=True):
        assert gradient.dtype == argument.dtype
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)
    return actual, expected


@pytest.mark.parametrize("key_dim,length,value_dim", [
    (8, 64, 8), (64, 192, 64), (8, 2048, 4),
    # Production length/value width for both small-key experiments, including
    # the masked K=8 portion of the minimum 16-wide key tile.
    (8, 2048, 64), (16, 2048, 64),
])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_fused_outputs_and_all_input_gradients_match_tensor_recurrence(key_dim, length, value_dim, dtype, tolerance):
    _fused()
    args, cotangent = polynomial_operands(dtype=dtype, key_dim=key_dim, length=length, heads=1, value_dim=value_dim)
    _check_against_tensor_reference(args, cotangent, tolerance=tolerance)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_fused_outputs_and_gradients_match_independent_fp64_oracle(dtype, tolerance):
    fused = _fused()
    args, cotangent = polynomial_operands(dtype=dtype, key_dim=4, length=64, heads=1, value_dim=3, seed=92)
    cotangent = cotangent.astype(dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)


@pytest.mark.parametrize("erase_value", [0., 1e-6])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_zero_and_tiny_erase_keep_signed_writes_and_correct_gradients(erase_value, dtype, tolerance):
    _fused()
    args, cotangent = polynomial_operands(dtype=dtype, key_dim=16, length=128, heads=2, value_dim=8, seed=93)
    args = (*args[:5], jnp.full_like(args[5], erase_value))
    actual, _ = _check_against_tensor_reference(args, cotangent, tolerance=tolerance)
    assert np.linalg.norm(np.asarray(actual[0], np.float32)) > 0
    assert np.linalg.norm(np.asarray(actual[-1], np.float32)) > 0


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_full_length_near_complete_erase_and_strong_decay_are_finite(dtype, tolerance):
    _fused()
    args, cotangent = polynomial_operands(dtype=dtype, key_dim=8, length=2048, heads=1, value_dim=4, seed=97)
    q, k, v, g, beta, erase = args
    # The distinct normalized keys produce noncommuting, nearly singular P_t.
    # Strong decay makes unmasked future-token exponentials overflow FP32.
    args = (q, k, v, jnp.full_like(g, -2.), beta, jnp.full_like(erase, .999999))
    _check_against_tensor_reference(args, cotangent, tolerance=tolerance)


def repeated_key_oracle(args, cotangent, scale):
    """Closed-form FP64 adjoint at unit q=k, full erasure, beta_0=0, g=0.

    The quadratic prior association is erased by two copies of P, so its first
    derivatives through P vanish here. Linear prior associations still contribute
    to key and erase gradients. Beta_0 gradients include every subsequent read.
    """
    q, k, v, g, beta, erase = (np.asarray(a, np.float64) for a in args)
    c = np.asarray(cotangent, np.float64)
    np.testing.assert_array_equal(q, k)
    np.testing.assert_array_equal(k, np.broadcast_to(k[:, :1], k.shape))
    np.testing.assert_array_equal(np.sum(k * k, axis=-1), np.ones_like(g))
    np.testing.assert_array_equal(g, np.zeros_like(g))
    np.testing.assert_array_equal(erase, np.ones_like(erase))
    np.testing.assert_array_equal(beta[..., 0], np.zeros_like(g))
    b1, b2 = beta[..., 1, None], beta[..., 2, None]
    previous = np.concatenate((np.zeros_like(v[:, :1]), (b1 * v)[:, :-1]), axis=1)
    dot = lambda value: np.sum(c * value, axis=-1)
    v_dot = dot(v)
    future_c = np.flip(np.cumsum(np.flip(c, axis=1), axis=1), axis=1)
    return (
        scale * (b1 + b2) * v,
        scale * q * dot((b1 + 2 * b2) * v)[..., None],
        scale * k * dot((b1 + 2 * b2) * v - 2 * previous)[..., None],
        scale * (b1 + b2) * c,
        np.zeros_like(g),
        scale * np.stack((np.sum(future_c * v, axis=-1), v_dot, v_dot), axis=-1),
        -scale * dot(previous),
    )


@pytest.mark.parametrize("direction", ["axis", "dense"])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_repeated_keys_and_complete_erase_match_closed_form_gradients(direction, dtype, tolerance):
    fused = _fused()
    args, cotangent = polynomial_operands(dtype=dtype, key_dim=64, length=128, heads=1, value_dim=8, seed=94)
    _, k, v, g, beta, erase = args
    if direction == "axis":
        key = jnp.zeros_like(k).at[..., 0].set(1)
    else:
        key = jnp.broadcast_to(jnp.where(jnp.arange(64) % 2, -.125, .125), k.shape).astype(dtype)
    args = (key, key, v, jnp.zeros_like(g), beta.at[..., 0].set(0), jnp.ones_like(erase))
    cotangent = cotangent.astype(dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, .125)
    expected = repeated_key_oracle(args, cotangent, .125)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)
    # This derivative must cancel in FP32 even when public operands are BF16.
    unit_roundoff = np.finfo(np.float32).eps / 2
    reduction_count = 8 + 2 * 64
    gamma = reduction_count * unit_roundoff / (1 - reduction_count * unit_roundoff)
    uncancelled = .125 * np.linalg.norm(np.asarray(cotangent, np.float64) * np.asarray(v, np.float64))
    assert np.linalg.norm(np.asarray(actual[4], np.float64)) <= gamma * uncancelled + 1e-7


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_linear_limit_recovers_existing_fused_gdn_including_all_gradients(dtype, tolerance):
    fused = _fused()
    from scaffolds.transformer.attention.backends.fla_gated_delta_rule import fla_independent_gated_delta_rule

    args, cotangent = polynomial_operands(dtype=dtype, key_dim=64, length=128, heads=1, value_dim=16, seed=95)
    q, k, v, g, beta, erase = args
    args = (q, k, v, g, beta[..., 1], erase)
    cotangent = cotangent.astype(dtype).astype(jnp.float32)

    def polynomial_linear(q, k, v, g, beta, erase, scale):
        writes = jnp.stack((jnp.zeros_like(beta), beta, jnp.zeros_like(beta)), axis=-1)
        return fused(q, k, v, g, writes, erase, scale)

    actual = value_and_input_gradients(polynomial_linear, args, cotangent, .125)
    expected = value_and_input_gradients(fla_independent_gated_delta_rule, args, cotangent, .125)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)


def test_fused_constant_degree_is_causal_across_chunk_boundaries():
    fused = _fused()
    args, _ = polynomial_operands(key_dim=16, length=192, heads=1, value_dim=8, seed=96)
    args = (*args[:4], args[4].at[..., 1:].set(0), args[5])
    original = fused(*args, .25)
    changed = tuple(a.at[:, 70:].add(.1) for a in args)
    perturbed = fused(*changed, .25)
    np.testing.assert_array_equal(original[:, :70], perturbed[:, :70])
    assert not np.allclose(original[:, 70:], perturbed[:, 70:])
