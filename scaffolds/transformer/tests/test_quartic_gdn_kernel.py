"""Quartic GPU qualification against independent tensor and source evolution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from scaffolds.transformer.tests.test_cubic_gdn import evolving_keys_oracle
from scaffolds.transformer.tests.test_polynomial_gdn import assert_outputs_and_gradients, value_and_input_gradients
from scaffolds.transformer.tests.test_polynomial_gdn_kernel import _fused
from scaffolds.transformer.tests.test_quartic_gdn import quartic_operands, quartic_tensor_oracle

pytestmark = pytest.mark.gpu


def check_quartic(args, cotangent, *, oracle=quartic_tensor_oracle, tolerance):
    fused = _fused()
    scale = args[0].shape[-1] ** -.5
    # The public BF16 output cast rounds its adjoint too; compare the same
    # cotangent against an oracle retaining FP32 state and intermediate values.
    cotangent = cotangent.astype(args[2].dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, scale)
    with jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(oracle, tuple(a.astype(jnp.float32) for a in args), cotangent, scale)
    assert actual[0].dtype == args[2].dtype
    for gradient, argument in zip(actual[1:], args, strict=True):
        assert gradient.dtype == argument.dtype
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)
    return actual, expected


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_quartic_gpu_matches_fp64_tensor_state_and_all_input_gradients(dtype, tolerance):
    fused = _fused()
    args, cotangent = quartic_operands(dtype=dtype, key_dim=4, length=64, heads=1, value_dim=3, seed=291)
    cotangent = cotangent.astype(dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(quartic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)


@pytest.mark.parametrize("key_dim", [8, 16])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_quartic_production_width_multichunk_matches_independent_source_evolution(key_dim, dtype, tolerance):
    _fused()
    # Dense normalized keys; K=8 also exercises padding of the 16-wide key tile.
    args, cotangent = quartic_operands(dtype=dtype, key_dim=key_dim, length=192, heads=1, value_dim=64, seed=292)
    check_quartic(args, cotangent, oracle=evolving_keys_oracle, tolerance=tolerance)


@pytest.mark.parametrize("key_dim", [8, 16])
@pytest.mark.parametrize("strong_decay", [False, True])
def test_quartic_full_length_bf16_matches_embedded_tensor_state(key_dim, strong_decay):
    _fused()
    args, cotangent = quartic_operands(dtype=jnp.bfloat16, key_dim=4, length=2048, heads=1, value_dim=64, seed=293)
    q, k, v, g, beta, erase = args
    # A four-dimensional invariant subspace supplies an exact manageable tensor
    # oracle at the production K,V,T. Outside-subspace gradients must be zero.
    q, k = (jnp.pad(a, ((0, 0), (0, 0), (0, 0), (0, key_dim - 4))) for a in (q, k))
    if strong_decay:
        # Unmasked future decay would overflow; P remains noncommuting and is
        # nearly singular (fully erased after BF16 rounding of this lambda).
        g, erase = jnp.full_like(g, -2.), jnp.full_like(erase, .999999)
    args = (q, k, v, g, beta, erase)

    def embedded_state(q, k, v, g, beta, erase, scale):
        return quartic_tensor_oracle(q[..., :4], k[..., :4], v, g, beta, erase, scale)

    check_quartic(args, cotangent, oracle=embedded_state, tolerance=1e-2)


@pytest.mark.parametrize("erase_value", [0., 1e-6])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_quartic_zero_and_tiny_erase_have_correct_signed_write_gradients(erase_value, dtype, tolerance):
    _fused()
    args, cotangent = quartic_operands(dtype=dtype, key_dim=4, length=128, heads=2, value_dim=8, seed=294)
    args = (*args[:5], jnp.full_like(args[5], erase_value))
    actual, _ = check_quartic(args, cotangent, tolerance=tolerance)
    assert np.linalg.norm(np.asarray(actual[5][..., 4], np.float32)) > .01
    assert np.linalg.norm(np.asarray(actual[-1], np.float32)) > .01


def quartic_repeated_key_oracle(args, cotangent, scale):
    """Closed-form adjoint at unit repeated q=k, full erase, beta0=g=0.

    Prior degree >= 2 writes have at least two erased factors, hence zero first
    derivatives. Prior linear writes survive in dk and derase. The current
    degree-r write contributes r*beta_r to the score derivative.
    """
    q, k, v, g, beta, erase = (np.asarray(a, np.float64) for a in args)
    c = np.asarray(cotangent, np.float64)
    np.testing.assert_array_equal(q, k)
    np.testing.assert_array_equal(k, np.broadcast_to(k[:, :1], k.shape))
    np.testing.assert_array_equal(np.sum(k * k, axis=-1), np.ones_like(g))
    np.testing.assert_array_equal(g, np.zeros_like(g))
    np.testing.assert_array_equal(erase, np.ones_like(erase))
    np.testing.assert_array_equal(beta[..., 0], np.zeros_like(g))
    weight = np.sum(beta[..., 1:], axis=-1, keepdims=True)
    derivative = np.sum(beta[..., 1:] * np.arange(1, 5), axis=-1, keepdims=True)
    linear = beta[..., 1, None] * v
    previous = np.concatenate((np.zeros_like(v[:, :1]), linear[:, :-1]), axis=1)
    dot = lambda value: np.sum(c * value, axis=-1)
    v_dot = dot(v)
    future_c = np.flip(np.cumsum(np.flip(c, axis=1), axis=1), axis=1)
    return (
        scale * weight * v,
        scale * q * dot(derivative * v)[..., None],
        scale * k * dot(derivative * v - 2 * previous)[..., None],
        scale * weight * c,
        np.zeros_like(g),
        scale * np.stack((np.sum(future_c * v, axis=-1), v_dot, v_dot, v_dot, v_dot), axis=-1),
        -scale * dot(previous),
    )


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_quartic_repeated_keys_complete_erase_match_closed_form_all_gradients(dtype, tolerance):
    fused = _fused()
    args, cotangent = quartic_operands(dtype=dtype, key_dim=16, length=128, heads=1, value_dim=8, seed=295)
    _, k, v, g, beta, erase = args
    key = jnp.broadcast_to(jnp.where(jnp.arange(16) % 2, -.25, .25), k.shape).astype(dtype)
    args = (key, key, v, jnp.zeros_like(g), beta.at[..., 0].set(0), jnp.ones_like(erase))
    cotangent = cotangent.astype(dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, .25)
    expected = quartic_repeated_key_oracle(args, cotangent, .25)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)
    # A truly zero decay derivative must cancel before the BF16 adjoint cast.
    unit_roundoff = np.finfo(np.float32).eps / 2
    reduction_count = 8 + 2 * 64
    gamma = reduction_count * unit_roundoff / (1 - reduction_count * unit_roundoff)
    uncancelled = .25 * np.linalg.norm(np.asarray(cotangent, np.float64) * np.asarray(v, np.float64))
    assert np.linalg.norm(np.asarray(actual[4], np.float64)) <= gamma * uncancelled + 1e-7


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_quartic_zero_write_limit_matches_cubic_kernel_common_gradients(dtype, tolerance):
    fused = _fused()
    args, cotangent = quartic_operands(dtype=dtype, key_dim=16, length=128, heads=1, value_dim=16, seed=296)
    args = (*args[:4], args[4][..., :4], args[5])
    cotangent = cotangent.astype(dtype).astype(jnp.float32)

    def zero_quartic(q, k, v, g, beta, erase, scale):
        return fused(q, k, v, g, jnp.pad(beta, ((0, 0),) * 3 + ((0, 1),)), erase, scale)

    actual = value_and_input_gradients(zero_quartic, args, cotangent, .25)
    expected = value_and_input_gradients(fused, args, cotangent, .25)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_zero_quartic_write_initialization_gets_correct_nonzero_quartic_gradient(dtype, tolerance):
    _fused()
    args, cotangent = quartic_operands(dtype=dtype, key_dim=4, length=64, heads=1, value_dim=8, seed=297)
    args = (*args[:4], args[4].at[..., 4].set(0), args[5])
    actual, _ = check_quartic(args, cotangent, tolerance=tolerance)
    assert np.linalg.norm(np.asarray(actual[5][..., 4], np.float32)) > .01


def test_quartic_kernel_is_causal_across_chunk_boundaries():
    fused = _fused()
    args, _ = quartic_operands(key_dim=8, length=192, heads=1, value_dim=8, seed=298)
    original = fused(*args, .25)
    perturbed = fused(*(a.at[:, 70:].add(.1) for a in args), .25)
    np.testing.assert_array_equal(original[:, :70], perturbed[:, :70])
    assert not np.allclose(original[:, 70:], perturbed[:, 70:])
