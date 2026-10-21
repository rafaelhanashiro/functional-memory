"""GPU qualification for cubic GDN against independent state-based oracles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from scaffolds.transformer.attention.polynomial_gdn_reference import polynomial_gated_delta_rule_reference
from scaffolds.transformer.tests.test_cubic_gdn import cubic_operands, cubic_tensor_oracle, evolving_keys_oracle
from scaffolds.transformer.tests.test_polynomial_gdn import assert_outputs_and_gradients, value_and_input_gradients
from scaffolds.transformer.tests.test_polynomial_gdn_kernel import _fused

pytestmark = pytest.mark.gpu


def check_cubic(args, cotangent, *, oracle=polynomial_gated_delta_rule_reference, tolerance):
    fused = _fused()
    scale = args[0].shape[-1] ** -.5
    # Match the public BF16 output cast's adjoint before comparing with an
    # FP32/FP64 oracle that does not round its internal state to BF16.
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
def test_cubic_gpu_matches_fp64_tensor_state_and_all_input_gradients(dtype, tolerance):
    fused = _fused()
    args, cotangent = cubic_operands(dtype=dtype, key_dim=4, length=64, heads=1, value_dim=3, seed=191)
    cotangent = cotangent.astype(dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(cubic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)


@pytest.mark.parametrize("key_dim", [8, 16, 64])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_cubic_production_width_multichunk_matches_independent_source_evolution(key_dim, dtype, tolerance):
    _fused()
    # K=8 exercises the masked portion of the minimum 16-wide key tile.
    args, cotangent = cubic_operands(dtype=dtype, key_dim=key_dim, length=192, heads=1, value_dim=64, seed=192)
    check_cubic(args, cotangent, oracle=evolving_keys_oracle, tolerance=tolerance)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_cubic_full_length_strong_decay_near_complete_erase_matches_tensor_state(dtype, tolerance):
    _fused()
    args, cotangent = cubic_operands(dtype=dtype, key_dim=4, length=2048, heads=1, value_dim=4, seed=193)
    q, k, v, g, beta, erase = args
    # Noncommuting nearly singular transitions, with future decay exponentials
    # large enough to overflow unless causality is handled before exp.
    args = (q, k, v, jnp.full_like(g, -2.), beta, jnp.full_like(erase, .999999))
    check_cubic(args, cotangent, tolerance=tolerance)


@pytest.mark.parametrize("key_dim", [8, 16, 64])
def test_cubic_full_production_width_and_length_bf16_matches_embedded_tensor_state(key_dim):
    _fused()
    args, cotangent = cubic_operands(dtype=jnp.bfloat16, key_dim=4, length=2048, heads=1, value_dim=64, seed=194)
    q, k, v, g, beta, erase = args
    # An invariant coordinate subspace gives an exact small tensor oracle for
    # the actual K=8/16/64,V=64,T=2048 kernel; gradients outside the subspace
    # are zero. Dense keys at each width are covered by the multi-chunk test.
    q, k = (jnp.pad(a, ((0, 0), (0, 0), (0, 0), (0, key_dim - 4))) for a in (q, k))
    args = (q, k, v, g, beta, erase)

    def embedded_state(q, k, v, g, beta, erase, scale):
        return polynomial_gated_delta_rule_reference(q[..., :4], k[..., :4], v, g, beta, erase, scale)

    check_cubic(args, cotangent, oracle=embedded_state, tolerance=1e-2)


@pytest.mark.parametrize("erase_value", [0., 1e-6])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_cubic_zero_and_tiny_erase_have_correct_signed_write_gradients(erase_value, dtype, tolerance):
    _fused()
    args, cotangent = cubic_operands(dtype=dtype, key_dim=4, length=128, heads=2, value_dim=8, seed=195)
    args = (*args[:5], jnp.full_like(args[5], erase_value))
    actual, _ = check_cubic(args, cotangent, tolerance=tolerance)
    assert np.linalg.norm(np.asarray(actual[5][..., 3], np.float32)) > .01
    assert np.linalg.norm(np.asarray(actual[-1], np.float32)) > .01


def cubic_repeated_key_oracle(args, cotangent, scale):
    """Closed-form adjoint at unit repeated q=k, full erase, beta0=g=0.

    Two or more erased tensor factors make the derivatives of prior higher
    degree writes vanish. A prior linear write still affects dk and d(erase).
    The current degree-r write contributes r*beta_r to score derivatives.
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
    derivative = np.sum(beta[..., 1:] * np.arange(1, 4), axis=-1, keepdims=True)
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
        scale * np.stack((np.sum(future_c * v, axis=-1), v_dot, v_dot, v_dot), axis=-1),
        -scale * dot(previous),
    )


@pytest.mark.parametrize("direction", ["axis", "dense"])
@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_cubic_repeated_keys_complete_erase_match_closed_form_all_gradients(direction, dtype, tolerance):
    fused = _fused()
    args, cotangent = cubic_operands(dtype=dtype, key_dim=64, length=128, heads=1, value_dim=8, seed=196)
    _, k, v, g, beta, erase = args
    if direction == "axis":
        key = jnp.zeros_like(k).at[..., 0].set(1)
    else:
        key = jnp.broadcast_to(jnp.where(jnp.arange(64) % 2, -.125, .125), k.shape).astype(dtype)
    args = (key, key, v, jnp.zeros_like(g), beta.at[..., 0].set(0), jnp.ones_like(erase))
    cotangent = cotangent.astype(dtype).astype(jnp.float32)
    actual = value_and_input_gradients(fused, args, cotangent, .125)
    expected = cubic_repeated_key_oracle(args, cotangent, .125)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)
    # The mathematically zero decay derivative must cancel in FP32 before
    # cotangents are cast back to BF16, using the same reduction error budget
    # as the existing quadratic hard-erase qualification.
    unit_roundoff = np.finfo(np.float32).eps / 2
    reduction_count = 8 + 2 * 64
    gamma = reduction_count * unit_roundoff / (1 - reduction_count * unit_roundoff)
    uncancelled = .125 * np.linalg.norm(np.asarray(cotangent, np.float64) * np.asarray(v, np.float64))
    assert np.linalg.norm(np.asarray(actual[4], np.float64)) <= gamma * uncancelled + 1e-7


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_cubic_zero_write_limit_matches_quadratic_kernel_common_gradients(dtype, tolerance):
    fused = _fused()
    args, cotangent = cubic_operands(dtype=dtype, key_dim=64, length=128, heads=1, value_dim=16, seed=197)
    args = (*args[:4], args[4][..., :3], args[5])
    cotangent = cotangent.astype(dtype).astype(jnp.float32)

    def zero_cubic(q, k, v, g, beta, erase, scale):
        return fused(q, k, v, g, jnp.pad(beta, ((0, 0),) * 3 + ((0, 1),)), erase, scale)

    actual = value_and_input_gradients(zero_cubic, args, cotangent, .125)
    expected = value_and_input_gradients(fused, args, cotangent, .125)
    assert_outputs_and_gradients(actual, expected, rtol=tolerance, atol=3e-5)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 3e-4), (jnp.bfloat16, 1e-2)])
def test_zero_cubic_write_initialization_still_gets_correct_nonzero_cubic_gradient(dtype, tolerance):
    _fused()
    args, cotangent = cubic_operands(dtype=dtype, key_dim=4, length=64, heads=1, value_dim=8, seed=198)
    args = (*args[:4], args[4].at[..., 3].set(0), args[5])
    actual, _ = check_cubic(args, cotangent, tolerance=tolerance)
    assert np.linalg.norm(np.asarray(actual[5][..., 3], np.float32)) > .01


def test_cubic_kernel_is_causal_across_chunk_boundaries():
    fused = _fused()
    args, _ = cubic_operands(key_dim=16, length=192, heads=1, value_dim=8, seed=199)
    original = fused(*args, .25)
    changed = tuple(a.at[:, 70:].add(.1) for a in args)
    perturbed = fused(*changed, .25)
    np.testing.assert_array_equal(original[:, :70], perturbed[:, :70])
    assert not np.allclose(original[:, 70:], perturbed[:, 70:])
