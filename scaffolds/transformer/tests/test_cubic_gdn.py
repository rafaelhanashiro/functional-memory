"""Independent mathematical contracts for cubic memory with key-space erasure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from scaffolds.transformer.attention.polynomial_gdn_reference import polynomial_gated_delta_rule_reference
from scaffolds.transformer.tests.test_polynomial_gdn import (
    assert_outputs_and_gradients, polynomial_operands, value_and_input_gradients,
)


def cubic_operands(**kwargs):
    args, cotangent = polynomial_operands(**kwargs)
    q, k, v, g, beta, erase = args
    rng = np.random.default_rng(kwargs.get("seed", 71) + 1)
    cubic = jnp.asarray(rng.uniform(-.8, .8, size=beta.shape[:-1] + (1,)), beta.dtype)
    return (q, k, v, g, jnp.concatenate((beta, cubic), axis=-1), erase), cotangent


def cubic_tensor_oracle(q, k, v, g, beta, erase, scale):
    """FP64 full-matrix transport on every tensor axis, with no low-rank trick.

    This independent oracle materializes the genuine cubic state. Use small key
    dimensions and call inside ``enable_x64()``. It shares no state-update or
    chunk algebra with either the FP32 reference or the production backend.
    """
    q, k, v, g, beta, erase = (jnp.asarray(a, jnp.float64) for a in (q, k, v, g, beta, erase))
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    initial = tuple(jnp.zeros((batch, heads) + (key_dim,) * degree + (value_dim,), jnp.float64)
                    for degree in range(4))

    def step(states, inputs):
        s0, s1, s2, s3 = states
        qt, kt, vt, gt, bt, et = inputs
        p = jnp.eye(key_dim, dtype=jnp.float64) - et[..., None, None] * kt[..., :, None] * kt[..., None, :]
        alpha = jnp.exp(gt)
        s0 = alpha[..., None] * s0 + bt[..., 0, None] * vt
        s1 = alpha[..., None, None] * (p @ s1) + jnp.einsum("bhi,bhv->bhiv", kt, bt[..., 1, None] * vt)
        s2 = jnp.einsum("bhia,bhajv->bhijv", p, s2)
        s2 = jnp.einsum("bhja,bhiav->bhijv", p, s2)
        s2 = alpha[..., None, None, None] * s2 + jnp.einsum("bhi,bhj,bhv->bhijv", kt, kt, bt[..., 2, None] * vt)
        s3 = jnp.einsum("bhia,bhajlv->bhijlv", p, s3)
        s3 = jnp.einsum("bhja,bhialv->bhijlv", p, s3)
        s3 = jnp.einsum("bhla,bhijav->bhijlv", p, s3)
        s3 = alpha[..., None, None, None, None] * s3
        s3 += jnp.einsum("bhi,bhj,bhl,bhv->bhijlv", kt, kt, kt, bt[..., 3, None] * vt)
        read = (s0 + jnp.einsum("bhi,bhiv->bhv", qt, s1)
                + jnp.einsum("bhi,bhj,bhijv->bhv", qt, qt, s2)
                + jnp.einsum("bhi,bhj,bhl,bhijlv->bhv", qt, qt, qt, s3))
        return (s0, s1, s2, s3), scale * read

    _, reads = jax.lax.scan(step, initial, tuple(jnp.moveaxis(a, 1, 0) for a in (q, k, v, g, beta, erase)))
    return jnp.moveaxis(reads, 0, 1)


def evolving_keys_oracle(q, k, v, g, beta, erase, scale):
    """Independent unrolling: evolve every previous source key with full P_t.

    Unlike production, this walks forward in time, with no query transport,
    chunks, inverse matrices, or custom derivatives. The stored sources are O(TK)
    instead of O(K**3 V), so all-input AD at production K=64 remains practical
    for a few chunks. Small cases verify this oracle against the FP64 tensor one.
    """
    dtype = jnp.result_type(q.dtype, jnp.float32)
    q, k, v, g, beta, erase = (jnp.asarray(a, dtype) for a in (q, k, v, g, beta, erase))
    batch, length, heads, key_dim = q.shape
    values = v.transpose(0, 2, 1, 3)
    writes = beta.transpose(0, 2, 1, 3)
    initial = (jnp.zeros((batch, heads, length, key_dim), dtype),
               jnp.zeros((batch, heads, length), dtype))

    def step(state, inputs):
        keys, weights = state
        index, qt, kt, gt, et = inputs
        p = jnp.eye(key_dim, dtype=dtype) - et[..., None, None] * kt[..., :, None] * kt[..., None, :]
        keys = jnp.einsum("bhij,bhkj->bhik", keys, p)
        keys = keys.at[:, :, index].set(kt)
        weights = (weights * jnp.exp(gt)[..., None]).at[:, :, index].set(1)
        scores = jnp.einsum("bhk,bhik->bhi", qt, keys)
        polynomial = sum(writes[..., degree] * scores ** degree for degree in range(writes.shape[-1]))
        read = scale * jnp.einsum("bhi,bhiv->bhv", weights * polynomial, values)
        return (keys, weights), read

    inputs = (jnp.arange(length),) + tuple(jnp.moveaxis(a, 1, 0) for a in (q, k, g, erase))
    _, reads = jax.lax.scan(step, initial, inputs)
    return jnp.moveaxis(reads, 0, 1)


@pytest.mark.parametrize("key_dim", [1, 3])
def test_cubic_reference_matches_independent_fp64_tensor_state_and_all_gradients(key_dim):
    args, cotangent = cubic_operands(key_dim=key_dim, length=9)
    scale = key_dim ** -.5
    with jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, scale)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(cubic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), scale)
    assert_outputs_and_gradients(actual, expected, rtol=5e-6, atol=3e-6)


def test_evolving_keys_oracle_matches_cubic_tensor_state_and_all_gradients():
    args, cotangent = cubic_operands(length=9, key_dim=3, heads=1, value_dim=2)
    with enable_x64(), jax.default_matmul_precision("highest"):
        double_args = tuple(a.astype(jnp.float64) for a in args)
        actual = value_and_input_gradients(evolving_keys_oracle, double_args, cotangent.astype(jnp.float64), .5)
        expected = value_and_input_gradients(cubic_tensor_oracle, double_args, cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=2e-13, atol=2e-13)


def test_zero_cubic_write_recovers_quadratic_outputs_and_common_gradients():
    args, cotangent = polynomial_operands(length=11, key_dim=3)

    def zero_cubic(q, k, v, g, beta, erase, scale):
        return polynomial_gated_delta_rule_reference(q, k, v, g, jnp.pad(beta, ((0, 0),) * 3 + ((0, 1),)), erase, scale)

    with jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(zero_cubic, args, cotangent, .5)
        expected = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, .5)
    np.testing.assert_array_equal(actual[0], expected[0])
    # Extra zero-valued AD contributions may change FP32 addition association
    # (dv differs by one ulp on CPU), while the forward remains bit-identical.
    assert_outputs_and_gradients(actual, expected, rtol=3e-7, atol=1e-7)
    q, k, v, g, beta, erase = args
    writes = jnp.pad(beta, ((0, 0),) * 3 + ((0, 1),))
    gradient = jax.grad(lambda b: jnp.sum(polynomial_gated_delta_rule_reference(q, k, v, g, b, erase, .5)
                                         * cotangent))(writes)
    assert float(jnp.linalg.norm(gradient[..., 3])) > .01


@pytest.mark.parametrize("decay", [False, True])
def test_zero_erase_recovers_causal_signed_cubic_attention(decay):
    args, _ = cubic_operands(key_dim=3)
    q, k, v, g, beta, erase = args
    g = g if decay else jnp.zeros_like(g)
    actual = polynomial_gated_delta_rule_reference(q, k, v, g, beta, jnp.zeros_like(erase), .5)
    scores = jnp.einsum("bthk,bshk->btsh", q, k)
    cumulative = jnp.cumsum(g, axis=1)
    decay_weights = jnp.exp(cumulative[:, :, None, :] - cumulative[:, None, :, :])
    polynomial = sum(beta[:, None, :, :, degree] * scores ** degree for degree in range(4))
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    weights = jnp.where(causal[None, :, :, None], decay_weights * polynomial, 0)
    expected = .5 * jnp.einsum("btsh,bshv->bthv", weights, v)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=3e-6)


def test_cubic_state_uses_one_decay_and_preserves_odd_score_sign():
    q = -jnp.ones((1, 2, 1, 1))
    k = jnp.ones_like(q)
    v = jnp.array([2., 0.]).reshape(1, 2, 1, 1)
    g = jnp.log(jnp.array([1., .25])).reshape(1, 2, 1)
    beta = jnp.zeros((1, 2, 1, 4)).at[..., 3].set(1)
    actual = polynomial_gated_delta_rule_reference(q, k, v, g, beta, jnp.zeros_like(g), 1.)
    np.testing.assert_allclose(actual.reshape(-1), [-2., -.5], rtol=1e-6, atol=1e-6)


def test_cubic_noncommuting_transitions_have_correct_time_order():
    args, _ = cubic_operands(length=7, heads=1, key_dim=2)
    args = (*args[:4], args[4].at[..., :3].set(0), args[5])
    q, k, v, g, beta, erase = (np.asarray(a, np.float64) for a in args)
    expected, wrong_order = np.zeros_like(v), np.zeros_like(v)
    for t in range(q.shape[1]):
        for i in range(t + 1):
            for output, time_order in ((expected, range(i + 1, t + 1)), (wrong_order, range(t, i, -1))):
                key = k[0, i, 0].copy()
                for j in time_order:
                    p = np.eye(2) - erase[0, j, 0] * np.outer(k[0, j, 0], k[0, j, 0])
                    key = p @ key
                score = q[0, t, 0] @ key
                output[0, t, 0] += np.exp(np.sum(g[0, i + 1:t + 1, 0])) * beta[0, i, 0, 3] * score ** 3 * v[0, i, 0]
    actual = polynomial_gated_delta_rule_reference(*args, 1.)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-6)
    assert np.linalg.norm(wrong_order - expected) > .005


@pytest.mark.parametrize("erase_value", [0., 1e-6, 1.])
def test_cubic_hard_gate_and_repeated_keys_gradients(erase_value):
    args, cotangent = cubic_operands(length=11, heads=1, key_dim=4, value_dim=2)
    _, k, v, g, beta, erase = args
    key = jnp.full_like(k, .5)
    args = (key, key, v, jnp.zeros_like(g), beta, jnp.full_like(erase, erase_value))
    with jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(cubic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=6e-6, atol=3e-6)


def test_cubic_closed_form_hard_erase_oracle_matches_full_tensor_gradients():
    from scaffolds.transformer.tests.test_cubic_gdn_kernel import cubic_repeated_key_oracle

    args, cotangent = cubic_operands(length=9, heads=1, key_dim=4, value_dim=2)
    _, k, v, g, beta, erase = args
    key = jnp.full_like(k, .5)
    args = (key, key, v, jnp.zeros_like(g), beta.at[..., 0].set(0), jnp.ones_like(erase))
    actual = cubic_repeated_key_oracle(args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(cubic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=1e-12, atol=1e-12)


def test_cubic_recurrence_is_causal():
    args, _ = cubic_operands(length=13, key_dim=3)
    original = polynomial_gated_delta_rule_reference(*args, .5)
    perturbed = polynomial_gated_delta_rule_reference(*(a.at[:, 8:].add(.25) for a in args), .5)
    np.testing.assert_array_equal(original[:, :8], perturbed[:, :8])
    assert not np.allclose(original[:, 8:], perturbed[:, 8:])
