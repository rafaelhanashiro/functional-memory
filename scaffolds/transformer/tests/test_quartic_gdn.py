"""Independent tensor-state contracts for quartic memory in key-space geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from scaffolds.transformer.attention.polynomial_gdn_reference import polynomial_gated_delta_rule_reference
from scaffolds.transformer.tests.test_cubic_gdn import cubic_operands, evolving_keys_oracle
from scaffolds.transformer.tests.test_polynomial_gdn import assert_outputs_and_gradients, value_and_input_gradients


def quartic_operands(**kwargs):
    args, cotangent = cubic_operands(**kwargs)
    q, k, v, g, beta, erase = args
    rng = np.random.default_rng(kwargs.get("seed", 71) + 2)
    quartic = jnp.asarray(rng.uniform(-.8, .8, size=beta.shape[:-1] + (1,)), beta.dtype)
    return (q, k, v, g, jnp.concatenate((beta, quartic), axis=-1), erase), cotangent


def quartic_tensor_oracle(q, k, v, g, beta, erase, scale):
    """Materialize every state tensor and multiply full P on every key axis.

    No rank-one update, transported-score algebra, chunks, or custom derivative
    is shared with production. FP64 callers obtain a full FP64 oracle; long GPU
    checks use FP32. Keep K <= 4, since the fourth state has K**4 V entries.
    """
    dtype = jnp.result_type(q.dtype, jnp.float32)
    q, k, v, g, beta, erase = (jnp.asarray(a, dtype) for a in (q, k, v, g, beta, erase))
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    initial = tuple(jnp.zeros((batch, heads) + (key_dim,) * degree + (value_dim,), dtype)
                    for degree in range(5))

    def step(states, inputs):
        s0, s1, s2, s3, s4 = states
        qt, kt, vt, gt, bt, et = inputs
        p = jnp.eye(key_dim, dtype=dtype) - et[..., None, None] * kt[..., :, None] * kt[..., None, :]
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
        s4 = jnp.einsum("bhia,bhajlmv->bhijlmv", p, s4)
        s4 = jnp.einsum("bhja,bhialmv->bhijlmv", p, s4)
        s4 = jnp.einsum("bhla,bhijamv->bhijlmv", p, s4)
        s4 = jnp.einsum("bhma,bhijlav->bhijlmv", p, s4)
        s4 = alpha[..., None, None, None, None, None] * s4
        s4 += jnp.einsum("bhi,bhj,bhl,bhm,bhv->bhijlmv", kt, kt, kt, kt, bt[..., 4, None] * vt)
        read = (s0 + jnp.einsum("bhi,bhiv->bhv", qt, s1)
                + jnp.einsum("bhi,bhj,bhijv->bhv", qt, qt, s2)
                + jnp.einsum("bhi,bhj,bhl,bhijlv->bhv", qt, qt, qt, s3)
                + jnp.einsum("bhi,bhj,bhl,bhm,bhijlmv->bhv", qt, qt, qt, qt, s4))
        return (s0, s1, s2, s3, s4), scale * read

    _, reads = jax.lax.scan(step, initial, tuple(jnp.moveaxis(a, 1, 0) for a in (q, k, v, g, beta, erase)))
    return jnp.moveaxis(reads, 0, 1)


@pytest.mark.parametrize("key_dim", [1, 3])
def test_quartic_reference_matches_fp64_tensor_state_and_all_gradients(key_dim):
    args, cotangent = quartic_operands(key_dim=key_dim, length=9)
    scale = key_dim ** -.5
    with jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, scale)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(quartic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), scale)
    assert_outputs_and_gradients(actual, expected, rtol=6e-6, atol=3e-6)


def test_evolving_keys_oracle_matches_quartic_tensor_state_and_all_gradients():
    args, cotangent = quartic_operands(length=9, key_dim=3, heads=1, value_dim=2)
    with enable_x64(), jax.default_matmul_precision("highest"):
        args = tuple(a.astype(jnp.float64) for a in args)
        actual = value_and_input_gradients(evolving_keys_oracle, args, cotangent.astype(jnp.float64), .5)
        expected = value_and_input_gradients(quartic_tensor_oracle, args, cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=2e-13, atol=2e-13)


def test_zero_quartic_write_recovers_cubic_outputs_and_common_gradients():
    args, cotangent = cubic_operands(length=11, key_dim=3)

    def zero_quartic(q, k, v, g, beta, erase, scale):
        return polynomial_gated_delta_rule_reference(q, k, v, g, jnp.pad(beta, ((0, 0),) * 3 + ((0, 1),)), erase, scale)

    with jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(zero_quartic, args, cotangent, .5)
        expected = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, .5)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert_outputs_and_gradients(actual, expected, rtol=3e-7, atol=1e-7)
    q, k, v, g, beta, erase = args
    writes = jnp.pad(beta, ((0, 0),) * 3 + ((0, 1),))
    gradient = jax.grad(lambda b: jnp.sum(polynomial_gated_delta_rule_reference(q, k, v, g, b, erase, .5)
                                         * cotangent))(writes)
    assert float(jnp.linalg.norm(gradient[..., 4])) > .01


@pytest.mark.parametrize("decay", [False, True])
def test_zero_erase_recovers_causal_signed_quartic_attention(decay):
    args, _ = quartic_operands(key_dim=3)
    q, k, v, g, beta, erase = args
    g = g if decay else jnp.zeros_like(g)
    actual = polynomial_gated_delta_rule_reference(q, k, v, g, beta, jnp.zeros_like(erase), .5)
    scores = jnp.einsum("bthk,bshk->btsh", q, k)
    cumulative = jnp.cumsum(g, axis=1)
    decay_weights = jnp.exp(cumulative[:, :, None, :] - cumulative[:, None, :, :])
    polynomial = sum(beta[:, None, :, :, degree] * scores ** degree for degree in range(5))
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    weights = jnp.where(causal[None, :, :, None], decay_weights * polynomial, 0)
    expected = .5 * jnp.einsum("btsh,bshv->bthv", weights, v)
    np.testing.assert_allclose(actual, expected, rtol=6e-6, atol=3e-6)


def test_quartic_state_uses_one_decay_and_even_score_power():
    q = -jnp.ones((1, 2, 1, 1))
    k = jnp.ones_like(q)
    v = jnp.array([2., 0.]).reshape(1, 2, 1, 1)
    g = jnp.log(jnp.array([1., .25])).reshape(1, 2, 1)
    beta = jnp.zeros((1, 2, 1, 5)).at[..., 4].set(-1)
    actual = polynomial_gated_delta_rule_reference(q, k, v, g, beta, jnp.zeros_like(g), 1.)
    np.testing.assert_allclose(actual.reshape(-1), [-2., -.5], rtol=1e-6, atol=1e-6)


def test_quartic_noncommuting_transitions_have_correct_time_order():
    args, _ = quartic_operands(length=7, heads=1, key_dim=2)
    args = (*args[:4], args[4].at[..., :4].set(0), args[5])
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
                output[0, t, 0] += np.exp(np.sum(g[0, i + 1:t + 1, 0])) * beta[0, i, 0, 4] * score ** 4 * v[0, i, 0]
    actual = polynomial_gated_delta_rule_reference(*args, 1.)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=2e-6)
    assert np.linalg.norm(wrong_order - expected) > .005


@pytest.mark.parametrize("erase_value", [0., 1.])
def test_quartic_hard_gate_and_repeated_keys_gradients(erase_value):
    args, cotangent = quartic_operands(length=11, heads=1, key_dim=4, value_dim=2)
    _, k, v, g, beta, erase = args
    key = jnp.full_like(k, .5)
    args = (key, key, v, jnp.zeros_like(g), beta, jnp.full_like(erase, erase_value))
    with jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(quartic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=7e-6, atol=3e-6)


def test_quartic_closed_form_hard_erase_oracle_matches_tensor_gradients():
    from scaffolds.transformer.tests.test_quartic_gdn_kernel import quartic_repeated_key_oracle

    args, cotangent = quartic_operands(length=9, heads=1, key_dim=4, value_dim=2)
    _, k, v, g, beta, erase = args
    key = jnp.full_like(k, .5)
    args = (key, key, v, jnp.zeros_like(g), beta.at[..., 0].set(0), jnp.ones_like(erase))
    actual = quartic_repeated_key_oracle(args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(quartic_tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=1e-12, atol=1e-12)


def test_quartic_recurrence_is_causal():
    args, _ = quartic_operands(length=13, key_dim=3)
    original = polynomial_gated_delta_rule_reference(*args, .5)
    perturbed = polynomial_gated_delta_rule_reference(*(a.at[:, 8:].add(.25) for a in args), .5)
    np.testing.assert_array_equal(original[:, :8], perturbed[:, :8])
    assert not np.allclose(original[:, 8:], perturbed[:, 8:])
