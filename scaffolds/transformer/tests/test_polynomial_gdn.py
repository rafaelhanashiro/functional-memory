"""Mathematical contracts for the original-key-space quadratic GDN recurrence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from scaffolds.transformer.attention.gated_deltanet import gated_delta_rule_reference
from scaffolds.transformer.attention.polynomial_gdn_reference import polynomial_gated_delta_rule_reference


def polynomial_operands(*, dtype=jnp.float32, batch=1, length=11, heads=2, key_dim=4, value_dim=3, seed=71):
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(batch, length, heads, key_dim)) for _ in range(2))
    q, k = (a / np.linalg.norm(a, axis=-1, keepdims=True) for a in (q, k))
    v = rng.normal(size=(batch, length, heads, value_dim))
    g = -rng.uniform(.002, .4, size=(batch, length, heads))
    beta = rng.uniform(-.8, .8, size=(batch, length, heads, 3))
    erase = rng.uniform(.02, .98, size=g.shape)
    arrays = (q, k, v, g, beta, erase)
    args = tuple(jnp.asarray(a, jnp.float32 if i == 3 else dtype) for i, a in enumerate(arrays))
    return args, jnp.asarray(rng.normal(size=v.shape), jnp.float32)


def tensor_oracle(q, k, v, g, beta, erase, scale):
    """Independent FP64 oracle using full P matrices on both tensor indices.

    Call inside ``enable_x64()``. No low-rank update, triangular solve, chunking,
    or production/reference implementation is shared with this oracle.
    """
    q, k, v, g, beta, erase = (jnp.asarray(a, jnp.float64) for a in (q, k, v, g, beta, erase))
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    initial = (jnp.zeros((batch, heads, value_dim), jnp.float64),
               jnp.zeros((batch, heads, key_dim, value_dim), jnp.float64),
               jnp.zeros((batch, heads, key_dim, key_dim, value_dim), jnp.float64))

    def step(states, inputs):
        s0, s1, s2 = states
        qt, kt, vt, gt, bt, et = inputs
        p = jnp.eye(key_dim, dtype=jnp.float64) - et[..., None, None] * kt[..., :, None] * kt[..., None, :]
        alpha = jnp.exp(gt)
        s0 = alpha[..., None] * s0 + bt[..., 0, None] * vt
        s1 = alpha[..., None, None] * (p @ s1) + kt[..., :, None] * (bt[..., 1, None] * vt)[..., None, :]
        s2 = jnp.einsum("bhij,bhjkv->bhikv", p, s2)
        s2 = alpha[..., None, None, None] * jnp.einsum("bhikv,bhkj->bhijv", s2, p)
        s2 += kt[..., :, None, None] * kt[..., None, :, None] * (bt[..., 2, None] * vt)[..., None, None, :]
        read = s0 + jnp.einsum("bhk,bhkv->bhv", qt, s1) + jnp.einsum("bhi,bhijv,bhj->bhv", qt, s2, qt)
        return (s0, s1, s2), read * scale

    _, read = jax.lax.scan(step, initial, tuple(jnp.moveaxis(a, 1, 0) for a in (q, k, v, g, beta, erase)))
    return jnp.moveaxis(read, 0, 1)


def value_and_input_gradients(function, args, cotangent, scale):
    output = function(*args, scale)
    gradients = jax.grad(lambda *a: jnp.sum(function(*a, scale) * cotangent), argnums=range(6))(*args)
    return (output, *gradients)


def assert_outputs_and_gradients(actual, expected, *, rtol, atol):
    for name, a, e in zip(("output", "dq", "dk", "dv", "dg", "dbeta", "derase"), actual, expected, strict=True):
        a, e = np.asarray(a, np.float64), np.asarray(e, np.float64)
        assert np.all(np.isfinite(a)), name
        error = np.linalg.norm(a - e)
        assert error <= rtol * np.linalg.norm(e) + atol, (name, error, np.linalg.norm(e))


@pytest.mark.parametrize("key_dim", [1, 4])
def test_reference_outputs_and_all_gradients_match_fp64_tensor_oracle(key_dim):
    args, cotangent = polynomial_operands(key_dim=key_dim)
    scale = key_dim ** -.5
    with jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, scale)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), scale)
    assert_outputs_and_gradients(actual, expected, rtol=3e-6, atol=2e-6)


def test_linear_limit_matches_independent_gdn_outputs_and_gradient_chain():
    args, cotangent = polynomial_operands()
    q, k, v, g, beta, erase = args
    write = beta[..., 1]
    linear = lambda q, k, v, g, b, e: gated_delta_rule_reference(q, k, v, g, b, .5, erase=e)
    quadratic = lambda q, k, v, g, b, e: polynomial_gated_delta_rule_reference(
        q, k, v, g, jnp.stack((jnp.zeros_like(b), b, jnp.zeros_like(b)), axis=-1), e, .5)
    linear_args = (q, k, v, g, write, erase)
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(quadratic(*linear_args), linear(*linear_args), rtol=2e-6, atol=2e-6)
        gradients = lambda f: jax.grad(lambda *a: jnp.sum(f(*a) * cotangent), argnums=range(6))(*linear_args)
        for actual, expected in zip(gradients(quadratic), gradients(linear), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("decay", [False, True])
def test_zero_erase_recovers_causal_polynomial_attention(decay):
    args, _ = polynomial_operands()
    q, k, v, g, beta, erase = args
    g = g if decay else jnp.zeros_like(g)
    actual = polynomial_gated_delta_rule_reference(q, k, v, g, beta, jnp.zeros_like(erase), .5)
    scores = jnp.einsum("bthk,bshk->btsh", q, k)
    cumulative = jnp.cumsum(g, axis=1)
    decay_weights = jnp.exp(cumulative[:, :, None, :] - cumulative[:, None, :, :])
    writes = beta[:, None, :, :, 0] + beta[:, None, :, :, 1] * scores + beta[:, None, :, :, 2] * scores ** 2
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    weights = jnp.where(causal[None, :, :, None], decay_weights * writes, 0)
    expected = .5 * jnp.einsum("btsh,bshv->bthv", weights, v)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)


def test_constant_degree_is_invariant_to_query_key_and_erase():
    args, cotangent = polynomial_operands()
    q, k, v, g, beta, erase = args
    beta = beta.at[..., 1:].set(0)
    function = lambda q, k, e: polynomial_gated_delta_rule_reference(q, k, v, g, beta, e, .5)
    actual = function(q, k, erase)
    np.testing.assert_array_equal(actual, function(q * 2, k * 3, jnp.ones_like(erase)))
    gradients = jax.grad(lambda q, k, e: jnp.sum(function(q, k, e) * cotangent), argnums=(0, 1, 2))(q, k, erase)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


def test_noncommuting_query_transitions_have_the_correct_time_order():
    args, _ = polynomial_operands(length=5, heads=1, key_dim=2)
    q, k, v, g, beta, erase = (np.asarray(a, np.float64) for a in args)
    expected = np.zeros_like(v)
    wrong_order = np.zeros_like(v)
    for t in range(q.shape[1]):
        for i in range(t + 1):
            key, reversed_key = k[0, i, 0].copy(), k[0, i, 0].copy()
            for j in range(i + 1, t + 1):
                p = np.eye(2) - erase[0, j, 0] * np.outer(k[0, j, 0], k[0, j, 0])
                key = p @ key
            for j in range(t, i, -1):
                p = np.eye(2) - erase[0, j, 0] * np.outer(k[0, j, 0], k[0, j, 0])
                reversed_key = p @ reversed_key
            decay = np.exp(np.sum(g[0, i + 1:t + 1, 0]))
            for output, source in ((expected, key), (wrong_order, reversed_key)):
                score = q[0, t, 0] @ source
                output[0, t, 0] += decay * np.polynomial.polynomial.polyval(score, beta[0, i, 0]) * v[0, i, 0]
    actual = polynomial_gated_delta_rule_reference(*args, 1.)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
    assert np.linalg.norm(wrong_order - expected) > .01


def test_quadratic_state_has_one_decay_not_its_square():
    q = k = jnp.ones((1, 2, 1, 1))
    v = jnp.array([2., 0.]).reshape(1, 2, 1, 1)
    g = jnp.log(jnp.array([1., .25])).reshape(1, 2, 1)
    beta = jnp.zeros((1, 2, 1, 3)).at[..., 2].set(1)
    erase = jnp.zeros_like(g)
    actual = polynomial_gated_delta_rule_reference(q, k, v, g, beta, erase, 1.)
    np.testing.assert_allclose(actual.reshape(-1), [2., .5], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("erase_value", [0., 1e-6, 1.])
def test_repeated_key_hard_gates_have_correct_fp64_gradients(erase_value):
    args, cotangent = polynomial_operands(length=17, heads=1, key_dim=4, value_dim=2)
    q, k, v, g, beta, erase = args
    key = jnp.full_like(k, .5)  # Exactly unit length even after BF16 quantization.
    args = (key, key, v, jnp.zeros_like(g), beta, jnp.full_like(erase, erase_value))
    actual = value_and_input_gradients(polynomial_gated_delta_rule_reference, args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        expected = value_and_input_gradients(tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                             cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=5e-6, atol=3e-6)


def test_zero_new_degrees_still_receive_nonzero_gradients():
    args, cotangent = polynomial_operands()
    q, k, v, g, beta, erase = args
    beta = beta.at[..., 0].set(0).at[..., 2].set(0)
    gradient = jax.grad(lambda b: jnp.sum(polynomial_gated_delta_rule_reference(q, k, v, g, b, erase, .5)
                                         * cotangent))(beta)
    assert float(jnp.linalg.norm(gradient[..., 0])) > 0
    assert float(jnp.linalg.norm(gradient[..., 2])) > 0


def test_repeated_key_closed_form_oracle_matches_full_tensor_derivatives():
    from scaffolds.transformer.tests.test_polynomial_gdn_kernel import repeated_key_oracle

    args, cotangent = polynomial_operands(length=9, heads=1, key_dim=4, value_dim=2)
    _, k, v, g, beta, erase = args
    key = jnp.full_like(k, .5)
    args = (key, key, v, jnp.zeros_like(g), beta.at[..., 0].set(0), jnp.ones_like(erase))
    expected = repeated_key_oracle(args, cotangent, .5)
    with enable_x64(), jax.default_matmul_precision("highest"):
        actual = value_and_input_gradients(tensor_oracle, tuple(a.astype(jnp.float64) for a in args),
                                           cotangent.astype(jnp.float64), .5)
    assert_outputs_and_gradients(actual, expected, rtol=1e-12, atol=1e-12)


def test_recurrence_is_causal():
    args, _ = polynomial_operands(length=13)
    original = polynomial_gated_delta_rule_reference(*args, .5)
    changed = tuple(a.at[:, 8:].add(.25) for a in args)
    perturbed = polynomial_gated_delta_rule_reference(*changed, .5)
    np.testing.assert_array_equal(original[:, :8], perturbed[:, :8])
    assert not np.allclose(original[:, 8:], perturbed[:, 8:])
