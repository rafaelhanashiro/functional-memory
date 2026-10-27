"""Free per-example query coefficients times free key factors: c_r(t) g_r(i)."""
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_coefficient_override import _static_config
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import make_query_key_functions, qualify

FACTORIAL = jnp.asarray([math.factorial(r) for r in range(5)], jnp.float32)


def _taylor(**updates):
    return _static_config(attention_degree=4, attention_coefficients="fixed", attention_coefficient_scale="none",
                          qk_norm_head_scale=True, **updates)


def _tokens():
    return jnp.arange(16).reshape(2, 8) % 16


@pytest.fixture(scope="module")
def taylor():
    model = build_model(_taylor(max_seq_len=8), 7)
    model.qk_norm_head_log_scale[...] = jnp.asarray([[[.3], [-.2]], [[.1], [.4]]])
    graph, params = nnx.split(model, nnx.Param)
    return graph, params


def test_key_factors_multiply_each_keys_order_term(taylor):
    graph, params = taylor
    model = nnx.merge(graph, params)
    tokens = _tokens()[:, :7]
    expected = model(tokens)
    _, coefficients = model(tokens, return_coefficients=True)
    ones = jnp.ones_like(coefficients)
    np.testing.assert_allclose(model(tokens, query_coefficients=coefficients, key_factors=ones), expected,
                               rtol=2e-5, atol=2e-6)
    # A key factor constant across keys is the same as rescaling the query coefficient.
    scale = jnp.asarray([1.3, .7, 2., .4, 1.1], jnp.float32)
    np.testing.assert_allclose(model(tokens, query_coefficients=coefficients, key_factors=ones * scale),
                               model(tokens, query_coefficients=coefficients * scale), rtol=2e-5, atol=2e-6)
    # Key-dependent factors change the function and receive gradients.
    varied = ones.at[:, :, 2].multiply(3.0)
    assert np.abs(np.asarray(model(tokens, query_coefficients=coefficients, key_factors=varied) - expected)).max() > 1e-4
    gradient = jax.grad(lambda g: model(tokens, query_coefficients=coefficients, key_factors=g).sum())(varied)
    assert np.isfinite(gradient).all() and float(jnp.linalg.norm(gradient)) > 0


def test_key_factors_require_replaced_static_coefficients(taylor):
    graph, params = taylor
    model = nnx.merge(graph, params)
    tokens = _tokens()[:, :7]
    _, coefficients = model(tokens, return_coefficients=True)
    with pytest.raises(ValueError, match="key_factors"):
        model(tokens, key_factors=jnp.ones_like(coefficients))
    with pytest.raises(ValueError, match="key_factors"):
        model(tokens, query_coefficients=coefficients, key_factors=jnp.ones_like(coefficients)[..., :4])
    query = build_model(_static_config(attention_degree=4, attention_coefficient_conditioning="query",
                                       attention_coefficient_activation="tanh",
                                       attention_coefficient_amplitude_scale="factorial",
                                       attention_coefficient_scale="factorial",
                                       attention_coefficient_projection_initial_std=.1), 7)
    _, query_coefficients = query(tokens, return_coefficients=True)
    with pytest.raises(ValueError, match="key_factors"):
        query(tokens, query_coefficients=query_coefficients, key_factors=jnp.ones_like(query_coefficients))


def test_step_is_independent_per_example_clipped_adam_on_both_tables(taylor):
    graph, params = taylor
    # A larger epsilon keeps g / (|g| + epsilon) well conditioned where a position's
    # gradient is tiny, so batched and per-example rounding cannot flip the update.
    options = dict(clip_norm=.05, beta1=.9, beta2=.95, epsilon=1e-3)
    capture, initialize, step, evaluate = make_query_key_functions(graph, graph, 4, options)
    tokens = _tokens()
    coefficients, parent_nll, _, _ = capture(params, tokens)
    rng = np.random.default_rng(3)
    tables = (coefficients * jnp.asarray(1 + .1 * rng.normal(size=coefficients.shape), jnp.float32),
              jnp.asarray(1 + .2 * rng.normal(size=coefficients.shape), jnp.float32))
    np.testing.assert_allclose(evaluate((coefficients, jnp.ones_like(coefficients)), params, tokens), parent_nll,
                               rtol=1e-5, atol=1e-5)
    before = state_sha256(params)
    updated, _, losses, norms, finite = step(tables, initialize(tables), params, tokens, 1e-2)
    assert bool(finite) and state_sha256(params) == before

    def one_example(c, g, example):
        logits = nnx.merge(graph, params)(tokens[example:example + 1, :-1],
                                          query_coefficients=c[None].swapaxes(0, 1),
                                          key_factors=g[None].swapaxes(0, 1))
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), tokens[example:example + 1, 1:]).mean()

    for example in range(2):
        gc, gg = jax.grad(one_example, argnums=(0, 1))(tables[0][example], tables[1][example], example)
        gc = gc / FACTORIAL
        norm = jnp.sqrt(jnp.sum(gc ** 2) + jnp.sum(gg ** 2))
        np.testing.assert_allclose(norms[example], norm, rtol=1e-4)
        scale = jnp.minimum(1., options["clip_norm"] / norm)
        # Adam's first bias-corrected step is g / (|g| + epsilon).
        adam = lambda grad: grad * scale / (jnp.abs(grad * scale) + options["epsilon"])
        np.testing.assert_allclose(updated[0][example], tables[0][example] - 1e-2 * adam(gc) / FACTORIAL,
                                   rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(updated[1][example], tables[1][example] - 1e-2 * adam(gg), rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(losses[example], one_example(tables[0][example], tables[1][example], example),
                                   rtol=1e-5)


def test_qualification_passes_with_identity_key_factors(taylor, tmp_path):
    graph, params = taylor
    options = dict(clip_norm=1., beta1=.9, beta2=.95, epsilon=1e-8)
    functions = make_query_key_functions(graph, graph, 4, options)
    capture, initialize = functions[:2]
    tokens = _tokens()
    coefficients = capture(params, tokens)[0]
    tables = (coefficients, jnp.ones_like(coefficients))
    acceptance = qualify(tmp_path, functions, params, tokens, tables, initialize(tables), 1e-2)
    assert acceptance["status"] == "passed" and acceptance["checkpoint_resume_exact"]
    assert acceptance["improved_training_nll"] < acceptance["separate_training_forward_nll"]


def test_launcher_scope_and_taylor_parent():
    from experiments.transformer_200m_4b.launch_coefficient_fit import fitting_scope

    assert fitting_scope("free_per_example_query_key_coefficients", 256, 256) == (
        256, "free_query_key_coefficients_full_batch")
    with pytest.raises(ValueError):
        fitting_scope("free_per_example_query_key_coefficients", 256, 1024)
