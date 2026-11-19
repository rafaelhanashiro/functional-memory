"""Composition on the blocked backend (attention/pieces.py): pieces no specialized route combines.

A per-query polynomial (static and fixed coefficients arrive broadcast per query) takes key factors, a
cumulative decay, per-degree row shifts, a score transform and a prefactor together:

    w_ti = f(s_ti) D_ti sum_r c_r(t) K_r(i) (phi(s_ti) + beta_r(t))^r,    beta_0 = 0,

checked against an FP64 dense evaluation, forward and VJP, on the scan and the causal strips; and recipes
the scaffold refused before the refactor (fixed Taylor coefficients with a forget gate, static coefficients
with score biases or other gates, transforms with gates) build and train through the model.
"""

from dataclasses import replace
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.core import attention_core


def _phi(s, transform):
    return {"identity": s, "softplus": jnp.logaddexp(s, 0.0), "sigmoid": jax.nn.sigmoid(s)}[transform]


def _f(s, prefactor):
    return {"none": jnp.ones_like(s), "softplus": jnp.logaddexp(s, 0.0), "sigmoid": jax.nn.sigmoid(s)}[prefactor]


def _dense(q, k, v, c, key_factors, log_decay, row_shift, *, transform, prefactor):
    time = q.shape[1]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones((time, time), bool))
    z = _phi(scores, transform)
    weights = jnp.zeros_like(scores)
    for order in range(c.shape[-1]):
        coefficient = jnp.swapaxes(c[..., order], 1, 2)[..., None]
        if key_factors is not None:
            coefficient = coefficient * jnp.swapaxes(key_factors[..., order], 1, 2)[:, :, None, :]
        base = z
        if row_shift is not None and order >= 1:
            base = z + jnp.swapaxes(row_shift[..., order - 1], 1, 2)[..., None]
        weights = weights + coefficient * base ** order
    if log_decay is not None:
        cumulative = jnp.swapaxes(log_decay, 1, 2)
        weights = weights * jnp.exp(jnp.where(causal, cumulative[..., :, None] - cumulative[..., None, :], 0.0))
    weights = jnp.where(causal, weights * _f(scores, prefactor), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def _operands(*, degree=3, seed=3, shape=(2, 41, 2, 16, 8)):
    batch, time, heads, qk_dim, value_dim = shape
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(0.6 * rng.standard_normal((batch, time, heads, qk_dim)), jnp.float32) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((batch, time, heads, value_dim)), jnp.float32)
    c = jnp.asarray(np.array([1 / math.factorial(r) for r in range(degree + 1)])
                    * (1 + 0.2 * rng.standard_normal((batch, time, heads, degree + 1))), jnp.float32)
    key_factors = jnp.asarray(1 + 0.2 * rng.standard_normal(c.shape), jnp.float32)
    log_decay = jnp.asarray(np.cumsum(-0.2 * rng.random((batch, time, heads)), axis=1), jnp.float32)
    row_shift = jnp.asarray(0.3 * rng.standard_normal((batch, time, heads, degree)), jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal((batch, time, heads, value_dim)), jnp.float32)
    return q, k, v, c, key_factors, log_decay, row_shift, cotangent


COMBINATIONS = {
    "transform+decay": dict(decay=True, transform="softplus"),
    "transform+bias": dict(bias=True, transform="softplus"),
    "bias+decay": dict(bias=True, decay=True),
    "prefactor+decay": dict(decay=True, prefactor="sigmoid"),
    "keys+decay+transform": dict(keys=True, decay=True, transform="sigmoid"),
    "everything": dict(keys=True, decay=True, bias=True, transform="softplus", prefactor="softplus"),
}


@pytest.mark.parametrize("implementation", ["scan", "strips"])
@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_composition_matches_fp64_dense(name, implementation):
    combination = COMBINATIONS[name]
    q, k, v, c, key_factors, log_decay, row_shift, cotangent = _operands()
    keys = key_factors if combination.get("keys") else None
    decay = log_decay if combination.get("decay") else None
    shift = row_shift if combination.get("bias") else None
    transform, prefactor = combination.get("transform", "identity"), combination.get("prefactor", "none")
    extra = {name: value for name, value in (("key_polynomial_factors", keys), ("query_log_decay", decay),
                                             ("score_bias_row_shift", shift)) if value is not None}

    def blocked(q, k, v, c):
        return attention_core(q, k, v, backend="blocked", attention_degree=c.shape[-1] - 1,
                              attention_normalization=False, query_polynomial_coefficients=c,
                              attention_coefficient_activation="tanh", score_transform=transform,
                              score_prefactor=prefactor, blocked_implementation=implementation, **extra)

    output, pullback = jax.vjp(blocked, q, k, v, c)
    actual = (output, *pullback(cotangent))
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        as64 = lambda x: None if x is None else x.astype(jnp.float64)
        dense = partial(_dense, key_factors=as64(keys), log_decay=as64(decay), row_shift=as64(shift),
                        transform=transform, prefactor=prefactor)
        reference, pullback64 = jax.vjp(dense, *(x.astype(jnp.float64) for x in (q, k, v, c)))
        expected = (reference, *pullback64(cotangent.astype(jnp.float64)))
    for label, observed, wanted in zip(("output", "dq", "dk", "dv", "dc"), actual, expected, strict=True):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        assert np.isfinite(observed).all(), f"{name} {label}: nonfinite values"
        error = np.linalg.norm(observed - wanted) / np.linalg.norm(wanted)
        assert error <= 3e-5, f"{name} {label}: relative L2 error {error:.3g}"


def _model(**changes):
    from scaffolds.transformer.configs.base import ModelConfig

    settings = dict(attention_normalization=False, attention_output_scaling="head_rms_norm", attention_backend="blocked",
                    attention_blocked_implementation="strips", qk_norm=True, qk_norm_type="rms_norm",
                    qk_norm_sharing="per_layer", qk_norm_head_scale=True, attention_degree=4)
    settings.update(changes)
    return ModelConfig(**settings)


STATIC = dict(attention_coefficients="learned", attention_coefficient_sharing="per_head")
QUERY = dict(STATIC, attention_coefficient_conditioning="query", attention_coefficient_activation="tanh")
RECIPES = {
    "fixed Taylor + FoX": dict(attention_forget_gate=True),
    "fixed Taylor + query-distance gate": dict(attention_forget_gate=True, attention_forget_gate_mode="query_distance"),
    "fixed Taylor + constant gate": dict(attention_forget_gate=True, attention_forget_gate_form="constant"),
    "fixed table + FoX": dict(attention_fixed_coefficients=(0.0, 1.0, 0.5, 0.0, 0.1), attention_forget_gate=True),
    "static + sigmoid gate": dict(STATIC, attention_forget_gate=True, attention_forget_gate_form="sigmoid"),
    "static + score bias": dict(STATIC, attention_score_bias="static"),
    "fixed Taylor + softplus input": dict(attention_score_transform="softplus"),
    "static + sigmoid prefactor + FoX": dict(STATIC, attention_score_prefactor="sigmoid", attention_forget_gate=True),
    "query + softplus + FoX": dict(QUERY, attention_score_transform="softplus", attention_forget_gate=True),
    "query + score bias + FoX": dict(QUERY, attention_score_bias="query", attention_forget_gate=True),
}


@pytest.mark.parametrize("name", list(RECIPES))
def test_recipes_refused_before_the_refactor_build_and_train(name):
    from flax import nnx

    from scaffolds.transformer.models import build_model

    model = build_model(replace(_model(**RECIPES[name]), num_layers=1, hidden_size=64, num_heads=2, num_kv_heads=2,
                                intermediate_size=128, max_seq_len=32, vocab_size=64), 0)
    tokens = jnp.arange(64).reshape(2, 32) % 64
    loss, grads = nnx.value_and_grad(lambda m: jnp.mean(m(tokens).astype(jnp.float32) ** 2))(model)
    assert np.isfinite(float(loss))
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)


def test_fixed_taylor_with_gate_is_static_taylor_with_gate():
    """Fixed Taylor coefficients per query compute what learned static coefficients at 1/r! compute."""
    from flax import nnx

    from scaffolds.transformer.models import build_model

    tiny = dict(num_layers=1, hidden_size=64, num_heads=2, num_kv_heads=2, intermediate_size=128, max_seq_len=32,
                vocab_size=64)
    tokens = jnp.arange(64).reshape(2, 32) % 64
    fixed = build_model(replace(_model(attention_forget_gate=True), **tiny), 0)
    static = build_model(replace(_model(**STATIC, attention_forget_gate=True), **tiny), 0)
    # Copy every shared array; the static model's coefficients are its Taylor start 1/r! (stored with factorial scale
    # applied inside the layer), so both models compute the same function.
    fixed_state, static_state = nnx.state(fixed, nnx.Param), nnx.state(static, nnx.Param)
    fixed_leaves = dict(jax.tree_util.tree_leaves_with_path(fixed_state))
    for path, leaf in jax.tree_util.tree_leaves_with_path(static_state):
        if path in fixed_leaves:
            leaf.value = fixed_leaves[path]
    nnx.update(static, static_state)
    np.testing.assert_allclose(np.asarray(fixed(tokens)), np.asarray(static(tokens)), rtol=2e-6, atol=2e-6)
