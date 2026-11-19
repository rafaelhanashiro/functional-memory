"""The exponential's score functions (attention/pieces.py): w = f(s) exp(phi(s)).

CPU: the blocked scan and causal strips against an FP64 dense reference, softmax and unnormalized,
with and without a cumulative forget gate; the recipes' validation. GPU (``pytest -m gpu`` in an
H100 or B300 allocation): the fused unnormalized kernel against the blocked backend and FP64,
forward and VJP, for every score function it declares.
"""

from dataclasses import replace
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.reference import blocked_backend_attention, blocked_dot_product_attention
from scaffolds.transformer.attention.pieces import (
    FUSED_EXPONENTIAL_PREFACTORS, FUSED_EXPONENTIAL_TRANSFORMS, ScoreFunction, fused_exponential_problem,
)

FUNCTIONS = [(transform, prefactor) for transform in FUSED_EXPONENTIAL_TRANSFORMS
             for prefactor in FUSED_EXPONENTIAL_PREFACTORS if (transform, prefactor) != ("identity", "none")]


def _inputs(shape, *, dtype=jnp.float32, seed=7, decay=False):
    batch, time, heads, qk_dim, value_dim = shape
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(0.8 * rng.standard_normal((batch, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(rng.standard_normal((batch, time, heads, value_dim)), dtype)
    cotangent = jnp.asarray(rng.standard_normal((batch, time, heads, value_dim)), dtype)
    log_decay = None
    if decay:
        # Cumulative L_t = sum_(j<=t) log alpha_j with log alpha in [-0.5, 0].
        log_decay = jnp.asarray(np.cumsum(-0.5 * rng.random((batch, time, heads)), axis=1), jnp.float32)
    return (q, k, v), cotangent, log_decay


def _phi(s, transform):
    return {"identity": s, "softplus": jnp.logaddexp(s, 0.0), "sigmoid": jax.nn.sigmoid(s)}[transform]


def _f(s, prefactor):
    return {"none": jnp.ones_like(s), "softplus": jnp.logaddexp(s, 0.0), "sigmoid": jax.nn.sigmoid(s)}[prefactor]


def _dense(q, k, v, *, transform, prefactor, normalized, log_decay=None):
    """y_t = sum_(i<=t) w_ti v_i [/ sum_(i<=t) w_ti], w_ti = f(s_ti) exp(phi(s_ti) + L_t - L_i)."""
    time = q.shape[1]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones((time, time), bool))
    exponent = _phi(scores, transform)
    if log_decay is not None:
        cumulative = jnp.swapaxes(log_decay, 1, 2)
        exponent = exponent + cumulative[..., :, None] - cumulative[..., None, :]
    weights = jnp.where(causal, _f(scores, prefactor) * jnp.exp(jnp.where(causal, exponent, 0.0)), 0.0)
    if normalized:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def _forward_and_vjp(attend, operands, cotangent):
    output, pullback = jax.vjp(attend, *operands)
    return output, pullback(cotangent)


def _fp64(operands, cotangent, **settings):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        operands64 = tuple(x.astype(jnp.float64) for x in operands)
        log_decay = settings.pop("log_decay")
        settings["log_decay"] = None if log_decay is None else log_decay.astype(jnp.float64)
        result = _forward_and_vjp(partial(_dense, **settings), operands64, cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, np.float64), result)


def _assert_close(actual, expected, *, rtol, label):
    for name, observed, reference in zip(("output", "dq", "dk", "dv"), (actual[0], *actual[1]),
                                         (expected[0], *expected[1]), strict=True):
        observed, reference = np.asarray(observed, np.float64), np.asarray(reference, np.float64)
        assert np.isfinite(observed).all(), f"{label} {name}: nonfinite values"
        error = np.linalg.norm(observed - reference) / max(np.linalg.norm(reference), 1e-30)
        assert error <= rtol, f"{label} {name}: relative L2 error {error:.3g} exceeds {rtol}"


@pytest.mark.parametrize("implementation", ["scan", "strips"])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("transform,prefactor", FUNCTIONS)
def test_blocked_score_functions_match_fp64_dense(transform, prefactor, decay, normalized, implementation):
    operands, cotangent, log_decay = _inputs((2, 37, 2, 16, 8), decay=decay)
    decay_operand = {} if log_decay is None else {"query_log_decay": log_decay}
    attend = partial(blocked_backend_attention, implementation=implementation, attention_normalization=normalized,
                     score_transform=transform, score_prefactor=prefactor, **decay_operand)
    actual = jax.jit(partial(_forward_and_vjp, attend))(operands, cotangent)
    expected = _fp64(operands, cotangent, transform=transform, prefactor=prefactor, normalized=normalized,
                     log_decay=log_decay)
    _assert_close(actual, expected, rtol=2e-5, label=f"{transform}/{prefactor}")


def test_identity_score_functions_are_the_historical_exponential():
    operands, _, _ = _inputs((1, 19, 2, 8, 8))
    for normalized in (False, True):
        historical = blocked_backend_attention(*operands, attention_normalization=normalized)
        explicit = blocked_backend_attention(*operands, attention_normalization=normalized,
                                             score_transform="identity", score_prefactor="none")
        np.testing.assert_array_equal(np.asarray(historical), np.asarray(explicit))


def test_softplus_input_is_one_plus_exponential():
    """exp(softplus(s)) = 1 + e^s: the softplus-input exponential adds a uniform causal average."""
    operands, _, _ = _inputs((1, 23, 2, 8, 8))
    transformed = blocked_backend_attention(*operands, attention_normalization=False, score_transform="softplus")
    plain = blocked_backend_attention(*operands, attention_normalization=False)
    np.testing.assert_allclose(np.asarray(transformed), np.asarray(plain + jnp.cumsum(operands[2], axis=1)),
                               rtol=2e-5, atol=2e-5)


def test_fused_declaration():
    for transform, prefactor in FUNCTIONS:
        assert fused_exponential_problem(ScoreFunction(transform, prefactor), attention_degree=None) is None
    assert fused_exponential_problem(ScoreFunction("identity", "none"), attention_degree=4) is None
    assert fused_exponential_problem(ScoreFunction("identity", "sigmoid"), attention_degree=4) is not None
    assert fused_exponential_problem(ScoreFunction("abs", "none"), attention_degree=None) is not None
    with pytest.raises(ValueError, match="Unknown attention score prefactor"):
        ScoreFunction("identity", "tanh")


def _exp_model(**changes):
    from scaffolds.transformer.configs.base import ModelConfig

    settings = dict(attention_degree=None, attention_normalization=False, attention_output_scaling="head_rms_norm",
                    attention_backend="fused", qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
                    qk_norm_head_scale=True)
    settings.update(changes)
    return ModelConfig(**settings)


def test_recipe_validation():
    for transform, prefactor in FUNCTIONS:
        _exp_model(attention_score_transform=transform, attention_score_prefactor=prefactor)
        _exp_model(attention_score_transform=transform, attention_score_prefactor=prefactor,
                   attention_backend="blocked", attention_normalization=True)
        _exp_model(attention_score_transform=transform, attention_score_prefactor=prefactor,
                   attention_backend="blocked", attention_forget_gate=True)
    with pytest.raises(ValueError, match="fused kernel is the unnormalized exponential"):
        _exp_model(attention_score_prefactor="softplus", attention_normalization=True)
    with pytest.raises(ValueError, match="blocked or fused"):
        _exp_model(attention_score_prefactor="sigmoid", attention_backend="auto")
    with pytest.raises(ValueError, match="prefactor applies so far to the exponential"):
        _exp_model(attention_score_prefactor="softplus", attention_backend="blocked", attention_degree=4)


def test_layers_build_and_differentiate():
    """Every score function builds through the layer and differentiates with finite gradients."""
    from flax import nnx

    from scaffolds.transformer.models import build_model

    tiny = dict(num_layers=1, hidden_size=64, num_heads=2, num_kv_heads=2, intermediate_size=128, max_seq_len=32,
                vocab_size=64, attention_backend="blocked")
    tokens = jnp.arange(64).reshape(2, 32) % 64
    for transform, prefactor in FUNCTIONS:
        model = build_model(replace(_exp_model(attention_score_transform=transform,
                                               attention_score_prefactor=prefactor), **tiny), 0)
        loss, grads = nnx.value_and_grad(lambda m: jnp.mean(m(tokens).astype(jnp.float32) ** 2))(model)
        assert np.isfinite(float(loss))
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(grads))


def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation; the compiled fused kernel is under test")


@pytest.mark.gpu
@pytest.mark.parametrize("transform,prefactor", FUNCTIONS)
@pytest.mark.parametrize("shape", [(2, 17, 3, 7, 5), (1, 131, 2, 33, 65), (1, 200, 2, 64, 64)])
def test_fused_fp32_matches_blocked_and_fp64(transform, prefactor, shape):
    _requires_gpu()
    from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention

    operands, cotangent, _ = _inputs(shape, seed=91)
    fused = partial(fused_unnormalized_attention, score_transform=transform, score_prefactor=prefactor)
    # As test_unnormalized_fused_attention: 32-query blocks (XLA's GPU layout pass rejects this scan's
    # 128-query blocks at full precision past 128 tokens).
    blocked = partial(blocked_dot_product_attention, attention_normalization=False, score_transform=transform,
                      score_prefactor=prefactor, query_block_size=32)
    with jax.default_matmul_precision("highest"):
        # The blocked reference's FP32 dot products at full precision, not TF32.
        actual = jax.jit(partial(_forward_and_vjp, fused))(operands, cotangent)
        reference = jax.jit(partial(_forward_and_vjp, blocked))(operands, cotangent)
    expected = _fp64(operands, cotangent, transform=transform, prefactor=prefactor, normalized=False, log_decay=None)
    _assert_close(actual, expected, rtol=1e-4, label=f"fused {transform}/{prefactor} vs FP64")
    _assert_close(actual, reference, rtol=1e-4, label=f"fused {transform}/{prefactor} vs blocked")


@pytest.mark.gpu
@pytest.mark.parametrize("transform,prefactor", FUNCTIONS)
def test_fused_bf16_production_context_matches_blocked(transform, prefactor):
    _requires_gpu()
    from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention

    operands, cotangent, _ = _inputs((1, 2048, 2, 64, 64), dtype=jnp.bfloat16, seed=5)
    fused = partial(fused_unnormalized_attention, score_transform=transform, score_prefactor=prefactor)
    blocked = partial(blocked_backend_attention, attention_normalization=False, score_transform=transform,
                      score_prefactor=prefactor, implementation="strips")
    actual = jax.jit(partial(_forward_and_vjp, fused))(operands, cotangent)
    reference = jax.jit(partial(_forward_and_vjp, blocked))(operands, cotangent)
    _assert_close(actual, reference, rtol=0.03, label=f"bf16 fused {transform}/{prefactor} vs blocked strips")


@pytest.mark.gpu
@pytest.mark.parametrize("transform,prefactor", FUNCTIONS)
def test_fused_masked_future_and_invalid_scores(transform, prefactor):
    _requires_gpu()
    from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention

    (q, k, v), _, _ = _inputs((1, 9, 1, 4, 4), seed=3)
    # A huge future key score never reaches earlier rows; a nonfinite causal score poisons only its row.
    k = k.at[0, 8, 0].set(1e4)
    output = fused_unnormalized_attention(q, k, v, score_transform=transform, score_prefactor=prefactor)
    assert np.isfinite(np.asarray(output[0, :8])).all()
    q = q.at[0, 4, 0, 0].set(jnp.nan)
    output = fused_unnormalized_attention(q, k, v, score_transform=transform, score_prefactor=prefactor)
    assert np.isnan(np.asarray(output[0, 4])).all() and np.isfinite(np.asarray(output[0, :4])).all()
