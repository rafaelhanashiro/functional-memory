"""Test the split kernels and train-time gradients independently of the blocked Horner implementation."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention import split_score


@pytest.mark.parametrize("branch", split_score.BRANCHES[1:])
def test_dense_values_and_gradients(branch):
    check_dense_values_and_gradients(branch, length=13, width=4)


@pytest.mark.gpu
@pytest.mark.parametrize("branch", split_score.BRANCHES[1:])
def test_gpu_causal_strip_boundary(branch):
    assert jax.default_backend() == "gpu"
    with jax.default_matmul_precision("highest"):
        check_dense_values_and_gradients(branch, length=519, width=64)


def check_dense_values_and_gradients(branch, length, width):
    key = jax.random.key(205)
    q, k, v = [jax.random.normal(jax.random.fold_in(key, i), (1, length, 2, width)) * .7 for i in range(3)]
    coeff = jax.random.normal(key, (1, length, 2, 9)) * .01
    coeff = coeff.at[..., 0].add(1)
    raw = jax.random.normal(key, (1, length, 2, 4)) * .08
    if branch == "signed_quartic":
        raw = raw + .3  # Avoid poles in a gradient equivalence test.
    has_d = branch in split_score.DENOMINATORS

    def dense(q, k, v, c, d):
        s = jnp.einsum("bqhd,bkhd->bqhk", q,k)/math.sqrt(width)
        u = jnp.where(s < 0, -s, 0.)
        p = sum(c[...,r,None]*s**r for r in range(9))
        neg = jnp.exp(-u)* (p if branch == "exp_times_sos" else c[...,0,None])
        if has_d:
            a,b,g,h = [d[...,i,None] for i in range(4)]
            if branch == "sos_quartic":
                den = (split_score.SOS_FLOOR+(1+a*u+b*u**2)**2+(g*u+h*u**2)**2)/(1+split_score.SOS_FLOOR)
            else:
                terms = jax.nn.softplus(d) if branch == "positive_quartic" else d
                den = 1 + sum(terms[...,r-1,None]*u**r/math.factorial(r) for r in range(1,5))
            neg = c[...,0,None]/den
        valid = jnp.tril(jnp.ones((length,length),bool))[None,:,None,:]
        return jnp.einsum("bqhk,bkhd->bqhd",jnp.where(valid,jnp.where(s<0,neg,p),0),v)

    def kernel(q,k,v,c,d,impl,diagnostics=False):
        extra = (split_score.denominator_parameters(d,branch) if has_d else None)
        return attention_core(q,k,v,backend="blocked", attention_degree=8,attention_normalization=False,
                              query_polynomial_coefficients=c,negative_score_branch=branch,
                              negative_quartic_parameters=extra,blocked_implementation=impl,return_diagnostics=diagnostics)
    args=(q,k,v,coeff,raw)
    expected=dense(*args)
    cot=jax.random.normal(jax.random.fold_in(key,7),v.shape)
    value, stats=kernel(*args,"scan",True)
    np.testing.assert_allclose(value,expected,atol=5e-5,rtol=2e-4)
    if has_d:
        assert float(stats["split_denominator_abs_min"]) > 0
    actual, pull=jax.vjp(lambda *args: kernel(*args,"strips"),*args)
    _, reference=jax.vjp(dense,*args)
    np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=2e-4)
    for a,b in zip(pull(cot),reference(cot),strict=True):
        if length > 512:
            # Two strips sum each dK/dV in a different FP32 order. Coefficient gradients
            # also sum over hundreds of keys and can cancel at individual coordinates.
            assert np.linalg.norm(np.asarray(a - b).ravel()) <= 3e-5 * max(1., np.linalg.norm(np.asarray(b).ravel()))
        else:
            np.testing.assert_allclose(a,b,atol=5e-5,rtol=3e-4)


@pytest.mark.parametrize("branch", split_score.DENOMINATORS)
def test_common_initial_denominator_and_fixed_intercept(branch):
    u=jnp.asarray([0.,.1,1.,2.,4.,8.])
    bias=jnp.asarray(split_score.initial_bias(branch))
    _, den=split_score.weights(-u,[1.]*9,split_score.denominator_parameters(bias,branch),branch)
    np.testing.assert_allclose(den,sum(u**r/math.factorial(r) for r in range(5)),rtol=2e-6)
    _, learned=split_score.weights(-u,[1.]*9,split_score.denominator_parameters(bias+.35,branch),branch)
    assert learned[0]==1.


def test_signed_denominator_is_not_clamped():
    score=jnp.array([-2.,-1.,0.,1.])
    out,_=split_score.weights(score,[1.,0.],[1.,0.,0.,0.],"signed_quartic")
    np.testing.assert_allclose(out,[1/3,1/2,1,1])
    out,_=split_score.weights(score,[1.,0.],[-1.,0.,0.,0.],"signed_quartic")
    assert out[0] == -1 and jnp.isinf(out[1])


# The query polynomial with direct coefficients takes the factorial scales of its control (r25fffc16c2f3).
DIRECT = dict(attention_coefficient_parameterization="direct", attention_coefficient_scale="factorial",
              attention_coefficient_amplitude_scale="factorial")


@pytest.mark.parametrize("parameterization", ("sum_of_squares", "direct"))
@pytest.mark.parametrize("branch", split_score.BRANCHES[1:])
def test_model_updates_and_reports_each_branch(branch, parameterization):
    from scaffolds.transformer.tests.test_softplus_sos import _config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity
    from scaffolds.transformer.training.state import create_state

    config = _config(attention_score_transform="identity", attention_negative_score_branch=branch,
                     attention_head_rms_prescale="max_abs", **(DIRECT if parameterization == "direct" else {}))
    state = create_state(config, jax.random.PRNGKey(1))
    batch = jnp.asarray(np.random.default_rng(9).integers(0, 32, (2, 9)), jnp.int32)
    after, *_ = loop.train_step(state, batch, jax.random.PRNGKey(0))
    assert attention_identity(config)["attention_negative_score_branch"] == branch
    names = ["query_polynomial_bias"]
    if branch in split_score.DENOMINATORS:
        names.append("negative_quartic_bias")
    for name in names:
        value = np.asarray(after.params[name][...])
        assert np.all(np.isfinite(value))
        assert np.any(value != np.asarray(state.params[name][...]))
    report = attention_diagnostics(after, config, np.asarray(batch[:1]), {"selection": "test"})
    assert report["valid"], report["issues"]


def test_the_direct_query_polynomial_starts_at_taylor_on_both_sides():
    """With direct coefficients h_t starts at T_8; e^s h_t(s), e^s h_t(0) and h_t(0)/D(-s) with D's initial quartic
    T_4(-s) start at e^s T_8(s), e^s and 1/T_4(-s)."""
    from scaffolds.transformer.tests.test_softplus_sos import _config
    from scaffolds.transformer.models import build_model
    from flax import nnx

    s = jnp.asarray([-3., -1., -.25, 0., .5, 2.])
    taylor = [1 / math.factorial(r) for r in range(9)]
    t8 = sum(c * s ** r for r, c in enumerate(taylor))
    for branch, expected in (("exp_times_sos", jnp.where(s < 0, jnp.exp(s) * t8, t8)),
                             ("exp_times_origin", jnp.where(s < 0, jnp.exp(s), t8)),
                             ("positive_quartic", jnp.where(s < 0, 1 / sum((-s) ** r / math.factorial(r)
                                                                         for r in range(5)), t8))):
        config = _config(attention_score_transform="identity", attention_negative_score_branch=branch,
                         attention_head_rms_prescale="max_abs", **DIRECT)
        params = nnx.state(build_model(config.model, rngs=nnx.Rngs(0)), nnx.Param)
        # Layers stack on the first axis and heads on the second; the factorial scale divides b_r = 1 by r!.
        bias = np.asarray(params["query_polynomial_bias"][...])
        np.testing.assert_allclose(bias, np.ones(bias.shape), rtol=1e-6)
        quartic = None
        if branch in split_score.DENOMINATORS:
            raw = jnp.asarray(params["negative_quartic_bias"][...])[0, 0]
            quartic = split_score.denominator_parameters(raw, branch)
        coefficients = [jnp.asarray(c) for c in taylor]
        weights, _ = split_score.weights(s, coefficients, quartic, branch)
        np.testing.assert_allclose(weights, expected, rtol=2e-6)
