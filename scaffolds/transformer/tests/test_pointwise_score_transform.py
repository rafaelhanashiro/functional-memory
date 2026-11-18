"""Fixed p(z)=z attention: independent CPU oracles and H100/B200 Mosaic qualification.

Only the score function changes: sigmoid(s) or softplus(s), no causal-sum division. The kernel's
coefficient operand is exactly [0, 1] at every query/head. GPU tests compare BF16 Q/K/V and FP32
coefficients with explicit NumPy FP64 output and all four VJPs, including the production context.
The oracle runs on the host and never compiles a dense attention reference on the allocated GPU.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import score_transform
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.mosaic import api, blackwell, common, hopper

TRANSFORMS = ("sigmoid", "softplus")


def _phi(transform, scores):
    logistic = np.exp(-np.logaddexp(0.0, -scores))
    if transform == "softplus":
        return np.logaddexp(0.0, scores), logistic
    # Separate log-domain formula: its positive-tail derivative does not subtract a rounded one.
    return logistic, np.exp(-np.logaddexp(0.0, scores) - np.logaddexp(0.0, -scores))


def _numpy_oracle(transform, operands, cotangent):
    q, k, v, c = (np.asarray(x, np.float64) for x in operands)
    do = np.asarray(cotangent, np.float64)
    scale, time = 1.0 / math.sqrt(q.shape[-1]), q.shape[1]
    causal = np.tril(np.ones((time, time), bool))
    scores = np.einsum("bthd,bihd->bhti", q, k, optimize=True) * scale
    z, derivative = _phi(transform, scores)
    c = c.transpose(0, 2, 1, 3)
    weights = np.where(causal, c[..., 0, None] + c[..., 1, None] * z, 0.0)
    dp = np.where(causal, np.einsum("bthd,bihd->bhti", do, v, optimize=True), 0.0)
    ds = dp * c[..., 1, None] * derivative
    return np.einsum("bhti,bihd->bthd", weights, v, optimize=True), (
        np.einsum("bhti,bihd->bthd", ds, k, optimize=True) * scale,
        np.einsum("bhti,bthd->bihd", ds, q, optimize=True) * scale,
        np.einsum("bhti,bthd->bihd", weights, do, optimize=True),
        np.stack((dp.sum(-1), (dp * z).sum(-1)), -1).transpose(0, 2, 1, 3),
    )


def _inputs(*, time, width=64, heads=2, dtype=jnp.float32, spread=1.0):
    rng = np.random.default_rng(143)
    q = jnp.asarray(spread * rng.normal(size=(1, time, heads, width)), dtype)
    k = jnp.asarray(rng.normal(size=q.shape), dtype)
    v = jnp.asarray(rng.normal(size=(1, time, heads, 64)), dtype)
    c = jnp.broadcast_to(jnp.array([0.0, 1.0], jnp.float32), (*q.shape[:3], 2))
    do = jnp.asarray(rng.normal(size=v.shape), dtype)
    return (q, k, v, c), do


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _assert_relative(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert np.isfinite(actual).all(), label
    scale = np.linalg.norm(expected)
    if scale == 0:
        np.testing.assert_array_equal(actual, expected, err_msg=label)
    else:
        error = np.linalg.norm(actual - expected) / scale
        assert error <= tolerance, f"{label}: relative L2 {error:.6g} > {tolerance}"


def _assert_all(actual, expected, *, gpu):
    tolerances = (0.006, 0.025, 0.025, 0.015, 0.01) if gpu else (4e-6,) * 5
    for label, got, want, tolerance in zip(
            ("output", "dq", "dk", "dv", "dc"), (actual[0], *actual[1]),
            (expected[0], *expected[1]), tolerances, strict=True):
        if label == "dc":
            for order in range(2):
                _assert_relative(got[..., order], want[..., order], tolerance, f"dc{order}")
        else:
            _assert_relative(got, want, tolerance, label)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_sigmoid_values_and_derivatives_include_both_tails_and_zero(dtype):
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        s = jnp.asarray([-1e30, -80., -20., -2., 0., 2., 20., 80., 1e30], dtype)
        expected, derivative = _phi("sigmoid", np.asarray(s, np.float64))
        function = partial(score_transform.apply, transform="sigmoid")
        observed, tangent = jax.jvp(function, (s,), (jnp.ones_like(s),))
        reverse = jax.vjp(function, s)[1](jnp.ones_like(s))[0]
        assert observed.dtype == dtype
        # The independent log-domain oracle rounds its exponent before exp; allow its few ulps too.
        np.testing.assert_allclose(observed, expected, rtol=16 * np.finfo(dtype).eps, atol=0.)
        np.testing.assert_allclose(tangent, derivative, rtol=16 * np.finfo(dtype).eps, atol=0.)
        np.testing.assert_array_equal(reverse, tangent)
        assert float(tangent[4]) == .25
        assert np.all(np.asarray(tangent)[[6, 7]] > 0), "A value rounded to one must still train"
        invalid = jnp.array([jnp.inf, -jnp.inf, jnp.nan], dtype)
        value, gradient = jax.jvp(function, (invalid,), (jnp.ones_like(invalid),))
        assert bool(jnp.isnan(value).all() & jnp.isnan(gradient).all())


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_tile_values_and_slopes_preserve_extreme_scores(transform):
    with jax.default_device(jax.devices("cpu")[0]):
        scores = jnp.array([-1e30, -80., -20., -2., 0., 2., 20., 80., 1e30], jnp.float32)
        expected, derivative = _phi(transform, np.asarray(scores, np.float64))
        values, slopes = common.transformed_score(api.TRANSFORM_FAMILY[transform], scores, slope=True)
        np.testing.assert_allclose(values, expected, rtol=8e-6, atol=0.)
        np.testing.assert_allclose(slopes, derivative, rtol=8e-6, atol=0.)
        invalid = jnp.array([jnp.inf, -jnp.inf, jnp.nan], jnp.float32)
        values, _ = common.transformed_score(api.TRANSFORM_FAMILY[transform], invalid, slope=True)
        assert not bool(jnp.isfinite(values).any())


@pytest.mark.parametrize("implementation", ("scan", "strips"))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_blocked_fixed_linear_attention_and_all_gradients_match_numpy(transform, implementation):
    with jax.default_device(jax.devices("cpu")[0]):
        operands, do = _inputs(time=13, width=8, spread=6.)
        q, k, v, c = operands
        operands = (q.at[:, 3].set(0), k, v, c)
        options = dict(attention_degree=1, attention_normalization=False, score_transform=transform)
        if implementation == "scan":
            function = lambda q, k, v, c: reference.blocked_dot_product_attention(
                q, k, v, query_block_size=4, query_polynomial_coefficients=c, **options)
        else:
            function = lambda q, k, v, c: reference.strip_dot_product_attention(
                q, k, v, strip_size=4, query_polynomial_coefficients=c, **options)
        actual = jax.jit(partial(_vjp, function))(operands, do)
        _assert_all(actual, _numpy_oracle(transform, operands, do), gpu=False)


@pytest.mark.parametrize("architecture", ("hopper", "blackwell"))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_fixed_linear_mosaic_entry_point_lowers_forward_and_all_gradients(monkeypatch, transform, architecture):
    kernels = {"hopper": hopper, "blackwell": blackwell}[architecture]
    monkeypatch.setattr(api, "_kernels", lambda: (architecture, kernels))
    monkeypatch.setattr(api, "_device_compute_capability", lambda: "9.0" if architecture == "hopper" else "10.0")
    q = jax.ShapeDtypeStruct((1, 2048, 2, 64), jnp.bfloat16)
    c = jax.ShapeDtypeStruct((1, 2048, 2, 2), jnp.float32)
    function = partial(api.transformed_query_polynomial_attention, transform=transform)
    jax.jit(function).trace(q, q, q, c).lower(lowering_platforms=("cuda",))
    jax.jit(lambda *a: jax.vjp(function, *a)[1](jnp.ones(a[2].shape, a[2].dtype))).trace(
        q, q, q, c).lower(lowering_platforms=("cuda",))


@pytest.fixture
def mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0"):
        pytest.skip("Requires an H100 or B200 allocation")


@pytest.mark.gpu
@pytest.mark.parametrize("time,spread", ((512, 1.), (512, 12.), (2048, 1.)))
@pytest.mark.parametrize("precision", common.PRECISIONS)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_mosaic_fixed_linear_output_and_all_gradients_match_numpy(mosaic_gpu, transform, precision, time, spread):
    operands, do = _inputs(time=time, dtype=jnp.bfloat16, spread=spread)
    function = partial(api.transformed_query_polynomial_attention, transform=transform, precision=precision)
    actual = jax.jit(partial(_vjp, function))(operands, do)
    assert actual[0].dtype == jnp.bfloat16 and actual[1][3].dtype == jnp.float32
    _assert_all(actual, _numpy_oracle(transform, operands, do), gpu=True)


@pytest.mark.gpu
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_mosaic_fixed_linear_extreme_tails_train_and_zero_has_correct_slope(mosaic_gpu, transform):
    time, width = 512, 64
    scores = jnp.resize(jnp.array([-80., -20., 0., 20., 80.], jnp.bfloat16), (time,))
    q = jnp.zeros((1, time, 1, width), jnp.bfloat16).at[0, :, 0, 0].set(scores)
    k = jnp.zeros_like(q).at[:, :, :, 0].set(8.)
    v, do = jnp.ones_like(q), jnp.ones_like(q)
    c = jnp.broadcast_to(jnp.array([0., 1.], jnp.float32), (1, time, 1, 2))
    operands = q, k, v, c
    function = partial(api.transformed_query_polynomial_attention, transform=transform)
    actual = jax.jit(partial(_vjp, function))(operands, do)
    expected = _numpy_oracle(transform, operands, do)
    _assert_all(actual, expected, gpu=True)
    # Every score group separately: large gradients around zero cannot hide a lost positive tail.
    for score in (-80., -20., 0., 20., 80.):
        selected = np.asarray(scores, np.float32) == score
        _assert_relative(actual[1][0][0, selected, 0], expected[1][0][0, selected, 0], .025,
                         f"{transform} dQ at score {score}")


@pytest.mark.gpu
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_mosaic_fixed_linear_is_causal_in_forward_and_all_gradients(mosaic_gpu, transform):
    operands, do = _inputs(time=512, dtype=jnp.bfloat16)
    prefix = 300
    function = partial(api.transformed_query_polynomial_attention, transform=transform)
    attend = jax.jit(partial(_vjp, function))
    do = do.at[:, prefix:].set(0)
    expected = attend(operands, do)
    changed = tuple(x.at[:, prefix:].add(2.) for x in operands)
    actual = attend(changed, do)
    np.testing.assert_array_equal(actual[0][:, :prefix], expected[0][:, :prefix])
    for got, want in zip(actual[1], expected[1], strict=True):
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got[:, prefix:], 0.)


@pytest.mark.gpu
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_mosaic_fixed_linear_repeated_forward_and_gradients_are_identical(mosaic_gpu, transform):
    operands, do = _inputs(time=2048, dtype=jnp.bfloat16)
    function = partial(api.transformed_query_polynomial_attention, transform=transform)
    attend = jax.jit(partial(_vjp, function))
    first = [np.asarray(x) for x in jax.tree.leaves(attend(operands, do))]
    for _ in range(3):
        for actual, expected in zip(jax.tree.leaves(attend(operands, do)), first, strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_mosaic_fixed_linear_nonfinite_causal_scores_fail_and_future_scores_are_masked(mosaic_gpu, transform):
    (q, k, v, c), _ = _inputs(time=512, dtype=jnp.bfloat16)
    function = jax.jit(partial(api.transformed_query_polynomial_attention, transform=transform))
    prefix = 300
    clean = function(q, k, v, c)
    future = function(q, k.at[:, prefix:].set(jnp.inf), v, c)
    np.testing.assert_array_equal(future[:, :prefix], clean[:, :prefix])
    assert bool(jnp.isnan(future[:, prefix:]).all())
    # Isolate one pair so both overflow signs are checked without changing any other causal score.
    q, k = q.at[:, :, :, 0].set(0), k.at[:, :, :, 0].set(0)
    clean = np.asarray(function(q, k, v, c), np.float32)
    for sign in (-1., 1.):
        poisoned = np.asarray(function(q.at[0, 300, 0, 0].set(2. ** 100),
                                       k.at[0, 37, 0, 0].set(sign * 2. ** 100), v, c), np.float32)
        invalid = np.zeros(poisoned.shape[:3], bool)
        invalid[0, 300, 0] = True
        np.testing.assert_array_equal(np.isnan(poisoned).all(-1), invalid)
        np.testing.assert_array_equal(poisoned[~invalid], clean[~invalid])
