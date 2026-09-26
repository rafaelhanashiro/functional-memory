"""Independent value/VJP gates for normalized learned direct and SOS polynomials.

CPU cases interpret the Pallas programs with full FP32 matmuls; GPU cases compile
the production Triton path. The dense oracle evaluates powers (and unexpanded
square factors) in FP64, independently of the fused Horner recurrences.
"""

from functools import partial
from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends import fused_normalized_polynomial as fused
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.learned_polynomial import normalized_learned_polynomial_attention
from scaffolds.transformer.attention.nonnegative_polynomial import taylor_square_factors
from scaffolds.transformer.attention.qk_norm import cosine_score_operands


FAMILIES = [("direct", degree) for degree in (1, 2, 3, 4, 8)] + [("sum_of_squares", degree) for degree in (2, 4, 8)]


def _inputs(shape, parameterization, degree, *, dtype=jnp.float32, seed=719):
    batch, time, heads, dq, dv = shape
    rng = np.random.default_rng(seed + degree)
    q, k = (jnp.asarray(0.7 * rng.normal(size=(batch, time, heads, dq)), dtype) for _ in range(2))
    v, cotangent = (jnp.asarray(rng.normal(size=(batch, time, heads, dv)), dtype) for _ in range(2))
    if parameterization == "direct":
        # Signed, head-specific parameters; the dominant constant keeps random
        # normalizers well-conditioned. Cancellation has a separate exact test.
        parameters = rng.uniform(-0.03, 0.03, size=(heads, degree + 1))
        parameters /= np.array([math.factorial(r) for r in range(degree + 1)])
        parameters[:, 0] = np.linspace(1.5, 2.5, heads)
        parameters[:, 1] = np.linspace(0.12, 0.23, heads)
    else:
        parameters = np.asarray(taylor_square_factors(degree))[None, :] * (
            1 + rng.uniform(-0.15, 0.15, size=(heads, degree + 1)))
    return (q, k, v, jnp.asarray(parameters, jnp.float32)), cotangent


def _dense_reference(q, k, v, parameters, *, parameterization, score_offset=None):
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    if score_offset is not None:
        scores = scores + score_offset
    degree = parameters.shape[-1] - 1
    parameters = jnp.broadcast_to(parameters, (q.shape[2], degree + 1))
    if parameterization == "direct":
        values = sum(parameters[:, order][None, :, None, None] * scores ** order
                     for order in range(degree + 1))
    else:
        half = degree // 2
        first = sum(parameters[:, order][None, :, None, None] * scores ** order
                    for order in range(half + 1))
        second = sum(parameters[:, half + 1 + order][None, :, None, None] * scores ** order
                     for order in range(half))
        values = first ** 2 + second ** 2
    values = jnp.where(jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool)), values, 0.0)
    return jnp.einsum("bhts,bshd->bthd", values / jnp.sum(values, axis=-1, keepdims=True), v)


def _blocked(q, k, v, parameters, *, parameterization):
    keyword = "polynomial_coefficients" if parameterization == "direct" else "polynomial_square_factors"
    return blocked_dot_product_attention(q, k, v, attention_degree=parameters.shape[-1] - 1,
                                         attention_normalization=True, query_block_size=32,
                                         **{keyword: parameters})


def _vjp(attend, operands, cotangent):
    output, pullback = jax.vjp(attend, *operands)
    return output, pullback(cotangent)


def _evaluate(attend, operands, cotangent):
    return jax.jit(partial(_vjp, attend))(operands, cotangent)


def _fp64(attend, operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _evaluate(attend, tuple(x.astype(jnp.float64) for x in operands), cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, np.float64), result)


def _assert_array(actual, reference, tolerance, name):
    actual, reference = (np.asarray(x, np.float64) for x in (actual, reference))
    assert actual.shape == reference.shape, name
    assert np.isfinite(actual).all() and np.isfinite(reference).all(), name
    if reference.ndim == 4:
        views = [(actual[b, :, h], reference[b, :, h])
                 for b in range(reference.shape[0]) for h in range(reference.shape[2])]
    elif reference.ndim == 2:
        views = list(zip(actual, reference, strict=True))
    else:
        views = [(actual, reference)]
    for head, (got, want) in enumerate(views):
        norm = np.linalg.norm(want)
        if norm == 0:
            np.testing.assert_array_equal(got, want, err_msg=name)
            continue
        error = np.linalg.norm(got - want) / norm
        assert error <= tolerance, f"{name}, view {head}: relative L2 {error:.6g} > {tolerance}"
        assert np.max(np.abs(got - want)) <= 3 * tolerance * np.max(np.abs(want)), f"{name}, view {head}: max error"


def _assert_result(actual, expected, *, output_tolerance=3e-5, gradient_tolerances=(2e-4,) * 4):
    _assert_array(actual[0], expected[0], output_tolerance, "output")
    for index, (got, want, tolerance) in enumerate(zip(actual[1], expected[1], gradient_tolerances, strict=True)):
        _assert_array(got, want, tolerance, f"gradient {index}")


@pytest.fixture(params=[pytest.param("interpret", id="cpu"), pytest.param("compiled", id="gpu", marks=pytest.mark.gpu)])
def kernel(request, monkeypatch):
    if request.param == "compiled":
        if jax.default_backend() != "gpu":
            pytest.skip("Compiled Pallas/Triton requires a CUDA allocation")
        return lambda parameterization: partial(fused.fused_normalized_polynomial_attention,
                                                 parameterization=parameterization)
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST))

    def interpreted_load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)

    monkeypatch.setattr(fused, "_load", interpreted_load)
    return lambda parameterization: partial(fused._attention, parameterization=parameterization, interpret=True)


@pytest.mark.parametrize("parameterization,degree", FAMILIES)
def test_values_and_all_gradients_against_independent_fp64(kernel, parameterization, degree):
    operands, cotangent = _inputs((2, 7, 3, 7, 5), parameterization, degree)
    reference = _fp64(partial(_dense_reference, parameterization=parameterization), operands, cotangent)
    for scale in (1.0, 1e-5):
        actual = _evaluate(kernel(parameterization), operands, cotangent * scale)
        expected = (reference[0], tuple(x * scale for x in reference[1]))
        _assert_result(actual, expected)
        assert all(x.dtype == jnp.float32 for x in (actual[0], *actual[1]))


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_odd_widths_and_sequence_tile_boundaries(kernel, parameterization):
    operands, cotangent = _inputs((1, 131, 2, 33, 65), parameterization, 4)
    actual = _evaluate(kernel(parameterization), operands, cotangent)
    expected = _fp64(partial(_dense_reference, parameterization=parameterization), operands, cotangent)
    _assert_result(actual, expected)


def _signed_inputs(case):
    time = 130
    q = jnp.zeros((1, time, 1, 1), jnp.float32).at[:, -1].set(1)
    scores = np.zeros(time, np.float32)
    if case == "zero_partial":
        scores[32:64] = -2.0
    elif case == "negative_partial":
        scores[:64] = -2.0
    elif case == "negative_final":
        scores = np.linspace(-3, -2, time, dtype=np.float32)
    elif case == "zero_kernel":
        scores[16:64] = -1.0
    elif case == "zero_final":
        scores[65:] = -2.0
    else:
        raise ValueError(case)
    rng = np.random.default_rng(56)
    v, cotangent = (jnp.asarray(rng.normal(size=q.shape), jnp.float32) for _ in range(2))
    # Even-degree unrestricted kernels can become signed too. Testing degree 4
    # with zero high coefficients also exercises their nonzero derivatives.
    return (q, jnp.asarray(scores).reshape(q.shape), v, jnp.asarray([[1., 1., 0., 0., 0.]], jnp.float32)), cotangent


@pytest.mark.parametrize("case", ("zero_partial", "negative_partial", "negative_final", "zero_kernel"))
def test_signed_partial_sums_and_negative_final_sums(kernel, case):
    operands, cotangent = _signed_inputs(case)
    actual = _evaluate(kernel("direct"), operands, cotangent)
    _assert_result(actual, _fp64(partial(_dense_reference, parameterization="direct"), operands, cotangent))


def test_exact_zero_final_sum_is_invalid_without_poisoning_earlier_rows(kernel):
    operands, _ = _signed_inputs("zero_final")
    output = jax.jit(kernel("direct"))(*operands)
    assert np.isfinite(np.asarray(output[:, :-1])).all()
    assert np.isnan(np.asarray(output[:, -1])).all()


def test_near_cancellation_preserves_large_signed_output_and_gradients(kernel):
    # All inputs and row sums are exactly representable dyadics. kappa describes
    # sensitivity to score rounding; it is not grounds for loosening every test.
    # The denominator is 2^-10 with kappa ~= 1.3e5, and must not be clipped.
    operands, _ = _signed_inputs("zero_final")
    q, k, _, _ = operands
    epsilon = 2.0 ** -10
    k = k.at[:, -1].add(epsilon)
    v = jnp.where(jnp.arange(130) < 65, 1.0, -1.0).reshape(q.shape)
    parameters = jnp.asarray([[1., 1.]], jnp.float32)
    cotangent = jnp.zeros_like(v).at[:, -1].set(1)
    operands = q, k, v, parameters
    weights = 1 + np.asarray(k, np.float64).reshape(-1)
    condition = np.sum(np.abs(weights)) / abs(np.sum(weights))
    assert 1e5 < condition < 2e5
    actual = _evaluate(kernel("direct"), operands, cotangent)
    expected = _fp64(partial(_dense_reference, parameterization="direct"), operands, cotangent)
    _assert_result(actual, expected, output_tolerance=2e-6, gradient_tolerances=(2e-5,) * 4)
    assert float(actual[0][0, -1, 0, 0]) == pytest.approx(130 / epsilon - 1, rel=2e-6)
    # A known perturbation of the denominator produces the predicted change;
    # this sensitivity is mathematical, separate from kernel implementation error.
    shifted = jax.jit(kernel("direct"))(q, k.at[:, -1].add(epsilon), v, parameters)
    assert float(shifted[0, -1, 0, 0]) == pytest.approx(130 / (2 * epsilon) - 1, rel=2e-6)


def test_sos_exact_roots_are_valid_but_zero_rows_are_invalid(kernel):
    q = jnp.ones((1, 4, 1, 1), jnp.float32)
    k = jnp.asarray([1., 3., 1., -1.], jnp.float32).reshape(q.shape)
    v = jnp.asarray([1., 2., 3., 4.], jnp.float32).reshape(q.shape)
    parameters = jnp.asarray([[-1., 1., 0.]], jnp.float32)  # (s - 1)^2
    output = jax.jit(kernel("sum_of_squares"))(q, k, v, parameters).reshape(-1)
    assert np.isnan(output[0])
    np.testing.assert_allclose(output[1:], [2., 2., 3.], rtol=1e-6)


def test_sos_all_zero_prefix_tiles_can_be_followed_by_positive_weights(kernel):
    q = jnp.zeros((1, 131, 1, 1), jnp.float32).at[:, -1].set(1)
    k = jnp.ones_like(q).at[:, -2].set(2).at[:, -1].set(3)
    positions = jnp.arange(131, dtype=jnp.float32).reshape(q.shape)
    operands = q, k, jnp.sin(positions), jnp.asarray([[-1., 1., 0.]], jnp.float32)
    cotangent = jnp.cos(positions)
    actual = _evaluate(kernel("sum_of_squares"), operands, cotangent)
    expected = _fp64(partial(_dense_reference, parameterization="sum_of_squares"), operands, cotangent)
    _assert_result(actual, expected)


def test_normalized_coefficient_gradients_do_not_form_overflowing_raw_powers(kernel):
    # s^16 overflows FP32 near 1000, but c16*s^16 and dp*s^16 are finite.
    # Starting the gradient recurrence at dp preserves that normalization.
    q = jnp.asarray([0.001, 0.002, 1.], jnp.float32).reshape(1, 3, 1, 1)
    k = jnp.asarray([900., 970., 1030.], jnp.float32).reshape(q.shape)
    v = jnp.asarray([1., -2., 4.], jnp.float32).reshape(q.shape)
    parameters = jnp.asarray([[1 / math.factorial(r) for r in range(17)]], jnp.float32)
    operands, cotangent = (q, k, v, parameters), jnp.ones_like(v)
    actual = _evaluate(kernel("direct"), operands, cotangent)
    reference = _fp64(partial(_dense_reference, parameterization="direct"), operands, cotangent)
    _assert_result((actual[0], actual[1][:3]), (reference[0], reference[1][:3]),
                   output_tolerance=3e-5, gradient_tolerances=(3e-4,) * 3)
    # dc16 subtracts contributions about 1852 times larger than their sum.
    # Bound its absolute rounding error by the contribution magnitudes, keeping
    # the ordinary relative bound for the better-conditioned coefficients.
    q64, k64, v64, coefficients = (np.asarray(x, np.float64) for x in operands)
    scores = np.einsum("bthd,bshd->bhts", q64, k64)
    causal = np.tril(np.ones((3, 3), bool))
    values = sum(coefficients[:, r][None, :, None, None] * scores ** r for r in range(17))
    normalizer = np.sum(np.where(causal, values, 0), axis=-1, keepdims=True)
    dp = (np.einsum("bthd,bshd->bhts", np.asarray(cotangent, np.float64), v64)
          - np.einsum("bthd,bthd->bht", np.asarray(cotangent, np.float64), reference[0])[..., None]) / normalizer
    magnitudes = np.stack([np.sum(np.where(causal, np.abs(dp * scores ** r), 0), axis=(0, 2, 3))
                           for r in range(17)], axis=-1)
    condition = magnitudes[0, -1] / abs(reference[1][3][0, -1])
    assert 1000 < condition < 3000
    bound = np.maximum(3e-4 * np.abs(reference[1][3]), 2 * np.finfo(np.float32).eps * magnitudes)
    error = np.abs(np.asarray(actual[1][3], np.float64) - reference[1][3])
    assert np.all(error <= bound), f"coefficient errors {error} exceed conditioning bounds {bound}"


def test_normalized_sos_score_gradient_does_not_form_overflowing_raw_derivative(kernel):
    # p=(2+1e38*s)^2 has p'(0)=4e38, which overflows FP32, but
    # (dO.V-dO.Y)/rowsum scales it down before the derivative is needed.
    q = jnp.zeros((1, 2, 1, 1), jnp.float32)
    k = jnp.asarray([0., 0.1], jnp.float32).reshape(q.shape)
    v = jnp.asarray([0., 1.], jnp.float32).reshape(q.shape)
    parameters = jnp.asarray([[2., 1e38, 0.]], jnp.float32)
    cotangent = jnp.asarray([0., 1.], jnp.float32).reshape(q.shape)
    actual = _evaluate(kernel("sum_of_squares"), (q, k, v, parameters), cotangent)
    np.testing.assert_array_equal(actual[0].reshape(-1), [0., 0.5])
    np.testing.assert_allclose(np.asarray(actual[1][0]).reshape(-1), [0., 2.5e36], rtol=2e-6, atol=0)
    np.testing.assert_array_equal(actual[1][1], jnp.zeros_like(k))
    np.testing.assert_array_equal(actual[1][2].reshape(-1), [0.5, 0.5])
    np.testing.assert_array_equal(actual[1][3], jnp.zeros_like(parameters))


def test_normalized_direct_score_gradient_does_not_form_overflowing_raw_derivative(kernel):
    # p'(1)=6e38 overflows FP32 despite finite values and row sums. The
    # nonzero exact Q/K/parameter gradients are smaller than 1e-37 here;
    # CUDA can flush these subnormals, but it must not produce NaN or Inf.
    q = jnp.asarray([0., 1.], jnp.float32).reshape(1, 2, 1, 1)
    parameters = jnp.asarray([[1., 0., 3e38]], jnp.float32)
    actual = _evaluate(kernel("direct"), (q, q, q, parameters), q)
    np.testing.assert_array_equal(actual[0].reshape(-1), [0., 1.])
    for gradient in (actual[1][0], actual[1][1], actual[1][3]):
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-37)
    np.testing.assert_allclose(actual[1][2].reshape(-1), [0., 1.], rtol=2e-6, atol=1e-37)


def test_sos_near_roots_use_factors_without_expansion_or_clipping(kernel):
    # The expanded FP32 polynomial can turn negative around its root at 20.
    # Scores within 0.02 of the root expose one-ULP tensor-core score errors.
    q = jnp.ones((1, 67, 1, 1), jnp.float32)
    scores = np.concatenate(([19.5], np.linspace(19.98, 20.02, 66))).astype(np.float32)
    k = jnp.asarray(scores).reshape(q.shape)
    t = jnp.arange(67, dtype=jnp.float32)
    v, cotangent = jnp.sin(t).reshape(q.shape), jnp.cos(t).reshape(q.shape)
    parameters = jnp.asarray([[-60 / 64, -17 / 64, 1 / 64, -20 / 1024, 1 / 1024]], jnp.float32)
    operands = q, k, v, parameters
    actual = _evaluate(kernel("sum_of_squares"), operands, cotangent)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        operands64 = tuple(x.astype(jnp.float64) for x in operands)

        def oracle(offsets):
            attend = partial(_dense_reference, parameterization="sum_of_squares", score_offset=offsets)
            return _vjp(attend, operands64, cotangent.astype(jnp.float64))

        offsets = jnp.zeros((67,), jnp.float64)
        expected = oracle(offsets)
        sensitivities = jax.jit(jax.jacfwd(oracle))(offsets)
        # Q is exactly one, so every query's score error for a key is the same.
        # Differentiate outputs and their VJPs with respect to independent score
        # offsets. This gives an input-specific bound instead of relaxing every
        # near-zero factor comparison. A factor of two covers the local
        # linearization remainder; separate budgets cover FP32 evaluation.
        ulps = jnp.asarray(np.spacing(scores), jnp.float64)
        envelopes = jax.tree.map(lambda derivative: jnp.sum(jnp.abs(derivative) * ulps, axis=-1), sensitivities)
        expected, envelopes = jax.tree.map(lambda x: np.asarray(x, np.float64), (expected, envelopes))
    for index, (got, want, envelope) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected),
                                                       jax.tree.leaves(envelopes), strict=True)):
        got = np.asarray(got, np.float64)
        evaluation_budget = 2e-5 if index == 0 else 1e-4
        bound = 2 * envelope + evaluation_budget * np.max(np.abs(want))
        assert np.isfinite(got).all(), f"nonfinite near-root result {index}"
        assert np.all(np.abs(got - want) <= bound), f"near-root result {index} exceeds one-ULP conditioning bound"


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_masked_overflow_and_invalid_causal_values(kernel, parameterization):
    parameters = jnp.asarray([[1., 1., 0.5]], jnp.float32)
    attend = jax.jit(kernel(parameterization))
    q = jnp.asarray([1., 0.], jnp.float32).reshape(1, 2, 1, 1)
    k = jnp.asarray([0., 1e30], jnp.float32).reshape(q.shape)
    v = jnp.asarray([3., 7.], jnp.float32).reshape(q.shape)
    np.testing.assert_array_equal(attend(q, k, v, parameters).reshape(-1), [3., 5.])
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    v = jnp.asarray([3., 7., 11.], jnp.float32).reshape(q.shape)
    for bad_score in (np.inf, -np.inf, np.nan, 1e30):
        k = jnp.asarray([0., 0., bad_score], jnp.float32).reshape(q.shape)
        output = attend(q, k, v, parameters)
        np.testing.assert_allclose(output[:, :2].reshape(-1), [3., 5.], rtol=1e-6)
        assert np.isnan(np.asarray(output[:, -1])).all()
    for bad_parameter in (np.inf, np.nan):
        output = attend(q, jnp.zeros_like(q), v, parameters.at[:, 0].set(bad_parameter))
        assert np.isnan(np.asarray(output)).all()


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_shared_parameter_wrapper_reduces_all_heads(kernel, monkeypatch, parameterization):
    operands, cotangent = _inputs((2, 7, 3, 7, 5), parameterization, 4)
    attend = kernel(parameterization)
    monkeypatch.setattr(fused, "fused_normalized_polynomial_attention",
                        lambda q, k, v, parameters, **unused: attend(q, k, v, parameters))
    keyword = "polynomial_coefficients" if parameterization == "direct" else "polynomial_square_factors"
    shared = operands[3][0]
    broadcast = jnp.broadcast_to(shared, operands[3].shape)
    expected = _evaluate(attend, (*operands[:3], broadcast), cotangent)

    def wrapper(q, k, v, parameters):
        return normalized_learned_polynomial_attention(q, k, v, attention_degree=4, backend="fused",
                                                       **{keyword: parameters})

    actual = _evaluate(wrapper, (*operands[:3], shared), cotangent)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1][3], expected[1][3].sum(axis=0), rtol=2e-6, atol=1e-7)
    _assert_result(actual, _fp64(partial(_dense_reference, parameterization=parameterization),
                                 (*operands[:3], shared), cotangent))


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_scale_invariance_and_parameter_gradient(kernel, parameterization):
    operands, cotangent = _inputs((1, 17, 2, 7, 5), parameterization, 4)
    attend = kernel(parameterization)
    output, grads = _evaluate(attend, operands, cotangent)
    for scale in (-4.0, 0.25):
        scaled_output, scaled_grads = _evaluate(attend, (*operands[:3], operands[3] * scale), cotangent)
        _assert_array(scaled_output, output, 3e-6, "scale invariant output")
        for index in range(3):
            _assert_array(scaled_grads[index], grads[index], 2e-5, f"scale invariant gradient {index}")
        _assert_array(scaled_grads[3] * scale, grads[3], 2e-5, "inverse scaled parameter gradient")
    products = np.asarray(grads[3]) * np.asarray(operands[3])
    np.testing.assert_allclose(products.sum(axis=-1), 0, rtol=0,
                               atol=2e-5 * np.max(np.sum(np.abs(products), axis=-1)))


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_future_tokens_do_not_change_prefix_or_parameter_gradient(kernel, parameterization):
    operands, cotangent = _inputs((1, 67, 2, 7, 5), parameterization, 4)
    cotangent = cotangent.at[:, 33:].set(0)
    attend = kernel(parameterization)
    original = _evaluate(attend, operands, cotangent)
    q, k, v, parameters = operands
    changed = q.at[:, 33:].multiply(-2), k.at[:, 33:].multiply(3), v.at[:, 33:].add(2), parameters
    result = _evaluate(attend, changed, cotangent)
    np.testing.assert_array_equal(result[0][:, :33], original[0][:, :33])
    np.testing.assert_array_equal(result[1][3], original[1][3])
    for value in result[1][:3]:
        np.testing.assert_array_equal(value[:, 33:], jnp.zeros_like(value[:, 33:]))


@pytest.fixture
def require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")


@pytest.mark.gpu
@pytest.mark.parametrize("parameterization,degree", [("direct", d) for d in (1, 2, 3, 4)] + [("sum_of_squares", 2), ("sum_of_squares", 4)])
def test_bf16_full_context_with_qk_norm_and_all_gradients(require_gpu, parameterization, degree):
    operands, cotangent = _inputs((1, 2048, 2, 64, 64), parameterization, degree, dtype=jnp.bfloat16, seed=75)
    log_scale = jnp.asarray([[math.log(8.0)], [math.log(6.0)]], jnp.float32)

    def compose(attend, q, k, v, parameters, scale):
        q, k = cosine_score_operands(q, k, scale, eps=1e-6)
        return attend(q, k, v, parameters)

    attend = partial(fused.fused_normalized_polynomial_attention, parameterization=parameterization)
    actual = _evaluate(partial(compose, attend), (*operands, log_scale), cotangent)
    expected = _evaluate(partial(compose, partial(_blocked, parameterization=parameterization)),
                         (*operands, log_scale), cotangent)
    _assert_result(actual, expected, output_tolerance=0.006,
                   gradient_tolerances=(0.025, 0.025, 0.015, 0.012, 0.025))
    assert actual[0].dtype == jnp.bfloat16
    assert all(x.dtype == jnp.bfloat16 for x in actual[1][:3])
    assert all(x.dtype == jnp.float32 for x in actual[1][3:])
    # Independent FP64 checks retain the BF16-rounded normalized operands;
    # normalization's own VJP is checked against the blocked full-context path.
    normalized_q, normalized_k = cosine_score_operands(*operands[:2], log_scale, eps=1e-6)
    prefix = normalized_q[:, :129], normalized_k[:, :129], operands[2][:, :129], operands[3]
    prefix_cotangent = cotangent[:, :129]
    prefix_actual = _evaluate(attend, prefix, prefix_cotangent)
    reference = _fp64(partial(_dense_reference, parameterization=parameterization), prefix, prefix_cotangent)
    _assert_result(prefix_actual, reference, output_tolerance=0.006,
                   gradient_tolerances=(0.025, 0.025, 0.015, 0.012))


@pytest.mark.gpu
@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_bf16_odd_widths_match_fp64(require_gpu, parameterization):
    operands, cotangent = _inputs((2, 131, 3, 33, 17), parameterization, 4, dtype=jnp.bfloat16)
    attend = partial(fused.fused_normalized_polynomial_attention, parameterization=parameterization)
    _assert_result(_evaluate(attend, operands, cotangent),
                   _fp64(partial(_dense_reference, parameterization=parameterization), operands, cotangent),
                   output_tolerance=0.006, gradient_tolerances=(0.025, 0.025, 0.015, 0.012))


@pytest.mark.gpu
@pytest.mark.parametrize("parameterization,sharing", [("direct", "per_head"), ("sum_of_squares", "global")])
def test_transformer_training_and_checkpoint_resume(require_gpu, tmp_path, parameterization, sharing):
    from scaffolds.transformer.tests.test_training import tiny_config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
    from scaffolds.transformer.training.state import create_state

    config = tiny_config(tmp_path / "full")
    config.model.attention_degree = 4
    config.model.attention_coefficients = "learned"
    config.model.attention_normalization = True
    config.model.attention_coefficient_parameterization = parameterization
    config.model.attention_coefficient_sharing = sharing
    config.model.attention_backend = "fused"
    config.model.qk_norm = True
    config.model.qk_norm_initial_scale = 0.3
    config.train.dtype = "bfloat16"
    config.optim.total_steps = 2
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-3, 1e-4
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.asarray(np.arange(20).reshape(4, 5) % 16)
    loss, grads = loop.microbatch_loss_and_grads(initial, batch, jax.random.PRNGKey(1))
    assert np.isfinite(loss)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))
    parameter = "polynomial_coefficients" if parameterization == "direct" else "polynomial_square_factors"
    assert np.linalg.norm(np.asarray(grads[parameter][...])) > 0
    assert np.linalg.norm(np.asarray(grads["qk_norm_log_scale"][...])) > 0

    full = loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert full["attention_backend"] == resumed["attention_backend"] == "fused"
    assert full["attention_diagnostics"]["valid"] and resumed["attention_diagnostics"]["valid"]
    first_audit, resume_audit = interrupted["checkpoint_audit"], resumed["checkpoint_audit"]
    assert first_audit["final_state_sha256"] == first_audit["checkpoint_state_sha256"]
    assert first_audit["final_state_sha256"] == resume_audit["start_state_sha256"]
    assert first_audit["next_update"] == resume_audit["first_update"]
    assert resumed["restored_step"] == 1
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
            assert step == 2
            states.append(state)
        finally:
            manager.close()
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    for name in (parameter, "qk_norm_log_scale"):
        trained, start = (np.asarray(state.params[name][...]) for state in (states[0], initial))
        assert np.isfinite(trained).all()
        assert np.any(trained != start), f"{name} did not update"
