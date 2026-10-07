"""Independent value/VJP gates for normalized learned direct, SOS, and rational polynomials.

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


# Families name a parameterization and, for rational kernels, the denominator:
# "rational" has a direct denominator of the numerator's degree, "rational_linear" a
# direct linear one, "rational_sos2" a nonnegative quadratic (1 + u s)^2 + (w s)^2,
# "rational_sos8" a nonnegative degree-8 (1 + sum u_r s^r)^2 + (sum w_r s^r)^2. The
# "_c" families learn the denominator's constant: b_0 + b_1 s, and q1(s)^2 + q2^2.
FAMILY_OPTIONS = {
    "direct": dict(parameterization="direct"),
    "sum_of_squares": dict(parameterization="sum_of_squares"),
    "rational": dict(parameterization="rational"),
    "rational_linear": dict(parameterization="rational", denominator_degree=1, denominator_form="direct"),
    "rational_sos2": dict(parameterization="rational", denominator_degree=2, denominator_form="sum_of_squares"),
    "rational_sos8": dict(parameterization="rational", denominator_degree=8, denominator_form="sum_of_squares"),
    "rational_linear_c": dict(parameterization="rational", denominator_degree=1, denominator_form="direct",
                              denominator_constant="learned"),
    "rational_sos2_c": dict(parameterization="rational", denominator_degree=2, denominator_form="sum_of_squares",
                            denominator_constant="learned"),
}
RATIONAL_FAMILIES = ("rational_linear", "rational_sos2", "rational_sos8", "rational_linear_c", "rational_sos2_c")
FAMILIES = ([("direct", degree) for degree in (1, 2, 3, 4, 8)] + [("sum_of_squares", degree) for degree in (2, 4, 8)]
            + [("rational", degree) for degree in (1, 2, 3, 4, 8)]
            + [("rational_linear", degree) for degree in (1, 2, 8)]
            + [("rational_sos2", degree) for degree in (1, 4, 8)]
            + [("rational_sos8", degree) for degree in (2, 8)]
            + [("rational_linear_c", degree) for degree in (1, 8)]
            + [("rational_sos2_c", degree) for degree in (2, 8)])
PARAMETERIZATIONS = ("direct", "sum_of_squares", "rational", *RATIONAL_FAMILIES)


def _denominator_count(parameters, family):
    options = FAMILY_OPTIONS[family]
    if options["parameterization"] != "rational":
        return 0
    if "denominator_degree" not in options:
        return (parameters.shape[-1] - 1) // 2
    return options["denominator_degree"] + (options.get("denominator_constant") == "learned")


def _degree(parameters, family):
    return parameters.shape[-1] - _denominator_count(parameters, family) - 1


def _inputs(shape, parameterization, degree, *, dtype=jnp.float32, seed=719):
    batch, time, heads, dq, dv = shape
    rng = np.random.default_rng(seed + degree)
    q, k = (jnp.asarray(0.7 * rng.normal(size=(batch, time, heads, dq)), dtype) for _ in range(2))
    v, cotangent = (jnp.asarray(rng.normal(size=(batch, time, heads, dv)), dtype) for _ in range(2))
    factorials = np.array([math.factorial(r) for r in range(degree + 1)])
    if parameterization == "direct":
        # Signed, head-specific parameters; the dominant constant keeps random
        # normalizers well-conditioned. Cancellation has a separate exact test.
        parameters = rng.uniform(-0.03, 0.03, size=(heads, degree + 1))
        parameters /= factorials
        parameters[:, 0] = np.linspace(1.5, 2.5, heads)
        parameters[:, 1] = np.linspace(0.12, 0.23, heads)
    elif parameterization == "rational":
        # A signed numerator of order one and a denominator 1 + small signed terms,
        # whose implicit constant keeps the row sums well-conditioned.
        numerator = rng.uniform(-0.5, 0.5, size=(heads, degree + 1)) / factorials
        numerator[:, 1] = np.linspace(0.8, 1.2, heads)
        denominator = rng.uniform(-0.03, 0.03, size=(heads, degree)) / factorials[1:]
        parameters = np.concatenate([numerator, denominator], axis=-1)
    elif parameterization in RATIONAL_FAMILIES:
        numerator = rng.uniform(-0.5, 0.5, size=(heads, degree + 1)) / factorials
        numerator[:, 1] = np.linspace(0.8, 1.2, heads)
        if parameterization == "rational_linear":
            denominator = rng.uniform(-0.03, 0.03, size=(heads, 1))
        elif parameterization == "rational_linear_c":
            denominator = np.stack([np.linspace(0.8, 1.2, heads), rng.uniform(-0.03, 0.03, heads)], axis=-1)
        elif parameterization == "rational_sos2_c":
            # q1 = c_0 + c_1 s with c_0 near 0.7, q2 = d_0 near 0.5.
            denominator = np.stack([np.linspace(0.6, 0.8, heads), rng.uniform(-0.1, 0.1, heads),
                                    np.linspace(0.4, 0.6, heads)], axis=-1)
        else:
            # u_r near 0 and w_r of order 0.2, each divided by r! like Taylor coefficients.
            half = FAMILY_OPTIONS[parameterization]["denominator_degree"] // 2
            scale = np.array([math.factorial(order) for order in range(1, half + 1)])
            u = rng.uniform(-0.1, 0.1, size=(heads, half)) / scale
            denominator = np.concatenate([u, rng.uniform(0.1, 0.3, size=(heads, half)) / scale], axis=-1)
        parameters = np.concatenate([numerator, denominator], axis=-1)
    else:
        parameters = np.asarray(taylor_square_factors(degree))[None, :] * (
            1 + rng.uniform(-0.15, 0.15, size=(heads, degree + 1)))
    return (q, k, v, jnp.asarray(parameters, jnp.float32)), cotangent


def _dense_reference(q, k, v, parameters, *, parameterization, score_offset=None):
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    if score_offset is not None:
        scores = scores + score_offset
    degree = _degree(parameters, parameterization)
    parameters = jnp.broadcast_to(parameters, (q.shape[2], parameters.shape[-1]))
    if parameterization in RATIONAL_FAMILIES:
        values = sum(parameters[:, order][None, :, None, None] * scores ** order
                     for order in range(degree + 1))
        rest = parameters[:, degree + 1:]
        if parameterization == "rational_linear_c":
            denominators = rest[:, 0][None, :, None, None] + rest[:, 1][None, :, None, None] * scores
        elif parameterization == "rational_sos2_c":
            denominators = ((rest[:, 0][None, :, None, None] + rest[:, 1][None, :, None, None] * scores) ** 2
                            + rest[:, 2][None, :, None, None] ** 2)
        elif parameterization == "rational_linear":
            denominators = 1 + rest[:, 0][None, :, None, None] * scores
        else:
            half = rest.shape[-1] // 2
            first = 1 + sum(rest[:, order - 1][None, :, None, None] * scores ** order for order in range(1, half + 1))
            second = sum(rest[:, half + order - 1][None, :, None, None] * scores ** order
                         for order in range(1, half + 1))
            denominators = first ** 2 + second ** 2
    elif parameterization == "direct":
        values = sum(parameters[:, order][None, :, None, None] * scores ** order
                     for order in range(degree + 1))
        denominators = values
    elif parameterization == "rational":
        values = sum(parameters[:, order][None, :, None, None] * scores ** order
                     for order in range(degree + 1))
        denominators = 1 + sum(parameters[:, degree + order][None, :, None, None] * scores ** order
                               for order in range(1, degree + 1))
    else:
        half = degree // 2
        first = sum(parameters[:, order][None, :, None, None] * scores ** order
                    for order in range(half + 1))
        second = sum(parameters[:, half + 1 + order][None, :, None, None] * scores ** order
                     for order in range(half))
        values = first ** 2 + second ** 2
        denominators = values
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    values, denominators = jnp.where(causal, values, 0.0), jnp.where(causal, denominators, 0.0)
    return jnp.einsum("bhts,bshd->bthd", values / jnp.sum(denominators, axis=-1, keepdims=True), v)


def _learned(parameters, parameterization):
    """Keyword arguments naming the learned arrays of one parameterization."""
    degree = _degree(parameters, parameterization)
    if parameterization.startswith("rational"):
        options = FAMILY_OPTIONS[parameterization]
        key = {("direct", "fixed"): "polynomial_denominator_coefficients",
               ("direct", "learned"): "polynomial_denominator_coefficients_with_constant",
               ("sum_of_squares", "fixed"): "polynomial_denominator_square_factors",
               ("sum_of_squares", "learned"): "polynomial_denominator_square_factors_with_constant"}[
            (options.get("denominator_form", "direct"), options.get("denominator_constant", "fixed"))]
        return {"polynomial_coefficients": parameters[..., :degree + 1], key: parameters[..., degree + 1:]}
    keyword = "polynomial_coefficients" if parameterization == "direct" else "polynomial_square_factors"
    return {keyword: parameters}


def _blocked(q, k, v, parameters, *, parameterization):
    return blocked_dot_product_attention(q, k, v, attention_degree=_degree(parameters, parameterization),
                                         attention_normalization=True, query_block_size=32,
                                         **_learned(parameters, parameterization))


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
        return lambda family: partial(fused.fused_normalized_polynomial_attention, **FAMILY_OPTIONS[family])
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST))

    def interpreted_load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)

    monkeypatch.setattr(fused, "_load", interpreted_load)
    return lambda family: partial(fused.fused_normalized_polynomial_attention, interpret=True,
                                  **FAMILY_OPTIONS[family])


@pytest.mark.parametrize("parameterization,degree", FAMILIES)
def test_values_and_all_gradients_against_independent_fp64(kernel, parameterization, degree):
    operands, cotangent = _inputs((2, 7, 3, 7, 5), parameterization, degree)
    reference = _fp64(partial(_dense_reference, parameterization=parameterization), operands, cotangent)
    for scale in (1.0, 1e-5):
        actual = _evaluate(kernel(parameterization), operands, cotangent * scale)
        expected = (reference[0], tuple(x * scale for x in reference[1]))
        _assert_result(actual, expected)
        assert all(x.dtype == jnp.float32 for x in (actual[0], *actual[1]))


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
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


def test_denominator_options_are_validated():
    q = jnp.zeros((1, 4, 2, 4), jnp.float32)
    attend = partial(fused.fused_normalized_polynomial_attention, q, q, q)
    with pytest.raises(ValueError, match="rational parameterization only"):
        attend(jnp.zeros((2, 3), jnp.float32), parameterization="direct", denominator_degree=1)
    with pytest.raises(ValueError, match="even degree"):
        attend(jnp.zeros((2, 6), jnp.float32), parameterization="rational", denominator_degree=3,
               denominator_form="sum_of_squares")
    with pytest.raises(ValueError, match="numerator degree >= 1"):
        attend(jnp.zeros((2, 3), jnp.float32), parameterization="rational", denominator_degree=2)
    with pytest.raises(ValueError, match="explicit denominator_degree"):
        attend(jnp.zeros((2, 4), jnp.float32), parameterization="rational", denominator_constant="learned")
    with pytest.raises(ValueError, match="denominator_constant"):
        attend(jnp.zeros((2, 4), jnp.float32), parameterization="rational", denominator_degree=1,
               denominator_constant="free")
    with pytest.raises(ValueError, match="denominator_form"):
        attend(jnp.zeros((2, 3), jnp.float32), parameterization="rational", denominator_degree=1,
               denominator_form="positive")


def test_rational_zero_denominator_sums_are_invalid_and_zero_numerators_are_valid(kernel):
    # p_a(s) = s and p_b(s) = 1 - s on key scores 1, 3, 5, -2: denominators 0, -2, -4, 3
    # sum to 0, -2, -6, -3. Row 0 is poisoned; the negative sums are valid.
    q = jnp.ones((1, 4, 1, 1), jnp.float32)
    k = jnp.asarray([1., 3., 5., -2.], jnp.float32).reshape(q.shape)
    v = jnp.asarray([1., 2., 3., 4.], jnp.float32).reshape(q.shape)
    output = np.asarray(jax.jit(kernel("rational"))(q, k, v, jnp.asarray([[0., 1., -1.]], jnp.float32))).reshape(-1)
    assert np.isnan(output[0])
    np.testing.assert_allclose(output[1:], [(1 + 6) / -2, (1 + 6 + 15) / -6, (1 + 6 + 15 - 8) / -3], rtol=1e-6)
    zero_numerator = np.asarray(jax.jit(kernel("rational"))(q, k, v, jnp.asarray([[0., 0., -1.]], jnp.float32))).reshape(-1)
    assert np.isnan(zero_numerator[0])
    np.testing.assert_array_equal(zero_numerator[1:], [0., 0., 0.])


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


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_masked_overflow_and_invalid_causal_values(kernel, parameterization):
    # Rational kernels use the same numerator with a constant denominator.
    denominator = {"rational": [0., 0.], "rational_linear": [0.], "rational_sos2": [0., 0.],
                   "rational_sos8": [0.] * 8, "rational_linear_c": [1., 0.],
                   "rational_sos2_c": [1., 0., 0.]}.get(parameterization, [])
    parameters = jnp.asarray([[1., 1., 0.5, *denominator]], jnp.float32)
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


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_shared_parameter_wrapper_reduces_all_heads(kernel, monkeypatch, parameterization):
    operands, cotangent = _inputs((2, 7, 3, 7, 5), parameterization, 4)
    attend = kernel(parameterization)
    monkeypatch.setattr(fused, "fused_normalized_polynomial_attention",
                        lambda q, k, v, parameters, **unused: attend(q, k, v, parameters))
    shared = operands[3][0]
    broadcast = jnp.broadcast_to(shared, operands[3].shape)
    expected = _evaluate(attend, (*operands[:3], broadcast), cotangent)

    def wrapper(q, k, v, parameters):
        return normalized_learned_polynomial_attention(q, k, v, attention_degree=4, backend="fused",
                                                       **_learned(parameters, parameterization))

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


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
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
@pytest.mark.parametrize("parameterization,degree", ([("direct", d) for d in (1, 2, 3, 4)]
                                                      + [("sum_of_squares", 2), ("sum_of_squares", 4)]
                                                      + [("rational", 4), ("rational", 8)]
                                                      + [("rational_linear", 8), ("rational_sos2", 8),
                                                         ("rational_sos8", 8), ("rational_linear_c", 8),
                                                         ("rational_sos2_c", 8)]))
def test_bf16_full_context_with_qk_norm_and_all_gradients(require_gpu, parameterization, degree):
    operands, cotangent = _inputs((1, 2048, 2, 64, 64), parameterization, degree, dtype=jnp.bfloat16, seed=75)
    log_scale = jnp.asarray([[math.log(8.0)], [math.log(6.0)]], jnp.float32)

    def compose(attend, q, k, v, parameters, scale):
        q, k = cosine_score_operands(q, k, scale, eps=1e-6)
        return attend(q, k, v, parameters)

    attend = partial(fused.fused_normalized_polynomial_attention, **FAMILY_OPTIONS[parameterization])
    actual = _evaluate(partial(compose, attend), (*operands, log_scale), cotangent)
    expected = _evaluate(partial(compose, partial(_blocked, parameterization=parameterization)),
                         (*operands, log_scale), cotangent)
    _assert_result((actual[0], actual[1][:4]), (expected[0], expected[1][:4]), output_tolerance=0.006,
                   gradient_tolerances=(0.025, 0.025, 0.015, 0.012))
    # The log-scale gradient sums ds * s over every causal pair and can nearly cancel
    # for a head; BF16 operands then leave both BF16 paths far from its exact value
    # while they agree with each other (rational_sos2 at degree 8: about 70% for one
    # head, against FP64). Compare both with an FP32 evaluation of the blocked
    # reference instead: the fused error must be within the old 2.5% of it, or no
    # larger than the reference's own error.
    with jax.default_matmul_precision("highest"):
        exact = _evaluate(partial(compose, partial(_blocked, parameterization=parameterization)),
                          (*(x.astype(jnp.float32) for x in operands), log_scale), cotangent.astype(jnp.float32))
    exact_scale = np.asarray(exact[1][4], np.float64)
    fused_error = np.abs(np.asarray(actual[1][4], np.float64) - exact_scale)
    reference_error = np.abs(np.asarray(expected[1][4], np.float64) - exact_scale)
    assert np.all(np.isfinite(fused_error))
    assert np.all(fused_error <= np.maximum(0.025 * np.abs(exact_scale), 1.25 * reference_error)), (
        f"log-scale gradient errors {fused_error.ravel()} against the reference's {reference_error.ravel()}")
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
@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_bf16_odd_widths_match_fp64(require_gpu, parameterization):
    operands, cotangent = _inputs((2, 131, 3, 33, 17), parameterization, 4, dtype=jnp.bfloat16)
    attend = partial(fused.fused_normalized_polynomial_attention, **FAMILY_OPTIONS[parameterization])
    _assert_result(_evaluate(attend, operands, cotangent),
                   _fp64(partial(_dense_reference, parameterization=parameterization), operands, cotangent),
                   output_tolerance=0.006, gradient_tolerances=(0.025, 0.025, 0.015, 0.012))


@pytest.mark.gpu
@pytest.mark.parametrize("parameterization,sharing", [("direct", "per_head"), ("sum_of_squares", "global"),
                                                       ("rational", "per_head"), ("rational_sos2", "per_head")])
def test_transformer_training_and_checkpoint_resume(require_gpu, tmp_path, parameterization, sharing):
    from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter
    from scaffolds.transformer.tests.test_training import tiny_config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
    from scaffolds.transformer.training.state import create_state

    config = tiny_config(tmp_path / "full")
    config.model.attention_degree = 4
    config.model.attention_coefficients = "learned"
    config.model.attention_normalization = True
    options = FAMILY_OPTIONS[parameterization]
    config.model.attention_coefficient_parameterization = options["parameterization"]
    if parameterization == "rational_sos2":
        config.model.attention_denominator_degree = 2
        config.model.attention_denominator_parameterization = "sum_of_squares"
    config.model.attention_coefficient_sharing = sharing
    config.model.attention_backend = "fused"
    config.model.qk_norm = True
    config.model.qk_norm_initial_scale = 0.3
    config.train.dtype = "bfloat16"
    config.optim.total_steps = 2
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-3, 1e-4
    # Exact independent replay needs deterministic gradient accumulation. Avoid
    # repeated embedding indices within a microbatch: GPU scatter-add into a
    # BF16 embedding gradient can otherwise depend on atomic update order.
    # Distinct successive batches still detect a wrong resumed data cursor.
    sequences = np.arange(40, dtype=np.int32).reshape(8, 5) % 16
    for microbatch in sequences.reshape(-1, config.train.micro_batch_size, 5):
        inputs = microbatch[:, :-1]
        assert np.unique(inputs).size == inputs.size
    assert not np.array_equal(sequences[:4], sequences[4:])
    config.data.source = "packed"
    for split, rows in (("train", sequences), ("eval", sequences[:4])):
        writer = PackedSequenceWriter(tmp_path / split, sequence_length=5,
                                       num_sequences=len(rows), shard_sequences=4)
        writer.write(rows)
        setattr(config.data, f"{split}_manifest", str(writer.finish()))
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.asarray(np.arange(20).reshape(4, 5) % 16)
    loss, grads = loop.microbatch_loss_and_grads(initial, batch, jax.random.PRNGKey(1))
    assert np.isfinite(loss)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))
    parameters = (("polynomial_square_factors",) if parameterization == "sum_of_squares" else
                  ("polynomial_coefficients", "polynomial_denominator_coefficients") if parameterization == "rational"
                  else ("polynomial_coefficients", "polynomial_denominator_square_factors")
                  if parameterization == "rational_sos2" else ("polynomial_coefficients",))
    for name in (*parameters, "qk_norm_log_scale"):
        assert np.linalg.norm(np.asarray(grads[name][...])) > 0, name

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
    for name in (*parameters, "qk_norm_log_scale"):
        trained, start = (np.asarray(state.params[name][...]) for state in (states[0], initial))
        assert np.isfinite(trained).all()
        assert np.any(trained != start), f"{name} did not update"
