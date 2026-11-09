"""Causal query strips (strip_dot_product_attention) reproduce the blocked scan.

The strips run the scan's per-block arithmetic, the same code, against causal key prefixes
(attention/backends/reference.py). blocked_dot_product_attention computes in FP32 whatever its
input dtype, so exact-arithmetic checks compare the strips with the scan's own code evaluated in
FP64 (``_blocked_scan(compute=float64)``); FP32 and BF16 checks compare with the scan itself.
"""
from functools import partial
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import (
    STRIP_SIZE, blocked_backend_attention, blocked_dot_product_attention, strip_dot_product_attention,
)
from scaffolds.transformer.attention.core import attention_core

UNNORMALIZED = {"attention_normalization": False}
# Every argument family of blocked_dot_product_attention: (static arguments, array arguments).
CASES = {
    "softmax": ({}, ()),
    "exp": (UNNORMALIZED, ()),
    "taylor0": ({"attention_degree": 0}, ()),
    "taylor1": ({"attention_degree": 1}, ()),
    "taylor2": ({"attention_degree": 2}, ()),
    "taylor2_unnormalized": ({"attention_degree": 2, **UNNORMALIZED}, ()),
    "taylor3_unnormalized": ({"attention_degree": 3, **UNNORMALIZED}, ()),
    "quadratic_positive": ({"attention_degree": 2, "quadratic_mode": "positive"}, ("coefficients",)),
    "quadratic_unrestricted": ({"attention_degree": 2, "quadratic_mode": "unrestricted"}, ("coefficients",)),
    "learned2": ({"attention_degree": 2}, ("polynomial_coefficients",)),
    "learned3_per_head": ({"attention_degree": 3}, ("polynomial_coefficients",)),
    "learned2_unnormalized": ({"attention_degree": 2, **UNNORMALIZED}, ("polynomial_coefficients",)),
    "square4": ({"attention_degree": 4}, ("polynomial_square_factors",)),
    "rational_direct": ({"attention_degree": 2},
                        ("polynomial_coefficients", "polynomial_denominator_coefficients")),
    "rational_direct_constant": ({"attention_degree": 2},
                                 ("polynomial_coefficients", "polynomial_denominator_coefficients_with_constant")),
    "rational_squares": ({"attention_degree": 2},
                         ("polynomial_coefficients", "polynomial_denominator_square_factors")),
    "rational_squares_constant": ({"attention_degree": 2},
                                  ("polynomial_coefficients", "polynomial_denominator_square_factors_with_constant")),
    "query1": ({"attention_degree": 1, **UNNORMALIZED}, ("query_polynomial_coefficients",)),
    "query4": ({"attention_degree": 4, **UNNORMALIZED}, ("query_polynomial_coefficients",)),
    "query_key3": ({"attention_degree": 3, **UNNORMALIZED},
                   ("query_polynomial_coefficients", "key_polynomial_factors")),
    "query4_cumulative": ({"attention_degree": 4, **UNNORMALIZED},
                          ("query_polynomial_coefficients", "query_log_decay")),
    "query4_distance": ({"attention_degree": 4, "query_decay_mode": "query_distance", **UNNORMALIZED},
                        ("query_polynomial_coefficients", "query_log_alpha")),
    "query4_endpoint": ({"attention_degree": 4, "query_decay_mode": "query_distance", **UNNORMALIZED},
                        ("query_polynomial_coefficients", "query_gate_input", "key_gate_input", "gate_log_rate")),
    "fox_softmax": ({}, ("query_log_decay",)),
    "fox_exp": (UNNORMALIZED, ("query_log_decay",)),
    # Appended families leave the earlier families' seeds (case_arrays seeds by position) unchanged.
    "query4_per_order_distance": ({"attention_degree": 4, "query_decay_mode": "query_distance", **UNNORMALIZED},
                                  ("query_polynomial_coefficients", "query_log_alpha_per_order")),
    "query4_per_order_cumulative": ({"attention_degree": 4, **UNNORMALIZED},
                                    ("query_polynomial_coefficients", "query_log_decay_per_order")),
    "fox_softmax_distance": ({"query_decay_mode": "query_distance"}, ("query_log_alpha",)),
    "fox_exp_distance": ({"query_decay_mode": "query_distance", **UNNORMALIZED}, ("query_log_alpha",)),
}


def case_arrays(name, *, batch=2, time=37, heads=2, width=8, value_width=5, seed=0):
    """q, k, v and the family's other arrays (float64 NumPy), well conditioned for every kernel."""
    static, extra = CASES[name]
    degree = static.get("attention_degree")
    rng = np.random.default_rng(seed + 1000 * list(CASES).index(name))
    arrays = {
        "q": rng.standard_normal((batch, time, heads, width)),
        "k": rng.standard_normal((batch, time, heads, width)),
        "v": rng.standard_normal((batch, time, heads, value_width)),
    }
    taylor = None if degree is None else np.asarray([1 / math.factorial(r) for r in range(degree + 1)])
    for key in extra:
        if key == "coefficients":
            shape = (2,) if static["quadratic_mode"] == "positive" else (heads, 2)
            arrays[key] = np.asarray([1.0, 0.7] if shape == (2,) else [[0.9, 0.4], [1.1, 0.6]])[:heads]
        elif key == "polynomial_coefficients":
            per_head = name.endswith("per_head") or name == "learned2_unnormalized"
            arrays[key] = taylor + 0.1 * rng.standard_normal((heads, degree + 1) if per_head else (degree + 1,))
        elif key == "polynomial_square_factors":
            arrays[key] = np.asarray([1.0, 0.5, 0.25, 0.4, 0.1]) + 0.05 * rng.standard_normal(5)
        elif key == "polynomial_denominator_coefficients":
            arrays[key] = np.asarray([[0.9, 0.45], [1.1, 0.55]])[:heads] + 0.05 * rng.standard_normal((heads, 2))
        elif key == "polynomial_denominator_coefficients_with_constant":
            arrays[key] = np.asarray([1.0, 0.9, 0.5]) + 0.05 * rng.standard_normal(3)
        elif key == "polynomial_denominator_square_factors":
            arrays[key] = np.asarray([0.6, 0.5]) + 0.05 * rng.standard_normal(2)
        elif key == "polynomial_denominator_square_factors_with_constant":
            arrays[key] = (np.asarray([[0.8, 0.5, 0.6], [0.7, 0.4, 0.7]])[:heads]
                           + 0.05 * rng.standard_normal((heads, 3)))
        elif key == "query_polynomial_coefficients":
            arrays[key] = taylor * (1 + 0.3 * np.tanh(rng.standard_normal((batch, time, heads, degree + 1))))
        elif key == "key_polynomial_factors":
            arrays[key] = 1 + 0.3 * np.tanh(rng.standard_normal((batch, time, heads, degree + 1)))
        elif key == "query_log_decay":
            # Weak decay: distant keys keep weights near one, so a misplaced key stays visible.
            arrays[key] = np.cumsum(-rng.uniform(0.005, 0.05, (batch, time, heads)), axis=1)
        elif key == "query_log_alpha":
            arrays["query_log_decay"] = -rng.uniform(0.005, 0.1, (batch, time, heads))
        elif key == "query_log_decay_per_order":
            # Each order's own cumulative sum, decaying at its own rates.
            arrays["query_log_decay"] = np.cumsum(-rng.uniform(0.005, 0.05, (batch, time, heads, degree + 1)), axis=1)
        elif key == "query_log_alpha_per_order":
            arrays["query_log_decay"] = -rng.uniform(0.005, 0.1, (batch, time, heads, degree + 1))
        elif key in ("query_gate_input", "key_gate_input"):
            arrays[key] = rng.standard_normal((batch, time, heads))
        elif key == "gate_log_rate":
            arrays[key] = np.log(rng.uniform(0.005, 0.03, (heads, 1)))
        else:
            raise KeyError(key)
    return arrays


def program(attend, names, static):
    """jit(values, cotangent) -> (output, gradients with respect to every array argument)."""
    def run(values, cotangent):
        def function(*values):
            given = dict(zip(names, values))
            return attend(given.pop("q"), given.pop("k"), given.pop("v"), **static, **given)

        output, pullback = jax.vjp(function, *values)
        return output, pullback(cotangent)

    return jax.jit(run)


def evaluate(name, attend, *, time, dtype, qkv_dtype=None, seed=0):
    arrays = case_arrays(name, time=time, seed=seed)
    names = list(arrays)
    values = [jnp.asarray(arrays[key], qkv_dtype if qkv_dtype is not None and key in ("q", "k", "v") else dtype)
              for key in names]
    cotangent = jnp.asarray(np.random.default_rng(seed + time).standard_normal(arrays["v"].shape),
                            qkv_dtype or dtype)
    output, gradients = program(attend, names, CASES[name][0])(values, cotangent)
    return [output, *gradients], names


def relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))


# (time, strip size, scan block): several ragged strips, one strip shorter than strip_size, one query.
SHAPES = ((37, 8, 4), (5, 8, 4), (1, 8, 4))


@pytest.mark.parametrize("time,strip,block", SHAPES)
@pytest.mark.parametrize("name", list(CASES))
def test_float64_strips_match_the_scan_for_every_argument_family(name, time, strip, block):
    """Output and the gradient of every differentiable input: the strips' prefixes, ragged last
    strip, and FP32-style cross-strip sums against the scan's own code in exact arithmetic."""
    with jax.enable_x64():
        expected, names = evaluate(name, partial(reference._blocked_scan, compute=jnp.float64,
                                                 query_block_size=block), time=time, dtype=jnp.float64)
        actual, _ = evaluate(name, partial(strip_dot_product_attention, strip_size=strip),
                             time=time, dtype=jnp.float64)
    for label, observed, wanted in zip(["output", *names], actual, expected, strict=True):
        observed, wanted = np.asarray(observed), np.asarray(wanted)
        assert observed.dtype == np.float64 and observed.shape == wanted.shape, label
        assert np.isfinite(wanted).all(), label
        scale = max(np.max(np.abs(wanted)), 1.0)
        np.testing.assert_allclose(observed, wanted, rtol=1e-11, atol=1e-11 * scale, err_msg=label)


@pytest.mark.parametrize("name", list(CASES))
def test_float32_strips_match_the_public_scan(name):
    """In its own FP32 arithmetic the scan and the strips differ only in reduction order."""
    expected, names = evaluate(name, partial(blocked_dot_product_attention, query_block_size=8),
                               time=37, dtype=jnp.float32)
    actual, _ = evaluate(name, partial(strip_dot_product_attention, strip_size=16), time=37, dtype=jnp.float32)
    for label, observed, wanted in zip(["output", *names], actual, expected, strict=True):
        assert observed.dtype == wanted.dtype == jnp.float32, label
        assert relative_l2(observed, wanted) < 2e-5, label


def test_one_strip_is_one_scan_block():
    """On BF16 inputs a single strip reproduces a single scan block bit for bit (output, dQ, dK,
    dV); FP32 parameter gradients differ only in the order of their FP32 reductions."""
    for name in ("softmax", "query4_cumulative", "rational_squares"):
        expected, names = evaluate(name, partial(blocked_dot_product_attention, query_block_size=64), time=37,
                                   dtype=jnp.float32, qkv_dtype=jnp.bfloat16)
        actual, _ = evaluate(name, partial(strip_dot_product_attention, strip_size=64), time=37,
                             dtype=jnp.float32, qkv_dtype=jnp.bfloat16)
        for label, observed, wanted in zip(["output", *names], actual, expected, strict=True):
            assert observed.dtype == wanted.dtype, label
            if label in ("output", "q", "k", "v"):
                assert observed.dtype == jnp.bfloat16, label
                np.testing.assert_array_equal(np.asarray(observed, np.float32), np.asarray(wanted, np.float32),
                                              err_msg=label)
            else:
                assert relative_l2(observed, wanted) < 1e-6, label


def nonfinite_rows(output, head=1):
    """Rows of ``head`` with a nonfinite entry, and whether every other head stayed finite."""
    bad = ~np.isfinite(np.asarray(output, np.float32))
    others = bad.copy()
    others[:, :, head] = False
    return bad[0, :, head].any(-1), not others.any()


POSITION, STRIP = 20, 8


def _inject(name, case):
    """Arrays of a family with one finite-input failure (or nonfinite input) at token POSITION, head 1."""
    arrays = {key: np.asarray(value, np.float32) for key, value in case_arrays(name, batch=1).items()}
    q, k, v = arrays["q"], arrays["k"], arrays["v"]
    if case == "score_overflow":
        q[0, POSITION, 1] = 3e38                 # finite query whose products overflow: row POSITION's s = inf
    elif case == "exponential_overflow":
        key = k[0, 3, 1]
        q[0, POSITION, 1] = 100 * math.sqrt(key.size) * key / np.dot(key, key)  # s = 100: exp(s) = inf
    elif case == "polynomial_overflow":
        q[0, POSITION, 1] *= 1e20               # finite scores near 1e20 whose squares overflow
    elif case == "zero_normalizer":
        q[0, POSITION, 1] = 0.0                 # every score of the row is zero: p(s) = s sums to zero
    elif case == "positive_log_alpha":
        arrays["query_log_decay"][0, POSITION, 1] = 0.1
    elif case == "nan_log_alpha":
        arrays["query_log_decay"][0, POSITION, 1] = np.nan
    elif case == "nan_cumulative_sum":
        arrays["query_log_decay"][0, POSITION:, 1, 2] = np.nan  # order two's sum, from a NaN log alpha on
    elif case == "q_nan":
        q[0, POSITION, 1, 0] = np.nan
    elif case == "k_nan":
        k[0, POSITION, 1, 0] = np.nan
    elif case == "k_negative_infinity":
        k[0, POSITION, 1] = -np.inf
    elif case == "v_nan":
        v[0, POSITION, 1, 0] = np.nan
    elif case == "v_infinity":
        v[0, POSITION, 1, 0] = np.inf
    else:
        raise KeyError(case)
    return arrays


ROWS = np.arange(37)
LINEAR = {"attention_degree": 1, "polynomial_coefficients": np.asarray([0.0, 1.0], np.float32)}  # p(s) = s
FAILURES = [
    # (family, case, rows of head 1 that both implementations poison, replaced arguments)
    ("softmax", "score_overflow", ROWS == POSITION, {}),
    ("exp", "exponential_overflow", ROWS == POSITION, {}),
    ("taylor2", "polynomial_overflow", ROWS == POSITION, {}),
    ("learned2", "zero_normalizer", ROWS == POSITION, LINEAR),
    ("query4_distance", "positive_log_alpha", ROWS == POSITION, {}),
    ("query4_cumulative", "q_nan", ROWS == POSITION, {}),
    ("fox_softmax", "k_nan", ROWS >= POSITION, {}),
    ("rational_direct", "k_negative_infinity", ROWS >= POSITION, {}),
    ("fox_softmax_distance", "positive_log_alpha", ROWS == POSITION, {}),
    ("fox_exp_distance", "nan_log_alpha", ROWS == POSITION, {}),
    ("query4_per_order_cumulative", "nan_cumulative_sum", ROWS >= POSITION, {}),
]


@pytest.mark.parametrize("name,case,expected,replaced", FAILURES, ids=[f"{n}-{c}" for n, c, *_ in FAILURES])
def test_invalid_rows_are_the_scans(name, case, expected, replaced):
    """Invalid scores, kernels, and normalizers poison the same rows, and only those."""
    arrays = {**_inject(name, case), **replaced}
    static = CASES[name][0]
    q, k, v = (jnp.asarray(arrays.pop(key)) for key in ("q", "k", "v"))
    if "attention_degree" in arrays:
        static = {**static, "attention_degree": arrays.pop("attention_degree")}
    others = {key: jnp.asarray(value) for key, value in arrays.items()}
    scan = jax.jit(partial(blocked_dot_product_attention, **static, query_block_size=STRIP))(q, k, v, **others)
    strips = jax.jit(partial(strip_dot_product_attention, **static, strip_size=STRIP))(q, k, v, **others)
    for output in (scan, strips):
        rows, others_finite = nonfinite_rows(output)
        np.testing.assert_array_equal(rows, expected)
        assert others_finite
    finite = ~expected
    np.testing.assert_allclose(np.asarray(strips)[0, finite, 1], np.asarray(scan)[0, finite, 1], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("case", ("v_nan", "v_infinity"))
@pytest.mark.parametrize("name", ("softmax", "query4_cumulative"))
def test_nonfinite_values_poison_rows_from_their_strip_onward(name, case):
    """The one intended difference: masked keys are not multiplied, so a bad V entry poisons the
    rows from the start of its strip onward; the scan's masked zero weights poison every row."""
    arrays = _inject(name, case)
    q, k, v = (jnp.asarray(arrays.pop(key)) for key in ("q", "k", "v"))
    others = {key: jnp.asarray(value) for key, value in arrays.items()}
    static = CASES[name][0]
    scan = jax.jit(partial(blocked_dot_product_attention, **static, query_block_size=STRIP))(q, k, v, **others)
    strips = jax.jit(partial(strip_dot_product_attention, **static, strip_size=STRIP))(q, k, v, **others)
    scan_rows, scan_others = nonfinite_rows(scan)
    strip_rows, strip_others = nonfinite_rows(strips)
    assert scan_others and strip_others
    assert scan_rows.all()
    np.testing.assert_array_equal(strip_rows, ROWS >= POSITION // STRIP * STRIP)
    # Rows before the bad value's strip are the clean result.
    clean = jax.jit(partial(strip_dot_product_attention, **static, strip_size=STRIP))(
        *(jnp.asarray(case_arrays(name, batch=1)[key], jnp.float32) for key in ("q", "k", "v")), **others)
    np.testing.assert_array_equal(np.asarray(strips)[:, :16], np.asarray(clean)[:, :16])


@pytest.mark.parametrize("name", ("softmax", "query4_cumulative"))
def test_bfloat16_strips_put_key_and_value_gradients_at_the_rounding_floor(name):
    """Four 256-query strips sum dK/dV in FP32 and round once. The scan's eight BF16 block sums
    cost 1.5-1.7x the floor (the float64 reference rounded to BF16) here: CPU XLA reproduces the
    BF16 sums measured on H100 (1.8-2.8x at 2,048 tokens)."""
    shape = (1, 1024, 2, 64)
    rng = np.random.default_rng(3)
    q, k, v, cotangent = (jnp.asarray(rng.standard_normal(shape), jnp.bfloat16) for _ in range(4))
    static, extra = {}, {}
    if name == "query4_cumulative":
        static = {"attention_degree": 4, "attention_normalization": False}
        taylor = np.asarray([1, 1, 1 / 2, 1 / 6, 1 / 24])
        extra = {"query_polynomial_coefficients": jnp.asarray(
                     taylor * (1 + 0.3 * np.tanh(rng.standard_normal((*shape[:3], 5)))), jnp.float32),
                 "query_log_decay": jnp.asarray(np.cumsum(-rng.uniform(0.001, 0.02, shape[:3]), axis=1), jnp.float32)}

    def run(attend, operands, g):
        def function(q, k, v, extra):
            return attend(q, k, v, **static, **extra)

        output, pullback = jax.vjp(function, *operands)
        return [output, *pullback(g)[:3]]

    with jax.enable_x64():
        operands = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), (q, k, v, extra))
        scan64 = partial(reference._blocked_scan, compute=jnp.float64, query_block_size=1024)
        exact = [np.asarray(x) for x in jax.jit(partial(run, scan64))(operands, jnp.asarray(cotangent, jnp.float64))]
    floor = [relative_l2(np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32)), x) for x in exact]

    def ratios(attend):
        observed = jax.jit(partial(run, attend))((q, k, v, extra), cotangent)
        assert all(x.dtype == jnp.bfloat16 for x in observed)
        errors = [relative_l2(np.asarray(x.astype(jnp.float32)), e) for x, e in zip(observed, exact)]
        return [error / best for error, best in zip(errors, floor)], observed

    strips, strip_results = ratios(partial(strip_dot_product_attention, strip_size=256))
    scan, scan_results = ratios(blocked_dot_product_attention)
    assert max(strips) < 1.1, strips                # output, dQ, dK, dV
    assert max(scan[:2]) < 1.1, scan                # the scan's output and dQ are at the floor too
    assert min(scan[2:]) > 1.3, scan                # its BF16 dK/dV block sums are not
    # The forward is the scan's: identical BF16-exact products, FP32 accumulation in another order.
    # Rows whose terms cancel to near zero differ by that order's rounding, which depends on the
    # host CPU's vectorization, so the absolute tolerance scales with the output.
    strip_output, scan_output = (np.asarray(x[0], np.float32) for x in (strip_results, scan_results))
    assert relative_l2(strip_output, scan_output) < 2 ** -9
    np.testing.assert_allclose(strip_output, scan_output, rtol=2 ** -7, atol=1e-4 * np.abs(scan_output).max())


def test_strips_recompute_inside_checkpoints_that_prevent_cse():
    """One checkpoint per strip with CSE prevention: unrolled strips have no loop boundary, so
    without it XLA merges each recomputation with its forward and keeps every layer's scores."""
    q = jnp.zeros((1, 37, 2, 8), jnp.bfloat16)
    text = str(jax.make_jaxpr(partial(strip_dot_product_attention, strip_size=8))(q, q, q))
    # jax.checkpoint prints as remat2; 37 tokens in strips of 8 are five strips.
    assert len(re.findall(r"\bremat2\[", text)) == text.count("prevent_cse=True") == 5
    assert "prevent_cse=False" not in text and "scan[" not in text


def test_bf16_conversions_keep_their_rounding_explicitly():
    """K and V once, and each strip's queries and output, convert through reduce_precision, which XLA
    GPU keeps where it removes f32 -> bf16 -> f32 conversion pairs (test_blocked_strips_h100.py
    checks the compiled programs); FP32 operands need no rounding."""
    q = jnp.zeros((1, 37, 2, 8), jnp.bfloat16)
    text = str(jax.make_jaxpr(partial(strip_dot_product_attention, strip_size=8))(q, q, q))
    assert len(re.findall(r"\breduce_precision\[exponent_bits=8 mantissa_bits=7\]", text)) == 2 + 2 * 5
    q = q.astype(jnp.float32)
    assert "reduce_precision" not in str(jax.make_jaxpr(partial(strip_dot_product_attention, strip_size=8))(q, q, q))


def test_strip_size_and_implementation_are_validated():
    q = jnp.zeros((1, 4, 1, 2))
    for size in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="strip_size must be a positive integer"):
            strip_dot_product_attention(q, q, q, strip_size=size)
    with pytest.raises(TypeError, match="return_diagnostics"):
        strip_dot_product_attention(q, q, q, return_diagnostics=True)
    with pytest.raises(ValueError, match="Unknown blocked attention implementation"):
        blocked_backend_attention(q, q, q, implementation="flash")
    with pytest.raises(ValueError, match="Unknown blocked attention implementation"):
        attention_core(q, q, q, blocked_implementation="flash")
    # The scan's own argument checks apply unchanged.
    with pytest.raises(ValueError, match="Fixed attention does not accept learned coefficients"):
        strip_dot_product_attention(q, q, q, coefficients=jnp.ones(2))
    assert STRIP_SIZE == 512


# Families as attention_core receives them: CASES' arrays and statics, plus the coefficient activation.
TANH = {"attention_coefficient_activation": "tanh"}
CORE_FAMILIES = {"softmax": {}, "taylor2": {}, "exp": {}, "learned2_unnormalized": {}, "learned2": {}, "square4": {},
                 "rational_squares_constant": {}, "quadratic_unrestricted": {}, "query1": {}, "query4": TANH,
                 "query_key3": TANH, "query4_cumulative": TANH, "query4_endpoint": TANH, "fox_softmax": {},
                 "fox_exp": {}, "query4_per_order_distance": TANH, "query4_per_order_cumulative": TANH,
                 "fox_softmax_distance": {}, "fox_exp_distance": {}}


@pytest.mark.parametrize("family", list(CORE_FAMILIES))
def test_every_blocked_mechanism_trains_on_strips_and_diagnoses_with_the_scan(family, monkeypatch):
    """attention_core routes each mechanism's blocked output to the strips (counted) and its
    diagnostics to the scan; the default keeps the scan's output exactly."""
    arrays = {key: jnp.asarray(value, jnp.float32) for key, value in case_arrays(family, time=9).items()}
    q, k, v = arrays.pop("q"), arrays.pop("k"), arrays.pop("v")
    static = CASES[family][0]
    original = reference.strip_dot_product_attention
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, strip_size=4, **kwargs)

    monkeypatch.setattr(reference, "strip_dot_product_attention", counted)

    def core(implementation, diagnostics=False):
        return jax.jit(lambda q, k, v, arrays: attention_core(
            q, k, v, backend="blocked", blocked_implementation=implementation, return_diagnostics=diagnostics,
            **static, **CORE_FAMILIES[family], **arrays))(q, k, v, arrays)

    def direct(attend, **options):
        return jax.jit(lambda q, k, v, arrays: attend(q, k, v, **static, **arrays, **options))(q, k, v, arrays)

    strips = core("strips")
    assert len(calls) == 1
    np.testing.assert_array_equal(strips, direct(original, strip_size=4))
    scan = core("scan")
    assert len(calls) == 1
    np.testing.assert_array_equal(scan, direct(blocked_dot_product_attention))
    assert relative_l2(strips, scan) < 2e-5
    output, diagnostics = core("strips", diagnostics=True)
    _, expected = core("scan", diagnostics=True)
    np.testing.assert_array_equal(output, strips)
    assert diagnostics.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(diagnostics[name], expected[name], err_msg=name)


def _model(settings, *, implementation, dtype):
    from flax import nnx

    from scaffolds.transformer.configs import ModelConfig
    from scaffolds.transformer.models import build_model

    config = ModelConfig(vocab_size=64, max_seq_len=32, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2,
                         qk_head_dim=16, value_head_dim=16, intermediate_size=64, attention_backend="blocked",
                         initialization="transformer_reference", attention_blocked_implementation=implementation,
                         **settings)
    model = build_model(config, nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)))
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    graphdef, params = nnx.split(model, nnx.Param)
    return graphdef, jax.tree.map(lambda x: x.astype(dtype), params)


def _loss_and_gradients(graphdef, params, tokens):
    from flax import nnx

    def loss(params):
        logits = nnx.merge(graphdef, params)(tokens[:, :-1]).astype(params_dtype(params))
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(log_probabilities, tokens[:, 1:, None], axis=-1))

    return jax.jit(jax.value_and_grad(loss))(params)


def params_dtype(params):
    return jax.tree.leaves(params)[0].dtype


MODEL_RECIPES = {
    # Forget-gate recipes on the blocked backend; plain QK keeps FP32 normalization out of the
    # float64 comparison (QK RMSNorm normalizes in FP32 by design).
    "fox_softmax": dict(attention_forget_gate=True, qk_norm=False),
    "query4_cumulative": dict(attention_degree=4, attention_normalization=False, attention_coefficients="learned",
                              attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                              attention_coefficient_activation="tanh", attention_forget_gate=True, qk_norm=False,
                              attention_coefficient_projection_initial_std=0.1),
    # Without head RMSNorm, which normalizes in FP32 like QK RMSNorm.
    "fox_exp_distance": dict(attention_forget_gate=True, attention_forget_gate_mode="query_distance",
                             attention_normalization=False, qk_norm=False),
    "query4_per_order_cumulative": dict(attention_degree=4, attention_normalization=False,
                                        attention_coefficients="learned", attention_coefficient_sharing="per_head",
                                        attention_coefficient_conditioning="query",
                                        attention_coefficient_activation="tanh", attention_forget_gate=True,
                                        attention_forget_gate_per_order=True, qk_norm=False,
                                        attention_coefficient_projection_initial_std=0.1),
}


@pytest.mark.parametrize("recipe", list(MODEL_RECIPES))
def test_model_strips_match_the_scan(recipe, monkeypatch):
    """A two-layer forget-gate model: the same loss and parameter gradients with either
    implementation, in exact arithmetic (the scan's code in float64) and in the scan's FP32;
    every layer's attention runs on the strips (counted), here four 8-query strips of 32 tokens."""
    tokens = jnp.asarray(np.random.default_rng(5).integers(0, 64, (2, 33)), jnp.int32)
    original = reference.strip_dot_product_attention
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, strip_size=8, **kwargs)

    monkeypatch.setattr(reference, "strip_dot_product_attention", counted)
    with jax.enable_x64():
        with monkeypatch.context() as patch:
            patch.setattr(reference, "blocked_dot_product_attention",
                          partial(reference._blocked_scan, compute=jnp.float64))
            scan = _loss_and_gradients(*_model(MODEL_RECIPES[recipe], implementation="scan", dtype=jnp.float64), tokens)
            assert calls == []
            strips = _loss_and_gradients(*_model(MODEL_RECIPES[recipe], implementation="strips", dtype=jnp.float64),
                                         tokens)
            assert len(calls) == 2  # one trace, two layers
        np.testing.assert_allclose(strips[0], scan[0], rtol=1e-11)
        for (path, observed), wanted in zip(jax.tree_util.tree_leaves_with_path(strips[1]),
                                            jax.tree.leaves(scan[1]), strict=True):
            assert observed.dtype == jnp.float64
            # The forget gate projects in FP32 by design (forget_gate.log_decay): its own parameters'
            # gradients are FP32 sums over tokens, which the two programs may order differently.
            tolerance = 2e-6 if "forget_gate" in jax.tree_util.keystr(path) else 1e-11
            np.testing.assert_allclose(observed, wanted, rtol=tolerance,
                                       atol=tolerance * max(np.max(np.abs(wanted)), 1.0), err_msg=str(path))
    calls.clear()
    scan = _loss_and_gradients(*_model(MODEL_RECIPES[recipe], implementation="scan", dtype=jnp.float32), tokens)
    strips = _loss_and_gradients(*_model(MODEL_RECIPES[recipe], implementation="strips", dtype=jnp.float32), tokens)
    assert len(calls) == 2
    np.testing.assert_allclose(strips[0], scan[0], rtol=1e-6)
    for observed, wanted in zip(jax.tree.leaves(strips[1]), jax.tree.leaves(scan[1]), strict=True):
        assert relative_l2(observed, wanted) < 5e-5


def test_metadata_records_the_strips_and_their_diagnostic_scan():
    from scaffolds.transformer.configs import Config, ModelConfig
    from scaffolds.transformer.training.diagnostics import attention_identity

    config = Config(model=ModelConfig(attention_backend="blocked", attention_forget_gate=True,
                                      attention_blocked_implementation="strips"))
    identity = attention_identity(config)
    assert identity["attention_backend"] == "blocked"
    assert identity["attention_implementation"] == "blocked_causal_strips"
    assert identity["attention_blocked_implementation"] == "strips"
    assert identity["attention_strip_size"] == STRIP_SIZE
    assert identity["query_block_size"] is None
    assert identity["diagnostic_backend"] == "blocked"
    assert identity["diagnostic_query_block_size"] == reference.QUERY_BLOCK_SIZE
    # The scan's record is unchanged, and a recipe's strips never relabel another backend.
    config.model.attention_blocked_implementation = "scan"
    scan = attention_identity(config)
    assert scan["attention_implementation"] == "blocked" and scan["query_block_size"] == reference.QUERY_BLOCK_SIZE
    assert not {"attention_blocked_implementation", "attention_strip_size", "diagnostic_backend"} & scan.keys()
    fused = Config(model=ModelConfig(attention_backend="fused", attention_degree=2,
                                     attention_blocked_implementation="strips"))
    assert "attention_strip_size" not in attention_identity(fused)
    assert attention_identity(fused)["attention_implementation"] == "pallas_triton"
