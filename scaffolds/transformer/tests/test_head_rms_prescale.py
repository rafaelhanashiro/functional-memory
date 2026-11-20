"""Head RMSNorm in units of each row's largest magnitude (``model.attention_head_rms_prescale: max_abs``).

``o / sqrt(mean(o^2) + eps) g`` is evaluated as ``(o / c) / sqrt(mean((o / c)^2) + eps / c^2) g`` with ``c > 0``
taken per token and head from the row and held constant (attention/output_scaling.py): the same function of ``o``,
so its values and gradients are the plain evaluation's in real arithmetic. These checks hold the production
arithmetic to the plain function in FP64 (values and VJPs at every output scale, the eps term included, both
branches of a row-shifted exponential's); hold its FP32 gradient to FP64 at every scale of the measured probe
(validation/README.md), where the plain evaluation loses the term that removes the gradient's component along the
row from an output RMS of 9e14 to 1e18, kept here as the regression case of that defect; check that "none" is the
historical arithmetic bit for bit, the composition with the exponential's row-maximum shift on the scan and the
strips, nonfinite rows, refusals, the recipe default, provenance, telemetry and exact training resumption; and
compare two-layer models of the four polynomial families and of the shifted exponential with the same models in
FP64 when every head output is forced above 1e15. The GPU cases repeat the probe on the device's own arithmetic and
run the Mosaic kernels (fixed Taylor, static, query, and softplus sum of squares with its input power) through the
prescaled normalization, at the kernel and through two-layer models at Q/K widths 4 and 64.
"""
from dataclasses import replace
from functools import partial
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.reference import (
    blocked_dot_product_attention, strip_dot_product_attention,
)
from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.output_scaling import (
    HEAD_RMS_PRESCALES, LINEAR_LOG_RANGE, SLOPE_HEADROOM, max_abs_head_rms, scale_attention_output,
)
from scaffolds.transformer.configs import Config, ModelConfig, validate_config
from scaffolds.transformer.configs.base import fresh_recipe_uses_head_rms_prescale
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.tests.test_exp_row_max_shift import SHIFTED, _arrays, _dense

EPS = 1e-6
TEST_PATH = "scaffolds/transformer/tests/test_head_rms_prescale.py"
# The scales of the probe that first measured the defect on a GPU (1e10 to 1e25), extended down to vanishing rows
# and up to the largest rows whose gradient FP32 still holds.
PROBE_RMS = (1e-25, 1e-13, 1e-6, 1e-3, 1.0, 1e6, 1e10, 1e11, 1e12, 3e12, 4.4e12, 6e12, 1e13, 3e13, 1e14, 2e14, 3e14,
             4.5e14, 6e14, 9e14, 1.2e15, 2e15, 1e16, 1e17, 1e18, 1e19, 3e19, 1e25, 1e30)
# Where the plain FP32 backward has lost the whole projection term on every device: m^(-3/2), m = mean(y^2), is
# below half of FP32's smallest subnormal number from an RMS of about 1.1e15, and the mean square overflows (the
# historical guard then rescales the row) past an L2 norm of 1.8e19, an RMS of 2.3e18 over 64 channels.
FAILING_RMS = tuple(rms for rms in PROBE_RMS if 9e14 <= rms <= 1e18)


def _reference(value, gain, cotangent, *, eps=EPS, shift=None):
    """NumPy FP64 output and VJP (d/dvalue, d/dgain) of ``value rsqrt(mean(value^2) + eps e^(-2 shift)) gain`` over
    [batch, tokens, heads, channels]: head RMSNorm of ``o = e^shift value`` written in the row's own units, which
    FP64 holds wherever FP32 holds ``value`` and ``shift`` (``shift`` None: zero)."""
    value, gain, cotangent = (np.asarray(array, np.float64) for array in (value, gain, cotangent))
    with np.errstate(all="ignore"):
        epsilon = eps if shift is None else eps * np.exp(-2.0 * np.asarray(shift, np.float64))[..., None]
        scale = 1.0 / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + epsilon)
        weighted = cotangent * gain
        radial = np.sum(weighted * value, axis=-1, keepdims=True) * scale ** 3 / value.shape[-1]
        return value * scale * gain, (weighted * scale - value * radial, np.sum(cotangent * value * scale, axis=(0, 1)))


def _relative(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))


def _row_relative(actual, expected):
    """The largest relative L2 error of any row (token and head): rows of very different scales share an array."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    norm = np.linalg.norm(expected, axis=-1)
    error = np.linalg.norm(actual - expected, axis=-1)
    return float(np.max(np.where(norm > 0, error / np.where(norm > 0, norm, 1.0), error)))


def _radial_fraction(value, gain, cotangent, gradient, *, eps=EPS):
    """sum |y . vjp| / sum |y . (g s)| over rows, g the gain-weighted cotangent and s = 1/sqrt(mean(y^2) + eps):
    zero for the exact Jacobian, whose result is orthogonal to y (up to eps), and one when the term that removes
    the radial component is lost entirely."""
    value, gradient = np.asarray(value, np.float64), np.asarray(gradient, np.float64)
    weighted = np.asarray(cotangent, np.float64) * np.asarray(gain, np.float64)
    scale = 1.0 / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)
    lost = np.sum(np.abs(np.sum(value * weighted * scale, axis=-1)))
    return float(np.sum(np.abs(np.sum(value * gradient, axis=-1))) / lost)


def _probe(*, rows=2048, heads=2, channels=64, seed=0):
    """The measured probe's rows: unit-RMS directions, a cotangent of scale 1e-4 and gains near one."""
    generator = np.random.default_rng(seed)
    direction = generator.standard_normal((1, rows, heads, channels))
    direction /= np.sqrt(np.mean(direction ** 2, axis=-1, keepdims=True))
    cotangent = 1e-4 * generator.standard_normal(direction.shape)
    gain = 1.0 + 0.1 * generator.standard_normal((heads, channels))
    return direction, cotangent, gain


def _evaluation(prescale, *, shifted=False, jit=True):
    """(y, gain, cotangent[, shift]) -> (output, d/dy, d/dgain) of head RMSNorm with ``prescale``."""
    def evaluate(y, gain, cotangent, shift=None):
        arguments = {} if shift is None else {"output_shift": shift}
        output, pullback = jax.vjp(lambda y, gain: scale_attention_output(
            y, "head_rms_norm", output_norm_scale=gain, head_rms_prescale=prescale, **arguments), y, gain)
        return (output, *pullback(cotangent))

    return jax.jit(evaluate) if jit else evaluate


def _assert_bitwise(actual, expected):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8))


def _plain(value, eps=EPS, shift=None):
    """Head RMSNorm as written, in ``value``'s dtype (the function without its gain)."""
    epsilon = eps if shift is None else eps * jnp.exp(-2.0 * shift)
    return value * jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + epsilon)


def _mixed_rows(generator, scales, channels=16):
    """One row per scale, plus a zero row and a row whose channels span twelve orders of magnitude."""
    rows = [generator.standard_normal(channels) * scale for scale in scales]
    rows.append(np.zeros(channels))
    rows.append(generator.standard_normal(channels) * 10.0 ** generator.uniform(-6, 6, channels))
    return np.stack(rows)


@pytest.mark.parametrize("eps", (1e-6, 1e-12, 1.0))
def test_the_prescaled_function_is_the_plain_one_in_fp64(eps):
    """The production arithmetic in FP64 against ``y rsqrt(mean(y^2) + eps)`` for rows of RMS 1e-25 to 1e30 (a zero
    row and a row of mixed channel scales with them): values and VJPs agree to 1e-12 of each row, the eps term
    included (rows from far below sqrt(eps) to far above it). Differentiating through the divisor as well gives
    the same gradient, so holding it constant loses nothing."""
    generator = np.random.default_rng(1)
    scales = 10.0 ** np.arange(-25.0, 30.5, 0.5)
    rows = _mixed_rows(generator, scales)
    direction = generator.standard_normal(rows.shape)
    with jax.enable_x64():
        value, cotangent = jnp.asarray(rows), jnp.asarray(direction)

        def differentiated(value):
            divisor = jnp.maximum(jnp.max(jnp.abs(value), axis=-1, keepdims=True), 1.0)
            scaled = value / divisor
            return scaled * jax.lax.rsqrt(jnp.mean(jnp.square(scaled), axis=-1, keepdims=True)
                                          + eps / jnp.square(divisor))

        results = {}
        for name, function in (("plain", partial(_plain, eps=eps)), ("prescaled", partial(max_abs_head_rms, eps=eps)),
                               ("differentiated divisor", differentiated)):
            output, pullback = jax.vjp(function, value)
            assert output.dtype == jnp.float64
            results[name] = (np.asarray(output), np.asarray(pullback(cotangent)[0]))
    for name in ("prescaled", "differentiated divisor"):
        for label, got, want in zip(("output", "VJP"), results[name], results["plain"]):
            assert np.isfinite(got).all(), (name, label)
            assert _row_relative(got, want) <= 1e-12, (name, label, _row_relative(got, want))
    # Rows of at most one keep the plain arithmetic: identical, not only close.
    small = np.max(np.abs(rows), axis=-1) <= 1.0
    assert small.sum() > 40
    np.testing.assert_array_equal(results["prescaled"][0][small], results["plain"][0][small])
    np.testing.assert_array_equal(results["prescaled"][1][small], results["plain"][1][small])


@pytest.mark.parametrize("eps", (1e-6, 1e-12))
def test_the_prescaled_shifted_function_is_the_plain_one_in_fp64(eps):
    """A row-shifted exponential's rows ``y = e^(-m) o`` with their shifts: the production arithmetic in FP64 against
    ``y rsqrt(mean(y^2) + eps e^(-2m))`` over shifts of -60 to 150 and rows from 1e-100 to 1e60 in the units of y
    (the range in which the plain FP64 backward itself stays finite), on both sides of the linear range's boundary
    (``max|o| = e^-23 sqrt(eps)``) and next to it."""
    generator = np.random.default_rng(2)
    shifts = np.array([-60.0, -20.0, -3.0, 0.0, 4.0, 30.0, 88.0, 150.0])
    # log10 of max|o| / sqrt(eps): the linear range ends at -23 / ln 10 = -9.99.
    relative = np.concatenate([np.arange(-30.0, 40.0, 2.5), [-10.2, -10.0, -9.98, -9.9]])
    boundary = 0.5 * math.log10(eps)
    rows, row_shifts = [], []
    for shift in shifts:
        for log_scale in relative:
            direction = generator.standard_normal(16)
            direction /= np.max(np.abs(direction))
            scale = 10.0 ** (log_scale + boundary - shift / math.log(10.0))  # the row's largest magnitude, in y
            rows.append(direction * scale)
            row_shifts.append(shift)
        rows.append(np.zeros(16))
        row_shifts.append(shift)
    rows, row_shifts = np.stack(rows), np.asarray(row_shifts)[:, None]
    direction = generator.standard_normal(rows.shape)
    with jax.enable_x64():
        value, shift, cotangent = (jnp.asarray(array) for array in (rows, row_shifts, direction))
        log_scale = shift + jnp.log(jnp.max(jnp.abs(value), axis=-1, keepdims=True)) - 0.5 * math.log(eps)
        linear = np.asarray(log_scale <= -LINEAR_LOG_RANGE)[:, 0]
        results = {}
        for name, function in (("plain", partial(_plain, eps=eps, shift=shift)),
                               ("prescaled", lambda value: max_abs_head_rms(value, eps, shift))):
            output, pullback = jax.vjp(function, value)
            results[name] = (np.asarray(output), np.asarray(pullback(cotangent)[0]))
    assert linear.sum() > 60 and (~linear).sum() > 100  # both branches
    for label, got, want in zip(("output", "VJP"), results["prescaled"], results["plain"]):
        assert np.isfinite(got).all() and np.isfinite(want).all(), label
        assert _row_relative(got, want) <= 1e-11, (label, _row_relative(got, want))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16), ids=("fp32", "bf16"))
def test_fp32_gradient_matches_fp64_at_every_scale_of_the_measured_probe(dtype):
    """The measured probe (2,048 tokens, two heads, 64 channels, a cotangent of scale 1e-4) at output RMS 1e-25 to
    1e30. Prescaled: output and VJP in the rows and the gain match FP64 at every scale, to FP32 rounding (to BF16
    rounding of the stored output and gradient for a BF16 layer), and the gradient has no component along its row.
    Plain: the same up to 1e12 and beyond the mean square's overflow, but from 9e14 to 1e18 its gradient keeps the
    whole radial component, the defect the prescale removes."""
    direction, cotangent, gain = _probe()
    cotangent, gain = jnp.asarray(cotangent, dtype), jnp.asarray(gain, jnp.float32)
    prescaled, plain = _evaluation("max_abs"), _evaluation("none")
    for rms in PROBE_RMS:
        # At 1e30 the gradient's smallest elements are below FP32's normal range, which XLA's CPU code flushes.
        tolerance = 1e-2 if dtype == jnp.bfloat16 else 1e-6 if rms <= 1e25 else 1e-5
        y = jnp.asarray(direction * rms, dtype)
        represented = np.asarray(y.astype(jnp.float32), np.float64)
        expected = _reference(represented, gain, cotangent.astype(jnp.float32))
        output, dy, dgain = prescaled(y, gain, cotangent)
        assert output.dtype == dy.dtype == dtype and dgain.dtype == jnp.float32
        actual = tuple(np.asarray(array.astype(jnp.float32), np.float64) for array in (output, dy, dgain))
        assert _relative(actual[0], expected[0]) <= tolerance, (rms, "output", _relative(actual[0], expected[0]))
        assert _relative(actual[1], expected[1][0]) <= tolerance, (rms, "dy", _relative(actual[1], expected[1][0]))
        assert _relative(actual[2], expected[1][1]) <= tolerance, (rms, "dgain", _relative(actual[2], expected[1][1]))
        if dtype == jnp.float32 and 1.0 <= rms <= 1e25:
            # The exact gradient's radial fraction is eps / (rms^2 + eps): 1e-6 at one, and one far below sqrt(eps).
            assert _radial_fraction(represented, gain, cotangent, actual[1]) <= 1e-5, rms
        historical = np.asarray(plain(y, gain, cotangent)[1].astype(jnp.float32), np.float64)
        if rms in FAILING_RMS:
            assert _relative(historical, expected[1][0]) >= 0.05, (rms, _relative(historical, expected[1][0]))
            assert _radial_fraction(represented, gain, cotangent, historical) >= 0.99, rms
        elif rms <= 1e12 or rms >= 1e19:
            assert _relative(historical, expected[1][0]) <= tolerance, (rms, _relative(historical, expected[1][0]))


def test_rows_of_at_most_one_keep_the_plain_arithmetic_bit_for_bit():
    """A row whose largest magnitude is at most one (zero rows included) keeps divisor one: output and VJP are the
    plain evaluation's bit for bit, compiled or not, FP32 and BF16."""
    generator = np.random.default_rng(3)
    magnitudes = generator.choice([0.0, 1e-30, 1e-13, 1e-6, 1e-3, 0.3, 1.0], size=(2, 37, 3, 1))
    direction = generator.standard_normal((2, 37, 3, 8))
    direction /= np.max(np.abs(direction), axis=-1, keepdims=True)
    gain = jnp.asarray(generator.uniform(0.5, 2.0, (3, 8)), jnp.float32)
    for dtype in (jnp.float32, jnp.bfloat16):
        y = jnp.asarray(direction * magnitudes, dtype)
        assert float(jnp.max(jnp.abs(y.astype(jnp.float32)))) == 1.0
        cotangent = jnp.asarray(generator.standard_normal(y.shape), dtype)
        for jit in (False, True):
            _assert_bitwise(_evaluation("max_abs", jit=jit)(y, gain, cotangent),
                            _evaluation("none", jit=jit)(y, gain, cotangent))


def _historical_head_rms_norm(y, gain, eps=EPS, output_shift=None):
    """Head RMSNorm as committed before the prescale (c7a9319e), kept here to hold "none" to it."""
    value = y.astype(jnp.float32)
    overflow = ~jnp.isfinite(jnp.mean(jnp.square(value), axis=-1, keepdims=True))
    divisor = jax.lax.stop_gradient(jnp.where(overflow, jnp.max(jnp.abs(value), axis=-1, keepdims=True), 1.0))
    value = value / divisor
    if output_shift is None:
        value = value * jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True)
                                      + eps / jnp.square(divisor))
    else:
        shift = jax.lax.stop_gradient(output_shift.astype(jnp.float32))[..., None]
        denominator = (jnp.mean(jnp.square(value), axis=-1, keepdims=True)
                       + eps * jnp.exp(-2.0 * shift) / jnp.square(divisor))
        zero = denominator == 0
        value = value * jnp.where(zero, 0.0, jax.lax.rsqrt(jnp.where(zero, 1.0, denominator)))
    value = value * gain.astype(jnp.float32)[None, None]
    return value.astype(y.dtype)


@pytest.mark.parametrize("shifted", (False, True), ids=("unshifted", "row_shifted"))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16), ids=("fp32", "bf16"))
def test_none_is_the_historical_arithmetic_bit_for_bit(dtype, shifted):
    """The default and an explicit "none" compute what the function computed before the prescale existed: output
    and VJP in the rows and the gain bit for bit, compiled or not, on rows from zero to beyond the mean square's
    overflow, with and without a row shift."""
    generator = np.random.default_rng(4)
    magnitudes = generator.choice([0.0, 1e-30, 1e-3, 1.0, 1e9, 1e15, 1e18, 1e25, 1e37], size=(2, 29, 3, 1))
    y = jnp.asarray(generator.standard_normal((2, 29, 3, 8)) * magnitudes, dtype)
    gain = jnp.asarray(generator.uniform(0.5, 2.0, (3, 8)), jnp.float32)
    cotangent = jnp.asarray(generator.standard_normal(y.shape), dtype)
    shift = jnp.asarray(generator.uniform(-30, 60, y.shape[:3]), jnp.float32) if shifted else None

    def historical(y, gain, cotangent, shift=None):
        output, pullback = jax.vjp(lambda y, gain: _historical_head_rms_norm(y, gain, output_shift=shift), y, gain)
        return (output, *pullback(cotangent))

    def default(y, gain, cotangent, shift=None):
        arguments = {} if shift is None else {"output_shift": shift}
        output, pullback = jax.vjp(lambda y, gain: scale_attention_output(
            y, "head_rms_norm", output_norm_scale=gain, **arguments), y, gain)
        return (output, *pullback(cotangent))

    arguments = (y, gain, cotangent) if shift is None else (y, gain, cotangent, shift)
    for transform in (lambda function: function, jax.jit):
        expected = transform(historical)(*arguments)
        _assert_bitwise(transform(default)(*arguments), expected)
        _assert_bitwise(_evaluation("none", jit=transform is jax.jit)(*arguments), expected)
    assert HEAD_RMS_PRESCALES == ("none", "max_abs")


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16), ids=("fp32", "bf16"))
def test_the_prescale_covers_rows_whose_fp32_mean_square_overflows(dtype):
    """Rows of peak 1.9e19 to 3e38, whose FP32 mean square is infinite: the prescale divides every row above one by
    its largest magnitude, so it needs no guard for them; output (to 3e38) and VJP (to 1e30, beyond which the exact
    gradient leaves FP32's normal range) match FP64."""
    generator = np.random.default_rng(11)
    ones = jnp.ones((1, 8), jnp.float32)
    scaled = partial(scale_attention_output, scaling="head_rms_norm", output_norm_scale=ones,
                     head_rms_prescale="max_abs")
    for peak in (1.9e19, 1e25, 1e30, 1e37, 3e38):
        row = generator.standard_normal((1, 1, 1, 8))
        row[..., 2], row[..., 5] = -peak, peak / 3
        row = np.asarray(jnp.asarray(row, dtype).astype(jnp.float32), np.float64)
        assert not np.isfinite(np.asarray(jnp.mean(jnp.square(jnp.asarray(row, jnp.float32)))))
        direction = jnp.asarray(generator.standard_normal(row.shape), dtype)
        output, pullback = jax.vjp(scaled, jnp.asarray(row, dtype))
        expected = _reference(row, ones, direction.astype(jnp.float32), eps=0.0)
        rtol = 1e-2 if dtype == jnp.bfloat16 else 1e-6
        assert _relative(output.astype(jnp.float32), expected[0]) <= rtol
        if peak <= 1e30:
            assert _relative(pullback(direction)[0].astype(jnp.float32), expected[1][0]) <= 2 * rtol


@pytest.mark.parametrize("prescale", HEAD_RMS_PRESCALES)
def test_nonfinite_rows_are_nan_and_leave_the_other_rows_alone(prescale):
    """A row with an infinite or NaN entry is NaN in every channel, with or without the prescale and a row shift
    (a NaN shift likewise); the finite rows beside it are what they are alone."""
    generator = np.random.default_rng(5)
    clean = generator.standard_normal((1, 6, 2, 8)) * 10.0 ** generator.uniform(-3, 17, (1, 6, 2, 1))
    gain = jnp.asarray(generator.uniform(0.5, 2.0, (2, 8)), jnp.float32)
    shift = generator.uniform(-5, 40, (1, 6, 2))
    scaled = partial(scale_attention_output, scaling="head_rms_norm", output_norm_scale=gain,
                     head_rms_prescale=prescale)
    for shifted in (False, True):
        arguments = {"output_shift": jnp.asarray(shift, jnp.float32)} if shifted else {}
        alone = np.asarray(scaled(jnp.asarray(clean, jnp.float32), **arguments))
        assert np.isfinite(alone).all()
        for bad in (np.inf, -np.inf, np.nan):
            poisoned = clean.copy()
            poisoned[0, 2, 1, 3] = bad
            result = np.asarray(scaled(jnp.asarray(poisoned, jnp.float32), **arguments))
            assert np.isnan(result[0, 2, 1]).all(), (shifted, bad)
            mask = np.ones(result.shape[:3], bool)
            mask[0, 2, 1] = False
            np.testing.assert_array_equal(result[mask], alone[mask])
    bad_shift = shift.copy()
    bad_shift[0, 4, 0] = np.nan
    result = np.asarray(scaled(jnp.asarray(clean, jnp.float32), output_shift=jnp.asarray(bad_shift, jnp.float32)))
    assert np.isnan(result[0, 4, 0]).all() and np.isfinite(np.delete(result.reshape(12, 8), 8, axis=0)).all()


# (shift, RMS of y): rows of ordinary size at shifts from far below to far beyond FP32's exponential, and rows
# whose own size is far from one, in both the normalizing and the linear range of the function.
SHIFTED_CASES = ((0.0, 1.0), (5.0, 1.0), (30.0, 1.0), (60.0, 1.0), (88.0, 1.0), (100.0, 1.0), (300.0, 1.0),
                 (2000.0, 1.0), (-5.0, 1.0), (-20.0, 1.0), (-40.0, 1.0), (-60.0, 1.0), (30.0, 1e-12), (40.0, 1e-14),
                 (50.0, 1e-14), (50.0, 1e-20), (80.0, 1e-30), (60.0, 1e-30), (20.0, 1e-12), (10.0, 1e-10),
                 (5.0, 1e-6), (0.0, 1e-25), (0.0, 1e-13), (0.0, 1e-3), (0.0, 1e6), (0.0, 1e15), (0.0, 1e25),
                 (-30.0, 1e15), (-40.0, 1e20), (40.0, 1e10), (10.0, 1e20), (-25.0, 1e17))


def test_composition_with_the_row_shift_matches_fp64_at_every_shift_and_size():
    """``scale_attention_output(e^(-m) o, offset m)`` with the prescale against the FP64 function of the same FP32
    rows and shifts: output and VJP to FP32 rounding at every case, including rows the unprescaled shifted
    arithmetic cannot evaluate (tiny rows at large shifts, whose mean square or ``rsqrt(x) / x`` leaves FP32) and
    rows inside the 9e14-1e18 band at shift zero."""
    direction, cotangent, gain = _probe(rows=64, seed=6)
    cotangent, gain = jnp.asarray(cotangent, jnp.float32), jnp.asarray(gain, jnp.float32)
    prescaled, plain = _evaluation("max_abs", shifted=True), _evaluation("none", shifted=True)
    broken = 0
    for shift, rms in SHIFTED_CASES:
        y = jnp.asarray(direction * rms, jnp.float32)
        offset = jnp.full(y.shape[:3], shift, jnp.float32)
        expected = _reference(y, gain, cotangent, shift=offset)
        output, dy, dgain = prescaled(y, gain, cotangent, offset)
        for label, got, want in (("output", output, expected[0]), ("dy", dy, expected[1][0]),
                                 ("dgain", dgain, expected[1][1])):
            assert np.isfinite(np.asarray(got)).all(), (shift, rms, label)
            assert _relative(got, want) <= 1e-6, (shift, rms, label, _relative(got, want))
        historical = plain(y, gain, cotangent, offset)
        broken += not (np.isfinite(np.asarray(historical[1])).all()
                       and _relative(historical[1], expected[1][0]) <= 1e-2)
    assert broken >= 6  # The cases this composition adds to what the shifted arithmetic alone evaluates.


def test_shifted_zero_and_vanishing_rows_keep_the_linear_function():
    """A zero row stays zero at any shift, with the exact slope ``e^m / sqrt(eps)`` as its gradient wherever FP32
    holds it and a finite one beyond (capped at e^80.5); a row in the linear range is ``y e^m / sqrt(eps)``."""
    gain = jnp.ones((1, 8), jnp.float32)
    scaled = partial(scale_attention_output, scaling="head_rms_norm", output_norm_scale=gain,
                     head_rms_prescale="max_abs")
    zero = jnp.zeros((1, 1, 1, 8), jnp.float32)
    cotangent = jnp.ones_like(zero)
    for offset in (-200.0, -50.0, 0.0, 20.0, 30.0, 36.0, 44.0, 60.0, 73.0, 74.0, 88.0, 100.0, 1e4, np.inf, -np.inf):
        output, pullback = jax.vjp(lambda y: scaled(y, output_shift=jnp.full((1, 1, 1), offset, jnp.float32)), zero)
        (gradient,) = pullback(cotangent)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(zero))
        assert np.isfinite(np.asarray(gradient)).all(), offset
        exact = math.exp(min(offset, 73.0)) / math.sqrt(EPS) if offset > -80 else 0.0
        if -80 < offset <= 73.0:
            np.testing.assert_allclose(np.asarray(gradient, np.float64), exact, rtol=1e-6)
        elif offset > 73.0:
            cap = math.exp(math.log(float(np.finfo(np.float32).max)) - SLOPE_HEADROOM)  # e^80.5
            assert float(gradient.min()) >= exact and float(gradient.max()) <= cap * 1.0001
        else:
            assert float(jnp.max(jnp.abs(gradient))) < 1e-30
    generator = np.random.default_rng(7)
    row = generator.standard_normal((1, 5, 1, 8))
    for offset, scale in ((0.0, 1e-20), (30.0, 1e-30), (-10.0, 1e-14), (45.0, 1e-34)):
        y = jnp.asarray(row * scale, jnp.float32)
        assert float(jnp.min(jnp.abs(y))) > float(jnp.finfo(jnp.float32).tiny)  # no subnormal input
        assert float(jnp.max(jnp.abs(y))) * math.exp(offset) < math.exp(-LINEAR_LOG_RANGE) * math.sqrt(EPS)
        expected = np.asarray(y, np.float64) * math.exp(offset) / math.sqrt(EPS)
        output = scaled(y, output_shift=jnp.full((1, 5, 1), offset, jnp.float32))
        np.testing.assert_allclose(np.asarray(output, np.float64), expected, rtol=2e-6)


@pytest.mark.parametrize("implementation", ("scan", "strips"))
@pytest.mark.parametrize("scale,largest", ((1.0, 5.0), (3.0, 45.0), (6.0, 190.0), (9.0, 440.0)))
def test_shifted_scan_and_strips_with_the_prescale_match_fp64(implementation, scale, largest):
    """The blocked backend's shifted exponential into the prescaled normalization against the FP64 function of the
    same FP32 inputs, at row maxima from a few units to far beyond FP32's e^88.7: output and VJP in Q, K, V and the
    gain, as without the prescale."""
    q, k, v, gain, cotangent = _arrays(scale=scale, seed=1)
    arrays = tuple(jnp.asarray(x, jnp.float32) for x in (q, k, v, gain))
    cotangent = jnp.asarray(cotangent, jnp.float32)
    attend = (partial(blocked_dot_product_attention, query_block_size=8) if implementation == "scan"
              else partial(strip_dot_product_attention, strip_size=16))

    def function(q, k, v, gain):
        output, shift = attend(q, k, v, **SHIFTED)
        return scale_attention_output(output, "head_rms_norm", output_norm_scale=gain, output_shift=shift,
                                      head_rms_prescale="max_abs")

    with jax.enable_x64():
        arrays64 = tuple(jnp.asarray(np.asarray(x, np.float64)) for x in arrays)
        output, pullback = jax.vjp(partial(_dense, shift="stop"), *arrays64)
        gradients = pullback(jnp.asarray(np.asarray(cotangent, np.float64)))
        expected = (np.asarray(output), [np.asarray(gradient) for gradient in gradients])
        from scaffolds.transformer.tests.test_exp_row_max_shift import _logits

        assert float(jnp.max(_logits(*arrays64[:2])[0])) >= largest
    with jax.default_matmul_precision("highest"):
        output, pullback = jax.jit(lambda *a: jax.vjp(function, *a))(*arrays)
        gradients = pullback(cotangent)
    assert _relative(output, expected[0]) <= 5e-6
    for name, got, want in zip("qkvg", gradients, expected[1], strict=True):
        assert np.isfinite(np.asarray(got)).all() and _relative(got, want) <= 2e-5, (name, _relative(got, want))


# The four polynomial recipes of the key-width runs (fixed Taylor, static, query, and the softplus sum of squares
# with its input power, z h_t(z)), the shifted exponential, and two recipes on the normalization's other paths.
FAMILIES = {
    "taylor": dict(attention_degree=8),
    "static": dict(attention_degree=8, attention_coefficients="learned", attention_coefficient_sharing="per_head",
                   attention_coefficient_scale="factorial"),
    "query": dict(attention_degree=8, attention_coefficients="learned", attention_coefficient_sharing="per_head",
                  attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                  attention_coefficient_kernel_weight_decay=True, attention_coefficient_amplitude_initial=0.0,
                  attention_coefficient_projection_initial_std=0.1, attention_coefficient_amplitude_scale="factorial",
                  attention_coefficient_scale="factorial"),
    "softplus_sos": dict(attention_degree=9, attention_coefficients="learned",
                         attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                         attention_coefficient_activation="tanh", attention_coefficient_kernel_weight_decay=True,
                         attention_coefficient_amplitude_initial=0.0,
                         attention_coefficient_projection_initial_std=0.1, attention_score_transform="softplus",
                         attention_coefficient_parameterization="sum_of_squares",
                         attention_coefficient_input_power=1),
    "shifted_exp": dict(attention_exp_shift="row_max"),
    "softmax": dict(attention_normalization=True),
    "path_exp": dict(positional_encoding="path", attention_blocked_implementation="scan"),
}
POLYNOMIAL_FAMILIES = ("taylor", "static", "query", "softplus_sos")
# Shapes: the CPU's (blocked backend) and the GPU's (Mosaic kernels: value width 64, a context of 512 tokens).
TINY = dict(vocab_size=64, max_seq_len=32, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2,
            qk_head_dim=4, value_head_dim=16, intermediate_size=64)
MOSAIC = dict(vocab_size=256, max_seq_len=512, num_layers=2, hidden_size=128, num_heads=2, num_kv_heads=2,
              qk_head_dim=4, value_head_dim=64, intermediate_size=256)
VALUE_FACTOR = 1e16  # on the value projections: every head output above 1e15


def _model_config(family, prescale, *, backend="blocked", shape=TINY, **updates):
    """A two-layer recipe with the canonical QK RMSNorm, head scales and head RMSNorm output scaling."""
    values = dict(shape, attention_backend=backend, attention_blocked_implementation="strips",
                  attention_normalization=False, attention_output_scaling="head_rms_norm",
                  attention_head_rms_prescale=prescale, rms_norm_fp32=True,
                  initialization="transformer_truncated_reference", qk_norm=True, qk_norm_type="rms_norm",
                  qk_norm_sharing="per_layer", qk_norm_head_scale=True)
    return ModelConfig(**{**values, **FAMILIES[family], **updates})


def _build(config, dtype=jnp.float32):
    """(graphdef, FP32 parameters) with the layers computing in ``dtype``, as training builds them."""
    from scaffolds.transformer.models import build_model

    model = build_model(config, nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)))
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    return nnx.split(model, nnx.Param)


def _path_text(path):
    return "/".join(str(getattr(entry, "key", getattr(entry, "idx", entry))) for entry in path)


def _scale_values(params, factor):
    """The parameters with every value projection times ``factor``: every head output scales with it, and head
    RMSNorm's result does not (up to eps)."""
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf * factor if "v_proj" in _path_text(path) else leaf, params)


def _loss(graphdef, tokens, dtype=jnp.float32):
    def loss(params):
        logits = nnx.merge(graphdef, params)(tokens[:, :-1]).astype(dtype)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tokens[:, 1:, None], axis=-1))

    return loss


def _fp64_head_rms_norm(y, scaling, *, output_norm_scale=None, eps=EPS, output_shift=None, head_rms_prescale="none"):
    """Head RMSNorm as written, in y's own dtype: the FP64 model's normalization."""
    assert scaling == "head_rms_norm" and y.dtype == jnp.float64
    shift = None if output_shift is None else output_shift.astype(y.dtype)[..., None]
    return _plain(y, eps, shift) * output_norm_scale.astype(y.dtype)[None, None]


def _fp64_gradients(config, params, tokens, monkeypatch):
    """Loss and gradient of the recipe's model computing in FP64 on the blocked backend: the parameters upcast,
    every layer in FP64, the scan and the strips in FP64 arithmetic, and head RMSNorm as written in FP64. The
    scaffold's explicit FP32 steps (QK RMSNorm, RoPE tables, coefficient generators) keep their FP32 rounding."""
    from scaffolds.transformer.models.components.normalization import RMSNorm

    with jax.enable_x64(), monkeypatch.context() as patch:
        model = nnx.merge(*_build(replace(config, attention_backend="blocked", attention_head_rms_prescale="none")))
        for _, module in nnx.iter_graph(model):
            if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
                module.dtype = jnp.float64
            if isinstance(module, RMSNorm):
                module.fp32 = False
        graphdef, _ = nnx.split(model, nnx.Param)
        params64 = jax.tree.map(lambda leaf: jnp.asarray(np.asarray(leaf, np.float64)), params)
        patch.setattr(reference, "blocked_dot_product_attention", partial(reference._blocked_scan, compute=jnp.float64))
        patch.setattr(multihead, "scale_attention_output", _fp64_head_rms_norm)
        value, gradient = jax.value_and_grad(_loss(graphdef, tokens, jnp.float64))(params64)
        assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(gradient))
        return float(value), jax.tree.map(lambda leaf: np.asarray(leaf, np.float64), gradient)


GROUPS = (("qk", ("q_proj", "k_proj", "qk_norm")), ("v", ("v_proj",)), ("o", ("o_proj",)),
          ("gain", ("output_norm_scale",)), ("coefficients", ("polynomial", "coefficient")))


def _group_errors(actual, expected):
    """Relative L2 error per parameter group: Q/K projections and QK gains, V, O, head gains, kernel coefficients,
    and everything else."""
    sums = {}
    for (path, got), want in zip(jax.tree_util.tree_leaves_with_path(actual), jax.tree.leaves(expected), strict=True):
        text = _path_text(path)
        group = next((name for name, parts in GROUPS if any(part in text for part in parts)), "other")
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        assert np.isfinite(got).all(), text
        total = sums.setdefault(group, [0.0, 0.0])
        total[0] += float(np.sum((got - want) ** 2))
        total[1] += float(np.sum(want ** 2))
    return {group: math.sqrt(difference / max(norm, 1e-300)) for group, (difference, norm) in sums.items()}


def _tokens(config, *, batch=2, seed=5):
    return jnp.asarray(np.random.default_rng(seed).integers(0, config.vocab_size, (batch, config.max_seq_len + 1)),
                       jnp.int32)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_model_with_the_prescale_is_the_model_without_it_at_ordinary_scales(family):
    """Two-layer models of every family (the polynomial kernels, the shifted exponential, softmax with head RMSNorm,
    and PaTH, which has its own call of the normalization): at initialization, where head outputs are of ordinary
    size, loss and every gradient with the prescale are those without it to FP32 rounding."""
    tokens = _tokens(_model_config(family, "none"))
    results = {}
    for prescale in HEAD_RMS_PRESCALES:
        graphdef, params = _build(_model_config(family, prescale))
        with jax.default_matmul_precision("highest"):
            results[prescale] = jax.jit(jax.value_and_grad(_loss(graphdef, tokens)))(params)
    np.testing.assert_allclose(results["max_abs"][0], results["none"][0], rtol=1e-6)
    errors = _group_errors(results["max_abs"][1], results["none"][1])
    assert max(errors.values()) <= 2e-5, errors


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16), ids=("fp32", "bf16"))
@pytest.mark.parametrize("family", (*POLYNOMIAL_FAMILIES, "shifted_exp"))
def test_model_gradients_match_fp64_with_every_head_output_above_1e15(family, dtype, monkeypatch):
    """A two-layer model at Q/K width 4 whose value projections are scaled by 1e16, so that every head output lies
    in the band where the plain backward fails. With the prescale every gradient matches the FP64 model's: to 1e-5
    with FP32 layers, and with BF16 layers as closely as the unscaled BF16 model matches its own FP64 counterpart
    (BF16 rounding, 1% to 3% of a group's norm: within three times that distance plus 1%). Without it the Q/K
    gradient is off by more than half its norm."""
    config = _model_config(family, "max_abs")
    tokens = _tokens(config)
    _, params = _build(config)
    errors = {}
    for factor in (1.0, VALUE_FACTOR):
        scaled = _scale_values(params, factor)
        expected_loss, expected = _fp64_gradients(config, scaled, tokens, monkeypatch)
        for prescale in HEAD_RMS_PRESCALES:
            graphdef, _ = _build(replace(config, attention_head_rms_prescale=prescale), dtype)
            with jax.default_matmul_precision("highest"):
                value, gradient = jax.jit(jax.value_and_grad(_loss(graphdef, tokens)))(scaled)
            np.testing.assert_allclose(value, expected_loss, rtol=1e-5 if dtype == jnp.float32 else 2e-3)
            errors[prescale, factor] = _group_errors(gradient, expected)
    forced, ordinary = errors["max_abs", VALUE_FACTOR], errors["max_abs", 1.0]
    for group, error in forced.items():
        bound = 1e-5 if dtype == jnp.float32 else 3 * ordinary[group] + 1e-2
        assert error <= bound, (group, error, bound, errors)
    assert errors["none", VALUE_FACTOR]["qk"] >= 0.5, errors
    assert max(errors["none", 1.0].values()) <= (1e-5 if dtype == jnp.float32 else 0.06), errors


@pytest.mark.parametrize("family", ("taylor", "softplus_sos"))
def test_scaling_the_values_leaves_the_prescaled_model_unchanged(family):
    """Head RMSNorm removes any positive scale of a head's values, up to eps. With the prescale, a model whose
    value projections are scaled by 1e16 has the unscaled model's loss and gradients, the value projections' own
    divided by 1e16, to the size of eps next to the unscaled rows' mean squares (5e-4 of the coefficient gradient
    for the softplus sum of squares, whose smallest initial outputs are near 0.1); without it the scaled model's
    Q/K gradient is not the unscaled one's."""
    config = _model_config(family, "max_abs")
    tokens = _tokens(config)
    graphdef, params = _build(config)
    evaluate = jax.jit(jax.value_and_grad(_loss(graphdef, tokens)))
    with jax.default_matmul_precision("highest"):
        unscaled = evaluate(params)
        scaled = evaluate(_scale_values(params, VALUE_FACTOR))
        plain = jax.jit(jax.value_and_grad(_loss(_build(replace(config, attention_head_rms_prescale="none"))[0],
                                                 tokens)))(_scale_values(params, VALUE_FACTOR))
    np.testing.assert_allclose(scaled[0], unscaled[0], rtol=1e-6)
    carried = _scale_values(scaled[1], VALUE_FACTOR)  # d/d(1e16 W) = (d/dW) / 1e16
    errors = _group_errors(carried, unscaled[1])
    assert max(errors.values()) <= 2e-3, errors
    assert _group_errors(_scale_values(plain[1], VALUE_FACTOR), unscaled[1])["qk"] >= 0.5


def test_layer_passes_the_prescale_and_refuses_it_without_head_rms_norm():
    common = dict(hidden_size=32, num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                  attention_normalization=False, backend="blocked", attention_degree=4)
    x = jnp.asarray(np.random.default_rng(8).standard_normal((2, 11, 32)), jnp.float32)
    gain = {"output_norm_scale": jnp.asarray(np.random.default_rng(9).uniform(0.5, 2.0, (4, 8)), jnp.float32)}
    layers = {prescale: MultiHeadAttention(**common, attention_output_scaling="head_rms_norm",
                                           attention_head_rms_prescale=prescale, rngs=nnx.Rngs(8))
              for prescale in HEAD_RMS_PRESCALES}
    with jax.default_matmul_precision("highest"):
        plain = layers["none"](x, attention_parameters=dict(gain))
        np.testing.assert_allclose(layers["max_abs"](x, attention_parameters=dict(gain)), plain, rtol=2e-5, atol=1e-6)
        # Scaling every value by 1e20 leaves the prescaled layer's output unchanged; FP32 cannot square such rows.
        layers["max_abs"].v_proj.kernel[...] = layers["max_abs"].v_proj.kernel[...] * 1e20
        np.testing.assert_allclose(layers["max_abs"](x, attention_parameters=dict(gain)), plain, rtol=2e-5, atol=1e-6)
    for scaling in ("none", "inverse_position"):
        with pytest.raises(ValueError, match="max-abs prescale evaluates head RMSNorm"):
            MultiHeadAttention(**common, attention_output_scaling=scaling, attention_head_rms_prescale="max_abs",
                               rngs=nnx.Rngs(8))
    with pytest.raises(ValueError, match="Unknown head RMSNorm prescale"):
        MultiHeadAttention(**common, attention_output_scaling="head_rms_norm", attention_head_rms_prescale="rms",
                           rngs=nnx.Rngs(8))


def test_function_refuses_inconsistent_arguments():
    y = jnp.ones((2, 5, 3, 6), jnp.float32)
    gain = jnp.ones((3, 6), jnp.float32)
    with pytest.raises(ValueError, match="Unknown head RMSNorm prescale"):
        scale_attention_output(y, "head_rms_norm", output_norm_scale=gain, head_rms_prescale="rms")
    for scaling in ("none", "inverse_position"):
        with pytest.raises(ValueError, match="Only head RMSNorm takes a prescale"):
            scale_attention_output(y, scaling, head_rms_prescale="max_abs")
    with pytest.raises(ValueError, match="shape"):
        scale_attention_output(y, "head_rms_norm", output_norm_scale=jnp.ones((6,)), head_rms_prescale="max_abs")
    with pytest.raises(ValueError, match="positive epsilon"):
        scale_attention_output(y, "head_rms_norm", output_norm_scale=gain, head_rms_prescale="max_abs", eps=0.0,
                               output_shift=jnp.zeros((2, 5, 3)))
    # Unshifted, eps = 0 is the scale-free normalization itself.
    output = scale_attention_output(3e20 * y, "head_rms_norm", output_norm_scale=gain, head_rms_prescale="max_abs",
                                    eps=0.0)
    np.testing.assert_allclose(output, np.ones(y.shape), rtol=1e-6)
    assert scale_attention_output(y, "none") is y


@pytest.mark.parametrize("settings,error", [
    ({"attention_output_scaling": "none"}, "head RMSNorm output scaling"),
    ({"attention_output_scaling": "inverse_position"}, "head RMSNorm output scaling"),
    ({"attention_output_scaling": "none", "attention_normalization": True}, "head RMSNorm output scaling"),
    ({"attention_head_rms_prescale": "rms"}, "Unknown model.attention_head_rms_prescale"),
])
def test_recipes_without_head_rms_norm_refuse_the_prescale(settings, error):
    base = dict(attention_backend="blocked", attention_normalization=False, attention_output_scaling="head_rms_norm",
                attention_head_rms_prescale="max_abs")
    with pytest.raises(ValueError, match=error):
        ModelConfig(**{**base, **settings})


@pytest.mark.parametrize("settings", [
    {}, {"attention_degree": 8}, {"attention_normalization": True}, {"attention_backend": "fused"},
    {"attention_backend": "auto"}, {"positional_encoding": "path"}, {"positional_encoding": "none"},
    {"attention_exp_shift": "row_max"}, {"attention_forget_gate": True},
    {"attention_backend": "mosaic", "attention_degree": 8, "qk_head_dim": 4, "value_head_dim": 64,
     "hidden_size": 128, "num_heads": 2, "num_kv_heads": 2},
    {"attention_degree": 4, "attention_coefficients": "learned"},
])
def test_the_prescale_applies_wherever_head_rms_norm_does(settings):
    """Any kernel, normalized or not, on any backend and positional encoding: fresh recipes use it."""
    model = ModelConfig(**{**dict(attention_backend="blocked", attention_normalization=False,
                                  attention_output_scaling="head_rms_norm"), **settings})
    assert model.attention_head_rms_prescale == "none" and fresh_recipe_uses_head_rms_prescale(model)
    model.attention_head_rms_prescale = "max_abs"
    model.__post_init__()


@pytest.mark.parametrize("settings", [
    {}, {"attention_normalization": False}, {"attention_normalization": False, "attention_degree": 4},
    {"attention_normalization": False, "attention_output_scaling": "inverse_position"},
    {"attention_mechanism": "gla", "qk_norm": False},
])
def test_fresh_recipes_without_head_rms_norm_keep_none(settings):
    assert not fresh_recipe_uses_head_rms_prescale(ModelConfig(**settings))


def test_recipes_without_the_field_are_not_prescaled():
    assert ModelConfig().attention_head_rms_prescale == "none"
    config = Config(model=ModelConfig(attention_backend="blocked", attention_normalization=False,
                                      attention_degree=4, attention_output_scaling="head_rms_norm"))
    validate_config(config)
    current = config_to_dict(config)
    assert current["model"]["attention_head_rms_prescale"] == "none"
    legacy = json.loads(json.dumps(current))
    del legacy["model"]["attention_head_rms_prescale"]
    assert config_dicts_equal(current, legacy)
    config.model.attention_head_rms_prescale = "max_abs"
    assert not config_dicts_equal(config_to_dict(config), legacy)


def _training_config(path, **settings):
    from scaffolds.transformer.tests.test_training import tiny_config

    config = tiny_config(path)
    config.model.attention_backend = "blocked"
    config.model.attention_normalization = False
    config.model.attention_output_scaling = "head_rms_norm"
    config.model.attention_head_rms_prescale = "max_abs"
    config.model.attention_blocked_implementation = "strips"
    config.model.qk_norm, config.model.qk_norm_type, config.model.qk_norm_sharing = True, "rms_norm", "per_layer"
    config.model.qk_norm_head_scale = True
    for key, value in settings.items():
        setattr(config.model, key, value)
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    validate_config(config)
    return config


def test_identity_acceptance_and_continuations_name_the_prescale(tmp_path):
    from scaffolds.transformer.acceptance import (
        HEAD_RMS_GPU_TESTS, attention_test_paths, matches_softmax_backbone_initialization,
        matches_softmax_initialization,
    )
    from scaffolds.transformer.training.diagnostics import attention_identity

    config = _training_config(tmp_path, attention_degree=4)
    identity = attention_identity(config)
    assert identity["attention_head_rms_prescale"] == "max_abs"
    assert identity["attention_formula"].endswith("an FP32 gradient that is exact at any output scale")
    assert "divided by its largest magnitude" in identity["attention_formula"]
    shifted = attention_identity(_training_config(tmp_path, attention_exp_shift="row_max"))
    assert shifted["attention_exp_shift"] == "row_max" and shifted["attention_head_rms_prescale"] == "max_abs"
    assert (shifted["attention_formula"].index("shifted by their causal maximum")
            < shifted["attention_formula"].index("divided by its largest magnitude"))
    recipe = config_to_dict(config)
    # CPU: the module. GPU: its two device cases, which run (and never skip) on any GPU and backend.
    device_cases = [f"{TEST_PATH}::{test}" for test in HEAD_RMS_GPU_TESTS]
    assert len(device_cases) == 2 and all(test in globals() for test in HEAD_RMS_GPU_TESTS)
    for resolved in ("blocked", "mosaic", "fused"):
        assert TEST_PATH in attention_test_paths(recipe, resolved, True)
        selected = attention_test_paths(recipe, resolved, False)
        assert all(case in selected for case in device_cases) and TEST_PATH not in selected
    config.model.attention_head_rms_prescale = "none"
    plain = attention_identity(config)
    assert "attention_head_rms_prescale" not in plain
    assert plain["attention_formula"] == identity["attention_formula"].split("; head RMSNorm evaluated")[0]
    for cpu in (True, False):
        assert not any(TEST_PATH in path for path in attention_test_paths(config_to_dict(config), "blocked", cpu))
    # The softmax controls of the initialization checks drop the prescale with the output scaling: the backbone
    # check passes, and the exact-softmax check reaches its comparison (head RMSNorm's gains are not softmax's).
    prescaled = _training_config(tmp_path, attention_degree=4)
    assert matches_softmax_backbone_initialization(prescaled)
    with pytest.raises(RuntimeError, match="differs from the exact-softmax counterpart"):
        matches_softmax_initialization(prescaled)


def test_a_continuation_may_change_the_prescale_as_an_execution_setting(tmp_path):
    """Like the backend and the exponential's shift, the prescale changes no parameter or optimizer shape, and it
    keeps the function: a prescaled child restores a parent trained without it."""
    import hashlib

    import yaml

    from scaffolds.transformer.training.continuation import _parent_config

    parent = _training_config(tmp_path / "parent", attention_degree=4)
    parent.model.attention_head_rms_prescale = "none"
    validate_config(parent)
    path = tmp_path / "parent.yaml"
    path.write_text(yaml.safe_dump(config_to_dict(parent)))
    child = _training_config(tmp_path / "child", attention_degree=4)
    child.continuation.parent_config = str(path)
    child.continuation.parent_config_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    child.continuation.preserve_all_parameters = True
    assert _parent_config(child).model.attention_head_rms_prescale == "none"
    child.model.attention_degree = 2
    with pytest.raises(ValueError, match="Continuation changes parent model.attention_degree"):
        _parent_config(child)


def test_telemetry_statistics_are_those_of_the_output_before_the_normalization():
    """Telemetry reads the head outputs before their normalization, so every statistic keeps its meaning and its
    value with the prescale: the output RMS statistics are the unscaled output's and ``head_rms_norm_gain_max`` is
    the factor ``1 / sqrt(mean(y^2) + eps)`` of the function, whatever units evaluate it."""
    from scaffolds.transformer.training import telemetry

    common = dict(hidden_size=32, num_heads=4, num_kv_heads=4, qk_head_dim=8, value_head_dim=8,
                  attention_normalization=False, backend="blocked", attention_degree=4,
                  attention_output_scaling="head_rms_norm")
    x = jnp.asarray(np.random.default_rng(8).standard_normal((2, 11, 32)), jnp.float32)
    gain = {"output_norm_scale": jnp.ones((4, 8), jnp.float32)}
    recorded = {}
    for prescale in HEAD_RMS_PRESCALES:
        layer = MultiHeadAttention(**common, attention_head_rms_prescale=prescale, rngs=nnx.Rngs(8))
        layer.v_proj.kernel[...] = layer.v_proj.kernel[...] * 1e6
        with telemetry.collect() as collector:
            layer(x, attention_parameters=dict(gain))
        recorded[prescale] = collector
    assert set(recorded["max_abs"].values) == set(recorded["none"].values) >= {
        "attention_output_rms", "attention_output_head_rms_min", "head_rms_norm_gain_max"}
    assert recorded["max_abs"].reductions == recorded["none"].reductions
    for key, value in recorded["none"].values.items():
        np.testing.assert_array_equal(np.asarray(recorded["max_abs"].values[key]), np.asarray(value))
    assert float(recorded["max_abs"].values["attention_output_rms"]) > 1e10  # a mean square: the unscaled output's


@pytest.mark.parametrize("settings", [
    {"attention_degree": 2, "attention_coefficients": "learned", "attention_coefficient_sharing": "per_head"},
    {"attention_exp_shift": "row_max"},
], ids=("learned_polynomial", "shifted_exp"))
def test_training_resumes_exactly_and_records_the_prescale(tmp_path, settings):
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
    from scaffolds.transformer.training.state import create_state

    config = _training_config(tmp_path / "full", **settings)
    loop.run_training(config)
    resumed_config = replace(config, train=replace(config.train, workdir=tmp_path / "resumed"))
    interrupted = loop.run_training(resumed_config, stop_after_step=1, audit_checkpoint=True)
    resumed = loop.run_training(resumed_config, audit_checkpoint=True)
    assert interrupted["checkpoint_audit"]["final_state_sha256"] == resumed["checkpoint_audit"]["start_state_sha256"]
    assert resumed["restored_step"] == 1
    states = []
    for current in (config, resumed_config):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
            assert step == config.optim.total_steps
            states.append(state)
        finally:
            manager.close()
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    assert np.any(np.asarray(states[0].params["output_norm_scale"][...]) != 1)
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["attention_head_rms_prescale"] == "max_abs"
    assert metadata["attention_output_scaling"] == "head_rms_norm"
    measurements = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    reports = [row["attention_diagnostics"] for row in measurements if "attention_diagnostics" in row]
    assert reports and all(report["valid"] for report in reports)


# ----------------------------------------------------------------------------------------------------------------
# GPU: the device's own FP32 arithmetic, and the Mosaic kernels through the prescaled normalization.

def _device_capability():
    return str(jax.devices()[0].compute_capability) if jax.default_backend() == "gpu" else None


def _require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation")


def _require_mosaic(family, width):
    """The Mosaic kernels a case needs, on the GPUs that admit and qualify them: the query-coefficient kernel on
    H100, B200 and B300 (Q/K width 4 on H100 and B200 alone), the transformed kernels on H100 and B200."""
    _require_gpu()
    capability = _device_capability()
    if capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("The Mosaic kernels run on H100, B200 and B300")
    if capability == "10.3" and (family == "softplus_sos" or width == 4):
        pytest.skip("B300 does not admit the transformed kernels, and its query kernel is not qualified at Q/K "
                    "width 4")


@pytest.mark.gpu
def test_gpu_fp32_gradient_matches_fp64_at_every_scale_of_the_measured_probe(record_property):
    """The probe on the device's own FP32 arithmetic (XLA's GPU code keeps subnormal numbers, unlike its CPU code,
    so the plain backward degrades gradually there). Prescaled: output and VJP match host FP64 to 1e-6 at every
    scale from 1e-25 to 1e30, with no radial component. Plain: its gradient keeps the whole radial component from
    9e14 to 1e18. Every scale's errors are recorded."""
    _require_gpu()
    direction, cotangent, gain = _probe()
    cotangent, gain = jnp.asarray(cotangent, jnp.float32), jnp.asarray(gain, jnp.float32)
    prescaled, plain = _evaluation("max_abs"), _evaluation("none")
    report = []
    for rms in PROBE_RMS:
        y = jnp.asarray(direction * rms, jnp.float32)
        expected = _reference(y, gain, cotangent)
        entry = {"rms": rms}
        for name, function in (("max_abs", prescaled), ("none", plain)):
            output, dy, dgain = (np.asarray(array, np.float64) for array in function(y, gain, cotangent))
            entry[name] = {"output": _relative(output, expected[0]), "dy": _relative(dy, expected[1][0]),
                           "dgain": _relative(dgain, expected[1][1]),
                           "radial_fraction": _radial_fraction(y, gain, cotangent, dy)}
        report.append(entry)
    record_property("head_rms_probe", json.dumps(report))
    for entry in report:
        rms, errors = entry["rms"], entry["max_abs"]
        assert max(errors["output"], errors["dy"], errors["dgain"]) <= 1e-6, entry
        if 1.0 <= rms <= 1e25:
            assert errors["radial_fraction"] <= 1e-5, entry
        if rms in FAILING_RMS:
            assert entry["none"]["dy"] >= 0.05 and entry["none"]["radial_fraction"] >= 0.99, entry
        elif rms <= 1e12 or rms >= 1e19:
            assert entry["none"]["dy"] <= 1e-6, entry


@pytest.mark.gpu
def test_gpu_composition_with_the_row_shift_matches_fp64(record_property):
    """The shifted composition's table (shifts of -60 to 2,000, rows of 1e-30 to 1e25) on the device: output and
    VJP against host FP64 to 1e-6; zero rows stay zero with finite gradients at every shift."""
    _require_gpu()
    direction, cotangent, gain = _probe(rows=64, seed=6)
    cotangent, gain = jnp.asarray(cotangent, jnp.float32), jnp.asarray(gain, jnp.float32)
    prescaled = _evaluation("max_abs", shifted=True)
    report = []
    for shift, rms in SHIFTED_CASES:
        y = jnp.asarray(direction * rms, jnp.float32)
        offset = jnp.full(y.shape[:3], shift, jnp.float32)
        expected = _reference(y, gain, cotangent, shift=offset)
        output, dy, dgain = prescaled(y, gain, cotangent, offset)
        errors = {"output": _relative(output, expected[0]), "dy": _relative(dy, expected[1][0]),
                  "dgain": _relative(dgain, expected[1][1])}
        report.append({"shift": shift, "rms": rms, **errors})
        assert all(np.isfinite(np.asarray(array)).all() for array in (output, dy, dgain)), (shift, rms)
        assert max(errors.values()) <= 1e-6, (shift, rms, errors)
    record_property("head_rms_shifted_composition", json.dumps(report))
    zero = jnp.zeros((1, 4, 2, 64), jnp.float32)
    for shift in (-200.0, 0.0, 30.0, 36.0, 44.0, 73.0, 100.0, 1e4):
        output, dy, dgain = prescaled(zero, gain, jnp.ones_like(zero), jnp.full((1, 4, 2), shift, jnp.float32))
        assert not np.asarray(output).any() and np.isfinite(np.asarray(dy)).all(), shift


def _kernel_operands(family, width, *, time=2048, heads=2, seed=41):
    """BF16 Q, K, V at the production context, FP32 coefficients and gain, a BF16 cotangent."""
    generator = np.random.default_rng(seed + width)
    q, k = (jnp.asarray(0.7 * generator.standard_normal((1, time, heads, width)), jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(generator.standard_normal((1, time, heads, 64)), jnp.bfloat16)
    cotangent = jnp.asarray(generator.standard_normal(v.shape), jnp.bfloat16)
    gain = jnp.asarray(1.0 + 0.1 * generator.standard_normal((heads, 64)), jnp.float32)
    taylor = np.array([1 / math.factorial(order) for order in range(9)])
    if family == "taylor":
        coefficients = None
    elif family == "static":
        coefficients = jnp.asarray(taylor * (1 + 0.5 * np.tanh(generator.standard_normal((heads, 9)))), jnp.float32)
    elif family == "query":
        coefficients = jnp.asarray(taylor * (1 + 0.5 * np.tanh(generator.standard_normal((1, time, heads, 9)))),
                                   jnp.float32)
    else:
        # z h_t(z), h_t = A_t^2 + B_t^2 with deg A = 4, deg B = 3: degree nine with a structural zero constant.
        a = generator.standard_normal((1, time, heads, 5)) * taylor[:5]
        b = generator.standard_normal((1, time, heads, 4)) * taylor[:4]
        square = np.zeros((1, time, heads, 9))
        for i in range(5):
            for j in range(5):
                square[..., i + j] += a[..., i] * a[..., j]
        for i in range(4):
            for j in range(4):
                square[..., i + j] += b[..., i] * b[..., j]
        coefficients = jnp.asarray(np.concatenate([np.zeros((1, time, heads, 1)), square], axis=-1), jnp.float32)
    return (q, k, v), coefficients, gain, cotangent


def _mosaic_kernel(family, q, k, v, coefficients):
    from scaffolds.transformer.attention.backends.mosaic import api
    from scaffolds.transformer.attention.unnormalized import unnormalized_attention

    if family == "taylor":
        return unnormalized_attention(q, k, v, attention_degree=8, backend="mosaic")
    if family == "static":
        return unnormalized_attention(q, k, v, attention_degree=8, backend="mosaic",
                                      polynomial_coefficients=coefficients)
    if family == "query":
        return api.query_polynomial_attention(q, k, v, coefficients)
    return api.transformed_query_polynomial_attention(q, k, v, coefficients, transform="softplus")


def _dense_kernel(family, q, k, v, coefficients):
    """The kernel's function in the operands' dtype (FP64): every causal pair, every power explicit."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    z = jnp.where(causal, jax.nn.softplus(scores) if family == "softplus_sos" else scores, 0.0)
    if family == "taylor":
        coefficients = jnp.asarray([1 / math.factorial(order) for order in range(9)], scores.dtype)
    if coefficients.ndim == 1:
        coefficients = jnp.broadcast_to(coefficients, (q.shape[2], coefficients.shape[-1]))
    if coefficients.ndim == 2:
        coefficients = jnp.broadcast_to(coefficients[None, None], (q.shape[0], q.shape[1], *coefficients.shape))
    per_query = jnp.swapaxes(coefficients, 1, 2)  # [batch, heads, time, orders]
    weights = sum(per_query[..., order, None] * z ** order for order in range(per_query.shape[-1]))
    return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, weights, 0.0), v)


KERNEL_BOUNDS = {"output": 0.02, "dq": 0.06, "dk": 0.06, "dv": 0.05, "dgain": 0.05, "dcoefficients": 0.06}


@pytest.mark.gpu
@pytest.mark.parametrize("width", (4, 64))
@pytest.mark.parametrize("family", POLYNOMIAL_FAMILIES)
def test_gpu_mosaic_kernels_through_the_prescaled_normalization_match_fp64(family, width, record_property):
    """Each Mosaic family at the production context (2,048 tokens, value width 64, degree eight; the softplus sum
    of squares at degree nine) into head RMSNorm with the prescale, against the dense FP64 function of the same
    BF16 operands: output and VJP in Q, K, V, the gain and the coefficients within the kernels' BF16 precision,
    with the values at their own scale and scaled by 1e16 (head outputs above 1e15), where the errors stay what
    they are at the values' own scale (within twice that plus 1%). Without the prescale the Q gradient at the
    forced scale is off by more than 15% of its norm (23% to 110% when the projection term is lost entirely)."""
    _require_mosaic(family, width)
    (q, k, v), coefficients, gain, cotangent = _kernel_operands(family, width)
    differentiated = (q, k, v, gain) if family == "taylor" else (q, k, v, gain, coefficients)
    names = ("dq", "dk", "dv", "dgain", "dcoefficients")[:len(differentiated)]

    def production(prescale, q, k, v, gain, coefficients=coefficients):
        return scale_attention_output(_mosaic_kernel(family, q, k, v, coefficients), "head_rms_norm",
                                      output_norm_scale=gain, head_rms_prescale=prescale)

    def exact(q, k, v, gain, coefficients=None):
        return _plain(_dense_kernel(family, q, k, v, coefficients)) * gain[None, None]

    report = {}
    for factor in (1.0, VALUE_FACTOR):
        operands = (differentiated[0], differentiated[1], (v.astype(jnp.float32) * factor).astype(jnp.bfloat16),
                    *differentiated[3:])
        with jax.enable_x64():
            operands64 = tuple(jnp.asarray(np.asarray(array.astype(jnp.float32), np.float64)) for array in operands)
            output, pullback = jax.vjp(exact, *operands64)
            gradients = pullback(jnp.asarray(np.asarray(cotangent.astype(jnp.float32), np.float64)))
            expected = (np.asarray(output), [np.asarray(gradient) for gradient in gradients])
        for prescale in HEAD_RMS_PRESCALES:
            output, pullback = jax.jit(lambda *a, prescale=prescale: jax.vjp(partial(production, prescale), *a))(
                *operands)
            gradients = pullback(cotangent)
            assert output.dtype == jnp.bfloat16
            errors = {"output": _relative(output.astype(jnp.float32), expected[0])}
            for name, got, want in zip(names, gradients, expected[1], strict=True):
                errors[name] = _relative(got.astype(jnp.float32), want)
            report[f"{prescale} x{factor:g}"] = errors
    record_property("head_rms_mosaic_kernel", json.dumps({"family": family, "width": width, **report}))
    ordinary, forced = report["max_abs x1"], report[f"max_abs x{VALUE_FACTOR:g}"]
    for name, bound in KERNEL_BOUNDS.items():
        if name in ordinary:
            assert ordinary[name] <= bound, (name, report)
            assert forced[name] <= 2 * ordinary[name] + 1e-2, (name, report)
    assert report[f"none x{VALUE_FACTOR:g}"]["dq"] >= 0.15, report
    assert max(report["none x1"].values()) <= max(KERNEL_BOUNDS.values()), report


@pytest.mark.gpu
@pytest.mark.parametrize("width", (4, 64))
@pytest.mark.parametrize("family", POLYNOMIAL_FAMILIES)
def test_gpu_mosaic_model_step_matches_fp64_with_every_head_output_above_1e15(family, width, monkeypatch,
                                                                             record_property):
    """One training gradient of a two-layer BF16 model on the Mosaic kernels (production dispatch: coefficient
    generators, kernels, telemetry-free forward, head RMSNorm with the prescale) at Q/K widths 4 and 64, with the
    value projections scaled by 1e16 so that every head output is above 1e15: the loss and every parameter
    gradient against the same model in FP64 on the blocked backend, as close as the unscaled model is to its own
    FP64 counterpart (BF16 rounding: within three times that distance plus 1%, and within 15% of each group's
    norm). Without the prescale the Q/K gradient is off by more than half its norm."""
    _require_mosaic(family, width)
    shape = dict(MOSAIC, qk_head_dim=width)
    config = _model_config(family, "max_abs", backend="mosaic", shape=shape)
    tokens = _tokens(config, batch=1)
    _, params = _build(config)
    report = {}
    for factor in (1.0, VALUE_FACTOR):
        scaled = _scale_values(params, factor)
        expected_loss, expected = _fp64_gradients(config, scaled, tokens, monkeypatch)
        for prescale in HEAD_RMS_PRESCALES:
            graphdef, _ = _build(replace(config, attention_head_rms_prescale=prescale), jnp.bfloat16)
            value, gradient = jax.jit(jax.value_and_grad(_loss(graphdef, tokens)))(scaled)
            report[f"{prescale} x{factor:g}"] = {"loss": float(value), "fp64_loss": expected_loss,
                                                 **_group_errors(gradient, expected)}
    record_property("head_rms_mosaic_model", json.dumps({"family": family, "width": width, **report}))
    ordinary, forced = report["max_abs x1"], report[f"max_abs x{VALUE_FACTOR:g}"]
    for group in ordinary:
        if group in ("loss", "fp64_loss"):
            continue
        assert ordinary[group] <= 0.1, (group, report)
        assert forced[group] <= min(3 * ordinary[group] + 1e-2, 0.15), (group, report)
    np.testing.assert_allclose(forced["loss"], forced["fp64_loss"], rtol=5e-3)
    assert report[f"none x{VALUE_FACTOR:g}"]["qk"] >= 0.5, report
