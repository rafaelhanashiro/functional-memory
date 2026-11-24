"""Row-power shifted per-query polynomials against dense FP64, their kernel arithmetic, and the layer they serve."""
from dataclasses import replace
import json
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.fused_shifted_polynomial import fused_shifted_polynomial
from scaffolds.transformer.attention.output_scaling import max_abs_head_rms
from scaffolds.transformer.attention.shifted_polynomial import TRANSFORMS, shifted_polynomial_attention
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_softplus_sos import _config

DEGREES = (9, 17)


def _phi(s, transform):
    return jnp.logaddexp(s, 0.0) if transform == "softplus" else s


def _oracle(q, k, v, c, transform, prefactor="none"):
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    scores = jnp.where(causal, scores, 0)
    z = _phi(scores, transform)
    weights = sum(c[..., r].transpose(0, 2, 1)[..., None] * z ** r for r in range(c.shape[-1]))
    shift = jnp.zeros(weights.shape[:-1] + (1,), weights.dtype)
    if prefactor == "softplus":
        weights = jnp.logaddexp(scores, 0.0) * weights
    elif prefactor == "exp":
        # e^s passes FP64's range at the stress scores: e^(s - m) with m = max(0, max_i s_i) per row, and head
        # RMSNorm's epsilon times e^(-2m), the same normalized function.
        shift = jax.lax.stop_gradient(jnp.maximum(0.0, jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1,
                                                               keepdims=True)))
        weights = jnp.exp(scores - shift) * weights
    output = jnp.einsum("bhqk,bkhd->bqhd", jnp.where(causal, weights, 0), v)
    eps = 1e-6 * jnp.exp(-2.0 * shift).transpose(0, 2, 1, 3)
    return output / jnp.sqrt(jnp.mean(jnp.square(output), axis=-1, keepdims=True) + eps)


def _inputs(time, stress, degree, prefactor="none"):
    rng = np.random.default_rng(124 + degree)
    q, k, v = (rng.normal(size=(1, time, 1, d)).astype(np.float32) / 2 for d in (4, 4, 8))
    if stress:
        # 1000^17 and 1100^17 overflow FP32 (as does 300^17); the last key is inaccessible to earlier rows.
        q[:, :, :, :] = np.array([2., 0., 0., 0.])
        k[0, 0, 0] = [-1000., 0., 0., 0.]
        k[0, time // 2, 0] = [1000., 0., 0., 0.]
        k[0, -1, 0] = [1100., 0., 0., 0.]
        if prefactor == "exp":
            # e^100 overflows FP32 too. Near s = 1000, FP32 holds the key bound s + M log s only to 6e-5, a relative
            # error in e^s beyond what the comparison resolves, and e^s leaves one key per row by e^100 and more
            # (whose coefficient gradients vanish in exact arithmetic): keep the positive keys near 100, each within
            # e^1 of another.
            k[0, time // 2, 0] = [100., 0., 0., 0.]
            k[0, time // 2 + 1, 0] = [99.5, 0., 0., 0.]
            k[0, -1, 0] = [100.5, 0., 0., 0.]
    # A sum of squares' coefficients need not all be positive; keep the polynomial positive with signed lower orders.
    c = (rng.uniform(0.1, 0.9, size=(*q.shape[:-1], degree + 1))
         / np.array([math.factorial(r) for r in range(degree + 1)])).astype(np.float32)
    c[..., 1:degree:2] *= -0.1
    cot = rng.normal(size=v.shape).astype(np.float32)
    return (q, k, v, c), cot


# Each transform alone, and each prefactor of the raw score.
SCORE_FUNCTIONS = (*((transform, "none") for transform in TRANSFORMS), ("identity", "softplus"), ("identity", "exp"))


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("transform, prefactor", SCORE_FUNCTIONS)
@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("backend", ["blocked", pytest.param("fused", marks=pytest.mark.gpu)])
def test_shifted_composition_and_all_gradients(backend, stress, transform, prefactor, degree):
    # 65 includes masked tail rows/columns and changes the running maximum on a later causal tile.
    if backend == "fused" and jax.default_backend() != "gpu":
        pytest.skip("The fused Triton program requires a GPU")
    args, cot = _inputs(65, stress, degree, prefactor)
    with jax.enable_x64(True):
        oracle, pull = jax.vjp(lambda *a: _oracle(*a, transform, prefactor),
                               *(jnp.asarray(x, jnp.float64) for x in args))
        oracle_grads = pull(jnp.asarray(cot, jnp.float64))

    def normalized(*args):
        result, shift = shifted_polynomial_attention(*args, transform=transform, prefactor=prefactor, backend=backend)
        return max_abs_head_rms(result, 1e-6, shift[..., None])

    y, pull = jax.vjp(jax.jit(normalized), *map(jnp.asarray, args))
    grads = pull(jnp.asarray(cot))
    np.testing.assert_allclose(y, oracle, rtol=2e-4, atol=1e-5)
    # Where one key dominates a stress row, head RMSNorm leaves its coefficient's gradient a near-cancellation of
    # terms of the gradient's largest scale, which FP32 resolves only to about 1e-3 of that scale.
    for name, actual, expected in zip("qkvc", grads, oracle_grads, strict=True):
        scale = float(jnp.max(jnp.abs(expected)))
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=(2e-3 if stress else 1e-4) * max(1.0, scale),
                                   err_msg=name)


def test_unshifted_evaluation_overflows_where_the_shift_stays_finite():
    """The stress rows' z^17 alone overflows FP32; the shifted output is finite and matches FP64 above."""
    (q, k, v, c), _ = _inputs(65, True, 17)
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) / 2
    assert not np.all(np.isfinite(np.float32(np.max(np.abs(scores))) ** np.float32(17)))
    y, shift = shifted_polynomial_attention(*map(jnp.asarray, (q, k, v, c)), transform="softplus")
    assert bool(jnp.all(jnp.isfinite(y))) and float(jnp.max(shift)) > 17 * math.log(1000)


def test_blocked_diagnostics_describe_the_causal_shifted_weights():
    args, _ = _inputs(35, False, 9)
    (y, shift), d = shifted_polynomial_attention(*map(jnp.asarray, args), transform="softplus", return_diagnostics=True)
    assert y.shape == args[2].shape and shift.shape == y.shape[:3]
    assert int(d["causal_entry_count"]) == 35 * 36 // 2
    assert int(d["invalid_kernel_count"]) == 0
    np.testing.assert_allclose(d["polynomial_shift_max"], shift.max())


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("scale", [0.5, 4.0, 40.0, 400.0])
def test_fused_horner_arithmetic_matches_the_shifted_terms(scale, transform, degree):
    """The fused programs' per-tile arithmetic (Horner in u, slopes, coefficient moments), run here over whole rows on
    CPU because Triton programs need a GPU, against FP64."""
    from scaffolds.transformer.attention.backends import fused_shifted_polynomial as fused

    rng = np.random.default_rng(int(scale * 10) + degree)
    rows, keys = 16, 256
    s = jnp.asarray(rng.normal(size=(rows, keys)) * scale, jnp.float32)
    coefficients = tuple(jnp.asarray(rng.uniform(0.05, 0.95, size=rows) / math.factorial(r), jnp.float32)
                         for r in range(degree + 1))
    dp = jnp.asarray(rng.normal(size=(rows, keys)), jnp.float32)
    z, dz = fused._transformed(s, transform)
    shift = fused._row_shift(z, degree)
    half, units, horner = fused._row_units(shift, coefficients)
    value, u, slope = fused._polynomial(z, half, horner, slope=True)
    slope = slope if dz is None else slope * dz
    product, moments = dp, [jnp.sum(dp, axis=1)]
    for _ in range(degree):
        product = product * u
        moments.append(jnp.sum(product, axis=1))
    with jax.enable_x64(True):
        s64, shift64 = jnp.asarray(s, jnp.float64), jnp.asarray(shift, jnp.float64)[:, None]
        z64 = _phi(s64, transform)
        dz64 = jax.nn.sigmoid(s64) if transform == "softplus" else jnp.ones_like(s64)
        c64 = [jnp.asarray(c, jnp.float64)[:, None] for c in coefficients]
        terms = [c * z64 ** r * jnp.exp(-shift64) for r, c in enumerate(c64)]
        slope_terms = [r * c * z64 ** (r - 1) * dz64 * jnp.exp(-shift64) for r, c in enumerate(c64) if r]
        bases = [z64 ** r * jnp.exp(-shift64) for r in range(degree + 1)]
        # Signed sums lose FP32 resolution relative to their terms: compare at the scale of the terms' magnitudes.
        exact = [(np.asarray(sum(x)), np.asarray(sum(jnp.abs(t) for t in x))) for x in (terms, slope_terms)]
        exact_moments = [(np.asarray(jnp.sum(jnp.asarray(dp, jnp.float64) * b, axis=1)),
                          np.asarray(jnp.sum(jnp.abs(b), axis=1))) for b in bases]
    floor = keys * np.finfo(np.float32).tiny
    for got, (want, size) in zip((value, slope), exact):
        np.testing.assert_array_less(np.abs(np.asarray(got, np.float64) - want), 8e-6 * size + floor)
    for got, unit, (want, size) in zip(moments, units, exact_moments, strict=True):
        np.testing.assert_array_less(np.abs(np.asarray(got * unit, np.float64) - want), 2e-5 * size + floor)


def _model_outputs(config, tokens, cotangent):
    """Logits and the gradient of <logits, cotangent> with respect to every parameter."""
    graphdef, params = nnx.split(build_model(config.model, 42), nnx.Param)

    def loss(params):
        logits = nnx.merge(graphdef, params)(tokens)
        return jnp.sum(logits * cotangent), logits

    (_, logits), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    return logits, dict(nnx.to_flat_state(grads))


def _assert_models_agree(actual_config, expected_config, *, rtol, grad_rtol, grad_atol=1e-4, logit_atol=0.0):
    rng = np.random.default_rng(7)
    vocab = expected_config.model.vocab_size
    tokens = jnp.asarray(rng.integers(0, vocab, size=(2, 8)), jnp.int32)
    cotangent = jnp.asarray(rng.normal(size=(2, 8, vocab)), jnp.float32)
    expected, expected_grads = _model_outputs(expected_config, tokens, cotangent)
    actual, actual_grads = _model_outputs(actual_config, tokens, cotangent)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=max(1e-5, logit_atol * float(np.max(np.abs(expected)))))
    assert actual_grads.keys() == expected_grads.keys()
    for path, value in expected_grads.items():
        scale = max(1e-6, float(np.max(np.abs(value[...]))))
        np.testing.assert_allclose(actual_grads[path][...], value[...], rtol=grad_rtol, atol=grad_atol * scale,
                                   err_msg=str(path))


@pytest.mark.parametrize("power", (0, 1))
def test_shifted_model_matches_the_unshifted_model_where_both_are_finite(power):
    """The h_t(z) and z h_t(z) models (core eight) with the shift equal the unshifted blocked models at moderate
    scores, logits and every parameter gradient: head RMSNorm divides the row factor out."""
    plain = _config(power, attention_head_rms_prescale="max_abs")
    shifted = _config(power, attention_head_rms_prescale="max_abs", attention_polynomial_shift="row_power")
    _assert_models_agree(shifted, plain, rtol=1e-4, grad_rtol=1e-3)


# The fused and blocked models' logits agree to rounding except where a token of the tiny model is ill-conditioned:
# on a B200 one token of the degree-sixteen tanh query model differs by 4.2e-5 (0.02% of the largest logit, 0.19),
# each evaluation deterministic. Logits are held to 5e-4 of the largest, as gradients to 1e-3 of each array's largest.
LOGIT_ATOL = 5e-4


@pytest.mark.gpu
@pytest.mark.parametrize("power", (0, 1))
def test_fused_model_matches_the_blocked_shifted_model(power):
    """The fused kernel's models equal the blocked reference's, whose attention products run at FP32 precision on
    the GPU (the unshifted blocked path takes the device's default matmul precision there), to 1e-3 of each
    gradient's largest entry: the kernel forms TF32x3 products and sums them in another order."""
    assert jax.default_backend() == "gpu"
    blocked, fused = (_config(power, attention_head_rms_prescale="max_abs", attention_backend=backend,
                              attention_polynomial_shift="row_power") for backend in ("blocked", "fused"))
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3, logit_atol=LOGIT_ATOL)


def test_the_shift_is_refused_off_its_recipe():
    with pytest.raises(ValueError, match="attention_polynomial_shift"):
        _config(0, attention_head_rms_prescale="none", attention_polynomial_shift="row_power")
    with pytest.raises(ValueError, match="mosaic"):
        _config(0, attention_head_rms_prescale="max_abs", attention_polynomial_shift="row_power",
                attention_backend="mosaic", value_head_dim=64, attention_score_transform="identity")
    # The softplus score has Mosaic row-power kernels (test_mosaic_row_power.py).
    _config(0, attention_head_rms_prescale="max_abs", attention_polynomial_shift="row_power",
            attention_backend="mosaic", value_head_dim=64)
    with pytest.raises(ValueError, match="attention_polynomial_shift"):
        _config(0, attention_head_rms_prescale="max_abs", attention_polynomial_shift="row_power",
                attention_coefficient_parameterization="direct", attention_score_transform="identity",
                attention_coefficient_conditioning="query_key")


@pytest.mark.parametrize("power", (0, 1))
def test_validation_diagnostics_report_the_shifted_model_as_the_unshifted_one(power):
    """After a training step, the trainer's attention report of the shifted model is valid and has the unshifted
    model's transform measurements, with its kernel values named for the shift."""
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.diagnostics import attention_diagnostics
    from scaffolds.transformer.training.state import create_state

    batch = jnp.asarray(np.random.default_rng(9).integers(0, 32, (2, 9)), jnp.int32)
    reports = []
    for shift in ("row_power", "none"):
        config = _config(power, attention_head_rms_prescale="max_abs", attention_polynomial_shift=shift)
        state, *_ = loop.train_step(create_state(config, jax.random.PRNGKey(0)), batch, jax.random.PRNGKey(0))
        reports.append(attention_diagnostics(state, config, np.asarray(batch[:1]), {"selection": "test"}))
    shifted, plain = reports
    assert shifted["valid"], shifted["issues"]
    assert sorted(shifted["coefficients"]["score_transform"]) == sorted(plain["coefficients"]["score_transform"])
    assert plain["kernel_values"] == "query_polynomial_of_softplus_scores"
    assert shifted["kernel_values"] == "row_power_shifted_query_polynomial_of_softplus_scores"


# The direct coefficient sources the shift also evaluates, on the raw score at degree sixteen as the Q/K width 4 runs
# set them: tanh query coefficients (with a nonzero amplitude, so their generator's gradients are exercised), fixed
# Taylor, and static learned coefficients from Taylor's.
SOURCES = {
    "query": dict(attention_coefficient_parameterization="direct", attention_score_transform="identity",
                  attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                  attention_coefficient_amplitude_initial=0.5),
    "fixed": dict(attention_coefficients="fixed", attention_coefficient_conditioning="static",
                  attention_coefficient_activation="identity", attention_coefficient_parameterization="direct",
                  attention_coefficient_sharing="global", attention_score_transform="identity",
                  attention_coefficient_amplitude_initial=1.0, attention_coefficient_projection_initial_std=0.0,
                  attention_coefficient_kernel_weight_decay=False),
    "static": dict(attention_coefficients="learned", attention_coefficient_conditioning="static",
                   attention_coefficient_activation="identity", attention_coefficient_parameterization="direct",
                   attention_score_transform="identity", attention_coefficient_amplitude_initial=1.0,
                   attention_coefficient_projection_initial_std=0.0, attention_coefficient_kernel_weight_decay=False,
                   attention_coefficient_scale="factorial", attention_coefficient_initialization="taylor"),
}


def _source_config(source, **changes):
    return _config(0, core_degree=16, **{"attention_head_rms_prescale": "max_abs", **SOURCES[source], **changes})


@pytest.mark.parametrize("source", SOURCES)
def test_shifted_model_matches_the_unshifted_model_for_every_coefficient_source(source):
    """Degree-sixteen tanh query, fixed Taylor and static learned coefficients on the raw score: with the shift each
    model equals the unshifted blocked model at moderate scores, logits and every parameter gradient."""
    _assert_models_agree(_source_config(source, attention_polynomial_shift="row_power"), _source_config(source),
                         rtol=1e-4, grad_rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("source", SOURCES)
def test_fused_model_matches_the_blocked_shifted_model_for_every_coefficient_source(source):
    """As for the SoS models: the fused kernel's model against the blocked reference's, to 1e-3 of each gradient's
    largest entry."""
    assert jax.default_backend() == "gpu"
    blocked, fused = (_source_config(source, attention_backend=backend, attention_polynomial_shift="row_power")
                      for backend in ("blocked", "fused"))
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3, logit_atol=LOGIT_ATOL)


def test_the_shift_admits_its_coefficient_sources_alone():
    """The recipe rule alone, on each source's settings with one changed (some changes are invalid recipes on other
    grounds, which ModelConfig would refuse first)."""
    from types import SimpleNamespace

    from scaffolds.transformer.attention.pieces import polynomial_shift_problem

    for source in SOURCES:
        assert polynomial_shift_problem(_source_config(source).model) is None, source
    # Pure tanh generates direct coefficients or square factors alike.
    for pure in (_pure_tanh_config(attention_polynomial_shift="row_power"),
                 _pure_tanh_sos_config(attention_polynomial_shift="row_power")):
        assert polynomial_shift_problem(pure.model) is None
    query, fixed = _source_config("query").model, _source_config("fixed").model
    for model, changes, reason in (
            (query, dict(attention_coefficient_conditioning="query_key"), "diagonal tanh or pure tanh query"),
            (query, dict(attention_coefficient_readout="dense"), "diagonal tanh or pure tanh query"),
            (query, dict(attention_coefficient_activation="sigmoid"), "diagonal tanh or pure tanh query"),
            (fixed, dict(attention_coefficient_parameterization="sum_of_squares"), "diagonal tanh or pure tanh query"),
            (query, dict(attention_coefficient_input_power=1), "every order from zero"),
            (query, dict(attention_coefficient_zero_constant=True), "every order from zero"),
            (fixed, dict(attention_fixed_square_factors=(1.0, 1.0)), "square factors"),
            (query, dict(attention_normalization=True), "unnormalized"),
            (query, dict(attention_score_prefactor="sigmoid"), "prefactor"),
            (query, dict(attention_score_prefactor="exp", attention_score_transform="softplus"), "prefactor"),
            (fixed, dict(attention_head_rms_prescale="none"), "max-abs prescale")):
        assert reason in polynomial_shift_problem(SimpleNamespace(**{**vars(model), **changes})), changes


@pytest.mark.parametrize("source, kernel_values", (("query", "query_polynomial"), ("fixed", "taylor"),
                                                   ("static", "learned_polynomial")))
def test_validation_diagnostics_name_each_shifted_source(source, kernel_values):
    """After a training step the trainer's attention report of each shifted model is valid, with its kernel values
    named for the shift."""
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.diagnostics import attention_diagnostics
    from scaffolds.transformer.training.state import create_state

    batch = jnp.asarray(np.random.default_rng(9).integers(0, 32, (2, 9)), jnp.int32)
    reports = {}
    for shift in ("row_power", "none"):
        config = _source_config(source, attention_polynomial_shift=shift)
        state, *_ = loop.train_step(create_state(config, jax.random.PRNGKey(0)), batch, jax.random.PRNGKey(0))
        reports[shift] = attention_diagnostics(state, config, np.asarray(batch[:1]), {"selection": "test"})
    assert reports["row_power"]["valid"], reports["row_power"]["issues"]
    assert reports["none"]["kernel_values"] == f"unshifted_{kernel_values}"
    assert reports["row_power"]["kernel_values"] == f"row_power_shifted_{kernel_values}"


def _taylor_row(top, coefficients, time=96):
    """One head whose first key scores `top` and whose others spread over [-9, 9], every query with the same
    coefficients."""
    scores = np.concatenate([[top], np.linspace(-9.0, 9.0, time - 1)]).astype(np.float32)
    q, k = np.ones((1, time, 1, 1), np.float32), scores.reshape(1, time, 1, 1)
    c = np.broadcast_to(np.asarray(coefficients, np.float32), (1, time, 1, len(coefficients))).copy()
    _, d = shifted_polynomial_attention(*map(jnp.asarray, (q, k, np.ones((1, time, 1, 2), np.float32), c)),
                                        transform="identity", return_diagnostics=True)
    return d


def test_weights_below_the_rows_resolution_are_not_negative_beyond_rounding():
    """Where a row's largest power passes FP32's range (a score of 300 at degree sixteen puts the row unit at
    2e-40), the weights of its small keys are rounding in row units and may come out below zero, although T_16 is
    positive everywhere; none is negative beyond its resolution, and a genuinely negative weight still is."""
    taylor = [1 / math.factorial(r) for r in range(17)]
    tiny = float(np.finfo(np.float32).tiny)
    for top in (5.0, 50.0, 300.0, 700.0):
        d = _taylor_row(top, taylor)
        assert int(d["negative_weight_beyond_rounding_count"]) == 0
        assert float(d["kernel_min"]) >= -34 * tiny and float(d["weight_sum_min"]) > 0
        if top <= 50.0:  # every weight is resolved: none rounds below zero
            assert int(d["negative_weight_count"]) == 0
    below_four = [-3.0, *taylor[1:]]  # T_16(s) - 4, negative for s below about 1.4
    linear = [0.0, 1.0, *([0.0] * 15)]
    for coefficients in (below_four, linear):
        d = _taylor_row(5.0, coefficients)
        assert int(d["negative_weight_beyond_rounding_count"]) == int(d["negative_weight_count"]) > 1000


def test_validation_accepts_rounding_below_zero_and_refuses_a_negative_weight_beyond_it(monkeypatch):
    """The trainer's attention report holds a nonnegative shifted kernel to its resolution: a kernel minimum of a
    few FP32 denormal steps is valid, a negative weight beyond rounding is not; the unshifted kernel keeps the
    strict rule."""
    from scaffolds.transformer.training import diagnostics
    from scaffolds.transformer.training.state import create_state

    batch = np.random.default_rng(9).integers(0, 32, (1, 9))
    step = diagnostics._diagnostic_step

    def report(shift, **overrides):
        config = _source_config("fixed", attention_polynomial_shift=shift)
        state = create_state(config, jax.random.PRNGKey(0))

        def measured(*arguments):
            layers = [dict(layer) for layer in step(*arguments)]
            for name, value in overrides.items():
                assert shift == "none" or name in layers[1]
                layers[1][name] = np.float32(value)
            return layers

        monkeypatch.setattr(diagnostics, "_diagnostic_step", measured)
        return diagnostics.attention_diagnostics(state, config, batch, {"selection": "test"})

    assert report("row_power")["valid"]
    rounded = report("row_power", kernel_min=-4.2e-45, negative_weight_beyond_rounding_count=0)
    assert rounded["valid"], rounded["issues"]
    assert rounded["layers"][1]["negative_weight_beyond_rounding_count"] == 0
    beyond = report("row_power", kernel_min=-1e-20, negative_weight_beyond_rounding_count=3)
    assert not beyond["valid"]
    assert beyond["issues"] == ["layer 1: 3 negative kernel values beyond rounding, kernel_min=" + str(float(np.float32(-1e-20)))]
    strict = report("none", kernel_min=-4.2e-45)
    assert strict["issues"] == ["layer 1: invalid kernel_min=" + str(float(np.float32(-4.2e-45)))]


def test_blocked_diagnostics_measure_the_cancellation_of_signed_rows():
    """On signed scores with signed coefficients the cancellation ratio is sum |w| / |sum w| of the unshifted
    weights per row, which the row factor leaves unchanged."""
    (q, k, v, c), _ = _inputs(35, False, 9)
    c[..., 1::2] *= -50.0
    _, d = shifted_polynomial_attention(*map(jnp.asarray, (q, k, v, c)), transform="identity",
                                        return_diagnostics=True)
    scores = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64)) / 2
    causal = np.tri(35, dtype=bool)
    weights = sum(c.astype(np.float64)[..., r].transpose(0, 2, 1)[..., None] * scores ** r for r in range(10))
    weights = np.where(causal, weights, 0.0)
    ratio = np.abs(weights).sum(-1) / np.abs(weights.sum(-1))
    assert ratio.max() > 1.5
    np.testing.assert_allclose(d["cancellation_ratio_max"], ratio.max(), rtol=1e-3)
    np.testing.assert_allclose(d["cancellation_ratio_mean"], ratio.mean(), rtol=1e-3)


# Pure tanh coefficients tanh(w_r^T x_hat + beta_r)/r! from the L2-normalized raw block input, W at zero and beta at
# atanh(1/2), as the Q/K width 4 prefactor runs set them.
PURE_TANH = dict(attention_coefficient_parameterization="direct", attention_score_transform="identity",
                 attention_coefficient_activation="pure_tanh", attention_coefficient_tanh_bias=True,
                 attention_coefficient_tanh_bias_initial=math.atanh(0.5),
                 attention_coefficient_projection_initial_std=0.0, attention_coefficient_amplitude_initial=1.0,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="none",
                 attention_coefficient_input_source="pre_rmsnorm", attention_coefficient_input_normalization="l2")
PREFACTORS = ("none", "softplus", "exp")


def _pure_tanh_config(prefactor="none", **changes):
    return _config(0, core_degree=16, **{"attention_head_rms_prescale": "max_abs", **PURE_TANH,
                                         "attention_score_prefactor": prefactor, **changes})


def test_pure_tanh_coefficients_start_at_half_the_taylor_coefficients():
    """W at zero and beta at atanh(1/2): every c_r starts at 0.5/r!, tanh' = 3/4, and only W decays."""
    from flax import nnx as flax_nnx

    from scaffolds.transformer.attention.input_dependent import project_coefficients
    from scaffolds.transformer.attention.parameters import attention_parameter_specs

    config = _pure_tanh_config()
    params = {"/".join(map(str, key)): np.asarray(value[...])
              for key, value in flax_nnx.to_flat_state(flax_nnx.state(build_model(config.model, 42), flax_nnx.Param))}
    kernel = next(value for key, value in params.items() if key.endswith("query_polynomial_kernel"))
    bias = next(value for key, value in params.items() if key.endswith("query_polynomial_tanh_bias"))
    assert not np.any(kernel) and np.allclose(bias, math.atanh(0.5))
    assert not any(key.endswith(("query_polynomial_bias", "query_polynomial_amplitude")) for key in params)
    decays = {spec.name: spec.weight_decay for spec in attention_parameter_specs(config.model)}
    assert decays["query_polynomial_kernel"] and not decays["query_polynomial_tanh_bias"]
    x = jnp.asarray(np.random.default_rng(3).normal(size=(1, 3, 16)), jnp.float32)
    c = project_coefficients(x, jnp.asarray(kernel[0]), None, activation="pure_tanh",
                             tanh_bias=jnp.asarray(bias[0]), amplitude_scale="factorial")
    factorials = np.array([math.factorial(r) for r in range(17)], np.float64)
    np.testing.assert_allclose(np.asarray(c, np.float64) * factorials, 0.5, rtol=1e-6)


def test_the_pure_tanh_shifted_model_matches_the_unshifted_one():
    _assert_models_agree(_pure_tanh_config(attention_polynomial_shift="row_power"), _pure_tanh_config(),
                         rtol=1e-4, grad_rtol=1e-3)


@pytest.mark.parametrize("prefactor", ("softplus", "exp"))
def test_shifted_prefactor_models_match_the_unshifted_composition(prefactor):
    """f(s) p_t(s) with tanh query coefficients at degree sixteen: the shifted model equals the blocked backend's
    unshifted composition of the prefactor and the polynomial at moderate scores."""
    plain = _source_config("query", attention_score_prefactor=prefactor)
    shifted = _source_config("query", attention_score_prefactor=prefactor, attention_polynomial_shift="row_power")
    _assert_models_agree(shifted, plain, rtol=1e-4, grad_rtol=1e-3)


@pytest.mark.parametrize("prefactor", PREFACTORS)
def test_the_fused_pure_tanh_models_build(prefactor):
    """The layer admits each fused recipe (building runs no kernel)."""
    build_model(_pure_tanh_config(prefactor, attention_backend="fused", attention_polynomial_shift="row_power").model, 0)


@pytest.mark.parametrize("prefactor", PREFACTORS)
def test_acceptance_checks_the_pure_tanh_start_and_selects_the_shifted_gpu_cases(prefactor):
    """Acceptance's initialization check builds a softmax control without the inner bias or its start, and its
    GPU stage takes the shifted module's cases, not the coefficient forms' Mosaic case, which skips on a B200."""
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.configs import config_to_dict

    config = _pure_tanh_config(prefactor, attention_backend="fused", attention_polynomial_shift="row_power")
    spec = acceptance.attention_spec(config, entry_module="tests")
    facts = spec.check_initialization(config)
    assert len(facts["softmax_counterpart_parameter_sha256"]) == 64
    bias = facts["mechanism_parameters"]["query_polynomial_tanh_bias"]
    assert set(bias["initial"]) == {math.atanh(0.5)}
    selected = {cpu: acceptance.test_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=cpu)
                for cpu in (True, False)}
    module = "scaffolds/transformer/tests/test_shifted_polynomial_attention.py"
    assert module in selected[True] and module in selected[False]
    assert acceptance.COEFFICIENT_FORM_TESTS in selected[True]
    assert acceptance.COEFFICIENT_FORM_TESTS not in selected[False]


@pytest.mark.parametrize("prefactor", PREFACTORS)
def test_training_resumes_a_pure_tanh_model_exactly(tmp_path, prefactor):
    """The trainer's updates, telemetry, validation diagnostics and checkpoint audit with the inner bias and each
    prefactor, on the blocked shifted reference: an interrupted run resumes to the uninterrupted one's state."""
    from scaffolds.transformer.training import loop

    def config(name):
        current = _pure_tanh_config(prefactor, attention_polynomial_shift="row_power")
        current.train.workdir = tmp_path / name
        current.train.eval_every, current.train.eval_batches = 2, 1
        current.train.ckpt_every, current.train.log_every = 2, 1
        return current

    full = loop.run_training(config("full"), audit_checkpoint=True)
    before = loop.run_training(config("restart"), stop_after_step=2, audit_checkpoint=True)
    after = loop.run_training(config("restart"), audit_checkpoint=True)
    first, second = before["checkpoint_audit"], after["checkpoint_audit"]
    assert first["final_state_sha256"] == first["checkpoint_state_sha256"] == second["start_state_sha256"]
    assert first["next_update"] == second["first_update"]
    assert second["final_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    rows = [json.loads(line) for line in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(row["validation_nll"]) for row in rows if "validation_nll" in row)
    assert [row["step"] for row in rows if "validation_nll" in row][-1] == 4


@pytest.mark.gpu
@pytest.mark.parametrize("prefactor", PREFACTORS)
def test_fused_pure_tanh_models_match_the_blocked_shifted_models(prefactor):
    assert jax.default_backend() == "gpu"
    blocked, fused = (_pure_tanh_config(prefactor, attention_backend=backend, attention_polynomial_shift="row_power")
                      for backend in ("blocked", "fused"))
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3, logit_atol=LOGIT_ATOL)


# z h_t(z), z = softplus(s), with pure tanh square factors F_j = f_j tanh(w_j^T x_hat + beta_j), from the L2-normalized
# raw block input, W at zero and beta at atanh(1/2), as the Q/K width 4 run with a degree-sixteen core sets them.
PURE_TANH_SOS = dict(attention_coefficient_activation="pure_tanh", attention_coefficient_tanh_bias=True,
                     attention_coefficient_tanh_bias_initial=math.atanh(0.5),
                     attention_coefficient_projection_initial_std=0.0, attention_coefficient_amplitude_initial=1.0,
                     attention_coefficient_input_source="pre_rmsnorm", attention_coefficient_input_normalization="l2")
# A random W, so that the factors differ between queries.
RANDOM_W = dict(attention_coefficient_projection_initial_std=0.25)


def _pure_tanh_sos_config(**changes):
    return _config(1, core_degree=16, **{"attention_head_rms_prescale": "max_abs", **PURE_TANH_SOS, **changes})


def test_pure_tanh_square_factors_are_fractions_of_the_taylor_factors():
    """Pure fractions g_j give the factors f_j g_j: a common fraction starts z h(z) at g^2 z T_16(z), and any
    fractions expand to the explicit z (A(z)^2 + B(z)^2) of those factors, which is never negative for z >= 0."""
    from scaffolds.transformer.attention.nonnegative_polynomial import (
        query_square_coefficients, split_factors, taylor_square_factors,
    )

    taylor = np.array([0.0, *(1 / math.factorial(r) for r in range(17))])  # z T_16(z)
    affine = query_square_coefficients(jnp.ones((17,), jnp.float32), 17, input_power=1)
    half = query_square_coefficients(jnp.full((17,), 0.5, jnp.float32), 17, input_power=1, pure=True)
    np.testing.assert_allclose(affine, taylor, rtol=1e-5)
    np.testing.assert_allclose(half, 0.25 * taylor, rtol=1e-5)
    assert float(half[0]) == 0.0
    fractions = np.random.default_rng(5).uniform(-1.0, 1.0, size=(6, 17))
    expanded = np.asarray(query_square_coefficients(jnp.asarray(fractions, jnp.float32), 17, input_power=1, pure=True),
                          np.float64)
    a, b = split_factors(np.asarray(taylor_square_factors(16), np.float64) * fractions, 16)
    z = np.linspace(0.0, 4.0, 17)
    powers = z[:, None] ** np.arange(18)
    explicit = z[:, None] * ((powers[:, :9] @ a.T) ** 2 + (powers[:, :8] @ b.T) ** 2)
    np.testing.assert_allclose(powers @ expanded.T, explicit, rtol=2e-5, atol=1e-7)
    assert np.all(explicit >= 0)
    with pytest.raises(ValueError, match="every factor constant"):
        query_square_coefficients(jnp.ones((15,), jnp.float32), 16, zero_constant=True, pure=True)


def test_the_pure_tanh_square_factor_model_starts_at_a_quarter_of_the_taylor_kernel():
    """W at zero and beta at atanh(1/2): every factor starts at half its Taylor factor, so each query's z h_t(z)
    starts at a quarter of z T_16(z), the affine generator's start scaled; tanh' = 3/4, and only W decays. (Head
    RMSNorm removes the scale wherever a head's output is well above its epsilon.)"""
    from scaffolds.transformer.attention.input_dependent import project_coefficients
    from scaffolds.transformer.attention.nonnegative_polynomial import query_square_coefficients
    from scaffolds.transformer.attention.parameters import attention_parameter_specs

    config = _pure_tanh_sos_config()
    params = {"/".join(map(str, key)): np.asarray(value[...])
              for key, value in nnx.to_flat_state(nnx.state(build_model(config.model, 42), nnx.Param))}
    kernel = next(value for key, value in params.items() if key.endswith("query_polynomial_kernel"))
    bias = next(value for key, value in params.items() if key.endswith("query_polynomial_tanh_bias"))
    assert kernel.shape[-1] == bias.shape[-1] == 17 and not np.any(kernel) and np.allclose(bias, math.atanh(0.5))
    assert not any(key.endswith(("query_polynomial_bias", "query_polynomial_amplitude")) for key in params)
    decays = {spec.name: spec.weight_decay for spec in attention_parameter_specs(config.model)}
    assert decays["query_polynomial_kernel"] and not decays["query_polynomial_tanh_bias"]
    x = jnp.asarray(np.random.default_rng(3).normal(size=(1, 3, 16)), jnp.float32)
    fractions = project_coefficients(x, jnp.asarray(kernel[0]), None, activation="pure_tanh",
                                     tanh_bias=jnp.asarray(bias[0]))
    np.testing.assert_allclose(fractions, 0.5, rtol=1e-6)
    coefficients = query_square_coefficients(fractions, 17, input_power=1, pure=True)
    taylor = np.array([0.0, *(1 / math.factorial(r) for r in range(17))])
    np.testing.assert_allclose(coefficients, np.broadcast_to(0.25 * taylor, coefficients.shape), rtol=1e-5)


@pytest.mark.parametrize("changes", ({}, RANDOM_W), ids=("start", "random-w"))
def test_the_pure_tanh_square_factor_shifted_model_matches_the_unshifted_one(changes):
    _assert_models_agree(_pure_tanh_sos_config(attention_polynomial_shift="row_power", **changes),
                         _pure_tanh_sos_config(**changes), rtol=1e-4, grad_rtol=1e-3)


def test_acceptance_and_validation_take_the_pure_tanh_square_factors():
    """The fused recipe builds, acceptance's initialization check and selection hold as for the direct pure tanh
    recipes, and after a training step the trainer's report is valid, nonnegative, and names the factors' form."""
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.configs import config_to_dict
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity
    from scaffolds.transformer.training.state import create_state

    fused = _pure_tanh_sos_config(attention_backend="fused", attention_polynomial_shift="row_power")
    build_model(fused.model, 0)
    spec = acceptance.attention_spec(fused, entry_module="tests")
    facts = spec.check_initialization(fused)
    assert set(facts["mechanism_parameters"]["query_polynomial_tanh_bias"]["initial"]) == {math.atanh(0.5)}
    selected = {cpu: acceptance.test_paths(spec, config_to_dict(fused), resolved_backend="fused", cpu=cpu)
                for cpu in (True, False)}
    module = "scaffolds/transformer/tests/test_shifted_polynomial_attention.py"
    assert module in selected[True] and module in selected[False]
    assert acceptance.COEFFICIENT_FORM_TESTS in selected[True]
    assert acceptance.COEFFICIENT_FORM_TESTS not in selected[False]

    config = _pure_tanh_sos_config(attention_polynomial_shift="row_power", **RANDOM_W)
    batch = jnp.asarray(np.random.default_rng(9).integers(0, 32, (2, 9)), jnp.int32)
    state, *_ = loop.train_step(create_state(config, jax.random.PRNGKey(0)), batch, jax.random.PRNGKey(0))
    report = attention_diagnostics(state, config, np.asarray(batch[:1]), {"selection": "test"})
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == "row_power_shifted_query_polynomial_of_softplus_scores"
    assert all(layer["negative_weight_count"] == 0 for layer in report["layers"])
    assert report["coefficients"]["activation"] == "pure_tanh"
    identity = attention_identity(config)
    assert "square factors F_j(x_t) = f_j tanh(W_j^T x_t + beta_j); fixed f factorizes T_16" in identity["attention_formula"]
    assert "beta initially 0.549306144" in identity["attention_formula"]
    assert identity["attention_coefficient_input"] == "unit_l2_of_pre_attention_residual_activation"
    assert identity["attention_coefficient_count"] == 2 * 2 * 17 * (16 + 1)  # layers, heads, factors, W and beta


def test_training_resumes_a_pure_tanh_square_factor_model_exactly(tmp_path):
    from scaffolds.transformer.training import loop

    def config(name):
        current = _pure_tanh_sos_config(attention_polynomial_shift="row_power")
        current.train.workdir = tmp_path / name
        current.train.eval_every, current.train.eval_batches = 2, 1
        current.train.ckpt_every, current.train.log_every = 2, 1
        return current

    full = loop.run_training(config("full"), audit_checkpoint=True)
    before = loop.run_training(config("restart"), stop_after_step=2, audit_checkpoint=True)
    after = loop.run_training(config("restart"), audit_checkpoint=True)
    first, second = before["checkpoint_audit"], after["checkpoint_audit"]
    assert first["final_state_sha256"] == first["checkpoint_state_sha256"] == second["start_state_sha256"]
    assert first["next_update"] == second["first_update"]
    assert second["final_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    rows = [json.loads(line) for line in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(row["validation_nll"]) for row in rows if "validation_nll" in row)


def test_pure_tanh_square_factors_are_refused_off_their_recipe():
    """They take every factor constant and no factorial scale, need the inner bias or a random W to start, and run
    on the blocked backend or the shifted fused kernel, not the Mosaic kernels."""
    for changes in (dict(attention_backend="mosaic"), dict(attention_coefficient_amplitude_scale="factorial"),
                    dict(attention_coefficient_tanh_bias=False, attention_coefficient_tanh_bias_initial=0.0),
                    dict(attention_backend="fused")):  # the fused kernel is the shifted one
        with pytest.raises(ValueError):
            _pure_tanh_sos_config(**changes)
    _pure_tanh_sos_config(attention_coefficient_tanh_bias=False, attention_coefficient_tanh_bias_initial=0.0, **RANDOM_W)


@pytest.mark.gpu
@pytest.mark.parametrize("changes", ({}, RANDOM_W), ids=("start", "random-w"))
def test_fused_pure_tanh_square_factor_model_matches_the_blocked_shifted_model(changes):
    assert jax.default_backend() == "gpu"
    blocked, fused = (_pure_tanh_sos_config(attention_backend=backend, attention_polynomial_shift="row_power", **changes)
                      for backend in ("blocked", "fused"))
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3, logit_atol=LOGIT_ATOL)


def test_exp_prefactor_overflows_unshifted_and_stays_finite_shifted():
    """e^1100 overflows FP32 (and FP64); the shifted weights stay finite and the shift holds the row maximum."""
    (q, k, v, c), _ = _inputs(65, True, 17)
    y, shift = shifted_polynomial_attention(*map(jnp.asarray, (q, k, v, c)), prefactor="exp")
    assert not np.isfinite(np.exp(np.float32(1100.0)))
    assert bool(jnp.all(jnp.isfinite(y))) and float(jnp.max(shift)) > 1000 + 17 * math.log(1000)


@pytest.mark.parametrize("prefactor", ("softplus", "exp"))
def test_validation_diagnostics_name_the_prefactor(prefactor):
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.diagnostics import attention_diagnostics
    from scaffolds.transformer.training.state import create_state

    batch = jnp.asarray(np.random.default_rng(9).integers(0, 32, (2, 9)), jnp.int32)
    config = _pure_tanh_config(prefactor, attention_polynomial_shift="row_power")
    state, *_ = loop.train_step(create_state(config, jax.random.PRNGKey(0)), batch, jax.random.PRNGKey(0))
    report = attention_diagnostics(state, config, np.asarray(batch[:1]), {"selection": "test"})
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == f"row_power_shifted_query_polynomial_times_{prefactor}_of_raw_scores"


def test_prefactors_are_refused_off_the_raw_score():
    with pytest.raises(ValueError, match="attention_polynomial_shift"):
        _config(1, attention_head_rms_prescale="max_abs", attention_polynomial_shift="row_power",
                attention_score_prefactor="exp")
    with pytest.raises(ValueError, match="attention_polynomial_shift"):
        _pure_tanh_config("sigmoid", attention_polynomial_shift="row_power")


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("prefactor", ("softplus", "exp"))
@pytest.mark.parametrize("scale", [0.5, 4.0, 40.0, 400.0])
def test_keyed_arithmetic_matches_fp64(scale, prefactor, degree):
    """The fused programs' per-key units: e^(b - m) h and its slope against f(s) p(s) e^(-m) and its derivative in
    FP64, m the row's largest bound, on rows mixing far negative keys with moderate ones (where units set by the
    largest |s| alone underflow the weights that matter)."""
    from scaffolds.transformer.attention.backends.fused_shifted_polynomial import _key_bound, _keyed_polynomial

    rng = np.random.default_rng(int(scale * 10) + degree)
    s = rng.normal(size=(8, 64)) * scale
    s[:, 0] = -25 * scale
    c = rng.uniform(-1, 1, size=(8, degree + 1)) / np.array([math.factorial(r) for r in range(degree + 1)])
    coefficients = tuple(jnp.asarray(c[:, r], jnp.float32) for r in range(degree + 1))
    scores = jnp.asarray(s, jnp.float32)
    bound, ratio = _key_bound(scores, prefactor, degree)
    shift = jnp.maximum(0.0, jnp.max(bound, axis=1))
    unit = jnp.exp(bound - shift[:, None])
    value, slope = _keyed_polynomial(scores, coefficients, slope=True)
    weight = unit * value
    derivative = unit * (value + slope if ratio is None else ratio * value + slope)
    s64 = np.asarray(scores, np.float64)
    # log softplus(s) and sigmoid(s) / softplus(s), with their limits s and 1 where softplus underflows FP64.
    tail = s64 < -30
    softplus = np.logaddexp(0.0, np.where(tail, 0.0, s64))
    logf = s64 if prefactor == "exp" else np.where(tail, s64, np.log(softplus))
    fprime_over_f = (np.ones_like(s64) if prefactor == "exp"
                     else np.where(tail, 1.0, 1 / (1 + np.exp(-np.where(tail, 0.0, s64))) / softplus))
    powers = [s64 ** r for r in range(degree + 1)]
    p = sum(c[:, r, None] * powers[r] for r in range(degree + 1))
    dp = sum(r * c[:, r, None] * powers[r - 1] for r in range(1, degree + 1))
    m = np.asarray(shift, np.float64)[:, None]
    exact = np.exp(logf - m) * p
    exact_derivative = np.exp(logf - m) * (fprime_over_f * p + dp)
    # Each key's terms reach |c_r| e^(b - m): signed sums resolve to FP32 relative to them.
    size = np.exp(np.minimum(np.asarray(bound, np.float64) - m, 0)) * np.abs(c).sum(1, keepdims=True)
    np.testing.assert_array_less(np.abs(np.asarray(weight, np.float64) - exact), 2e-5 * size + 1e-30)
    np.testing.assert_array_less(np.abs(np.asarray(derivative, np.float64) - exact_derivative),
                                 2e-5 * size * (degree + 1) + 1e-30)
    assert float(jnp.max(jnp.abs(weight))) > 0
