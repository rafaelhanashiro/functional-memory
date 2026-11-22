"""Row-power shifted per-query polynomials against dense FP64, their kernel arithmetic, and the layer they serve."""
from dataclasses import replace
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


def _oracle(q, k, v, c, transform):
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(q.shape[1])[None, :]
    z = _phi(jnp.where(causal, scores, 0), transform)
    weights = sum(c[..., r].transpose(0, 2, 1)[..., None] * z ** r for r in range(c.shape[-1]))
    output = jnp.einsum("bhqk,bkhd->bqhd", jnp.where(causal, weights, 0), v)
    return output / jnp.sqrt(jnp.mean(jnp.square(output), axis=-1, keepdims=True) + 1e-6)


def _inputs(time, stress, degree):
    rng = np.random.default_rng(124 + degree)
    q, k, v = (rng.normal(size=(1, time, 1, d)).astype(np.float32) / 2 for d in (4, 4, 8))
    if stress:
        # 1000^17 and 1100^17 overflow FP32 (as does 300^17); the last key is inaccessible to earlier rows.
        q[:, :, :, :] = np.array([2., 0., 0., 0.])
        k[0, 0, 0] = [-1000., 0., 0., 0.]
        k[0, time // 2, 0] = [1000., 0., 0., 0.]
        k[0, -1, 0] = [1100., 0., 0., 0.]
    # A sum of squares' coefficients need not all be positive; keep the polynomial positive with signed lower orders.
    c = (rng.uniform(0.1, 0.9, size=(*q.shape[:-1], degree + 1))
         / np.array([math.factorial(r) for r in range(degree + 1)])).astype(np.float32)
    c[..., 1:degree:2] *= -0.1
    cot = rng.normal(size=v.shape).astype(np.float32)
    return (q, k, v, c), cot


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("backend", ["blocked", pytest.param("fused", marks=pytest.mark.gpu)])
def test_shifted_composition_and_all_gradients(backend, stress, transform, degree):
    # 65 includes masked tail rows/columns and changes the running maximum on a later causal tile.
    if backend == "fused" and jax.default_backend() != "gpu":
        pytest.skip("The fused Triton program requires a GPU")
    args, cot = _inputs(65, stress, degree)
    with jax.enable_x64(True):
        oracle, pull = jax.vjp(lambda *a: _oracle(*a, transform), *(jnp.asarray(x, jnp.float64) for x in args))
        oracle_grads = pull(jnp.asarray(cot, jnp.float64))

    def normalized(*args):
        result, shift = shifted_polynomial_attention(*args, transform=transform, backend=backend)
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


def _assert_models_agree(actual_config, expected_config, *, rtol, grad_rtol, grad_atol=1e-4):
    rng = np.random.default_rng(7)
    vocab = expected_config.model.vocab_size
    tokens = jnp.asarray(rng.integers(0, vocab, size=(2, 8)), jnp.int32)
    cotangent = jnp.asarray(rng.normal(size=(2, 8, vocab)), jnp.float32)
    expected, expected_grads = _model_outputs(expected_config, tokens, cotangent)
    actual, actual_grads = _model_outputs(actual_config, tokens, cotangent)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-5)
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


@pytest.mark.gpu
@pytest.mark.parametrize("power", (0, 1))
def test_fused_model_matches_the_blocked_shifted_model(power):
    """The fused kernel's models equal the blocked reference's, whose attention products run at FP32 precision on
    the GPU (the unshifted blocked path takes the device's default matmul precision there), to 1e-3 of each
    gradient's largest entry: the kernel forms TF32x3 products and sums them in another order."""
    assert jax.default_backend() == "gpu"
    blocked, fused = (_config(power, attention_head_rms_prescale="max_abs", attention_backend=backend,
                              attention_polynomial_shift="row_power") for backend in ("blocked", "fused"))
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3)


def test_the_shift_is_refused_off_its_recipe():
    with pytest.raises(ValueError, match="attention_polynomial_shift"):
        _config(0, attention_head_rms_prescale="none", attention_polynomial_shift="row_power")
    with pytest.raises(ValueError, match="attention_polynomial_shift"):
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
    _assert_models_agree(fused, blocked, rtol=2e-4, grad_rtol=2e-3, grad_atol=1e-3)


def test_the_shift_admits_its_coefficient_sources_alone():
    """The recipe rule alone, on each source's settings with one changed (some changes are invalid recipes on other
    grounds, which ModelConfig would refuse first)."""
    from types import SimpleNamespace

    from scaffolds.transformer.attention.pieces import polynomial_shift_problem

    for source in SOURCES:
        assert polynomial_shift_problem(_source_config(source).model) is None, source
    query, fixed = _source_config("query").model, _source_config("fixed").model
    for model, changes, reason in (
            (query, dict(attention_coefficient_conditioning="query_key"), "diagonal tanh query"),
            (query, dict(attention_coefficient_readout="dense"), "diagonal tanh query"),
            (query, dict(attention_coefficient_activation="sigmoid"), "diagonal tanh query"),
            (fixed, dict(attention_coefficient_parameterization="sum_of_squares"), "diagonal tanh query"),
            (query, dict(attention_coefficient_input_power=1), "every order from zero"),
            (query, dict(attention_coefficient_zero_constant=True), "every order from zero"),
            (fixed, dict(attention_fixed_square_factors=(1.0, 1.0)), "square factors"),
            (query, dict(attention_normalization=True), "unnormalized"),
            (query, dict(attention_score_prefactor="softplus"), "prefactor"),
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
