"""Score-MLP attention: independent FP64 mathematics, numerics, fits, recipes, and diagnostics."""
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.records import create_run, regenerate_index
from scaffolds.transformer import acceptance
from scaffolds.transformer.attention import score_mlp
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.parameters import attention_parameter_count, attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import attention_weight_decay_mask, make_optimizer
from scaffolds.transformer.training.state import create_state

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/base.yaml"
GATE_PARAMETERS = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")


def _dense_units(q, k, v, a, b, w, log_decay=None, *, form):
    """Every query, key, and unit written out explicitly, from units a, b [B,T,H,J], without production helpers."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    pre = w[None, :, None, None, :] * scores[..., None] + jnp.swapaxes(b, 1, 2)[:, :, :, None, :]
    units = jnp.sum(jnp.swapaxes(a, 1, 2)[:, :, :, None, :] * pre / (1.0 + jnp.exp(-pre)), axis=-1)
    kernel = units if form == "kernel" else jnp.exp(scores) * (1.0 + units)
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    if log_decay is not None:
        cumulative = jnp.swapaxes(log_decay, 1, 2)
        kernel = kernel * jnp.exp(jnp.where(causal, cumulative[..., :, None] - cumulative[..., None, :], 0.0))
    return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, kernel, 0.0), v)


def _dense(q, k, v, x, amplitude_kernel, amplitude_bias, offset_kernel, offset_bias, slope, log_decay=None, *, form):
    """The definition from the attention input: a = alpha + x A and b = beta + x B per head and unit."""
    a = jnp.einsum("btd,hdj->bthj", x, amplitude_kernel) + amplitude_bias
    b = jnp.einsum("btd,hdj->bthj", x, offset_kernel) + offset_bias
    return _dense_units(q, k, v, a, b, slope, log_decay, form=form)


def _production(q, k, v, x, amplitude_kernel, amplitude_bias, offset_kernel, offset_bias, slope, log_decay=None, *,
                form, block=8):
    return blocked_dot_product_attention(
        q, k, v, attention_normalization=False,
        score_mlp_amplitude=score_mlp.project_units(x, amplitude_kernel, amplitude_bias),
        score_mlp_offset=score_mlp.project_units(x, offset_kernel, offset_bias), score_mlp_slope=slope,
        score_mlp_form=form, query_log_decay=log_decay, query_block_size=block)


def _arrays(width, *, gate=False, time=19, batch=2, heads=2, qk=6, value=3, hidden=5, seed=0):
    """Signed slopes, query-dependent units, and nontrivial cumulative gates (FP64 NumPy)."""
    rng = np.random.default_rng(seed + 17 * width + gate)
    arrays = [rng.normal(size=(batch, time, heads, qk)), rng.normal(size=(batch, time, heads, qk)),
              rng.normal(size=(batch, time, heads, value)), rng.normal(size=(batch, time, hidden)),
              rng.normal(size=(heads, hidden, width)) * .3, rng.normal(size=(heads, width)),
              rng.normal(size=(heads, hidden, width)) * .3, rng.normal(size=(heads, width)),
              rng.uniform(-2.0, 2.0, size=(heads, width))]
    if gate:
        arrays.append(np.cumsum(-rng.uniform(.01, .4, size=(batch, time, heads)), axis=1))
    return arrays, rng.normal(size=(batch, time, heads, value))


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _relative_error(observed, expected):
    observed, expected = np.asarray(observed, np.float64), np.asarray(expected, np.float64)
    assert observed.shape == expected.shape and np.isfinite(observed).all()
    scale = np.linalg.norm(expected)
    assert scale > 0
    return np.linalg.norm(observed - expected) / scale


@pytest.mark.parametrize("keys", (1, 5, 7, 19))
@pytest.mark.parametrize("width", (1, 3, 8))
def test_unit_sum_custom_derivative_matches_autodiff_of_its_definition(width, keys):
    rng = np.random.default_rng(width * 31 + keys)

    def plain(s, a, b, w):
        pre = w[None, :, None, None, :] * s[..., None] + b[..., None, :]
        return jnp.sum(a[..., None, :] * pre / (1.0 + jnp.exp(-pre)), axis=-1)

    with jax.enable_x64():
        operands = (jnp.asarray(rng.normal(size=(2, 3, 4, keys)) * 2), jnp.asarray(rng.normal(size=(2, 3, 4, width))),
                    jnp.asarray(rng.normal(size=(2, 3, 4, width))), jnp.asarray(rng.uniform(-2, 2, size=(3, width))))
        cotangent = jnp.asarray(rng.normal(size=(2, 3, 4, keys)))
        actual, expected = _vjp(score_mlp.unit_sum, operands, cotangent), _vjp(plain, operands, cotangent)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    for observed, wanted in zip(actual[1], expected[1], strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("gate", (False, True))
@pytest.mark.parametrize("width", (1, 4, 8))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_outputs_and_all_gradients_match_independent_float64(form, width, gate):
    """q, k, v, x, A, alpha, B, beta, w (and the gate's L) through padded query blocks and key chunks."""
    arrays, cotangent = _arrays(width, gate=gate)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = _vjp(lambda *x: _dense(*x, form=form), tuple(jnp.asarray(x, jnp.float64) for x in arrays),
                        jnp.asarray(cotangent, jnp.float64))
        expected = jax.tree.map(lambda x: np.asarray(x, np.float64), expected)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *x: _vjp(lambda *y: _production(*y, form=form), x,
                                         jnp.asarray(cotangent, jnp.float32)))(*operands)
    names = ["output", "q", "k", "v", "x", "A", "alpha", "B", "beta", "w", "L"]
    for name, observed, wanted in zip(names, [actual[0], *actual[1]], [expected[0], *expected[1]]):
        assert _relative_error(observed, wanted) < 3e-5, name


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_causality_across_blocks(form):
    arrays, cotangent = _arrays(4, time=21)
    operands = [jnp.asarray(x, jnp.float32) for x in arrays]
    output = _production(*operands, form=form)
    changed = list(operands)
    for index in range(4):  # q, k, v, and the attention input x of every later token
        changed[index] = operands[index].at[:, 9:].multiply(-3.0).at[:, 9:].add(.5)
    np.testing.assert_array_equal(_production(*changed, form=form)[:, :9], output[:, :9])
    gradients = jax.grad(lambda *x: jnp.sum(_production(*x, form=form)[:, :9] * cotangent[:, :9]),
                         argnums=(0, 1, 2, 3))(*operands)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient[:, 9:], np.zeros_like(gradient[:, 9:]))
        assert np.abs(np.asarray(gradient[:, :9])).max() > 0


def _units(width=3, time=7, heads=2, seed=4):
    rng = np.random.default_rng(seed)
    q, k = (jnp.asarray(rng.normal(size=(1, time, heads, 4)), jnp.float32) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(1, time, heads, 3)), jnp.float32)
    a, b = (jnp.asarray(rng.normal(size=(1, time, heads, width)), jnp.float32) for _ in range(2))
    return q, k, v, a, b, jnp.asarray(rng.uniform(.5, 1.5, size=(heads, width)), jnp.float32)


def _attend(q, k, v, a, b, w, form="kernel", **options):
    return blocked_dot_product_attention(q, k, v, attention_normalization=False, score_mlp_amplitude=a,
                                         score_mlp_offset=b, score_mlp_slope=w, score_mlp_form=form,
                                         query_block_size=4, **options)


@pytest.mark.parametrize("form", score_mlp.FORMS)
@pytest.mark.parametrize("case", ("score", "unit", "kernel", "output"))
def test_nonfinite_scores_units_kernels_and_outputs_poison_only_their_rows(form, case):
    q, k, v, a, b, w = _units()
    if case == "output":
        # Large finite weights in head 1, so one huge value overflows every output that reads it.
        a, b = a.at[0, :, 1].set(1e3), b.at[0, :, 1].set(10.0)
    clean = _attend(q, k, v, a, b, w, form)
    expected = np.zeros((1, 7, 2), dtype=bool)
    if case == "score":
        k = k.at[0, 4, 0].set(jnp.inf)        # every query that sees key 4 in head 0
        expected[0, 4:, 0] = True
    elif case == "unit":
        a = a.at[0, 5, 1, 2].set(jnp.nan)     # one query's unit amplitude in head 1
        expected[0, 5, 1] = True
    elif case == "kernel":
        a, b = a.at[0, 3, 0].set(1e37), b.at[0, 3, 0].set(100.0)   # finite units, infinite kernel
        expected[0, 3, 0] = True
    else:
        v = v.at[0, 2, 1, 0].set(3e38)        # finite weights and values, overflowing outputs from key 2 on
        expected[0, 2:, 1] = True
    output, diagnostics = _attend(q, k, v, a, b, w, form, return_diagnostics=True)
    invalid = ~np.isfinite(np.asarray(output))
    np.testing.assert_array_equal(invalid.any(axis=-1), expected)
    np.testing.assert_array_equal(invalid.all(axis=-1), expected)   # a poisoned row is NaN in every channel
    np.testing.assert_array_equal(np.asarray(output)[~expected], np.asarray(clean)[~expected])
    count = {"score": "invalid_score_count", "unit": "invalid_kernel_count", "kernel": "invalid_kernel_count",
             "output": "invalid_output_count"}[case]
    assert int(diagnostics[count]) > 0
    assert int(diagnostics["invalid_output_count"]) == int(expected.sum()) * 3


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_masked_future_overflow_is_never_evaluated(form):
    # Only the masked entry (query 0, key 1) has a huge score; evaluating it would overflow either form.
    q = jnp.array([1.0, 0.0]).reshape(1, 2, 1, 1)
    k = jnp.array([0.0, 1e30]).reshape(1, 2, 1, 1)
    v = jnp.array([3.0, 7.0]).reshape(1, 2, 1, 1)
    a, b, w = jnp.full((1, 2, 1, 2), 1e10), jnp.ones((1, 2, 1, 2)), jnp.ones((1, 2))
    output, diagnostics = _attend(q, k, v, a, b, w, form, return_diagnostics=True)
    at_zero = 2e10 / (1.0 + math.exp(-1.0)) + (1.0 if form == "exp_residual" else 0.0)  # f(0)
    np.testing.assert_allclose(np.asarray(output).reshape(-1), [3 * at_zero, 10 * at_zero], rtol=1e-6)
    assert float(diagnostics["score_max"]) == 0
    assert all(int(value) == 0 for name, value in diagnostics.items() if name.startswith("invalid_"))


@pytest.mark.parametrize("width", (1, 4, 8))
def test_exp_residual_start_is_exactly_the_unnormalized_exp_kernel(width):
    q, k, v, _, b, w = _units(width, time=11)
    a = jnp.zeros_like(b)
    residual = lambda q, k, v: _attend(q, k, v, a, b, w, "exp_residual")
    exponential = lambda q, k, v: blocked_dot_product_attention(q, k, v, attention_normalization=False,
                                                                query_block_size=4)
    np.testing.assert_array_equal(residual(q, k, v), exponential(q, k, v))
    cotangent = jnp.asarray(np.random.default_rng(2).normal(size=v.shape), jnp.float32)
    for observed, expected in zip(_vjp(residual, (q, k, v), cotangent)[1], _vjp(exponential, (q, k, v), cotangent)[1]):
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("width", (4, 8))
def test_kernel_form_starts_within_its_fit_error(width):
    fit = score_mlp.EXP_FITS[width]
    constants = [jnp.asarray(values, jnp.float32) for values in (fit.amplitude, fit.offset, fit.slope)]
    grid = jnp.linspace(-4.0, 4.0, 4001, dtype=jnp.float32).reshape(1, 1, 1, -1)
    kernel = score_mlp.score_kernel(grid, constants[0].reshape(1, 1, 1, -1), constants[1].reshape(1, 1, 1, -1),
                                    constants[2].reshape(1, -1), "kernel")
    relative = np.abs(np.asarray(kernel, np.float64) / np.exp(np.asarray(grid, np.float64)) - 1.0)
    assert relative.max() <= fit.max_relative_error + 1e-4
    # In attention: scores within [-4, 4] and nonnegative values bound each output by the fit error.
    rng = np.random.default_rng(width)
    directions = rng.normal(size=(2, 2, 13, 2, 4))
    q, k = (jnp.asarray(2.0 * 4 ** .25 * d / np.linalg.norm(d, axis=-1, keepdims=True), jnp.float32) for d in directions)
    v = jnp.asarray(rng.uniform(0, 1, size=(2, 13, 2, 3)), jnp.float32)
    a, b = (jnp.broadcast_to(c, (2, 13, 2, width)) for c in constants[:2])
    with jax.default_matmul_precision("highest"):
        output = _attend(q, k, v, a, b, jnp.broadcast_to(constants[2], (2, width)))
        exponential = blocked_dot_product_attention(q, k, v, attention_normalization=False, query_block_size=4)
    np.testing.assert_array_less(np.abs(np.asarray(output - exponential)),
                                 (fit.max_relative_error + 1e-4) * np.asarray(exponential) + 1e-6)


def _objective_and_error(fit):
    amplitude, slope, offset = (np.asarray(values, np.float64) for values in (fit.amplitude, fit.slope, fit.offset))
    f = lambda s: (pre := slope * s[:, None] + offset) / (1.0 + np.exp(-pre)) @ amplitude
    points = np.linspace(*score_mlp.FIT_INTERVAL, score_mlp.FIT_POINTS)
    weights = np.full(points.size, points[1] - points[0])
    weights[[0, -1]] *= .5
    fine = np.linspace(*score_mlp.FIT_INTERVAL, score_mlp.ERROR_POINTS)
    return (np.sum(weights * (f(points) / np.exp(points) - 1) ** 2), np.max(np.abs(f(fine) / np.exp(fine) - 1)))


def test_pinned_exp_fits_are_feasible_ordered_and_reproduced():
    for width, fit in score_mlp.EXP_FITS.items():
        amplitude, slope, offset = (np.asarray(values) for values in (fit.amplitude, fit.slope, fit.offset))
        transition = -offset / slope
        assert len(amplitude) == len(slope) == len(offset) == width
        assert (amplitude >= 0).all() and np.all(np.diff(transition) >= 0)
        assert (slope >= score_mlp.FIT_SLOPES[0]).all() and (slope <= score_mlp.FIT_SLOPES[1]).all()
        assert (transition >= score_mlp.FIT_TRANSITIONS[0] - 1e-12).all()
        assert (transition <= score_mlp.FIT_TRANSITIONS[1] + 1e-12).all()
        objective, error = _objective_and_error(fit)
        np.testing.assert_allclose([objective, error], [fit.objective, fit.max_relative_error], rtol=1e-9)
    assert score_mlp.EXP_FITS[4].max_relative_error < .035 and score_mlp.EXP_FITS[8].max_relative_error < 3.1e-4
    # J = 4 needs random starts; J = 8's best start is the spread one, which restarts=1 runs alone.
    for refit, pinned in ((score_mlp.fit_exp_initialization(4), score_mlp.EXP_FITS[4]),
                          (score_mlp.fit_exp_initialization(8, restarts=1), score_mlp.EXP_FITS[8])):
        for name in ("amplitude", "slope", "offset"):
            np.testing.assert_allclose(getattr(refit, name), getattr(pinned, name), rtol=1e-3, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(refit.objective, pinned.objective, rtol=1e-3)
    assert score_mlp.kernel_initialization(8) == {name: getattr(score_mlp.EXP_FITS[8], name)
                                                   for name in ("amplitude", "slope", "offset")}


def test_exp_fit_is_deterministic_and_validated():
    first, second = (score_mlp.fit_exp_initialization(3, restarts=3, seed=5) for _ in range(2))
    assert first == second
    np.testing.assert_allclose(_objective_and_error(first), [first.objective, first.max_relative_error], rtol=1e-9)
    for width in (0, True, 2.0):
        with pytest.raises(ValueError, match="positive integer"):
            score_mlp.fit_exp_initialization(width)
    with pytest.raises(ValueError, match="start"):
        score_mlp.fit_exp_initialization(2, restarts=0)


def test_exp_residual_constants_start_exactly_at_exp():
    assert score_mlp.exp_residual_initialization(1) == {"amplitude": (0.0,), "slope": (1.0,), "offset": (0.0,)}
    constants = score_mlp.initial_constants(4, "exp_residual")
    assert constants["amplitude"] == (0.0,) * 4 and constants["slope"] == (1.0,) * 4
    np.testing.assert_allclose(constants["offset"], [-3.0, -1.0, 1.0, 3.0])
    np.testing.assert_allclose(score_mlp.initial_constants(8, "exp_residual")["offset"], np.linspace(-3, 3, 8))
    with pytest.raises(ValueError, match="form"):
        score_mlp.initial_constants(4, "mixture")


def _model(form="kernel", width=4, **changes):
    constants = (score_mlp.initial_constants(width, form) if form == "exp_residual" or width in score_mlp.EXP_FITS
                 else {"amplitude": (1.0,) * width, "slope": (1.0,) * width,
                       "offset": tuple(np.linspace(-1, 1, width).tolist())})
    model = dict(vocab_size=16, max_seq_len=8, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
                 qk_head_dim=4, value_head_dim=4, intermediate_size=12, attention_backend="blocked",
                 attention_normalization=False, attention_output_scaling="head_rms_norm", qk_norm=True,
                 qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference", attention_score_mlp_width=width,
                 attention_score_mlp_form=form, attention_score_mlp_amplitude_initial=constants["amplitude"],
                 attention_score_mlp_slope_initial=constants["slope"],
                 attention_score_mlp_offset_initial=constants["offset"])
    return {**model, **changes}


def _config(form="kernel", width=4, **changes):
    config = Config(model=ModelConfig(**_model(form, width, **changes)))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = 1e-3
    validate_config(config)
    return config


def _control(**changes):
    """The unnormalized exp recipe the score MLP replaces: identical except for its settings."""
    model = {key: value for key, value in _model().items() if not key.startswith("attention_score_mlp_")}
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length, config.data.synthetic_vocab_size = 9, 16
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    validate_config(config)
    return config


@pytest.mark.parametrize("width", (1, 4, 8))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_configuration_accepts_both_forms_with_rope_none_and_the_cumulative_gate(form, width):
    for changes in ({}, {"positional_encoding": "none"}, {"attention_forget_gate": True},
                    {"attention_score_mlp_kernel_weight_decay": True}):
        config = _config(form, width, **changes)
        assert config.model.attention_score_mlp_amplitude_initial == tuple(_model(form, width)[
            "attention_score_mlp_amplitude_initial"])
        assert config.model.qk_norm and not config.model.mosaic_supported()


@pytest.mark.parametrize(("changes", "message"), (
    ({"positional_encoding": "path"}, "PaTH"),
    ({"attention_forget_gate": True, "attention_forget_gate_mode": "query_distance"}, "cumulative input forget gate"),
    ({"attention_forget_gate": True, "attention_forget_gate_form": "constant"}, "cumulative input forget gate"),
    ({"attention_normalization": True}, "replaces the kernel"),
    ({"attention_degree": 4}, "replaces the kernel"),
    ({"attention_degree": 0}, "replaces the kernel"),
    ({"attention_degree": 4, "attention_coefficients": "learned"}, "replaces the kernel"),
    ({"attention_degree": 2, "attention_fixed_coefficients": (1.0, 1.0, .5)}, "replaces the kernel"),
    ({"attention_degree": 2, "quadratic_mode": "unrestricted"}, "replaces the kernel"),
    ({"attention_degree": 4, "attention_coefficients": "learned", "attention_coefficient_sharing": "per_head",
      "attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh"}, "replaces the kernel"),
    ({"attention_mechanism": "gla", "qk_norm": False}, "replaces the kernel"),
    ({"attention_backend": "fused"}, "blocked"),
    ({"attention_backend": "mosaic"}, "blocked"),
    ({"attention_backend": "auto"}, "blocked"),
    ({"attention_backend": "xla"}, "blocked"),
    ({"attention_score_mlp_width": 0}, "positive integer"),
    ({"attention_score_mlp_width": True}, "positive integer"),
    ({"attention_score_mlp_width": 4.0}, "positive integer"),
    ({"attention_score_mlp_form": "mixture"}, "Unknown score-MLP form"),
    ({"attention_score_mlp_activation": "gelu"}, "Unknown score-MLP activation"),
    ({"attention_score_mlp_kernel_weight_decay": 1}, "boolean"),
    ({"attention_score_mlp_amplitude_initial": (1.0,) * 3}, "must hold 4 finite numbers"),
    ({"attention_score_mlp_slope_initial": (1.0, 1.0, float("nan"), 1.0)}, "finite numbers"),
    ({"attention_score_mlp_offset_initial": None}, "finite numbers"),
    ({"attention_score_mlp_offset_initial": (1.0, True, 0.0, 0.0)}, "finite numbers"),
))
def test_configuration_rejects_unsupported_settings(changes, message):
    with pytest.raises(ValueError, match=message):
        _config(**changes)


@pytest.mark.parametrize("changes", (
    {"attention_score_mlp_form": "exp_residual"}, {"attention_score_mlp_kernel_weight_decay": True},
    {"attention_score_mlp_amplitude_initial": (0.0,)}, {"attention_score_mlp_slope_initial": (1.0,)},
))
def test_score_mlp_settings_require_a_width(changes):
    with pytest.raises(ValueError, match="require model.attention_score_mlp_width"):
        _control(**changes)


def test_continuation_cannot_introduce_a_score_mlp():
    config = _config()
    continuation = config.continuation
    continuation.parent_config, continuation.parent_checkpoint = "parent.yaml", "checkpoint"
    continuation.parent_config_sha256 = continuation.parent_checkpoint_sha256 = "0" * 64
    continuation.parent_step = 1
    with pytest.raises(ValueError, match="score-MLP"):
        validate_config(config)


@pytest.mark.parametrize(("changes", "message"), (
    ({"positional_encoding": "path"}, "PaTH"),
    ({"backend": "fused"}, "blocked backend"),
    ({"attention_normalization": True}, "replaces the kernel"),
    ({"attention_degree": 2}, "replaces the kernel"),
    ({"attention_forget_gate": True, "attention_forget_gate_mode": "query_distance"}, "cumulative input forget gate"),
    ({"attention_forget_gate": True, "attention_forget_gate_form": "constant"}, "cumulative input forget gate"),
    ({"score_mlp_width": 0}, "positive integer"),
    ({"score_mlp_form": "mixture"}, "Unknown score-MLP form"),
))
def test_module_rejects_unsupported_settings(changes, message):
    settings = {"attention_normalization": False, "backend": "blocked", "score_mlp_width": 4, **changes}
    with pytest.raises(ValueError, match=message):
        MultiHeadAttention(16, 2, 2, 4, 4, rngs=nnx.Rngs(0), **settings)


def test_dispatch_rejects_incomplete_or_conflicting_arrays():
    q, k, v, a, b, w = _units()
    with pytest.raises(ValueError, match="amplitudes, offsets, and slopes"):
        _attend(q, k, v, a, None, w)
    with pytest.raises(ValueError, match="alone"):
        _attend(q, k, v, a, b, w, attention_degree=2)
    with pytest.raises(ValueError, match="cumulative"):
        _attend(q, k, v, a, b, w, query_log_decay=jnp.zeros(q.shape[:3]), query_decay_mode="query_distance")
    with pytest.raises(ValueError, match=r"\[heads,J\]"):
        _attend(q, k, v, a, b, w[:, :2])
    with pytest.raises(ValueError, match="form"):
        _attend(q, k, v, a, b, w, form="mixture")
    with pytest.raises(ValueError, match="alone"):
        attention_core(q, k, v, attention_normalization=False, attention_degree=2, score_mlp_amplitude=a,
                       score_mlp_offset=b, score_mlp_slope=w)
    with pytest.raises(ValueError, match="blocked"):
        score_mlp.score_mlp_attention(q, k, v, a, b, w, backend="fused")


def test_existing_recipes_are_unchanged(tmp_path):
    base = load_config(BASE_CONFIG)
    assert base.model.attention_score_mlp_width is None and base.model.attention_score_mlp_form == "kernel"
    control = _control(attention_forget_gate=False)
    recipe = config_to_dict(control)
    legacy = {**recipe, "model": {key: value for key, value in recipe["model"].items()
                                  if not key.startswith("attention_score_mlp_")}}
    assert config_dicts_equal(recipe, legacy) and not config_dicts_equal(config_to_dict(_config()), legacy)
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy))
    assert asdict(load_config(path).model) == asdict(control.model)
    assert not any(spec.name.startswith("score_mlp") for spec in attention_parameter_specs(control.model))
    identity = attention_identity(control)
    assert identity["attention_kernel"] == "exp_unnormalized" and not any("score_mlp" in key for key in identity)
    assert model_parameter_count(control.model) == model_parameter_count(_config().model) - 2 * 2 * 4 * (2 * 8 + 3)
    # A score-MLP recipe stores its constants as YAML lists and loads them back exactly.
    config = _config("kernel", 8)
    save_config(config, tmp_path / "score_mlp.yaml")
    stored = yaml.safe_load((tmp_path / "score_mlp.yaml").read_text())["model"]
    assert stored["attention_score_mlp_offset_initial"] == list(score_mlp.EXP_FITS[8].offset)
    assert asdict(load_config(tmp_path / "score_mlp.yaml").model) == asdict(config.model)


@pytest.mark.parametrize("decay", (False, True))
def test_parameters_initialization_and_weight_decay(decay):
    config = _config("kernel", 4, attention_score_mlp_kernel_weight_decay=decay)
    specs = {spec.name: spec for spec in attention_parameter_specs(config.model) if spec.name.startswith("score_mlp")}
    assert tuple(specs) == score_mlp.PARAMETERS
    fit = score_mlp.EXP_FITS[4]
    for name, initial in ((score_mlp.AMPLITUDE_BIAS, fit.amplitude), (score_mlp.OFFSET_BIAS, fit.offset),
                          (score_mlp.SLOPE, fit.slope)):
        assert specs[name].initial == initial and not specs[name].weight_decay and specs[name].sharing == "per_head"
    for name in (score_mlp.AMPLITUDE_KERNEL, score_mlp.OFFSET_KERNEL):
        assert specs[name].initial_shape == (8, 4) and specs[name].weight_decay is decay
        assert specs[name].initial_std == 0 and specs[name].initializer is None
    assert attention_parameter_count(config.model) - attention_parameter_count(_control().model) == 2 * 2 * 4 * (2 * 8 + 3)
    model, control = build_model(config.model, 11), build_model(_control().model, 11)
    params, control_params = nnx.state(model, nnx.Param), nnx.state(control, nnx.Param)
    flat, control_flat = dict(nnx.to_flat_state(params)), dict(nnx.to_flat_state(control_params))
    assert set(flat) - set(control_flat) == {(name,) for name in score_mlp.PARAMETERS}
    for path, value in control_flat.items():
        np.testing.assert_array_equal(flat[path][...], value[...], err_msg=str(path))
    for name in (score_mlp.AMPLITUDE_KERNEL, score_mlp.OFFSET_KERNEL):
        np.testing.assert_array_equal(params[name][...], np.zeros((2, 2, 8, 4), np.float32))
    for name, initial in ((score_mlp.AMPLITUDE_BIAS, fit.amplitude), (score_mlp.OFFSET_BIAS, fit.offset),
                          (score_mlp.SLOPE, fit.slope)):
        assert params[name][...].dtype == jnp.float32
        np.testing.assert_array_equal(params[name][...], np.broadcast_to(np.asarray(initial, np.float32), (2, 2, 4)))
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(params, excluded_names=excluded))}
    assert {name: mask[(name,)] for name in score_mlp.PARAMETERS} == {
        score_mlp.AMPLITUDE_KERNEL: decay, score_mlp.AMPLITUDE_BIAS: False, score_mlp.OFFSET_KERNEL: decay,
        score_mlp.OFFSET_BIAS: False, score_mlp.SLOPE: False}
    # AdamW's actual zero-gradient update decays exactly the masked arrays.
    for name in score_mlp.PARAMETERS:
        params[name][...] = jnp.ones_like(params[name][...])
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    updates = dict(nnx.to_flat_state(updates))
    for name in score_mlp.PARAMETERS:
        expected = -1e-3 * config.optim.weight_decay if mask[(name,)] else 0.0
        np.testing.assert_allclose(updates[(name,)][...], expected, rtol=1e-6, atol=0, err_msg=name)
    facts = acceptance.matches_softmax_backbone_initialization(config)
    assert {name: facts["mechanism_parameters"][name]["weight_decay"] for name in score_mlp.PARAMETERS} == {
        name: mask[(name,)] for name in score_mlp.PARAMETERS}


def test_exp_residual_model_starts_at_the_exp_control():
    tokens = jnp.asarray(np.random.default_rng(1).integers(0, 16, size=(2, 8)), jnp.int32)
    for width in (1, 8):
        model = build_model(_config("exp_residual", width).model, 23)
        control = build_model(_control().model, 23)
        np.testing.assert_array_equal(model(tokens), control(tokens))
        # The kernel form starts near exp but not at it; its units are live.
        kernel = build_model(_config("kernel", 4).model, 23)
        assert 0 < np.abs(np.asarray(kernel(tokens) - control(tokens))).max() < 1.0


@pytest.mark.parametrize(("form", "gate"), (("kernel", False), ("exp_residual", False), ("kernel", True)))
def test_training_updates_every_score_mlp_parameter(form, gate):
    config = _config(form, 4, attention_forget_gate=gate, attention_score_mlp_kernel_weight_decay=True)
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.asarray(np.random.default_rng(2).integers(0, 16, size=(2, 9)), jnp.int32)
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    second, _, _ = step(first, jax.random.PRNGKey(2))   # at alpha = 0, only alpha and A move on the first update
    assert np.isfinite(float(loss))
    for name in score_mlp.PARAMETERS + (GATE_PARAMETERS if gate else ()):
        assert np.linalg.norm(np.asarray(second.params[name][...] - initial.params[name][...])) > 0, name


@pytest.mark.parametrize(("form", "gate"), (("kernel", False), ("exp_residual", True)))
def test_diagnostics_identify_the_score_mlp_and_summarize_its_units(form, gate):
    config = _config(form, 4, attention_forget_gate=gate)
    identity = attention_identity(config)
    kernel = score_mlp.KERNEL_NAMES[form]
    assert identity["attention_kernel"] == kernel and identity["attention_signed"] is True
    assert "SiLU(w_j s_ti + b_j(x_t))" in identity["attention_formula"]
    assert ("exp(s_ti) (1 + " in identity["attention_formula"]) is (form == "exp_residual")
    assert ("exp(L_t - L_i)" in identity["attention_formula"]) is gate
    assert identity["attention_formula"].endswith("per-head RMSNorm over value channels with a learned gain")
    assert identity["attention_score_mlp_width"] == 4 and identity["attention_score_mlp_form"] == form
    assert identity["attention_score_mlp_parameter_count"] == 2 * 2 * 4 * (2 * 8 + 3)
    assert identity["attention_score_mlp_offset_initial"] == list(config.model.attention_score_mlp_offset_initial)
    assert identity.get("attention_forget_gate", False) is gate
    state = create_state(config, jax.random.PRNGKey(3))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == ("unshifted_score_mlp" if form == "kernel"
                                       else "unshifted_exp_times_one_plus_score_mlp")
    json.dumps(report, allow_nan=False)
    for layer in report["layers"]:
        assert {"kernel_min", "kernel_max", "weight_sum_min", "weight_sum_max", "negative_weight_fraction"} <= set(layer)
        assert not any(name.startswith("score_mlp_") for name in layer)
    units = report["coefficients"]
    assert units["kind"] == "score_mlp" and units["width"] == 4 and units["finite"]
    assert units["effective_coefficient_names"] == ["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3"]
    summary = {name: np.asarray(units["effective_summary"][name]) for name in ("mean", "std", "min", "max")}
    model = config.model
    start = np.float32(model.attention_score_mlp_amplitude_initial + model.attention_score_mlp_offset_initial)
    # At A = B = 0 every token's units are the starting constants (up to FP32 averaging).
    for name in ("mean", "min", "max"):
        np.testing.assert_allclose(summary[name], np.broadcast_to(start, (2, 2, 8)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(summary["std"], 0, atol=1e-5)
    np.testing.assert_array_equal(units["slope"], np.broadcast_to(np.float32(model.attention_score_mlp_slope_initial),
                                                                  (2, 2, 4)))
    np.testing.assert_array_equal(units["amplitude_kernel_l2"], np.zeros((2, 2, 4)))
    assert ("forget_gate" in units) is gate and (not gate or units["forget_gate"]["finite"])
    metrics = tracking.attention_metrics(report)
    assert f"attention/{kernel}/coefficients/layer01/head01/b3/mean" in metrics
    assert f"attention/{kernel}/layer00/kernel_max" in metrics
    tags = tracking.run_tags(config, {})
    assert {kernel, f"score_mlp_{form}", "score_mlp_width4", "output_head_rms_norm"} <= set(tags)
    assert ("forget_gate" in tags) is gate


def test_acceptance_selects_the_score_mlp_tests_on_cpu_and_gpu():
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for form in score_mlp.FORMS:
        recipe = config_to_dict(_config(form))
        for cpu in (True, False):
            assert acceptance.SCORE_MLP_TESTS in acceptance.test_paths(spec, recipe, resolved_backend="blocked", cpu=cpu)
        production = acceptance.attention_spec(_config(form), entry_module="tests")
        paths = acceptance.test_paths(production, recipe, resolved_backend="blocked", cpu=True)
        assert "scaffolds/transformer/tests/test_unnormalized_attention.py" in paths
    for cpu in (True, False):
        assert acceptance.SCORE_MLP_TESTS not in acceptance.test_paths(
            spec, config_to_dict(_control()), resolved_backend="blocked", cpu=cpu)
    facts = acceptance.matches_softmax_backbone_initialization(_config("exp_residual", 8))
    assert facts["mechanism_parameters"][score_mlp.OFFSET_BIAS]["initial"] == list(
        score_mlp.exp_residual_initialization(8)["offset"])


def test_run_index_exposes_width_and_form(tmp_path):
    study = tmp_path / "experiments/demo"
    study.mkdir(parents=True)
    (study / "sweep.yaml").write_text(json.dumps({"study_id": "demo", "run_ids": []}))
    create_run(study, config_to_dict(_config("exp_residual", 8)), label="score-mlp")
    create_run(study, config_to_dict(_config("kernel", 4, attention_score_mlp_kernel_weight_decay=True)),
               label="score-mlp-kernel")
    create_run(study, config_to_dict(_control()), label="exp")
    regenerate_index(study)
    with (study / "runs.csv").open() as stream:
        rows = {row["label"]: row for row in csv.DictReader(stream)}
    fields = ("attention_family", "score_mlp_width", "score_mlp_form", "score_mlp_kernel_weight_decay")
    assert {label: tuple(row[field] for field in fields) for label, row in rows.items()} == {
        "score-mlp": ("score_mlp", "8", "exp_residual", "False"),
        "score-mlp-kernel": ("score_mlp", "4", "kernel", "True"),
        "exp": ("exponential", "", "", ""),
    }


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_backward_memory_does_not_grow_with_width(form):
    """The compiled blocked backward keeps a fixed number of score tiles for J = 1 and J = 16."""
    def temporary_bytes(width):
        q, k, v, a, b, w = _units(width, time=512, heads=2)
        loss = lambda *x: jnp.sum(blocked_dot_product_attention(
            *x[:3], attention_normalization=False, score_mlp_amplitude=x[3], score_mlp_offset=x[4],
            score_mlp_slope=x[5], score_mlp_form=form) ** 2)
        compiled = jax.jit(jax.grad(loss, argnums=tuple(range(6)))).lower(q, k, v, a, b, w).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    tile = 1 * 2 * 128 * 512 * 4
    narrow, wide = temporary_bytes(1), temporary_bytes(16)
    assert wide <= narrow + tile, (narrow / tile, wide / tile)


def _production_shape_errors(form, width, *, gate=False, time=2048, heads=4, channels=64, seed=91):
    """Relative L2 errors of BF16 Q/K/V through the blocked kernel against the dense FP32 HIGHEST definition."""
    rng = np.random.default_rng(seed + width)
    q, k, v = (rng.normal(size=(1, time, heads, channels)) for _ in range(3))
    constants = score_mlp.initial_constants(width, form)
    alpha, beta, slope = (np.broadcast_to(np.asarray(constants[name]), (heads, width))
                          for name in ("amplitude", "offset", "slope"))
    # Query-dependent units around the recipe constants: the form's starting point after some training.
    a = alpha + (.1 + .1 * np.abs(alpha)) * rng.normal(size=(1, time, heads, width))
    b = beta + .5 * rng.normal(size=(1, time, heads, width))
    arrays = [q, k, v, a, b, slope] + ([np.cumsum(-np.exp(rng.uniform(-8, -2, size=(1, time, heads))), axis=1)]
                                       if gate else [])
    cotangent = jnp.asarray(rng.normal(size=v.shape), jnp.bfloat16)
    bf16 = tuple(jnp.asarray(x, jnp.bfloat16 if index < 3 else jnp.float32) for index, x in enumerate(arrays))
    reference_inputs = tuple(jnp.asarray(x, jnp.float32) for x in bf16)   # the same BF16 values, upcast

    def production(q, k, v, a, b, w, log_decay=None):
        return blocked_dot_product_attention(q, k, v, attention_normalization=False, score_mlp_amplitude=a,
                                             score_mlp_offset=b, score_mlp_slope=w, score_mlp_form=form,
                                             query_log_decay=log_decay)

    def dense(q, k, v, a, b, w, log_decay=None):
        return _dense_units(q, k, v, a, b, w, log_decay, form=form)

    with jax.default_matmul_precision("highest"):
        expected = jax.jit(lambda *x: _vjp(dense, x, cotangent.astype(jnp.float32)))(*reference_inputs)
    # The trainer's default matmul precision (TF32 for the FP32 weight-value products on GPUs).
    actual = jax.jit(lambda *x: _vjp(production, x, cotangent))(*bf16)
    names = ["output", "q", "k", "v", "a", "b", "w", "L"]
    return {name: _relative_error(observed, wanted)
            for name, observed, wanted in zip(names, [actual[0], *actual[1]], [expected[0], *expected[1]])}


def test_bfloat16_qualification_runs_at_a_small_shape():
    """The GPU qualification's comparison, at a CPU-sized context."""
    errors = _production_shape_errors("exp_residual", 4, gate=True, time=160, heads=2, channels=16)
    assert max(errors.values()) < .02, errors


@pytest.mark.gpu
@pytest.mark.parametrize(("form", "width", "gate"), (
    ("kernel", 4, False), ("kernel", 8, False), ("exp_residual", 4, False), ("exp_residual", 8, False),
    ("exp_residual", 8, True),
))
def test_gpu_production_shape_bfloat16_matches_float32_highest(form, width, gate):
    """Qualify the production 2,048-token, 64-channel blocked score MLP on its allocated GPU."""
    if jax.default_backend() != "gpu":
        pytest.skip("Requires the experiment's GPU allocation")
    errors = _production_shape_errors(form, width, gate=gate)
    assert max(errors.values()) < .012, errors
