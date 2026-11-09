"""Per-order GDN coefficient gates: one shared start, epsilon-level parity, gradients, diagnostics, Mosaic."""

from dataclasses import replace
import json
import math
import zlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention import input_dependent
from scaffolds.transformer.attention.gated_deltanet import A_LOG_RANGE, DT_FLOOR, DT_RANGE
from scaffolds.transformer.attention.input_dependent import (
    GDN_GATE_PARAMETERS, coefficient_gate_logits, coefficient_gate_statistics, gdn_gate_log_multipliers,
    project_coefficients,
)
from scaffolds.transformer.attention.parameters import attention_parameter_specs, factorial_scale, mechanism_key
from scaffolds.transformer.configs import config_to_dict, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_coefficient_gates import (
    _config as _sigmoid_config, _flat, _model as _sigmoid_model, _params, _per_order, _relative,
    _requires_mosaic_gpu, _vjp,
)
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import attention_weight_decay_mask, make_optimizer
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import attention_metrics, run_tags


DEGREES = (4, 8)
KERNEL, LOG_RATE, DT_BIAS = GDN_GATE_PARAMETERS


def _config(degree=4, **changes):
    """The sigmoid gate tests' backbone and production generator, with the GDN gate instead."""
    return _sigmoid_config(degree, **{"attention_coefficient_gate": "gdn", **changes})


def _ungated(config):
    return replace(config, model=replace(config.model, attention_coefficient_gate="none"))


def _softplus(values):
    return np.logaddexp(0.0, values)


@pytest.mark.parametrize("degree", DEGREES)
def test_every_order_starts_as_one_forget_gate_draw_and_the_backbone_is_unchanged(degree):
    config = _config(degree)
    key = jax.random.PRNGKey(42)
    gated, ungated = build_model(config.model, key), build_model(_ungated(config).model, key)
    actual, baseline = _params(gated), _params(ungated)
    assert set(actual) - set(baseline) == {(name,) for name in GDN_GATE_PARAMETERS}
    for path, value in baseline.items():
        np.testing.assert_array_equal(actual[path], value, err_msg=str(path))
    layers, heads, hidden, count = 2, 2, 16, degree + 1
    assert model_parameter_count(config.model) - model_parameter_count(_ungated(config).model) == (
        layers * heads * (hidden + 2) * count)
    shapes = {KERNEL: (layers, heads, hidden, count), LOG_RATE: (layers, heads, count),
              DT_BIAS: (layers, heads, count)}
    for name, shape in shapes.items():
        value = np.asarray(actual[(name,)])
        assert value.shape == shape and value.dtype == np.float32, name
        np.testing.assert_array_equal(value, np.broadcast_to(value[..., :1], shape), err_msg=name)
    # Order 0 is the input-form forget gate's draw per layer and head, regenerated here from the
    # mechanism stream by parameter name with Gated DeltaNet's formulas.
    stream = lambda name: jax.random.fold_in(mechanism_key(key), zlib.crc32(name.encode()) & 0x7FFFFFFF)
    bound = 1 / math.sqrt(hidden)
    dt = jnp.maximum(jnp.exp(jax.random.uniform(stream(DT_BIAS), (layers, heads), jnp.float32,
                                                math.log(DT_RANGE[0]), math.log(DT_RANGE[1]))), DT_FLOOR)
    expected = {
        KERNEL: jax.random.uniform(stream(KERNEL), (layers, heads, hidden), jnp.float32, -bound, bound),
        LOG_RATE: jnp.log(jax.random.uniform(stream(LOG_RATE), (layers, heads), jnp.float32, *A_LOG_RANGE)),
        DT_BIAS: dt + jnp.log(-jnp.expm1(-dt)),
    }
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[(name,)][..., 0], value, err_msg=name)
    assert np.all(np.abs(actual[(KERNEL,)]) <= bound)
    rate, dt = np.exp(np.asarray(actual[(LOG_RATE,)], np.float64)), _softplus(np.asarray(actual[(DT_BIAS,)], np.float64))
    assert np.all((rate > 0) & (rate <= A_LOG_RANGE[1] * (1 + 1e-6)))
    assert np.all((dt >= DT_RANGE[0] * (1 - 1e-5)) & (dt <= DT_RANGE[1] * (1 + 1e-5)))
    # At every token all orders of a head share one multiplier; the generator is the production one.
    x = gated.layers_0.attention_norm(gated.token_embedding(jnp.arange(32).reshape(2, 16) % 32))
    multiplier = np.asarray(jnp.exp(gdn_gate_log_multipliers(
        coefficient_gate_logits(x, actual[(KERNEL,)][0], actual[(DT_BIAS,)][0]), actual[(LOG_RATE,)][0])))
    np.testing.assert_array_equal(multiplier, np.broadcast_to(multiplier[..., :1], multiplier.shape))
    assert np.all((multiplier > 0) & (multiplier <= 1))


@pytest.mark.parametrize("degree", DEGREES)
def test_initial_logits_match_the_ungated_model_up_to_the_rms_epsilon(degree):
    """With the RMSNorm epsilon removed, alpha cancels to FP32 rounding; at 1e-6 the logits move by
    the epsilon's share of the smallest head output (acceptance.gdn_gate_start_parity)."""
    config = _config(degree)
    config.train.seed = 42
    tokens = jnp.asarray(np.random.default_rng(degree).integers(0, 32, size=(2, 16)), jnp.int32)
    parity = acceptance.gdn_gate_start_parity(config, tokens)
    assert parity["gate_parameters_repeat_one_draw_over_orders"]
    assert len(parity["head_output_relative_change_bound"]) == config.model.num_layers
    assert parity["epsilon_free_logit_difference_relative_to_largest"] <= acceptance.GATE_START_ROUNDING
    assert parity["logit_difference_relative_to_largest"] <= parity["tolerance_relative_to_largest"]
    # The epsilon, not rounding, separates the two starts at the recipe's setting.
    assert (parity["logit_difference_relative_to_largest"]
            > 10 * parity["epsilon_free_logit_difference_relative_to_largest"])


def test_independent_draws_per_order_fail_the_start_parity(monkeypatch):
    """Orders drawn independently no longer share alpha, and the RMSNorm cannot cancel it."""
    original = input_dependent.parameter_specs
    monkeypatch.setattr(input_dependent, "parameter_specs", lambda **settings: tuple(
        replace(spec, repeat_initial_axis=None) for spec in original(**settings)))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 32, size=(2, 16)), jnp.int32)
    with pytest.raises(RuntimeError, match="initial logits differ"):
        acceptance.gdn_gate_start_parity(_config(), tokens)


@pytest.mark.parametrize("degree", DEGREES)
def test_gdn_gated_generator_matches_an_independent_float64_formula_and_gradients(degree):
    rng = np.random.default_rng(21 + degree)
    heads, width, count = 2, 6, degree + 1
    factorials = np.asarray([math.factorial(r) for r in range(count)], np.float64)
    x = rng.normal(size=(2, 5, width))
    w, u = (rng.normal(size=(heads, width, count)) * 0.4 for _ in range(2))
    b, a = 1 + 0.3 * rng.normal(size=(heads, count)), 0.7 * rng.normal(size=(heads, count))
    dt_bias, log_rate = rng.normal(size=(heads, count)) - 2, 0.5 * rng.normal(size=(heads, count))
    cotangent = rng.normal(size=(2, 5, heads, count)) * factorials  # comparable cotangents for every order

    def production(x, w, b, a, u, dt_bias, log_rate):
        return project_coefficients(x, w, factorial_scale(b), amplitude=a, activation="tanh",
                                    amplitude_scale="factorial", gate_kernel=u, gate_bias=dt_bias,
                                    gate_log_rate=log_rate)

    def oracle(x, w, b, a, u, dt_bias, log_rate):
        c = (b + a * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, w))) / factorials
        rate = jnp.exp(log_rate) * jnp.log1p(jnp.exp(jnp.einsum("btd,hdr->bthr", x, u) + dt_bias))
        return c * jnp.exp(-rate)

    operands = tuple(jnp.asarray(value, jnp.float32) for value in (x, w, b, a, u, dt_bias, log_rate))
    actual = jax.tree.map(np.asarray, _vjp(production, operands, jnp.asarray(cotangent, jnp.float32)))
    with jax.enable_x64():
        expected = jax.tree.map(np.asarray, _vjp(
            oracle, tuple(jnp.asarray(value, jnp.float64) for value in operands), jnp.asarray(cotangent)))
    assert actual[0].dtype == np.float32 and all(gradient.dtype == np.float32 for gradient in actual[1])
    _per_order(actual[0], expected[0], 2e-6, "coefficients")
    _relative(actual[1][0], expected[1][0], 2e-5, "dx")
    for name, observed, wanted in zip(("dW", "db", "da", "dw", "ddt_bias", "dA_log"), actual[1][1:], expected[1][1:]):
        _per_order(observed, wanted, 2e-5, name)


def test_nonfinite_gate_inputs_give_nan_rather_than_a_saturated_gate():
    logits = jnp.asarray([[[[0.0, jnp.inf, -jnp.inf, jnp.nan]]]], jnp.float32)
    log_alpha = np.asarray(gdn_gate_log_multipliers(logits, jnp.zeros((1, 4), jnp.float32)))
    assert np.isfinite(log_alpha[..., 0]).all() and np.isnan(log_alpha[..., 1:]).all()
    infinite_rate = np.asarray(gdn_gate_log_multipliers(jnp.zeros((1, 1, 1, 2), jnp.float32),
                                                        jnp.asarray([[0.0, jnp.inf]], jnp.float32)))
    assert np.isfinite(infinite_rate[..., 0]).all() and np.isnan(infinite_rate[..., 1]).all()


def test_weight_decay_covers_both_projections_and_spares_a_b_and_the_gate_rates():
    config = _config()
    model = build_model(config.model, 42)
    decayed = ("query_polynomial_kernel", KERNEL)
    spared = ("query_polynomial_bias", "query_polynomial_amplitude", LOG_RATE, DT_BIAS)
    for name in (*decayed, *spared):  # nonzero, so that a zero-gradient update reveals any decay
        getattr(model, name)[...] = jnp.full_like(getattr(model, name)[...], 0.25)
    params = nnx.state(model, nnx.Param)

    def mask(recipe, parameters):
        excluded = frozenset(spec.name for spec in attention_parameter_specs(recipe) if not spec.weight_decay)
        return {path: bool(value.get_value()) for path, value in
                nnx.to_flat_state(attention_weight_decay_mask(parameters, excluded_names=excluded))}

    gated = mask(config.model, params)
    assert all(gated[(name,)] for name in decayed) and not any(gated[(name,)] for name in spared)
    control = _ungated(config).model
    assert {path: decay for path, decay in gated.items() if path[0] not in GDN_GATE_PARAMETERS} == mask(
        control, nnx.state(build_model(control, 42), nnx.Param))
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    values = _flat(updates)
    for name in decayed:
        np.testing.assert_allclose(values[(name,)], -1e-3 * 0.1 * 0.25, rtol=2e-6, atol=1e-10)
    for name in spared:
        np.testing.assert_array_equal(values[(name,)], 0)
    # w follows attention_coefficient_kernel_weight_decay, like W and the sigmoid gate's U.
    undecayed = replace(config.model, attention_coefficient_kernel_weight_decay=False)
    assert not any(spec.weight_decay for spec in attention_parameter_specs(undecayed)
                   if spec.name in GDN_GATE_PARAMETERS)


def test_gdn_gate_statistics_match_a_dense_numpy_oracle():
    rng = np.random.default_rng(9)
    batch, time, heads, width, count = 2, 7, 3, 4, 5
    logits, ungated = rng.normal(size=(batch, time, heads, count)) * 2 - 1, rng.normal(size=(batch, time, heads, count))
    log_rate = 0.5 * rng.normal(size=(heads, count))
    gated = np.exp(-np.exp(log_rate) * _softplus(logits)) * ungated
    q, k = (rng.normal(size=(batch, time, heads, width)) for _ in range(2))
    inputs = tuple(np.asarray(value, np.float32) for value in (logits, ungated, gated, q, k, log_rate))
    statistics = jax.device_get(coefficient_gate_statistics(*(jnp.asarray(value) for value in inputs[:5]),
                                                            gate_log_rate=jnp.asarray(inputs[5])))
    logits, ungated, gated, q, k, log_rate = (value.astype(np.float64) for value in inputs)
    log_alpha = -np.exp(log_rate) * _softplus(logits)
    alpha = np.exp(log_alpha)
    tilt = log_alpha - log_alpha.mean(axis=-1, keepdims=True)
    expected = {"mean": alpha.mean(axis=(0, 1)), "std": alpha.std(axis=(0, 1)), "min": alpha.min(axis=(0, 1)),
                "max": alpha.max(axis=(0, 1)), "log_tilt_mean": tilt.mean(axis=(0, 1)),
                "log_tilt_std": tilt.std(axis=(0, 1)), "ungated_mean": ungated.mean(axis=(0, 1)),
                "ungated_std": ungated.std(axis=(0, 1))}
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(width)
    causal = np.tril(np.ones((time, time), bool))[None, None, :, :, None]
    terms = np.where(causal, gated.transpose(0, 2, 1, 3)[:, :, :, None, :]
                     * scores[..., None] ** np.arange(count), 0.0)  # [B,H,T,T,R]
    pairs = batch * time * (time + 1) / 2
    rows = np.abs(terms).sum(axis=3)
    expected.update(contribution_rms=np.sqrt((terms ** 2).sum(axis=(0, 2, 3)) / pairs),
                    contribution_abs_mean=np.abs(terms).sum(axis=(0, 2, 3)) / pairs,
                    contribution_row_share=(rows / rows.sum(axis=-1, keepdims=True)).mean(axis=(0, 2)),
                    attention_kernel_abs_mean=np.abs(terms.sum(axis=-1)).sum(axis=(0, 2, 3)) / pairs)
    assert set(statistics) == {f"coefficient_gate_{name}" for name in expected}
    for name, values in expected.items():
        np.testing.assert_allclose(statistics[f"coefficient_gate_{name}"], values, rtol=2e-4, atol=1e-6,
                                   err_msg=name)
    np.testing.assert_allclose(statistics["coefficient_gate_contribution_row_share"].sum(axis=-1), 1, rtol=1e-5)


@pytest.mark.parametrize("degree", DEGREES)
def test_diagnostics_report_multipliers_rates_ungated_coefficients_and_contributions(degree):
    config = _config(degree)
    state = create_state(config, jax.random.PRNGKey(0))
    rng = np.random.default_rng(4)
    # Independent per-order gates, so every reported order differs.
    for name, scale, offset in ((KERNEL, 0.3, 0.0), (LOG_RATE, 0.5, 0.0), (DT_BIAS, 1.0, -1.0),
                                ("query_polynomial_amplitude", 0.5, 0.0)):
        state.params[name][...] = jnp.asarray(rng.normal(size=state.params[name].shape) * scale + offset,
                                              jnp.float32)
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["attention_coefficient_gate"] == "gdn"
    assert report["attention_coefficient_count"] == 2 * 2 * (degree + 1) * (2 * 16 + 4)
    assert "alpha_r(x_t) c_r(x_t) s_ti^r" in report["attention_formula"]
    assert "alpha_r(x_t) = exp(-exp(A_log_r) softplus(x_t w_r + dt_bias_r))" in report["attention_formula"]
    gates = report["coefficients"]["coefficient_gates"]
    assert gates["finite"] and len(gates["order_names"]) == degree + 1 and "gate_bias" not in gates
    assert "alpha_r(x_t) = exp(-exp(A_log_r) softplus(x_t w_r + dt_bias_r))" in gates["formula"]
    model = nnx.merge(state.graphdef, state.params)
    kernel, log_rate, dt_bias = (np.asarray(getattr(model, name)[...], np.float64) for name in GDN_GATE_PARAMETERS)
    np.testing.assert_allclose(gates["gate_rate_exp_A_log"], np.exp(log_rate), rtol=1e-6)
    np.testing.assert_allclose(gates["gate_softplus_dt_bias"], _softplus(dt_bias), rtol=1e-6)
    np.testing.assert_allclose(gates["gate_kernel_l2"], np.linalg.norm(kernel, axis=-2), rtol=1e-6)
    x = np.asarray(model.layers_0.attention_norm(model.token_embedding(jnp.asarray(sample[:, :-1]))), np.float64)
    log_alpha = -np.exp(log_rate[0]) * _softplus(np.einsum("btd,hdr->bthr", x, kernel[0]) + dt_bias[0])
    alpha = np.exp(log_alpha)
    summary, tilt = gates["gate_summary"], log_alpha - log_alpha.mean(axis=-1, keepdims=True)
    for statistic, values in (("mean", alpha.mean(axis=(0, 1))), ("std", alpha.std(axis=(0, 1))),
                              ("min", alpha.min(axis=(0, 1))), ("max", alpha.max(axis=(0, 1)))):
        np.testing.assert_allclose(summary[statistic][0], values, rtol=1e-4, atol=1e-7, err_msg=statistic)
    np.testing.assert_allclose(gates["gate_log_tilt_summary"]["mean"][0], tilt.mean(axis=(0, 1)), rtol=1e-4, atol=1e-6)
    ungated = np.asarray(project_coefficients(
        jnp.asarray(x, jnp.float32), model.query_polynomial_kernel[0],
        factorial_scale(model.query_polynomial_bias[0]), amplitude=model.query_polynomial_amplitude[0],
        activation="tanh", amplitude_scale="factorial"), np.float64)
    np.testing.assert_allclose(gates["ungated_coefficient_summary"]["mean"][0], ungated.mean(axis=(0, 1)),
                               rtol=1e-5, atol=1e-9)
    gated = alpha * ungated  # what the kernel receives, summarized as the effective coefficients
    np.testing.assert_allclose(report["coefficients"]["effective_summary"]["mean"][0], gated.mean(axis=(0, 1)),
                               rtol=1e-4, atol=1e-9)
    # Order 0 multiplies s^0 = 1: its mean |contribution| counts each query once per causal key.
    time = gated.shape[1]
    weights = np.arange(1, time + 1)[None, :, None]
    np.testing.assert_allclose(np.asarray(gates["contribution_summary"]["abs_mean"])[0][..., 0],
                               (np.abs(gated[..., 0]) * weights).sum(axis=(0, 1)) / (time * (time + 1) / 2),
                               rtol=1e-4)
    np.testing.assert_allclose(np.sum(gates["contribution_summary"]["row_share"], axis=-1), 1, rtol=1e-5)
    metrics = attention_metrics(report)
    for statistic in ("gate_mean", "gate_log_tilt_mean", "contribution_row_share"):
        assert f"attention/query_polynomial_unnormalized/coefficient_gates/{gates['order_names'][-1]}/{statistic}" in metrics
    assert "coefficient_gate_gdn" in run_tags(config, {}) and "coefficient_gate_sigmoid" not in run_tags(config, {})


def test_historical_recipes_keep_their_parameters_identity_and_tests():
    gdn, ungated, sigmoid = _config(), _ungated(_config()), _sigmoid_config()
    identity = attention_identity(ungated)
    assert "attention_coefficient_gate" not in identity and "alpha_r(x_t)" not in identity["attention_formula"]
    assert identity["attention_coefficient_count"] == 2 * 2 * 5 * (16 + 2)
    assert attention_identity(sigmoid)["attention_coefficient_count"] == 2 * 2 * 9 * (2 * 16 + 3)
    assert "2 g_r(x_t) c_r(x_t)" in attention_identity(sigmoid)["attention_formula"]
    assert attention_identity(gdn)["attention_coefficient_count"] == 2 * 2 * 5 * (2 * 16 + 4)
    # Specs without a repeated draw are unchanged: the sigmoid gate keeps its exact zeros and draws nothing.
    for recipe in (ungated, sigmoid):
        specs = attention_parameter_specs(recipe.model)
        assert all(spec.repeat_initial_axis is None for spec in specs)
    sigmoid_specs = {spec.name: spec for spec in attention_parameter_specs(sigmoid.model)}
    for name in ("query_polynomial_gate_kernel", "query_polynomial_gate_bias"):
        assert set(sigmoid_specs[name].initial) == {0.0} and sigmoid_specs[name].initializer is None
    assert not set(GDN_GATE_PARAMETERS[1:]) & set(sigmoid_specs)
    legacy = config_to_dict(ungated)
    del legacy["model"]["attention_coefficient_gate"]
    assert config_dicts_equal(config_to_dict(ungated), legacy)
    assert not config_dicts_equal(config_to_dict(gdn), legacy)
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for recipe, selected, other in ((sigmoid, acceptance.COEFFICIENT_GATE_TESTS, acceptance.COEFFICIENT_GDN_GATE_TESTS),
                                    (gdn, acceptance.COEFFICIENT_GDN_GATE_TESTS, acceptance.COEFFICIENT_GATE_TESTS)):
        for cpu in (True, False):
            paths = acceptance.test_paths(spec, config_to_dict(recipe), resolved_backend="mosaic", cpu=cpu)
            assert any(path.startswith(selected) for path in paths)
            assert not any(path.startswith(other) for path in paths)
    sigmoid_gpu = acceptance.test_paths(spec, config_to_dict(sigmoid), resolved_backend="mosaic", cpu=False)
    assert [path for path in sigmoid_gpu if path.startswith(acceptance.COEFFICIENT_GATE_TESTS)] == [
        f"{acceptance.COEFFICIENT_GATE_TESTS}::{test}[{degree}]"
        for test in acceptance.COEFFICIENT_GATE_GPU_TESTS for degree in acceptance.COEFFICIENT_GATE_DEGREES]


@pytest.mark.parametrize("changes", [
    dict(attention_coefficient_activation="affine_sigmoid"), dict(attention_coefficient_readout="dense"),
    dict(attention_coefficient_orders=(0, 2, 4)), dict(attention_coefficient_tanh_bias=True),
    dict(positional_encoding="none"), dict(attention_forget_gate=True, attention_forget_gate_mode="query_distance"),
    dict(attention_coefficient_conditioning="query_key"),
])
def test_unqualified_gdn_gate_recipes_are_rejected(changes):
    with pytest.raises(ValueError):
        _sigmoid_model(4, **{"attention_coefficient_gate": "gdn", **changes})


def test_gdn_gated_continuations_and_learning_rate_scales_are_rejected():
    validate_config(_config())
    scaled = _config()
    scaled.optim.query_coefficient_learning_rate_scale = 3.0
    with pytest.raises(ValueError, match="does not cover coefficient gates"):
        validate_config(scaled)
    for preserve in (False, True):
        continued = _config()
        continued.continuation.parent_checkpoint, continued.continuation.parent_config = "parent", "parent.yaml"
        continued.continuation.parent_config_sha256 = continued.continuation.parent_checkpoint_sha256 = "0" * 64
        continued.continuation.parent_step = 1
        continued.continuation.preserve_all_parameters = preserve
        with pytest.raises(ValueError, match="Continuation with coefficient gates"):
            validate_config(continued)


def test_acceptance_selects_the_gdn_gate_checks_and_its_qualified_degree_on_the_gpu():
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for degree in DEGREES:
        recipe = config_to_dict(_config(degree))
        cpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=True)
        gpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=False)
        assert acceptance.COEFFICIENT_GDN_GATE_TESTS in cpu
        degrees = sorted({*acceptance.COEFFICIENT_GDN_GATE_DEGREES, degree})
        assert [path for path in gpu if path.startswith(acceptance.COEFFICIENT_GDN_GATE_TESTS)] == [
            f"{acceptance.COEFFICIENT_GDN_GATE_TESTS}::{test}[{selected}]"
            for test in acceptance.COEFFICIENT_GDN_GATE_GPU_TESTS for selected in degrees]
    parameters = acceptance.matches_softmax_backbone_initialization(_config())["mechanism_parameters"]
    assert parameters[KERNEL]["weight_decay"] is True and parameters[KERNEL]["initializer"] == "uniform_fan_in"
    assert {name: parameters[name]["initializer"] for name in (LOG_RATE, DT_BIAS)} == {
        LOG_RATE: "mamba2_log_rate", DT_BIAS: "mamba2_dt_bias"}
    assert all(parameters[name]["repeat_initial_axis"] == -1 for name in GDN_GATE_PARAMETERS)
    assert not parameters[LOG_RATE]["weight_decay"] and not parameters[DT_BIAS]["weight_decay"]


def test_tiny_training_moves_every_order_of_the_gates_and_resumes_exactly(tmp_path):
    config = _config()
    config.train.workdir = tmp_path / "full"
    config.train.wandb_mode = "disabled"
    config.train.eval_every, config.train.eval_batches, config.train.ckpt_every = 1, 1, 1
    full = loop.run_training(config, audit_checkpoint=True)
    restarted = replace(config, train=replace(config.train, workdir=tmp_path / "restarted"))
    before = loop.run_training(restarted, stop_after_step=1, audit_checkpoint=True)
    after = loop.run_training(restarted, audit_checkpoint=True)
    a, b = before["checkpoint_audit"], after["checkpoint_audit"]
    assert a["final_state_sha256"] == a["checkpoint_state_sha256"] == b["start_state_sha256"]
    assert a["next_update"] == b["first_update"]
    assert b["final_state_sha256"] == b["checkpoint_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    manager = make_checkpoint_manager(restarted)
    try:
        initial = create_state(restarted, jax.random.PRNGKey(0))
        state, step = restore_latest_checkpoint(manager, initial)
        assert step == config.optim.total_steps
        for name in (*GDN_GATE_PARAMETERS, "query_polynomial_bias"):
            moved = np.asarray(state.params[name][...]) != np.asarray(initial.params[name][...])
            assert moved.any(), name
        # Each order trains on its own: the copies of the shared draw separate.
        for name in GDN_GATE_PARAMETERS:
            values = np.asarray(state.params[name][...])
            assert not np.array_equal(values, np.broadcast_to(values[..., :1], values.shape)), name
    finally:
        manager.close()
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert records and all(row["attention_diagnostics"]["valid"] for row in records)
    assert all(row["attention_diagnostics"]["coefficients"]["coefficient_gates"]["finite"] for row in records)


def test_gate_gradients_are_finite_and_nonzero_for_every_order_at_initialization():
    config = _config()
    batch = jnp.asarray(np.random.default_rng(3).integers(0, 32, size=(2, 17)), jnp.int32)
    _, gradients = loop.loss_and_grads(create_state(config, jax.random.PRNGKey(42)), batch, jax.random.PRNGKey(0))
    gradients = _flat(gradients)
    for name in GDN_GATE_PARAMETERS:
        gradient = gradients[(name,)]
        magnitude = np.linalg.norm(gradient, axis=-2) if name == KERNEL else np.abs(gradient)
        assert np.isfinite(gradient).all() and np.all(magnitude > 0), name


GPU_DEGREES = acceptance.COEFFICIENT_GDN_GATE_DEGREES


@pytest.mark.gpu
@pytest.mark.parametrize("degree", GPU_DEGREES)
def test_gdn_gated_coefficients_through_mosaic_match_fp64(degree):
    """Production shapes (2,048 tokens, d_K 64): x -> GDN-gated generator -> Mosaic, against FP64, every gradient."""
    _requires_mosaic_gpu()
    from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
    from scaffolds.transformer.tests.test_fused_attention import _assert_relative
    from scaffolds.transformer.tests.test_mosaic_query import COEFFICIENT_TOLERANCE, QKV_TOLERANCES

    rng = np.random.default_rng(400 + degree)
    time, heads, width, hidden, count = 2048, 2, 64, 32, degree + 1
    q, k = (rng.normal(size=(1, time, heads, width)) * 0.7 for _ in range(2))
    v = rng.normal(size=(1, time, heads, 64))
    x = rng.normal(size=(1, time, hidden))
    w, u = (rng.normal(size=(heads, hidden, count)) * hidden ** -0.5 for _ in range(2))
    b, a = 1 + 0.2 * rng.normal(size=(heads, count)), 0.5 * rng.normal(size=(heads, count))
    dt_bias, log_rate = rng.normal(size=(heads, count)) - 2, 0.5 * rng.normal(size=(heads, count))
    # Both sides differentiate the same BF16 cotangent: an unrounded reference cotangent would add an
    # independent ~0.2% error to every per-head sum, enough to fail a per-order bound by chance.
    cotangent = jnp.asarray(rng.normal(size=(1, time, heads, 64)), jnp.bfloat16)

    def generator(x, w, b, a, u, dt_bias, log_rate):
        return project_coefficients(x, w, factorial_scale(b), amplitude=a, activation="tanh",
                                    amplitude_scale="factorial", gate_kernel=u, gate_bias=dt_bias,
                                    gate_log_rate=log_rate)

    def production(q, k, v, *parameters):
        return query_polynomial_attention(q, k, v, generator(*parameters))

    def reference(q, k, v, x, w, b, a, u, dt_bias, log_rate):
        factorials = jnp.asarray([math.factorial(r) for r in range(count)], jnp.float64)
        c = (b + a * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, w))) / factorials
        rate = jnp.exp(log_rate) * jax.nn.softplus(jnp.einsum("btd,hdr->bthr", x, u) + dt_bias)
        c = jnp.swapaxes(c * jnp.exp(-rate), 1, 2)
        scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(width)
        polynomial = jnp.broadcast_to(c[..., count - 1, None], scores.shape)
        for order in range(count - 2, -1, -1):
            polynomial = polynomial * scores + c[..., order, None]
        weights = jnp.where(jnp.tril(jnp.ones((time, time), bool)), polynomial, 0.0)
        return jnp.einsum("bhti,bihd->bthd", weights, v)

    operands = tuple(jnp.asarray(value, jnp.bfloat16 if index < 3 else jnp.float32)
                     for index, value in enumerate((q, k, v, x, w, b, a, u, dt_bias, log_rate)))
    actual = jax.tree.map(np.asarray, jax.jit(lambda *values: _vjp(production, values, cotangent))(*operands))
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = jax.tree.map(np.asarray, _vjp(
            reference, tuple(jnp.asarray(value, jnp.float64) for value in operands),
            jnp.asarray(cotangent, jnp.float64)))
    _assert_relative(actual[0], expected[0], rtol=QKV_TOLERANCES[0], name=f"m={degree} output")
    for name, observed, wanted, tolerance in zip("qkv", actual[1][:3], expected[1][:3], QKV_TOLERANCES[1:]):
        _assert_relative(observed, wanted, rtol=tolerance, name=f"m={degree} d{name}")
    _relative(actual[1][3], expected[1][3], COEFFICIENT_TOLERANCE, f"m={degree} dx")
    for name, observed, wanted in zip(("dW", "db", "da", "dw", "ddt_bias", "dA_log"), actual[1][4:], expected[1][4:]):
        _per_order(observed, wanted, COEFFICIENT_TOLERANCE, f"m={degree} {name}")


@pytest.mark.gpu
@pytest.mark.parametrize("degree", GPU_DEGREES)
def test_gdn_gated_mosaic_model_matches_float32_blocked_model(degree, monkeypatch):
    """End-to-end GDN-gated model on Mosaic in BF16 against the FP32 blocked model: loss and gradient
    direction, as for the sigmoid gate, with every gate array's gradient required to be finite and nonzero."""
    _requires_mosaic_gpu()
    from scaffolds.transformer.attention.backends.mosaic import api

    rng = np.random.default_rng(67 + degree)
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    # Perturbed per order around the shared start, so the orders' multipliers differ.
    perturbations = {"query_polynomial_amplitude": (0.5, 0.0), KERNEL: (0.05, 0.0), LOG_RATE: (0.5, 0.0),
                     DT_BIAS: (0.5, -2.0)}
    results = {}
    for backend, dtype, precision in (("blocked", "float32", None), ("blocked", "bfloat16", None),
                                      ("mosaic", "bfloat16", "bf16"), ("mosaic", "bfloat16", "bf16x3")):
        monkeypatch.setattr(api, "default_precision", lambda family, degree, precision=precision: precision)
        config = _config(degree, vocab_size=64, max_seq_len=256, hidden_size=128, qk_head_dim=64, value_head_dim=64,
                         intermediate_size=256, attention_backend=backend,
                         attention_coefficient_projection_initial_std=128 ** -0.5)
        config.train.dtype, config.data.sequence_length, config.data.synthetic_vocab_size = dtype, batch.shape[1], 64
        state = create_state(config, jax.random.PRNGKey(3))
        perturbation = np.random.default_rng(7)
        for name, (scale, offset) in perturbations.items():
            state.params[name][...] = jnp.asarray(
                perturbation.normal(size=state.params[name].shape) * scale + offset, jnp.float32)
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            loss, gradients = jax.jit(lambda *values: loop.loss_and_grads(*values))(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype, precision] = float(loss), _flat(gradients)
    reference_loss, reference = results["blocked", "float32", None]
    bf16 = results["blocked", "bfloat16", None][1]
    loss, actual = results["mosaic", "bfloat16", "bf16"]
    assert math.isfinite(loss) and abs(loss - reference_loss) < 0.02
    assert all(np.isfinite(value).all() for value in actual.values())
    for name in GDN_GATE_PARAMETERS:
        assert np.linalg.norm(actual[(name,)]) > 0, name

    def cosine(gradients):
        a, b = (np.concatenate([g[path].astype(np.float64).ravel() for path in sorted(reference, key=str)])
                for g in (gradients, reference))
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    exact = results["mosaic", "bfloat16", "bf16x3"][1]
    assert cosine(exact) >= min(0.995, cosine(bf16) - 0.01), (cosine(exact), cosine(bf16))
    assert cosine(actual) >= min(0.995, cosine(bf16) - 0.03), (cosine(actual), cosine(bf16))
