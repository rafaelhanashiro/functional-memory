"""Tanh-conditioned polynomials: independent derivatives, initialization, resume, CUDA."""

from dataclasses import replace
from functools import partial
import json
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.backends import fused_query_polynomial as fused
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.input_dependent import project_coefficients, query_polynomial_attention
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import attention_metrics


DEGREES = (1, 2, 3, 4)


def _model(degree=3, **updates):
    settings = dict(vocab_size=16, max_seq_len=13, num_layers=2, hidden_size=8, num_heads=2,
                    num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
                    attention_degree=degree, attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                    attention_coefficient_activation="tanh", attention_backend="blocked",
                    initialization="transformer_reference", qk_norm=True,
                    attention_output_scaling="head_rms_norm")
    return ModelConfig(**{**settings, **updates})


def _config(degree=3):
    config = Config(model=_model(degree))
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 14, 2
    config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 3, 0
    return config


def _inputs(degree, *, time=17, dtype=jnp.float32, qk_dim=7, value_dim=5, batch=2, heads=3):
    random = np.random.default_rng(237 + degree)
    q, k = (jnp.asarray(random.normal(size=(batch, time, heads, qk_dim)), dtype) for _ in range(2))
    v = jnp.asarray(random.normal(size=(batch, time, heads, value_dim)), dtype)
    x = jnp.asarray(random.normal(size=(batch, time, 6)), dtype)
    kernel = jnp.asarray(random.normal(size=(heads, 6, degree + 1)) * 0.2, jnp.float32)
    bias = jnp.asarray(random.normal(size=(heads, degree + 1)) * 0.5, jnp.float32)
    amplitude = jnp.asarray(random.normal(size=(heads, degree + 1)) * 0.7, jnp.float32)
    cotangent = jnp.asarray(random.normal(size=v.shape), dtype)
    return (q, k, v, x, kernel, bias, amplitude), cotangent


def _dense(q, k, v, x, kernel, bias, amplitude):
    """Independent expanded-power oracle: no production projection or polynomial helpers."""
    coefficients = bias + amplitude * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, kernel))
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    weights = sum(jnp.swapaxes(coefficients[..., order], 1, 2)[..., None] * scores ** order
                  for order in range(coefficients.shape[-1]))
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    return jnp.einsum("bhts,bshd->bthd", jnp.where(causal, weights, 0.0), v)


def _operation(q, k, v, x, kernel, bias, amplitude, *, backend="blocked"):
    coefficients = project_coefficients(x, kernel, bias, amplitude=amplitude, activation="tanh")
    return query_polynomial_attention(q, k, v, coefficients, backend=backend)


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return output, pullback(cotangent)


def _oracle(operands, cotangent):
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _vjp(_dense, tuple(x.astype(jnp.float64) for x in operands), cotangent.astype(jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, np.float64), result)


def _assert_close(actual, expected, tolerance, label):
    actual, expected = (np.asarray(x, np.float64) for x in (actual, expected))
    assert actual.shape == expected.shape, label
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    scale = np.linalg.norm(expected)
    if scale == 0:
        np.testing.assert_array_equal(actual, expected, err_msg=label)
    else:
        error = np.linalg.norm(actual - expected) / scale
        assert error < tolerance, f"{label}: relative L2 {error} >= {tolerance}"
        assert np.max(np.abs(actual - expected)) <= 4 * tolerance * np.max(np.abs(expected)), label


def _compare(actual, expected, tolerance=1e-4):
    _assert_close(actual[0], expected[0], tolerance, "output")
    for index, (observed, reference) in enumerate(zip(actual[1], expected[1], strict=True)):
        _assert_close(observed, reference, tolerance, f"gradient {index}")
        # Every head and coefficient order must train; large lower orders must
        # not conceal a wrong reduction or missing higher-order derivative.
        if index >= 4:
            for head in range(reference.shape[0]):
                for order in range(reference.shape[-1]):
                    _assert_close(observed[head, ..., order], reference[head, ..., order],
                                  tolerance, f"gradient {index}, head {head}, order {order}")


def _interpret(monkeypatch):
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST))

    def load(ref, rows, width, time):
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)

    monkeypatch.setattr(fused, "_load", load)
    compiled_entry = fused.fused_query_polynomial_attention
    monkeypatch.setattr(fused, "fused_query_polynomial_attention",
                        lambda q, k, v, c: compiled_entry(q, k, v, c, interpret=True))


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_forward_and_all_seven_gradients_match_independent_fp64(monkeypatch, degree, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    operands, cotangent = _inputs(degree)
    with jax.default_matmul_precision("highest"):
        actual = _vjp(partial(_operation, backend=backend), operands, cotangent)
    _compare(actual, _oracle(operands, cotangent))
    for gradient in actual[1][4:]:
        assert gradient.dtype == jnp.float32
        assert np.linalg.norm(gradient) > 0


def test_interpreted_fused_backward_crosses_query_and_key_tile_boundaries(monkeypatch):
    _interpret(monkeypatch)
    operands, cotangent = _inputs(3, time=67, batch=1, heads=2, qk_dim=17, value_dim=9)
    with jax.default_matmul_precision("highest"):
        actual = _vjp(partial(_operation, backend="fused"), operands, cotangent)
    _compare(actual, _oracle(operands, cotangent))


@pytest.mark.parametrize("degree", DEGREES)
def test_blocked_horner_crosses_padding_boundaries(degree):
    operands, cotangent = _inputs(degree, time=13)
    q, k, v, x, kernel, bias, amplitude = operands
    c = project_coefficients(x, kernel, bias, amplitude=amplitude, activation="tanh")
    actual, diagnostics = blocked_dot_product_attention(
        q, k, v, attention_degree=degree, attention_normalization=False, query_block_size=4,
        query_polynomial_coefficients=c, return_diagnostics=True)
    _assert_close(actual, _oracle(operands, cotangent)[0], 1e-4, "padded reference output")
    assert int(diagnostics["row_count"]) == 2 * 13 * 3
    assert int(diagnostics["causal_entry_count"]) == 2 * 3 * 13 * 14 // 2


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_query_conditioned_outputs_and_all_gradients_are_causal(monkeypatch, degree, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    operands, cotangent = _inputs(degree, time=7, batch=1, heads=2)
    prefix = 3
    cotangent = cotangent.at[:, prefix:].set(0)
    operation = partial(_operation, backend=backend)
    actual = _vjp(operation, operands, cotangent)
    changed = tuple(x.at[:, prefix:].add(2) if index < 4 else x for index, x in enumerate(operands))
    altered = _vjp(operation, changed, cotangent)
    np.testing.assert_allclose(actual[0][:, :prefix], altered[0][:, :prefix], rtol=1e-5, atol=2e-5)
    for index, (left, right) in enumerate(zip(actual[1], altered[1], strict=True)):
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=2e-5)
        if index < 4:
            np.testing.assert_array_equal(left[:, prefix:], 0)


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_nonfinite_weight_cannot_hide_behind_zero_values(monkeypatch, degree, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    q = jnp.full((1, 3, 1, 1), 10., jnp.float32)
    c = jnp.ones((1, 3, 1, degree + 1), jnp.float32).at[:, 1, :, -1].set(1e38)
    output = query_polynomial_attention(q, q, jnp.zeros_like(q), c, backend=backend)
    assert bool(jnp.isfinite(output[:, 0]).all())
    assert bool(jnp.isnan(output[:, 1]).all())
    assert bool(jnp.isfinite(output[:, 2]).all())


@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_future_nonfinite_scores_and_coefficients_do_not_poison_prefix(monkeypatch, backend):
    if backend == "fused":
        _interpret(monkeypatch)
    q = jnp.ones((1, 3, 1, 1), jnp.float32)
    c = jnp.ones((1, 3, 1, 4), jnp.float32).at[:, 2].set(jnp.nan)
    k = q.at[:, 2].set(jnp.inf)
    output = query_polynomial_attention(q, k, q, c, backend=backend)
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [4, 8])
    assert bool(jnp.isnan(output[:, 2]).all())


@pytest.mark.parametrize("degree", DEGREES)
def test_zero_projection_has_taylor_bias_unit_amplitude_and_live_weight_gradient(degree):
    random = np.random.default_rng(81)
    x = jnp.asarray(random.normal(size=(2, 7, 6)), jnp.float32)
    kernel = jnp.zeros((3, 6, degree + 1), jnp.float32)
    taylor = jnp.asarray([1 / math.factorial(r) for r in range(degree + 1)], jnp.float32)
    bias = jnp.broadcast_to(taylor, (3, degree + 1))
    amplitude = jnp.ones_like(bias)
    cotangent = jnp.asarray(random.normal(size=(2, 7, 3, degree + 1)), jnp.float32)
    operation = lambda x, w, b, a: project_coefficients(x, w, b, amplitude=a, activation="tanh")
    values, pullback = jax.vjp(operation, x, kernel, bias, amplitude)
    dx, dw, db, da = pullback(cotangent)
    np.testing.assert_array_equal(values, jnp.broadcast_to(taylor, values.shape))
    np.testing.assert_array_equal(dx, 0)
    np.testing.assert_array_equal(da, 0)
    np.testing.assert_allclose(dw, jnp.einsum("btd,bthr->hdr", x, cotangent), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(db, jnp.sum(cotangent, axis=(0, 1)), rtol=1e-6, atol=1e-6)
    assert np.linalg.norm(dw) > 0


@pytest.mark.parametrize("degree", DEGREES)
def test_canonical_backbone_initial_function_and_parameter_count_match_static(degree):
    dynamic_config = _model(degree)
    static_config = replace(dynamic_config, attention_coefficient_conditioning="static",
                            attention_coefficient_activation="identity")
    dynamic, static = build_model(dynamic_config, 13), build_model(static_config, 13)
    dynamic_params = dict(nnx.to_flat_state(nnx.state(dynamic, nnx.Param)))
    static_params = dict(nnx.to_flat_state(nnx.state(static, nnx.Param)))
    for path, parameter in static_params.items():
        if path != ("polynomial_coefficients",):
            np.testing.assert_array_equal(parameter[...], dynamic_params[path][...])
    np.testing.assert_array_equal(dynamic.query_polynomial_kernel[...], 0)
    np.testing.assert_array_equal(dynamic.query_polynomial_amplitude[...], 1)
    np.testing.assert_array_equal(dynamic.query_polynomial_bias[...], static.polynomial_coefficients[...])
    assert dynamic.query_polynomial_kernel.shape == (2, 2, 8, degree + 1)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(dynamic_params))
    assert model_parameter_count(dynamic_config) - model_parameter_count(static_config) == 2 * 2 * 9 * (degree + 1)
    tokens = jnp.arange(26).reshape(2, 13) % 16
    np.testing.assert_allclose(dynamic(tokens), static(tokens), rtol=2e-5, atol=2e-6)
    # The static submodel includes arbitrary signed coefficients, not just Taylor.
    coefficients = jnp.asarray(np.random.default_rng(9).normal(size=(2, 2, degree + 1)), jnp.float32)
    dynamic.query_polynomial_bias[...] = coefficients
    static.polynomial_coefficients[...] = coefficients
    np.testing.assert_allclose(dynamic(tokens), static(tokens), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("degree", DEGREES)
def test_zero_amplitude_with_random_projection_starts_static_and_learns_through_the_amplitude(degree):
    spread = 0.002
    config = _model(degree, attention_coefficient_amplitude_initial=0.0,
                    attention_coefficient_projection_initial_std=spread)
    static_config = replace(config, attention_coefficient_conditioning="static",
                            attention_coefficient_activation="identity",
                            attention_coefficient_amplitude_initial=1.0,
                            attention_coefficient_projection_initial_std=0.0)
    dynamic, again, static = build_model(config, 13), build_model(config, 13), build_model(static_config, 13)
    dynamic_params = dict(nnx.to_flat_state(nnx.state(dynamic, nnx.Param)))
    for path, parameter in dict(nnx.to_flat_state(nnx.state(static, nnx.Param))).items():
        if path != ("polynomial_coefficients",):
            np.testing.assert_array_equal(parameter[...], dynamic_params[path][...])
    kernel = np.asarray(dynamic.query_polynomial_kernel[...])
    np.testing.assert_array_equal(kernel, again.query_polynomial_kernel[...])  # a function of the seed
    assert not np.array_equal(kernel, build_model(config, 14).query_polynomial_kernel[...])
    assert 0.5 * spread < kernel.std() < 1.5 * spread and len(np.unique(kernel)) == kernel.size
    np.testing.assert_array_equal(dynamic.query_polynomial_amplitude[...], 0)
    tokens = jnp.arange(26).reshape(2, 13) % 16
    np.testing.assert_allclose(dynamic(tokens), static(tokens), rtol=2e-5, atol=2e-6)
    # a = 0 silences W's gradient; the random W gives the amplitude a nonzero one.
    random = np.random.default_rng(5)
    x = jnp.asarray(random.normal(size=(2, 7, 8)), jnp.float32)
    cotangent = jnp.asarray(random.normal(size=(2, 7, 2, degree + 1)), jnp.float32)
    operation = lambda w, a: project_coefficients(x, w, dynamic.query_polynomial_bias[0], amplitude=a,
                                                  activation="tanh")
    _, pullback = jax.vjp(operation, dynamic.query_polynomial_kernel[0], dynamic.query_polynomial_amplitude[0])
    dw, da = pullback(cotangent)
    np.testing.assert_array_equal(dw, 0)
    assert np.all(np.abs(np.asarray(da)) > 0)
    facts = acceptance.matches_softmax_backbone_initialization(Config(model=config))
    assert facts["mechanism_parameters"]["query_polynomial_kernel"]["initial_std"] == spread
    assert facts["mechanism_parameters"]["query_polynomial_amplitude"]["initial"] == [0.0] * (degree + 1)


def test_new_initialization_fields_keep_legacy_recipes_and_identity_unchanged():
    config = _config(2)
    current, legacy = config_to_dict(config), config_to_dict(config)
    for key in ("attention_coefficient_amplitude_initial", "attention_coefficient_projection_initial_std"):
        del legacy["model"][key]
    assert config_dicts_equal(current, legacy)
    changed = _config(2)
    changed.model = replace(changed.model, attention_coefficient_amplitude_initial=0.0,
                            attention_coefficient_projection_initial_std=0.002)
    assert not config_dicts_equal(current, config_to_dict(changed))
    from scaffolds.transformer.training.diagnostics import attention_identity

    default = attention_identity(config)
    assert "attention_coefficient_amplitude_initial" not in default
    assert "W initially zero, a initially 1, b_r initially 1/r!" in default["attention_formula"]
    started = attention_identity(changed)
    assert started["attention_coefficient_amplitude_initial"] == 0.0
    assert started["attention_coefficient_projection_initial_std"] == 0.002
    assert "W initially N(0, 0.002^2) per entry, a initially 0" in started["attention_formula"]


def test_all_projection_parameters_keep_fp32_and_are_excluded_from_weight_decay():
    config = _config()
    state = create_state(config, jax.random.PRNGKey(0))
    for name in ("query_polynomial_kernel", "query_polynomial_bias", "query_polynomial_amplitude"):
        state.params[name][...] += 0.125
    after = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    for name in ("query_polynomial_kernel", "query_polynomial_bias", "query_polynomial_amplitude"):
        assert state.params[name].dtype == jnp.float32
        np.testing.assert_array_equal(state.params[name][...], after.params[name][...])
    values = project_coefficients(jnp.ones((1, 13, 8), jnp.bfloat16), state.params["query_polynomial_kernel"][0],
                                  state.params["query_polynomial_bias"][0],
                                  amplitude=state.params["query_polynomial_amplitude"][0], activation="tanh")
    assert values.dtype == jnp.float32


@pytest.mark.parametrize("degree", DEGREES)
def test_mandatory_acceptance_checks_include_new_backend_and_canonical_initialization(degree):
    config = _config(degree)
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (False, True):
        paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=cpu)
        assert acceptance.QUERY_TANH_POLYNOMIAL_TESTS in paths
        assert len(paths) == len(set(paths))
    facts = acceptance.matches_softmax_backbone_initialization(config)
    assert facts["mechanism_parameters"]["query_polynomial_amplitude"]["shape"] == [2, 2, degree + 1]
    assert facts["mechanism_parameters"]["query_polynomial_bias"]["initial"] == [
        1 / math.factorial(r) for r in range(degree + 1)]


def test_legacy_identity_config_resumes_without_new_activation_field():
    config = _config(1)
    config.model.attention_coefficient_activation = "identity"
    current, legacy = config_to_dict(config), config_to_dict(config)
    del legacy["model"]["attention_coefficient_activation"]
    assert config_dicts_equal(current, legacy)
    assert not config_dicts_equal(current, config_to_dict(_config(1)))


@pytest.mark.parametrize("updates", [
    {"attention_normalization": True}, {"attention_degree": 0},
    {"attention_coefficients": "fixed"}, {"attention_coefficient_sharing": "global"},
    {"attention_coefficient_conditioning": "static"}, {"attention_coefficient_activation": "relu"},
    {"attention_coefficient_initialization": "linear"},
    {"attention_coefficient_amplitude_initial": 0.0},
    {"attention_coefficient_projection_initial_std": -0.1},
    {"attention_coefficient_amplitude_initial": float("nan")},
    {"attention_coefficient_activation": "identity", "attention_degree": 1,
     "attention_coefficient_projection_initial_std": 0.01},
])
def test_rejects_unsupported_tanh_coefficient_configurations(updates):
    with pytest.raises(ValueError):
        _model(**updates)


@pytest.mark.parametrize("degree", DEGREES)
def test_diagnostics_use_actual_tanh_coefficients_not_baselines(degree):
    config = _config(degree)
    state = create_state(config, jax.random.PRNGKey(0))
    shape = state.params["query_polynomial_kernel"].shape
    state.params["query_polynomial_kernel"][...] = jnp.asarray(
        np.random.default_rng(2).normal(size=shape) * 0.2, jnp.float32)
    state.params["query_polynomial_amplitude"][...] *= -0.75
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["attention_coefficient_activation"] == "tanh"
    assert report["attention_coefficient_count"] == 2 * 2 * (8 + 2) * (degree + 1)
    coefficients = report["coefficients"]
    assert np.asarray(coefficients["effective_summary"]["mean"]).shape == (2, 2, degree + 1)
    assert np.min(coefficients["effective_summary"]["std"]) > 0
    model = nnx.merge(state.graphdef, state.params)
    x = model.layers_0.attention_norm(model.token_embedding(jnp.asarray(sample[:, :-1])))
    expected = project_coefficients(x, model.query_polynomial_kernel[0], model.query_polynomial_bias[0],
                                    amplitude=model.query_polynomial_amplitude[0], activation="tanh")
    np.testing.assert_allclose(coefficients["effective_summary"]["mean"][0], jnp.mean(expected, axis=(0, 1)),
                               rtol=1e-5, atol=1e-6)
    metrics = attention_metrics(report)
    assert any(name.endswith("/constant/std") and value > 0 for name, value in metrics.items())


def test_tiny_learning_and_checkpoint_resume_preserve_amplitudes_optimizer_and_rng(tmp_path):
    config = _config()
    config.train.workdir = tmp_path / "full"
    config.train.wandb_mode = "disabled"
    config.train.eval_every, config.train.eval_batches, config.train.ckpt_every = 1, 1, 1
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-3, 1e-4
    full = loop.run_training(config, audit_checkpoint=True)
    restarted = replace(config, train=replace(config.train, workdir=tmp_path / "restarted"))
    before = loop.run_training(restarted, stop_after_step=1, audit_checkpoint=True)
    after = loop.run_training(restarted, audit_checkpoint=True)
    a, b = before["checkpoint_audit"], after["checkpoint_audit"]
    assert a["final_state_sha256"] == a["checkpoint_state_sha256"] == b["start_state_sha256"]
    assert a["next_update"] == b["first_update"]
    assert b["final_state_sha256"] == b["checkpoint_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    assert after["restored_step"] == 1
    manager = make_checkpoint_manager(restarted)
    try:
        initial = create_state(restarted, jax.random.PRNGKey(0))
        state, step = restore_latest_checkpoint(manager, initial)
        assert step == config.optim.total_steps
        for name in ("query_polynomial_kernel", "query_polynomial_bias", "query_polynomial_amplitude"):
            assert not np.array_equal(state.params[name][...], initial.params[name][...]), name
    finally:
        manager.close()
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert records and all(row["attention_diagnostics"]["valid"] for row in records)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_gpu_forward_and_all_gradients_match_independent_fp64(degree, dtype):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    operands, cotangent = _inputs(degree, time=131, dtype=dtype, qk_dim=33, value_dim=17)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(partial(_vjp, partial(_operation, backend="fused")))(operands, cotangent)
    _compare(actual, _oracle(operands, cotangent), tolerance=0.025 if dtype == jnp.bfloat16 else 2e-4)
    assert actual[0].dtype == dtype
    assert all(gradient.dtype == jnp.float32 for gradient in actual[1][4:])


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_bf16_production_context_matches_blocked_forward_and_backward(degree):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    operands, cotangent = _inputs(degree, time=2048, dtype=jnp.bfloat16, qk_dim=64, value_dim=64,
                                 batch=1, heads=4)
    actual = jax.jit(partial(_vjp, partial(_operation, backend="fused")))(operands, cotangent)
    with jax.default_matmul_precision("highest"):
        reference = jax.jit(partial(_vjp, _operation))(operands, cotangent)
    _compare(actual, reference, tolerance=0.025)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gpu_actual_kernel_overflow_and_nonfinite_future_key(degree):
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation")
    attend = jax.jit(partial(query_polynomial_attention, backend="fused"))
    q = jnp.full((1, 3, 1, 1), 10., jnp.float32)
    c = jnp.ones((1, 3, 1, degree + 1), jnp.float32).at[:, 1, :, -1].set(1e38)
    output = attend(q, q, jnp.zeros_like(q), c)
    assert bool(jnp.isfinite(output[:, 0]).all()) and bool(jnp.isnan(output[:, 1]).all())
    assert bool(jnp.isfinite(output[:, 2]).all())
    q = jnp.ones_like(q)
    c = jnp.ones_like(c)
    output = attend(q, q.at[:, 2].set(jnp.inf), q, c)
    np.testing.assert_array_equal(output[:, :2].reshape(-1), [degree + 1, 2 * (degree + 1)])
    assert bool(jnp.isnan(output[:, 2]).all())
