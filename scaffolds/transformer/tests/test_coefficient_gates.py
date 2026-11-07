"""Per-order sigmoid coefficient gates: exact ungated start, decay policy, gradients, diagnostics, Mosaic."""

from dataclasses import replace
import json
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.input_dependent import coefficient_gate_statistics, project_coefficients
from scaffolds.transformer.attention.parameters import attention_parameter_specs, factorial_scale
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import attention_weight_decay_mask, make_optimizer
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import attention_metrics, run_tags


DEGREES = (8, 10)
GATE_NAMES = ("query_polynomial_gate_kernel", "query_polynomial_gate_bias")


def _model(degree=8, **changes):
    """The production generator (r25fffc16c2f3's coefficient settings) on a small backbone."""
    settings = dict(vocab_size=32, max_seq_len=16, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                    qk_head_dim=8, value_head_dim=8, intermediate_size=32, attention_backend="blocked",
                    attention_degree=degree, attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                    attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.0,
                    attention_coefficient_projection_initial_std=16 ** -0.5,
                    attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                    attention_coefficient_kernel_weight_decay=True, attention_coefficient_gate="sigmoid",
                    attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                    qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                    initialization="transformer_truncated_reference")
    return ModelConfig(**{**settings, **changes})


def _config(degree=8, **changes):
    config = Config(model=_model(degree, **changes))
    config.data.synthetic_vocab_size, config.data.sequence_length = 32, 17
    config.data.batch_size = config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 3, 0
    config.optim.learning_rate = config.optim.min_learning_rate = 1e-3
    config.optim.weight_decay = 0.1
    return config


def _ungated(config):
    return replace(config, model=replace(config.model, attention_coefficient_gate="none"))


def _params(model):
    return {path: value[...] for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _flat(tree):
    return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(tree)}


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _relative(observed, expected, tolerance, label):
    observed, expected = np.asarray(observed, np.float64), np.asarray(expected, np.float64)
    assert observed.shape == expected.shape and np.isfinite(observed).all(), label
    error = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    assert error <= tolerance, f"{label}: relative L2 error {error:.3g} exceeds {tolerance}"


def _per_order(observed, expected, tolerance, label):
    """Every order must train: compare each order's slice relative to its own norm."""
    for order in range(np.shape(expected)[-1]):
        _relative(np.asarray(observed)[..., order], np.asarray(expected)[..., order], tolerance,
                  f"{label} order {order}")


def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


@pytest.mark.parametrize("degree", DEGREES)
def test_zero_gates_reproduce_the_ungated_model_and_its_gradients_exactly(degree):
    config = _config(degree)
    control = _ungated(config)
    gated, ungated = build_model(config.model, 42), build_model(control.model, 42)
    actual, baseline = _params(gated), _params(ungated)
    assert set(actual) - set(baseline) == {(name,) for name in GATE_NAMES}
    for path, value in baseline.items():
        np.testing.assert_array_equal(actual[path], value, err_msg=str(path))
    np.testing.assert_array_equal(actual[("query_polynomial_gate_kernel",)], np.zeros((2, 2, 16, degree + 1)))
    np.testing.assert_array_equal(actual[("query_polynomial_gate_bias",)], np.zeros((2, 2, degree + 1)))
    assert all(value.dtype == jnp.float32 for path, value in actual.items() if path[0] in GATE_NAMES)
    assert model_parameter_count(config.model) - model_parameter_count(control.model) == 2 * 2 * 17 * (degree + 1)
    tokens = jnp.arange(32).reshape(2, 16) % 32
    np.testing.assert_array_equal(gated(tokens), ungated(tokens))
    # Raw (unclipped) gradients of every shared parameter equal the ungated model's.
    batch = jnp.asarray(np.random.default_rng(3).integers(0, 32, size=(2, 17)), jnp.int32)
    results = [loop.loss_and_grads(create_state(recipe, jax.random.PRNGKey(42)), batch, jax.random.PRNGKey(0))
               for recipe in (config, control)]
    (gated_loss, gated_gradients), (loss, gradients) = results
    assert float(gated_loss) == float(loss)
    gated_gradients, gradients = _flat(gated_gradients), _flat(gradients)
    for path, value in gradients.items():
        np.testing.assert_array_equal(gated_gradients[path], value, err_msg=str(path))
    # The gates learn at once: finite and nonzero for every layer, head and order.
    for name in GATE_NAMES:
        gradient = gated_gradients[(name,)]
        magnitude = np.abs(gradient) if name == "query_polynomial_gate_bias" else np.linalg.norm(gradient, axis=-2)
        assert np.isfinite(gradient).all() and np.all(magnitude > 0), name
    # With a = 0: dL/dbeta_r = (c_r / 2) dL/dc~_r and dL/db_r = (1 / r!) dL/dc~_r with c_r = 1/r!; the
    # two FP32 token sums are associated differently, so they agree to rounding rather than bitwise.
    np.testing.assert_allclose(gated_gradients[("query_polynomial_gate_bias",)],
                               0.5 * gated_gradients[("query_polynomial_bias",)], rtol=1e-4, atol=0)


def test_weight_decay_covers_both_projections_and_spares_a_b_and_the_gate_bias():
    config = _config()
    model = build_model(config.model, 42)
    names = ("query_polynomial_kernel", "query_polynomial_gate_kernel", "query_polynomial_bias",
             "query_polynomial_amplitude", "query_polynomial_gate_bias")
    for name in names:  # nonzero, so that a zero-gradient update reveals any decay
        getattr(model, name)[...] = jnp.full_like(getattr(model, name)[...], 0.25)
    params = nnx.state(model, nnx.Param)

    def mask(recipe, parameters):
        excluded = frozenset(spec.name for spec in attention_parameter_specs(recipe) if not spec.weight_decay)
        return {path: bool(value.get_value()) for path, value in
                nnx.to_flat_state(attention_weight_decay_mask(parameters, excluded_names=excluded))}

    gated = mask(config.model, params)
    assert gated[("query_polynomial_kernel",)] and gated[("query_polynomial_gate_kernel",)]
    assert not any(gated[(name,)] for name in names[2:])
    control = _ungated(config).model
    assert {path: decay for path, decay in gated.items() if path[0] not in GATE_NAMES} == mask(
        control, nnx.state(build_model(control, 42), nnx.Param))
    optimizer = make_optimizer(config)
    updates, _ = optimizer.update(jax.tree.map(jnp.zeros_like, params), optimizer.init(params), params)
    values = _flat(updates)
    for name in names[:2]:
        np.testing.assert_allclose(values[(name,)], -1e-3 * 0.1 * 0.25, rtol=2e-6, atol=1e-10)
    for name in names[2:]:
        np.testing.assert_array_equal(values[(name,)], 0)


@pytest.mark.parametrize("degree", DEGREES)
def test_gated_generator_matches_an_independent_float64_formula_and_gradients(degree):
    rng = np.random.default_rng(11 + degree)
    heads, width, count = 2, 6, degree + 1
    factorials = np.asarray([math.factorial(r) for r in range(count)], np.float64)
    x = rng.normal(size=(2, 5, width))
    w, u = (rng.normal(size=(heads, width, count)) * 0.4 for _ in range(2))
    b, a, beta = (1 + 0.3 * rng.normal(size=(heads, count)), 0.7 * rng.normal(size=(heads, count)),
                  rng.normal(size=(heads, count)))
    cotangent = rng.normal(size=(2, 5, heads, count)) * factorials  # comparable cotangents for every order

    def production(x, w, b, a, u, beta):
        return project_coefficients(x, w, factorial_scale(b), amplitude=a, activation="tanh",
                                    amplitude_scale="factorial", gate_kernel=u, gate_bias=beta)

    def oracle(x, w, b, a, u, beta):
        c = (b + a * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, w))) / factorials
        return 2 * c / (1 + jnp.exp(-(jnp.einsum("btd,hdr->bthr", x, u) + beta)))

    operands = tuple(jnp.asarray(value, jnp.float32) for value in (x, w, b, a, u, beta))
    actual = jax.tree.map(np.asarray, _vjp(production, operands, jnp.asarray(cotangent, jnp.float32)))
    with jax.enable_x64():
        expected = jax.tree.map(np.asarray, _vjp(
            oracle, tuple(jnp.asarray(value, jnp.float64) for value in operands), jnp.asarray(cotangent)))
    assert actual[0].dtype == np.float32 and all(gradient.dtype == np.float32 for gradient in actual[1])
    _per_order(actual[0], expected[0], 2e-6, "coefficients")
    _relative(actual[1][0], expected[1][0], 2e-5, "dx")
    for name, observed, wanted in zip(("dW", "db", "da", "dU", "dbeta"), actual[1][1:], expected[1][1:]):
        _per_order(observed, wanted, 2e-5, name)


def test_gate_statistics_match_a_dense_numpy_oracle():
    rng = np.random.default_rng(8)
    batch, time, heads, width, count = 2, 7, 3, 4, 5
    logits, ungated = rng.normal(size=(batch, time, heads, count)) * 2, rng.normal(size=(batch, time, heads, count))
    gated = 2 / (1 + np.exp(-logits)) * ungated
    q, k = (rng.normal(size=(batch, time, heads, width)) for _ in range(2))
    inputs = tuple(np.asarray(value, np.float32) for value in (logits, ungated, gated, q, k))
    statistics = jax.device_get(coefficient_gate_statistics(*(jnp.asarray(value) for value in inputs)))
    logits, ungated, gated, q, k = (value.astype(np.float64) for value in inputs)
    gates = 1 / (1 + np.exp(-logits))
    log_gates = -np.logaddexp(0, -logits)
    tilt = log_gates - log_gates.mean(axis=-1, keepdims=True)
    expected = {"mean": gates.mean(axis=(0, 1)), "std": gates.std(axis=(0, 1)), "min": gates.min(axis=(0, 1)),
                "max": gates.max(axis=(0, 1)), "log_tilt_mean": tilt.mean(axis=(0, 1)),
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
def test_diagnostics_report_gates_ungated_coefficients_and_contributions(degree):
    config = _config(degree)
    state = create_state(config, jax.random.PRNGKey(0))
    rng = np.random.default_rng(4)
    for name, scale in (("query_polynomial_gate_kernel", 0.3), ("query_polynomial_gate_bias", 1.0),
                        ("query_polynomial_amplitude", 0.5)):
        state.params[name][...] = jnp.asarray(rng.normal(size=state.params[name].shape) * scale, jnp.float32)
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["attention_coefficient_gate"] == "sigmoid"
    assert report["attention_coefficient_count"] == 2 * 2 * (degree + 1) * (2 * 16 + 3)
    assert "2 g_r(x_t) c_r(x_t) s_ti^r" in report["attention_formula"]
    assert "g_r(x_t) = sigmoid(x_t U_r + beta_r)" in report["attention_formula"]
    gates = report["coefficients"]["coefficient_gates"]
    assert gates["finite"] and gates["order_names"][-1] == f"degree_{degree}"
    assert np.asarray(gates["gate_summary"]["mean"]).shape == (2, 2, degree + 1)
    model = nnx.merge(state.graphdef, state.params)
    x = np.asarray(model.layers_0.attention_norm(model.token_embedding(jnp.asarray(sample[:, :-1]))), np.float64)
    gate = 1 / (1 + np.exp(-(np.einsum("btd,hdr->bthr", x, np.asarray(model.query_polynomial_gate_kernel[0]))
                             + np.asarray(model.query_polynomial_gate_bias[0]))))
    ungated = np.asarray(project_coefficients(
        jnp.asarray(x, jnp.float32), model.query_polynomial_kernel[0],
        factorial_scale(model.query_polynomial_bias[0]), amplitude=model.query_polynomial_amplitude[0],
        activation="tanh", amplitude_scale="factorial"), np.float64)
    np.testing.assert_allclose(gates["gate_summary"]["mean"][0], gate.mean(axis=(0, 1)), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gates["ungated_coefficient_summary"]["mean"][0], ungated.mean(axis=(0, 1)),
                               rtol=1e-5, atol=1e-9)
    gated = 2 * gate * ungated  # what the kernel receives, summarized as the effective coefficients
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
    assert f"attention/query_polynomial_unnormalized/coefficient_gates/degree_{degree}/gate_mean" in metrics
    assert "coefficient_gate_sigmoid" in run_tags(config, {})


def test_ungated_recipes_keep_their_identity_and_legacy_recipes_load_unchanged():
    gated, ungated = _config(), _ungated(_config())
    identity = attention_identity(ungated)
    assert "attention_coefficient_gate" not in identity and "g_r(x_t)" not in identity["attention_formula"]
    assert identity["attention_coefficient_count"] == 2 * 2 * 9 * (16 + 2)
    assert attention_identity(gated)["attention_coefficient_count"] == 2 * 2 * 9 * (2 * 16 + 3)
    legacy = config_to_dict(ungated)
    del legacy["model"]["attention_coefficient_gate"]
    assert config_dicts_equal(config_to_dict(ungated), legacy)
    assert not config_dicts_equal(config_to_dict(gated), legacy)
    assert "coefficient_gate_sigmoid" not in run_tags(ungated, {})


@pytest.mark.parametrize("changes", [
    dict(attention_coefficient_gate="softmax"), dict(attention_coefficient_gate=False),
    dict(attention_coefficient_activation="affine_sigmoid"), dict(attention_coefficient_readout="dense"),
    dict(attention_coefficient_orders=(0, 2, 4, 6, 8)), dict(attention_coefficient_tanh_bias=True),
    dict(positional_encoding="none"), dict(attention_forget_gate=True, attention_forget_gate_mode="query_distance"),
    dict(attention_coefficient_conditioning="query_key"),
])
def test_unqualified_gate_recipes_are_rejected(changes):
    with pytest.raises(ValueError):
        _model(**changes)


def test_gated_continuations_and_learning_rate_scales_are_rejected():
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


def test_acceptance_selects_the_gate_checks_and_both_qualified_degrees_on_the_gpu():
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    expected = [f"{acceptance.COEFFICIENT_GATE_TESTS}::{test}[{degree}]"
                for test in acceptance.COEFFICIENT_GATE_GPU_TESTS for degree in DEGREES]
    for degree in DEGREES:
        recipe = config_to_dict(_config(degree))
        cpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=True)
        gpu = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=False)
        assert acceptance.COEFFICIENT_GATE_TESTS in cpu
        assert [path for path in gpu if path.startswith(acceptance.COEFFICIENT_GATE_TESTS)] == expected
    ungated = config_to_dict(_ungated(_config()))
    assert not any(path.startswith(acceptance.COEFFICIENT_GATE_TESTS) for cpu in (True, False)
                   for path in acceptance.test_paths(spec, ungated, resolved_backend="mosaic", cpu=cpu))
    facts = acceptance.matches_softmax_backbone_initialization(_config())
    parameters = facts["mechanism_parameters"]
    assert parameters["query_polynomial_gate_kernel"]["weight_decay"] is True
    assert parameters["query_polynomial_gate_bias"]["weight_decay"] is False
    assert set(parameters["query_polynomial_gate_bias"]["initial"]) == {0.0}


def test_tiny_training_moves_the_gates_and_resumes_exactly(tmp_path):
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
        for name in (*GATE_NAMES, "query_polynomial_bias"):
            assert not np.array_equal(state.params[name][...], initial.params[name][...]), name
    finally:
        manager.close()
    records = [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]
    assert records and all(row["attention_diagnostics"]["valid"] for row in records)
    assert all(row["attention_diagnostics"]["coefficients"]["coefficient_gates"]["finite"] for row in records)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gated_coefficients_through_mosaic_match_fp64(degree):
    """Production shapes (2,048 tokens, d_K 64): x -> gated generator -> Mosaic, against FP64, every gradient."""
    _requires_mosaic_gpu()
    from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
    from scaffolds.transformer.tests.test_fused_attention import _assert_relative
    from scaffolds.transformer.tests.test_mosaic_query import COEFFICIENT_TOLERANCE, QKV_TOLERANCES

    rng = np.random.default_rng(300 + degree)
    time, heads, width, hidden, count = 2048, 2, 64, 32, degree + 1
    q, k = (rng.normal(size=(1, time, heads, width)) * 0.7 for _ in range(2))
    v = rng.normal(size=(1, time, heads, 64))
    x = rng.normal(size=(1, time, hidden))
    w, u = (rng.normal(size=(heads, hidden, count)) * hidden ** -0.5 for _ in range(2))
    b, a, beta = (1 + 0.2 * rng.normal(size=(heads, count)), 0.5 * rng.normal(size=(heads, count)),
                  rng.normal(size=(heads, count)))
    # Both sides differentiate the same BF16 cotangent: an unrounded reference cotangent would add an
    # independent ~0.2% error to every per-head sum, enough to fail a per-order bound by chance.
    cotangent = jnp.asarray(rng.normal(size=(1, time, heads, 64)), jnp.bfloat16)

    def generator(x, w, b, a, u, beta):
        return project_coefficients(x, w, factorial_scale(b), amplitude=a, activation="tanh",
                                    amplitude_scale="factorial", gate_kernel=u, gate_bias=beta)

    def production(q, k, v, *parameters):
        return query_polynomial_attention(q, k, v, generator(*parameters))

    def reference(q, k, v, x, w, b, a, u, beta):
        factorials = jnp.asarray([math.factorial(r) for r in range(count)], jnp.float64)
        c = (b + a * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, w))) / factorials
        c = jnp.swapaxes(2 * c * jax.nn.sigmoid(jnp.einsum("btd,hdr->bthr", x, u) + beta), 1, 2)
        scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(width)
        polynomial = jnp.broadcast_to(c[..., count - 1, None], scores.shape)
        for order in range(count - 2, -1, -1):
            polynomial = polynomial * scores + c[..., order, None]
        weights = jnp.where(jnp.tril(jnp.ones((time, time), bool)), polynomial, 0.0)
        return jnp.einsum("bhti,bihd->bthd", weights, v)

    operands = tuple(jnp.asarray(value, jnp.bfloat16 if index < 3 else jnp.float32)
                     for index, value in enumerate((q, k, v, x, w, b, a, u, beta)))
    actual = jax.tree.map(np.asarray, jax.jit(lambda *values: _vjp(production, values, cotangent))(*operands))
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = jax.tree.map(np.asarray, _vjp(
            reference, tuple(jnp.asarray(value, jnp.float64) for value in operands),
            jnp.asarray(cotangent, jnp.float64)))
    _assert_relative(actual[0], expected[0], rtol=QKV_TOLERANCES[0], name=f"m={degree} output")
    for name, observed, wanted, tolerance in zip("qkv", actual[1][:3], expected[1][:3], QKV_TOLERANCES[1:]):
        _assert_relative(observed, wanted, rtol=tolerance, name=f"m={degree} d{name}")
    _relative(actual[1][3], expected[1][3], COEFFICIENT_TOLERANCE, f"m={degree} dx")
    for name, observed, wanted in zip(("dW", "db", "da", "dU", "dbeta"), actual[1][4:], expected[1][4:]):
        _per_order(observed, wanted, COEFFICIENT_TOLERANCE, f"m={degree} {name}")


@pytest.mark.gpu
@pytest.mark.parametrize("degree", DEGREES)
def test_gated_mosaic_model_matches_float32_blocked_model(degree, monkeypatch):
    """End-to-end gated model on Mosaic in BF16 against the FP32 blocked model: loss and gradient direction.
    As for the endpoint gates, single parameters can be cancellation-dominated in BF16, so the gradient is
    compared as a whole, with every gate array's gradient required to be finite and nonzero."""
    _requires_mosaic_gpu()
    from scaffolds.transformer.attention.backends.mosaic import api

    rng = np.random.default_rng(57 + degree)
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    perturbations = {"query_polynomial_amplitude": 0.5, "query_polynomial_gate_kernel": 0.05,
                     "query_polynomial_gate_bias": 0.5}
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
        for name, scale in perturbations.items():
            state.params[name][...] = jnp.asarray(perturbation.normal(size=state.params[name].shape) * scale,
                                                  jnp.float32)
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            loss, gradients = jax.jit(lambda *values: loop.loss_and_grads(*values))(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype, precision] = float(loss), _flat(gradients)
    reference_loss, reference = results["blocked", "float32", None]
    bf16 = results["blocked", "bfloat16", None][1]
    loss, actual = results["mosaic", "bfloat16", "bf16"]
    assert math.isfinite(loss) and abs(loss - reference_loss) < 0.02
    assert all(np.isfinite(value).all() for value in actual.values())
    for name in GATE_NAMES:
        assert np.linalg.norm(actual[(name,)]) > 0, name

    def cosine(gradients):
        a, b = (np.concatenate([g[path].astype(np.float64).ravel() for path in sorted(reference, key=str)])
                for g in (gradients, reference))
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    exact = results["mosaic", "bfloat16", "bf16x3"][1]
    assert cosine(exact) >= min(0.995, cosine(bf16) - 0.01), (cosine(exact), cosine(bf16))
    assert cosine(actual) >= min(0.995, cosine(bf16) - 0.03), (cosine(actual), cosine(bf16))
