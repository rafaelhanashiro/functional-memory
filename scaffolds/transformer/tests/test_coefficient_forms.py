"""Pure tanh and affine sigmoid query coefficients, and the Gated DeltaNet forget gate."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.forget_gate import log_decay, parameter_specs as forget_gate_specs
from scaffolds.transformer.attention.input_dependent import parameter_specs, project_coefficients
from scaffolds.transformer.attention.parameters import (
    attention_parameter_count, attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
)
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_identity, query_coefficient_report
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state

FACTORIALS = np.array([math.factorial(r) for r in range(5)], np.float64)


def _config(form="affine_sigmoid", forget_gate=False, **changes):
    """Query-only m=4 recipes: canonical affine generators, unscaled pure tanh, optional forget gate."""
    pure = form == "pure_tanh"
    model = dict(vocab_size=16, max_seq_len=8, num_layers=1, hidden_size=8,
                 num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
                 intermediate_size=12, attention_degree=4, attention_backend="blocked",
                 attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation=form,
                 attention_coefficient_amplitude_initial=1.0 if pure else 0.0,
                 attention_coefficient_projection_initial_std=8 ** -.5,
                 attention_coefficient_amplitude_scale="none" if pure else "factorial",
                 attention_coefficient_scale="none" if pure else "factorial",
                 attention_forget_gate=forget_gate,
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    return config


def _float64_gradients(reference, arrays, cotangent):
    with jax.enable_x64():
        inputs = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        value = np.asarray(reference(*inputs))
        gradients = jax.grad(lambda *x: jnp.sum(reference(*x) * jnp.asarray(cotangent)),
                             argnums=tuple(range(len(arrays))))(*inputs)
    return value, [np.asarray(g) for g in gradients]


@pytest.mark.parametrize("factorial", (False, True))
def test_pure_tanh_and_derivatives_match_float64(factorial):
    rng = np.random.default_rng(5)
    arrays = (rng.normal(size=(2, 7, 8)), rng.normal(size=(2, 8, 5)) * .4)
    cotangent = rng.normal(size=(2, 7, 2, 5))
    factors = FACTORIALS if factorial else np.ones(5)
    reference = lambda x, w: jnp.tanh(jnp.einsum("btd,hdj->bthj", x, w)) / jnp.asarray(factors)
    actual = lambda x, w: project_coefficients(x, w, None, activation="pure_tanh",
                                               amplitude_scale="factorial" if factorial else "none")
    wanted, gradients64 = _float64_gradients(reference, arrays, cotangent)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    np.testing.assert_allclose(jax.jit(actual)(*operands), wanted, rtol=3e-6, atol=1e-7)
    gradients = jax.grad(lambda *x: jnp.sum(actual(*x) * jnp.asarray(cotangent, jnp.float32)), argnums=(0, 1))(*operands)
    for name, value, expected in zip("xW", gradients, gradients64):
        np.testing.assert_allclose(value, expected, rtol=3e-5, atol=1e-6, err_msg=name)


def test_affine_sigmoid_and_derivatives_match_float64():
    rng = np.random.default_rng(6)
    arrays = (rng.normal(size=(2, 7, 8)), rng.normal(size=(2, 8, 5)) * .4, rng.normal(size=(2, 5)),
              rng.normal(size=(2, 5)))
    cotangent = rng.normal(size=(2, 7, 2, 5))
    reference = lambda x, w, b, a: b + (a / jnp.asarray(FACTORIALS)) * jax.nn.sigmoid(jnp.einsum("btd,hdj->bthj", x, w))
    actual = lambda x, w, b, a: project_coefficients(x, w, b, amplitude=a, activation="affine_sigmoid",
                                                     amplitude_scale="factorial")
    wanted, gradients64 = _float64_gradients(reference, arrays, cotangent)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    np.testing.assert_allclose(jax.jit(actual)(*operands), wanted, rtol=3e-6, atol=1e-6)
    gradients = jax.grad(lambda *x: jnp.sum(actual(*x) * jnp.asarray(cotangent, jnp.float32)),
                         argnums=(0, 1, 2, 3))(*operands)
    for name, value, expected in zip(("x", "W", "b", "a"), gradients, gradients64):
        np.testing.assert_allclose(value, expected, rtol=3e-5, atol=1e-6, err_msg=name)


def test_generator_parameters():
    pure = parameter_specs(hidden_size=8, degree=4, activation="pure_tanh", projection_initial_std=.3)
    assert [spec.name for spec in pure] == ["query_polynomial_kernel"] and pure[0].initial_std == .3
    affine = parameter_specs(hidden_size=8, degree=4, activation="affine_sigmoid", amplitude_initial=0.0,
                             projection_initial_std=.3)
    assert [spec.name for spec in affine] == ["query_polynomial_bias", "query_polynomial_kernel",
                                             "query_polynomial_amplitude"]
    with pytest.raises(ValueError, match="no bias"):
        project_coefficients(jnp.ones((1, 2, 8)), jnp.ones((1, 8, 5)), jnp.ones((1, 5)), activation="pure_tanh")


def test_config_validation_and_legacy_default(tmp_path):
    with pytest.raises(ValueError, match="random W start"):
        _config("pure_tanh", attention_coefficient_projection_initial_std=0.0)
    with pytest.raises(ValueError, match="tanh activation"):
        _config("affine_sigmoid", attention_coefficient_conditioning="query_key")
    for backend in ("mosaic", "fused"):
        with pytest.raises(ValueError, match="Backend 'mosaic'|RoPE/none on the blocked backend"):
            _config("tanh", forget_gate=True, attention_backend=backend)
    with pytest.raises(ValueError, match="boolean"):
        _config(forget_gate=1)
    assert not _config("tanh", forget_gate=True, max_seq_len=2048, qk_head_dim=64,
                       value_head_dim=64).model.mosaic_supported()
    for form in ("pure_tanh", "affine_sigmoid"):
        assert _config(form, attention_backend="auto", max_seq_len=2048, qk_head_dim=64,
                       value_head_dim=64).model.mosaic_supported()
    recipe = config_to_dict(_config())
    legacy = {**recipe, "model": {k: v for k, v in recipe["model"].items() if k != "attention_forget_gate"}}
    assert config_dicts_equal(recipe, legacy)
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy))
    assert load_config(path).model.attention_forget_gate is False


def _naive_decayed_polynomial(q, k, v, c, L):
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    weights = sum(jnp.swapaxes(c[..., r], 1, 2)[..., None] * s ** r for r in range(c.shape[-1]))
    gap = jnp.swapaxes(L, 1, 2)[..., :, None] - jnp.swapaxes(L, 1, 2)[..., None, :]
    causal = jnp.tril(jnp.ones(s.shape[-2:], bool))
    weights = jnp.where(causal, weights * jnp.exp(jnp.where(causal, gap, 0.0)), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


def test_blocked_forget_gate_and_all_derivatives_match_float64():
    rng = np.random.default_rng(7)
    b, t, h, d = 2, 37, 3, 4
    log_alpha = -np.abs(rng.normal(size=(b, t, h))) * .3
    arrays = (rng.normal(size=(b, t, h, d)), rng.normal(size=(b, t, h, d)), rng.normal(size=(b, t, h, d)),
              rng.normal(size=(b, t, h, 5)) / FACTORIALS, np.cumsum(log_alpha, axis=1))
    cotangent = rng.normal(size=(b, t, h, d))
    wanted, gradients64 = _float64_gradients(_naive_decayed_polynomial, arrays, cotangent)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    actual = lambda q, k, v, c, L: blocked_dot_product_attention(
        q, k, v, attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
        query_log_decay=L, query_block_size=16)
    # FP32 throughout: GPUs otherwise default FP32 matmuls to TF32, which s^4 amplifies.
    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(jax.jit(actual)(*operands), wanted, rtol=2e-5, atol=2e-5)
        gradients = jax.grad(lambda *x: jnp.sum(actual(*x) * jnp.asarray(cotangent, jnp.float32)),
                             argnums=tuple(range(5)))(*operands)
    for name, value, expected in zip(("q", "k", "v", "c", "L"), gradients, gradients64):
        scale = np.linalg.norm(expected)
        assert scale > 0, name
        assert np.linalg.norm(np.asarray(value) - expected) / scale < 2e-5, name
    # Without decay the same call is the ordinary per-query polynomial.
    with jax.default_matmul_precision("highest"):
        undecayed = blocked_dot_product_attention(*operands[:3], attention_degree=4, attention_normalization=False,
                                                  query_polynomial_coefficients=operands[3], query_block_size=16)
        zero = actual(*operands[:4], jnp.zeros_like(operands[4]))
    np.testing.assert_allclose(zero, undecayed, rtol=1e-6, atol=1e-6)


def test_forget_gate_formula_initialization_and_decay_policy():
    x = jnp.asarray(np.random.default_rng(8).normal(size=(2, 5, 8)), jnp.float32)
    kernel = jnp.asarray(np.random.default_rng(9).normal(size=(3, 8)) * .1, jnp.float32)
    log_rate, dt_bias = jnp.asarray([[0.5], [1.0], [2.0]]), jnp.asarray([[-1.0], [0.0], [1.0]])
    expected = -np.exp(np.asarray(log_rate)[:, 0]) * np.logaddexp(
        0, np.einsum("btd,hd->bth", np.asarray(x, np.float64), np.asarray(kernel, np.float64)) + np.asarray(dt_bias)[:, 0])
    np.testing.assert_allclose(log_decay(x, kernel, log_rate, dt_bias), expected, rtol=1e-5)
    values = initial_attention_parameter_values(forget_gate_specs(hidden_size=896), num_layers=15, num_heads=14,
                                                key=mechanism_key(jax.random.PRNGKey(42)))
    rate, dt = np.exp(values["forget_gate_log_rate"]), np.logaddexp(0, values["forget_gate_dt_bias"])
    assert rate.shape == (15, 14, 1) and (rate >= 0).all() and (rate < 16).all() and rate.std() > 3
    assert (dt >= 1e-3 * (1 - 1e-5)).all() and (dt <= .1 * (1 + 1e-5)).all()
    bound = 896 ** -.5
    assert values["forget_gate_kernel"].shape == (15, 14, 896)
    assert np.abs(values["forget_gate_kernel"]).max() <= bound and np.abs(values["forget_gate_kernel"]).max() > .9 * bound
    config = _config("tanh", forget_gate=True)
    state = create_state(config, jax.random.PRNGKey(0))
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(state.params, excluded_names=excluded))}
    assert mask[("forget_gate_kernel",)] is True
    assert mask[("forget_gate_log_rate",)] is False and mask[("forget_gate_dt_bias",)] is False
    facts = acceptance.matches_softmax_backbone_initialization(config)
    assert facts["mechanism_parameters"]["forget_gate_log_rate"]["initializer"] == "mamba2_log_rate"


def test_model_starts():
    """affine_sigmoid with a = 0 is fixed Taylor; a forget gate that never decays changes nothing."""
    tokens = jnp.arange(16).reshape(2, 8) % 16
    affine = build_model(_config("affine_sigmoid").model, 19)
    tanh = build_model(_config("tanh").model, 19)
    np.testing.assert_array_equal(affine(tokens), tanh(tokens))
    gated = build_model(_config("tanh", forget_gate=True).model, 19)
    assert np.abs(np.asarray(gated(tokens) - tanh(tokens))).max() > 1e-4
    gated.forget_gate_dt_bias[...] = jnp.full_like(gated.forget_gate_dt_bias[...], -200.0)
    np.testing.assert_allclose(gated(tokens), tanh(tokens), rtol=1e-6, atol=1e-6)
    pure = build_model(_config("pure_tanh").model, 19)
    assert np.isfinite(pure(tokens)).all() and not hasattr(pure, "query_polynomial_bias")


@pytest.mark.parametrize(("form", "forget_gate"), (("pure_tanh", False), ("affine_sigmoid", False), ("tanh", True)))
def test_each_run_trains_its_new_parameters(form, forget_gate):
    config = _config(form, forget_gate)
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.arange(18).reshape(2, 9) % 16
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    second, _, _ = step(first, jax.random.PRNGKey(2))
    assert np.isfinite(loss)
    names = ["query_polynomial_kernel"] + (["query_polynomial_amplitude"] if form != "pure_tanh" else []) + (
        ["forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias"] if forget_gate else [])
    for name in names:
        assert np.linalg.norm(np.asarray(second.params[name][...] - initial.params[name][...])) > 0, name


@pytest.mark.parametrize(("form", "forget_gate"), (("pure_tanh", False), ("affine_sigmoid", False), ("tanh", True)))
def test_diagnostics_describe_each_run(form, forget_gate):
    config = _config(form, forget_gate)
    identity = attention_identity(config)
    formula = identity["attention_formula"]
    assert {"pure_tanh": "c_r(x_t) = tanh(x_t W_r)", "affine_sigmoid": "b + (a/r!)*sigmoid(x_t W)",
            "tanh": "b + (a/r!)*tanh(x_t W)"}[form] in formula
    assert ("prod_(j=i+1..t) alpha_j" in formula) is forget_gate
    assert identity.get("attention_forget_gate", False) is forget_gate
    gate_count = (8 + 2) * 2 if forget_gate else 0
    assert identity["attention_coefficient_count"] + gate_count == attention_parameter_count(replace(
        config.model, qk_norm=False, qk_norm_head_scale=False, attention_output_scaling="none"))
    state = create_state(config, jax.random.PRNGKey(3))
    raw = [{f"query_coefficient_{stat}": np.full((2, 5), .1) for stat in ("mean", "std", "min", "max")}]
    report = query_coefficient_report(state.params, raw, amplitude_scale=config.model.attention_coefficient_amplitude_scale,
                                      coefficient_scale=config.model.attention_coefficient_scale, activation=form)
    assert report["activation"] == form and report["finite"]


def test_forget_gate_diagnostics_on_the_reference_path():
    from scaffolds.transformer.training.diagnostics import forget_gate_report

    config = _config("tanh", forget_gate=True)
    model = build_model(config.model, 19)
    _, layers = model(jnp.arange(16).reshape(2, 8) % 16, return_diagnostics=True)
    assert layers[0]["forget_gate_log_alpha_mean"].shape == (2,)
    params = nnx.state(model, nnx.Param)
    report = forget_gate_report(params, [jax.device_get(layer) for layer in layers])
    assert report["finite"] and np.all(np.asarray(report["log_alpha_sample"]["max"]) <= 0)


@pytest.mark.parametrize(("form", "forget_gate"), (("pure_tanh", False), ("affine_sigmoid", False), ("tanh", True)))
def test_training_diagnostics_end_to_end(form, forget_gate):
    """The trainer's full diagnostic pass, as run at each validation, accepts every run."""
    from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample

    config = _config(form, forget_gate)
    config.data.batch_size = config.train.micro_batch_size = 2
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["coefficients"]["activation"] == form
    assert ("forget_gate" in report["coefficients"]) is forget_gate
    if forget_gate:
        assert report["coefficients"]["forget_gate"]["finite"]


def test_acceptance_selects_form_checks():
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for form, forget_gate in (("pure_tanh", False), ("affine_sigmoid", False), ("tanh", True)):
        paths = acceptance.test_paths(spec, config_to_dict(_config(form, forget_gate)), resolved_backend="blocked",
                                      cpu=True)
        assert acceptance.COEFFICIENT_FORM_TESTS in paths and acceptance.QUERY_TANH_POLYNOMIAL_TESTS in paths


@pytest.mark.gpu
@pytest.mark.parametrize(("form", "forget_gate"), (("pure_tanh", False), ("affine_sigmoid", False), ("tanh", True)))
def test_gpu_model_forward_and_gradients_match_float32_reference(form, forget_gate):
    """Production backend in BF16 (Mosaic, or blocked with the forget gate) against the FP32 blocked model."""
    if jax.default_backend() != "gpu" or not jax.devices()[0].device_kind.startswith(("NVIDIA H100", "NVIDIA B300")):
        pytest.skip("Requires an H100 or B300 allocation")
    results = {}
    rng = np.random.default_rng(51)
    amplitude = rng.normal(size=(2, 2, 5)) * .5
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    production = "blocked" if forget_gate else "mosaic"
    for backend, dtype in (("blocked", "float32"), ("blocked", "bfloat16"), (production, "bfloat16")):
        config = _config(form, forget_gate, vocab_size=64, max_seq_len=256, num_layers=2, hidden_size=128,
                         qk_head_dim=64, value_head_dim=64, intermediate_size=256, attention_backend=backend,
                         attention_coefficient_projection_initial_std=128 ** -.5)
        config.train.dtype = dtype
        state = create_state(config, jax.random.PRNGKey(3))
        if form != "pure_tanh":
            state.params["query_polynomial_amplitude"][...] = jnp.asarray(amplitude, jnp.float32)
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            loss, gradients = jax.jit(loop.loss_and_grads)(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype] = float(loss), {path: np.asarray(value[...], np.float64)
                                               for path, value in nnx.to_flat_state(gradients)}
    (reference_loss, reference), (_, bf16) = results["blocked", "float32"], results["blocked", "bfloat16"]
    loss, actual = results[production, "bfloat16"]
    assert math.isfinite(loss) and abs(loss - reference_loss) < .02
    for path, expected in reference.items():
        assert np.isfinite(actual[path]).all(), path
        scale = np.linalg.norm(expected)
        if scale == 0:
            np.testing.assert_array_equal(actual[path], 0, err_msg=str(path))
            continue
        error = np.linalg.norm(actual[path] - expected) / scale
        rounding = np.linalg.norm(bf16[path] - expected) / scale
        assert error <= max(2.5 * rounding, .03), (path, error, rounding)
    if forget_gate:
        for name in ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias"):
            assert np.linalg.norm(actual[(name,)]) > 0, name
