"""Pure sigmoid query coefficients c_r = sigmoid(w_r^T x) [/ r!], diagonal or dense, with optional W decay."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.input_dependent import parameter_specs, project_coefficients
from scaffolds.transformer.attention.parameters import attention_parameter_count, attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_identity, query_coefficient_report
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state


def _config(degree=4, dense=False, factorial=True, decay=False, **changes):
    model = dict(vocab_size=16, max_seq_len=8, num_layers=1, hidden_size=8,
                 num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
                 intermediate_size=12, attention_degree=degree, attention_backend="blocked",
                 attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="sigmoid",
                 attention_coefficient_readout="dense" if dense else "diagonal",
                 attention_coefficient_feature_count=degree + 1 if dense else None,
                 attention_coefficient_readout_scale=5 ** -.5 if dense and factorial else 1.0,
                 attention_coefficient_amplitude_scale="factorial" if factorial else "none",
                 attention_coefficient_kernel_weight_decay=decay,
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    return config


@pytest.mark.parametrize("degree", (4, 8))
@pytest.mark.parametrize("dense", (False, True))
@pytest.mark.parametrize("factorial", (False, True))
def test_coefficients_and_all_derivatives_match_independent_float64(degree, dense, factorial):
    rng = np.random.default_rng(31 + degree)
    count = degree + 1
    arrays = [rng.normal(size=(2, 7, 8)), rng.normal(size=(2, 8, count)) * .4]
    if dense:
        arrays.append(rng.normal(size=(2, count, count)) * .5)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    cotangent = rng.normal(size=(2, 7, 2, count))
    factors = np.array([math.factorial(r) for r in range(count)], np.float64) if factorial else np.ones(count)
    scale = 5 ** -.5 if dense else 1.0

    def reference(x, w, v=None):
        features = jax.nn.sigmoid(jnp.einsum("btd,hdj->bthj", x, w))
        if v is None:
            return features / jnp.asarray(factors)
        return scale * jnp.einsum("bthj,hrj->bthr", features, v / jnp.asarray(factors)[None, :, None])

    def actual(x, w, v=None):
        return project_coefficients(x, w, None, mixing=v, activation="sigmoid",
                                    amplitude_scale="factorial" if factorial else "none", readout_scale=scale)

    arguments = tuple(range(len(arrays)))
    with jax.enable_x64():
        inputs64 = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        wanted = np.asarray(reference(*inputs64))
        gradients64 = [np.asarray(g) for g in jax.grad(
            lambda *x: jnp.sum(reference(*x) * jnp.asarray(cotangent)), argnums=arguments)(*inputs64)]
    got = jax.jit(actual)(*operands)
    assert got.dtype == jnp.float32 and got.shape == (2, 7, 2, count)
    np.testing.assert_allclose(got, wanted, rtol=3e-6, atol=1e-7)
    gradients = jax.grad(lambda *x: jnp.sum(actual(*x) * jnp.asarray(cotangent, jnp.float32)),
                         argnums=arguments)(*operands)
    for name, value, expected in zip(("x", "W", "V"), gradients, gradients64):
        np.testing.assert_allclose(value, expected, rtol=3e-5, atol=1e-6, err_msg=name)
        assert np.linalg.norm(expected) > 0, name


def test_projection_rejects_tanh_parameters_and_misshaped_mixing():
    x, w = jnp.ones((1, 2, 3)), jnp.ones((1, 3, 5))
    for extra in ({"amplitude": jnp.ones((1, 5))}, {"tanh_bias": jnp.zeros((1, 5))}):
        with pytest.raises(ValueError, match="no bias"):
            project_coefficients(x, w, None, activation="sigmoid", **extra)
    with pytest.raises(ValueError, match="no bias"):
        project_coefficients(x, w, jnp.ones((1, 5)), activation="sigmoid")
    with pytest.raises(ValueError, match="mixing"):
        project_coefficients(x, w, None, mixing=jnp.ones((1, 5, 4)), activation="sigmoid")
    with pytest.raises(ValueError, match="dense"):
        project_coefficients(x, w, None, activation="sigmoid", readout_scale=.5)


@pytest.mark.parametrize("decay", (False, True))
def test_specs_hold_only_w_and_dense_identity_v(decay):
    diagonal = parameter_specs(hidden_size=8, degree=4, activation="sigmoid", kernel_weight_decay=decay)
    assert [spec.name for spec in diagonal] == ["query_polynomial_kernel"]
    assert diagonal[0].initial == (0.0,) * 40 and diagonal[0].initial_shape == (8, 5)
    assert diagonal[0].weight_decay is decay
    dense = parameter_specs(hidden_size=8, degree=4, activation="sigmoid", readout="dense", kernel_weight_decay=decay)
    assert [spec.name for spec in dense] == ["query_polynomial_kernel", "query_polynomial_mixing"]
    np.testing.assert_array_equal(np.reshape(dense[1].initial, dense[1].initial_shape), np.eye(5))
    assert dense[1].weight_decay is False
    tanh = parameter_specs(hidden_size=8, degree=4, activation="tanh", amplitude_initial=0.0, kernel_weight_decay=decay)
    assert {spec.name: spec.weight_decay for spec in tanh} == {
        "query_polynomial_bias": False, "query_polynomial_kernel": decay, "query_polynomial_amplitude": False}
    for bad in ({"tanh_bias": True}, {"conditioning": "query_key"}, {"amplitude_initial": 0.0}):
        with pytest.raises(ValueError, match="[Ss]igmoid"):
            parameter_specs(hidden_size=8, degree=4, activation="sigmoid", **bad)


@pytest.mark.parametrize("decay", (False, True))
def test_weight_decay_reaches_w_alone(decay):
    config = _config(dense=True, decay=decay)
    state = create_state(config, jax.random.PRNGKey(0))
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = {path: bool(value.get_value()) for path, value in
            nnx.to_flat_state(attention_weight_decay_mask(state.params, excluded_names=excluded))}
    assert mask[("query_polynomial_kernel",)] is decay
    for name in ("query_polynomial_mixing", "qk_norm_head_log_scale", "qk_norm_q_gain", "output_norm_scale"):
        assert mask[(name,)] is False, name
    assert any(value for path, value in mask.items() if path[0] != "query_polynomial_kernel")


@pytest.mark.parametrize("dense", (False, True))
def test_order_scaled_start_is_taylor_and_dense_identity_start_is_diagonal(dense):
    """At W = 0 every c_r = 0.5/r! (dense: times 1/sqrt(5)); head RMSNorm removes the common scale.

    The toy model's outputs are small enough for RMSNorm's epsilon to resolve a factor of two,
    so every model here uses a negligible epsilon; the common scale is otherwise exact.
    """
    config = _config(dense=dense, rms_norm_eps=1e-12)
    model = build_model(config.model, 19)
    fixed = build_model(replace(config.model, attention_coefficients="fixed", attention_coefficient_readout="diagonal",
                                attention_coefficient_feature_count=None, attention_coefficient_readout_scale=1.0,
                                attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
                                attention_coefficient_amplitude_scale="none"), 19)
    diagonal = build_model(_config(rms_norm_eps=1e-12).model, 19)
    tokens = jnp.arange(16).reshape(2, 8) % 16
    np.testing.assert_allclose(model(tokens), fixed(tokens), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(model(tokens), diagonal(tokens), rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(model.query_polynomial_kernel[...], 0)
    assert not hasattr(model, "query_polynomial_bias")
    unscaled = build_model(_config(dense=dense, factorial=False, rms_norm_eps=1e-12).model, 19)
    assert np.isfinite(unscaled(tokens)).all()
    assert np.abs(np.asarray(unscaled(tokens) - model(tokens))).max() > 1e-3


@pytest.mark.parametrize("dense", (False, True))
@pytest.mark.parametrize("factorial", (False, True))
@pytest.mark.parametrize("decay", (False, True))
def test_every_regime_trains_its_generator(dense, factorial, decay):
    config = _config(dense=dense, factorial=factorial, decay=decay)
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.arange(18).reshape(2, 9) % 16
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    second, _, _ = step(first, jax.random.PRNGKey(2))
    assert np.isfinite(loss)
    names = ("query_polynomial_kernel", "query_polynomial_mixing") if dense else ("query_polynomial_kernel",)
    for name in names:
        assert np.linalg.norm(np.asarray(second.params[name][...] - initial.params[name][...])) > 0, name


def test_config_validation_and_legacy_default(tmp_path):
    for changes, match in (
            ({"attention_coefficient_amplitude_initial": 0.0}, "no amplitude"),
            ({"attention_coefficient_tanh_bias": True}, "tanh"),
            ({"attention_coefficient_conditioning": "query_key"}, "tanh"),
            ({"attention_coefficient_scale": "factorial"}, "coefficient scale"),
            ({"attention_coefficient_feature_count": 9, "attention_coefficient_readout": "dense"}, "degree\\+1")):
        with pytest.raises(ValueError, match=match):
            _config(**changes)
    with pytest.raises(ValueError, match="kernel weight decay"):
        _config(attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
                attention_coefficient_amplitude_scale="none", decay=True)
    with pytest.raises(ValueError, match="boolean"):
        _config(decay=1)
    _config(attention_coefficient_projection_initial_std=.1)
    recipe = config_to_dict(_config())
    legacy = {**recipe, "model": {key: value for key, value in recipe["model"].items()
                                  if key != "attention_coefficient_kernel_weight_decay"}}
    assert config_dicts_equal(recipe, legacy)
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy))
    assert load_config(path).model.attention_coefficient_kernel_weight_decay is False


@pytest.mark.parametrize("dense", (False, True))
@pytest.mark.parametrize("factorial", (False, True))
def test_diagnostics_describe_the_sigmoid_generator(dense, factorial):
    config = _config(dense=dense, factorial=factorial, decay=True)
    identity = attention_identity(config)
    formula = identity["attention_formula"]
    assert "sigmoid(x_t W" in formula and "b_r" not in formula
    assert ("/r!" in formula) is factorial
    assert ("V initially the identity" in formula) is dense
    assert ("0.447213595 * sum_j" in formula) is (dense and factorial)
    assert identity["attention_coefficient_kernel_weight_decay"] is True
    assert identity["attention_coefficient_activation"] == "sigmoid"
    assert identity["attention_coefficient_count"] == attention_parameter_count(replace(
        config.model, qk_norm=False, qk_norm_head_scale=False, attention_output_scaling="none"))
    state = create_state(config, jax.random.PRNGKey(3))
    raw = [{f"query_coefficient_{stat}": np.full((2, 5), .5) for stat in ("mean", "std", "min", "max")}]
    report = query_coefficient_report(state.params, raw, amplitude_scale=config.model.attention_coefficient_amplitude_scale,
                                      readout_scale=config.model.attention_coefficient_readout_scale,
                                      activation="sigmoid")
    assert report["activation"] == "sigmoid" and report["finite"] and "projection_bias" not in report
    assert len(report["effective_coefficient_names"]) == 5
    assert ("readout" in report) is dense


@pytest.mark.parametrize("dense", (False, True))
@pytest.mark.parametrize("decay", (False, True))
def test_acceptance_initialization_controls_accept_every_regime(dense, decay):
    """The softmax controls drop the generator, including its W decay, and keep the backbone."""
    facts = acceptance.matches_softmax_backbone_initialization(_config(dense=dense, decay=decay))
    parameters = facts["mechanism_parameters"]
    assert parameters["query_polynomial_kernel"]["weight_decay"] is decay
    assert "query_polynomial_bias" not in parameters
    assert ("query_polynomial_mixing" in parameters) is dense


def test_acceptance_selects_sigmoid_checks():
    config = config_to_dict(_config())
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (True, False):
        paths = acceptance.test_paths(spec, config, resolved_backend="mosaic", cpu=cpu)
        assert acceptance.SIGMOID_COEFFICIENT_TESTS in paths
        assert acceptance.QUERY_TANH_POLYNOMIAL_TESTS in paths
        assert acceptance.DENSE_QUERY_READOUT_TESTS not in paths
    assert _config(max_seq_len=2048, qk_head_dim=64, value_head_dim=64).model.mosaic_supported()


@pytest.mark.gpu
@pytest.mark.parametrize("dense", (False, True))
@pytest.mark.parametrize("factorial", (False, True))
def test_gpu_sigmoid_model_forward_and_gradients_match_reference(dense, factorial):
    """Production Mosaic composition against FP32 and BF16 blocked references at degree 4."""
    if jax.default_backend() != "gpu" or not jax.devices()[0].device_kind.startswith(("NVIDIA H100", "NVIDIA B300")):
        pytest.skip("Requires an H100 or B300 allocation")
    results = {}
    rng = np.random.default_rng(41 + 2 * dense + factorial)
    kernel = rng.normal(size=(2, 2, 128, 5)) * 128 ** -.5
    mixing = np.eye(5) + rng.normal(size=(2, 2, 5, 5)) * .15
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    head_scales = jnp.log(jnp.asarray([[[.8], [1.2]], [[1.1], [.9]]], jnp.float32))
    for backend, dtype in (("blocked", "float32"), ("blocked", "bfloat16"), ("mosaic", "bfloat16")):
        config = _config(dense=dense, factorial=factorial, decay=True, vocab_size=64, max_seq_len=256,
                         num_layers=2, hidden_size=128, qk_head_dim=64, value_head_dim=64,
                         intermediate_size=256, attention_backend=backend)
        config.train.dtype = dtype
        state = create_state(config, jax.random.PRNGKey(3))
        state.params["query_polynomial_kernel"][...] = jnp.asarray(kernel, jnp.float32)
        if dense:
            state.params["query_polynomial_mixing"][...] = jnp.asarray(mixing, jnp.float32)
        state.params["qk_norm_head_log_scale"][...] = head_scales
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            loss, gradients = jax.jit(loop.loss_and_grads)(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype] = float(loss), {path: np.asarray(value[...], np.float64)
                                               for path, value in nnx.to_flat_state(gradients)}
    (reference_loss, reference), (_, bf16), (loss, mosaic) = (
        results["blocked", "float32"], results["blocked", "bfloat16"], results["mosaic", "bfloat16"])
    assert math.isfinite(loss) and abs(loss - reference_loss) < .02
    for path, expected in reference.items():
        assert np.isfinite(mosaic[path]).all(), path
        scale = np.linalg.norm(expected)
        if scale == 0:
            np.testing.assert_array_equal(mosaic[path], 0, err_msg=str(path))
            continue
        error = np.linalg.norm(mosaic[path] - expected) / scale
        rounding = np.linalg.norm(bf16[path] - expected) / scale
        assert error <= max(2.5 * rounding, .03), (path, error, rounding)
    assert np.linalg.norm(mosaic[("query_polynomial_kernel",)]) > 0
    if dense:
        assert np.linalg.norm(mosaic[("query_polynomial_mixing",)]) > 0
