"""Dense query readouts preserve diagonal checkpoints and learn cross-order features."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.input_dependent import project_coefficients
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity, query_coefficient_report


def model_config(**changes):
    settings = dict(vocab_size=16, max_seq_len=8, num_layers=2, hidden_size=8,
                    num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
                    intermediate_size=12, attention_degree=4, attention_backend="blocked",
                    attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                    attention_coefficient_activation="tanh", attention_coefficient_readout="dense",
                    attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                    attention_coefficient_projection_initial_std=.2, initialization="transformer_reference",
                    attention_output_scaling="head_rms_norm", qk_norm=True)
    return ModelConfig(**{**settings, **changes})


@pytest.mark.parametrize("degree", (4, 8))
@pytest.mark.parametrize("scaling", ("none", "factorial"))
def test_dense_diagonal_replays_coefficients_and_matches_amplitude_derivative(degree, scaling):
    rng = np.random.default_rng(31)
    count = degree + 1
    x = jnp.asarray(rng.normal(size=(3, 7, 8)), jnp.float32)
    kernel = jnp.asarray(rng.normal(size=(2, 8, count)) * .4, jnp.float32)
    bias = jnp.asarray(rng.normal(size=(2, count)), jnp.float32)
    amplitude = jnp.asarray(rng.normal(size=(2, count)), jnp.float32)
    inner = jnp.asarray(rng.normal(size=(2, count)) * .1, jnp.float32)
    mixing = amplitude[..., :, None] * jnp.eye(count)
    kwargs = dict(activation="tanh", tanh_bias=inner, amplitude_scale=scaling)
    diagonal = lambda a: project_coefficients(x, kernel, bias, amplitude=a, **kwargs)
    dense = lambda v: project_coefficients(x, kernel, bias, mixing=v, **kwargs)
    np.testing.assert_allclose(dense(mixing), diagonal(amplitude), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(jax.jit(dense)(mixing), jax.jit(diagonal)(amplitude))
    cotangent = jnp.asarray(rng.normal(size=diagonal(amplitude).shape), jnp.float32)
    dense_gradient = jax.grad(lambda v: jnp.sum(dense(v) * cotangent))(mixing)
    diagonal_gradient = jax.grad(lambda a: jnp.sum(diagonal(a) * cotangent))(amplitude)
    np.testing.assert_allclose(jnp.diagonal(dense_gradient, axis1=-2, axis2=-1),
                               diagonal_gradient, rtol=2e-5, atol=2e-6)
    # Independent derivative checks output-order scaling rather than feature scaling.
    features = np.tanh(np.einsum("btd,hdj->bthj", np.asarray(x, np.float64), np.asarray(kernel, np.float64))
                       + np.asarray(inner, np.float64))
    scales = np.asarray([math.factorial(r) for r in range(count)]) if scaling == "factorial" else np.ones(count)
    expected = np.einsum("bthr,bthj->hrj", np.asarray(cotangent, np.float64) / scales, features)
    np.testing.assert_allclose(dense_gradient, expected, rtol=4e-5, atol=4e-6)
    assert np.linalg.norm(np.asarray(dense_gradient) * (1 - np.eye(count))) > 0
    mixed = mixing + jnp.asarray(rng.normal(size=mixing.shape) * .1, jnp.float32)
    expected_output = np.asarray(bias) + np.einsum(
        "bthj,hrj->bthr", features, np.asarray(mixed, np.float64) / scales[None, :, None])
    np.testing.assert_allclose(dense(mixed), expected_output, rtol=4e-5, atol=4e-6)


def test_diagonal_conversion_replays_full_model_and_offdiagonal_changes_output():
    config = model_config()
    old = build_model(replace(config, attention_coefficient_readout="diagonal"), 19)
    dense = build_model(config, 19)
    rng = np.random.default_rng(9)
    old.query_polynomial_bias[...] = jnp.asarray(rng.normal(size=old.query_polynomial_bias.shape), jnp.float32)
    old.query_polynomial_amplitude[...] = jnp.asarray(rng.normal(size=old.query_polynomial_amplitude.shape), jnp.float32)
    dense.query_polynomial_bias[...] = old.query_polynomial_bias[...]
    dense.query_polynomial_mixing[...] = old.query_polynomial_amplitude[..., :, None] * jnp.eye(5)
    np.testing.assert_array_equal(dense.query_polynomial_kernel[...], old.query_polynomial_kernel[...])
    tokens = jnp.arange(24).reshape(3, 8) % 16
    reference, actual = old(tokens), dense(tokens)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-6)
    assert not hasattr(dense, "query_polynomial_amplitude")
    assert dense.query_polynomial_mixing.shape == (2, 2, 5, 5)
    assert model_parameter_count(config) - model_parameter_count(replace(config, attention_coefficient_readout="diagonal")) == 2 * 2 * (25 - 5)
    dense.query_polynomial_mixing[...] = dense.query_polynomial_mixing[...].at[0, 0, 0, 1].add(.5)
    changed = dense(tokens)
    assert np.max(np.abs(np.asarray(changed - actual))) > 1e-6
    # A coefficient may mix a future query's features, but no future token is visible.
    altered_tokens = tokens.at[:, 4:].set((tokens[:, 4:] + 3) % 16)
    np.testing.assert_allclose(dense(altered_tokens)[:, :4], changed[:, :4], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("changes", [
    {"attention_coefficient_readout": "unknown"},
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_activation": "identity", "attention_degree": 1},
    {"attention_coefficients": "fixed"},
    {"attention_coefficient_conditioning": "static"},
])
def test_dense_requires_its_declared_query_generator(changes):
    with pytest.raises(ValueError, match="readout"):
        model_config(**changes)


def test_dense_reports_mixing_and_preserves_backbone_initialization():
    config = Config(model=model_config())
    config.data.sequence_length = 9
    config.data.synthetic_vocab_size = 16
    facts = acceptance.matches_softmax_backbone_initialization(config)
    assert facts["mechanism_parameters"]["query_polynomial_mixing"]["shape"] == [2, 2, 5, 5]
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    assert acceptance.DENSE_QUERY_READOUT_TESTS in acceptance.test_paths(
        spec, config_to_dict(config), resolved_backend="blocked", cpu=True)
    identity = attention_identity(config)
    assert identity["attention_coefficient_readout"] == "dense"
    assert "sum_j (V_rj/r!)" in identity["attention_formula"]
    model = build_model(config.model, 3)
    params = nnx.state(model, nnx.Param)
    raw = [{f"query_coefficient_{stat}": np.ones((2, 5)) for stat in ("mean", "std", "min", "max")}
           for _ in range(2)]
    report = query_coefficient_report(params, raw, amplitude_scale="factorial", coefficient_scale="factorial")
    assert report["activation"] == "tanh" and report["readout"] == "dense" and report["finite"]
    effective = np.asarray(report["effective_modulation_mixing"])
    expected = np.asarray(model.query_polynomial_mixing[...]) / np.array([1, 1, 2, 6, 24])[:, None]
    np.testing.assert_array_equal(effective, expected)


def test_legacy_config_defaults_to_diagonal_and_dense_mixing_stays_fp32_without_decay():
    config = Config(model=model_config(attention_coefficient_readout="diagonal"))
    current = config_to_dict(config)
    legacy = config_to_dict(config)
    del legacy["model"]["attention_coefficient_readout"]
    assert config_dicts_equal(legacy, current)
    config.model = replace(config.model, attention_coefficient_readout="dense")
    assert not config_dicts_equal(legacy, config_to_dict(config))
    config.data.sequence_length = 9
    config.data.synthetic_vocab_size = 16
    config.optim.total_steps = 4
    config.optim.warmup_steps = 0
    state = create_state(config, jax.random.PRNGKey(0))
    state.params["query_polynomial_mixing"][...] += .125
    after = state.apply_gradients(grads=jax.tree.map(jnp.zeros_like, state.params))
    assert state.params["query_polynomial_mixing"].dtype == jnp.float32
    np.testing.assert_array_equal(state.params["query_polynomial_mixing"][...],
                                  after.params["query_polynomial_mixing"][...])


def _feature_config(degree, features, **changes):
    return model_config(attention_degree=degree, attention_coefficient_feature_count=features,
                        attention_coefficient_amplitude_initial=0.0, **changes)


@pytest.mark.parametrize("degree,features", [(2, 3), (2, 9), (4, 5), (4, 9), (8, 9)])
def test_feature_count_shapes_zero_start_reproduce_fixed_taylor(degree, features):
    config = _feature_config(degree, features)
    dense = build_model(config, 19)
    count = degree + 1
    assert dense.query_polynomial_kernel.shape == (2, 2, 8, features)
    assert dense.query_polynomial_mixing.shape == (2, 2, count, features)
    np.testing.assert_array_equal(dense.query_polynomial_mixing[...], 0)
    np.testing.assert_array_equal(dense.query_polynomial_bias[...], np.broadcast_to(1.0, (2, 2, count)))
    assert float(jnp.std(dense.query_polynomial_kernel[...])) > 0
    original = replace(config, attention_coefficient_readout="diagonal", attention_coefficient_feature_count=None)
    expected = 2 * 2 * (8 * features + count * features) - 2 * 2 * (8 * count + count)
    assert model_parameter_count(config) - model_parameter_count(original) == expected
    # V = 0 makes every coefficient exactly 1/r!: the same function as fixed Taylor with the
    # same backbone draw, since mechanism parameters use their own random streams.
    fixed = build_model(replace(original, attention_coefficients="fixed", attention_coefficient_sharing="global",
                                attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
                                attention_coefficient_scale="none", attention_coefficient_amplitude_scale="none",
                                attention_coefficient_amplitude_initial=1.0,
                                attention_coefficient_projection_initial_std=0.0), 19)
    tokens = jnp.arange(24).reshape(3, 8) % 16
    np.testing.assert_allclose(dense(tokens), fixed(tokens), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("degree,features", [(2, 9), (4, 9), (4, 2)])
@pytest.mark.parametrize("scaling", ("none", "factorial"))
def test_rectangular_readout_and_gradients_match_fp64(degree, features, scaling):
    rng = np.random.default_rng(57 + degree + features)
    count = degree + 1
    x = jnp.asarray(rng.normal(size=(3, 7, 8)), jnp.float32)
    kernel = jnp.asarray(rng.normal(size=(2, 8, features)) * .4, jnp.float32)
    bias = jnp.asarray(rng.normal(size=(2, count)), jnp.float32)
    mixing = jnp.asarray(rng.normal(size=(2, count, features)), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=(3, 7, 2, count)), jnp.float32)
    scales = np.asarray([math.factorial(r) for r in range(count)], np.float64) if scaling == "factorial" else np.ones(count)

    def reference(x, kernel, bias, mixing):
        features_ = jnp.tanh(jnp.einsum("btd,hdj->bthj", x, kernel))
        return bias + jnp.einsum("bthj,hrj->bthr", features_, mixing / jnp.asarray(scales)[None, :, None])

    project = lambda *a: project_coefficients(*a[:3], mixing=a[3], activation="tanh", amplitude_scale=scaling)
    loss = lambda f: lambda *a: jnp.sum(f(*a) * cotangent.astype(f(*a).dtype))
    with jax.enable_x64():
        operands64 = tuple(jnp.asarray(np.asarray(v), jnp.float64) for v in (x, kernel, bias, mixing))
        expected = np.asarray(reference(*operands64))
        expected_grads = jax.grad(loss(reference), argnums=(0, 1, 2, 3))(*operands64)
        expected_grads = [np.asarray(g) for g in expected_grads]
    np.testing.assert_allclose(project(x, kernel, bias, mixing), expected, rtol=2e-5, atol=2e-6)
    grads = jax.grad(loss(project), argnums=(0, 1, 2, 3))(x, kernel, bias, mixing)
    for name, actual, wanted in zip(("x", "kernel", "bias", "mixing"), grads, expected_grads):
        np.testing.assert_allclose(actual, wanted, rtol=5e-5, atol=5e-5, err_msg=name)
    # At V = 0 every V entry has a gradient and the projection has none: the zero-amplitude start.
    zero = jnp.zeros_like(mixing)
    np.testing.assert_array_equal(project(x, kernel, bias, zero), np.broadcast_to(bias, (3, 7, 2, count)))
    at_zero = jax.grad(loss(project), argnums=(1, 3))(x, kernel, bias, zero)
    np.testing.assert_array_equal(at_zero[0], 0)
    assert np.all(np.abs(np.asarray(at_zero[1])) > 0)


@pytest.mark.parametrize("changes,message", [
    ({"attention_coefficient_feature_count": 9, "attention_coefficient_readout": "diagonal",
      "attention_coefficient_amplitude_initial": 0.0}, "requires dense"),
    ({"attention_coefficient_feature_count": 9, "attention_coefficient_amplitude_initial": 1.0}, "no diagonal"),
    ({"attention_coefficient_feature_count": 0}, "positive integer"),
    ({"attention_coefficient_feature_count": True}, "positive integer"),
])
def test_feature_count_is_validated(changes, message):
    with pytest.raises(ValueError, match=message):
        model_config(**changes)


def test_legacy_recipes_default_to_degree_plus_one_features():
    config = Config(model=model_config())
    current, legacy = config_to_dict(config), config_to_dict(config)
    del legacy["model"]["attention_coefficient_feature_count"]
    assert config_dicts_equal(legacy, current)
    explicit = config_to_dict(Config(model=model_config(attention_coefficient_feature_count=5)))
    assert not config_dicts_equal(legacy, explicit)
    identity = attention_identity(Config(model=_feature_config(2, 9)))
    assert identity["attention_coefficient_feature_count"] == 9 and "j = 1..9" in identity["attention_formula"]


def _gpu_backend_widths():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires an H100 or B300 allocation")
    kind = jax.devices()[0].device_kind
    if not (kind.startswith("NVIDIA H100") or kind.startswith("NVIDIA B300")):
        pytest.skip(f"Mosaic qualification targets H100 and B300, not {kind}")
    return 256  # a sequence multiple of both GPUs' Mosaic tiles


# Mosaic's canonical BF16 weight precision rounds the degree-2 coefficient-bias gradient
# more than the blocked BF16 path once coefficients are strongly signed: on B300 and H100,
# 7.6-8.3% relative L2 against FP32 versus 1.9-3.2% (validation/README.md, dense query K).
_DEGREE_TWO_STRESS = pytest.mark.xfail(strict=False, reason="degree-2 coefficient-bias gradient under "
                                       "strongly signed coefficients exceeds 2.5x blocked BF16 rounding")


@pytest.mark.gpu
@pytest.mark.parametrize("degree,features,mixing_std", [
    pytest.param(degree, features, std, marks=_DEGREE_TWO_STRESS if degree == 2 and std == 0.3 else ())
    for degree, features in ((2, 3), (2, 9), (4, 5), (4, 9), (8, 9)) for std in (0.0, 0.05, 0.3)])
def test_gpu_mosaic_dense_model_gradients_stay_within_bf16_rounding(degree, features, mixing_std):
    """The production composition (QK RMSNorm, dense tanh coefficients, Mosaic, head RMSNorm) in
    BF16 against the blocked FP32 reference: Mosaic's whole-parameter gradient error stays within
    2.5x the blocked BF16 path's own error (floor 3%). V = 0 is the production start; std 0.3 is a
    strongly signed stress case."""
    from scaffolds.transformer.training.loop import loss_and_grads

    time = _gpu_backend_widths()
    mixing = None
    results = {}
    for backend, dtype in (("blocked", "float32"), ("blocked", "bfloat16"), ("mosaic", "bfloat16")):
        config = Config(model=_feature_config(
            degree, features, vocab_size=64, max_seq_len=time, num_layers=2, hidden_size=128,
            num_heads=2, num_kv_heads=2, qk_head_dim=64, value_head_dim=64, intermediate_size=256,
            attention_backend=backend, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
            initialization="transformer_truncated_reference", attention_coefficient_projection_initial_std=128 ** -.5))
        config.train.dtype = dtype
        config.data.sequence_length = time + 1
        config.data.synthetic_vocab_size = 64
        state = create_state(config, jax.random.PRNGKey(3))
        if mixing is None:
            mixing = np.random.default_rng(5 + degree + features).normal(
                size=state.params["query_polynomial_mixing"].shape) * mixing_std
        state.params["query_polynomial_mixing"][...] = jnp.asarray(mixing, jnp.float32)
        batch = jnp.asarray(np.random.default_rng(11).integers(0, 64, size=(2, time + 1)), jnp.int32)
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            loss, grad = jax.jit(loss_and_grads)(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype] = float(loss), {path: np.asarray(value[...], np.float64)
                                               for path, value in nnx.to_flat_state(grad)}
    (reference_loss, reference), (_, standard), (loss, mosaic) = (
        results["blocked", "float32"], results["blocked", "bfloat16"], results["mosaic", "bfloat16"])
    assert np.isfinite(loss) and abs(loss - reference_loss) < 2e-2
    for path, expected in reference.items():
        scale = np.linalg.norm(expected)
        assert np.isfinite(mosaic[path]).all(), path
        if scale == 0:
            continue
        error = np.linalg.norm(mosaic[path] - expected) / scale
        bf16 = np.linalg.norm(standard[path] - expected) / scale
        assert error <= max(2.5 * bf16, 0.03), (path, error, bf16)
    if mixing_std:
        assert np.linalg.norm(mosaic[("query_polynomial_kernel",)]) > 0
