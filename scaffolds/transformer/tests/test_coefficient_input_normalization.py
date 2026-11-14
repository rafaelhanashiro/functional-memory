"""Unit-L2 coefficient inputs: independent derivatives and attention-branch isolation."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer.attention import input_dependent, multihead
from scaffolds.transformer.attention.nonnegative_polynomial import query_square_coefficients, taylor_square_factors
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model

EPS = 1e-6
DEGREE = 8
FACTORIALS = np.asarray([math.factorial(r) for r in range(DEGREE + 1)], np.float64)


def _config(parameterization="direct", normalization="l2", **changes):
    scale = "none" if parameterization == "sum_of_squares" else "factorial"
    settings = dict(vocab_size=32, max_seq_len=8, num_layers=1, hidden_size=16,
                    num_heads=2, num_kv_heads=2, qk_head_dim=8, value_head_dim=8,
                    intermediate_size=32, positional_encoding="rope", attention_backend="blocked",
                    attention_degree=DEGREE, attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                    attention_coefficient_activation="tanh", attention_coefficient_amplitude_initial=0.,
                    attention_coefficient_projection_initial_std=.25, attention_coefficient_amplitude_scale=scale,
                    attention_coefficient_scale=scale, attention_coefficient_kernel_weight_decay=True,
                    attention_coefficient_parameterization=parameterization,
                    attention_coefficient_input_normalization=normalization,
                    attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                    qk_norm_sharing="per_layer", qk_norm_head_scale=True, rms_norm_fp32=True,
                    initialization="transformer_reference")
    config = Config(model=ModelConfig(**(settings | changes)))
    config.data.sequence_length, config.data.synthetic_vocab_size = 9, 32
    validate_config(config)
    return config


def _unit_numpy(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def test_l2_value_and_vjp_match_analytic_jacobian_including_zero():
    rng = np.random.default_rng(21)
    directions = rng.normal(size=(2, 5, 11))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    # Both sides of the epsilon floor, without relying on the derivative at its kink.
    values = (directions * np.asarray([0., 1e-8, 1e-3, 1., 1e4])[None, :, None]).astype(np.float32)
    cotangent = rng.normal(size=values.shape).astype(np.float32)
    x = values.astype(np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.maximum(norm, EPS)
    expected_gradient = cotangent / denominator
    expected_gradient -= np.where(norm > EPS, x * np.sum(x * cotangent, axis=-1, keepdims=True)
                                  / denominator ** 3, 0.)
    operation = jax.jit(lambda x: input_dependent.normalize_coefficient_input(x, "l2"))
    actual, backward = jax.vjp(operation, jnp.asarray(values))
    gradient, = backward(jnp.asarray(cotangent))
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(actual, _unit_numpy(x), rtol=3e-6, atol=1e-8)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-5, atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(actual[:, 2:], axis=-1), 1., rtol=2e-6)
    np.testing.assert_array_equal(actual[:, 0], 0.)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_none_is_exact_legacy_identity_and_l2_uses_float32(dtype):
    x = jnp.asarray(np.random.default_rng(11).normal(size=(2, 3, 16)), dtype)
    assert input_dependent.normalize_coefficient_input(x, "none") is x
    normalized = input_dependent.normalize_coefficient_input(x, "l2")
    assert normalized.dtype == jnp.float32
    np.testing.assert_allclose(normalized, _unit_numpy(np.asarray(x, np.float32)), rtol=3e-6, atol=1e-7)


def _oracle(x, w, b, a, *, parameterization):
    # Independent FP64 unit-vector projection and tanh; no production normalization/projector.
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), EPS)
    g = b + a * jnp.tanh(jnp.einsum("btd,hdr->bthr", x, w))
    if parameterization == "direct":
        return g / jnp.asarray(FACTORIALS)
    factors = jnp.asarray(taylor_square_factors(DEGREE)) * (1. + g) / 2.
    def convolve(f):
        u, v = f[:5], f[5:]
        return jnp.convolve(u, u) + jnp.pad(jnp.convolve(v, v), (0, 2))
    return jax.vmap(convolve)(factors.reshape(-1, DEGREE + 1)).reshape(g.shape)


def _production(x, w, b, a, *, parameterization):
    x = input_dependent.normalize_coefficient_input(x, "l2")
    if parameterization == "direct":
        return input_dependent.project_coefficients(
            x, w, b / jnp.asarray(FACTORIALS, jnp.float32), amplitude=a,
            activation="tanh", amplitude_scale="factorial")
    g = input_dependent.project_coefficients(x, w, b, amplitude=a, activation="tanh")
    return query_square_coefficients(g, DEGREE)


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_generator_and_all_operand_gradients_match_independent_float64(parameterization):
    rng = np.random.default_rng(31)
    arrays = (rng.normal(size=(2, 3, 16)) * 4., rng.normal(size=(2, 16, 9)) * .5,
              rng.uniform(.6, 1.2, size=(2, 9)), rng.uniform(-.8, .9, size=(2, 9)))
    # Equalize the effective coefficient orders in the objective; high-order errors must not hide.
    cotangent = rng.normal(size=(2, 3, 2, 9)) * FACTORIALS
    with jax.enable_x64():
        wide = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        reference = lambda *args: _oracle(*args, parameterization=parameterization)
        expected, backward = jax.vjp(reference, *wide)
        expected_gradients = jax.tree.map(np.asarray, backward(jnp.asarray(cotangent)))
        expected = np.asarray(expected)
    operation = jax.jit(lambda *args: _production(*args, parameterization=parameterization))
    actual, backward = jax.vjp(operation, *(jnp.asarray(x, jnp.float32) for x in arrays))
    gradients = backward(jnp.asarray(cotangent, jnp.float32))
    np.testing.assert_allclose(np.asarray(actual) * FACTORIALS, expected * FACTORIALS,
                               rtol=5e-6, atol=2e-6)
    for name, actual, wanted in zip(("x", "W", "b", "a"), gradients, expected_gradients, strict=True):
        assert np.linalg.norm(wanted) > 0, name
        np.testing.assert_allclose(actual, wanted, rtol=2e-4, atol=2e-5, err_msg=name)


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_model_changes_only_coefficient_branch_preserving_gain_direction_and_qkv(monkeypatch, parameterization):
    base = _config(parameterization, "none")
    changed = _config(parameterization, "l2")
    assert attention_parameter_specs(base.model) == attention_parameter_specs(changed.model)
    models = [build_model(config.model, 19) for config in (base, changed)]
    first = dict(nnx.to_flat_state(nnx.state(models[0], nnx.Param)))
    second = dict(nnx.to_flat_state(nnx.state(models[1], nnx.Param)))
    assert first.keys() == second.keys()
    for path in first:
        np.testing.assert_array_equal(first[path][...], second[path][...])
    tokens = jnp.arange(16).reshape(2, 8) % 32
    # The standard a=0 start must remain identical, not merely statistically equivalent.
    np.testing.assert_array_equal(models[0](tokens), models[1](tokens))
    for model in models:
        model.query_polynomial_amplitude[...] = jnp.linspace(-.7, .9, 18).reshape(1, 2, 9)
        model.query_polynomial_bias[...] = jnp.linspace(.6, 1.3, 18).reshape(1, 2, 9)
        model.layers_0.attention_norm.scale[...] = jnp.linspace(.4, 1.8, 16)
    projection_inputs, operands, generated = [], [], []
    original_projection = input_dependent.project_coefficients
    original_core = multihead.attention_core
    def capture_projection(x, *args, **kwargs):
        projection_inputs.append(np.asarray(x))
        result = original_projection(x, *args, **kwargs)
        generated.append(np.asarray(result))
        return result
    def capture_core(q, k, v, **kwargs):
        operands.append(tuple(np.asarray(value) for value in (q, k, v)))
        return original_core(q, k, v, **kwargs)
    monkeypatch.setattr(input_dependent, "project_coefficients", capture_projection)
    monkeypatch.setattr(multihead, "attention_core", capture_core)
    outputs = [model(tokens) for model in models]
    assert len(projection_inputs) == len(operands) == 2
    np.testing.assert_allclose(projection_inputs[1], _unit_numpy(projection_inputs[0]), rtol=3e-6, atol=1e-7)
    for before, after in zip(operands[0], operands[1], strict=True):
        np.testing.assert_array_equal(before, after)
    assert not np.allclose(generated[0], generated[1], rtol=1e-5, atol=1e-7)
    assert not np.allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)
    # Injecting the exact changed generator output into the unmodified backbone reproduces the new model.
    # SoS expansion remains on its native path; no unsupported model-level SoS override is used.
    changed_generator = jnp.asarray(generated[1])
    monkeypatch.setattr(input_dependent, "project_coefficients", lambda *args, **kwargs: changed_generator)
    np.testing.assert_array_equal(models[0](tokens), outputs[1])


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_config_roundtrip_and_legacy_resume_semantics(tmp_path, parameterization):
    current = config_to_dict(_config(parameterization, "none"))
    legacy = {**current, "model": dict(current["model"])}
    del legacy["model"]["attention_coefficient_input_normalization"]
    assert config_dicts_equal(current, legacy)
    normalized = config_to_dict(_config(parameterization, "l2"))
    assert not config_dicts_equal(normalized, legacy)
    for label, recipe, expected in (("legacy", legacy, "none"), ("l2", normalized, "l2")):
        path = tmp_path / (label + ".yaml")
        path.write_text(yaml.safe_dump(recipe))
        assert load_config(path).model.attention_coefficient_input_normalization == expected
    mosaic = replace(_config(parameterization).model, attention_backend="mosaic", max_seq_len=2048,
                     qk_head_dim=64, value_head_dim=64)
    assert mosaic.mosaic_supported()


@pytest.mark.parametrize("changes", (
    {"attention_coefficient_input_normalization": "rms"},
    {"attention_coefficient_input_normalization": True},
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_tanh_bias": True},
    {"attention_score_transform": "softplus"},
    {"attention_forget_gate": True},
    {"attention_coefficient_readout": "dense"},
    {"attention_coefficient_gate": "sigmoid"},
))
def test_unsupported_combinations_are_rejected(changes):
    with pytest.raises(ValueError):
        _config(**changes)


def test_diagnostics_measure_actual_normalized_inputs_and_projection_distribution():
    rng = np.random.default_rng(41)
    raw = jnp.asarray(rng.normal(size=(2, 5, 16)) * 7, jnp.float32)
    normalized = input_dependent.normalize_coefficient_input(raw, "l2")
    kernel = jnp.asarray(rng.normal(size=(2, 16, 9)) * 3, jnp.float32)
    report = input_dependent.coefficient_input_statistics(raw, normalized, kernel)
    reference_z = np.einsum("btd,hdr->bthr", _unit_numpy(raw), np.asarray(kernel, np.float64))
    features = np.tanh(reference_z)
    for prefix, values in (("coefficient_input_raw_l2", np.linalg.norm(raw, axis=-1)),
                           ("coefficient_input_l2", np.linalg.norm(normalized, axis=-1)),
                           ("coefficient_projection", reference_z)):
        axes = (0, 1) if prefix == "coefficient_projection" else None
        for suffix, operation in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
            np.testing.assert_allclose(report[prefix + "_" + suffix], operation(values, axis=axes),
                                       rtol=1e-5, atol=2e-6)
    np.testing.assert_array_equal(report["coefficient_tanh_saturation_fraction"],
                                  (np.abs(features) >= .99).mean(axis=(0, 1)).astype(np.float32))
    np.testing.assert_allclose(report["coefficient_tanh_derivative_mean"],
                               (1. - features ** 2).mean(axis=(0, 1)), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_training_diagnostics_collect_generator_coordinates_and_baseline_omits_them(parameterization):
    from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
    from scaffolds.transformer.training.state import create_state

    for normalization in ("none", "l2"):
        config = _config(parameterization, normalization)
        config.data.batch_size = config.train.micro_batch_size = 1
        state = create_state(config, jax.random.PRNGKey(23))
        sample, provenance = fixed_diagnostic_sample(config)
        report = attention_diagnostics(state, config, sample, provenance)
        if normalization == "none":
            assert "generator_input" not in report["coefficients"]
            continue
        generator = report["coefficients"]["generator_input"]
        assert generator["finite"] and generator["normalization"] == "l2"
        assert generator["epsilon"] == EPS
        assert generator["projection_axes"] == ["layer", "head", "generator_coordinate"]
        expected_labels = ([f"r{i}" for i in range(9)] if parameterization == "direct" else
                           [f"U{i}" for i in range(5)] + [f"V{i}" for i in range(4)])
        assert generator["coordinate_labels"] == expected_labels
        model = nnx.merge(state.graphdef, state.params)
        _, layers = model(jnp.asarray(sample[:, :-1]), return_diagnostics=True)
        for statistic in ("mean", "std", "min", "max"):
            np.testing.assert_allclose(generator["projection_summary"][statistic][0],
                layers[0]["coefficient_projection_" + statistic], rtol=2e-6, atol=1e-7)
            for kind, prefix in (("before", "coefficient_input_raw_l2"), ("after", "coefficient_input_l2")):
                np.testing.assert_allclose(generator["input_norms"][kind][statistic][0],
                    layers[0][prefix + "_" + statistic], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(generator["input_norms"]["after"]["mean"], 1., rtol=2e-6)
        np.testing.assert_allclose(generator["saturation_fraction"][0],
                                   layers[0]["coefficient_tanh_saturation_fraction"], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(generator["mean_tanh_derivative"][0],
                                   layers[0]["coefficient_tanh_derivative_mean"], rtol=2e-6, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.parametrize("parameterization,precision", (
    ("direct", "bf16"), ("sum_of_squares", "bf16"),
    ("direct", "bf16x3"), ("sum_of_squares", "bf16x3"),
), ids=("direct-bf16", "sum_of_squares-bf16", "direct-bf16x3", "sum_of_squares-bf16x3"))
def test_mosaic_composition_matches_fp64_with_all_gradients(parameterization, precision):
    """Native Mosaic VJPs through unit normalization and both m8 generator parameterizations."""
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires H100/B200/B300")
    from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
    from scaffolds.transformer.tests.test_mosaic_query import _numpy_fp64
    from scaffolds.transformer.tests.test_fused_attention import _assert_relative

    rng = np.random.default_rng(103)
    q, k = (jnp.asarray(.7 * rng.normal(size=(1, 512, 2, 64)), jnp.bfloat16) for _ in range(2))
    v = jnp.asarray(rng.normal(size=(1, 512, 2, 64)), jnp.bfloat16)
    generator = (rng.normal(size=(1, 512, 16)).astype(np.float32) * 4.,
                 rng.normal(size=(2, 16, 9)).astype(np.float32) * .5,
                 rng.uniform(.7, 1.2, size=(2, 9)).astype(np.float32),
                 rng.uniform(-.6, .8, size=(2, 9)).astype(np.float32))
    cotangent = jnp.asarray(rng.normal(size=v.shape), jnp.bfloat16)
    operands = (q, k, v, *(jnp.asarray(value) for value in generator))
    def composed(q, k, v, x, w, b, a):
        coefficients = _production(x, w, b, a, parameterization=parameterization)
        return query_polynomial_attention(q, k, v, coefficients, precision=precision)
    def compute(*args):
        output, backward = jax.vjp(composed, *args)
        return output, backward(cotangent)
    actual, gradients = jax.jit(compute)(*operands)
    with jax.enable_x64():
        wide = tuple(jnp.asarray(value, jnp.float64) for value in generator)
        coefficients, backward = jax.vjp(lambda *args: _oracle(*args, parameterization=parameterization), *wide)
        expected, expected_qkvc = _numpy_fp64((q, k, v, np.asarray(coefficients)), cotangent)
        expected_generator = jax.tree.map(np.asarray, backward(jnp.asarray(expected_qkvc[3])))
    _assert_relative(actual, expected, rtol=.006, name="composed output")
    for name, got, want, tolerance in zip("qkv", gradients[:3], expected_qkvc[:3], (.025, .025, .015), strict=True):
        _assert_relative(got, want, rtol=tolerance, name="composed d" + name)
    _assert_relative(gradients[3], expected_generator[0], rtol=.025, name="composed dx")
    for name, got, want in zip(("W", "b", "a"), gradients[4:], expected_generator[1:], strict=True):
        assert got.dtype == jnp.float32
        for coordinate in range(DEGREE + 1):
            _assert_relative(got[..., coordinate], want[..., coordinate], rtol=.025,
                             name=f"composed d{name} coordinate {coordinate}")
