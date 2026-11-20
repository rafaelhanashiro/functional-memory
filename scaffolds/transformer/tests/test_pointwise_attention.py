"""Fixed sigmoid/softplus/swish baselines: parameter identity, complete MHA math, and dispatch."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, load_config, save_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.tests.composition_checks import refused_unless_composed


def _config(transform="softplus", **changes):
    model = dict(vocab_size=32, max_seq_len=16, num_layers=1, hidden_size=16,
                 num_heads=2, num_kv_heads=2, qk_head_dim=8, value_head_dim=8,
                 intermediate_size=32, attention_backend="blocked", attention_blocked_implementation="strips",
                 attention_degree=1, attention_normalization=False, attention_fixed_coefficients=(0., 1.),
                 attention_score_transform=transform, attention_output_scaling="head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 rms_norm_fp32=True, initialization="transformer_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length, config.data.synthetic_vocab_size = 17, 32
    config.data.batch_size = config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    if config.model.attention_backend == "mosaic":
        config.train.dtype = "bfloat16"
    return config


@pytest.mark.parametrize("transform", ("sigmoid", "softplus", "swish"))
def test_pointwise_recipe_roundtrip_and_mosaic_dispatch(tmp_path, transform):
    config = _config(transform, max_seq_len=2048, qk_head_dim=64, value_head_dim=64,
                     attention_backend="mosaic")
    assert config.model.mosaic_score_transform_supported()
    assert resolve_attention_backend(config.model, dtype="bfloat16", platform="gpu") == "mosaic"
    path = tmp_path / "config.yaml"
    save_config(config, path)
    restored = load_config(path)
    assert restored.model.attention_fixed_coefficients == (0., 1.)
    assert restored.model.attention_score_transform == transform
    identity = attention_identity(restored)
    assert identity["attention_kernel"] == f"{transform}_unnormalized"
    assert identity["attention_signed"] == (transform == "swish")  # swish reaches -0.2785
    assert identity["attention_coefficient_count"] == 0
    assert f"{transform}(s_ti)" in identity["attention_formula"]
    assert "per-head RMSNorm" in identity["attention_formula"]


@pytest.mark.parametrize("changes", (
    {"attention_fixed_coefficients": (1., 1.)}, {"attention_fixed_coefficients": None},
    {"attention_degree": 2, "attention_fixed_coefficients": (0., 0., 1.)},
    {"attention_normalization": True}, {"attention_score_transform": "abs"},
    {"attention_backend": "auto"}, {"attention_backend": "fused"},
    {"attention_backend": "manual"}, {"attention_coefficient_activation": "tanh"},
    {"attention_coefficient_conditioning": "query"}, {"attention_forget_gate": True},
    {"attention_score_bias": "static"}, {"attention_coefficient_input_power": 1},
))
def test_pointwise_refuses_other_mechanisms(changes):
    refused_unless_composed(_config, changes)


def test_sigmoid_transform_is_qualified_only_for_the_fixed_baseline():
    """On Mosaic; the blocked backend composes the sigmoid transform with any polynomial (attention/pieces.py)."""
    query = dict(attention_coefficients="learned", attention_fixed_coefficients=None,
                 attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                 attention_coefficient_sharing="per_head")
    _config("sigmoid", **query)
    with pytest.raises(ValueError, match="sigmoid score transform is qualified only"):
        _config("sigmoid", **query, attention_backend="mosaic")
    with pytest.raises(ValueError, match="fixed static linear"):
        multihead.MultiHeadAttention(
            hidden_size=8, num_heads=2, num_kv_heads=2, qk_head_dim=4, value_head_dim=4, backend="mosaic",
            attention_degree=1, attention_normalization=False, attention_score_transform="sigmoid",
            attention_coefficient_conditioning="query", attention_coefficient_activation="tanh", rngs=nnx.Rngs(0))


@pytest.mark.parametrize("transform", ("sigmoid", "softplus", "swish"))
def test_pointwise_preserves_every_exp_parameter_and_decay_mask(transform):
    config = _config(transform)
    exp = replace(config.model, attention_degree=None, attention_fixed_coefficients=None,
                  attention_score_transform="identity")
    models = [build_model(recipe, 42) for recipe in (config.model, exp)]
    params = [nnx.state(model, nnx.Param) for model in models]
    flat = [dict(nnx.to_flat_state(state)) for state in params]
    assert flat[0].keys() == flat[1].keys()
    for path in flat[0]:
        np.testing.assert_array_equal(flat[0][path][...], flat[1][path][...])
    assert attention_parameter_specs(config.model) == attention_parameter_specs(exp)
    assert model_parameter_count(config.model) == model_parameter_count(exp)
    masks = [attention_weight_decay_mask(state, excluded_names=frozenset(
        spec.name for spec in attention_parameter_specs(recipe) if not spec.weight_decay))
        for state, recipe in zip(params, (config.model, exp), strict=True)]
    assert jax.tree.structure(masks[0]) == jax.tree.structure(masks[1])
    assert all(a == b for a, b in zip(jax.tree.leaves(masks[0]), jax.tree.leaves(masks[1]), strict=True))


@pytest.mark.parametrize("transform", ("sigmoid", "softplus", "swish"))
@pytest.mark.parametrize("implementation", ("scan", "strips"))
def test_pointwise_complete_mha_matches_independent_dense_math_and_gradient(monkeypatch, transform, implementation):
    module = multihead.MultiHeadAttention(
        hidden_size=8, num_heads=2, num_kv_heads=2, qk_head_dim=4, value_head_dim=4,
        attention_degree=1, attention_normalization=False, attention_fixed_coefficients=(0., 1.),
        attention_score_transform=transform, attention_blocked_implementation=implementation, rngs=nnx.Rngs(11))
    x = jax.random.normal(jax.random.PRNGKey(17), (1, 9, 8))
    cotangent = jax.random.normal(jax.random.PRNGKey(19), x.shape)
    actual, backward = jax.vjp(lambda inputs: module(inputs), x)
    gradient, = backward(cotangent)

    def dense(q, k, v, **arguments):
        coefficients = arguments["query_polynomial_coefficients"]
        assert coefficients.dtype == jnp.float32
        np.testing.assert_array_equal(coefficients, jnp.broadcast_to(jnp.array([0., 1.]), coefficients.shape))
        assert "polynomial_coefficients" not in arguments
        assert not arguments["attention_normalization"]
        scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
        weights = {"sigmoid": 1 / (1 + jnp.exp(-scores)), "softplus": jnp.logaddexp(scores, 0.),
                   "swish": scores / (1 + jnp.exp(-scores))}[transform]
        causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(k.shape[1])[None, :]
        return jnp.einsum("bhti,bihd->bthd", jnp.where(causal, weights, 0.), v)

    monkeypatch.setattr(multihead, "attention_core", dense)
    expected, backward = jax.vjp(lambda inputs: module(inputs), x)
    expected_gradient, = backward(cotangent)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-5, atol=3e-6)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0


@pytest.mark.parametrize("transform", ("sigmoid", "softplus", "swish"))
def test_pointwise_diagnostics_measure_the_transformed_weights(transform):
    config = _config(transform)
    state = create_state(config, jax.random.PRNGKey(0))
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == f"unshifted_{transform}"
    assert not report["coefficients"]["trainable"]
    assert report["coefficients"]["score_transform"]["transform"] == transform


@pytest.mark.gpu
@pytest.mark.parametrize("transform", ("sigmoid", "softplus", "swish"))
@pytest.mark.parametrize("unique_tokens", (True, False), ids=("unique", "repeated"))
def test_pointwise_mosaic_full_model_bf16_gradients_and_repeatability(transform, unique_tokens):
    from scaffolds.transformer.attention.backends.mosaic.api import transformed_compute_capability_supported

    gpu = jax.devices()[0]
    if gpu.platform != "gpu" or not transformed_compute_capability_supported(gpu.compute_capability):
        pytest.skip("Pointwise Mosaic attention is qualified on H100/B200")
    cpu = jax.devices("cpu")[0]
    vocab_size = 512 if unique_tokens else 64
    config = _config(transform, max_seq_len=256, vocab_size=vocab_size, hidden_size=64,
                     qk_head_dim=64, value_head_dim=64, intermediate_size=128, attention_backend="mosaic")
    assert resolve_attention_backend(config.model, dtype="bfloat16") == "mosaic"
    with jax.default_device(cpu):
        model = build_model(config.model, 29)
        for _, module in nnx.iter_graph(model):
            if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
                module.dtype = jnp.bfloat16
        mosaic_graph, params = nnx.split(model, nnx.Param)
        for _, module in nnx.iter_graph(model):
            if isinstance(module, multihead.MultiHeadAttention):
                module.backend = "blocked"
        blocked_graph, _ = nnx.split(model, nnx.Param)
        rng = np.random.default_rng(31)
        ids = rng.permutation(vocab_size)[:256] if unique_tokens else rng.integers(vocab_size, size=256)
        tokens = jnp.asarray(ids[None, :], jnp.int32)

    def program(graph):
        def loss(parameters, inputs):
            logits = nnx.merge(graph, parameters)(inputs).astype(jnp.float32)
            return jnp.mean(logits ** 2), logits
        return jax.jit(jax.value_and_grad(loss, has_aux=True))

    gpu_program = program(mosaic_graph)
    gpu_inputs = jax.device_put((params, tokens), gpu)
    actual = gpu_program(*gpu_inputs)
    for replay in (gpu_program, program(mosaic_graph)):
        repeated = replay(*gpu_inputs)
        for (path, first), second in zip(jax.tree.leaves_with_path(actual), jax.tree.leaves(repeated), strict=True):
            np.testing.assert_array_equal(first, second, err_msg=jax.tree_util.keystr(path))

    expected = program(blocked_graph)(*jax.device_put((params, tokens), cpu))
    for (path, first), reference in zip(jax.tree.leaves_with_path(actual), jax.tree.leaves(expected), strict=True):
        name = jax.tree_util.keystr(path)
        got, want = np.asarray(first, np.float64), np.asarray(reference, np.float64)
        assert np.isfinite(got).all(), name
        # BF16 matmuls in the complete model accumulate the tile-level rounding differences.
        scale = max(float(np.max(np.abs(want))), 1e-8)
        np.testing.assert_allclose(got, want, rtol=.03, atol=.015 * scale, err_msg=name)
        assert np.linalg.norm(got - want) <= .03 * max(np.linalg.norm(want), 1e-8), name
    gradients = dict(nnx.to_flat_state(actual[1]))
    for projection in ("q_proj", "k_proj", "v_proj"):
        arrays = [np.asarray(value[...]) for path, value in gradients.items() if projection in path]
        assert arrays and all(np.linalg.norm(value) > 0 for value in arrays)
