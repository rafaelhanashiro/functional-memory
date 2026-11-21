"""CPU checks of measure_head_rms_backward.py.

Run from the repository root:
  JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest -q \
    -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
    scaffolds/transformer/validation/test_measure_head_rms_backward.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.validation import measure_head_rms_backward as measure

EPS = 1e-6


def _rows(scale, *, seed=0, shape=(1, 7, 2, 16)):
    generator = np.random.default_rng(seed)
    y = (scale * generator.standard_normal(shape)).astype(np.float32)
    gain = (1.0 + 0.1 * generator.standard_normal(shape[-2:])).astype(np.float32)
    cotangent = (1e-3 * generator.standard_normal(shape)).astype(np.float32)
    return y, gain, cotangent


def _relative(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def _vjp(function, y, gain, cotangent):
    output, pull_back = jax.vjp(function, jnp.asarray(y), jnp.asarray(gain))
    return (output, *pull_back(jnp.asarray(cotangent)))


@pytest.mark.parametrize("scale", (1e-12, 1.0, 1e9, 1e13, 1e16, 3e19, 1e30))
def test_the_stable_evaluation_is_head_rms_norm_and_its_gradient_at_every_scale(scale):
    """Output, dy and dgain against FP64, from rows far below eps to rows whose FP32 square overflows."""
    y, gain, cotangent = _rows(scale)
    output, value, weight = _vjp(lambda y, gain: measure.stable_head_rms_norm(y, gain, EPS), y, gain, cotangent)
    expected_value, expected_weight = measure._vjp_fp64(y, gain, cotangent, eps=EPS)
    assert _relative(output, measure._host_forward(y, gain, eps=EPS)) < 1e-6
    assert _relative(value, expected_value) < 2e-6 and _relative(weight, expected_weight) < 2e-6


def test_the_host_vjp_is_the_fp64_derivative_of_the_function_as_written():
    y, gain, cotangent = _rows(1e5, seed=1)
    with jax.enable_x64():
        _, value, weight = _vjp(lambda y, gain: measure.plain_head_rms_norm(y, gain, EPS),
                                *(np.asarray(array, np.float64) for array in (y, gain, cotangent)))
    expected_value, expected_weight = measure._vjp_fp64(y, gain, cotangent, eps=EPS)
    assert _relative(value, expected_value) < 1e-12 and _relative(weight, expected_weight) < 1e-12


@pytest.mark.parametrize("backward", ("stable", "fp64"))
def test_production_forward_keeps_productions_output_and_takes_the_other_backward(backward):
    y, gain, cotangent = _rows(1e15, seed=2)
    y = jnp.asarray(y, jnp.bfloat16)
    cotangent = np.asarray(jnp.asarray(cotangent, jnp.bfloat16))
    production = scale_attention_output(y, "head_rms_norm", output_norm_scale=jnp.asarray(gain), eps=EPS)
    output, value, weight = _vjp(
        lambda y, gain: measure.production_forward(scale_attention_output, y, gain, EPS, backward), y, gain, cotangent)
    np.testing.assert_array_equal(np.asarray(output, np.float32), np.asarray(production, np.float32))
    assert value.dtype == jnp.bfloat16 and weight.dtype == jnp.float32
    expected_value, expected_weight = measure._vjp_fp64(np.asarray(y, np.float32), gain, cotangent, eps=EPS)
    # dy is rounded to BF16, as the cotangent of a BF16 output is.
    assert _relative(value, expected_value) < 5e-3 and _relative(weight, expected_weight) < 1e-5


def test_the_probe_finds_where_this_backend_loses_the_radial_term():
    """On CPU, which flushes subnormals, production's VJP is exact below RMS 4.4e12 and keeps the whole radial
    component from there to the FP32 overflow of the mean square, past which the current source divides the row
    first; the stable and host FP64 evaluations are exact at every scale."""
    if jax.default_backend() != "cpu":
        pytest.skip("the flush threshold asserted here is the CPU's")
    probe = measure.backward_probe({"production": scale_attention_output}, host_callback=True, rows=64)
    by_scale = {entry["rms"]: entry for entry in probe["scales"]}
    for rms, entry in by_scale.items():
        assert entry["stable"]["relative_l2"] < 1e-6 and entry["fp64_head"]["relative_l2"] < 1e-6, rms
        lost = 4.4e12 <= rms <= 1e18
        assert (entry["production"]["radial_fraction"] > 0.99) == lost, rms
        assert (entry["production"]["relative_l2"] < 1e-6) == (not lost), rms
    assert measure.host_callback_problem() is None


def _model(*, dtype=jnp.bfloat16, **settings):
    from scaffolds.transformer.models import build_model

    config = ModelConfig(vocab_size=64, max_seq_len=32, num_layers=2, hidden_size=32, num_heads=2, num_kv_heads=2,
                         qk_head_dim=4, value_head_dim=16, intermediate_size=64, attention_backend="blocked",
                         attention_normalization=False, attention_output_scaling="head_rms_norm",
                         initialization="transformer_reference", qk_norm=True, qk_norm_type="rms_norm",
                         qk_norm_sharing="per_layer", qk_norm_head_scale=True, **settings)
    model = build_model(config, nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)))
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    model.qk_norm_head_log_scale = nnx.Param(model.qk_norm_head_log_scale[...] + 5.0)   # scores of a few hundred
    return nnx.split(model, nnx.Param)


def test_parameter_groups_cover_the_attention_parameters():
    _, params = _model(attention_degree=8, attention_coefficients="learned")
    groups = {measure._path_text(path).strip("/").removesuffix("/value"): measure.parameter_group(path)
              for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]}
    assert groups["layers_0/attention/q_proj/kernel"] == groups["layers_1/attention/k_proj/kernel"] == "qk"
    assert groups["qk_norm_q_gain"] == groups["qk_norm_k_gain"] == groups["qk_norm_head_log_scale"] == "qk"
    assert groups["layers_0/attention/v_proj/kernel"] == "v"
    assert groups["layers_0/attention/o_proj/kernel"] == "attention_output"
    assert groups["output_norm_scale"] == "head_rms_gain"
    assert groups["polynomial_coefficients"] == "kernel_coefficients"
    assert {groups[name] for name in ("token_embedding/embedding", "lm_head/kernel", "final_norm/scale",
                                      "layers_0/feed_forward/up_proj/kernel")} == {"other"}


def test_group_metrics_are_the_relative_difference_and_cosine_of_each_group():
    generator = np.random.default_rng(3)
    left = {"layers_0": {"attention": {"q_proj": {"kernel": generator.standard_normal((4, 3)) * 1e-30},
                                       "v_proj": {"kernel": generator.standard_normal((4, 3)) * 1e20}}},
            "lm_head": {"kernel": generator.standard_normal((5,))}}
    right = jax.tree.map(lambda leaf: leaf * (1.0 + 0.1 * generator.standard_normal(leaf.shape)), left)
    left, right = (jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), tree) for tree in (left, right))
    paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(left)[0]]
    metrics = measure.group_metrics([measure.parameter_group(path) for path in paths],
                                    [leaf.size for leaf in jax.tree.leaves(left)], measure.leaf_sums(left, right))
    for group, a, b in (("qk", left["layers_0"]["attention"]["q_proj"]["kernel"],
                         right["layers_0"]["attention"]["q_proj"]["kernel"]),
                        ("v", left["layers_0"]["attention"]["v_proj"]["kernel"],
                         right["layers_0"]["attention"]["v_proj"]["kernel"])):
        a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
        np.testing.assert_allclose(metrics[group]["relative_l2"], np.linalg.norm(a - b) / np.linalg.norm(b), rtol=1e-5)
        np.testing.assert_allclose(metrics[group]["cosine"], a @ b / np.linalg.norm(a) / np.linalg.norm(b), rtol=1e-5)
    assert metrics["kernel_coefficients"] is None and metrics["all"]["parameters"] == 29


def test_one_program_evaluates_every_head_rms_norm_of_a_model():
    """A two-layer BF16 model whose head outputs pass the CPU's threshold: the run-time selection reproduces the
    production evaluation traced alone; the stable and host FP64 evaluations agree with each other and not with
    production; the two that take production's forward return its loss; the FP64 model runs in FP64."""
    graphdef, params = _model(attention_degree=8)
    tokens = jnp.asarray(np.random.default_rng(4).integers(0, 64, (1, 33)), jnp.int32)
    head_norm = measure.HeadNorm(scale_attention_output)
    measure.install(head_norm)
    try:
        evaluate = measure.gradient_function(graphdef, head_norm)
        results = {name: evaluate(params, tokens, jnp.asarray(index, jnp.int32))
                   for index, name in enumerate(measure.VARIANTS)}
        (direct_loss, _), direct = measure.gradient_function(graphdef, head_norm, variant="production")(params, tokens)
        (fp64_loss, fp64_rows, fp64_gradient, dtypes) = measure.fp64_model_gradient(graphdef, params, tokens, head_norm)
    finally:
        from scaffolds.transformer.attention import multihead

        multihead.scale_attention_output = scale_attention_output
    (loss, rows), production = results["production"]
    assert len(rows) == 2 and float(max(layer["largest_rms"] for layer in rows)) > 1e13
    assert int(sum(layer["rows_subnormal"] for layer in rows)) > 0

    def difference(left, right):
        sums = np.asarray(measure.leaf_sums(left, right), np.float64)
        return float(np.sqrt((sums[:, 0] ** 2 * sums[:, 1]).sum() / (sums[:, 0] ** 2 * sums[:, 3]).sum()))

    assert float(direct_loss) == float(loss) and difference(direct, production) < 1e-6
    assert difference(results["stable"][1], results["fp64_head"][1]) < 1e-3
    assert difference(results["stable_backward"][1], results["fp64_backward"][1]) < 1e-3
    assert difference(production, results["stable"][1]) > 0.05
    assert difference(production, results["stable_backward"][1]) > 0.05
    for name in ("stable_backward", "fp64_backward"):
        assert float(results[name][0][0]) == float(loss)
    assert dtypes == ["float64"] and abs(fp64_loss - float(loss)) < 0.05
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(fp64_gradient)) and len(fp64_rows) == 2
