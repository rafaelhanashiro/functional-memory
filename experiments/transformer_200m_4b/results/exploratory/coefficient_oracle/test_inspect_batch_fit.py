"""CPU checks for diagnostic projection and native-forward preservation."""
from dataclasses import replace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_training import tiny_config
from experiments.transformer_200m_4b.results.exploratory.coefficient_oracle.inspect_batch_fit import (
    bounded_replay_control, first_layer_token_directions, fit_direction_basis, make_native_evaluator,
    make_query_inspector, nll_parity, require_diagnostic_parity,
    replace_first_layer_by_token, transform_coefficients,
)


def test_pca_full_rank_and_unit_radius_controls():
    rng = np.random.default_rng(982)
    table = rng.normal(size=(8, 2, 9, 3, 5)).astype(np.float32)
    means, bases, _ = fit_direction_basis(table, 6)
    reconstructed = transform_coefficients(table, means, bases, rank=5)
    np.testing.assert_array_equal(reconstructed, table)
    normalized = transform_coefficients(table, means, bases, unit=True)
    factorial = np.asarray([1, 1, 2, 6, 24])
    np.testing.assert_allclose(np.linalg.norm(normalized * factorial, axis=-1), 1., atol=1e-7)
    direction = table * factorial / np.linalg.norm(table * factorial, axis=-1, keepdims=True)
    np.testing.assert_allclose(normalized * factorial, direction, atol=1e-7)
    mean_only = transform_coefficients(table, means, bases, rank=0) * factorial
    mean_direction = means / np.linalg.norm(means, axis=-1, keepdims=True)
    expected = mean_direction[None, :, None] * np.linalg.norm(table * factorial, axis=-1, keepdims=True)
    np.testing.assert_allclose(mean_only, expected, atol=2e-6, rtol=1e-6)
    for rank in range(5):
        compressed = transform_coefficients(table, means, bases, rank=rank) * factorial
        np.testing.assert_allclose(np.linalg.norm(compressed, axis=-1),
                                   np.linalg.norm(table * factorial, axis=-1), atol=2e-6, rtol=1e-6)


def test_pca_does_not_fit_the_other_sequences():
    rng = np.random.default_rng(341)
    table = rng.normal(size=(8, 2, 9, 3, 5)).astype(np.float32)
    reference = fit_direction_basis(table, 6)
    table[6:] *= rng.uniform(-100, 100, size=table[6:].shape)
    changed = fit_direction_basis(table, 6)
    for left, right in zip(reference, changed, strict=True):
        np.testing.assert_array_equal(left, right)


def test_token_replacement_changes_only_layer_zero_direction():
    rng = np.random.default_rng(42)
    table = rng.normal(size=(8, 2, 9, 3, 5)).astype(np.float32)
    inputs = rng.integers(0, 16, size=(8, 9))
    directions, diagnostics = first_layer_token_directions([(0, table[:4]), (4, table[4:])], inputs, 16)
    result = replace_first_layer_by_token(table, inputs, directions)
    np.testing.assert_array_equal(result[:, 1:], table[:, 1:])
    factorial = np.asarray([1, 1, 2, 6, 24])
    transformed = result[:, 0] * factorial
    norms = np.linalg.norm(transformed, axis=-1, keepdims=True)
    np.testing.assert_allclose(norms, np.linalg.norm(table[:, 0] * factorial, axis=-1, keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(transformed / norms, directions[inputs], atol=1e-7)
    fixed_radius = replace_first_layer_by_token(table, inputs, directions, unit=True)
    np.testing.assert_array_equal(fixed_radius[:, 1:], table[:, 1:])
    np.testing.assert_allclose(fixed_radius[:, 0] * factorial, directions[inputs], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(fixed_radius[:, 0] * factorial, axis=-1), 1., atol=1e-7)
    for token in np.unique(inputs):
        repeated = fixed_radius[:, 0][inputs == token]
        np.testing.assert_array_equal(repeated, np.broadcast_to(repeated[0], repeated.shape))
    assert diagnostics["token_occurrences"] == 72


def test_parity_discrepancy_mode_preserves_failure_and_aggregate_gate():
    measured = nll_parity([2., 2.], [2.003, 1.997], .002)
    assert measured["aggregate_nll_passed"]
    assert not measured["original_per_sequence_bound_passed"]
    assert measured["sequences_exceeding_original_bound"] == 2
    assert len(measured["per_sequence_nll_deltas"]) == 2
    with pytest.raises(ValueError, match="FAILED"):
        require_diagnostic_parity(measured, record_discrepancies=False)
    require_diagnostic_parity(measured, record_discrepancies=True)
    failed_mean = nll_parity([2., 2.], [2.003, 2.004], .002)
    with pytest.raises(ValueError, match="aggregate"):
        require_diagnostic_parity(failed_mean, record_discrepancies=True)


def test_query_inspector_preserves_native_forward_and_reads_live_inputs(tmp_path):
    config = tiny_config(tmp_path)
    config.model = replace(config.model, attention_degree=4, attention_backend="blocked",
        attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_sharing="per_head", attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        initialization="transformer_truncated_reference")
    model = build_model(config.model, 7)
    model.query_polynomial_kernel[...] = jnp.zeros_like(model.query_polynomial_kernel[...])
    model.query_polynomial_bias[...] = jnp.ones_like(model.query_polynomial_bias[...])
    model.query_polynomial_amplitude[...] = jnp.ones_like(model.query_polynomial_amplitude[...])
    model.qk_norm_head_log_scale[...] = jnp.asarray([[[.4], [-.2]]])
    graph, params = nnx.split(model, nnx.Param)
    tokens = jnp.arange(20).reshape(4, 5) % 16
    inspect, compare = make_query_inspector(graph, np.asarray([0, 3]))
    parity = jax.device_get(compare(params, tokens))
    assert float(parity["logit_max_absolute_error"]) < 1e-6
    losses, stats, norms, activation, coefficients = jax.device_get(inspect(params, tokens))
    native = make_native_evaluator(graph)(params, tokens)
    np.testing.assert_allclose(losses, native, atol=1e-6)
    assert norms.shape == (1, 4, 2, 2)
    np.testing.assert_allclose(norms, np.sqrt(5), atol=1e-6)
    np.testing.assert_allclose(coefficients, 1., atol=1e-6)
    np.testing.assert_array_equal(stats["saturated_099_fraction"], 0.)
    np.testing.assert_array_equal(stats["tanh_derivative_mean"], 1.)
    np.testing.assert_array_equal(stats["cancellation_ratio_mean"], 1.)
    assert stats["direction_bias_jacobian_mean_square"].shape == (1, 2, 5)
    assert "layer_00/attention_output_head_rms_min" in activation
    assert float(activation["layer_00/head_rms_norm_gain_max"]) > 0
    assert 0 <= float(activation["layer_00/head_mean_square_lt_1eps_fraction_mean"]) <= 1
    generator, frozen = params.split(lambda path, value: str(path[-1]).startswith("query_polynomial"), ...)
    merged = nnx.merge_state(frozen, generator)
    for a, b in zip(jax.tree.leaves(merged), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)


def test_native_hooks_capture_nontrivial_bfloat16_generator_without_reprojection(tmp_path):
    """Nonzero W, signed b/a, two causal layers, and the real BF16 module policy."""
    config = tiny_config(tmp_path)
    config.model = replace(config.model, num_layers=2, attention_degree=4, attention_backend="blocked",
        attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_scale="factorial", attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_sharing="per_head", attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        initialization="transformer_truncated_reference")
    model = build_model(config.model, 51)
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = jnp.bfloat16
    rng = np.random.default_rng(34)
    for name, scale in (("query_polynomial_kernel", .7), ("query_polynomial_bias", .2),
                        ("query_polynomial_amplitude", .4)):
        parameter = getattr(model, name)
        parameter[...] = jnp.asarray(rng.normal(size=parameter.shape) * scale, jnp.float32)
    graph, params = nnx.split(model, nnx.Param)
    tokens = jnp.asarray(rng.integers(0, 16, size=(4, 5)))
    positions = np.asarray([0, 2, 3])
    inspect, compare = make_query_inspector(graph, positions)
    parity = jax.device_get(compare(params, tokens))
    assert float(parity["logit_max_absolute_error"]) == 0.
    losses, stats, norms, activation, coordinates = jax.device_get(inspect(params, tokens))
    np.testing.assert_allclose(losses, make_native_evaluator(graph)(params, tokens), atol=1e-6)

    @jax.jit
    def native_coefficients(state, inputs):
        return nnx.merge(graph, state)(inputs[:, :-1], deterministic=True, return_coefficients=True)[1]

    native = np.asarray(native_coefficients(params, tokens))
    expected = np.take(native * np.asarray([1, 1, 2, 6, 24]), positions, axis=2)
    np.testing.assert_allclose(coordinates, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(norms, np.linalg.norm(expected, axis=-1), atol=1e-6)
    assert np.max(stats["saturated_099_fraction"]) > 0
    assert np.min(stats["tanh_derivative_mean"]) < .5
    assert len(activation) == 10
    assert not any("residual" in name for name in activation)
    native_loss = make_native_evaluator(graph)
    repeated = bounded_replay_control(graph, params, np.asarray(tokens), 2, positions,
        native_loss, np.asarray(native_loss(params, tokens)), losses)
    assert repeated["microbatch_sequence_count"] == 2
    assert all(item["all_returned_arrays_repeat_exactly"] for item in repeated["anchor_forwards"].values())
    assert all(item["all_returned_arrays_repeat_exactly"] for item in repeated["materialized_logit_forwards"].values())
    assert repeated["within_compiled_native_hook_logit_parity"]["logit_max_absolute_error"] == 0.
