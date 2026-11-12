"""Sampling, causal score summaries, and represented coefficient capture."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.transformer_pp.calibrate_spline_scores import (
    _accumulate, _self_check, effective_coefficients, layer_statistics, sample_indices, summarize,
)


def test_statistics_match_independent_dense_numpy_reference():
    assert _self_check()["status"] == "passed"


def test_causal_pair_sampling_is_reproducible_and_uniform():
    rows, q, k = sample_indices(7, 4, 100_000, 123, 0)
    for left, right in zip((rows, q, k), sample_indices(7, 4, 100_000, 123, 0)):
        np.testing.assert_array_equal(left, right)
    assert len(set(rows.tolist())) == 4
    assert np.all((0 <= k) & (k <= q) & (q < 7))
    flattened = q * (q + 1) // 2 + k
    counts = np.bincount(flattened, minlength=28)
    # All causal pairs, not query positions, have the same sampling probability.
    np.testing.assert_allclose(counts / len(q), np.full(28, 1 / 28), atol=0.002)
    assert not np.array_equal(q, sample_indices(7, 4, 100_000, 123, 1)[1])


def test_static_and_query_capture_give_identical_effective_coefficients():
    q = jnp.ones((1, 5, 2, 4))
    coefficients = jnp.array([[1, -0.5, 0.2], [1, 0.5, 0.3]], jnp.float32)
    common = {"attention_degree": 2, "attention_normalization": False}
    static = effective_coefficients(q, {**common, "polynomial_coefficients": coefficients}, "static")
    query = effective_coefficients(q, {**common, "query_polynomial_coefficients":
                                         jnp.broadcast_to(coefficients, (1, 5, 2, 3))}, "query")
    np.testing.assert_array_equal(static, query)
    with pytest.raises(ValueError, match="without FoX"):
        effective_coefficients(q, {**common, "polynomial_coefficients": coefficients,
                                   "query_log_decay": jnp.zeros((1, 5, 2))}, "static")


def test_histogram_endpoints_causal_mask_and_tail_summaries():
    # Scores are [[1,2,3],[2,4,6],[3,6,9]]; only the lower triangle counts.
    q = jnp.arange(1, 4, dtype=jnp.float32).reshape(3, 1, 1)
    coefficients = jnp.broadcast_to(jnp.array([1.0, 0.0]), (3, 1, 2))
    radii, edges = np.array([2.0, 4.0], np.float32), np.array([1, 2, 4], np.float32)
    rows = np.array([0, 1, 2], np.int32)
    pq, pk = np.array([0, 1, 2], np.int32), np.array([0, 1, 2], np.int32)
    values = jax.device_get(layer_statistics(q, q, coefficients, rows, pq, pk, radii, edges))
    accumulator = {}
    _accumulate(accumulator, values)
    _accumulate(accumulator, values)
    result = list(summarize(accumulator, quantiles=[0.0, 0.5, 1.0], radii=radii))[0]
    assert result["score_count"] == 12
    assert result["histogram_counts"] == [2, 6]
    assert result["histogram_overflow"] == 4
    assert result["histogram_underflow"] == 0
    assert result["score_mean"] == pytest.approx(25 / 6)
    assert result["score_quantiles"] == {"0.0": 1.0, "0.5": 4.0, "1.0": 9.0}
    assert result["query_absmax_quantiles"] == {"0.0": 1.0, "0.5": 4.0, "1.0": 9.0}
    assert result["radius_statistics"][0]["causal_pair_fraction_outside"] == pytest.approx(4 / 6)
    assert result["radius_statistics"][0]["mean_query_absolute_weight_share_outside"] == pytest.approx(0.5)
    assert result["radius_statistics"][1]["mean_query_absolute_weight_share_outside"] == pytest.approx(2 / 9)


def test_zero_weight_rows_are_counted_and_never_report_nan_shares():
    q = jnp.ones((3, 1, 2), jnp.float32)
    indices = sample_indices(3, 3, 4, 42, 0)
    values = jax.device_get(layer_statistics(q, q, jnp.zeros((3, 1, 2)), *indices,
                                            jnp.array([1.0]), jnp.array([-4.0, 4.0])))
    accumulator = {}
    _accumulate(accumulator, values)
    result = list(summarize(accumulator, quantiles=[0.5], radii=[1.0]))[0]
    assert result["zero_weight_query_count"] == 3
    assert result["radius_statistics"][0]["mean_query_absolute_weight_share_outside"] == 0.0
