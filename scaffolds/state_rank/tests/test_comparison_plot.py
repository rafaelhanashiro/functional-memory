import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scaffolds.state_rank.plot_comparison import (
    Condition, FIGURE_NAMES, SCALAR_METRICS, aggregate_condition, load_conditions,
    plot_comparison, summary_rows,
)


def fixture_arrays(value=2.):
    arrays = {
        "layers": np.array([0, 3]), "head_indices": np.array([0, 1]),
        "prefixes": np.array([1, 2, 3]), "spectrum_prefixes": np.array([1, 3]),
        "token_ids": np.array([[1, 2, 3], [4, 5, 6]]),
    }
    for key in SCALAR_METRICS:
        arrays[key] = np.full((2, 2, 3, 2), value, dtype=float)
    arrays["normalized_eigenvalues"] = np.broadcast_to(
        np.array([.8, .2]), (2, 2, 2, 2, 2)).copy()
    return arrays


def write_condition(root, name, *, value=2., status="complete", arrays=None):
    directory = root / name
    directory.mkdir()
    (directory / "metadata.json").write_text(json.dumps({"status": status}))
    if status == "complete":
        np.savez_compressed(directory / "metrics.npz", **(fixture_arrays(value) if arrays is None else arrays))


def make_manifest(root, entries):
    manifest = {"schema": 1, "config": {"comparison_layer": 8, "model_id": "tiny-fixture"}, "conditions": entries}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_singular_ratios_and_rank_metrics_are_aggregated_after_per_head_computation():
    arrays = fixture_arrays()
    arrays["normalized_eigenvalues"][..., 1, :] = [.5, .5]
    arrays["participation_ratio"][..., 0] = 1. / (.8**2 + .2**2)
    arrays["participation_ratio"][..., 1] = 2.
    arrays["key_norms"][..., 0] = 3.
    arrays["key_norms"][..., 1] = 4.
    condition = Condition("random-seed0", "random", 0, {}, arrays)
    curves = aggregate_condition(condition)
    np.testing.assert_allclose(curves["relative_singular_values"], np.broadcast_to([1., .75], (2, 2, 2)))
    np.testing.assert_allclose(curves["participation_ratio"], (1. / .68 + 2.) / 2)
    np.testing.assert_allclose(curves["key_l2_rms"], np.sqrt((3.**2 + 4.**2) / 2))
    assert not np.isclose(curves["relative_singular_values"][0, 0, 1], np.sqrt(.35 / .65))
    assert len(summary_rows([condition])) == 2


def test_complete_conditions_make_four_figure_groups_and_partial_status_report(tmp_path):
    entries = [
        {"name": "pretrained", "weights": "pretrained", "seed": None},
        {"name": "random-seed0", "weights": "random", "seed": 0},
        {"name": "random-seed1", "weights": "random", "seed": 1},
        {"name": "random-seed2", "weights": "random", "seed": 2},
    ]
    make_manifest(tmp_path, entries)
    write_condition(tmp_path, "pretrained", value=3.)
    write_condition(tmp_path, "random-seed0", value=2.)
    write_condition(tmp_path, "random-seed1", status="running")
    result = plot_comparison(tmp_path)
    assert result == {"complete_conditions": 2, "planned_conditions": 4, "comparison_layer": 3, "summary_rows": 4}
    for name in FIGURE_NAMES:
        for extension in ("png", "pdf"):
            assert (tmp_path / f"{name}.{extension}").stat().st_size > 1000
    with (tmp_path / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and {row["condition"] for row in rows} == {"pretrained", "random-seed0"}
    assert rows[0]["final_participation_ratio_mean"] == "3.0"
    report = (tmp_path / "README.md").read_text()
    assert "random-seed1 | random | 1 | running" in report
    assert "random-seed2 | random | 2 | pending" in report
    assert "not a confidence interval" in report
    assert "not per-component RMS" in report


def test_pending_only_produces_status_and_empty_summary(tmp_path):
    make_manifest(tmp_path, [{"name": "random-seed0", "weights": "random", "seed": 0}])
    result = plot_comparison(tmp_path)
    assert result["complete_conditions"] == 0
    assert result["comparison_layer"] is None
    assert "No complete measurements" in (tmp_path / "README.md").read_text()
    assert len((tmp_path / "summary.csv").read_text().splitlines()) == 1
    assert not list(tmp_path.glob("*.png"))


def test_mismatched_inputs_cannot_silently_enter_comparison(tmp_path):
    make_manifest(tmp_path, [
        {"name": "pretrained", "weights": "pretrained", "seed": None},
        {"name": "random-seed0", "weights": "random", "seed": 0},
    ])
    write_condition(tmp_path, "pretrained")
    arrays = fixture_arrays()
    arrays["token_ids"][0, 0] = 99
    write_condition(tmp_path, "random-seed0", arrays=arrays)
    with pytest.raises(ValueError, match="token_ids differs"):
        load_conditions(tmp_path)


def test_zero_state_spectra_are_zero_and_missing_complete_data_is_rejected(tmp_path):
    arrays = fixture_arrays(0.)
    arrays["normalized_eigenvalues"][:] = 0
    arrays["log_trace"][:] = -np.inf
    curves = aggregate_condition(Condition("random-seed0", "random", 0, {}, arrays))
    assert np.all(curves["relative_singular_values"] == 0)
    assert np.all(np.isneginf(curves["log_trace"]))
    make_manifest(tmp_path, [{"name": "pretrained", "weights": "pretrained", "seed": None}])
    write_condition(tmp_path, "pretrained")
    (tmp_path / "pretrained" / "metrics.npz").unlink()
    with pytest.raises(ValueError, match="no metrics.npz"):
        load_conditions(tmp_path)


def test_random_band_represents_seed_means_and_pretrained_is_separate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scaffolds.state_rank.plot_comparison import _plot_comparison

    conditions = [Condition("r0", "random", 0, {}, {}),
                  Condition("r1", "random", 1, {}, {}),
                  Condition("p", "pretrained", None, {}, {})]
    values = {"r0": np.array([1., 3.]), "r1": np.array([5., 7.]), "p": np.array([9., 11.])}
    figure, axis = plt.subplots()
    try:
        _plot_comparison(axis, np.array([1, 2]), conditions, values)
        lines = {line.get_label(): line for line in axis.lines}
        np.testing.assert_array_equal(lines["Random mean (2 seeds)"].get_ydata(), [3., 5.])
        np.testing.assert_array_equal(lines["Pretrained mean"].get_ydata(), [9., 11.])
        vertices = axis.collections[0].get_paths()[0].vertices
        for x, expected in ((1, (1., 5.)), (2, (3., 7.))):
            y = vertices[vertices[:, 0] == x, 1]
            assert (y.min(), y.max()) == expected
    finally:
        plt.close(figure)


def test_native_recurrent_states_plot_without_fabricating_vector_norms(tmp_path):
    make_manifest(tmp_path, [{"name": "pretrained", "weights": "pretrained", "seed": None}])
    arrays = fixture_arrays()
    arrays.pop("key_norms")
    arrays.pop("value_norms")
    write_condition(tmp_path, "pretrained", arrays=arrays)
    result = plot_comparison(tmp_path)
    assert result["summary_rows"] == 2
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert "final_log_trace_mean" in rows[0]
    assert not any("key" in key or "value_l2" in key for key in rows[0])
    assert "vector norms were not collected" in (tmp_path / "README.md").read_text()
    for name in FIGURE_NAMES:
        assert (tmp_path / f"{name}.png").is_file()


def test_comparison_rejects_inconsistent_optional_norms(tmp_path):
    make_manifest(tmp_path, [
        {"name": "pretrained", "weights": "pretrained", "seed": None},
        {"name": "random-seed0", "weights": "random", "seed": 0},
    ])
    write_condition(tmp_path, "pretrained")
    arrays = fixture_arrays()
    arrays.pop("key_norms")
    write_condition(tmp_path, "random-seed0", arrays=arrays)
    with pytest.raises(ValueError, match="both be present"):
        load_conditions(tmp_path)
    arrays.pop("value_norms")
    np.savez_compressed(tmp_path / "random-seed0/metrics.npz", **arrays)
    with pytest.raises(ValueError, match="norm availability differs"):
        load_conditions(tmp_path)


def test_completed_archive_provenance_and_checksum_are_verified(tmp_path):
    entries = [{"name": "pretrained", "weights": "pretrained", "seed": None}]
    manifest = make_manifest(tmp_path, entries)
    manifest["identity"] = "fixed-identity"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    write_condition(tmp_path, "pretrained")
    path = tmp_path / "pretrained/metadata.json"
    metadata = {"status": "complete", "identity": manifest["identity"], "weights": "pretrained", "seed": None,
                "metrics_sha256": hashlib.sha256((tmp_path / "pretrained/metrics.npz").read_bytes()).hexdigest()}
    path.write_text(json.dumps(metadata))
    assert len(load_conditions(tmp_path)[1]) == 1
    for field, value, message in (("identity", "wrong", "identity"), ("seed", 2, "seed"),
                                  ("metrics_sha256", "wrong", "checksum")):
        path.write_text(json.dumps({**metadata, field: value}))
        with pytest.raises(ValueError, match=message):
            load_conditions(tmp_path)
