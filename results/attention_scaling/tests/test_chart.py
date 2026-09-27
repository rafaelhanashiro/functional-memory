"""Behavioral checks for the degree chart: placement, legacy marks, failures, references."""

import matplotlib.pyplot as plt
import pytest

from results.attention_scaling.chart import _build_figure, _spread, render_degree_chart


def _config():
    degrees = lambda *ds: [{"id": f"m{d}", "label": f"m={d}"} for d in ds]
    return {"d_k": [64], "panels": [
        {"id": "normalized_learned_taylor", "title": "Normalized, learned", "columns": degrees(1, 2, 3, 4)},
        {"id": "normalized_nonnegative", "title": "Normalized, learned nonnegative", "columns": degrees(2, 4)},
        {"id": "unnormalized_learned", "title": "Unnormalized, learned", "columns": degrees(1, 2, 3, 4)},
        {"id": "references", "title": "References", "columns": [
            {"id": "softmax", "label": "QK-norm softmax"}, {"id": "unnormalized_exp", "label": "unnormalized exp"}]},
    ]}


def _row(panel, column, status="complete", value=None, failure_step=None, legacy=False):
    return {"panel": panel, "column": column, "d_k": 64, "status": status, "terminal_ppl": value,
            "failure_step": failure_step, "legacy": legacy}


ROWS = [
    _row("normalized_learned_taylor", "m1", "numerical_failure", failure_step=27),
    _row("normalized_learned_taylor", "m3", "numerical_failure", failure_step=318),
    _row("normalized_nonnegative", "m2", value=16.2, legacy=True),
    _row("normalized_nonnegative", "m4", "not_run"),
    _row("unnormalized_learned", "m1", value=20.3),
    _row("unnormalized_learned", "m2", value=17.6),
    _row("references", "softmax", value=14.46),
    _row("references", "unnormalized_exp", value=17.46, legacy=True),
]


def _gids(axis):
    return {artist.get_gid() for artist in axis.get_children() if artist.get_gid()}


def test_points_legacy_marks_failures_and_references_are_placed_by_degree():
    figure = _build_figure(ROWS, _config())
    try:
        axes = {axis.get_gid(): axis for axis in figure.axes if axis.get_gid()}
        main, strip = axes["chart:perplexity"], axes["chart:failures"]
        gids = _gids(main)
        assert {"point:unnormalized_learned:m1:current", "point:unnormalized_learned:m2:current",
                "point:normalized_nonnegative:m2:legacy"} <= gids
        assert not any(gid.startswith("point:normalized_nonnegative:m4") for gid in gids)  # not run: no point
        assert {"reference:softmax", "reference:unnormalized_exp"} <= gids
        assert _gids(strip) >= {"failure:normalized_learned_taylor:m1", "failure:normalized_learned_taylor:m3"}
        legacy_point = next(line for line in main.get_lines() if line.get_gid() == "point:normalized_nonnegative:m2:legacy")
        assert legacy_point.get_markerfacecolor() == "#ffffff"
        texts = {text.get_text() for text in strip.texts}
        assert {"@27", "@318"} <= texts
    finally:
        plt.close(figure)


def test_unknown_family_and_coordinate_are_rejected():
    config = _config()
    config["panels"].append({"id": "mystery", "title": "?", "columns": [{"id": "m1", "label": "m=1"}]})
    with pytest.raises(ValueError, match="No style"):
        _build_figure(ROWS, config)
    with pytest.raises(ValueError, match="Unexpected chart coordinate"):
        _build_figure([*ROWS, _row("normalized_nonnegative", "m3", value=1.0)], _config())


def test_margin_labels_are_spread_apart():
    placed = _spread([14.46, 14.67, 16.2], gap=0.5)
    assert placed[1] - placed[0] >= 0.5 - 1e-9 and placed[2] - placed[1] >= 0.5 - 1e-9
    assert _spread([10.0, 20.0], gap=1.0) == [10.0, 20.0]


def test_png_and_pdf_regenerate_byte_identically(tmp_path):
    first = {name: path.read_bytes() for name, path in render_degree_chart(ROWS, _config(), tmp_path).items()}
    second = {name: path.read_bytes() for name, path in render_degree_chart(ROWS, _config(), tmp_path).items()}
    assert set(first) == {"degree_chart_png", "degree_chart_pdf"} and first == second
