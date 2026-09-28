"""Behavioral checks for the degree chart: placement, legacy marks, failures, references."""

import matplotlib.pyplot as plt
import pytest

from results.attention_scaling.chart import _build_facet_figure, _build_figure, _spread, render_degree_chart


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


def _facet_config():
    degrees = [{"id": f"m{d}", "label": f"m={d}"} for d in (1, 2, 3, 4)]
    return {"d_k": [64], "facets": [{"id": "normalized", "title": "Row-normalized"},
                                    {"id": "unnormalized", "title": "Unnormalized"}],
            "panels": [
                {"id": "normalized_fixed_taylor", "title": "Taylor", "family": "taylor", "facet": "normalized", "columns": degrees},
                {"id": "normalized_fixed_hermite", "title": "Hermite", "family": "hermite", "facet": "normalized", "columns": degrees},
                {"id": "unnormalized_fixed", "title": "Taylor", "family": "taylor", "facet": "unnormalized", "columns": degrees},
                {"id": "unnormalized_fixed_hermite", "title": "Hermite", "family": "hermite", "facet": "unnormalized", "columns": degrees},
                {"id": "references", "title": "References", "columns": [{"id": "softmax", "label": "QK-norm softmax"}]},
            ]}


FACET_ROWS = [
    _row("normalized_fixed_taylor", "m1", "numerical_failure", failure_step=31),
    _row("normalized_fixed_taylor", "m2", value=16.5),
    _row("normalized_fixed_taylor", "m3", "numerical_failure", failure_step=419),
    _row("normalized_fixed_taylor", "m4", value=14.9),
    *(_row("normalized_fixed_hermite", f"m{d}", "numerical_failure", failure_step=step)
      for d, step in ((1, 5), (2, 3829), (3, 802), (4, 1390))),
    *(_row("unnormalized_fixed", f"m{d}", value=v) for d, v in ((1, 20.3), (2, 21.7), (3, 17.6), (4, 18.2))),
    *(_row("unnormalized_fixed_hermite", f"m{d}", value=v) for d, v in ((1, 20.9), (2, 17.5), (3, 17.9), (4, 17.2))),
    _row("references", "softmax", value=14.46),
]


def test_facets_separate_normalization_and_never_bridge_a_failed_degree():
    figure = _build_facet_figure(FACET_ROWS, _facet_config())
    try:
        by_gid = {axis.get_gid(): axis for axis in figure.axes}
        normalized, unnormalized = by_gid["chart:normalized"], by_gid["chart:unnormalized"]
        assert {"point:normalized:taylor:m2", "point:normalized:taylor:m4",
                "reference:normalized:softmax"} <= _gids(normalized)
        assert not any(gid.startswith("point:normalized:hermite") for gid in _gids(normalized))
        assert {f"failure:normalized:hermite:m{d}" for d in (1, 2, 3, 4)} <= _gids(by_gid["chart:normalized:failures"])
        assert {"failure:normalized:taylor:m1", "failure:normalized:taylor:m3"} <= _gids(by_gid["chart:normalized:failures"])
        assert not _gids(by_gid["chart:unnormalized:failures"])
        connectors = lambda axis: [line for line in axis.get_lines() if line.get_gid() is None and len(line.get_xdata()) == 2]
        assert connectors(normalized) == []  # m2 and m4 are separated by the failed m3
        assert len(connectors(unnormalized)) == 6
        assert normalized.get_ylim() == unnormalized.get_ylim()
    finally:
        plt.close(figure)


def test_facet_figure_rejects_unstyled_family_or_orphan_facet():
    config = _facet_config()
    config["panels"][0]["family"] = "legendre"
    with pytest.raises(ValueError, match="No style"):
        _build_facet_figure(FACET_ROWS, config)
    config = _facet_config()
    config["facets"].append({"id": "empty", "title": "?"})
    with pytest.raises(ValueError, match="declared facet"):
        _build_facet_figure(FACET_ROWS, config)
