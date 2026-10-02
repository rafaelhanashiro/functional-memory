"""Behavioral checks for the combined, common-scale heatmap."""

import matplotlib.colors as colors
import numpy as np
import pytest

from results.attention_scaling.render import _build_figure, render_heatmap


def _config():
    widths = [8, 16, 32, 64, 128]
    specifications = [
        ("normalized_fixed", 6), ("normalized_learned", 4), ("references", 3),
        ("unnormalized_fixed", 6), ("unnormalized_learned", 6),
    ]
    return {
        "d_k": widths,
        "panels": [{"id": name, "title": name.replace("_", " ").title(),
                    "columns": [{"id": f"c{i}", "label": f"column {i}"}
                                for i in range(count)]}
                   for name, count in specifications],
    }


def _row(panel, column, width, status="not_run", value=None, failure_step=None, legacy=False):
    return {"panel": panel, "column": column, "d_k": width, "status": status,
            "terminal_ppl": value, "failure_step": failure_step, "legacy": legacy}


def _axes(figure):
    return {axis.get_gid().removeprefix("panel:"): axis for axis in figure.axes
            if axis.get_gid() and axis.get_gid().startswith("panel:")}


def test_cell_placement_and_one_linear_scale_across_all_panels():
    config = _config()
    rows = [
        _row("unnormalized_learned", "c5", 128, "complete", 31.25),
        _row("normalized_fixed", "c0", 8, "complete", 12.5),
        _row("references", "c1", 64, "complete", 17.125),
    ]
    figure = _build_figure(list(reversed(rows)), config)
    try:
        axes = _axes(figure)
        assert list(axes) == [panel["id"] for panel in config["panels"]]
        assert axes["normalized_fixed"].images[0].get_array().shape == (5, 6)
        assert axes["normalized_learned"].images[0].get_array().shape == (5, 4)
        assert axes["references"].images[0].get_array().shape == (5, 3)
        assert axes["unnormalized_learned"].images[0].get_array()[4, 5] == 31.25
        assert axes["normalized_fixed"].images[0].get_array()[0, 0] == 12.5
        assert axes["references"].images[0].get_array()[3, 1] == 17.125
        assert np.ma.is_masked(axes["references"].images[0].get_array()[0, 0])
        norms = [axis.images[0].norm for axis in axes.values()]
        assert all(norm is norms[0] for norm in norms)
        assert isinstance(norms[0], colors.Normalize)
        assert norms[0].vmin == 12.5
        assert norms[0].vmax == 31.25
        assert figure.axes[-1].get_gid() == "colorbar"
        assert axes["unnormalized_fixed"].get_position().y0 < axes["references"].get_position().y0
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_failure_incomplete_and_missing_cells_have_distinct_marks():
    config = _config()
    rows = [
        _row("normalized_fixed", "c0", 8, "numerical_failure", failure_step=12_345),
        _row("normalized_fixed", "c1", 8, "incomplete"),
        _row("normalized_fixed", "c2", 8, "not_run"),
        # c3 is absent altogether; the renderer must still mark its missing run.
        _row("normalized_fixed", "c4", 8, "complete", 14.125),
    ]
    figure = _build_figure(rows, config)
    try:
        axis = _axes(figure)["normalized_fixed"]
        labels = {(int(text.get_position()[0]), int(text.get_position()[1])): text.get_text()
                  for text in axis.texts}
        assert labels[(0, 0)] == "FAIL\n@12,345"
        assert labels[(1, 0)] == "INCOMPLETE"
        assert labels[(2, 0)] == labels[(3, 0)] == "NOT\nRUN"
        assert labels[(4, 0)] == "14.125"
        patches = {int(round(patch.get_x() + 0.5)): patch for patch in axis.patches
                   if patch.get_y() == -0.5}
        assert len({patches[index].get_facecolor() for index in (0, 1, 2)}) == 3
        assert patches[2].get_facecolor() == patches[3].get_facecolor()
        assert len({patches[index].get_hatch() for index in (0, 1, 2)}) == 3
        assert len(figure.legends) == 1
        # Without a legacy cell, neither the legend nor the subtitle mentions outlines.
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Numerical failure (step shown)", "Not run", "Incomplete"]
        assert not any("Outlined" in text.get_text() for text in figure.texts)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_legacy_runs_keep_their_color_and_get_an_outline():
    config = _config()
    rows = [
        _row("references", "c0", 8, "complete", 12.5, legacy=True),
        _row("references", "c1", 8, "complete", 13.5),
        _row("references", "c2", 16, "numerical_failure", failure_step=7, legacy=True),
    ]
    figure = _build_figure(rows, config)
    try:
        axis = _axes(figure)["references"]
        outlines = [patch for patch in axis.patches if patch.get_gid() == "legacy"]
        assert [(round(patch.get_x() + 0.43), round(patch.get_y() + 0.43)) for patch in outlines] == [(0, 0), (2, 1)]
        assert all(not patch.get_fill() for patch in outlines)
        # The outline does not replace the perplexity color: the legacy cell is still drawn from the grid.
        assert axis.images[0].get_array()[0, 0] == 12.5
        assert figure.legends[0].get_texts()[-1].get_text() == (
            "Legacy run: decay on all parameters, no QK-norm (to be rerun)")
        assert any("Outlined cells" in text.get_text() for text in figure.texts)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
    with pytest.raises(ValueError, match="cannot be a legacy run"):
        _build_figure([_row("references", "c0", 8, "not_run", legacy=True)], config)


def test_png_and_pdf_regenerate_byte_identically(tmp_path):
    config = _config()
    rows = [_row("normalized_fixed", "c0", 8, "complete", 12.5),
            _row("references", "c1", 32, "numerical_failure", failure_step=42)]
    paths = render_heatmap(rows, config, tmp_path)
    assert set(paths) == {"combined_terminal_ppl_png", "combined_terminal_ppl_pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    first = {name: path.read_bytes() for name, path in paths.items()}
    second = render_heatmap(rows, config, tmp_path)
    assert {name: path.read_bytes() for name, path in second.items()} == first


def test_caption_follows_the_subtitle():
    config = dict(_config(), caption="References use their reference initializations.")
    figure = _build_figure([], config)
    try:
        assert any(text.get_text().endswith("\nReferences use their reference initializations.")
                   for text in figure.texts)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_duplicate_coordinates_are_rejected():
    config = _config()
    row = _row("references", "c0", 8, "complete", 12.5)
    with pytest.raises(ValueError, match="Duplicate heatmap coordinate"):
        _build_figure([row, row], config)
