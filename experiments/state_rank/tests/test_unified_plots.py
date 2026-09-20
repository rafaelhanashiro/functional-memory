"""Inspect plotted scientific curves without expensive full-size exports."""

from pathlib import Path

import numpy as np
import pytest

from experiments.state_rank.plot_comparison import Condition
from experiments.state_rank.unified_metrics import ModelAnalysis, RANK_METRICS
from experiments.state_rank import unified_plots


def make_analysis(key="llama", label="Llama", width=64, layers=(0, 2, 4), depth=5):
    layers = np.array(layers)
    conditions = []
    shape = (len(layers), 2, 5, 8)
    # Every axis carries a distinguishable signal, including nonmonotonic time.
    signals = (np.arange(len(layers))[:, None, None, None] * 5
               + np.arange(2)[None, :, None, None] * 2
               + np.array([0, 3, 1, 4, 2])[None, None, :, None]
               + np.arange(8)[None, None, None, :])
    for condition_index, seed in enumerate([None, 0, 1, 2, 3, 4]):
        weights = "pretrained" if seed is None else "random"
        name = "pretrained" if seed is None else f"random-seed{seed}"
        arrays = {
            "layers": layers.copy(), "head_indices": np.arange(8), "prefixes": np.arange(1, 6),
            "spectrum_prefixes": np.array([1, 3, 5]), "token_ids": np.array([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]]),
            "normalized_eigenvalues": np.full((len(layers), 2, 3, 8, width), 1.0 / width),
            "log_trace": np.zeros(shape),
        }
        for metric_index, (metric, _) in enumerate(RANK_METRICS):
            arrays[metric] = (1 + condition_index * 2 + signals) * (metric_index + 1) / 4
        conditions.append(Condition(name, weights, seed, {"model_config": {"num_hidden_layers": depth}}, arrays))
    return ModelAnalysis(key, label, Path("unused"), {}, conditions, int(layers[1]), [
        {"title": "First saved article"}, {"title": "Second saved article"},
    ])


@pytest.fixture
def capture_figures(monkeypatch):
    figures = {}
    monkeypatch.setattr(unified_plots, "_save_figure", lambda figure, output, name: figures.setdefault(name, figure))
    return figures


def visible_lines(axis):
    return {line.get_label(): line for line in axis.lines if not line.get_label().startswith("_")}


def test_heads_seeds_and_selected_layer_remain_separate(tmp_path, capture_figures):
    analysis = make_analysis()
    paths = unified_plots.plot_analysis(analysis, tmp_path)
    assert len(paths) == 24  # Six networks, separate head/layer figures, PNG and PDF.
    assert len(set(paths)) == len(paths)
    for condition in analysis.conditions:
        figure = capture_figures[f"llama/heads/aggregate/{condition.name}"]
        assert "layer index 2" in figure._suptitle.get_text()
        if condition.seed is not None:
            assert f"seed {condition.seed}" in figure._suptitle.get_text()
        for axis, (metric, _) in zip(figure.axes, RANK_METRICS, strict=True):
            lines = visible_lines(axis)
            assert list(lines) == [f"KV head {head}" for head in range(8)]
            assert len({line.get_color() for line in lines.values()}) == 8
            expected = condition.arrays[metric][1].mean(axis=0)
            for head in range(8):
                np.testing.assert_array_equal(lines[f"KV head {head}"].get_xdata(), analysis.prefixes)
                np.testing.assert_array_equal(lines[f"KV head {head}"].get_ydata(), expected[:, head])
            # Band is the sequence min/max for this head, never a seed envelope.
            vertices = axis.collections[0].get_paths()[0].vertices
            for token_index, prefix in enumerate(analysis.prefixes):
                observed = vertices[np.isclose(vertices[:, 0], prefix), 1]
                expected_values = condition.arrays[metric][1, :, token_index, 0]
                assert observed.min() == expected_values.min()
                assert observed.max() == expected_values.max()
    reference = capture_figures["llama/heads/aggregate/pretrained"]
    for condition in analysis.conditions[1:]:
        current = capture_figures[f"llama/heads/aggregate/{condition.name}"]
        assert [axis.get_ylim() for axis in current.axes] == [axis.get_ylim() for axis in reference.axes]


def test_single_input_uses_exact_head_curves_without_bands_or_layer_exports(tmp_path, capture_figures):
    analysis = make_analysis()
    paths = unified_plots.plot_analysis(analysis, tmp_path, sequence_index=1)
    assert len(paths) == 12
    assert all("/heads/sequence-2/" in path for path in paths)
    assert all("/layers/" not in name for name in capture_figures)
    for condition in analysis.conditions:
        figure = capture_figures[f"llama/heads/sequence-2/{condition.name}"]
        assert "Input 2: Second saved article" in figure._suptitle.get_text()
        for axis, (metric, _) in zip(figure.axes, RANK_METRICS, strict=True):
            assert not axis.collections
            for head, line in enumerate(visible_lines(axis).values()):
                np.testing.assert_array_equal(line.get_ydata(), condition.arrays[metric][1, 1, :, head])


def test_layer_figures_keep_every_token_and_every_layer(tmp_path, capture_figures):
    analysis = make_analysis()
    unified_plots.plot_analysis(analysis, tmp_path)
    for condition in analysis.conditions:
        figure = capture_figures[f"llama/layers/{condition.name}"]
        assert "all 3 measured layers" in figure._suptitle.get_text()
        assert "mean across 2 inputs and 8 heads" in figure._suptitle.get_text()
        assert figure.axes[-1].get_ylabel() == "Layer index (zero based)"
        for axis, (metric, _) in zip(figure.axes[:4], RANK_METRICS, strict=True):
            lines = visible_lines(axis)
            assert list(lines) == ["Layer 0", "Layer 2", "Layer 4"]
            expected = condition.arrays[metric].mean(axis=(1, 3))
            for row, layer in enumerate(analysis.layers):
                line = lines[f"Layer {layer}"]
                np.testing.assert_array_equal(line.get_xdata(), analysis.prefixes)
                np.testing.assert_array_equal(line.get_ydata(), expected[row])
                assert len(np.unique(line.get_ydata())) > 1


def test_depth_views_use_terminal_values_real_depth_and_each_state_width(tmp_path, capture_figures):
    llama = make_analysis()
    deltanet = make_analysis("gdn", "Gated DeltaNet", width=256, layers=(0, 1, 5), depth=7)
    paths = unified_plots.plot_depth_comparison([llama, deltanet], tmp_path)
    assert paths == ["depth_counts.png", "depth_counts.pdf", "depth_fractional.png", "depth_fractional.pdf"]
    absolute = capture_figures["depth_counts"]
    fractional = capture_figures["depth_fractional"]
    assert len(absolute.axes) == 8 and len(fractional.axes) == 4
    for metric_index, (metric, _) in enumerate(RANK_METRICS):
        fractional_lines = visible_lines(fractional.axes[metric_index])
        assert len(fractional_lines) == 4
        for column, analysis in enumerate([llama, deltanet]):
            expected = analysis.conditions[0].arrays[metric][:, :, -1].mean(axis=(1, 2))
            random = np.stack([condition.arrays[metric][:, :, -1].mean(axis=(1, 2))
                               for condition in analysis.conditions[1:]])
            axis = absolute.axes[metric_index * 2 + column]
            lines = visible_lines(axis)
            np.testing.assert_array_equal(lines["pretrained"].get_xdata(), analysis.layers)
            np.testing.assert_array_equal(lines["pretrained"].get_ydata(), expected)
            np.testing.assert_array_equal(lines["random mean (5 seeds)"].get_ydata(), random.mean(axis=0))
            line = fractional_lines[f"{analysis.label} pretrained"]
            np.testing.assert_array_equal(line.get_xdata(), analysis.layers / (analysis.model_depth - 1))
            np.testing.assert_array_equal(line.get_ydata(), expected / analysis.width)
            np.testing.assert_array_equal(fractional_lines[f"{analysis.label} random mean (5 seeds)"].get_ydata(),
                                          (random / analysis.width).mean(axis=0))
        assert absolute.axes[metric_index * 2].get_ylim() == absolute.axes[metric_index * 2 + 1].get_ylim()


@pytest.mark.parametrize("index", [-1, 2, True, 0.5])
def test_invalid_input_selection_fails_before_plotting(tmp_path, index):
    with pytest.raises(ValueError, match="sequence_index"):
        unified_plots.plot_analysis(make_analysis(), tmp_path, sequence_index=index)


def test_depth_export_freezes_layout_and_writes_both_formats(tmp_path):
    # One real bounded export exercises constrained layout, outside legends,
    # frozen PNG/PDF rendering and output paths without writing retained data.
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(3, 2), constrained_layout=True)
    axis.plot([0, 1], [1, 2], label="Head 0")
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center")
    paths = unified_plots._save(figure, tmp_path, "nested/smoke")
    assert paths == ["nested/smoke.png", "nested/smoke.pdf"]
    assert all((tmp_path / path).stat().st_size > 1000 for path in paths)
    assert figure.get_layout_engine().__class__.__name__ == "PlaceHolderLayoutEngine"
    plt.close(figure)
