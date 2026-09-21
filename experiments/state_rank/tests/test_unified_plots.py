"""Inspect grouped scientific curves without expensive full-size exports."""

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
    # Unmatched head permutations across seeds expose unintended feature matching.
    signals = (np.arange(len(layers))[:, None, None, None] * 5
               + np.arange(2)[None, :, None, None] * 2
               + np.array([0, 3, 1, 4, 2])[None, None, :, None])
    for condition_index, seed in enumerate([None, 0, 1, 2, 3, 4]):
        weights = "pretrained" if seed is None else "random"
        name = "pretrained" if seed is None else f"random-seed{seed}"
        head_values = np.arange(8) if seed is None else np.roll(np.arange(8), seed)
        arrays = {
            "layers": layers.copy(), "head_indices": np.arange(8), "prefixes": np.arange(1, 6),
            "spectrum_prefixes": np.array([1, 3, 5]), "token_ids": np.array([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]]),
            "normalized_eigenvalues": np.full((len(layers), 2, 3, 8, width), 1.0 / width),
            "log_trace": np.zeros(shape),
        }
        for metric_index, (metric, _) in enumerate(RANK_METRICS):
            arrays[metric] = (1 + condition_index * 2 + signals + head_values) * (metric_index + 1) / 4
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


def test_grouped_heads_average_inputs_then_seeds_without_matching_features(tmp_path, capture_figures):
    analysis = make_analysis()
    paths = unified_plots.plot_analysis(analysis, tmp_path)
    assert paths == ["llama/heads.png", "llama/heads.pdf", "llama/layers.png", "llama/layers.pdf"]
    figure = capture_figures["llama/heads"]
    assert len(figure.axes) == 8
    assert "layer index 2" in figure._suptitle.get_text()
    assert "positional head indices across seeds, not matched learned features" in figure._suptitle.get_text()
    assert "5 seeds" in figure.axes[1].get_title()
    for row, (metric, _) in enumerate(RANK_METRICS):
        pretrained_inputs = analysis.conditions[0].arrays[metric][1]
        random_seed_means = np.stack([condition.arrays[metric][1].mean(axis=0)
                                      for condition in analysis.conditions[1:]])
        for column, samples in enumerate((pretrained_inputs, random_seed_means)):
            axis = figure.axes[row * 2 + column]
            lines = visible_lines(axis)
            assert list(lines) == [f"KV head index {head}" for head in range(8)]
            assert len({line.get_color() for line in lines.values()}) == 8
            assert len(axis.lines) == 8 + (metric == "threshold_rank")
            assert len(axis.collections) == 8
            expected = samples.mean(axis=0)
            for head in range(8):
                line = lines[f"KV head index {head}"]
                np.testing.assert_array_equal(line.get_xdata(), analysis.prefixes)
                np.testing.assert_array_equal(line.get_ydata(), expected[:, head])
                vertices = axis.collections[head].get_paths()[0].vertices
                for token_index, prefix in enumerate(analysis.prefixes):
                    observed = vertices[np.isclose(vertices[:, 0], prefix), 1]
                    assert observed.min() == samples[:, token_index, head].min()
                    assert observed.max() == samples[:, token_index, head].max()
        assert figure.axes[row * 2].get_ylim() == figure.axes[row * 2 + 1].get_ylim()


def test_grouped_layers_keep_tokens_and_average_inputs_heads_seeds(tmp_path, capture_figures):
    analysis = make_analysis()
    unified_plots.plot_analysis(analysis, tmp_path)
    figure = capture_figures["llama/layers"]
    assert "all 3 measured layers" in figure._suptitle.get_text()
    assert "2 inputs and 8 heads" in figure.axes[0].get_title()
    assert "5 seeds" in figure.axes[1].get_title()
    assert figure.axes[-1].get_ylabel() == "Layer index (zero based)"
    for row, (metric, _) in enumerate(RANK_METRICS):
        expected_groups = [
            analysis.conditions[0].arrays[metric].mean(axis=(1, 3)),
            np.stack([condition.arrays[metric].mean(axis=(1, 3)) for condition in analysis.conditions[1:]]).mean(axis=0),
        ]
        for column, expected in enumerate(expected_groups):
            axis = figure.axes[row * 2 + column]
            lines = visible_lines(axis)
            assert list(lines) == ["Layer 0", "Layer 2", "Layer 4"]
            assert len(axis.lines) == 3 + (metric == "threshold_rank")
            assert not axis.collections
            for layer_axis, layer in enumerate(analysis.layers):
                line = lines[f"Layer {layer}"]
                np.testing.assert_array_equal(line.get_xdata(), analysis.prefixes)
                np.testing.assert_array_equal(line.get_ydata(), expected[layer_axis])
                assert len(np.unique(line.get_ydata())) > 1
        assert figure.axes[row * 2].get_ylim() == figure.axes[row * 2 + 1].get_ylim()


def test_depth_counts_preserve_terminal_values_and_individual_seed_lines(tmp_path, capture_figures):
    analyses = [make_analysis(), make_analysis("gdn", "Gated DeltaNet", width=256, layers=(0, 1, 5), depth=7)]
    paths = unified_plots.plot_depth_comparison(analyses, tmp_path)
    assert paths == ["depth_counts.png", "depth_counts.pdf"]
    assert set(capture_figures) == {"depth_counts"}
    figure = capture_figures["depth_counts"]
    assert len(figure.axes) == 8
    for row, (metric, _) in enumerate(RANK_METRICS):
        for column, analysis in enumerate(analyses):
            expected = analysis.conditions[0].arrays[metric][:, :, -1].mean(axis=(1, 2))
            random = np.stack([condition.arrays[metric][:, :, -1].mean(axis=(1, 2))
                               for condition in analysis.conditions[1:]])
            axis = figure.axes[row * 2 + column]
            lines = visible_lines(axis)
            np.testing.assert_array_equal(lines["pretrained"].get_xdata(), analysis.layers)
            np.testing.assert_array_equal(lines["pretrained"].get_ydata(), expected)
            np.testing.assert_array_equal(lines["random mean (5 seeds)"].get_ydata(), random.mean(axis=0))
            faint = [line for line in axis.lines if line.get_label() == "_random_seed"]
            assert len(faint) == 5
            for line, seed_values in zip(faint, random, strict=True):
                np.testing.assert_array_equal(line.get_ydata(), seed_values)
        assert figure.axes[row * 2].get_ylim() == figure.axes[row * 2 + 1].get_ylim()


def test_inventory_is_five_figures_without_seed_input_or_fractional_exports(tmp_path, capture_figures):
    analyses = [make_analysis(), make_analysis("gdn", "Gated DeltaNet", width=256)]
    paths = []
    for analysis in analyses:
        paths.extend(unified_plots.plot_analysis(analysis, tmp_path))
    paths.extend(unified_plots.plot_depth_comparison(analyses, tmp_path))
    expected = {"llama/heads", "llama/layers", "gdn/heads", "gdn/layers", "depth_counts"}
    assert set(capture_figures) == expected
    assert set(paths) == {f"{name}.{extension}" for name in expected for extension in ("png", "pdf")}
    assert len(paths) == 10


def test_export_freezes_layout_and_writes_both_formats(tmp_path):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(3, 2), constrained_layout=True)
    axis.plot([0, 1], [1, 2], label="Head index 0")
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center")
    paths = unified_plots._save(figure, tmp_path, "nested/smoke")
    assert paths == ["nested/smoke.png", "nested/smoke.pdf"]
    assert all((tmp_path / path).stat().st_size > 1000 for path in paths)
    assert figure.get_layout_engine().__class__.__name__ == "PlaceHolderLayoutEngine"
    plt.close(figure)
