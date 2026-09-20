"""Cross-architecture comparisons must preserve the intended controls."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.state_rank.unify_analysis import validate_comparison, validate_output


def analysis(key, results):
    convention = {"participation_ratio_convention": "squared-singular-values", "threshold_rank_singular_tolerance": 1e-4}
    return SimpleNamespace(
        key=key, results=results,
        inputs=[{"title": "Article", "article_index": 4, "body_sha256": "abc", "text": f"{key} tokenizer prefix"}],
        prefixes=np.arange(1, 5),
        conditions=[SimpleNamespace(name="pretrained", weights="pretrained", seed=None, metadata=convention.copy()),
                    SimpleNamespace(name="random-seed0", weights="random", seed=0, metadata=convention.copy())],
    )


def test_same_articles_allow_different_native_text_prefixes(tmp_path):
    a, b = analysis("llama", tmp_path / "llama"), analysis("delta", tmp_path / "delta")
    result = validate_comparison([a, b])
    assert result["same_source_articles_in_same_order"]
    assert not result["identical_text_spans_across_architectures"]
    assert result["seeds"] == [0]


@pytest.mark.parametrize("mismatch", ["article", "prefixes", "seeds", "convention"])
def test_rejects_comparisons_with_different_controls(tmp_path, mismatch):
    a, b = analysis("llama", tmp_path / "llama"), analysis("delta", tmp_path / "delta")
    if mismatch == "article":
        b.inputs[0]["body_sha256"] = "different-source"
    elif mismatch == "prefixes":
        b.prefixes = np.arange(1, 6)
    elif mismatch == "seeds":
        b.conditions[1].seed = 1
    else:
        b.conditions[0].metadata["threshold_rank_singular_tolerance"] = 1e-3
    with pytest.raises(ValueError):
        validate_comparison([a, b])


def test_output_cannot_overlap_original_results_or_replace_old_outputs(tmp_path):
    a = analysis("llama", tmp_path / "original")
    a.results.mkdir()
    for output in (tmp_path, a.results, a.results / "figures"):
        with pytest.raises(ValueError, match="separate"):
            validate_output(output, [a])
    output = tmp_path / "derived"
    validate_output(output, [a])
    output.mkdir()
    (output / "retained.txt").write_text("scientific result")
    with pytest.raises(ValueError, match="not empty"):
        validate_output(output, [a])
