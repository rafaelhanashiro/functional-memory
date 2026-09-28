"""GLA summaries use explicit runs and width-matched, complete controls."""
import json
from pathlib import Path

import pytest

from results.gla_key_dimension import plot
from scaffolds.transformer.analysis import run_view, selected_run_directory, selected_views

OUTPUT = Path(__file__).resolve().parents[1]
STUDY = OUTPUT.parents[1] / "experiments/attention_scaling"
KEY_DIMENSIONS = (8, 16, 32, 64, 128)


def test_comparison_points_cover_every_width_and_degree():
    manifest = selected_views(OUTPUT, STUDY)
    assert {reference["study"] for reference in manifest["comparisons"]["runs"]} == {"attention_scaling"}
    assert set(plot.comparison_table(manifest)) == {
        (width, degree) for width in KEY_DIMENSIONS for degree in (None, *plot.DEGREES)}
    assert [run["qk_head_dim"] for run in manifest["runs"]] == list(KEY_DIMENSIONS)


def test_summary_uses_completed_runs_and_width_matched_comparisons(tmp_path):
    manifest = selected_views(OUTPUT, STUDY)
    # Keep the common study name while limiting the copy to the selected GLA runs.
    study = tmp_path / "attention_scaling"
    for entry in manifest["runs"]:
        source = STUDY / "runs" / entry["run_id"]
        directory = study / "runs" / entry["run_id"]
        directory.mkdir(parents=True)
        for name in ("record.json", "config.yaml"):
            (directory / name).write_bytes((source / name).read_bytes())

    def copied():
        return dict(manifest, runs=[run_view(study / "runs" / entry["run_id"], entry["name"])
                                    for entry in manifest["runs"]])

    run = selected_run_directory(OUTPUT, study, "200m-dk016-gla-seed0")
    (run / "completion.json").write_text(json.dumps({"final_step": 19074, "training_tokens": 10_000_269_312,
                                                     "validation_nll": 2.8, "validation_perplexity": 16.4}))
    (run / "best_validation.json").write_text(json.dumps({"step": 19074, "validation_perplexity": 16.4}))
    rows = {row["run"]: row for row in plot.gla_rows(copied(), study)}
    assert rows["200m-dk008-gla-seed0"]["status"] == "incomplete"
    completed = rows["200m-dk016-gla-seed0"]
    assert completed["softmax_perplexity"] == plot.comparison_table(manifest)[(16, None)]
    assert completed["delta_perplexity_vs_softmax"] == pytest.approx(16.4 - completed["softmax_perplexity"])
    (run / "completion.json").write_text(json.dumps({"final_step": 1000, "training_tokens": 524_288_000,
                                                     "validation_nll": 3.0, "validation_perplexity": 20.0}))
    with pytest.raises(ValueError, match="full training budget"):
        plot.gla_rows(copied(), study)
