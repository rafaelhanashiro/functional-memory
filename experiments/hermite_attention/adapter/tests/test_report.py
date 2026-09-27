"""Failures never acquire terminal results; derived tables reproduce run evidence."""
import json
from pathlib import Path
import shutil

import pytest

from experiments.hermite_attention.adapter.analysis import report
from scaffolds.transformer.analysis import regeneration_differences

RECORD = Path(__file__).resolve().parents[2]


def test_regeneration_is_byte_identical_and_preserves_authored_text(tmp_path):
    # Qualification can run while other production runs append metrics. Use an
    # immutable snapshot so this check never depends on a concurrently changing run.
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    authored = "# Authored method\n\nMethod stays unchanged.\n"
    (snapshot / "README.md").write_text(authored)
    report.regenerate(snapshot)
    output = tmp_path / "regenerated"
    output.mkdir()
    shutil.copyfile(snapshot / "README.md", output / "README.md")
    report.regenerate(snapshot, output)
    assert regeneration_differences(snapshot, output) == []
    assert (output / "README.md").read_text().startswith(authored)


def test_numerical_failure_preserves_last_validation_without_terminal_endpoint(tmp_path):
    directory = tmp_path / "runs" / report.RUNS[0]
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({"status": "failed", "step": 17}))
    (directory / "metrics.jsonl").write_text(json.dumps({
        "event": "validation", "step": 10, "checkpoint_step": 10,
        "validation_nll": 2.0, "validation_perplexity": 7.389,
    }) + "\n")
    (directory / "numerical_failure_step_00000017.json").write_text(json.dumps({
        "step": 17, "training_tokens": 8912896, "error": "nonfinite denominator",
    }))
    row = report.rows(tmp_path)[0]
    assert row["status"] == "numerical-failure"
    assert row["failure_step"] == 17
    assert row["last_valid_step"] == row["best_step"] == 10
    assert row["last_valid_perplexity"] == 7.389
    assert row["terminal_validation_perplexity"] == ""


def test_pilot_failure_is_not_a_production_metric(tmp_path):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "manifest.json").write_text(json.dumps({"runs": [
        {"name": report.RUNS[0], "status": "acceptance-numerical-failure"},
    ]}))
    row = report.rows(tmp_path)[0]
    assert row["status"] == "acceptance-numerical-failure"
    assert row["failure_step"] == row["updates"] == row["terminal_validation_perplexity"] == ""


def test_incomplete_completion_is_rejected(tmp_path):
    directory = tmp_path / "runs" / report.RUNS[0]
    directory.mkdir(parents=True)
    (directory / "completion.json").write_text(json.dumps({"final_step": 19074}))
    with pytest.raises(ValueError, match="incomplete terminal evidence"):
        report.rows(tmp_path)
