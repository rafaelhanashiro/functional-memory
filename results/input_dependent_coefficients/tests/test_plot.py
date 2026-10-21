"""The input-dependent coefficient comparison regenerates from its pinned runs."""

import copy
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml

from results.input_dependent_coefficients.plot import DIRECTORY, load_runs, main, variant_of


SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("learning_curves.png", "learning_curves.pdf", "summary.csv", "README.md")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_runs_controls_and_differences():
    rows, curves = load_runs(SELECTION)
    by_id = {row["run_id"]: row for row in rows}
    assert len(rows) == 37 and all(row["status"] == "complete" for row in rows)
    assert Counter(row["degree"] for row in rows) == {1: 8, 2: 9, 3: 7, 4: 9, 8: 4}
    assert Counter(row["variant"] for row in rows)["static"] == 5
    # Every input-dependent run is compared with the static run of its own degree.
    for row in rows:
        if row["variant"] != "static":
            assert by_id[row["control_run_id"]]["degree"] == row["degree"]
    assert by_id["raa043e978a3d"]["terminal_validation_perplexity"] == pytest.approx(13.461116, abs=1e-6)
    assert by_id["ra39c41604704"]["perplexity_percent_above_control"] == pytest.approx(0.11, abs=5e-3)
    assert by_id["rccf19c31669d"]["perplexity_percent_above_control"] == pytest.approx(0.76, abs=5e-3)
    assert all(len(curves[row["run_id"]]) == row["diagnostic_count"] for row in rows)


def test_variant_follows_the_recipe():
    base = {"attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh",
            "attention_backend": "mosaic", "hidden_size": 1024, "attention_coefficient_amplitude_initial": 0.0,
            "attention_coefficient_projection_initial_std": 1 / 32}
    assert variant_of({"attention_coefficient_conditioning": "static"})[2] == "static"
    assert variant_of(base)[2] == "tanh_zero_amplitude_wide"
    assert variant_of(dict(base, attention_coefficient_amplitude_scale="factorial"))[2] == \
        "tanh_zero_amplitude_order_scaled"
    assert variant_of(dict(base, attention_coefficient_projection_initial_std=10 / 32))[2] == \
        "tanh_zero_amplitude_saturating_10"
    assert variant_of(dict(base, attention_coefficient_conditioning="query_key"))[2] == "tanh_query_key"
    assert variant_of(dict(base, attention_coefficient_amplitude_initial=1.0))[2] == "tanh_original"


def test_changed_inputs_and_mismatched_controls_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][0]["sources"]["metrics.jsonl"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        load_runs(changed)
    mismatched = copy.deepcopy(SELECTION)
    # Order-scaled m = 8 against the static m = 4 run.
    next(entry for entry in mismatched["runs"] if entry["run_id"] == "ra39c41604704")["control"] = "rc2b555b905dd"
    with pytest.raises(ValueError, match="not degree matched"):
        load_runs(mismatched)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
