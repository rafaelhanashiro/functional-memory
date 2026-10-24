"""The rational polynomial comparison regenerates from its pinned runs."""

import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from results.rational_polynomial.plot import DIRECTORY, kernel_of, load_runs, main


SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("learning_curves.png", "learning_curves.pdf", "summary.csv", "README.md")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_runs_failures_and_coefficients():
    rows, curves = load_runs(SELECTION)
    by_id = {row["run_id"]: row for row in rows}
    assert len(rows) == 7
    # Both signed degree-8 denominators stay visible as numerical failures.
    assert {run_id: (row["status"], row["terminal_step"]) for run_id, row in by_id.items()
            if row["status"] != "complete"} == {"r90f6452bba82": ("numerical_failure", 73),
                                                 "rc8aeb26cf795": ("numerical_failure", 44)}
    assert by_id["rc8aeb26cf795"]["weight_decay"] == 0.01 and curves["rc8aeb26cf795"] == []
    assert by_id["rbafa372b4e8b"]["terminal_validation_perplexity"] == pytest.approx(15.006128, abs=1e-6)
    assert by_id["r05cd33157d94"]["terminal_validation_perplexity"] == pytest.approx(15.131604, abs=1e-6)
    assert by_id["rd5ac0e3013e3"]["perplexity_above_softmax"] == pytest.approx(1.4275, abs=1e-4)
    assert by_id["rdf00da0866a8"]["terminal_validation_perplexity"] == pytest.approx(13.731335, abs=1e-6)
    assert not by_id["rd5ac0e3013e3"]["penalized_denominator"] and by_id["rbafa372b4e8b"]["penalized_denominator"]
    # Coefficient statistics quoted in the findings.
    assert by_id["rd5ac0e3013e3"]["median_denominator_constant"] == pytest.approx(1.19, abs=5e-3)
    assert by_id["r05cd33157d94"]["min_denominator_constant"] == pytest.approx(0.649, abs=1e-3)
    assert by_id["rbafa372b4e8b"]["max_abs_denominator_coefficient"] == pytest.approx(0.0031, abs=5e-5)
    assert by_id["rbafa372b4e8b"]["median_numerator_high_order_ratio"] == pytest.approx(11.2, abs=0.05)
    assert all(len(curves[row["run_id"]]) == 20 for row in rows if row["status"] == "complete")


def test_kernel_follows_the_recipe():
    rational = {"attention_coefficient_parameterization": "rational", "attention_degree": 8}
    assert kernel_of(dict(rational, attention_denominator_degree=1, attention_denominator_constant="learned")) == \
        "1/direct/learned"
    assert kernel_of(rational) == "8/direct/fixed"
    assert kernel_of(dict(rational, attention_denominator_parameterization="sum_of_squares")) == \
        "8/sum_of_squares/fixed"
    assert kernel_of({"attention_degree": None, "attention_coefficients": "fixed"}) == "softmax"
    assert kernel_of({"attention_degree": 8, "attention_coefficients": "fixed"}) == "taylor"
    with pytest.raises(ValueError, match="Unlisted rational denominator"):
        kernel_of(dict(rational, attention_denominator_degree=4))


def test_changed_inputs_and_wrong_roles_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][2]["sources"]["metrics.jsonl"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        load_runs(changed)
    swapped = copy.deepcopy(SELECTION)
    next(entry for entry in swapped["runs"] if entry["run_id"] == "rdf00da0866a8")["role"] = "primary"
    with pytest.raises(ValueError, match="Rational runs are primary"):
        load_runs(swapped)
    other_start = copy.deepcopy(SELECTION)
    other_start["settings"]["coefficient_initialization"] = "taylor"
    with pytest.raises(ValueError, match="differs from the selection settings"):
        load_runs(other_start)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
