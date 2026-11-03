"""The degree-scaling comparison regenerates from its pinned runs."""

import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.transformer_pp.results.degree_scaling.plot import (
    DIRECTORY, family_of, hermite_coefficients, load_runs, main)


SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("degree_scaling.png", "degree_scaling.pdf", "summary.csv", "README.md")
# Terminal validation perplexity at update 7,570, as quoted in the README.
TERMINAL = {
    ("taylor", 1): 19.556, ("taylor", 2): 15.912, ("taylor", 3): 15.974, ("taylor", 4): 14.724,
    ("hermite", 1): 19.513, ("hermite", 2): 16.168, ("hermite", 3): 15.732, ("hermite", 4): 14.987,
    ("static", 1): 19.531, ("static", 2): 15.822, ("static", 3): 16.649, ("static", 4): 14.773,
    ("query_w_decay", 1): 19.219, ("query_w_decay", 2): 15.832, ("query_w_decay", 3): 16.443,
    ("query_w_decay", 4): 14.714,
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_runs_and_quoted_values():
    rows = load_runs(SELECTION)
    cells = {(row["family"], row["degree"]): row for row in rows if row["role"] == "primary"}
    assert len(rows) == 17 and set(cells) == set(TERMINAL)
    assert {row["terminal_step"] for row in rows} == {7570}
    assert {row["best_validation_step"] for row in rows} == {7500}
    for cell, value in TERMINAL.items():
        assert cells[cell]["terminal_validation_perplexity"] == pytest.approx(value, abs=5e-4), cell
    assert cells["static", 3]["change_from_previous_degree_percent"] == pytest.approx(5.22, abs=5e-3)
    assert cells["query_w_decay", 3]["change_from_previous_degree_percent"] == pytest.approx(3.86, abs=5e-3)
    # The m = 1 Taylor and Hermite runs train the same function; their gap is the noise scale.
    twin = cells["taylor", 1]["terminal_validation_perplexity"] - cells["hermite", 1]["terminal_validation_perplexity"]
    assert twin == pytest.approx(0.042, abs=5e-4)
    (softmax,) = [row for row in rows if row["role"] == "reference"]
    assert softmax["run_id"] == "r9040f12f3d80"
    assert softmax["terminal_validation_perplexity"] == pytest.approx(14.546, abs=5e-4)


def test_family_follows_the_recipe():
    fixed = {"attention_degree": 2, "attention_coefficients": "fixed"}
    assert family_of(fixed) == "taylor"
    assert hermite_coefficients(2) == pytest.approx([0.8243606353500641, 1.6487212707001282, 0.8243606353500641])
    assert family_of(dict(fixed, attention_fixed_coefficients=hermite_coefficients(2))) == "hermite"
    with pytest.raises(ValueError, match="neither Taylor nor the Hermite truncation"):
        family_of(dict(fixed, attention_fixed_coefficients=[1.0, 1.0, 0.5]))
    learned = {"attention_degree": 3, "attention_coefficients": "learned"}
    assert family_of(dict(learned, attention_coefficient_conditioning="static",
                          attention_coefficient_sharing="per_head")) == "static"
    query = dict(learned, attention_coefficient_conditioning="query", attention_coefficient_activation="tanh")
    assert family_of(dict(query, attention_coefficient_kernel_weight_decay=True)) == "query_w_decay"
    with pytest.raises(ValueError, match="Unlisted coefficient family"):
        family_of(dict(query, attention_coefficient_kernel_weight_decay=False))
    assert family_of({"attention_degree": None, "attention_normalization": True}) == "softmax"


def test_changed_inputs_and_mislabeled_cells_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][5]["sources"]["completion.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        load_runs(changed)
    wrong_degree = copy.deepcopy(SELECTION)
    wrong_degree["runs"][0]["degree"] = 2
    with pytest.raises(ValueError, match="selection says 2"):
        load_runs(wrong_degree)
    wrong_family = copy.deepcopy(SELECTION)
    wrong_family["runs"][4]["family"] = "taylor"
    with pytest.raises(ValueError, match="trains hermite"):
        load_runs(wrong_family)
    missing = copy.deepcopy(SELECTION)
    del missing["runs"][3]
    with pytest.raises(ValueError, match="exactly one run for every family and degree"):
        load_runs(missing)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
