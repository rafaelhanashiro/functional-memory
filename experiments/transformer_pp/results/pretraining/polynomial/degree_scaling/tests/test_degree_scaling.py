"""The degree-scaling comparison regenerates from its pinned runs."""

import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.transformer_pp.results.pretraining.polynomial.degree_scaling.plot import (
    DIRECTORY, ROOT, check_recipes, family_of, hermite_coefficients, load_runs, main)


SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("degree_scaling.png", "degree_scaling.pdf", "summary.csv", "README.md")
# Terminal validation perplexity at update 7,570, as quoted in the README.
TERMINAL = {
    ("taylor", 1): 19.556, ("taylor", 2): 15.912, ("taylor", 3): 15.974, ("taylor", 4): 14.724,
    ("taylor", 5): 14.843, ("taylor", 6): 14.334, ("taylor", 7): 14.3095,
    ("hermite", 1): 19.513, ("hermite", 2): 16.168, ("hermite", 3): 15.732, ("hermite", 4): 14.987,
    ("hermite", 5): 14.609, ("hermite", 6): 14.396, ("hermite", 7): 14.271,
    ("static", 1): 19.531, ("static", 2): 15.822, ("static", 3): 16.649, ("static", 4): 14.773,
    ("static", 5): 14.871, ("static", 6): 14.408, ("static", 7): 14.427,
    ("query_w_decay", 1): 19.219, ("query_w_decay", 2): 15.832, ("query_w_decay", 3): 16.443,
    ("query_w_decay", 4): 14.714, ("query_w_decay", 5): 15.080, ("query_w_decay", 6): 14.641,
    ("query_w_decay", 7): 14.384,
    ("taylor", 8): 14.223, ("hermite", 8): 14.231, ("static", 8): 14.2985, ("query_w_decay", 8): 14.328,
    ("taylor", 12): 14.311, ("query_w_decay", 12): 14.230,
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_runs_and_quoted_values():
    rows = load_runs(SELECTION)
    cells = {(row["family"], row["degree"]): row for row in rows if row["role"] == "primary"}
    assert len(rows) == 36 and set(cells) == set(TERMINAL)
    assert {row["terminal_step"] for row in rows} == {7570}
    assert {row["best_validation_step"] for row in rows} == {7500}
    for cell, value in TERMINAL.items():
        assert cells[cell]["terminal_validation_perplexity"] == pytest.approx(value, abs=5e-4), cell
    assert cells["static", 3]["change_from_previous_degree_percent"] == pytest.approx(5.22, abs=5e-3)
    assert cells["query_w_decay", 3]["change_from_previous_degree_percent"] == pytest.approx(3.86, abs=5e-3)
    # m = 5: the Taylor-shaped families end above m = 4 and Hermite below it.
    assert cells["taylor", 5]["change_from_previous_degree_percent"] == pytest.approx(0.81, abs=5e-3)
    assert cells["static", 5]["change_from_previous_degree_percent"] == pytest.approx(0.67, abs=5e-3)
    assert cells["query_w_decay", 5]["change_from_previous_degree_percent"] == pytest.approx(2.49, abs=5e-3)
    assert cells["hermite", 5]["change_from_previous_degree_percent"] == pytest.approx(-2.53, abs=5e-3)
    assert cells["hermite", 5]["percent_above_softmax"] == pytest.approx(0.43, abs=5e-3)
    # m = 6: every family improves; Taylor, Hermite, and static end below softmax.
    assert cells["taylor", 6]["change_from_previous_degree_percent"] == pytest.approx(-3.43, abs=5e-3)
    assert cells["taylor", 6]["percent_above_softmax"] == pytest.approx(-1.46, abs=5e-3)
    assert cells["taylor", 6]["percent_above_exp"] == pytest.approx(0.51, abs=5e-3)
    assert cells["query_w_decay", 6]["percent_above_softmax"] == pytest.approx(0.65, abs=5e-3)
    # m = 7: Taylor and static stay level with m = 6; Hermite comes within 0.07% of exp.
    assert cells["taylor", 7]["change_from_previous_degree_percent"] == pytest.approx(-0.17, abs=5e-3)
    assert cells["static", 7]["change_from_previous_degree_percent"] == pytest.approx(0.13, abs=5e-3)
    assert cells["query_w_decay", 7]["change_from_previous_degree_percent"] == pytest.approx(-1.75, abs=5e-3)
    assert cells["hermite", 7]["percent_above_exp"] == pytest.approx(0.07, abs=5e-3)
    # m = 8: every family improves; the fixed families end 0.2-0.3% below exp.
    assert cells["taylor", 8]["change_from_previous_degree_percent"] == pytest.approx(-0.60, abs=5e-3)
    assert cells["static", 8]["change_from_previous_degree_percent"] == pytest.approx(-0.89, abs=5e-3)
    assert cells["taylor", 8]["percent_above_exp"] == pytest.approx(-0.27, abs=5e-3)
    assert cells["hermite", 8]["percent_above_exp"] == pytest.approx(-0.21, abs=5e-3)
    assert cells["query_w_decay", 8]["percent_above_exp"] == pytest.approx(0.46, abs=5e-3)
    assert cells["taylor", 8]["percent_above_softmax"] == pytest.approx(-2.22, abs=5e-3)
    # m = 12 (Taylor and query only), compared with m = 8: query improves and ends below exp; Taylor worsens.
    assert cells["query_w_decay", 12]["change_from_previous_degree_percent"] == pytest.approx(-0.69, abs=5e-3)
    assert cells["taylor", 12]["change_from_previous_degree_percent"] == pytest.approx(0.62, abs=5e-3)
    assert cells["query_w_decay", 12]["percent_above_exp"] == pytest.approx(-0.23, abs=5e-3)
    assert cells["taylor", 12]["percent_above_exp"] == pytest.approx(0.34, abs=5e-3)
    # The m = 1 Taylor and Hermite runs train the same function; their gap is the noise scale.
    twin = cells["taylor", 1]["terminal_validation_perplexity"] - cells["hermite", 1]["terminal_validation_perplexity"]
    assert twin == pytest.approx(0.042, abs=5e-4)
    references = {row["family"]: row for row in rows if row["role"] == "reference"}
    assert references["softmax"]["run_id"] == "r9040f12f3d80"
    assert references["softmax"]["terminal_validation_perplexity"] == pytest.approx(14.546, abs=5e-4)
    assert references["exp"]["run_id"] == "r6a76e7fbde45"
    assert references["exp"]["terminal_validation_perplexity"] == pytest.approx(14.262, abs=5e-4)
    assert cells["query_w_decay", 4]["percent_above_exp"] == pytest.approx(3.17, abs=5e-3)


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
    assert family_of({"attention_degree": None, "attention_normalization": False,
                      "attention_coefficients": "fixed"}) == "exp"


def test_recipes_differ_only_in_the_compared_settings():
    configs = {entry["run_id"]: yaml.safe_load(
        (ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] / "config.yaml").read_text())
        for entry in SELECTION["runs"]}
    check_recipes(SELECTION["runs"], configs, SELECTION["settings"])
    # The blocked backend's implementation (scan or causal strips) is execution, not a compared setting.
    executed = copy.deepcopy(configs)
    executed["r81fad1755b11"]["model"]["attention_blocked_implementation"] = "strips"
    check_recipes(SELECTION["runs"], executed, SELECTION["settings"])
    # (run, section, key, value, expected message): settings outside the comparison.
    mutations = [
        ("r81fad1755b11", "model", "attention_forget_gate", True, "attention_forget_gate"),
        ("r9040f12f3d80", "model", "attention_forget_gate", True, "attention_forget_gate"),
        ("r816ab0fbe119", "model", "qk_norm_token_scale", "cosine", "qk_norm_token_scale"),
        ("rdf6363a4b83c", "continuation", "parent_checkpoint", "parent/checkpoints/7500", "parent_checkpoint"),
        ("ra7f2fda1c792", "model", "attention_mechanism", "path", "attention_mechanism"),
        ("r00f1f91a7803", "optim", "learning_rate", 0.002, "optim.learning_rate"),
        ("r1f880c4ee659", "model", "hidden_size", 1024, "model.hidden_size"),
        ("r9040f12f3d80", "optim", "weight_decay", 0.0, "optim.weight_decay"),
        ("r6a76e7fbde45", "optim", "learning_rate", 0.002, "optim.learning_rate"),
    ]
    for run, section, key, value, message in mutations:
        changed = copy.deepcopy(configs)
        changed[run].setdefault(section, {})[key] = value
        with pytest.raises(ValueError, match=message):
            check_recipes(SELECTION["runs"], changed, SELECTION["settings"])


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
    next(entry for entry in wrong_family["runs"] if entry["run_id"] == "rddf0b0173f42")["family"] = "taylor"
    with pytest.raises(ValueError, match="trains hermite"):
        load_runs(wrong_family)
    missing = copy.deepcopy(SELECTION)
    del missing["runs"][3]
    with pytest.raises(ValueError, match="exactly one run for every family and degree"):
        load_runs(missing)
    undeclared = copy.deepcopy(SELECTION)
    undeclared["settings"]["not_run"] = []
    with pytest.raises(ValueError, match="exactly one run for every family and degree"):
        load_runs(undeclared)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
