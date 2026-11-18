"""The score-MLP comparison regenerates from its pinned runs."""

import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.transformer_pp.results.score_mlp.plot import (
    DIRECTORY, ROOT, check_recipes, load_runs, main, variant_of)


SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("score_mlp.png", "score_mlp.pdf", "summary.csv", "README.md")
# Validation-best perplexity and its difference from exp, as quoted in the README.
BEST = {"score_mlp_j4": 20.321, "score_mlp_j8": 20.976, "exp": 14.257, "query_m4": 14.774,
        "fixed_taylor_m4": 14.722}
VERSUS_EXP = {"score_mlp_j4": 42.5, "score_mlp_j8": 47.1, "query_m4": 3.6, "fixed_taylor_m4": 3.3}
# Fraction of updates 4,000-7,570 whose gradient was clipped, as quoted in the findings.
LATE_CLIPPED = {"score_mlp_j4": 0.827, "score_mlp_j8": 0.390}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configs():
    return {entry["run_id"]: yaml.safe_load(
        (ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] / "config.yaml").read_text())
        for entry in SELECTION["runs"]}


def test_committed_runs_and_quoted_values():
    rows = {row["variant"]: row for row in load_runs()}
    assert set(rows) == set(BEST)
    for variant, best in BEST.items():
        assert rows[variant]["best_validation_perplexity"] == pytest.approx(best, abs=5e-4), variant
    for variant, gap in VERSUS_EXP.items():
        assert rows[variant]["best_percent_vs_exp"] == pytest.approx(gap, abs=0.05), variant
    for variant, fraction in LATE_CLIPPED.items():
        assert rows[variant]["late_clipped_fraction"] == pytest.approx(fraction, abs=5e-4), variant
    assert max(rows[variant]["late_clipped_fraction"] for variant in ("exp", "query_m4", "fixed_taylor_m4")) < 0.002
    assert {row["terminal_step"] for row in rows.values()} == {7570}
    assert {row["training_tokens"] for row in rows.values()} == {7570 * 256 * 2048}
    assert (rows["score_mlp_j4"]["parameter_count"], rows["score_mlp_j8"]["parameter_count"]) == (
        199_208_956, 199_964_116)


def test_variant_follows_the_recipe():
    unnormalized = {"attention_normalization": False, "attention_output_scaling": "head_rms_norm"}
    mlp = dict(unnormalized, attention_score_mlp_width=4, attention_score_mlp_architecture="conventional")
    assert variant_of(mlp) == "score_mlp_j4"
    assert variant_of(dict(mlp, attention_score_mlp_width=8)) == "score_mlp_j8"
    with pytest.raises(ValueError, match="conventional SiLU kernel"):
        variant_of(dict(mlp, attention_score_mlp_architecture="dynamic_readout"))
    with pytest.raises(ValueError, match="conventional SiLU kernel"):
        variant_of(dict(mlp, attention_score_mlp_form="exp_residual"))
    assert variant_of(dict(unnormalized, attention_degree=None)) == "exp"
    assert variant_of(dict(unnormalized, attention_degree=4, attention_coefficients="fixed")) == "fixed_taylor_m4"
    query = dict(unnormalized, attention_degree=4, attention_coefficients="learned",
                 attention_coefficient_conditioning="query", attention_coefficient_activation="tanh")
    assert variant_of(query) == "query_m4"
    with pytest.raises(ValueError, match="Unlisted variant"):
        variant_of(dict(query, attention_coefficient_kernel_weight_decay=True))
    with pytest.raises(ValueError, match="head output RMSNorm"):
        variant_of(dict(query, attention_normalization=True))
    with pytest.raises(ValueError, match="RoPE without a forget gate"):
        variant_of(dict(mlp, attention_forget_gate=True))


def test_runs_share_the_control_recipe_and_the_widths_differ_only_in_j():
    configs = _configs()
    check_recipes(SELECTION["runs"], configs, SELECTION["settings"])
    wide = next(entry["run_id"] for entry in SELECTION["runs"] if entry["variant"] == "score_mlp_j8")
    faster = copy.deepcopy(configs)
    faster[wide]["optim"]["learning_rate"] = 3e-3
    with pytest.raises(ValueError, match="outside its attention mechanism"):
        check_recipes(SELECTION["runs"], faster, SELECTION["settings"])
    other_backend = copy.deepcopy(configs)
    other_backend[wide]["model"]["attention_backend"] = "blocked"
    with pytest.raises(ValueError, match="differ beyond their width"):
        check_recipes(SELECTION["runs"], other_backend, SELECTION["settings"])


def test_changed_inputs_and_mislabeled_runs_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][0]["sources"]["completion.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        load_runs(changed)
    mislabeled = copy.deepcopy(SELECTION)
    entries = {entry["variant"]: entry for entry in mislabeled["runs"]}
    entries["score_mlp_j4"]["variant"], entries["score_mlp_j8"]["variant"] = "score_mlp_j8", "score_mlp_j4"
    with pytest.raises(ValueError, match="trains score_mlp_j4, not score_mlp_j8"):
        load_runs(mislabeled)
    missing = copy.deepcopy(SELECTION)
    missing["runs"] = [entry for entry in missing["runs"] if entry["variant"] != "query_m4"]
    with pytest.raises(ValueError, match="exactly one run of every variant"):
        load_runs(missing)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
