"""The budget comparison at the 12-layer width-640 shape regenerates from its pinned runs."""

import copy
import dataclasses
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.transformer_pp.results.l12_d640.budgets.plot import (
    DIRECTORY, ROOT, load_runs, main, matched_gaps, parse_role, variant_of)


SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("budgets.png", "budgets.pdf", "summary.csv", "README.md")
# Values quoted in the findings, keyed by (variant, degree).
BEST_2X = {("fox_query", 8): 15.314, ("fox_exp", None): 15.320, ("sum_of_squares", 12): 15.715,
           ("query", 12): 15.742, ("exp", None): 15.772, ("fixed_taylor", 8): 15.783, ("softmax", None): 16.192}
VERSUS_EXP_2X = {("fox_query", 8): -2.90, ("fox_exp", None): -2.87, ("fox_query", 12): -2.77,
                 ("sum_of_squares", 12): -0.36, ("fixed_taylor", 8): 0.07, ("softmax", None): 2.66}
VERSUS_1X = {("query", 8): -10.30, ("exp", None): -10.14, ("softplus_input", 8): -9.99, ("softmax", None): -9.86,
             ("fox_exp", None): -8.56}
VERSUS_REFERENCE = {("fox_query", 8): 10.12, ("fixed_taylor", 12): 10.16, ("softmax", None): 11.34,
                    ("fox_exp", None): 12.41}
BEST_4X = {("sum_of_squares", 12): 14.565, ("softplus_input", 12): 14.581, ("exp", None): 14.605,
           ("query", 12): 14.606, ("fixed_taylor", 12): 14.615, ("static", 12): 14.624, ("softmax", None): 15.092}
VERSUS_EXP_4X = {("sum_of_squares", 12): -0.28, ("softplus_input", 12): -0.16, ("query", 12): 0.01,
                 ("fixed_taylor", 12): 0.07, ("static", 12): 0.13, ("softmax", None): 3.34}
TWO_TO_FOUR = {("exp", None): -7.40, ("query", 12): -7.22, ("sum_of_squares", 12): -7.32, ("softmax", None): -6.79}
# Recipe keys that differ between runs by design: run names and locations (the data manifests' contents are
# checked by hash) and the blocked backend's summation order, which evaluates the same arithmetic.
LOCATIONS = {"data.train_manifest", "data.eval_manifest", "train.checkpoint_dir", "train.workdir",
             "train.wandb_group", "train.wandb_run_name", "model.attention_blocked_implementation"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cells(rows, shape, budget):
    return {row["key"][2:]: row for row in rows if (row["shape"], row["budget"]) == (shape, budget)}


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _resolved(path: Path) -> dict:
    """A recipe as the scaffold's loader resolves it, so that explicit and omitted defaults agree."""
    from scaffolds.transformer.configs import load_config

    flat = _flatten(dataclasses.asdict(load_config(path)))
    return {key: value for key, value in flat.items() if key not in LOCATIONS}


def test_committed_runs_and_quoted_values():
    rows = load_runs()
    two, four = _cells(rows, "l12_d640", 2), _cells(rows, "l12_d640", 4)
    assert (len(two), len(four), len(rows)) == (15, 7, 40)
    for cell, best in BEST_2X.items():
        assert two[cell]["best_validation_perplexity"] == pytest.approx(best, abs=5e-4), cell
    for cell, gap in VERSUS_EXP_2X.items():
        assert two[cell]["best_percent_vs_exp"] == pytest.approx(gap, abs=5e-3), cell
    for cell, gap in VERSUS_1X.items():
        assert two[cell]["best_percent_vs_half_budget"] == pytest.approx(gap, abs=5e-3), cell
    for cell, gap in VERSUS_REFERENCE.items():
        assert two[cell]["best_percent_vs_reference_shape"] == pytest.approx(gap, abs=5e-3), cell
    against_reference = [row["best_percent_vs_reference_shape"] for row in two.values()
                         if row["best_percent_vs_reference_shape"] != ""]
    assert (min(against_reference), max(against_reference)) == (
        pytest.approx(10.12, abs=5e-3), pytest.approx(12.41, abs=5e-3))
    rope = [row["best_percent_vs_exp"] for cell, row in two.items()
            if cell[0] in ("query", "static", "fixed_taylor", "softplus_input", "sum_of_squares")]
    assert len(rope) == 10 and max(rope) - min(rope) < 0.45
    for cell, best in BEST_4X.items():
        assert four[cell]["best_validation_perplexity"] == pytest.approx(best, abs=5e-4), cell
    for cell, gap in VERSUS_EXP_4X.items():
        assert four[cell]["best_percent_vs_exp"] == pytest.approx(gap, abs=5e-3), cell
    for cell, gap in TWO_TO_FOUR.items():
        assert four[cell]["best_percent_vs_half_budget"] == pytest.approx(gap, abs=5e-3), cell
    gaps = matched_gaps(rows, SELECTION["settings"])
    assert gaps["softmax", None][-1] == (pytest.approx(2.70, abs=5e-3), pytest.approx(3.34, abs=5e-3))
    assert gaps["query", 12][0] == (pytest.approx(-0.18, abs=5e-3), pytest.approx(0.05, abs=5e-3))


def test_roles_parse_and_name_their_recipes():
    assert parse_role("l12_d640_2x_sum_of_squares_m12") == ("l12_d640", 2, "sum_of_squares", 12)
    assert parse_role("reference_1x_fox_exp") == ("reference", 1, "fox_exp", None)
    with pytest.raises(ValueError, match="Unlisted role"):
        parse_role("l12_d640_2x_dense_query_m4")
    rope = {"attention_normalization": False, "attention_output_scaling": "head_rms_norm", "attention_degree": 12,
            "attention_coefficients": "learned", "attention_coefficient_conditioning": "query",
            "attention_coefficient_activation": "tanh", "attention_coefficient_kernel_weight_decay": True}
    assert variant_of(rope) == ("query", 12)
    assert variant_of(dict(rope, attention_score_transform="softplus")) == ("softplus_input", 12)
    assert variant_of(dict(rope, attention_coefficient_parameterization="sum_of_squares")) == ("sum_of_squares", 12)
    assert variant_of(dict(rope, attention_forget_gate=True, positional_encoding="none")) == ("fox_query", 12)
    with pytest.raises(ValueError, match="no positional encoding"):
        variant_of(dict(rope, attention_forget_gate=True))
    with pytest.raises(ValueError, match="Unlisted variant"):
        variant_of(dict(rope, attention_score_transform="softplus",
                        attention_coefficient_parameterization="sum_of_squares"))


def test_budget_partners_are_derived_by_the_study_rules(tmp_path):
    from experiments.transformer_pp.derive_recipe import derive

    runs = {parse_role(entry["role"]): entry["run_id"] for entry in SELECTION["runs"]}
    path = lambda run: ROOT / "experiments" / "transformer_pp" / "runs" / run / "config.yaml"
    checked = []
    for (shape, budget, variant, degree), run in runs.items():
        partner = runs.get((shape, budget // 2, variant, degree)) if shape == "l12_d640" and budget > 1 else None
        if partner is None:
            continue
        derived, _ = derive(yaml.safe_load(path(partner).read_text()), budget=budget)
        (tmp_path / f"{run}.yaml").write_text(yaml.safe_dump(derived))
        expected, actual = _resolved(tmp_path / f"{run}.yaml"), _resolved(path(run))
        changed = {key: (expected.get(key), actual.get(key)) for key in set(expected) | set(actual)
                   if expected.get(key) != actual.get(key)}
        # The 2x softplus run trains on the Mosaic transformed kernels, its 1x partner on the causal strips.
        assert changed == ({"model.attention_backend": ("blocked", "mosaic")}
                           if (variant, budget) == ("softplus_input", 2) else {}), (variant, degree, budget)
        checked.append((budget, variant, degree))
    assert sorted(budget for budget, _, _ in checked) == [2] * 5 + [4] * 7


def test_changed_inputs_and_mislabeled_runs_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][3]["sources"]["completion.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        load_runs(changed)
    mislabeled = copy.deepcopy(SELECTION)
    entry = next(entry for entry in mislabeled["runs"] if entry["role"] == "l12_d640_2x_query_m8")
    entry["role"] = "l12_d640_2x_static_m8_"
    with pytest.raises(ValueError, match="Unlisted role"):
        load_runs(mislabeled)
    entry["role"] = "l12_d640_4x_query_m8"
    with pytest.raises(ValueError, match="does not train the l12_d640 4x schedule"):
        load_runs(mislabeled)
    swapped = copy.deepcopy(SELECTION)
    roles = {entry["role"]: entry for entry in swapped["runs"]}
    roles["l12_d640_2x_query_m8"]["role"], roles["l12_d640_2x_static_m8"]["role"] = (
        "l12_d640_2x_static_m8", "l12_d640_2x_query_m8")
    with pytest.raises(ValueError, match="trains query m = 8, not static m = 8"):
        load_runs(swapped)
    without_exp = copy.deepcopy(SELECTION)
    without_exp["runs"] = [entry for entry in without_exp["runs"] if entry["role"] != "l12_d640_4x_exp"]
    with pytest.raises(ValueError, match="No exp run at the l12_d640 shape and 4x"):
        load_runs(without_exp)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
