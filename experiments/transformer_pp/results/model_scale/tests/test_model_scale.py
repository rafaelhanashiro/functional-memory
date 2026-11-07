"""The model-scale comparison regenerates from its pinned runs."""

import copy
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.transformer_pp.results.model_scale.plot import (
    DIRECTORY, ROOT, check_recipes, load_runs, main, variant_of)


SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("model_scale.png", "model_scale.pdf", "summary.csv", "README.md")
# Validation-best perplexity as quoted in the README: (15-layer width-896, 12-layer width-640).
BEST = {"softmax": (14.543, 17.964), "exp": (14.257, 17.550), "fixed_taylor_m4": (14.722, 18.087),
        "static_m4": (14.767, 18.010), "query_m4_w_decay": (14.709, 17.944),
        "fox_no_pe_query_m4_w_decay": (13.873, 16.911), "fox_no_pe_exp": (13.628, 16.753)}
# Percent difference from softmax within each shape, as quoted in the findings.
VERSUS_SOFTMAX = {"exp": (-1.96, -2.30), "fixed_taylor_m4": (1.23, 0.68), "static_m4": (1.54, 0.26),
                  "query_m4_w_decay": (1.14, -0.11), "fox_no_pe_query_m4_w_decay": (-4.60, -5.86),
                  "fox_no_pe_exp": (-6.29, -6.74)}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configs():
    return {entry["run_id"]: yaml.safe_load(
        (ROOT / "experiments" / entry["study"] / "runs" / entry["run_id"] / "config.yaml").read_text())
        for entry in SELECTION["runs"]}


def test_committed_runs_and_quoted_values():
    rows = load_runs(SELECTION)
    cells = {(row["variant"], row["shape"]): row for row in rows}
    assert len(rows) == 14 and set(cells) == {(variant, shape) for variant in BEST for shape in ("reference", "l12_d640")}
    for variant, (reference, small) in BEST.items():
        assert cells[variant, "reference"]["best_validation_perplexity"] == pytest.approx(reference, abs=5e-4), variant
        assert cells[variant, "l12_d640"]["best_validation_perplexity"] == pytest.approx(small, abs=5e-4), variant
    for variant, (reference, small) in VERSUS_SOFTMAX.items():
        assert cells[variant, "reference"]["best_percent_vs_softmax"] == pytest.approx(reference, abs=5e-3), variant
        assert cells[variant, "l12_d640"]["best_percent_vs_softmax"] == pytest.approx(small, abs=5e-3), variant
    assert {row["terminal_step"] for row in rows if row["shape"] == "reference"} == {7570}
    assert {row["terminal_step"] for row in rows if row["shape"] == "l12_d640"} == {7551}
    assert {row["training_tokens"] for row in rows if row["shape"] == "l12_d640"} == {7551 * 128 * 2048}
    assert cells["softmax", "reference"]["parameter_count"] == 198_440_146
    assert cells["softmax", "l12_d640"]["parameter_count"] == 98_977_016
    ratios = [row["best_ratio_to_reference_shape"] for row in rows if row["shape"] == "l12_d640"]
    assert min(ratios) == pytest.approx(1.219, abs=5e-4) and max(ratios) == pytest.approx(1.235, abs=5e-4)


def test_variant_follows_the_recipe():
    assert variant_of({"attention_degree": None, "attention_normalization": True}) == "softmax"
    exp = {"attention_degree": None, "attention_normalization": False, "attention_coefficients": "fixed"}
    assert variant_of(exp) == "exp"
    assert variant_of(dict(exp, attention_forget_gate=True, positional_encoding="none")) == "fox_no_pe_exp"
    with pytest.raises(ValueError, match="no positional encoding"):
        variant_of(dict(exp, attention_forget_gate=True, positional_encoding="rope"))
    assert variant_of({"attention_degree": 4, "attention_coefficients": "fixed"}) == "fixed_taylor_m4"
    learned = {"attention_degree": 4, "attention_coefficients": "learned"}
    assert variant_of(dict(learned, attention_coefficient_conditioning="static",
                           attention_coefficient_sharing="per_head")) == "static_m4"
    query = dict(learned, attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                 attention_coefficient_kernel_weight_decay=True)
    assert variant_of(query) == "query_m4_w_decay"
    assert variant_of(dict(query, attention_forget_gate=True, positional_encoding="none")) \
        == "fox_no_pe_query_m4_w_decay"
    with pytest.raises(ValueError, match="degree 4"):
        variant_of(dict(query, attention_degree=8))
    with pytest.raises(ValueError, match="Unlisted variant"):
        variant_of(dict(query, attention_coefficient_kernel_weight_decay=False))


def test_pairs_differ_only_by_the_shape_rules():
    configs = _configs()
    check_recipes(SELECTION["runs"], configs, SELECTION["settings"])
    # (run, section, key, value, expected message): settings outside the shape rules.
    mutations = [
        ("rf81c96f76a1d", "optim", "learning_rate", 0.002, "optim.learning_rate"),
        ("r3e2fa2532c23", "optim", "weight_decay", 0.0, "optim.weight_decay"),
        ("r2c432bd3a2a4", "model", "attention_backend", "fused", "model.attention_backend"),
        ("r94e0f2b8cff9", "model", "qk_norm_token_scale", "cosine", "model.qk_norm_token_scale"),
        ("r3d14b598d88f", "model", "attention_forget_gate_form", "sigmoid", "attention_forget_gate_form"),
        ("ra5dfd978d3bd", "data", "shuffle_seed", 1, "data.shuffle_seed"),
        ("r9040f12f3d80", "model", "hidden_size", 1024, "does not have the reference shape"),
        ("r4cce9a985a70", "data", "batch_size", 256, "does not have the l12_d640 shape"),
        ("r4cce9a985a70", "model", "attention_coefficient_projection_initial_std", 0.0334, "projection std"),
        ("r2c432bd3a2a4", "train", "eval_batches", 19, "shared 4864 validation sequences"),
        ("r183f08346ddc", "train", "seed", 43, "seed"),
    ]
    for run, section, key, value, message in mutations:
        changed = copy.deepcopy(configs)
        changed[run].setdefault(section, {})[key] = value
        with pytest.raises(ValueError, match=message):
            check_recipes(SELECTION["runs"], changed, SELECTION["settings"])


def test_changed_inputs_and_mislabeled_runs_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][5]["sources"]["completion.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        load_runs(changed)
    mislabeled = copy.deepcopy(SELECTION)
    next(entry for entry in mislabeled["runs"] if entry["run_id"] == "r94e0f2b8cff9")["variant"] = "fixed_taylor_m4"
    with pytest.raises(ValueError, match="trains static_m4"):
        load_runs(mislabeled)
    missing = copy.deepcopy(SELECTION)
    del missing["runs"][3]
    with pytest.raises(ValueError, match="exactly one run of every variant at each shape"):
        load_runs(missing)


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
