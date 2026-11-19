"""The spline attention collection regenerates from its pinned runs."""

import copy
import dataclasses
import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from experiments.transformer_pp.results.pretraining.alternative_attention.splines.spline_attention import plot_training, summarize
from scaffolds.records.selections import verify_pinned_runs


DIRECTORY = plot_training.HERE
SELECTION = yaml.safe_load((DIRECTORY / "selection.yaml").read_text())
DERIVED = ("summary.csv", "coverage.csv", "score_coverage.png", "training_summary.csv", "training.png",
           "initialization_summary.csv", "README.md")
# Validation-best perplexity (static, query + W decay) and the difference from the Taylor start, as quoted.
BEST = {"taylor": (14.7454, 14.7463), "affine": (17.9328, 16.9676), "quadratic": (15.7690, 15.6748),
        "binomial": (16.2902, 16.3599)}
VERSUS_TAYLOR = {"affine": (21.6, 15.1), "quadratic": (6.9, 6.3), "binomial": (10.5, 10.9)}
CONTROLS = (14.7667, 14.7085)
# Each start's recipe is its Taylor-started partner's except for these initial-function settings.
INITIAL_FUNCTION = {"model.attention_coefficient_initialization", "model.attention_spline_initial_intercept",
                    "model.attention_spline_initial_slope", "model.attention_spline_initial_power_coefficients"}
LOCATIONS = {"data.train_manifest", "data.eval_manifest", "train.checkpoint_dir", "train.workdir",
             "train.wandb_group", "train.wandb_run_name"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _resolved(run_id: str) -> dict:
    """A recipe as the scaffold's loader resolves it, so that explicit and omitted defaults agree."""
    from scaffolds.transformer.configs import load_config

    path = plot_training.ROOT / "experiments" / "transformer_pp" / "runs" / run_id / "config.yaml"
    return {key: value for key, value in _flatten(dataclasses.asdict(load_config(path))).items()
            if key not in LOCATIONS}


def test_committed_runs_and_quoted_values():
    runs = plot_training._runs()
    rows = {(row["start"], row["generator"]): row for row in plot_training.initializations(runs)}
    assert len(rows) == 8
    for start, (static, query) in BEST.items():
        assert rows[start, "static"]["best_validation_perplexity"] == pytest.approx(static, abs=5e-5), start
        assert rows[start, "query + W decay"]["best_validation_perplexity"] == pytest.approx(query, abs=5e-5), start
    for start, (static, query) in VERSUS_TAYLOR.items():
        assert rows[start, "static"]["best_percent_vs_taylor_spline"] == pytest.approx(static, abs=0.05), start
        assert rows[start, "query + W decay"]["best_percent_vs_taylor_spline"] == pytest.approx(query, abs=0.05), start
    assert {row["best_validation_step"] for row in rows.values()} == {7500}
    controls = [plot_training._best_and_terminal(runs[role])[0]["validation_perplexity"]
                for role in ("control_static", "control_query")]
    assert controls == [pytest.approx(value, abs=5e-5) for value in CONTROLS]


def test_each_start_changes_only_its_initial_function():
    runs = {entry["role"]: entry["run_id"] for entry in SELECTION["runs"]}
    checked = 0
    for start, _, static_role, query_role in plot_training.STARTS[1:]:
        for role, partner in ((static_role, "static_spline"), (query_role, "query_spline")):
            expected, actual = _resolved(runs[partner]), _resolved(runs[role])
            changed = {key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key)}
            assert changed and changed <= INITIAL_FUNCTION, (start, role, sorted(changed))
            assert actual["model.attention_coefficient_initialization"] == ("affine" if start == "affine" else "polynomial")
            checked += 1
    assert checked == 6


def test_changed_inputs_are_rejected():
    changed = copy.deepcopy(SELECTION)
    changed["runs"][5]["sources"]["completion.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ from the selection"):
        verify_pinned_runs(plot_training.ROOT, changed["runs"])


def test_regenerates_the_committed_outputs_byte_identically(tmp_path):
    shutil.copyfile(DIRECTORY / "README.md", tmp_path / "README.md")
    summarize.main(tmp_path)
    plot_training.main(tmp_path)
    assert {name: _digest(tmp_path / name) for name in DERIVED} == {name: _digest(DIRECTORY / name) for name in DERIVED}
