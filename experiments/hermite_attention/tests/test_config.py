"""Canonical QK-norm Hermite recipes match their Taylor controls and retain provenance."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.hermite_attention.acceptance import (
    DEGREES, EXPECTED_FACTORS, HERMITE_COEFFICIENTS, HERMITE_SCALE, check_config,
)
from scaffolds.records import load_study, run_directory, validate_record
from scaffolds.records.selections import verify_pinned_runs
from scaffolds.transformer.attention.nonnegative_polynomial import effective_coefficients_numpy
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments/hermite_attention"


def _recipes():
    return [(record, load_config(STUDY / "runs" / record["run_id"] / "config.yaml"))
            for record in load_study(STUDY)["runs"]]


def _hermite(degree=4, normalized=True):
    return next(config for record, config in _recipes() if record["polynomial_family"] == "hermite"
                and config.model.attention_degree == degree and config.model.attention_normalization == normalized)


def test_complete_hermite_grid_and_optional_complete_unnormalized_controls():
    coordinates = [(record["polynomial_family"], config.model.attention_degree, config.model.attention_normalization)
                   for record, config in _recipes()]
    assert len(coordinates) == len(set(coordinates))
    assert {(degree, normalized) for family, degree, normalized in coordinates if family == "hermite"} == {
        (degree, normalized) for degree in DEGREES for normalized in (False, True)}
    controls = {(degree, normalized) for family, degree, normalized in coordinates if family == "taylor"}
    assert controls in (set(), {(degree, False) for degree in DEGREES})


def test_all_recipes_enforce_qk_norm_and_match_the_normalized_taylor_budget():
    controls = {row["degree"]: load_config(ROOT / row["sources"]["config.yaml"]["path"])
                for row in load_study(STUDY)["normalized_taylor_controls"]}
    for record, config in _recipes():
        check_config(config)
        run = run_directory(STUDY, record["run_id"])
        raw = yaml.safe_load((run / "config.yaml").read_text())
        assert raw["model"]["qk_norm"] is True
        assert config.train.source_commit == "auto"
        assert config.train.workdir == run.relative_to(ROOT)
        assert config.train.checkpoint_dir == (run / "checkpoints").relative_to(ROOT)
        assert config.train.wandb_group == "hermite_attention"
        assert config.train.wandb_run_name == record["label"]
        assert config.train.seed == config.data.shuffle_seed == 0
        assert model_parameter_count(config.model) == 199_906_464
        assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 10_000_269_312
        actual, baseline = config_to_dict(config), config_to_dict(controls[config.model.attention_degree])
        for recipe in (actual, baseline):
            for name in ("attention_normalization", "attention_fixed_coefficients", "attention_fixed_square_factors",
                         "attention_coefficient_parameterization"):
                recipe["model"].pop(name)
            for name in ("workdir", "checkpoint_dir", "source_commit", "wandb_group", "wandb_run_name", "wandb_dir"):
                recipe["train"].pop(name)
        assert actual == baseline, record["run_id"]
        # Other jobs may be updating their outputs while this allocation qualifies.
        # Only immutable recipes belong to this cross-run acceptance check.
        validate_record(run, verify_hashes=False)
        assert record["config"]["path"] == str((run / "config.yaml").relative_to(ROOT))
        assert record["config"]["sha256"] == hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest()


def test_hermite_recipes_equal_the_independent_standard_normal_projection():
    for record, config in _recipes():
        model = config.model
        if record["polynomial_family"] == "taylor":
            assert model.attention_fixed_coefficients is None
            assert model.attention_fixed_square_factors is None
            continue
        expansion = [math.exp(0.5) / math.factorial(order) for order in range(model.attention_degree + 1)]
        expected = np.polynomial.hermite_e.herme2poly(expansion)
        if model.attention_degree == 2 and model.attention_normalization:
            assert model.attention_fixed_coefficients is None
            assert model.attention_fixed_square_factors == EXPECTED_FACTORS
            actual = effective_coefficients_numpy(model.attention_fixed_square_factors)
        else:
            assert model.attention_fixed_square_factors is None
            actual = model.attention_fixed_coefficients
        np.testing.assert_allclose(actual, expected, rtol=2e-16, atol=3e-16)


def test_normalized_control_pins_identify_matching_qk_norm_executions():
    controls = load_study(STUDY)["normalized_taylor_controls"]
    assert {row["degree"] for row in controls} == set(DEGREES)
    verify_pinned_runs(ROOT, controls)
    for row in controls:
        config = load_config(ROOT / row["sources"]["config.yaml"]["path"])
        assert config.model.qk_norm is True and config.model.qk_head_dim == 64
        assert config.model.attention_degree == row["degree"]
        assert config.model.attention_normalization is True
        assert config.optim.weight_decay_policy == "exclude_embeddings_norms_biases_v1"


@pytest.mark.parametrize("changes", (
    {"qk_norm": False}, {"qk_norm_sharing": "global"}, {"qk_norm_initial_scale": 4.0},
    {"qk_norm_eps": 1e-5}, {"qk_head_dim": 32}, {"attention_coefficients": "learned"},
    {"attention_backend": "blocked"},
))
def test_acceptance_rejects_score_or_execution_mismatch(changes):
    config = _hermite()
    for name, value in changes.items():
        setattr(config.model, name, value)
    with pytest.raises(ValueError, match="QK-norm"):
        check_config(config)


def test_acceptance_rejects_changed_projection_or_expanded_double_root():
    config = _hermite()
    config.model.attention_fixed_coefficients = tuple(1 / math.factorial(r) for r in range(5))
    with pytest.raises(ValueError, match="standard-normal Hermite"):
        check_config(config)
    config = _hermite(degree=2)
    config.model = replace(config.model, attention_coefficient_parameterization="direct",
                           attention_fixed_square_factors=None,
                           attention_fixed_coefficients=tuple(HERMITE_SCALE * x for x in HERMITE_COEFFICIENTS[2]))
    with pytest.raises(ValueError, match="double root"):
        check_config(config)
