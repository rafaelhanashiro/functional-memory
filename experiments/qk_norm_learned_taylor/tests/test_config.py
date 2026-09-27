"""Each recipe is its learned-coefficient control with QK-norm added, and the normalized twin differs only in normalization."""
import math
from pathlib import Path
from scaffolds.records import load_study, run_directory

import pytest

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity
from experiments.qk_norm_learned_taylor.acceptance import DEGREES, QK_HEAD_DIM, SPEC, expected_backend

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "experiments/qk_norm_learned_taylor"
CONTROLS = ROOT / "experiments/learned_unnormalized_taylor"
VARIANTS = ("unnormalized", "normalized")
SOFTMAX_CONTROL_PARAMETERS = 199_906_304


def run_name(degree: int, variant: str) -> str:
    return f"200m-qk-m{degree:03d}-{variant}-seed0"


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_recipe_changes_only_qk_norm_normalization_backend_and_output_locations(degree, variant):
    run = run_name(degree, variant)
    candidate = config_to_dict(load_config(run_directory(RECORD, run) / "config.yaml"))
    control = config_to_dict(load_config(run_directory(CONTROLS, f"200m-m{degree:03d}-seed0") / "config.yaml"))
    assert candidate["model"].pop("qk_norm") is True
    assert candidate["model"].pop("qk_norm_sharing") == "per_head"
    assert candidate["model"].pop("qk_norm_initial_scale") is None
    assert candidate["model"].pop("qk_norm_eps") == 1e-6
    assert control["model"].pop("qk_norm") is False
    for key in ("qk_norm_sharing", "qk_norm_initial_scale", "qk_norm_eps"):
        control["model"].pop(key)
    normalized = variant == "normalized"
    assert candidate["model"].pop("attention_normalization") is normalized
    assert control["model"].pop("attention_normalization") is False
    assert candidate["model"].pop("attention_backend") == expected_backend(normalized)
    assert control["model"].pop("attention_backend") == "auto"
    assert candidate["train"].pop("workdir") == str(run_directory(RECORD, run).relative_to(ROOT))
    assert candidate["train"].pop("checkpoint_dir") == str((run_directory(RECORD, run) / "checkpoints").relative_to(ROOT))
    assert candidate["train"].pop("source_commit") == "auto"
    for key in ("workdir", "checkpoint_dir", "source_commit"):
        control["train"].pop(key)
    # The controls trained under the earlier all-parameter decay; loading them now resolves the
    # canonical policy, which these recipes state explicitly.
    assert candidate["optim"]["weight_decay_policy"] == "exclude_embeddings_norms_biases_v1"
    assert candidate == control


@pytest.mark.parametrize("degree", DEGREES)
def test_normalized_and_unnormalized_twins_differ_only_in_normalization_backend_and_locations(degree):
    twins = {variant: config_to_dict(load_config(run_directory(RECORD, run_name(degree, variant)) / "config.yaml"))
             for variant in VARIANTS}
    for variant, config in twins.items():
        assert config["model"].pop("attention_normalization") is (variant == "normalized")
        config["model"].pop("attention_backend")
        config["train"].pop("workdir")
        config["train"].pop("checkpoint_dir")
    assert twins["normalized"] == twins["unnormalized"]


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_recipes_pass_acceptance_and_add_coefficients_and_scales_per_layer_and_head(degree, variant):
    config = load_config(run_directory(RECORD, run_name(degree, variant)) / "config.yaml")
    SPEC.check_config(config)
    model = config.model
    heads = model.num_layers * model.num_heads
    assert model.qk_head_dim == QK_HEAD_DIM
    assert model_parameter_count(model) == SOFTMAX_CONTROL_PARAMETERS + heads * (degree + 1) + heads
    identity = attention_identity(config)
    assert identity["attention_kernel"] == ("learned_polynomial" if variant == "normalized"
                                            else "learned_polynomial_unnormalized")
    assert identity["attention_coefficient_count"] == 160 * (degree + 1)
    assert identity["attention_signed"] is True
    assert identity["qk_norm"] is True and identity["qk_norm_parameter_count"] == 160
    assert identity["qk_norm_initial_scale"] == math.sqrt(QK_HEAD_DIM)
    assert identity["attention_backend"] == expected_backend(variant == "normalized")


def test_acceptance_rejects_controls_and_mismatched_backends():
    with pytest.raises(ValueError, match="QK-norm"):
        SPEC.check_config(load_config(run_directory(CONTROLS, "200m-m002-seed0") / "config.yaml"))
    config = load_config(run_directory(RECORD, run_name(2, "unnormalized")) / "config.yaml")
    config.model.attention_backend = "blocked"
    with pytest.raises(ValueError, match="backend"):
        SPEC.check_config(config)
    config = load_config(run_directory(RECORD, run_name(2, "normalized")) / "config.yaml")
    config.model.qk_norm_initial_scale = 2.0
    with pytest.raises(ValueError, match="initial scale"):
        SPEC.check_config(config)
