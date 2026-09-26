"""Each recipe is the retained d_K = 64 grid recipe at its degree, under QK-norm and selective decay."""
from pathlib import Path

import pytest

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity
from experiments.normalized_taylor_rerun.adapter.execution.acceptance import DEGREES, SPEC

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "experiments/normalized_taylor_rerun"
CONTROL = ROOT / "experiments/key_dimension_taylor/configs/200m-m002-seed0.yaml"


def run_name(degree):
    return f"200m-m{degree:03d}-seed0"


@pytest.mark.parametrize("degree", DEGREES)
def test_recipes_change_only_degree_qk_norm_source_and_outputs(degree):
    candidate = config_to_dict(load_config(RECORD / "configs" / f"{run_name(degree)}.yaml"))
    control = config_to_dict(load_config(CONTROL))
    assert candidate["model"].pop("attention_degree") == degree
    control["model"].pop("attention_degree")
    assert candidate["model"].pop("qk_norm") is True and control["model"].pop("qk_norm") is False
    assert candidate["train"].pop("source_commit") == "auto"
    control["train"].pop("source_commit")
    assert candidate["train"].pop("workdir") == f"experiments/normalized_taylor_rerun/runs/{run_name(degree)}"
    assert candidate["train"].pop("checkpoint_dir") == f"experiments/normalized_taylor_rerun/checkpoints/{run_name(degree)}"
    for key in ("workdir", "checkpoint_dir"):
        control["train"].pop(key)
    assert candidate == control
    assert candidate["optim"]["weight_decay_policy"] == "exclude_embeddings_norms_biases_v1"


@pytest.mark.parametrize("degree", DEGREES)
def test_recipes_pass_acceptance_with_one_scale_per_layer_and_head(degree):
    config = load_config(RECORD / "configs" / f"{run_name(degree)}.yaml")
    SPEC.check_config(config)
    assert model_parameter_count(config.model) == 199_906_304 + 160
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "taylor"
    assert identity["attention_implementation"] == "pallas_triton"
    assert identity["qk_norm"] is True
    assert identity.get("attention_signed", False) is (degree % 2 == 1)


def test_acceptance_rejects_the_legacy_recipes():
    for path in (CONTROL, ROOT / "experiments/odd_taylor_attention/configs/200m-m001-seed0.yaml"):
        with pytest.raises(ValueError, match="QK-norm"):
            SPEC.check_config(load_config(path))
