"""The recipe is the 200M softmax control with QK-norm added and nothing else changed."""
import math
from pathlib import Path
from scaffolds.records import load_study, run_directory

import pytest

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity
from experiments.qk_norm_attention.acceptance import QK_HEAD_DIM, SPEC

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "experiments/qk_norm_attention"
CONTROL = run_directory(ROOT / "experiments/key_dimension_taylor", "200m-softmax-seed0") / "config.yaml"
RUN = "200m-qk-softmax-seed0"
SOFTMAX_CONTROL_PARAMETERS = 199_906_304


def test_recipe_changes_only_qk_norm_and_output_locations():
    candidate = config_to_dict(load_config(run_directory(RECORD, RUN) / "config.yaml"))
    control = config_to_dict(load_config(CONTROL))
    assert candidate["model"].pop("qk_norm") is True
    assert candidate["model"].pop("qk_norm_sharing") == "per_head"
    assert candidate["model"].pop("qk_norm_initial_scale") is None
    assert candidate["model"].pop("qk_norm_eps") == 1e-6
    assert control["model"].pop("qk_norm") is False
    for key in ("qk_norm_sharing", "qk_norm_initial_scale", "qk_norm_eps"):
        control["model"].pop(key)
    assert candidate["train"].pop("workdir") == str(run_directory(RECORD, RUN).relative_to(ROOT))
    assert candidate["train"].pop("checkpoint_dir") == str((run_directory(RECORD, RUN) / "checkpoints").relative_to(ROOT))
    assert candidate["train"].pop("source_commit") == "auto"
    for key in ("workdir", "checkpoint_dir", "source_commit"):
        control["train"].pop(key)
    # The control trained under the earlier all-parameter decay; loading it now
    # resolves the canonical policy, which this recipe states explicitly.
    assert candidate["optim"]["weight_decay_policy"] == "exclude_embeddings_norms_biases_v1"
    assert candidate == control


def test_recipe_passes_acceptance_and_adds_one_scale_per_layer_and_head():
    config = load_config(run_directory(RECORD, RUN) / "config.yaml")
    SPEC.check_config(config)
    model = config.model
    assert model.qk_head_dim == QK_HEAD_DIM
    assert model_parameter_count(model) == SOFTMAX_CONTROL_PARAMETERS + model.num_layers * model.num_heads
    identity = attention_identity(config)
    assert identity["attention_kernel"] == "softmax"
    assert identity["attention_implementation"] == "cudnn"
    assert identity["qk_norm"] is True
    assert identity["qk_norm_parameter_count"] == 160
    assert identity["qk_norm_initial_scale"] == math.sqrt(QK_HEAD_DIM)
    assert "attention_signed" not in identity


def test_acceptance_rejects_the_control_and_other_variants():
    control = load_config(CONTROL)
    with pytest.raises(ValueError, match="QK-norm"):
        SPEC.check_config(control)
    config = load_config(run_directory(RECORD, RUN) / "config.yaml")
    config.model.qk_norm_initial_scale = 4.0
    with pytest.raises(ValueError, match="initial scale"):
        SPEC.check_config(config)
    config = load_config(run_directory(RECORD, RUN) / "config.yaml")
    config.model.qk_norm_sharing = "global"
    with pytest.raises(ValueError, match="per-head"):
        SPEC.check_config(config)
