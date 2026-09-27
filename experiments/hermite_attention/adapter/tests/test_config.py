"""All eight recipes share the baseline budget and use the stated fixed projection."""
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.hermite_attention.adapter.execution.acceptance import check_config
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import model_parameter_count

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "experiments/hermite_attention"


@pytest.mark.parametrize("degree", (1, 2, 3, 4))
@pytest.mark.parametrize("normalized", (False, True))
def test_recipe_has_fixed_hermite_projection_and_baseline_budget(degree, normalized):
    variant = "normalized" if normalized else "unnormalized"
    name = f"200m-{variant}-m{degree:03d}-seed0"
    config = load_config(RECORD / "configs" / f"{name}.yaml")
    check_config(config)
    assert config.model.attention_normalization is normalized
    expansion = np.array([math.exp(0.5) / math.factorial(order) for order in range(degree + 1)])
    expected = np.polynomial.hermite_e.herme2poly(expansion)
    np.testing.assert_allclose(config.model.attention_fixed_coefficients, expected, rtol=2e-16, atol=2e-16)
    assert model_parameter_count(config.model) == 199906304
    actual = config_to_dict(config)
    baseline = config_to_dict(load_config(ROOT / "experiments/unnormalized_taylor_attention/configs/200m-m001-seed0.yaml"))
    for recipe in (actual, baseline):
        for key in ("attention_degree", "attention_normalization", "attention_backend", "attention_fixed_coefficients"):
            recipe["model"].pop(key)
        for key in ("workdir", "checkpoint_dir", "wandb_group", "wandb_run_name", "wandb_dir"):
            recipe["train"].pop(key)
    assert actual == baseline
    assert str(config.train.workdir) == f"experiments/hermite_attention/runs/{name}"
    assert str(config.train.checkpoint_dir) == f"experiments/hermite_attention/checkpoints/{name}"
    assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 10000269312


def test_acceptance_rejects_accidental_taylor_coefficients():
    config = load_config(RECORD / "configs/200m-normalized-m004-seed0.yaml")
    config.model.attention_fixed_coefficients = (1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0)
    with pytest.raises(ValueError, match="standard-normal Hermite"):
        check_config(config)
