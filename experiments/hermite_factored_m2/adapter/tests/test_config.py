"""The recipe preserves the failed direct run's scientific setup apart from evaluation."""
import math
from pathlib import Path

import pytest

from experiments.hermite_factored_m2.adapter.execution.acceptance import check_config
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.models import model_parameter_count

ROOT = Path(__file__).resolve().parents[4]
NAME = "200m-normalized-factored-m002-seed0"
CONFIG = ROOT / "experiments/hermite_factored_m2/configs" / f"{NAME}.yaml"


def test_recipe_has_fixed_factorization_and_baseline_budget():
    config = load_config(CONFIG)
    check_config(config)
    factor = math.sqrt(math.exp(0.5) / 2.0)
    assert tuple(config.model.attention_fixed_square_factors) == (factor, factor, 0.0)
    assert config.model.attention_fixed_coefficients is None
    assert model_parameter_count(config.model) == 199906304
    assert config.optim.total_steps == 19074
    assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 10000269312
    assert config.data.batch_size == 256 and config.train.micro_batch_size == 64
    assert config.train.seed == config.data.shuffle_seed == 0
    assert str(config.train.workdir) == f"experiments/hermite_factored_m2/runs/{NAME}"
    assert str(config.train.checkpoint_dir) == f"experiments/hermite_factored_m2/checkpoints/{NAME}"
    assert config.train.wandb_group == "hermite_factored_m2"
    assert config.train.wandb_run_name == NAME


def test_acceptance_rejects_expanded_coefficients():
    config = load_config(CONFIG)
    config.model.attention_fixed_coefficients = (math.exp(0.5) / 2,
                                                 math.exp(0.5),
                                                 math.exp(0.5) / 2)
    with pytest.raises(ValueError, match="fixed normalized"):
        check_config(config)


def test_acceptance_rejects_wrong_factors():
    config = load_config(CONFIG)
    config.model.attention_fixed_square_factors = (1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="Square factors"):
        check_config(config)
