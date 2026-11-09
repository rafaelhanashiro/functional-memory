"""Scientific checks for the combined study use recipe fields and permanent run IDs."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from scaffolds.transformer.attention.nonnegative_polynomial import effective_coefficients_numpy
from scaffolds.transformer.configs import Config, ModelConfig, load_config
from scaffolds.transformer.models import model_parameter_count

STUDY = Path(__file__).resolve().parents[1]
# Recipes that declare an experiment_kind (sparse polynomial head distillation, fixed-input
# coefficient oracles) are not Transformer training recipes and keep their own formats.
RUNS = sorted(path for path in (STUDY / "runs").glob("*/config.yaml")
              if "experiment_kind" not in yaml.safe_load(path.read_text()))


def retained_config(path):
    # Missing historical QK-norm means plain scores. Loading an old recipe for
    # this audit must not insert today's defaults into its scientific identity.
    raw = yaml.safe_load(path.read_text())
    factory = Config if "qk_norm" in raw["model"] else lambda: Config(model=ModelConfig(qk_norm=False))
    return load_config(path, config_factory=factory)


@pytest.mark.parametrize("path", RUNS, ids=lambda path: path.parent.name)
def test_retained_recipes_match_executed_parameter_counts(path):
    config = retained_config(path)
    count = model_parameter_count(config.model)
    evidence = [path.with_name(name) for name in ("record.json", "metadata.json", "completion.json")]
    for source in evidence:
        if source.exists():
            recorded = json.loads(source.read_text()).get("parameter_count")
            if recorded is not None:
                assert count == recorded, source


def test_hermite_recipes_equal_the_independent_standard_normal_projection():
    checked = 0
    for path in RUNS:
        record = json.loads(path.with_name("record.json").read_text())
        if record.get("polynomial_family") != "hermite":
            continue
        model = retained_config(path).model
        expansion = [math.exp(0.5) / math.factorial(order) for order in range(model.attention_degree + 1)]
        expected = np.polynomial.hermite_e.herme2poly(expansion)
        if model.attention_fixed_square_factors is not None:
            actual = effective_coefficients_numpy(model.attention_fixed_square_factors)
        else:
            actual = model.attention_fixed_coefficients
        np.testing.assert_allclose(actual, expected, rtol=2e-16, atol=3e-16)
        checked += 1
    assert checked
