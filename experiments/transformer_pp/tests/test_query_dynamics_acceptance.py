from pathlib import Path

import pytest

from experiments.transformer_pp.query_dynamics_acceptance import check_recipe
from scaffolds.transformer.configs import load_config


@pytest.mark.parametrize("run_id", ["r1422e8000207", "r6f55fe172f41"])
def test_recipe_accepts_resolved_source_but_checks_scientific_settings(run_id):
    config = load_config(Path(__file__).resolve().parents[1] / "runs" / run_id / "config.yaml")
    config.train.source_commit = "7" * 40
    check_recipe(config)
    config.train.seed += 1
    with pytest.raises(ValueError, match="differs from the parent"):
        check_recipe(config)
