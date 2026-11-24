"""The unscaled-cosine pair's recipes: each its width-four control's with the cosine score alone."""
from dataclasses import replace

import pytest

from experiments.transformer_pp import affine_query_cosine_acceptance as adapter
from scaffolds.transformer.configs import load_config

RUNS = ("r46c9b99d3338", "r1571399b4f27")


@pytest.mark.parametrize("run", RUNS)
def test_an_allocated_recipe_passes_at_its_launch_commit(run):
    config = load_config(adapter.RUNS / run / "config.yaml")
    adapter.check_recipe(config)
    # Acceptance replaces "auto" with the launch commit before it checks the recipe.
    adapter.check_recipe(replace(config, train=replace(config.train, source_commit="0" * 40)))


@pytest.mark.parametrize("run", RUNS)
def test_a_changed_setting_is_refused(run):
    config = load_config(adapter.RUNS / run / "config.yaml")
    for changed in (replace(config, model=replace(config.model, qk_norm_eps=2 * config.model.qk_norm_eps)),
                    replace(config, optim=replace(config.optim, learning_rate=2 * config.optim.learning_rate)),
                    replace(config, train=replace(config.train, seed=config.train.seed + 1))):
        with pytest.raises(ValueError, match="differs from its width-four control"):
            adapter.check_recipe(changed)
