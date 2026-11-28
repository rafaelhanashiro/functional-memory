"""Scientific recipe and full-shape coefficient motion checks for the static checkpoint fork."""
from dataclasses import replace
import json

import numpy as np
import pytest

from experiments.transformer_pp import static_coefficient_fork as fork
from scaffolds.transformer.configs import load_config


def test_prefix_repeats_source_with_absolute_schedule():
    config = load_config(fork.SOURCE / "config.yaml")
    config.train = replace(config.train, wandb_run_name=fork.LABELS["prefix"],
                           eval_every=500, eval_early_every=None, eval_early_until=0)
    config.optim = replace(config.optim, total_steps=500, schedule_steps=7570)
    fork.check_recipe(config)
    config.optim = replace(config.optim, warmup_steps=10)
    with pytest.raises(ValueError, match="fork differs"):
        fork.check_recipe(config)


def test_pilot_checks_coefficients_against_trained_prefix(tmp_path):
    config = load_config(fork.SOURCE / "config.yaml")
    values = [[[0.83, 1.42]]]
    def diagnostic(table):
        return {"event": "validation", "step": 500, "attention_diagnostics": {"coefficients": {"raw": table}}}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(diagnostic(values)) + "\n")
    config.continuation = replace(config.continuation, parent_config=str(tmp_path / "config.yaml"),
                                  parent_checkpoint="checkpoints/500", freeze_static_coefficients=True)
    outcome = {"start_step": 500}
    assert fork.check_pilot(config, outcome, diagnostic(values))["changed_coefficient_count"] == 0
    with pytest.raises(ValueError, match="Unexpected static coefficient motion"):
        fork.check_pilot(config, outcome, diagnostic(np.ones_like(values)))
    config.continuation = replace(config.continuation, freeze_static_coefficients=False)
    assert fork.check_pilot(config, outcome, diagnostic(np.ones_like(values)))["changed_coefficient_count"] == 2
