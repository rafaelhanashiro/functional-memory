"""Pilot and continuation share a frozen evaluation schedule indexed by global step."""

from scaffolds.transformer.configs import Config
from scaffolds.transformer.training.cadence import evaluation_due, scheduled_work_counts


def test_early_cadence_switches_after_pilot_and_always_evaluates_terminal_step():
    config = Config()
    config.optim.total_steps = 7570
    config.train.eval_every = 500
    config.train.eval_early_every = 100
    config.train.eval_early_until = 1500
    actual = [step for step in range(1, 7571) if evaluation_due(config, step)]
    expected = [*range(100, 1501, 100), *range(2000, 7501, 500), 7570]
    assert actual == expected
    # Resuming at the pilot's checkpoint does not restart the early cadence.
    assert [step for step in range(1501, 2501) if evaluation_due(config, step)] == [2000, 2500]


def test_unset_early_cadence_preserves_legacy_schedule():
    config = Config()
    config.optim.total_steps = 17
    config.train.eval_every = 5
    assert [step for step in range(1, 18) if evaluation_due(config, step)] == [5, 10, 15, 17]


def test_runtime_projection_counts_early_evaluations_and_their_checkpoints():
    config = Config()
    config.optim.total_steps = 7570
    config.train.eval_every = config.train.ckpt_every = 500
    config.train.eval_early_every = 100
    config.train.eval_early_until = 1500
    assert scheduled_work_counts(config) == (28, 28)
    config.train.eval_early_every = None
    config.train.eval_early_until = 0
    assert scheduled_work_counts(config) == (16, 16)
    config.optim.total_steps = 1000
    assert scheduled_work_counts(config) == (2, 2)  # terminal already on the cadence
    config.train.ckpt_every = 100
    assert scheduled_work_counts(config) == (2, 10)
