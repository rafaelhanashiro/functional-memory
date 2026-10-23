"""Global-step evaluation scheduling, usable before initializing an accelerator."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scaffolds.transformer.configs import Config


def evaluation_due(config: "Config", step: int) -> bool:
    """Use the frozen early cadence on global steps, including across a pilot resume."""
    cadence = config.train.eval_every
    if config.train.eval_early_every is not None and step <= config.train.eval_early_until:
        cadence = config.train.eval_early_every
    return step == config.optim.total_steps or (cadence > 0 and step % cadence == 0)


def scheduled_work_counts(config: "Config") -> tuple[int, int]:
    """Count validation and checkpoint events for acceptance's runtime projection."""
    total = config.optim.total_steps
    evaluations = {step for step in range(1, total + 1) if evaluation_due(config, step)}
    checkpoints = evaluations | set(range(config.train.ckpt_every, total + 1, config.train.ckpt_every)) | {total}
    return len(evaluations), len(checkpoints)
