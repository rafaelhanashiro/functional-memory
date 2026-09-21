import shutil
from collections.abc import Mapping
from typing import Any

import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as ocp_managers

from experiments.polynomial_attention.configs import Config
from experiments.polynomial_attention.training.state import TrainState


VALIDATION_NLL_METRIC = "validation_nll"


def _validation_nll(metrics: Mapping[str, Any]) -> float:
    return float(metrics[VALIDATION_NLL_METRIC])


def cleanup_stale_checkpoint_temps(config: Config) -> list[str]:
    checkpoint_dir = config.train.workdir / "checkpoints"
    if not checkpoint_dir.exists():
        return []

    removed = []
    for path in checkpoint_dir.glob("*.orbax-checkpoint-tmp"):
        if not path.is_dir():
            continue
        shutil.rmtree(path)
        removed.append(str(path))
    return removed


def make_checkpoint_manager(config: Config):
    policies = [
        ocp_managers.LatestN(n=1),
        ocp_managers.BestN(
            get_metric_fn=_validation_nll,
            reverse=True,
            n=1,
            keep_checkpoints_without_metrics=False,
        ),
    ]
    options = ocp.CheckpointManagerOptions(
        preservation_policy=ocp_managers.AnyPreservationPolicy(policies),
        # Setting best_fn makes Orbax persist the metrics supplied with each
        # checkpoint. The explicit preservation policy above additionally
        # retains the latest checkpoint when it is not the validation best.
        best_fn=_validation_nll,
        best_mode="min",
        keep_checkpoints_without_metrics=False,
        create=True,
        enable_async_checkpointing=True,
    )
    return ocp.CheckpointManager(
        config.train.workdir / "checkpoints",
        ocp.StandardCheckpointer(),
        options,
    )


def save_checkpoint(
    manager,
    step: int,
    state: TrainState,
    *,
    metrics: Mapping[str, Any] | None,
) -> bool:
    # CheckpointManager stages/finalizes this save asynchronously. A later save
    # waits only if this one is still pending; normal completion waits once at
    # the end of the training loop.
    serialized_metrics = None if metrics is None else dict(metrics)
    return manager.save(step, {"state": state}, metrics=serialized_metrics, force=True)


def restore_latest_checkpoint(manager, state: TrainState) -> tuple[TrainState, int | None]:
    step = manager.latest_step()
    if step is None:
        return state, None
    restored = manager.restore(step, args=ocp.args.StandardRestore({"state": state}))
    return restored["state"], int(step)
