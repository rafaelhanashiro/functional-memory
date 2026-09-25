import json
import shutil
from collections.abc import Mapping
from typing import Any

import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as ocp_managers

from scaffolds.transformer.configs import Config
from scaffolds.transformer.training.state import TrainState


VALIDATION_NLL_METRIC = "validation_nll"


def checkpoint_directory(config: Config):
    """Use explicit artifact storage; preserve historical configs on resume."""
    from pathlib import Path

    if config.train.checkpoint_dir is not None:
        return Path(config.train.checkpoint_dir)
    return config.train.workdir / "checkpoints"


def validate_resume_progress(config: Config, latest_checkpoint_step: int | None) -> None:
    """Reject missing saved state before a resume can overwrite the run record.

    Validation records are committed only after their checkpoint is durable.
    The metadata RNG clock is likewise written at checkpointed boundaries; the
    top-level step of a numerical failure instead names an attempted update.
    Pending validation is deliberately excluded because its save may have failed.
    Completed runs, including best-only retention, have their own validation.
    """
    workdir = config.train.workdir
    if (workdir / "completion.json").exists():
        return

    recorded_step = 0
    metrics_path = workdir / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "validation":
                step = record.get("checkpoint_step", record["step"])
                recorded_step = max(recorded_step, int(step))

    best_path = workdir / "best_validation.json"
    if best_path.exists():
        recorded_step = max(recorded_step, int(json.loads(best_path.read_text())["step"]))

    metadata_path = workdir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        saved_rng_step = metadata.get("rng_state", {}).get("completed_updates", 0)
        recorded_step = max(recorded_step, int(saved_rng_step))
        if metadata.get("status") in {"running", "paused"}:
            recorded_step = max(recorded_step, int(metadata.get("step", 0)))

    available_step = latest_checkpoint_step if latest_checkpoint_step is not None else 0
    if recorded_step > available_step:
        available = "no checkpoint is available" if latest_checkpoint_step is None else (
            f"the latest available checkpoint is step {latest_checkpoint_step}"
        )
        raise ValueError(
            f"Cannot resume {workdir}: recorded completed progress is step {recorded_step}, "
            f"but {available}. Restore the missing checkpoints or use a new run directory."
        )


def state_sha256(state) -> str:
    """Fingerprint every finite array, including its path, shape, and dtype."""
    import hashlib
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.asarray(leaf)
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"Nonfinite checkpoint/audit array at {path}")
        digest.update(str(path).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validation_nll(metrics: Mapping[str, Any]) -> float:
    return float(metrics[VALIDATION_NLL_METRIC])


def cleanup_stale_checkpoint_temps(config: Config) -> list[str]:
    checkpoint_dir = checkpoint_directory(config)
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
        checkpoint_directory(config),
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
