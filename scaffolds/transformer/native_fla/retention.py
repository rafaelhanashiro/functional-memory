"""Retain resumable progress during training and validation best after completion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpointing import latest_checkpoint, prune_checkpoints


def retain_checkpoints(root: str | Path, *, best_step: int | None, completed: bool = False) -> list[Path]:
    latest = latest_checkpoint(root)
    if completed and best_step is None:
        raise ValueError("Completed training must identify a validation-best checkpoint")
    if latest is None:
        if best_step is not None:
            raise ValueError("Validation-best checkpoint is missing")
        return []
    keep = set() if completed else {int(latest.name)}
    if best_step is not None:
        keep.add(best_step)
    return prune_checkpoints(root, keep)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="run's checkpoints directory")
    parser.add_argument("--best-step", required=True, type=int)
    parser.add_argument("--completed", action="store_true")
    args = parser.parse_args(argv)
    removed = retain_checkpoints(args.root, best_step=args.best_step, completed=args.completed)
    print(json.dumps({"removed_steps": [int(path.name) for path in removed]}))


if __name__ == "__main__":
    main()
