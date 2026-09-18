"""Entry point: python -m experiments.key_dimension.train CONFIG.yaml."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from experiments.key_dimension.configs import load_config
from experiments.key_dimension.configs.base import validate_config
from experiments.key_dimension.training.loop import run_training


def should_resume(*, no_resume: bool, slurm_restart_count: str | None) -> bool:
    """A Slurm requeue resumes even if the original attempt used --no-resume."""
    restart_count = int(slurm_restart_count or "0")
    if restart_count < 0:
        raise ValueError("SLURM_RESTART_COUNT must be non-negative")
    return not no_resume or restart_count > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a Transformer at a chosen query/key width.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--qk-head-dim", type=int)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--stop-after-step", type=int, help="Stop with a resume checkpoint; retain the full learning-rate schedule.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.qk_head_dim is not None:
        config.model.qk_head_dim = args.qk_head_dim
    if args.workdir is not None:
        config.train.workdir = args.workdir
    if args.wandb_mode is not None:
        config.train.wandb_mode = args.wandb_mode
    if config.train.source_commit == "auto":
        source = os.environ.get("KEY_DIMENSION_SOURCE_COMMIT")
        if source is None:
            source = subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[2], text=True
            ).strip()
        if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
            raise ValueError(f"Invalid source commit: {source!r}")
        config.train.source_commit = source
    validate_config(config)
    run_training(
        config,
        resume=should_resume(no_resume=args.no_resume, slurm_restart_count=os.environ.get("SLURM_RESTART_COUNT")),
        stop_after_step=args.stop_after_step,
    )


if __name__ == "__main__":
    main()
