"""Entry point: python -m experiments.unnormalized_attention.train CONFIG.yaml."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from experiments.unnormalized_attention.configs import load_config
from experiments.unnormalized_attention.configs.base import validate_config
from experiments.unnormalized_attention.training.loop import run_training


def should_resume(*, no_resume: bool, slurm_restart_count: str | None) -> bool:
    """A Slurm requeue resumes even if the original attempt used --no-resume."""
    restart_count = int(slurm_restart_count or "0")
    if restart_count < 0:
        raise ValueError("SLURM_RESTART_COUNT must be non-negative")
    return not no_resume or restart_count > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain with exponential attention, optionally normalized for a softmax control.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--stop-after-step", type=int, help="Stop with a resume checkpoint; retain the full learning-rate schedule.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.workdir is not None:
        config.train.workdir = args.workdir
    if args.wandb_mode is not None:
        config.train.wandb_mode = args.wandb_mode
    if config.train.source_commit == "auto":
        source = os.environ.get("UNNORMALIZED_ATTENTION_SOURCE_COMMIT")
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
