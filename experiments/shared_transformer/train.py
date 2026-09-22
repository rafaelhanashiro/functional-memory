"""Entry point: python -m experiments.shared_transformer.train CONFIG.yaml."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from experiments.shared_transformer.configs import Config, load_config
from experiments.shared_transformer.configs.base import validate_config
from experiments.shared_transformer.training.loop import run_training
from experiments.shared_transformer.training.metrics import write_json_atomic


def should_resume(*, no_resume: bool, slurm_restart_count: str | None) -> bool:
    """A Slurm requeue resumes even if the original attempt used --no-resume."""
    restart_count = int(slurm_restart_count or "0")
    if restart_count < 0:
        raise ValueError("SLURM_RESTART_COUNT must be non-negative")
    return not no_resume or restart_count > 0


def attention_degree_argument(value: str) -> int | None:
    if value == "softmax":
        return None
    try:
        degree = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an integer degree or 'softmax'") from error
    if degree < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer degree")
    return degree


def main(argv=None, *, config_factory=Config,
         source_commit_env="SHARED_TRANSFORMER_SOURCE_COMMIT") -> None:
    parser = argparse.ArgumentParser(description="Pretrain a Transformer with configurable attention.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--attention-degree", type=attention_degree_argument, default=argparse.SUPPRESS,
                        help="Nonnegative degree (odd degrees permit signed weights), or softmax.")
    parser.add_argument("--qk-head-dim", type=int, help="Query/key channels per head.")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--stop-after-step", type=int, help="Stop with a resume checkpoint; retain the full learning-rate schedule.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    parser.add_argument("--result-json", type=Path, help="Write the training result, including synchronized pilot timings.")
    args = parser.parse_args(argv)
    config = load_config(args.config, config_factory=config_factory)
    if hasattr(args, "attention_degree"):
        config.model.attention_degree = args.attention_degree
    if args.qk_head_dim is not None:
        config.model.qk_head_dim = args.qk_head_dim
    if args.workdir is not None:
        config.train.workdir = args.workdir
    if args.wandb_mode is not None:
        config.train.wandb_mode = args.wandb_mode
    if config.train.source_commit == "auto":
        source = os.environ.get(source_commit_env)
        if source is None:
            source = subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[2], text=True
            ).strip()
        if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
            raise ValueError(f"Invalid source commit: {source!r}")
        config.train.source_commit = source
    validate_config(config)
    result = run_training(
        config,
        resume=should_resume(no_resume=args.no_resume, slurm_restart_count=os.environ.get("SLURM_RESTART_COUNT")),
        stop_after_step=args.stop_after_step,
    )
    if args.result_json is not None:
        write_json_atomic(args.result_json, result)


if __name__ == "__main__":
    main()
