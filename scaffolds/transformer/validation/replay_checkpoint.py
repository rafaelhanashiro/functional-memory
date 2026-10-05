"""Replay a training run from one of its retained checkpoints in a scratch directory.

The run's own files are only read. The scratch directory receives the run's recipe with
its paths moved there and W&B disabled, its data identity, its validation history up to
the checkpoint, and the checkpoint. Training resumes from the checkpoint and stops at
--stop, so a numerical failure that recurs is caught by the same per-update checks as in
production. The last stdout line is a JSON summary prefixed with REPLAY_RESULT.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import yaml


def prepare(run: Path, checkpoint: int, scratch: Path) -> Path:
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / "checkpoints").mkdir(parents=True)
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    recipe["train"].update(workdir=str(scratch), checkpoint_dir=str(scratch / "checkpoints"), wandb_mode="disabled")
    (scratch / "config.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    for name in ("data_identity.json", "best_validation.json"):
        shutil.copy2(run / name, scratch / name)
    kept = [line for line in (run / "metrics.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line).get("step", 0) <= checkpoint]
    (scratch / "metrics.jsonl").write_text("".join(line + "\n" for line in kept))
    shutil.copytree(run / "checkpoints" / str(checkpoint), scratch / "checkpoints" / str(checkpoint))
    return scratch / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory with the retained checkpoint")
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True, help="last update to train")
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()

    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.training import loop

    recipe = prepare(args.run.resolve(), args.checkpoint, args.scratch.resolve())
    start = time.time()
    try:
        outcome = loop.run_training(load_config(recipe), resume=True, stop_after_step=args.stop)
        result = {"status": "passed", "final_step": outcome.get("final_step")}
    except FloatingPointError as error:
        failures = sorted(args.scratch.resolve().glob("numerical_failure_step_*.json"))
        failure = json.loads(failures[-1].read_text()) if failures else {}
        result = {"status": "numerical_failure", "error": str(error), "step": failure.get("step"),
                  "update_numerics": failure.get("update_numerics")}
    result["seconds"] = round(time.time() - start, 1)
    print("REPLAY_RESULT " + json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
