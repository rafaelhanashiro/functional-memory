"""Keep only the validation-best checkpoint once a run has completed.

The trainer retains the latest and best checkpoints while a run can resume.
After completion, :func:`finalize_run` verifies the terminal record, prunes every
other checkpoint step, and records the selection in `retention.json` and the
run metadata. It is idempotent, so a requeue after pruning is harmless.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scaffolds.transformer.configs import load_config
from scaffolds.transformer.training.checkpointing import checkpoint_directory
from scaffolds.transformer.training.metrics import write_json_atomic

# Completion fields that must agree with the recipe when the completion records them.
IDENTITY_KEYS = ("attention_degree", "attention_mechanism", "quadratic_mode", "quadratic_sharing", "qk_head_dim")


def finalize_run(config) -> dict:
    workdir = Path(config.train.workdir)
    completion_path = workdir / "completion.json"
    completion = json.loads(completion_path.read_text())
    best = json.loads((workdir / "best_validation.json").read_text())
    tokens = config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1)
    model = vars(config.model)
    if (completion["source_commit"] != config.train.source_commit
            or completion["final_step"] != config.optim.total_steps
            or completion["training_tokens"] != tokens
            or completion["retained_best_checkpoint_step"] != best["step"]
            or any(key in completion and completion[key] != model[key] for key in IDENTITY_KEYS)):
        raise ValueError("Cannot finalize an incomplete or mismatched run")
    checkpoints = checkpoint_directory(config).resolve()
    if not (checkpoints / str(best["step"])).is_dir():
        raise ValueError("Validation-best checkpoint is missing from checkpoint storage")
    records = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not any(row["step"] == completion["final_step"] and row["validation_nll"] == completion["validation_nll"]
               for row in records):
        raise ValueError("Terminal validation has not been durably recorded")
    candidates = [row for row in records if row.get("event") == "validation"]
    selected = min(candidates, key=lambda row: (row["validation_nll"], -row["step"]))
    if selected["step"] != best["step"] or selected["validation_nll"] != best["validation_nll"]:
        raise ValueError("Selected checkpoint does not match the best committed validation")
    record = {
        "format": "checkpoint-retention", "version": 1,
        "source_commit": config.train.source_commit, "seed": config.train.seed,
        "completion_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        "selection_rule": "lowest_validation_nll_latest_on_tie", "best_checkpoint_step": best["step"],
        "validation_nll": best["validation_nll"], "terminal_step": completion["final_step"],
        "terminal_metrics_preserved": True,
    }
    # Record the intended retention before pruning, so a requeue after any deletion
    # recognizes that terminal metrics are complete even when the best step is earlier.
    write_json_atomic(workdir / "retention.json", record)
    for path in checkpoints.iterdir():
        if path.is_dir() and ((path.name.isdigit() and int(path.name) != best["step"])
                              or path.name.endswith(".orbax-checkpoint-tmp")):
            shutil.rmtree(path)
    metadata_path = workdir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(checkpoint_retention=record,
                    retained_terminal_checkpoint_step=best["step"] if best["step"] == completion["final_step"] else None)
    write_json_atomic(metadata_path, metadata)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="The run's recipe, with train.source_commit set to the executed commit")
    parser.add_argument("--source-commit", help="Executed source commit when the recipe says 'auto'")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.source_commit:
        config.train.source_commit = args.source_commit
    print(json.dumps(finalize_run(config), sort_keys=True))


if __name__ == "__main__":
    main()
