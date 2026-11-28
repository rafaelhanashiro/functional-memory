"""Prepare and qualify the common prefix and two matched static-coefficient continuations."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scaffolds.records import create_run, load_run, recipe_receipt, regenerate_index, update_record
from scaffolds.transformer.acceptance import attention_spec, main as shared_acceptance
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.loader import config_to_dict

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
SOURCE = STUDY / "runs/r6f55fe172f41"
FORK_STEP = 500
END_STEP = 2000
LABELS = {
    "prefix": "static-fork-prefix-m8-step500",
    "full": "static-fork-full-m8-500-2000",
    "frozen": "static-fork-frozen-m8-500-2000",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fork_checkpoint(prefix: Path) -> dict:
    record = load_run(prefix)
    if (record.get("experiment_kind") != "static_coefficient_fork" or record.get("condition") != "prefix"
            or record.get("status") != "complete" or record.get("source_run_id") != SOURCE.name):
        raise ValueError("Both arms require a completed common prefix of the instrumented static recipe")
    checkpoint = record["selected_checkpoint"]
    if checkpoint["step"] != FORK_STEP or record["retained_terminal_checkpoint_step"] != FORK_STEP:
        raise ValueError("The prefix must retain its terminal update-500 checkpoint")
    identity = file_hash(prefix / "data_identity.json")
    if identity != file_hash(SOURCE / "data_identity.json"):
        raise ValueError("The prefix and source packed-data identities differ")
    if file_hash(prefix / "config.yaml") != record["config"]["sha256"]:
        raise ValueError("The common prefix recipe differs from its recorded hash")
    return {
        "parent_config": str((prefix / "config.yaml").relative_to(ROOT)),
        "parent_config_sha256": record["config"]["sha256"],
        "parent_checkpoint": str((prefix / "checkpoints" / str(FORK_STEP)).relative_to(ROOT)),
        "parent_checkpoint_sha256": checkpoint["sha256"],
        "parent_data_identity_sha256": identity, "parent_step": FORK_STEP,
        "preserve_all_parameters": True,
    }


def recipe(role: str, prefix: Path | None) -> dict:
    # Preserve the historical model's head RMSNorm arithmetic and every data and optimizer setting.
    raw = yaml.safe_load((SOURCE / "config.yaml").read_text())
    raw["optim"]["schedule_steps"] = raw["optim"]["total_steps"]
    raw["optim"]["total_steps"] = FORK_STEP if role == "prefix" else END_STEP
    raw["train"]["wandb_run_name"] = LABELS[role]
    if role == "prefix":
        raw["train"].update(eval_every=FORK_STEP, eval_early_every=None, eval_early_until=0)
    else:
        raw["train"].update(eval_every=100, eval_early_every=1, eval_early_until=FORK_STEP + 1)
        raw["continuation"].update(fork_checkpoint(prefix), freeze_static_coefficients=(role == "frozen"))
    return raw


def prepare(roles: tuple[str, ...], prefix: Path | None = None) -> dict[str, str]:
    # Repeating the command after a partial registration reuses the exact recipe.
    existing = {}
    for path in (STUDY / "runs").glob("*/record.json"):
        record = json.loads(path.read_text())
        if record.get("experiment_kind") == "static_coefficient_fork" or record.get("label") in LABELS.values():
            label = record["label"]
            if label in existing:
                raise ValueError(f"Ambiguous existing run label {label}")
            existing[label] = path.parent
    runs = {}
    source = load_run(SOURCE)
    if file_hash(SOURCE / "config.yaml") != source["config"]["sha256"]:
        raise ValueError("The instrumented static reference recipe has changed")
    # Tiny-model gradient receipts are bitwise comparable on the same CPU type.
    source_receipt = recipe_receipt(SOURCE / "config.yaml")
    for role in roles:
        settings = recipe(role, prefix)
        run = existing.get(LABELS[role])
        if run is None:
            run = create_run(STUDY, settings, label=LABELS[role])
        expected = deepcopy(settings)
        expected["train"].update(workdir=str(run.relative_to(ROOT)),
                                  checkpoint_dir=str((run / "checkpoints").relative_to(ROOT)))
        if yaml.safe_load((run / "config.yaml").read_text()) != expected:
            raise ValueError(f"Run {run.name} is already allocated with different settings")
        receipt = recipe_receipt(run / "config.yaml")
        if receipt != source_receipt:
            raise ValueError(f"Fork model at initialization differs from the source: {receipt}")
        metadata = {
            "experiment_kind": "static_coefficient_fork", "condition": role,
            "home_cluster": "kennel", "seed": settings["train"]["seed"], "receipt": receipt,
            "source_run_id": SOURCE.name, "source_configuration": source["config"],
        }
        if role != "prefix":
            metadata.update(parent_run_id=prefix.name, source_checkpoint=load_run(prefix)["selected_checkpoint"])
        update_record(run, metadata)
        runs[role] = run.name
        print(f"{role} {run.name}", flush=True)
    regenerate_index(STUDY)
    return runs


def check_recipe(config):
    source = load_config(SOURCE / "config.yaml")
    role = "prefix"
    if config.continuation.parent_checkpoint:
        role = "frozen" if config.continuation.freeze_static_coefficients else "full"
        prefix = (ROOT / config.continuation.parent_config).resolve().parent
        if prefix.parent != STUDY / "runs":
            raise ValueError("Expected a prefix from this study")
        parent = fork_checkpoint(prefix)
        for name, expected in parent.items():
            if getattr(config.continuation, name) != expected:
                raise ValueError(f"Unexpected fork checkpoint setting {name}")
        source.continuation = replace(source.continuation, **parent,
                                      freeze_static_coefficients=(role == "frozen"))
    source.optim = replace(source.optim, schedule_steps=source.optim.total_steps,
                           total_steps=FORK_STEP if role == "prefix" else END_STEP)
    source.train = replace(source.train, workdir=config.train.workdir, checkpoint_dir=config.train.checkpoint_dir,
                           source_commit=config.train.source_commit, wandb_run_name=LABELS[role],
                           eval_every=FORK_STEP if role == "prefix" else 100,
                           eval_early_every=None if role == "prefix" else 1,
                           eval_early_until=0 if role == "prefix" else FORK_STEP + 1)
    if config_to_dict(config) != config_to_dict(source):
        raise ValueError("The fork differs from its source outside the prescribed duration, evaluation cadence and treatment")


def check_pilot(config, outcome, validation):
    learned = np.asarray(validation["attention_diagnostics"]["coefficients"]["raw"], dtype=np.float32)
    if not config.continuation.parent_checkpoint:
        baseline = np.ones_like(learned)
    else:
        prefix_metrics = Path(config.continuation.parent_config).parent / "metrics.jsonl"
        prefix_row, = [r for r in map(json.loads, prefix_metrics.read_text().splitlines())
                       if r["step"] == FORK_STEP and r.get("event") == "validation"]
        baseline = np.asarray(prefix_row["attention_diagnostics"]["coefficients"]["raw"], dtype=np.float32)
    if learned.shape != baseline.shape or not np.isfinite(learned).all():
        raise ValueError("The pilot's static coefficient table is invalid")
    changed = int(np.count_nonzero(learned != baseline))
    if (changed == 0) != config.continuation.freeze_static_coefficients:
        raise ValueError(f"Unexpected static coefficient motion over the pilot, {changed} changed entries")
    return {"start_step": outcome["start_step"], "stop_step": validation["step"],
            "changed_coefficient_count": changed, "coefficient_count": int(learned.size)}


def spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_pilot=check_pilot,
                   cpu_tests=(*shared.cpu_tests, "scaffolds/transformer/tests/test_static_checkpoint_fork.py",
                              "experiments/transformer_pp/tests/test_static_coefficient_fork.py"))


def main():
    import sys
    if sys.argv[1:2] == ["prepare"]:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="phase", required=True)
        commands.add_parser("prefix")
        arms = commands.add_parser("arms")
        arms.add_argument("prefix_run_id")
        args = parser.parse_args(sys.argv[2:])
        if args.phase == "prefix":
            prepare(("prefix",))
        else:
            prepare(("full", "frozen"), STUDY / "runs" / args.prefix_run_id)
    else:
        shared_acceptance(partial(spec, entry_module=__name__))


if __name__ == "__main__":
    main()
