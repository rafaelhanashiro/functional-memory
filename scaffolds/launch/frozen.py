"""Check and hash an experiment's frozen execution set against its launch commit.

While an experiment's jobs can run or requeue, what they execute must not change:
the scaffold's code (without its validation record), the experiment's `configs/`,
and its adapter's execution modules. The README, figures, summaries, and analysis
modules may change freely, and `runs/`, `logs/`, and `checkpoints/` are outputs.
Launchers call `check` at every start and requeue; the manifest records `hash`.

Standard library only: every scaffold's launcher runs this with the system Python.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


def pathspecs(experiment: str, scaffold: str) -> list[str]:
    return [f"scaffolds/{scaffold}", f":(exclude)scaffolds/{scaffold}/validation",
            f"experiments/{experiment}/configs", f"experiments/{experiment}/adapter/execution"]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)


def changes(root: Path, launch_commit: str, experiment: str, scaffold: str) -> list[str]:
    """Frozen paths that differ from the launch commit, committed or not."""
    specs = pathspecs(experiment, scaffold)
    diff = _git(root, "diff", "--name-only", launch_commit, "--", *specs)
    if diff.returncode:
        raise RuntimeError(f"Cannot compare with launch commit {launch_commit}: {diff.stderr.strip()}")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *specs)
    return sorted(set(diff.stdout.split()) | set(untracked.stdout.split()))


def frozen_hash(root: Path, launch_commit: str, experiment: str, scaffold: str) -> str:
    """SHA256 of the frozen set's tree entries at the launch commit."""
    included = [spec for spec in pathspecs(experiment, scaffold) if not spec.startswith(":(exclude)")]
    tree = _git(root, "ls-tree", "-r", launch_commit, "--", *included)
    if tree.returncode:
        raise RuntimeError(tree.stderr.strip())
    # ls-tree has no exclude pathspecs; drop the validation record's entries here.
    excluded = f"\tscaffolds/{scaffold}/validation/"
    entries = [line for line in tree.stdout.splitlines() if excluded not in line]
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "hash"))
    parser.add_argument("--launch-commit", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--scaffold", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "hash":
        print(frozen_hash(args.root, args.launch_commit, args.experiment, args.scaffold))
        return 0
    changed = changes(args.root, args.launch_commit, args.experiment, args.scaffold)
    if changed:
        print("Frozen execution set changed since launch; refusing to run:", *changed, sep="\n  ", file=sys.stderr)
        return 1
    print(f"frozen_set_unchanged_since={args.launch_commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
