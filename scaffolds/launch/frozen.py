"""Check and hash an experiment's frozen execution set against its launch commit.

While an experiment's jobs can run or requeue, what they execute must not change:
the scaffold's code (without its validation record), shared record/launch helpers,
the study's sweep.yaml, each run's config.yaml, and study execution modules.
Run records, metrics, logs, checkpoints, and results may change freely.
Launchers call `check` at every start and requeue; record.json records `hash`.

Standard library only: every scaffold's launcher runs this with the system Python.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


def pathspecs(experiment: str, scaffold: str) -> list[str]:
    """Candidate roots; the same predicate filters diffs and committed trees."""
    return [f"scaffolds/{scaffold}", "scaffolds/records", "scaffolds/launch",
            f"experiments/{experiment}"]


def is_frozen(path: str, experiment: str, scaffold: str) -> bool:
    for prefix in ("scaffolds/records/", "scaffolds/launch/"):
        if path.startswith(prefix):
            return True
    if path.startswith(f"scaffolds/{scaffold}/"):
        return not path.startswith(f"scaffolds/{scaffold}/validation/")
    prefix = f"experiments/{experiment}/"
    if not path.startswith(prefix):
        return False
    relative = Path(path.removeprefix(prefix))
    parts = relative.parts
    return (
        relative.as_posix() == "sweep.yaml"
        or len(parts) == 1 and relative.suffix in {".py", ".sbatch", ".sh"}
        or len(parts) == 3 and parts[0] == "runs" and parts[2] == "config.yaml"
        # Historical layouts remain protected when auditing their source commits.
        or parts[0] == "configs"
        or parts[:2] == ("adapter", "execution")
    )


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)


def changes(root: Path, launch_commit: str, experiment: str, scaffold: str) -> list[str]:
    """Frozen paths that differ from the launch commit, committed or not."""
    specs = pathspecs(experiment, scaffold)
    diff = _git(root, "diff", "--name-only", launch_commit, "--", *specs)
    if diff.returncode:
        raise RuntimeError(f"Cannot compare with launch commit {launch_commit}: {diff.stderr.strip()}")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *specs)
    candidates = set(diff.stdout.splitlines()) | set(untracked.stdout.splitlines())
    return sorted(path for path in candidates if is_frozen(path, experiment, scaffold))


def frozen_hash(root: Path, launch_commit: str, experiment: str, scaffold: str) -> str:
    """SHA256 of the frozen set's tree entries at the launch commit."""
    tree = _git(root, "ls-tree", "-r", launch_commit, "--", *pathspecs(experiment, scaffold))
    if tree.returncode:
        raise RuntimeError(tree.stderr.strip())
    entries = [line for line in tree.stdout.splitlines()
               if is_frozen(line.split("\t", 1)[1], experiment, scaffold)]
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
