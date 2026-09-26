"""Regenerate this record's summary, figures, and README results block from runs/.

Until every run has concluded (a completion record or a numerical-failure record),
the report writes nothing and prints ``pending``, so no derived output can be
fabricated from partial runs. The full analysis lands with the completed record.
"""
from __future__ import annotations

import argparse
from pathlib import Path

RECORD = Path(__file__).resolve().parents[2]
DEGREES = (1, 2, 3, 4)
VARIANTS = ("unnormalized", "normalized")
RUNS = tuple(f"200m-qk-m{degree:03d}-{variant}-seed0" for degree in DEGREES for variant in VARIANTS)


def concluded(run_directory: Path) -> bool:
    """A run has concluded when it completed or stopped with a recorded numerical failure."""
    return (run_directory / "completion.json").exists() or any(run_directory.glob("numerical_failure_step_*.json"))


def completed(record: Path = RECORD) -> bool:
    return all(concluded(record / "runs" / run) for run in RUNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write outputs here instead of the record (regeneration checks)")
    args = parser.parse_args()
    if not completed():
        done = sum(concluded(RECORD / "runs" / run) for run in RUNS)
        print(f"pending: {done}/{len(RUNS)} runs concluded; no summary, figures, or results block written")
        return
    raise NotImplementedError("The analysis for the concluded sweep is added with the completed record")


if __name__ == "__main__":
    main()
