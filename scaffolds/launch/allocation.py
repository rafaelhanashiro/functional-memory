"""Read a Slurm job's granted wall-clock limit for runtime qualification."""
from __future__ import annotations

import argparse
import re
import subprocess


def parse_time_limit(value: str) -> int:
    """Parse scontrol's finite ``[days-]hours:minutes:seconds`` format."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", value)
    if match is None:
        raise ValueError(f"Expected a finite Slurm TimeLimit, got {value!r}")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if match.group(1) is not None and hours >= 24:
        raise ValueError(f"Invalid hours in Slurm TimeLimit {value!r}")
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ValueError(f"Slurm TimeLimit must be positive, got {value!r}")
    return total


def allocation_seconds(job_id: str) -> int:
    """Use the allocated limit, including any reduction made by Slurm backfill."""
    allocation = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
    limits = [field.partition("=")[2] for field in allocation.split() if field.startswith("TimeLimit=")]
    if len(limits) != 1:
        raise ValueError(f"Expected exactly one TimeLimit for Slurm job {job_id}")
    return parse_time_limit(limits[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    print(allocation_seconds(parser.parse_args().job_id))


if __name__ == "__main__":
    main()
