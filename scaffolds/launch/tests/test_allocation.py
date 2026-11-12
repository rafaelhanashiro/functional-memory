"""Scheduler wall-clock budgets must reflect the granted allocation."""
from pathlib import Path

import pytest

from scaffolds.launch import allocation


@pytest.mark.parametrize(("value", "seconds"), [
    ("00:00:01", 1), ("06:00:00", 6 * 3600), ("48:00:00", 48 * 3600),
    ("2-00:00:00", 48 * 3600), ("7-00:00:00", 168 * 3600),
    ("1-03:02:01", 27 * 3600 + 121),
])
def test_finite_scheduler_time_limits(value, seconds):
    assert allocation.parse_time_limit(value) == seconds


@pytest.mark.parametrize("value", ["", "UNLIMITED", "NOT_SET", "N/A", "0", "00:00:00",
                                   "-1:00:00", "01:60:00", "01:00:60", "1-24:00:00", "1.5:00:00"])
def test_missing_unlimited_or_invalid_scheduler_time_limits_are_rejected(value):
    with pytest.raises(ValueError):
        allocation.parse_time_limit(value)


def test_reads_granted_time_limit_from_scheduler(monkeypatch):
    def show(command, **kwargs):
        assert command == ["scontrol", "show", "job", "-o", "123"]
        assert kwargs == {"text": True}
        return "JobId=123 TimeLimit=06:00:00 TimeMin=06:00:00 RunTime=00:10:00\n"

    monkeypatch.setattr(allocation.subprocess, "check_output", show)
    assert allocation.allocation_seconds("123") == 6 * 3600


@pytest.mark.parametrize("output", ["JobId=123 TimeMin=06:00:00", "TimeLimit=01:00:00 TimeLimit=02:00:00"])
def test_scheduler_must_return_exactly_one_limit(monkeypatch, output):
    monkeypatch.setattr(allocation.subprocess, "check_output", lambda *args, **kwargs: output)
    with pytest.raises(ValueError, match="exactly one"):
        allocation.allocation_seconds("123")


def test_only_kennel_limits_full_training_to_the_current_allocation():
    root = Path(__file__).resolve().parents[3]
    kennel = (root / "scaffolds/transformer/launchers/kennel.sbatch").read_text()
    orcd = (root / "scaffolds/transformer/launchers/orcd.sbatch").read_text()
    assert 'scaffolds/launch/allocation.py "$SLURM_JOB_ID"' in kennel
    assert "export TRANSFORMER_ALLOCATION_SECONDS" in kennel
    # ORCD resumes across allocations, including limits reduced by backfill. Its
    # full-training qualification horizon must not become one allocation's limit.
    assert "TRANSFORMER_ALLOCATION_SECONDS" not in orcd
    assert "scaffolds/launch/allocation.py" not in orcd
