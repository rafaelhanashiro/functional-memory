"""Bound CPU test pools on large login nodes; leave Slurm allocations intact.

Tests also start outside a launch even when run by a launcher's acceptance stage;
tests that model a launch set its source commit themselves.
"""
import os

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
if not os.environ.get("SLURM_JOB_ID") and hasattr(os, "sched_getaffinity"):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) > 8:
        os.sched_setaffinity(0, allowed[:8])


@pytest.fixture(autouse=True)
def _outside_a_launch(monkeypatch):
    monkeypatch.delenv("SHARED_TRANSFORMER_SOURCE_COMMIT", raising=False)
