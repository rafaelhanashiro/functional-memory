"""Shared test process settings: bounded CPU pools and bounded memory mappings.

CPU test pools are bounded on large login nodes; Slurm allocations stay intact.
Tests also start outside a launch even when run by a launcher's acceptance stage;
tests that model a launch set its source commit themselves.

Cached XLA executables keep memory mappings, so one process running a whole test
stage can exhaust ``vm.max_map_count`` (Linux default 65,530) and fail a later
compile. After any test that leaves the process above half that limit, JAX's caches
are cleared; tests stop with an explicit error if clearing cannot keep the process
below nine tenths of it.
"""
import gc
import os
import warnings

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
if not os.environ.get("SLURM_JOB_ID") and hasattr(os, "sched_getaffinity"):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) > 8:
        os.sched_setaffinity(0, allowed[:8])


def _read(path: str) -> bytes | None:
    try:
        with open(path, "rb") as stream:
            return stream.read()
    except OSError:
        return None


def memory_map_count() -> int | None:
    maps = _read("/proc/self/maps")
    return None if maps is None else maps.count(b"\n")


_LIMIT = _read("/proc/sys/vm/max_map_count")
MEMORY_MAP_LIMIT = int(_LIMIT) if _LIMIT and _LIMIT.strip().isdigit() else 65530
MEMORY_MAP_BUDGET = MEMORY_MAP_LIMIT // 2


@pytest.fixture(autouse=True)
def _outside_a_launch(monkeypatch):
    monkeypatch.delenv("SHARED_TRANSFORMER_SOURCE_COMMIT", raising=False)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    # After every fixture of the test is torn down, so its executables are unreferenced.
    result = yield
    before = memory_map_count()
    if before is None or before <= MEMORY_MAP_BUDGET:
        return result
    import jax

    jax.clear_caches()
    gc.collect()
    after = memory_map_count()
    if after is None:
        return result
    if after > MEMORY_MAP_LIMIT * 9 // 10:
        pytest.exit(f"{after} memory mappings of vm.max_map_count {MEMORY_MAP_LIMIT} remain after "
                    f"clearing JAX caches following {item.nodeid}", returncode=1)
    if after > MEMORY_MAP_BUDGET:
        warnings.warn(f"{after} memory mappings remain above {MEMORY_MAP_BUDGET} after clearing "
                      f"JAX caches following {item.nodeid}")
    return result


def pytest_addoption(parser):
    parser.addoption("--mosaic-precision", choices=("bf16", "bf16x3"),
                     help="Run Mosaic kernel tests only at this weight precision (acceptance passes the run's).")


def pytest_collection_modifyitems(config, items):
    precision = config.getoption("--mosaic-precision")
    if precision is None:
        return
    keep, drop = [], []
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        (drop if params.get("precision", precision) != precision else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
