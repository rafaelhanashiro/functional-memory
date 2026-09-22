"""Compare with immutable pre-refactor code, independently of compatibility aliases.

The original code is exported from its recorded Git commit into a temporary
directory and executed with the test interpreter. No execution checkout,
historical result, or retained checkpoint is imported or modified.
"""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
PROBE = Path(__file__).with_name("frozen_probe.py")
REFERENCES = {
    "key_dimension": "c2b0d6e5514995cbe7b5e564f902201cfde1ba4e",
    "polynomial_attention": "2e0e88d9cc955b57cd07cdd40a743d2c49bafd4c",
}
VARIANTS = [
    ("key_dimension", {"attention_backend": "manual"}),
    ("key_dimension", {"attention_backend": "xla"}),
    ("polynomial_attention", {"attention_backend": "blocked", "attention_degree": None}),
    *[("polynomial_attention", {"attention_backend": "blocked", "attention_degree": degree})
      for degree in (0, 1, 3, 4, 16)],
    *[("polynomial_attention", {"attention_backend": "blocked", "attention_degree": 2,
                                "quadratic_mode": mode, "quadratic_sharing": sharing})
      for mode in ("positive", "unrestricted") for sharing in ("global", "per_head")],
]


@pytest.fixture(scope="module")
def frozen_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen-scaffolds")
    for family, commit in REFERENCES.items():
        destination = root / family
        destination.mkdir()
        archive = subprocess.run(
            ["git", "archive", commit, f"experiments/{family}"], cwd=ROOT,
            check=True, capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(destination, filter="data")
    return root


def probe(root, package, settings, output, checkpoint, action="inspect"):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(root),
               OMP_NUM_THREADS="2", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    subprocess.run(
        [sys.executable, str(PROBE), package, json.dumps(settings), str(output),
         str(checkpoint), action], cwd=root, env=env, check=True,
        capture_output=True, text=True, timeout=120,
    )
    with np.load(output) as saved:
        return dict(saved)


def compare(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert np.isfinite(left[name]).all(), name
        assert np.isfinite(right[name]).all(), name
        if name.startswith("initial:"):
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)
        else:
            np.testing.assert_allclose(left[name], right[name], rtol=3e-5, atol=2e-7, err_msg=name)


@pytest.mark.parametrize("family,settings", VARIANTS)
def test_frozen_forward_gradients_and_optimizer_parity(frozen_sources, tmp_path, family, settings):
    expected = probe(frozen_sources / family, f"experiments.{family}", settings,
                     tmp_path / "old.npz", tmp_path / "old")
    actual = probe(ROOT, "experiments.shared_transformer", settings,
                   tmp_path / "new.npz", tmp_path / "new")
    compare(expected, actual)


@pytest.mark.parametrize("family,settings", [VARIANTS[0], VARIANTS[5], VARIANTS[8], VARIANTS[11]])
def test_frozen_checkpoint_restores_and_continues(frozen_sources, tmp_path, family, settings):
    checkpoint = tmp_path / "checkpoint"
    probe(frozen_sources / family, f"experiments.{family}", settings,
          tmp_path / "saved.npz", checkpoint, "save")
    expected = probe(frozen_sources / family, f"experiments.{family}", settings,
                     tmp_path / "old.npz", checkpoint, "restore")
    actual = probe(ROOT, "experiments.shared_transformer", settings,
                   tmp_path / "new.npz", checkpoint, "restore")
    compare(expected, actual)
