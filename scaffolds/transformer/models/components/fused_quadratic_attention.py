"""Compatibility alias for the shared fused_quadratic execution backend."""

import sys

from scaffolds.transformer.attention.backends import fused_quadratic as _implementation

# Preserve private numerical-test helpers and historical monkeypatch seams.
sys.modules[__name__] = _implementation
