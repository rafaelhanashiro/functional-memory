"""Compatibility alias for the shared fused_taylor execution backend."""

import sys

from scaffolds.transformer.attention.backends import fused_taylor as _implementation

# Preserve private numerical-test helpers and historical monkeypatch seams.
sys.modules[__name__] = _implementation
