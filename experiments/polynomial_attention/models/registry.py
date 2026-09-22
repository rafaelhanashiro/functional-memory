"""Compatibility import for :mod:`experiments.shared_transformer.models.registry`."""

from importlib import import_module
import sys

# Preserve module identity so private imports and monkeypatches use the shared code.
sys.modules[__name__] = import_module("experiments.shared_transformer.models.registry")
