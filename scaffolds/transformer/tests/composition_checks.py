"""Refusal checks after the attention-pieces refactor (attention/pieces.py).

On the blocked backend every unnormalized polynomial with RoPE or no positional encoding composes with any
forget gate, score transform, score biases and prefactor (configs.base._blocked_polynomial). A combination
the scaffold once refused everywhere is therefore refused only where no kernel implements it: on the fast
backends, whose declarations name what their kernels implement.
"""

import re

import pytest


def refused_unless_composed(build, changes, *, match=None):
    """``build(**changes)`` raises ValueError (matching ``match``), unless the recipe composes on the blocked
    backend: then it builds there, and the same recipe on the Mosaic backend is refused."""
    from scaffolds.transformer.configs.base import _blocked_polynomial

    try:
        config = build(**changes)
    except ValueError as error:
        if match is not None:
            assert re.search(match, str(error)), f"{changes}: {error}"
        return
    model = getattr(config, "model", config)
    assert _blocked_polynomial(model), f"{changes} was accepted although it does not compose on the blocked backend"
    with pytest.raises(ValueError):
        build(**{**changes, "attention_backend": "mosaic"})
