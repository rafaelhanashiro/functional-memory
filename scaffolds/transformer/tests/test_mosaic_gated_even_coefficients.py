"""Mosaic degree-eight gates at key width four against an independent FP64 forward/backward."""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic.api import default_precision, query_polynomial_attention
from scaffolds.transformer.attention.input_dependent import normalize_coefficient_input, project_coefficients
from scaffolds.transformer.attention.qk_norm import unit_cosine_score_operands
from scaffolds.transformer.tests.test_fused_attention import _assert_relative
from scaffolds.transformer.tests.test_mosaic_query import _assert_all, _inputs, _numpy_fp64, _vjp

ORDERS = (2, 4, 6, 8)


@pytest.mark.gpu
@pytest.mark.parametrize(("cosine", "taylor"), [(True, False), (False, False), (False, True)])
def test_even_gate_mosaic_matches_fp64(cosine, taylor):
    assert jax.default_backend() == "gpu"
    (q, k, v, _), cotangent = _inputs(8, 4)
    # A positive value/cotangent pairing keeps each tiny Taylor gate's bias
    # gradient measurable after summing over the token dimension.
    v, cotangent = jnp.abs(v), jnp.abs(cotangent)
    if cosine:
        q, k = unit_cosine_score_operands(q, k, eps=1e-6)
    q = q.at[:, 0].set(0)
    k = k.at[:, 1].set(-k[:, 0])
    # A fully zero row, arbitrary signed scores, and a pair of antiparallel keys.
    rng = np.random.default_rng(47)
    x = rng.standard_normal((*q.shape[:2], 32), dtype=np.float32)
    w = rng.standard_normal((q.shape[2], 32, len(ORDERS)), dtype=np.float32) / math.sqrt(32)
    offsets = -np.log([math.factorial(r) - 1 for r in ORDERS]) if taylor else np.zeros(4)
    b = np.tile(offsets, (q.shape[2], 1)).astype(np.float32)
    norm = np.linalg.norm(x.astype(np.float64), axis=-1, keepdims=True)
    unit = x / norm
    logits = np.einsum("btd,hdr->bthr", unit, w) + b
    gates = 1 / (1 + np.exp(-logits))
    c = np.zeros((*q.shape[:-1], 9), np.float32)
    c[..., list(ORDERS)] = gates
    attend = partial(query_polynomial_attention, precision=default_precision("query", 8))
    operands = (q, k, v, jnp.asarray(c))
    actual = jax.jit(partial(_vjp, attend))(operands, cotangent)
    expected = _numpy_fp64(operands, cotangent)
    _assert_all(actual, expected, f"cosine={cosine}, Taylor={taylor}")
    np.testing.assert_array_equal(np.asarray(actual[0][:, 0], dtype=np.float32), 0.)

    def gated(x, w, b):
        c = project_coefficients(normalize_coefficient_input(x, "l2"), w, None,
                                 activation="sigmoid", sigmoid_logit_bias=b, orders=ORDERS)
        return attend(q, k, v, c)

    out, (dx, dw, db) = jax.jit(partial(_vjp, gated))((jnp.asarray(x), jnp.asarray(w), jnp.asarray(b)), cotangent)
    _assert_relative(out, expected[0], rtol=.006, name="gated output")
    dz = expected[1][3][..., list(ORDERS)] * gates * (1 - gates)
    dunit = np.einsum("bthr,hdr->btd", dz, w)
    dx64 = (dunit - unit * (unit * dunit).sum(axis=-1, keepdims=True)) / norm
    # The comparison helper measures [batch, time, head, channel] per head.
    _assert_relative(dx[:, :, None, :], dx64[:, :, None, :], rtol=.025, name="raw L2 input gradient")
    dw64 = np.einsum("btd,bthr->hdr", unit, dz)
    db64 = dz.sum(axis=(0, 1))
    for channel, r in enumerate(ORDERS):
        _assert_relative(dw[None, None, :, :, channel], dw64[None, None, :, :, channel],
                         rtol=.025, name=f"gate W for order {r}")
        _assert_relative(db[None, None, :, channel, None], db64[None, None, :, channel, None],
                         rtol=.025, name=f"gate bias for order {r}")
