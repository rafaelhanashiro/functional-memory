"""Exercise packed-history addresses across the signed 32-bit element boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import triton
import triton.language as tl

from scaffolds.transformer.attention.backends.ejkernel_bridge import triton_call
from scaffolds.transformer.attention.backends.polynomial_gdn_kernels import (
    _history, _history_rows, _store_history,
)


@triton.jit
def _write_boundary_history(head_ids, history, T: tl.constexpr, K: tl.constexpr,
                            BT: tl.constexpr):
    case = tl.program_id(0)
    bh = tl.load(head_ids + case)
    rows, cols = tl.arange(0, BT), tl.arange(0, K)
    value = (case * BT * K + rows[:, None] * K + cols[None, :]).to(tl.float32)
    _store_history(history, value, bh, 31, 16, T, K, K, BT)


@triton.jit
def _read_boundary_history(head_ids, history, output, T: tl.constexpr, K: tl.constexpr,
                           BT: tl.constexpr, CASES: tl.constexpr):
    case = tl.program_id(0)
    bh = tl.load(head_ids + case)
    full = _history(history, bh, 31, 16, T, K, K, BT)
    sliced = _history_rows(history, bh, 31, 16, T, K, K, BT, BT)
    rows, cols = tl.arange(0, BT), tl.arange(0, K)
    offsets = (case * BT + rows[:, None]) * K + cols[None, :]
    tl.store(output + offsets, full)
    tl.store(output + CASES * BT * K + offsets, sliced)


@pytest.mark.gpu
def test_history_helpers_roundtrip_past_signed_int32_boundary():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a GPU allocation with room for an 8.25 GiB history")
    # At B64,H16,T2048,K64, head 992 / query chunk31 / key chunk16
    # begins at element 2**31. The neighboring heads exercise both sides.
    heads = jnp.asarray([991, 992, 1023], jnp.int32)
    length, width, chunk, pairs = 2048, 64, 64, 528
    history_shape = jax.ShapeDtypeStruct((64 * 16, pairs, chunk, width), jnp.float32)
    output_shape = jax.ShapeDtypeStruct((2, 3, chunk, width), jnp.float32)

    @jax.jit
    def roundtrip(head_ids):
        # Allocate one real production-size history, but write/read only three
        # tiles. Separate launches ensure both load helpers access device memory.
        history = triton_call(
            head_ids, kernel=_write_boundary_history, out_shape=history_shape,
            grid=(3,), num_warps=2, num_stages=1, T=length, K=width, BT=chunk)
        return triton_call(
            head_ids, history, kernel=_read_boundary_history, out_shape=output_shape,
            grid=(3,), num_warps=2, num_stages=1, T=length, K=width, BT=chunk, CASES=3)

    actual = np.asarray(roundtrip(heads))
    expected = np.arange(3 * chunk * width, dtype=np.float32).reshape(3, chunk, width)
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(actual[1], expected)
