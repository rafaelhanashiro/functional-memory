"""The bridge for Triton kernels written for PyTorch: device facts, menus, launches, interpreter."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl

from scaffolds.transformer.attention.backends import triton_port
from scaffolds.transformer.attention.backends.triton_port import Device, empty, update, zeros


def test_shared_memory_tiers_and_tiles_reproduce_fla():
    pytest.importorskip("torch")
    from fla.utils._device import Backend

    assert triton_port.SHARED_MEMORY_TIERS == {tier.name.lower(): tier.value for tier in Backend}
    h100, a100, ada = Device(9, 0, 232448), Device(8, 0, 166912), Device(8, 9, 101376)
    for device, expected in ((h100, (True, True, True, True)), (a100, (False, True, True, True)),
                             (ada, (False, False, True, False))):
        assert tuple(device.has_shared_memory(t) for t in ("hopper", "ampere", "ada", "default")) == expected
    # FLA's GDN tile caps: 128 with Hopper-class shared memory, 64 with Ada-class, else 32.
    assert [d.tile({"hopper": 128, "ada": 64}, 32) for d in (h100, a100, ada, Device(7, 5, 65536))] == [128, 64, 64, 32]


def test_architecture_menus():
    hopper, blackwell, ampere = Device(9, 0, 232448), Device(10, 3, 232448), Device(8, 0, 166912)
    assert (hopper.arch, blackwell.arch, ampere.arch, Device(12, 0, 101376).arch) == (
        "hopper", "blackwell", "other", "blackwell")
    choose = lambda d: triton_port.per_architecture(d, (2, 4, 8), hopper=(2, 4), blackwell=(2,))
    assert (choose(hopper), choose(blackwell), choose(ampere)) == ((2, 4), (2,), (2, 4, 8))
    assert triton_port.per_architecture(blackwell, [1], hopper=[2]) == [1]


def test_chunk_tables_match_fla():
    torch = pytest.importorskip("torch")
    from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets

    lengths = [5, 64, 65, 130, 1]
    cu_seqlens = torch.tensor(np.concatenate([[0], np.cumsum(lengths)]), dtype=torch.long)
    for chunk in (16, 64):
        np.testing.assert_array_equal(triton_port.chunk_indices(lengths, chunk),
                                      prepare_chunk_indices(cu_seqlens, chunk).numpy())
        np.testing.assert_array_equal(triton_port.chunk_offsets(lengths, chunk),
                                      prepare_chunk_offsets(cu_seqlens, chunk).numpy())


@triton.jit
def _scaled_update(x, scale, y_in, y, total, copy, BLOCK: tl.constexpr):
    """y += scale x in place; total accumulates x atomically; copy writes x."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets)
    tl.store(y + offsets, tl.load(y + offsets) + scale * values)
    tl.atomic_add(total + tl.arange(0, BLOCK), values)
    tl.store(copy + offsets, values)


def _scaled(x, y):
    return triton_port.launch(_scaled_update, x, 2.0, outputs=(update(y), zeros((16,), jnp.float32),
                                                               empty(x.shape, x.dtype)),
                              grid=(x.shape[0] // 16,), name="test::scaled_update", dot_precision="tf32", BLOCK=16)


def test_launch_declares_aliases_zeroed_outputs_and_precision():
    x, y = jnp.ones(64), jnp.arange(64.0)
    (equation,) = jax.make_jaxpr(lambda x, y: _scaled(x, y))(x, y).eqns
    params = equation.params
    assert equation.primitive.name == "triton_kernel_call"
    assert params["input_output_aliases"] == ((1, 0),)  # y, the second array, is output 0
    assert params["zeroed_outputs"] == (1,)
    assert params["scalar_args"] == ((1, "fp32", 2.0),)
    assert params["default_dot_input_precision"] == "tf32"
    assert params["grid"] == (4,) and params["BLOCK"] == 16 and params["name"] == "test::scaled_update"
    assert [tuple(s.shape) for s in params["out_shapes"]] == [(64,), (16,), (64,)]


def test_interpreter_runs_updates_atomic_zeros_and_writes():
    x, y = jnp.arange(64.0) % 5, jnp.arange(64.0)
    with triton_port.interpreter():
        updated, total, copy = jax.jit(_scaled)(x, y)
    np.testing.assert_array_equal(updated, y + 2 * x)
    np.testing.assert_array_equal(total, np.asarray(x).reshape(4, 16).sum(0))
    np.testing.assert_array_equal(copy, x)


@triton.jit
def _first_half(x, y, BLOCK: tl.constexpr):
    tl.store(y + tl.arange(0, BLOCK), tl.load(x + tl.arange(0, BLOCK)))


def test_interpreter_marks_unwritten_outputs_and_rejects_bf16():
    x = jnp.ones(16)
    with triton_port.interpreter():
        result = triton_port.launch(_first_half, x, outputs=empty((32,), jnp.float32), grid=(1,), name="t", BLOCK=16)
        assert np.isnan(np.asarray(result)[16:]).all() and (np.asarray(result)[:16] == 1).all()
        with pytest.raises(ValueError, match="BF16"):
            triton_port.launch(_first_half, x.astype(jnp.bfloat16), outputs=empty((32,), jnp.bfloat16), grid=(1,),
                               name="t", BLOCK=16)


@triton.jit
def _blocked_copy(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offsets, tl.load(x + offsets, mask=offsets < N), mask=offsets < N)


def test_interpreter_resolves_an_autotuned_menu_and_grid_function():
    tuned = triton_port.autotune(_blocked_copy, [triton.Config({"BLOCK": 16}, num_warps=2),
                                                 triton.Config({"BLOCK": 32}, num_warps=4)], key=["N"])
    x = jnp.arange(40.0)
    with triton_port.interpreter() as device:
        assert triton_port.device() == device
        result = triton_port.launch(tuned, x, outputs=empty(x.shape, x.dtype),
                                    grid=lambda meta: (triton_port.cdiv(40, meta["BLOCK"]),), name="t", N=40)
    np.testing.assert_array_equal(result, x)
    assert triton_port.jit_function(tuned) is _blocked_copy
