"""Run Triton kernels written for PyTorch from JAX.

Projects such as flash-linear-attention (FLA) write Triton kernels for PyTorch: their host
code allocates torch tensors, asks torch about the GPU, autotunes with torch's benchmarker,
and updates some outputs in place. The scaffold installs a CPU-only torch that only
satisfies their imports, so this module replaces that host side:

- ``device()`` reads the compute capability from JAX and the opt-in shared memory from the
  CUDA driver; ``Device.has_shared_memory`` reproduces FLA's ``check_shared_mem`` tiers and
  ``Device.tile`` the tile sizes FLA derives from them.
- ``per_architecture`` picks an autotune menu by GPU family (for example FLA's reduced
  Blackwell menus) and ``autotune`` wraps a kernel with it; jaxlib benchmarks every
  configuration on the call's first execution.
- ``jit_function`` returns an upstream kernel's undecorated body, which a bridge-order
  wrapper (inputs, updated arrays, outputs, constexprs) calls as a device function with
  the upstream heuristic flags fixed.
- ``chunk_indices`` and ``chunk_offsets`` build FLA's variable-length chunk tables.
- ``launch`` declares one launch: its inputs, its outputs (``empty``, ``zeros``, or
  ``update`` of an array the kernel modifies in place), grid, launch options, and
  constexprs. On a GPU it lowers to the vendored ejkernel ``triton_call``; inside
  ``interpreter()`` it runs the same wrapper in Triton's interpreter on the CPU through
  ``jax.pure_callback``, so a port's JAX orchestration and its gradients can be checked
  against a JAX reference without a GPU.

Only Triton kernels are in scope; CUDA/C++ extensions and calls into PyTorch are not.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import lru_cache
import threading
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

# FLA's ``Backend`` shared-memory tiers (fla.utils): check_shared_mem(tier) is
# ``max_shared_memory_per_block >= SHARED_MEMORY_TIERS[tier]``; unknown tiers use "default".
SHARED_MEMORY_TIERS = {"ada": 101376, "ampere": 166912, "hopper": 232448, "default": 102400}


def max_shared_memory_per_block(ordinal: int = 0) -> int:
    """The opt-in shared-memory limit per block that Triton reports, from the CUDA driver.

    Triton's own driver object asks torch whether CUDA is available, which the scaffold's
    CPU-only torch denies.
    """
    import ctypes

    cuda = ctypes.CDLL("libcuda.so.1")
    device, value = ctypes.c_int(), ctypes.c_int()
    for status in (cuda.cuInit(0), cuda.cuDeviceGet(ctypes.byref(device), ordinal),
                   # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
                   cuda.cuDeviceGetAttribute(ctypes.byref(value), 97, device)):
        if status:
            raise RuntimeError(f"CUDA driver query failed with status {status}")
    return value.value


@dataclass(frozen=True)
class Device:
    """The facts upstream host code reads from torch: compute capability and shared memory."""

    major: int
    minor: int
    shared_memory: int  # opt-in maximum shared memory per block, bytes

    @property
    def arch(self) -> str:
        return "hopper" if self.major == 9 else "blackwell" if self.major in (10, 12) else "other"

    @property
    def blackwell(self) -> bool:
        return self.arch == "blackwell"

    def has_shared_memory(self, tier: str = "default") -> bool:
        """FLA's ``check_shared_mem(tier)``."""
        return self.shared_memory >= SHARED_MEMORY_TIERS.get(tier, SHARED_MEMORY_TIERS["default"])

    def tile(self, tiles: Mapping[str, int], default: int) -> int:
        """The first tile whose shared-memory tier the device meets, in the given order.

        For example FLA's ``128 if check_shared_mem('hopper') else 64 if
        check_shared_mem('ada') else 32`` is ``tile({"hopper": 128, "ada": 64}, 32)``.
        """
        return next((size for tier, size in tiles.items() if self.has_shared_memory(tier)), default)


_STATE = threading.local()


def _emulated() -> Device | None:
    return getattr(_STATE, "device", None)


@lru_cache(maxsize=None)
def _local_device() -> Device:
    device = jax.local_devices()[0]
    capability = device.compute_capability
    capability = capability() if callable(capability) else capability
    major, minor = (int(part) for part in str(capability).split(".")[:2])
    return Device(major, minor, max_shared_memory_per_block(0))


def interpreting() -> bool:
    """Whether ``launch`` calls are being traced into Triton's interpreter."""
    return _emulated() is not None


def device() -> Device:
    """The local GPU (the device emulated by ``interpreter()``, when active)."""
    return _emulated() or _local_device()


def per_architecture(device: Device, default, *, hopper=None, blackwell=None):
    """``blackwell`` on Blackwell and ``hopper`` on Hopper when given, else ``default``."""
    choice = {"hopper": hopper, "blackwell": blackwell}.get(device.arch)
    return default if choice is None else choice


def jit_function(kernel):
    """An upstream kernel is Heuristics(Autotuner(JITFunction)); return the JITFunction."""
    import triton

    while not isinstance(kernel, triton.JITFunction):
        kernel = kernel.fn
    return kernel


def autotune(kernel, configs: Sequence, key: Sequence[str]):
    """Wrap ``kernel`` with an autotune menu that jaxlib benchmarks at first execution.

    A dummy benchmarker keeps Triton's Autotuner from asking its (torch-backed) driver for
    one; the bridge passes every configuration to jaxlib's autotuned kernel call.
    """
    import triton

    return triton.autotune(configs=list(configs), key=list(key), do_bench=lambda *args, **kwargs: 0.0)(kernel)


def next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def chunk_indices(lengths: Sequence[int], chunk: int) -> np.ndarray:
    """FLA's ``prepare_chunk_indices``: (sequence, chunk) per chunk of packed sequences, int32 [N, 2]."""
    pairs = [(sequence, index) for sequence, length in enumerate(lengths) for index in range(cdiv(int(length), chunk))]
    return np.asarray(pairs, np.int32).reshape(-1, 2)


def chunk_offsets(lengths: Sequence[int], chunk: int) -> np.ndarray:
    """FLA's ``prepare_chunk_offsets``: each sequence's first chunk, then the total, int32 [N + 1]."""
    return np.concatenate([[0], np.cumsum([cdiv(int(length), chunk) for length in lengths])]).astype(np.int32)


@dataclass(frozen=True)
class Output:
    """One kernel output; ``source`` is the array an ``update`` output starts from."""

    shape: tuple[int, ...]
    dtype: Any
    init: str = "empty"
    source: Any = None


def empty(shape, dtype) -> Output:
    """An output the kernel writes completely."""
    return Output(tuple(shape), jnp.dtype(dtype))


def zeros(shape, dtype) -> Output:
    """An output zeroed before launch (the upstream ``torch.zeros``), e.g. for atomic sums."""
    return Output(tuple(shape), jnp.dtype(dtype), "zeros")


def update(array) -> Output:
    """An output holding ``array`` at launch, which the kernel modifies in place.

    The array is appended to the kernel's inputs, after the explicit inputs, and aliased
    to this output, so the wrapper receives both pointers and must use the output's.
    """
    return Output(tuple(array.shape), jnp.dtype(array.dtype), "update", array)


def launch(kernel, *inputs, outputs: Output | Sequence[Output], grid, name: str,
           num_warps: int | None = None, num_stages: int | None = None,
           dot_precision: str | None = None, **constexprs):
    """Launch ``kernel(*inputs, *updated, *outputs, **constexprs)`` and return its outputs.

    ``inputs`` are arrays and Python scalars in the wrapper's order; ``outputs`` is one
    ``Output`` (returning one array) or a sequence (returning a tuple). ``grid`` is a tuple
    or a function of the launch metaparameters. ``dot_precision`` sets the default
    ``tl.dot`` precision for FP32 operands (``"tf32"`` is Triton's and FLA's default on
    sm80+; None keeps Triton's ambient default). Autotuned kernels take ``num_warps`` and
    ``num_stages`` from their configurations.
    """
    single = isinstance(outputs, Output)
    outputs = (outputs,) if single else tuple(outputs)
    updated = [output.source for output in outputs if output.init == "update"]
    arguments = (*inputs, *updated)
    array_positions = [index for index, value in enumerate(arguments) if not _is_scalar(value)]
    first_update = len(array_positions) - len(updated)
    aliases, update_index = {}, 0
    for index, output in enumerate(outputs):
        if output.init == "update":
            aliases[first_update + update_index] = index
            update_index += 1
    zeroed = tuple(index for index, output in enumerate(outputs) if output.init == "zeros")
    shapes = [jax.ShapeDtypeStruct(output.shape, output.dtype) for output in outputs]
    options = {key: value for key, value in (("num_warps", num_warps), ("num_stages", num_stages))
               if value is not None}
    if _emulated() is not None:
        result = _interpret(kernel, arguments, outputs, shapes, grid, constexprs)
        return result[0] if single else result
    from scaffolds.transformer.attention.backends.ejkernel_bridge import triton_call

    result = triton_call(*arguments, kernel=kernel, out_shape=shapes[0] if single else shapes, grid=grid,
                         name=name, zeroed_outputs=zeroed, default_dot_input_precision=dot_precision,
                         input_output_aliases=aliases or None, **options, **constexprs)
    return result if single else tuple(result)


def _is_scalar(value) -> bool:
    return isinstance(value, bool | int | float | np.float32)


# ---------------------------------------------------------------------------------------
# Triton's interpreter on the CPU
# ---------------------------------------------------------------------------------------

_INTERPRETER_LOCK = threading.RLock()
_INTERPRETED: dict[int, Any] = {}
_MISSING = object()


@contextlib.contextmanager
def interpreter(emulate: Device = Device(9, 0, SHARED_MEMORY_TIERS["hopper"])):
    """Trace ``launch`` calls into Triton-interpreter callbacks, choosing tiles for ``emulate``.

    Use it around tracing (``jax.jit``, ``jax.vjp``) on a CPU, for tests of a port's
    orchestration and gradients. Products run in the interpreter's NumPy arithmetic, not
    in tensor cores, autotuned kernels run their first configuration, and operands must be
    FP32 or FP16 (the interpreter cannot compute in BF16).
    """
    previous = _emulated()
    # Traces record the execution mode and tiles; JAX's caches must not carry them across.
    jax.clear_caches()
    _STATE.device = emulate
    try:
        yield emulate
    finally:
        _STATE.device = previous
        jax.clear_caches()


def _interpreted(function):
    from triton.runtime.interpreter import InterpretedFunction

    key = id(function)  # hashing a JITFunction would hash its source under patched builtins
    if key not in _INTERPRETED:
        _INTERPRETED[key] = (function, InterpretedFunction(function.fn))
    return _INTERPRETED[key][1]


@contextlib.contextmanager
def _interpreting():
    """Route nested ``@triton.jit`` calls (upstream device functions) to the interpreter.

    The interpreter patches ``triton.language`` for each nested call and restores only the
    launch's own patches, so the language modules and classes are restored here; later
    compilations in the process would otherwise build interpreter values.
    """
    import triton
    import triton.language as tl
    from triton.runtime.jit import JITFunction

    with _INTERPRETER_LOCK:
        objects = (tl, tl.core, tl.math, tl.tensor, tl.core.tensor, tl.dtype, tl.core.tensor_descriptor_base)
        saved = [(obj, dict(vars(obj))) for obj in objects]
        original = JITFunction.__call__
        JITFunction.__call__ = lambda self, *args, **kwargs: _interpreted(self)(*args, **kwargs)
        try:
            with triton.knobs.runtime.scope():
                triton.knobs.runtime.interpret = True
                yield
        finally:
            JITFunction.__call__ = original
            for obj, attributes in saved:
                for name in set(vars(obj)) - set(attributes):
                    delattr(obj, name)
                for name, value in attributes.items():
                    if vars(obj).get(name, _MISSING) is not value:
                        setattr(obj, name, value)


def _resolve(kernel, named: dict) -> tuple[Any, dict]:
    """The JIT function and its metaparameters: the first configuration, then heuristics."""
    from triton.runtime.autotuner import Autotuner, Heuristics

    if isinstance(kernel, Autotuner):
        function, meta = _resolve(kernel.fn, named)
        return function, {**meta, **kernel.configs[0].kwargs}
    if isinstance(kernel, Heuristics):
        function, meta = _resolve(kernel.fn, named)
        return function, {**meta, **{key: rule({**named, **meta}) for key, rule in kernel.values.items()}}
    return kernel, {}


def _to_torch(array):
    import ml_dtypes
    import torch

    array = np.asarray(array)
    if array.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(array.view(np.int16).copy()).view(torch.bfloat16)
    return torch.from_numpy(array.copy())


def _from_torch(tensor, dtype):
    import ml_dtypes
    import torch

    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy().astype(dtype, copy=False)


def _torch_dtype(dtype):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
            "int32": torch.int32, "int64": torch.int64, "bool": torch.bool}[jnp.dtype(dtype).name]


def _interpret(kernel, arguments, outputs, shapes, grid, constexprs):
    arrays = [value for value in arguments if not _is_scalar(value)]
    if any(jnp.dtype(x.dtype) == jnp.bfloat16 for x in (*arrays, *shapes)):
        # Triton 3.7's interpreter stores BF16 as uint16 and multiplies the raw bits.
        raise ValueError("Triton's interpreter cannot compute in BF16; use FP32 or FP16 operands")

    def run(*host_arrays):
        import torch

        host = iter(host_arrays)
        values = [value if _is_scalar(value) else _to_torch(next(host)) for value in arguments]
        sources = iter(values[len(values) - sum(o.init == "update" for o in outputs):])
        buffers = []
        for output in outputs:
            dtype = _torch_dtype(output.dtype)
            if output.init == "update":
                buffers.append(next(sources).clone())
            elif output.init == "zeros":
                buffers.append(torch.zeros(output.shape, dtype=dtype))
            elif dtype.is_floating_point:
                # Unwritten elements surface as NaN rather than as stale memory.
                buffers.append(torch.full(output.shape, float("nan"), dtype=dtype))
            else:
                buffers.append(torch.full(output.shape, -1, dtype=dtype))
        with _interpreting():
            function = jit_function(kernel)
            call = dict(zip(function.arg_names, [*values, *buffers]))
            _, meta = _resolve(kernel, {**call, **constexprs})
            meta = {**meta, **constexprs}
            launch_grid = grid(meta) if callable(grid) else grid
            launch_grid = (launch_grid,) if isinstance(launch_grid, int) else tuple(launch_grid)
            call.update({name: value for name, value in meta.items() if name in function.arg_names})
            _interpreted(function)[launch_grid](**call)
        return tuple(_from_torch(buffer, shape.dtype) for buffer, shape in zip(buffers, shapes, strict=True))

    return jax.pure_callback(run, tuple(shapes), *arrays, vmap_method="sequential")
