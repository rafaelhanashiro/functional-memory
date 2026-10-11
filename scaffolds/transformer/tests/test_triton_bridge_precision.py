"""Per-call dot precision reaches nested JIT code and isolates bridge caches."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
from threading import Event
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl

from scaffolds.transformer.attention.backends.ejkernel_bridge import _triton_call as bridge


@triton.jit
def _nested_dot(x, y):
    return tl.dot(x, y)


@triton.jit
def _dot_kernel(x, y, default_out, explicit_out):
    offsets = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    a = tl.load(x + offsets)
    b = tl.load(y + offsets)
    tl.store(default_out + offsets, _nested_dot(a, b))
    tl.store(explicit_out + offsets, tl.dot(a, b, input_precision="ieee"))


def _compile(precision):
    return bridge.get_or_create_triton_kernel(
        bridge.get_cuda_backend, "cuda", _dot_kernel, ["*fp32"] * 4, (), 0,
        num_warps=4, num_stages=3, num_ctas=1, compute_capability=80,
        enable_fp_fusion=True, default_dot_input_precision=precision,
        metaparams={}, dump=False,
    )[0]


@pytest.fixture
def compile_harness(monkeypatch, tmp_path):
    """Generate real TTIR on CPU; replace only GPU codegen/runtime objects."""
    captured = []

    def finish(ttir, backend, options, compute_capability, platform):
        captured.append((str(ttir), options.default_dot_input_precision))
        return bridge.CompilationResult(
            binary="test PTX", name="test_kernel", shared_mem_bytes=0,
            cluster_dims=(1, 1, 1), ttgir=None, llir=None,
        )

    monkeypatch.setattr(bridge, "compile_ttir_inplace", finish)
    monkeypatch.setattr(bridge.triton_kernel_call_lib, "TritonKernel", lambda *args: SimpleNamespace(args=args), raising=False)
    monkeypatch.setattr(bridge, "_COMPILED_KERNEL_CACHE", OrderedDict())
    monkeypatch.setattr(bridge, "TRITON_CACHE_ENABLED", True)
    monkeypatch.setattr(bridge, "TRITON_CACHE_DIR", tmp_path)
    with triton.knobs.language.scope():
        triton.knobs.language.fp32_default = None
        yield captured, tmp_path


def test_nested_dot_precision_and_explicit_override_have_isolated_caches(compile_harness):
    captured, cache_dir = compile_harness
    legacy = _compile(None)
    accurate = _compile("tf32x3")
    ieee = _compile("ieee")
    assert len(captured) == 3
    assert 'inputPrecision = tf32' in captured[0][0]
    assert 'inputPrecision = tf32x3' in captured[1][0]
    # The explicit dot remains IEEE even when the nested default is tf32x3.
    dots = [line for line in captured[1][0].splitlines() if '= tt.dot ' in line]
    assert len(dots) == 2
    assert sum('inputPrecision = tf32x3' in line for line in dots) == 1
    assert sum('inputPrecision' not in line for line in dots) == 1  # IEEE is the IR default
    assert 'inputPrecision = tf32' not in captured[2][0]
    assert [options for _, options in captured] == ["tf32", "tf32x3", "ieee"]
    assert _compile(None) is legacy
    assert _compile("tf32x3") is accurate
    assert _compile("ieee") is ieee
    assert len(captured) == 3
    assert len(list(cache_dir.glob("*/kernel.pkl"))) == 3
    bridge._COMPILED_KERNEL_CACHE.clear()
    for precision in (None, "tf32x3", "ieee"):
        _compile(precision)
    assert len(captured) == 3  # all three modes reload from their own disk entry
    assert triton.knobs.language.fp32_default is None
    assert "TRITON_F32_DEFAULT" not in os.environ


def test_ambient_precision_is_in_cache_identity(compile_harness):
    captured, _ = compile_harness
    _compile(None)
    triton.knobs.language.fp32_default = "ieee"
    _compile(None)
    assert len(captured) == 2
    assert 'inputPrecision = tf32' in captured[0][0]
    assert 'inputPrecision = tf32' not in captured[1][0]
    assert triton.knobs.language.fp32_default == "ieee"
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_precision_scope_restores_ambient_setting_when_ast_generation_fails(compile_harness, monkeypatch):
    triton.knobs.language.fp32_default = "ieee"

    def fail(*args, **kwargs):
        assert triton.knobs.language.fp32_default == "tf32x3"
        assert os.environ["TRITON_F32_DEFAULT"] == "tf32x3"
        raise RuntimeError("AST generation failed")

    monkeypatch.setattr(bridge.code_gen, "ast_to_ttir", fail)
    with pytest.raises(RuntimeError, match="AST generation failed"):
        _compile("tf32x3")
    assert triton.knobs.language.fp32_default == "ieee"
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_other_compilation_does_not_inherit_an_active_precision_scope():
    entered, release, resolving = Event(), Event(), Event()

    def scoped_compile():
        with bridge._triton_dot_precision_scope("tf32x3"):
            entered.set()
            assert release.wait(timeout=5)

    def resolve_ambient():
        resolving.set()
        return bridge._resolve_dot_input_precision(None)

    with triton.knobs.language.scope():
        triton.knobs.language.fp32_default = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            compilation = pool.submit(scoped_compile)
            try:
                assert entered.wait(timeout=5)
                ambient = pool.submit(resolve_ambient)
                assert resolving.wait(timeout=5)
                assert not ambient.done()
            finally:
                release.set()
            compilation.result(timeout=5)
            assert ambient.result(timeout=5) is None
        assert triton.knobs.language.fp32_default is None
        assert "TRITON_F32_DEFAULT" not in os.environ


def test_public_precision_option_is_a_primitive_parameter(monkeypatch):
    captured = []

    def bind(*args, **kwargs):
        captured.append(kwargs)
        return [jnp.zeros((16, 16), dtype=jnp.float32)]

    monkeypatch.setattr(bridge.triton_kernel_call_p, "bind", bind)
    monkeypatch.setattr(bridge, "_has_multi_accelerators", lambda: False)
    for precision in (None, "tf32x3"):
        bridge.triton_call(
            jnp.zeros((16, 16)), kernel=_dot_kernel,
            out_shape=jax.ShapeDtypeStruct((16, 16), jnp.float32), grid=(1,),
            default_dot_input_precision=precision,
        )
    assert [params["default_dot_input_precision"] for params in captured] == [None, "tf32x3"]


def test_lowering_separates_autotune_precision_and_forwards_compiler_option(monkeypatch):
    modes, autotune_keys = [], []

    def compile_mock(*args, **kwargs):
        modes.append(kwargs["default_dot_input_precision"])
        return object(), {(i,): [] for i in range(4)}

    def autotuned(key, calls, aliases):
        autotune_keys.append(key)
        return SimpleNamespace(to_proto=lambda *args: b"test proto")

    monkeypatch.setattr(bridge, "get_or_create_triton_kernel", compile_mock)
    monkeypatch.setattr(bridge.triton_kernel_call_lib, "create_array_parameter", lambda *args: args, raising=False)
    monkeypatch.setattr(bridge.triton_kernel_call_lib, "TritonKernelCall", lambda *args: args, raising=False)
    monkeypatch.setattr(bridge.triton_kernel_call_lib, "TritonAutotunedKernelCall", autotuned, raising=False)
    monkeypatch.setattr(bridge.jax.ffi, "ffi_lowering", lambda *args, **kwargs: lambda *args: [])
    kernel = triton.autotune(configs=[triton.Config({}, num_warps=n) for n in (4, 8)], key=[])(_dot_kernel)
    aval = bridge.core.ShapedArray((16, 16), jnp.float32)
    ctx = SimpleNamespace(avals_in=[aval, aval], avals_out=[aval, aval], module_context=SimpleNamespace(platforms=["cuda"]))
    for precision in (None, "tf32x3", "ieee"):
        bridge.triton_kernel_call_lowering(
            bridge.get_cuda_backend, ctx, fn=kernel, scalar_args=(), name="dot",
            custom_call_target_name="triton_kernel_call", out_shapes=(aval, aval),
            grid=(1,), num_warps=4, num_stages=3, num_ctas=1, device=0,
            compute_capability=80, enable_fp_fusion=True,
            default_dot_input_precision=precision, input_output_aliases=(),
            zeroed_outputs=(), debug=False, serialized_metadata=b"",
        )
    assert modes == [None, None, "tf32x3", "tf32x3", "ieee", "ieee"]
    assert len(set(autotune_keys)) == 3
