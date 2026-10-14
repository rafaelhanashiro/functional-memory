"""Per-call precision/occupancy controls preserve compiler and cache isolation."""
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


_TEST_PTX = """.version 8.7
.target sm_80
.address_size 64
.visible .entry test_kernel(
    .param .u64 .ptr .global .align 1 input
)
.reqntid 128
{
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [input];
    ret;
}
"""


def _compile(precision, min_ctas_per_sm=None):
    return bridge.get_or_create_triton_kernel(
        bridge.get_cuda_backend, "cuda", _dot_kernel, ["*fp32"] * 4, (), 0,
        num_warps=4, num_stages=3, num_ctas=1, compute_capability=80,
        enable_fp_fusion=True, default_dot_input_precision=precision,
        min_ctas_per_sm=min_ctas_per_sm,
        metaparams={}, dump=False,
    )[0]


@pytest.fixture
def compile_harness(monkeypatch, tmp_path):
    """Generate real TTIR on CPU; replace only GPU codegen/runtime objects."""
    captured = []

    def finish(ttir, backend, options, compute_capability, platform):
        captured.append((str(ttir), options.default_dot_input_precision))
        return bridge.CompilationResult(
            binary=_TEST_PTX, name="test_kernel", shared_mem_bytes=0,
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


def test_occupancy_hint_isolates_memory_disk_caches_and_reaches_runtime_ptx(compile_harness):
    captured, cache_dir = compile_harness
    kernels = {hint: _compile("tf32x3", hint) for hint in (None, 1, 2)}
    assert len(captured) == 3
    assert len({id(kernel) for kernel in kernels.values()}) == 3
    assert kernels[None].args[4] == _TEST_PTX
    for hint in (1, 2):
        ptx = kernels[hint].args[4]
        assert ptx.count(".minnctapersm") == 1
        assert f".minnctapersm {hint}" in ptx
    for hint in (None, 1, 2):
        assert _compile("tf32x3", hint) is kernels[hint]
    assert len(captured) == 3
    assert len(list(cache_dir.glob("*/kernel.pkl"))) == 3
    bridge._COMPILED_KERNEL_CACHE.clear()
    for hint in (None, 1, 2):
        reloaded = _compile("tf32x3", hint)
        assert reloaded.args[4] == kernels[hint].args[4]
    assert len(captured) == 3  # each annotated binary reloads from its own entry


def test_public_occupancy_hint_is_a_primitive_parameter(monkeypatch):
    captured = []

    def bind(*args, **kwargs):
        captured.append(kwargs)
        return [jnp.zeros((16, 16), dtype=jnp.float32)]

    monkeypatch.setattr(bridge.triton_kernel_call_p, "bind", bind)
    monkeypatch.setattr(bridge, "_has_multi_accelerators", lambda: False)
    for hint in (None, 1, 2):
        bridge.triton_call(
            jnp.zeros((16, 16)), kernel=_dot_kernel,
            out_shape=jax.ShapeDtypeStruct((16, 16), jnp.float32), grid=(1,),
            min_ctas_per_sm=hint,
        )
    assert [params["min_ctas_per_sm"] for params in captured] == [None, 1, 2]


def test_lowering_separates_autotune_occupancy_and_does_not_make_it_a_constexpr(monkeypatch):
    hints, autotune_keys = [], []

    def compile_mock(*args, **kwargs):
        hints.append(kwargs["min_ctas_per_sm"])
        assert "min_ctas_per_sm" not in kwargs["metaparams"]
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
    for hint in (None, 1, 2):
        bridge.triton_kernel_call_lowering(
            bridge.get_cuda_backend, ctx, fn=kernel, scalar_args=(), name="dot",
            custom_call_target_name="triton_kernel_call", out_shapes=(aval, aval),
            grid=(1,), num_warps=4, num_stages=3, num_ctas=1, device=0,
            compute_capability=80, enable_fp_fusion=True,
            min_ctas_per_sm=hint, input_output_aliases=(), zeroed_outputs=(),
            debug=False, serialized_metadata=b"",
        )
    assert hints == [None, None, 1, 1, 2, 2]
    assert len(set(autotune_keys)) == 3


@pytest.mark.parametrize("directive", [".reqntid 128", ".maxntid 128, 1, 1"])
def test_ptx_occupancy_directive_preserves_header_and_kernel_body(directive):
    ptx = _TEST_PTX.replace(".reqntid 128", directive)
    annotated = bridge._set_ptx_min_ctas_per_sm(ptx, "test_kernel", 1)
    assert annotated.count(".minnctapersm 1") == 1
    assert annotated.index(directive) < annotated.index(".minnctapersm 1") < annotated.index("{")
    assert annotated[annotated.index("{"):] == ptx[ptx.index("{"):]
    assert annotated[:annotated.index(directive) + len(directive)] == ptx[:ptx.index(directive) + len(directive)]
    assert bridge._set_ptx_min_ctas_per_sm(annotated, "test_kernel", 1) == annotated


def test_ptx_default_preserves_bytes_and_hint_only_changes_the_named_entry():
    other = _TEST_PTX.replace("test_kernel", "other_kernel")
    source = other + "\n" + _TEST_PTX[_TEST_PTX.index(".visible"):]
    assert bridge._set_ptx_min_ctas_per_sm(source, "test_kernel", None) == source
    annotated = bridge._set_ptx_min_ctas_per_sm(source, "test_kernel", 2)
    assert annotated.startswith(other + "\n")
    assert annotated.count(".minnctapersm") == 1
    assert annotated.index(".entry test_kernel") < annotated.index(".minnctapersm 2")
    # A disabled option does not parse or reject arbitrary legacy PTX.
    assert bridge._set_ptx_min_ctas_per_sm("legacy opaque binary", "missing", None) == "legacy opaque binary"


@pytest.mark.parametrize("source,name", [
    (_TEST_PTX, "missing"),
    (_TEST_PTX + _TEST_PTX, "test_kernel"),
    (_TEST_PTX.replace(".reqntid 128", ".reqntid 128\n.minnctapersm 2"), "test_kernel"),
], ids=["missing", "ambiguous", "conflicting"])
def test_ptx_hint_rejects_missing_ambiguous_or_conflicting_kernel_entries(source, name):
    with pytest.raises(ValueError):
        bridge._set_ptx_min_ctas_per_sm(source, name, 1)


@pytest.mark.parametrize("hint", [True, False, 0, -1, 1.5, "1"])
def test_invalid_occupancy_hint_rejected_before_compilation_or_primitive_binding(hint, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid occupancy hint reached compilation or primitive binding")

    monkeypatch.setattr(bridge.triton_kernel_call_p, "bind", unexpected)
    monkeypatch.setattr(bridge, "get_cuda_backend", unexpected)
    with pytest.raises(ValueError):
        bridge.triton_call(
            jnp.zeros((16, 16)), kernel=_dot_kernel,
            out_shape=jax.ShapeDtypeStruct((16, 16), jnp.float32), grid=(1,),
            min_ctas_per_sm=hint,
        )
    with pytest.raises(ValueError):
        _compile("tf32x3", hint)


def test_occupancy_hint_rejects_rocm_before_backend_initialization():
    def unexpected(*args, **kwargs):
        pytest.fail("CUDA-specific occupancy hint reached ROCm backend initialization")

    with pytest.raises(ValueError, match="CUDA|cuda"):
        bridge.get_or_create_triton_kernel(
            unexpected, "rocm", _dot_kernel, ["*fp32"] * 4, (), 0,
            num_warps=4, num_stages=3, num_ctas=1, compute_capability=80,
            enable_fp_fusion=True, min_ctas_per_sm=1, metaparams={}, dump=False,
        )
