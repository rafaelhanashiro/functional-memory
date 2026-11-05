"""CPU checks of the blocked softmax candidates in blocked_attention_benchmark.py.

Run from the repository root:
  JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest -q \
    -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
    scaffolds/transformer/validation/test_blocked_attention_benchmark.py
"""
import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.validation import blocked_attention_benchmark as bench


def lean(block, weights, *, checkpoint=True, causal_skip=True):
    return functools.partial(bench.lean_blocked_attention, block_size=block, weights=weights,
                             checkpoint=checkpoint, causal_skip=causal_skip)


def flash(block, weights, *, backward="columns"):
    return functools.partial(bench.flash_blocked_attention, block_size=block, weights=weights, backward=backward)


CANDIDATES = {
    "lean-skip-ckpt": lean,
    "lean-skip-nockpt": functools.partial(lean, checkpoint=False),
    "lean-scan-ckpt": functools.partial(lean, causal_skip=False),
    "lean-scan-nockpt": functools.partial(lean, checkpoint=False, causal_skip=False),
    "flash-columns": flash,
    "flash-rows": functools.partial(flash, backward="rows"),
    "flash-saved": functools.partial(flash, backward="saved"),
}


def inputs(shape, seed=0, sharp=2.0):
    rng = np.random.default_rng(seed)
    q, k, v, g = (rng.standard_normal(shape) for _ in range(4))
    return q * sharp, k, v, g


def run(attend, q, k, v, g):
    def program(q, k, v, g):
        output, pullback = jax.vjp(attend, q, k, v)
        return output, *pullback(g)

    return jax.jit(program)(*(jnp.asarray(x) for x in (q, k, v, g)))


def test_oracle_matches_float64_autodiff():
    check = bench.oracle_check()
    assert check["passed"], check
    assert check["worst_relative_l2"] < 1e-12


@pytest.mark.parametrize("length,block", [(1, 1), (1, 4), (37, 4), (37, 16), (32, 8), (37, 64)])
@pytest.mark.parametrize("weights", bench.WEIGHTS)
@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_float64_outputs_and_gradients_match_oracle(candidate, weights, length, block):
    q, k, v, g = inputs((2, length, 3, 8), seed=length + block)
    v = v[..., :5]
    g = g[..., :5]
    with jax.enable_x64():
        results = run(CANDIDATES[candidate](block, weights), q, k, v, g)
        assert all(x.dtype == jnp.float64 for x in results)
        for actual, expected in zip(results, bench.oracle(q, k, v, g), strict=True):
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_float32_matches_production_blocked(candidate):
    q, k, v, g = (x.astype(np.float32) for x in inputs((2, 45, 2, 16), seed=4))
    expected = run(functools.partial(blocked_dot_product_attention, query_block_size=8), q, k, v, g)
    actual = run(CANDIDATES[candidate](8, "float32"), q, k, v, g)
    reference = bench.oracle(q, k, v, g)
    for observed, production, exact in zip(actual, expected, reference, strict=True):
        assert observed.dtype == jnp.float32
        np.testing.assert_allclose(observed, production, rtol=2e-5, atol=2e-6)
        assert bench.relative_l2(observed, exact) < 1e-5


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_bfloat16_dtypes_and_weight_rounding(candidate):
    q, k, v, g = (jnp.asarray(x, jnp.bfloat16) for x in inputs((1, 64, 2, 16), seed=5))
    exact = bench.oracle(*(np.asarray(x.astype(jnp.float32)) for x in (q, k, v, g)))
    results = {}
    for weights in bench.WEIGHTS:
        results[weights] = run(CANDIDATES[candidate](16, weights), q, k, v, g)
        assert all(x.dtype == jnp.bfloat16 for x in results[weights])
        for observed, reference in zip(results[weights], exact, strict=True):
            assert bench.relative_l2(np.asarray(observed.astype(jnp.float32)), reference) < 2e-2
    # Input-dtype weights round P (and the backward's weights) to BF16: results differ.
    assert any(not np.array_equal(np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32)))
               for a, b in zip(results["float32"], results["input"], strict=True))


def test_gradient_sums_over_blocks_round_once_with_fp32_accumulation():
    """Autodiff over BF16 blocks sums dK/dV in BF16; the custom VJP and upcast operands do not."""
    q, k, v, g = (jnp.asarray(x, jnp.bfloat16) for x in inputs((1, 512, 2, 32), seed=8, sharp=1.0))
    exact = bench.oracle(*(np.asarray(x.astype(jnp.float32)) for x in (q, k, v, g)))
    floor = [bench.relative_l2(np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32)), x) for x in exact]

    def errors(attend):
        return [bench.relative_l2(np.asarray(x.astype(jnp.float32)), reference)
                for x, reference in zip(run(attend, q, k, v, g), exact, strict=True)]

    single = errors(functools.partial(blocked_dot_product_attention, query_block_size=512))
    blocked = errors(functools.partial(blocked_dot_product_attention, query_block_size=64))
    for attend in (flash(64, "float32"), flash(64, "float32", backward="rows"), flash(64, "float32", backward="saved"),
                   functools.partial(bench.lean_blocked_attention, block_size=64, upcast=True)):
        for observed, best in zip(errors(attend), floor, strict=True):
            assert observed < 1.05 * best
    for index in (2, 3):  # dK and dV: eight BF16 block sums in production
        assert single[index] < 1.05 * floor[index] < blocked[index] / 1.3


@pytest.mark.parametrize("backward", bench.BACKWARDS)
def test_custom_backward_starts_behind_a_barrier_with_the_cotangent(backward):
    """Keeps XLA from merging the backward's score products with the forward's across layers."""
    q = jnp.ones((1, 8, 1, 4))
    jaxpr = jax.make_jaxpr(lambda q, k, v, g: jax.vjp(flash(4, "float32", backward=backward), q, k, v)[1](g))(
        q, q, q, q)
    assert "optimization_barrier" in str(jaxpr)


def test_rounded_to_equals_the_bfloat16_round_trip():
    """reduce_precision(8, 7) rounds exactly as a BF16 conversion, subnormals and specials included."""
    rng = np.random.default_rng(10)
    x = (rng.standard_normal(20000) * 10.0 ** rng.integers(-40, 38, 20000)).astype(np.float32)
    x = np.concatenate([x, np.array([np.inf, -np.inf, np.nan, 0.0, -0.0, 1e-45, -1e-40, 3.39e38], np.float32)])
    rounded = np.asarray(jax.jit(lambda x: bench._rounded_to(x, jnp.bfloat16))(x))
    expected = np.asarray(jax.jit(lambda x: x.astype(jnp.bfloat16).astype(jnp.float32))(x))
    assert rounded.dtype == np.float32
    np.testing.assert_array_equal(rounded.view(np.uint32)[~np.isnan(x)], expected.view(np.uint32)[~np.isnan(x)])
    assert np.isnan(rounded[np.isnan(x)]).all()
    for dtype in (jnp.float32, jnp.float64):  # not narrower: unchanged
        assert bench._rounded_to(jnp.asarray(x), dtype) is not None
        np.testing.assert_array_equal(np.asarray(bench._rounded_to(jnp.asarray(x), dtype)), x)


# Compile-only H100 target (no GPU needed): cuDNN 9.22, CUDA 12.9, as on dogo.
H100_TARGET = """
gpu_device_info {
  threads_per_block_limit: 1024 threads_per_warp: 32 shared_memory_per_block: 49152
  shared_memory_per_core: 233472 threads_per_core_limit: 2048 core_count: 132 fpus_per_core: 128
  block_dim_limit_x: 2147483647 block_dim_limit_y: 65535 block_dim_limit_z: 65535
  memory_bandwidth: 3352320000000 l2_cache_size: 52428800 clock_rate_ghz: 1.98
  device_memory_size: 85049802752 shared_memory_per_block_optin: 232448
  cuda_compute_capability { major: 9 minor: 0 }
  registers_per_core_limit: 65536 registers_per_block_limit: 65536
}
platform_name: "CUDA"
dnn_version_info { major: 9 minor: 22 patch: 0 }
runtime_version { major: 12 minor: 9 }
device_description_str: "NVIDIA H100 80GB HBM3"
"""


@functools.cache
def h100_sharding():
    from jax.experimental import topologies

    try:
        device = topologies.get_topology_desc("h100", "cuda", target_config=H100_TARGET, topology="1x1x1").devices[0]
    except Exception as error:  # noqa: BLE001  (no CUDA plugin in this environment)
        pytest.skip(f"No compile-only CUDA topology: {type(error).__name__}: {error}")
    return jax.sharding.SingleDeviceSharding(device)


def h100_hlo(function, *shapes, dtype=jnp.bfloat16) -> str:
    """Optimized HLO of ``function`` compiled for an H100 without a GPU, as the accuracy programs."""
    arguments = [jax.ShapeDtypeStruct(shape, dtype, sharding=h100_sharding()) for shape in shapes]
    return jax.jit(function).lower(*arguments).compile({"xla_gpu_autotune_level": "0"}).as_text()


def test_xla_gpu_removes_bf16_round_trips():
    """The hazard: XLA GPU drops an f32 -> bf16 -> f32 conversion pair and keeps reduce_precision."""
    def program(x, g):
        return (jnp.sum(g * x.astype(jnp.bfloat16).astype(jnp.float32), axis=-1),
                jnp.sum(g * bench._rounded_to(x, jnp.bfloat16), axis=-1))

    text = h100_hlo(program, (256, 64), (256, 64), dtype=jnp.float32)
    assert "bf16[" not in text
    assert len(re.findall(r"reduce-precision\(.*exponent_bits=8, mantissa_bits=7", text)) == 1


@pytest.mark.parametrize("backward", bench.BACKWARDS)
def test_bf16_residual_rounding_survives_xla_gpu(backward):
    """One strip, BF16 weights: the BF16 rounding of O in delta (and of saved weights) reaches the
    optimized H100 program; FP32 weights round nothing."""
    shape = (1, 256, 2, 64)
    rounded = h100_hlo(bench.forward_backward(flash(256, "input", backward=backward)), *[shape] * 4)
    assert re.search(r"reduce-precision\(.*exponent_bits=8, mantissa_bits=7", rounded)
    plain = h100_hlo(bench.forward_backward(flash(256, "float32", backward=backward)), *[shape] * 4)
    assert "reduce-precision(" not in plain


def test_hlo_estimate_counts_the_strip_and_scan_gemms():
    """On real H100 programs the estimate's GEMM FLOPs equal the analytic count: seven GEMMs per strip
    (S, P V; recomputed S, dP, dQ, dK, dV), and seven per production block against all keys."""
    batch, length, heads, width = shape = (2, 512, 2, 64)
    strips = sum((end - start) * end for start, end in bench._strips(length, 256))
    estimate = bench.hlo_kernel_estimate(h100_hlo(bench.forward_backward(flash(256, "float32", backward="rows")),
                                                  *[shape] * 4))
    assert estimate["unresolved"] == []
    assert estimate["classes"]["gemm"]["flops"] == pytest.approx(7 * 2 * width * batch * heads * strips, rel=1e-9)
    production = bench.hlo_kernel_estimate(h100_hlo(bench.forward_backward(
        functools.partial(blocked_dot_product_attention, query_block_size=128)), *[shape] * 4))
    assert production["unresolved"] == []
    assert production["classes"]["gemm"]["flops"] == pytest.approx(7 * 2 * width * batch * heads * length * length,
                                                                     rel=1e-9)
    for result in (estimate, production):
        assert result["classes"]["other"]["bytes"] > 0 and result["classes"]["gemm"]["bytes"] > 0


HLO_SAMPLE = """HloModule sample, entry_computation_layout={(f32[8,4]{1,0})->f32[8,4]{1,0}}

%sliced (p0: f32[8,4]) -> f32[2,4] {
  %p0 = f32[8,4]{1,0} parameter(0)
  ROOT %s = f32[2,4]{1,0} slice(%p0), slice={[0:2], [0:4]}
}

%update (b: f32[8,4], u: f32[2,4], i: s32[]) -> f32[8,4] {
  %b = f32[8,4]{1,0} parameter(0)
  %u = f32[2,4]{1,0} parameter(1)
  %i = s32[] parameter(2)
  %z = s32[] constant(0)
  ROOT %d = f32[8,4]{1,0} dynamic-update-slice(%b, %u, %i, %z)
}

%gemm (a: f32[2,4], c: bf16[4,16]) -> f32[2,16] {
  %a = f32[2,4]{1,0} parameter(0)
  %c = bf16[4,16]{1,0} parameter(1)
  %e = f32[4,16]{1,0} convert(%c)
  ROOT %dot.1 = f32[2,16]{1,0} dot(%a, %e), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%body (t: (s32[], f32[8,4])) -> (s32[], f32[8,4]) {
  %t = (s32[], f32[8,4]{1,0}) parameter(0)
  %n = s32[] get-tuple-element(%t), index=0
  %x = f32[8,4]{1,0} get-tuple-element(%t), index=1
  %f = f32[2,4]{1,0} fusion(%x), kind=kLoop, calls=%sliced
  %w = f32[8,4]{1,0} fusion(%x, %f, %n), kind=kLoop, calls=%update
  ROOT %r = (s32[], f32[8,4]{1,0}) tuple(%n, %w)
}

%condition (t: (s32[], f32[8,4])) -> pred[] {
  %t = (s32[], f32[8,4]{1,0}) parameter(0)
  %n = s32[] get-tuple-element(%t), index=0
  %limit = s32[] constant(3)
  ROOT %lt = pred[] compare(%n, %limit), direction=LT
}

ENTRY %main (x: f32[8,4], k: bf16[4,16]) -> (f32[8,4], f32[2,16], f32[2,16]) {
  %x = f32[8,4]{1,0} parameter(0)
  %k = bf16[4,16]{1,0} parameter(1)
  %zero = s32[] constant(0)
  %init = (s32[], f32[8,4]{1,0}) tuple(%zero, %x)
  %loop = (s32[], f32[8,4]{1,0}) while(%init), condition=%condition, body=%body, backend_config={"known_trip_count":{"n":"3"}}
  %out = f32[8,4]{1,0} get-tuple-element(%loop), index=1
  %head = f32[2,4]{1,0} slice(%x), slice={[0:2], [0:4]}
  %g = f32[2,16]{1,0} fusion(%head, %k), kind=kCustom, calls=%gemm, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  %h = (f32[2,16]{1,0}, s8[1024]{0}) custom-call(%head, %k), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]}}}
  %hr = f32[2,16]{1,0} get-tuple-element(%h), index=0
  ROOT %result = (f32[8,4]{1,0}, f32[2,16]{1,0}, f32[2,16]{1,0}) tuple(%out, %g, %hr)
}
"""


def test_hlo_estimate_bytes_flops_and_trip_counts():
    estimate = bench.hlo_kernel_estimate(HLO_SAMPLE)
    gemm, other = estimate["classes"]["gemm"], estimate["classes"]["other"]
    assert estimate["unresolved"] == []
    # Two GEMM launches (Triton fusion, cuBLAS custom call), each 2 x (2 x 16) x 4 FLOPs and reading
    # [2,4] f32 + [4,16] bf16, writing [2,16] f32 (the workspace is not traffic).
    assert gemm["launches"] == 2
    assert gemm["flops"] == 2 * (2 * 2 * 16 * 4)
    assert gemm["bytes"] == 2 * (2 * 4 * 4 + 4 * 16 * 2 + 2 * 16 * 4)
    # Per trip: the slice fusion reads its [2,4] slice and writes it; the in-place update reads the
    # update and its index and writes the update. Three trips; four condition evaluations (two s32
    # read, one pred written); one standalone slice, which reads its whole operand.
    slice_bytes = 2 * 4 * 4
    per_trip = (slice_bytes + slice_bytes) + (slice_bytes + 4 + slice_bytes)
    assert other["launches"] == 3 * 2 + 4 * 1 + 1
    assert other["bytes"] == 3 * per_trip + 4 * (4 + 4 + 1) + (8 * 4 * 4 + slice_bytes)


@pytest.mark.parametrize("checkpoint", (True, False))
def test_upcast_lean_matches_float64_oracle(checkpoint):
    q, k, v, g = inputs((2, 37, 3, 8), seed=9)
    with jax.enable_x64():
        results = run(functools.partial(bench.lean_blocked_attention, block_size=8, checkpoint=checkpoint,
                                        upcast=True), q, k, v, g)
        for actual, expected in zip(results, bench.oracle(q, k, v, g), strict=True):
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match="upcast"):
        bench.lean_blocked_attention(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), weights="input", upcast=True)


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_outputs_are_causal(candidate):
    q, k, v, _ = (x.astype(np.float32) for x in inputs((1, 37, 2, 8), seed=6))
    attend = jax.jit(CANDIDATES[candidate](8, "float32"))
    output = attend(q, k, v)
    changed = attend(np.concatenate([q[:, :19], -4 * q[:, 19:]], axis=1),
                     np.concatenate([k[:, :19], 3 * k[:, 19:]], axis=1),
                     np.concatenate([v[:, :19], v[:, 19:] + 2], axis=1))
    np.testing.assert_array_equal(np.asarray(changed[:, :19]), np.asarray(output[:, :19]))


def nonfinite_rows(attend, q, k, v, g, head=1):
    output, *grads = run(attend, q, k, v, g)
    output = np.asarray(output)
    bad = ~np.isfinite(output)
    others = bad.copy()
    others[0, :, head] = False
    return bad[0, :, head].any(-1), output, not others.any(), all(bool(np.isfinite(np.asarray(x)).all()) for x in grads)


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_nonfinite_inputs_poison_rows_as_production(candidate):
    """Rows with a nonfinite causal score are NaN, as production; V and overflow propagate."""
    position, block = 17, 8
    base = [x.astype(np.float32) for x in inputs((1, 37, 2, 8), seed=7)]
    g = base[3]
    production = functools.partial(blocked_dot_product_attention, query_block_size=block)
    clean = np.asarray(jax.jit(production)(*base[:3]))
    for name, (q, k, v, expected) in bench.failure_cases(*base[:3], position=position).items():
        rows, output, others_finite, grads_finite = nonfinite_rows(CANDIDATES[candidate](block, "float32"), q, k, v, g)
        reference_rows, reference, _, _ = nonfinite_rows(production, q, k, v, g)
        assert others_finite, name
        assert rows[expected].all(), name
        assert not grads_finite, name
        first = position
        if name in ("v_nan", "v_inf"):
            # A masked zero weight times a nonfinite value is NaN: rows sharing the product with the
            # bad key may also be nonfinite (its strip with causal skipping; all rows in production).
            first = 0 if not candidate.startswith(("lean-skip", "flash")) else position // block * block
            assert reference_rows.all(), name
            # Rows before the bad key do not depend on it: compare with clean inputs.
            reference = clean
        elif name == "score_huge":
            # Huge finite scores in later rows may also overflow exp(s - max) after a fused
            # multiply-add; production behaves the same way, and earlier rows are unaffected.
            assert reference_rows[expected].all(), name
        else:
            np.testing.assert_array_equal(rows, reference_rows, err_msg=name)
            assert np.array_equal(rows, expected), name
        assert not rows[:first].any(), name
        finite = ~rows & np.isfinite(reference[0, :, 1]).all(-1)
        np.testing.assert_allclose(output[0, finite, 1], reference[0, finite, 1], rtol=2e-5, atol=2e-6)


def test_dense_manual_does_not_poison_negative_infinite_scores():
    """The textbook dense path (and cuDNN on GPU) treat a -inf score as a zero weight."""
    base = [x.astype(np.float32) for x in inputs((1, 37, 2, 8), seed=7)]
    q, k, v, _ = bench.failure_cases(*base[:3], position=17)["score_neg_inf"]
    output = np.asarray(jax.jit(bench.dense_manual_attention)(q, k, v))
    assert np.isfinite(output).all()
    assert np.isnan(np.asarray(jax.jit(flash(8, "float32"))(q, k, v))[0, 17:, 1]).all()


@pytest.mark.parametrize("arguments,message", [
    ({"block_size": 0}, "block_size"), ({"block_size": True}, "block_size"),
    ({"weights": "bfloat16"}, "weights"), ({"backward": "diagonal"}, "backward"),
])
def test_invalid_options_are_rejected(arguments, message):
    q = jnp.zeros((1, 4, 1, 2))
    with pytest.raises(ValueError, match=message):
        bench.flash_blocked_attention(q, q, q, **arguments)
    if "backward" not in arguments:
        with pytest.raises(ValueError, match=message):
            bench.lean_blocked_attention(q, q, q, **arguments)


def test_mismatched_shapes_are_rejected():
    q = jnp.zeros((1, 4, 2, 8))
    with pytest.raises(ValueError, match="shape"):
        bench.flash_blocked_attention(q, q[:, :3], q)
    with pytest.raises(ValueError, match="shape"):
        bench.lean_blocked_attention(q, q, q[..., 0])


def test_variant_table_builds_every_function():
    table = bench.variants()
    assert len(table) == len(set(table))
    q = jnp.zeros((1, 4, 1, 8), jnp.bfloat16)
    for name, variant in table.items():
        if variant.family != "cudnn":
            assert variant.function()(q, q, q).shape == q.shape, name


def test_profile_summary_splits_every_kernel_by_class():
    """Every kernel counts (not only the top ones); cuDNN's helper kernels count as fused attention."""
    kernels = [{"name": "gemm_fusion_dot_1", "kernel_sum_ms": 0.5}, {"name": "loop_select_fusion", "kernel_sum_ms": 0.3},
               {"name": "cudnn_generated_fort_native_sdpa_sm90_flash_fprop", "kernel_sum_ms": 0.6},
               {"name": "void cudnn::fusion::compute_dot_do_o_specialized", "kernel_sum_ms": 0.2},
               {"name": "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm>", "kernel_sum_ms": 0.1}]
    mean = {"wall_ms": 2.0, "kernel_count": 5, "kernel_sum_ms": 1.7, "gpu_activity_union_ms": 1.7,
            "no_recorded_gpu_activity_ms": 0.3}
    summary = {"devices": [{"mean_per_update": mean, "updates": [{"top_kernels": kernels}] * 2}]}
    result = bench.profile_summary(summary, HLO_SAMPLE)
    assert result["class_ms"] == pytest.approx({"gemm": 0.6, "fused_attention": 0.8, "other": 0.3})
    assert result["achieved"]["gemm"]["tflop_per_s"] == pytest.approx(512 / 0.6e-3 / 1e12)
    assert result["achieved"]["gemm"]["tb_per_s"] == pytest.approx(576 / 0.6e-3 / 1e12)
    assert result["achieved"]["other"]["tflop_per_s"] is None
    assert [kernel["class"] for kernel in result["top_kernels"]][:2] == ["fused_attention", "gemm"]


def tiny_model_config(backend="blocked", layers=2):
    config = bench.load_model_config(bench.RECIPE, micro_batch=2, backend=backend)
    model = config.model
    model.num_layers, model.hidden_size, model.num_heads, model.num_kv_heads = layers, 32, 2, 2
    model.qk_head_dim, model.value_head_dim, model.intermediate_size = 16, 16, 64
    model.vocab_size = config.data.synthetic_vocab_size = 64
    model.max_seq_len, config.data.sequence_length = 32, 33
    config.train.dtype = "float32"
    return config


def test_routed_model_matches_production_blocked(monkeypatch):
    """Routing sends every layer's attention to the candidate (counted per traced call) and
    reproduces the scaffold's blocked loss and gradients; agreement alone would not show routing,
    as the unrouted program agrees too."""
    import scaffolds.transformer.attention.softmax as softmax_module

    batch = jnp.asarray(np.random.default_rng(0).integers(0, 64, (2, 33), dtype=np.int32))
    rng = jax.random.PRNGKey(1)
    state, program, router = bench.micro_step_program(tiny_model_config(), variant=None)
    assert router is None
    expected_nll, expected_grads = jax.jit(program)(state.params, batch, rng)
    for attend in (flash(8, "float32"), flash(8, "input", backward="rows"), flash(8, "float32", backward="saved"),
                   lean(8, "float32")):
        router = bench._routed_softmax(attend)
        monkeypatch.setattr(softmax_module, "softmax_attention", router)
        state, program, _ = bench.micro_step_program(tiny_model_config(), variant=None)
        assert router.calls == 0
        nll, grads = jax.jit(program)(state.params, batch, rng)
        assert router.calls == 2  # one trace, two layers
        np.testing.assert_allclose(nll, expected_nll, rtol=1e-6)
        for observed, expected in zip(jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True):
            np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("layers", (1, 3))
def test_model_case_routing_installs_the_named_variant(monkeypatch, layers):
    """micro_step_program(variant=...) routes each layer's attention through that variant."""
    import scaffolds.transformer.attention.softmax as softmax_module

    monkeypatch.setattr(softmax_module, "softmax_attention", softmax_module.softmax_attention)  # restored after
    state, program, router = bench.micro_step_program(tiny_model_config(layers=layers), variant="flash-f32-rows-b512")
    assert softmax_module.softmax_attention is router
    assert router.attend.func is bench.flash_blocked_attention
    assert router.attend.keywords == {"block_size": 512, "weights": "float32", "backward": "rows"}
    batch = jnp.zeros((2, 33), jnp.int32)
    jax.jit(program).lower(state.params, batch, jax.random.PRNGKey(0))
    assert router.calls == layers


def test_block_sum_check_runs_and_one_block_is_at_the_floor():
    """The historical-kernel check: one query block rounds dK/dV once; eight blocks sum them in BF16."""
    report = bench.block_sum_check((1, 256, 2, 32), seed=3, block_sizes=(32, 256))
    assert set(report["kernels"]) == {"softmax", "query polynomial m=4, unnormalized",
                                      "normalized learned polynomial m=2"}
    for name, record in report["kernels"].items():
        floor = record["bf16_rounding_floor"]
        for label in ("dk", "dv"):
            assert record["blocks"]["256"][label] < 1.1 * floor[label], (name, label)
            assert record["blocks"]["32"][label] > 1.3 * floor[label], (name, label)
