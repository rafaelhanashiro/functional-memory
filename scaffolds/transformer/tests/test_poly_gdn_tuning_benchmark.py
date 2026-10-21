"""Fixture identity, accuracy accounting, and per-variant benchmark failure isolation."""
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.benchmarks import benchmark_poly_gdn_tuning as benchmark
from scaffolds.transformer.benchmarks.upstream_inputs import kernel_inputs, polynomial_coefficients


def test_equal_width_fixture_preserves_historical_bytes():
    # Generated from the pre-extension function with the locked NumPy version.
    expected = {
        "q": "be7d684ea4f9eaf82615d2a6a1bb5987add85c5e21816780acbbf25607048130",
        "k": "1a6436e487dc688347c2defad19324e081a1d180b7a12f6934c7a79cb232cccc",
        "v": "18b041c0aac541be0474fccb057da5fe38a6a67527cf19416efcd76d595716cb",
        "do": "6f64387f69ea74877ef2c1f86e5db4d102a2c37d3f5ca97b82ffcd93a16be5ca",
        "g": "3057420d828dc80c859b7ac1d1319224aaed6e22b608f40ab7aadbb070e3e041",
        "beta": "2aedc300e687e07adc1b30ce485fcda33ffb0ca50827c4d094d61cd2a748c305",
        "erase": "7fdd6d95e8e18b70628c44e810cbcb553892bc59313fe594ddaee6e0b4994a81",
    }
    for kwargs in ({}, {"value_width": 4}):
        actual = kernel_inputs(2, 8, 2, 4, 46, **kwargs)
        assert {key: benchmark.array_digest(value) for key, value in actual.items()} == expected


def test_unequal_width_fixture_and_separate_coefficients_are_reproducible():
    before = kernel_inputs(1, 64, 2, 16, value_width=64)
    coefficient = polynomial_coefficients(1, 64, 2)
    after = kernel_inputs(1, 64, 2, 16, value_width=64)
    assert before["q"].shape == (1, 64, 2, 16)
    assert before["v"].shape == before["do"].shape == (1, 64, 2, 64)
    assert coefficient.shape == (1, 64, 2, 4)
    assert coefficient.dtype == np.float32
    assert np.all(np.abs(coefficient) < 1)
    assert np.any(coefficient < 0) and np.any(coefficient > 0)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_accuracy_metrics_detect_nonfinite_and_keep_valid_json():
    actual = jnp.array([2., 0.])
    expected = jnp.array([1., 1.])
    result = benchmark.compare_result(actual, expected, backward=False, rms_limit=.1, max_limit=.1)
    assert not result["passed"]
    assert result["channels"]["output"]["relative_rms"] == 1
    result = benchmark.compare_result(jnp.array([jnp.nan]), jnp.array([0.]),
                                      backward=False, rms_limit=.1, max_limit=.1)
    assert not result["all_finite"] and not result["passed"]
    json.dumps(benchmark.json_safe(result), allow_nan=False)


def test_one_compile_failure_preserves_other_variants_and_all_gradient_checks(tmp_path):
    args = SimpleNamespace(output=tmp_path / "report.json", length=2, heads=1, degree=3, seed=46,
        value_dim=3, precisions=["tf32x3", "bf16_read"], warps=[2, 4, 8], reference_warps=2,
        min_ctas=[0, 1], reference_min_ctas=0, reference_batch_size=0, warmup=1, iterations=2,
        rounds=2, exact_relative_rms=.003, exact_relative_max=.01,
        max_relative_rms=.01, max_relative_max=.05, profile_stages=False)
    calls = []

    def forward(q, k, v, g, beta, erase, *, scale, precision, num_warps, min_ctas_per_sm):
        calls.append((precision, num_warps, min_ctas_per_sm))
        if num_warps in (4, 8):
            raise ValueError("synthetic launch failure")
        multiplier = jnp.sum(q.astype(jnp.float32) * k, axis=-1) + g + jnp.sum(beta, axis=-1) + erase
        return (v * multiplier[..., None] * scale).astype(v.dtype)

    entry = {"variants": {}}
    report = {"shapes": {"test": entry}}
    runner = benchmark.Runner(args, report)
    benchmark.benchmark_shape(runner, entry, 1, 2, SimpleNamespace(polynomial_gated_delta_rule=forward))
    assert calls[0] == ("tf32x3", 2, 0)
    assert len(entry["variants"]) == 12
    for precision in args.precisions:
        for min_ctas in args.min_ctas:
            assert (precision, 2, min_ctas) in calls
            assert entry["variants"][f"{precision}_w4_cta{min_ctas}"]["status"] == "failed"
            assert entry["variants"][f"{precision}_w8_cta{min_ctas}"]["status"] == "failed"
            passed = entry["variants"][f"{precision}_w2_cta{min_ctas}"]
            assert passed["min_ctas_per_sm"] == min_ctas
            assert passed["status"] == "complete"
            measured = passed["operations"]["forward_backward"]
            assert measured["summary"]["samples"] == 4
            assert set(measured["accuracy"]["channels"]) == {"output", "q", "k", "v", "g", "beta", "erase"}
    saved = json.loads(args.output.read_text())
    assert saved["shapes"]["test"]["variants"]["bf16_read_w2_cta1"]["status"] == "complete"


def test_resource_inspection_uses_jax_ptxas_flag_tokenization(monkeypatch):
    monkeypatch.setenv("JAX_TRITON_PTXAS_EXTRA_FLAGS", "  --regAllocOptLevel=2  -O3\t--keep ")
    monkeypatch.setenv("JAX_TRITON_PTXAS_DEVICE_DEBUG", "true")
    assert benchmark.runtime_ptxas_flags() == ["--regAllocOptLevel=2", "-O3\t--keep", "--device-debug"]
    monkeypatch.setenv("JAX_TRITON_PTXAS_EXTRA_FLAGS", "-g")
    assert benchmark.runtime_ptxas_flags() == ["-g"]
    monkeypatch.delenv("JAX_TRITON_PTXAS_EXTRA_FLAGS")
    monkeypatch.setenv("JAX_TRITON_PTXAS_DEVICE_DEBUG", "0")
    assert benchmark.runtime_ptxas_flags() == []


def test_cli_deduplicates_occupancy_sweep_and_keeps_reference_zero(monkeypatch, tmp_path):
    import sys
    captured = {}

    def capture(args):
        captured.update(vars(args))
        return {"status": "complete"}

    monkeypatch.setattr(benchmark, "benchmark", capture)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(tmp_path / "report.json"),
        "--min-ctas", "1", "0", "1", "--warps", "2", "2", "--precisions", "tf32x3", "tf32x3"])
    benchmark.main()
    assert captured["min_ctas"] == [1, 0]
    assert captured["warps"] == [2]
    assert captured["precisions"] == ["tf32x3"]
    assert captured["reference_min_ctas"] == 0


def test_split_reference_preserves_batch_slices_and_all_six_vjps(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "split.json", reference_batch_size=2,
                           exact_relative_rms=.003, exact_relative_max=.01)
    runner = benchmark.Runner(args, {})
    operands = tuple((jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 100 + index + 1).astype(
                     jnp.float32 if index == 3 else jnp.bfloat16) for index in range(6))
    cotangent = (jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10 + .5).astype(jnp.bfloat16)

    def forward(*values):
        return sum((index + 1) * value.astype(jnp.float32) ** 2
                   for index, value in enumerate(values)).astype(values[0].dtype)

    observed = {}
    original_compile = runner.compile

    def capture(name, function, arguments, destination):
        observed[name] = tuple(np.asarray(value) for value in arguments)
        return original_compile(name, function, arguments, destination)

    monkeypatch.setattr(runner, "compile", capture)
    reference = benchmark.value_and_vjp(forward, *operands, cotangent)
    entry = {}
    benchmark.check_batch_split_reference(runner, entry, forward, operands, cotangent, reference)
    checks = entry["batch_split_reference"]
    assert checks["status"] == "complete" and checks["passed"]
    assert checks["policy"] == "production_dtype_and_promoted_fp32"
    assert checks["accuracy"]["limits"] == {"relative_rms": .003, "relative_max": .01}
    promoted = checks["promoted_fp32"]
    assert promoted["status"] == "complete" and promoted["promotion_preserves_values_exactly"]
    assert promoted["accuracy"]["limits"] == {"relative_rms": 1e-5, "relative_max": 1e-5}
    assert set(promoted["promotion_channels"]) == {*benchmark.INPUT_NAMES, "cotangent"}
    for scope in (checks, promoted):
        assert scope["chunk_sizes"] == [2, 2, 1]
        assert set(scope["accuracy"]["channels"]) == {"output", *benchmark.INPUT_NAMES}
        assert all(value["relative_max"] <= 1e-5 for value in scope["accuracy"]["channels"].values())
    for index, name in enumerate((*benchmark.INPUT_NAMES, "cotangent")):
        original = (*operands, cotangent)[index]
        channel = promoted["promotion_channels"][name]
        assert channel == {"source_dtype": str(original.dtype), "target_dtype": "float32",
                           "values_preserved_exactly": True}
        promoted_input = observed["batch_split_promoted_fp32_full"][index]
        np.testing.assert_array_equal(promoted_input, np.asarray(original, dtype=np.float32))
        np.testing.assert_array_equal(observed["batch_split_promoted_fp32_b2"][index], promoted_input[:2])
        np.testing.assert_array_equal(observed["batch_split_promoted_fp32_b1"][index], promoted_input[4:])
    assert "effective_batch" not in runner.context

    # A corruption confined to the final chunk must fail even under the
    # production-dtype limits, before the additional FP32 check is attempted.
    damaged = (reference[0].at[-1, -1].add(100), reference[1])
    failed = {}
    with pytest.raises(FloatingPointError, match="batch-split reference"):
        benchmark.check_batch_split_reference(runner, failed, forward, operands, cotangent, damaged)
    assert failed["batch_split_reference"]["status"] == "failed"
    assert not failed["batch_split_reference"]["passed"]
    assert "promoted_fp32" not in failed["batch_split_reference"]


@pytest.mark.parametrize("channel", ("output", *benchmark.INPUT_NAMES))
def test_promoted_fp32_failure_in_any_channel_blocks_candidate_timing(tmp_path, monkeypatch, channel):
    args = SimpleNamespace(output=tmp_path / "rejected.json", length=2, heads=1, degree=3, seed=46,
        value_dim=3, reference_warps=2, reference_min_ctas=0, reference_batch_size=2,
        precisions=["tf32x3"], warps=[2], min_ctas=[0], exact_relative_rms=.003, exact_relative_max=.01)

    def forward(q, k, v, g, beta, erase, **options):
        multiplier = jnp.sum(q.astype(jnp.float32) * k, axis=-1) + g + jnp.sum(beta, axis=-1) + erase
        return (v * multiplier[..., None]).astype(v.dtype)

    entry = {"variants": {}}
    runner = benchmark.Runner(args, {"shapes": {"test": entry}})
    original_compile = runner.compile
    calls = []

    def corrupt_full_fp32(name, function, arguments, destination):
        calls.append(name)
        executable, result = original_compile(name, function, arguments, destination)
        if name == "batch_split_promoted_fp32_full":
            values = [result[0], *result[1]]
            index = ("output", *benchmark.INPUT_NAMES).index(channel)
            value = values[index]
            # A small error at the batch boundary passes the BF16 limits but
            # must fail the mandatory strict FP32 gate, for every VJP channel.
            values[index] = value.at[-1].add(jnp.max(jnp.abs(value)) * .001)
            result = values[0], tuple(values[1:])
        return executable, result

    def forbidden_timing(*args, **kwargs):
        pytest.fail("A rejected split-reference check reached candidate timing")

    monkeypatch.setattr(runner, "compile", corrupt_full_fp32)
    monkeypatch.setattr(runner, "measure", forbidden_timing)
    with pytest.raises(FloatingPointError, match="batch_split_promoted_fp32"):
        benchmark.benchmark_shape(runner, entry, 3, 2, SimpleNamespace(polynomial_gated_delta_rule=forward))
    check = entry["batch_split_reference"]
    assert check["accuracy"]["passed"] and check["accuracy"]["all_finite"]
    assert check["status"] == "failed" and not check["passed"]
    promoted = check["promoted_fp32"]
    assert promoted["promotion_preserves_values_exactly"]
    assert promoted["status"] == "failed" and not promoted["accuracy"]["passed"]
    assert promoted["accuracy"]["all_finite"]
    assert not promoted["accuracy"]["channels"][channel]["passed"]
    assert set(promoted["accuracy"]["channels"]) == {"output", *benchmark.INPUT_NAMES}
    assert entry["variants"] == {} and calls.count("forward_backward") == 1
    assert "effective_batch" not in runner.context
    saved = json.loads(args.output.read_text())["shapes"]["test"]["batch_split_reference"]
    assert saved["status"] == "failed" and saved["promoted_fp32"]["status"] == "failed"


def test_disabled_batch_split_check_does_no_extra_work(tmp_path, monkeypatch):
    runner = benchmark.Runner(SimpleNamespace(output=tmp_path / "unused.json", reference_batch_size=0), {})
    monkeypatch.setattr(runner, "compile", lambda *args: pytest.fail("Disabled check compiled a call"))
    entry = {}
    benchmark.check_batch_split_reference(runner, entry, None, None, None, None)
    assert entry == {} and not runner.args.output.exists()
