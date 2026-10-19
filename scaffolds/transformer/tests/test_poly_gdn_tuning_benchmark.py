"""Fixture identity, accuracy accounting, and per-variant benchmark failure isolation."""
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

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


def test_split_reference_preserves_batch_slices_and_all_six_vjps(tmp_path):
    args = SimpleNamespace(output=tmp_path / "split.json", reference_batch_size=2)
    runner = benchmark.Runner(args, {})
    operands = tuple(jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 100 + index + 1
                     for index in range(6))
    cotangent = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10 + .5

    def forward(*values):
        return sum((index + 1) * value ** 2 for index, value in enumerate(values))

    reference = benchmark.value_and_vjp(forward, *operands, cotangent)
    entry = {}
    benchmark.check_batch_split_reference(runner, entry, forward, operands, cotangent, reference)
    checks = entry["batch_split_reference"]
    assert checks["status"] == "complete" and checks["chunk_sizes"] == [2, 2, 1]
    assert set(checks["accuracy"]["channels"]) == {"output", "q", "k", "v", "g", "beta", "erase"}
    assert all(value["relative_max"] <= 1e-5 for value in checks["accuracy"]["channels"].values())
    assert "effective_batch" not in runner.context

    # A corruption confined to the final chunk must fail, even if all other
    # chunks and all gradients are correct.
    damaged = (reference[0].at[-1, -1].add(1), reference[1])
    import pytest
    with pytest.raises(FloatingPointError, match="batch-split reference"):
        benchmark.check_batch_split_reference(runner, {}, forward, operands, cotangent, damaged)
