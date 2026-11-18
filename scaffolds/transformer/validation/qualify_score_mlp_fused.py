"""The fused score-MLP GPU qualification record: the tests' outcome and the errors behind it.

The GPU tests (``tests/test_score_mlp_fused.py -m gpu``) assert thresholds. This module
measures the same cases with the tests' inputs and oracles, without asserting, summarizes the
JUnit file of the GPU test run that preceded it, and writes one record to ``--output``.
``qualify_score_mlp_fused.sbatch`` (step ``tests``) runs the GPU tests of both score-MLP test
files and then, on the same GPU:

    python -m scaffolds.transformer.validation.qualify_score_mlp_fused --launch-commit <commit> \
      --junit <gpu-tests.xml> --exit-code <the tests' status> \
      --output scaffolds/transformer/validation/runs/score-mlp/fused-gpu-qualification.json
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import shlex
import sys
import time
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention import score_mlp
from scaffolds.transformer.attention.backends import fused_score_mlp as fused
from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention
from scaffolds.transformer.benchmarks.benchmark_attention import write_report
from scaffolds.transformer.tests import test_score_mlp_fused as tests
from scaffolds.transformer.tests.test_score_mlp_attention import _numpy_dense_vjp

ROOT = Path(__file__).resolve().parents[3]
ODD_SHAPES = ((3, 33, 17), (3, 72, 40), (3, 128, 128), (8, 128, 128))   # (J, Q/K width, value width)


def _float(value):
    return float(f"{value:.3g}") if np.isfinite(value) else None


def _relative(observed, expected):
    """Relative L2 error, or None (nonfinite) where the tests would fail."""
    observed, expected = np.asarray(observed, np.float64), np.asarray(expected, np.float64)
    if not (np.isfinite(observed).all() and np.isfinite(expected).all()) or not np.linalg.norm(expected):
        return None
    return _float(np.linalg.norm(observed - expected) / np.linalg.norm(expected))


def _errors(actual, expected, magnitude=None):
    """The output and every gradient; dw also over the FP64 absolute sum of its terms, as the tests bound it."""
    errors = {name: _relative(observed, wanted) for name, observed, wanted in zip(
        tests.NAMES, (actual[0], *actual[1]), (expected[0], *expected[1]), strict=True)}
    if magnitude is not None:
        scaled = np.abs(np.asarray(actual[1][5], np.float64) - np.asarray(expected[1][5], np.float64)) / magnitude
        errors["w_over_term_magnitude"] = _float(np.max(scaled))
    return errors


def _program(form, cotangent, backend="fused"):
    return jax.jit(lambda *x: tests._vjp(tests._attend(form, backend), x, cotangent))


def production_shape():
    """2,048 tokens, 4 heads, 64-channel BF16 Q/K/V: against the FP64 oracle and the blocked path."""
    oracle, blocked, repeatable = {}, {}, {}
    for form in score_mlp.FORMS:
        for width in (4, 8):
            name = f"{form}-{width}"
            operands, cotangent = tests._production_inputs(form, width)
            expected = _numpy_dense_vjp(operands, cotangent, form=form)
            first, second = _program(form, cotangent)(*operands), _program(form, cotangent)(*operands)
            reference = _program(form, cotangent, "blocked")(*operands)
            repeatable[name] = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
                jax.tree.leaves(first), jax.tree.leaves(second), strict=True))
            oracle[name] = _errors(first, expected, tests._slope_magnitude(form, operands, cotangent))
            blocked[name] = {"fused_vs_blocked": _errors(first, reference),
                             "blocked_vs_oracle": _errors(reference, expected)}
            print(name, oracle[name], blocked[name], "repeatable", repeatable[name], flush=True)
    return {"shape": {"batch": 1, "time": 2048, "heads": 4, "qk_head_dim": 64, "value_head_dim": 64},
            "relative_l2_errors": oracle, "blocked_comparison": blocked, "bitwise_repeatable": repeatable}


def zero_amplitude():
    """exp_residual at a = 0 against the fused unnormalized exp kernel, whose 64x64 tiles sum in another FP32 order."""
    report = {}
    for width in (4, 8):
        (q, k, v, a, b, w), cotangent = tests._production_inputs("exp_residual", width)
        actual = _program("exp_residual", cotangent)(q, k, v, jnp.zeros_like(a), b, w)
        exponential = jax.jit(lambda *x, cotangent=cotangent: tests._vjp(
            fused_unnormalized_attention, x, cotangent))(q, k, v)
        pairs = tuple(zip(tests.NAMES, (actual[0], *actual[1][:3]), (exponential[0], *exponential[1])))
        report[f"exp_residual-{width}"] = {
            "relative_l2": {name: _relative(o, e) for name, o, e in pairs},
            "bitwise_equal": {name: bool(np.array_equal(np.asarray(o), np.asarray(e))) for name, o, e in pairs},
            "db_dw_exactly_zero": bool(np.all(np.asarray(actual[1][4]) == 0) and np.all(np.asarray(actual[1][5]) == 0)),
        }
        print("a = 0", width, report[f"exp_residual-{width}"], flush=True)
    return report


def odd_shapes():
    """131 tokens, 2 batches, 3 heads: partial tiles, padded channels, and widths up to 128, FP32 and BF16."""
    report = {}
    for width, qk, value in ODD_SHAPES:
        for form in score_mlp.FORMS:
            for dtype in (jnp.float32, jnp.bfloat16):
                operands, cotangent = tests._inputs(width, time=131, batch=2, heads=3, qk=qk, value=value, dtype=dtype)
                name = f"{form}-J{width}-{qk}-{value}-{jnp.dtype(dtype).name}"
                report[name] = _errors(_program(form, cotangent)(*operands),
                                       _numpy_dense_vjp(operands, cotangent, form=form),
                                       tests._slope_magnitude(form, operands, cotangent))
                print(name, report[name], flush=True)
    return {"time": 131, "batch": 2, "heads": 3, "relative_l2_errors": report}


def masked_entries():
    """FP32 masked entries whose score, exp, or f(0) overflows, and a NaN unit in one query's row."""
    report = {"score_overflow_vs_oracle": {}, "value_overflow": {}, "poisoned_row": {}}
    for form in score_mlp.FORMS:
        operands, cotangent = tests._masked_score_overflow(form)
        report["score_overflow_vs_oracle"][form] = _errors(
            _program(form, cotangent)(*operands), _numpy_dense_vjp(operands, cotangent, form=form),
            tests._slope_magnitude(form, operands, cotangent))
        operands, cotangent, unreached = tests._poisoned_row()
        with jax.default_matmul_precision("highest"):
            actual, blocked = (_program(form, cotangent, backend)(*operands) for backend in ("fused", "blocked"))
        arrays = tuple((name, np.asarray(x, np.float64), np.asarray(y, np.float64), mask) for name, x, y, mask in zip(
            tests.NAMES, (actual[0], *actual[1]), (blocked[0], *blocked[1]), unreached, strict=True))
        report["poisoned_row"][form] = {
            "nan_entries": {name: int(np.isnan(x).sum()) for name, x, _, _ in arrays},
            "nan_entries_match_blocked": all(np.array_equal(np.isnan(x), np.isnan(y)) for _, x, y, _ in arrays),
            "unreached_vs_blocked": {name: _relative(x[mask], y[mask]) for name, x, y, mask in arrays}}
        print("masked", form, report["score_overflow_vs_oracle"][form], report["poisoned_row"][form], flush=True)
    for length in (1, 2):
        operands, cotangent = tests._masked_value_overflow(length)
        actual = _program("exp_residual", cotangent)(*operands)
        with np.errstate(over="ignore"):
            expected = _numpy_dense_vjp(operands, cotangent, form="exp_residual")
        with jax.default_matmul_precision("highest"):
            blocked = _program("exp_residual", cotangent, "blocked")(*operands)
        row = {"finite": all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves(actual))}
        for label, reference in (("max_relative_vs_oracle", expected), ("max_relative_vs_blocked", blocked)):
            # Elementwise: da, db, and dw are exactly zero here.
            row[label] = {name: _float(np.max(np.abs(np.asarray(o, np.float64) - np.asarray(e, np.float64))
                                              / np.maximum(np.abs(np.asarray(e, np.float64)), 1e-30)))
                          for name, o, e in zip(tests.NAMES, (actual[0], *actual[1]), (reference[0], *reference[1]))}
        report["value_overflow"][f"tokens-{length}"] = row
        print("value overflow", length, row, flush=True)
    return report


def model_shape():
    """Micro-batch 16, 14 heads, 2,048 tokens, J = 8: finite, and against the blocked path on the same GPU."""
    report = {"shape": {"batch": 16, "time": 2048, "heads": 14, "qk_head_dim": 64, "value_head_dim": 64}}
    for form in score_mlp.FORMS:
        operands, cotangent = tests._production_inputs(form, 8, batch=16, heads=14)
        actual = _program(form, cotangent)(*operands)
        report[f"{form}-8"] = {"finite": all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(actual)),
                               "fused_vs_blocked": _errors(actual, _program(form, cotangent, "blocked")(*operands))}
        print("model shape", form, report[f"{form}-8"], flush=True)
    return report


def whole_model():
    """Two layers, 256 tokens, FP32 at HIGHEST precision: loss and every parameter gradient, fused against blocked."""
    changes = dict(max_seq_len=256, hidden_size=32, num_heads=2, num_kv_heads=2, qk_head_dim=16, value_head_dim=16,
                   intermediate_size=48)
    report = {"layers": 2, "tokens": 256, "hidden": 32, "heads": 2, "head_dims": 16, "width": 4}
    tokens = jnp.asarray(np.random.default_rng(3).integers(0, 16, size=(2, 257)), jnp.int32)
    with jax.default_matmul_precision("highest"):
        for form in score_mlp.FORMS:
            fused_loss, fused_gradients = tests._model_loss_and_gradients(form, "fused", tokens, **changes)
            blocked_loss, blocked_gradients = tests._model_loss_and_gradients(form, "blocked", tokens, **changes)
            errors = {"/".join(map(str, path)): _relative(fused_gradients[path], expected)
                      for path, expected in blocked_gradients.items()}
            worst = max(errors, key=lambda path: np.inf if errors[path] is None else errors[path])
            report[form] = {"loss_fused": fused_loss, "loss_blocked": blocked_loss,
                            "loss_relative_difference": _float(abs(fused_loss - blocked_loss) / abs(blocked_loss)),
                            "max_parameter_gradient_relative_l2": errors[worst], "worst_parameter": worst}
            print("model", form, report[form], flush=True)
    return report


def junit_summary(path: Path) -> dict:
    cases = list(ET.parse(path).iter("testcase"))
    return {"cases": len(cases),
            "failed": sum(any(case.find(tag) is not None for tag in ("failure", "error")) for case in cases),
            "skipped": sum(case.find("skipped") is not None for case in cases),
            "seconds": round(sum(float(case.get("time", 0)) for case in cases), 1),
            "junit": {"path": os.path.relpath(path.resolve(), ROOT),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch-commit", required=True)
    parser.add_argument("--junit", type=Path, nargs="+", required=True, help="JUnit files of the preceding test runs")
    parser.add_argument("--exit-code", type=int, required=True, help="the preceding test runs' exit status")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    devices = [device.device_kind for device in jax.local_devices()]
    if jax.default_backend() != "gpu" or len(devices) != 1:
        raise SystemExit(f"Qualify on exactly one GPU, not {devices}")
    suites = {path.stem: junit_summary(path) for path in args.junit}
    report = {
        "producer": "scaffolds/transformer/validation/qualify_score_mlp_fused.py",
        "command": shlex.join(["python", "-m", "scaffolds.transformer.validation.qualify_score_mlp_fused",
                               *sys.argv[1:]]),
        "source_commit": args.launch_commit, "home_cluster": "kennel", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"), "gpu_type": devices[0], "jax": jax.__version__,
        "dependency_lock_sha256": hashlib.sha256((ROOT / "scaffolds/transformer/uv.lock").read_bytes()).hexdigest(),
        "runtime_environment": {key: os.environ.get(key) for key in (
            "TRANSFORMER_CUDA_ENVIRONMENT", "CUDA_HOME", "TRITON_PTXAS_PATH", "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_PREALLOCATE")},
        "production": (f"fused Pallas Triton kernel (attention/backends/fused_score_mlp.py), {fused.BLOCK_Q}x"
                       f"{fused.BLOCK_K} tiles, {fused.NUM_WARPS} warps, {fused.NUM_STAGES} stages"),
        "reference": "host NumPy FP64 forward and analytic VJP on the same (rounded) values, crosschecked "
                     "against CPU float64 autodiff; the blocked path on the same GPU",
        "tolerances": {"bfloat16_output_and_qkv_gradients": .012, "float32": 1e-4, "unit_gradients": 1e-4},
        "tolerance_scope": "relative L2 for the whole array, each head, and (a, b) each head and unit; each dw_hj "
                           "error over the FP64 absolute sum of its terms, which may cancel",
        "tests": suites,
        "production_shape": production_shape(),
        "exp_residual_zero_amplitude_vs_fused_exp": zero_amplitude(),
        "odd_shapes": odd_shapes(),
        "masked_entries": masked_entries(),
        "model_shape": model_shape(),
        "whole_model_fused_vs_blocked": whole_model(),
    }
    passed = args.exit_code == 0 and all(suite["cases"] and not suite["failed"] and not suite["skipped"]
                                         for suite in suites.values())
    report.update(status="passed" if passed else "failed", exit_code=args.exit_code,
                  completed_utc=datetime.now(UTC).isoformat(), measurement_seconds=round(time.time() - started, 1))
    write_report(args.output, report)
    print("status", report["status"], flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
