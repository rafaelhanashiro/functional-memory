"""Qualify frozen-head capture and sparse-polynomial fitting on one CUDA device.

This produces software checks and timing evidence, never scientific run metrics.
Invoke through the matching Slurm launcher after committing all execution inputs.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.state_rank.head_capture import (
    DEFAULT_HEADS, capture_llama_heads, load_llama_for_capture, load_token_corpus,
    sha256_file,
)
from scaffolds.state_rank.spectrum import MODEL_ID, MODEL_REVISION


ROOT = Path(__file__).resolve().parents[3]


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def assert_finite(values, description):
    if not all(bool(torch.isfinite(tensor).all()) for tensor in values):
        raise FloatingPointError(f"Nonfinite {description}")


def tiny_cuda_capture():
    from transformers import LlamaConfig, LlamaModel

    config = LlamaConfig(vocab_size=80, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    torch.manual_seed(16)
    model = LlamaModel(config).cuda().eval().requires_grad_(False)
    ids = torch.tensor([[1, 7, 11, 13, 17, 19]], device="cuda")
    captured = capture_llama_heads(model, ids, [(0, 0), (2, 3)])
    errors = {f"{layer}:{head}": item.native_max_abs_error
              for (layer, head), item in captured.items()}
    del model
    torch.cuda.empty_cache()
    return {"passed": True, "native_max_abs_errors": errors}


def tiny_cuda_polynomial():
    from scaffolds.state_rank.sparse_polynomial import PolynomialFeatures, SparsePolynomialHead

    generator = torch.Generator().manual_seed(71)
    q, k = (torch.randn(length, 4, generator=generator) * .2 for length in (3, 10))
    v = torch.randn(10, 6, generator=generator)
    positions = torch.tensor([3, 6, 9])
    cpu_bank, gpu_bank = PolynomialFeatures(4).double(), PolynomialFeatures(4).cuda()
    scores = q.cuda() @ k.cuda().T / 2
    gram = gpu_bank(q.cuda()) @ gpu_bank(k.cuda()).T
    expected = 1 + scores + scores.square()/2 + scores.pow(3)/6
    torch.testing.assert_close(gram, expected, rtol=2e-6, atol=2e-6)
    results = []
    for mode in ("dense", "fixed", "random", "dynamic"):
        budget = cpu_bank.width if mode == "dense" else 8
        cpu = SparsePolynomialHead(cpu_bank, mode=mode, budget=budget, seed=71).double()
        gpu = SparsePolynomialHead(gpu_bank, mode=mode, budget=budget, seed=71).cuda()
        gpu.load_state_dict(cpu.state_dict())
        reference = cpu(q.double(), k.double(), v.double(), positions, training_safe=True)
        actual = gpu(q.cuda(), k.cuda(), v.cuda(), positions.cuda(), training_safe=True)
        for name in ("kernel", "denominator", "attention", "output", "denominator_penalty"):
            torch.testing.assert_close(actual[name].cpu(), reference[name], rtol=2e-5, atol=2e-6,
                                       check_dtype=False, msg=f"CPU/CUDA mismatch: {mode}/{name}")
        torch.testing.assert_close(actual["support"].cpu(), reference["support"], rtol=0, atol=0)
        reference["output"].square().mean().backward()
        actual["output"].square().mean().backward()
        for (name, cpu_parameter), (_, gpu_parameter) in zip(cpu.named_parameters(), gpu.named_parameters(), strict=True):
            if cpu_parameter.grad is None:
                if gpu_parameter.grad is not None:
                    raise AssertionError(f"Unexpected CUDA gradient: {mode}/{name}")
                continue
            if gpu_parameter.grad is None:
                raise AssertionError(f"Missing CUDA gradient: {mode}/{name}")
            torch.testing.assert_close(gpu_parameter.grad.cpu(), cpu_parameter.grad, rtol=3e-4, atol=2e-7,
                                       check_dtype=False, msg=f"CPU/CUDA gradient mismatch: {mode}/{name}")
        results.append({"mode": mode, "budget": budget, "cpu_float64_vs_cuda_float32": "passed"})
    return {"passed": True, "feature_gram_taylor_identity": True, "forward_backward_checks": results}


def benchmark_mode(bank, capture, calibration, mode, budget, args, directory):
    from scaffolds.state_rank.sparse_polynomial import SparsePolynomialHead

    torch.manual_seed(20261015)
    model = SparsePolynomialHead(bank, mode=mode, budget=budget, seed=20261015).cuda()
    model.set_calibration(*calibration)
    q, k, v, teacher = [getattr(capture, name)[0].cuda() for name in ("q", "k", "v", "teacher")]
    positions = torch.linspace(0, len(q)-1, args.query_batch, device="cuda").long().unique()
    eval_positions = torch.linspace(0, len(q)-1, args.eval_query_batch, device="cuda").long().unique()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)

    def forward_backward():
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_features(bank(q[positions]), bank(k), v, positions,
                                            training_safe=True)
        target = teacher[positions]
        loss = (prediction["output"]-target).square().mean() / target.square().mean().clamp_min(1e-12)
        objective = loss + prediction["denominator_penalty"]
        objective.backward()
        return prediction, objective

    # Warm the kernels, allocator, and optimizer state before measuring.
    for _ in range(2):
        prediction, objective = forward_backward()
        assert_finite([objective, prediction["output"]], "warmup output or loss")
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            raise AssertionError("No parameter received a gradient")
        assert_finite(gradients, "warmup gradients")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    backward_times, optimizer_times = [], []
    for _ in range(args.iterations):
        (prediction, objective), elapsed = timed(forward_backward)
        backward_times.append(elapsed)
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert_finite([objective, prediction["output"], *gradients], "timed loss, output, or gradients")

        def update():
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

        _, elapsed = timed(update)
        optimizer_times.append(elapsed)
        assert_finite(parameters, "updated parameters")
    train_peak = torch.cuda.max_memory_allocated()
    model.eval()
    with torch.no_grad():
        key_features, feature_seconds = timed(lambda: bank(k))
        torch.cuda.reset_peak_memory_stats()
        eval_times = []
        for _ in range(args.iterations):
            _, elapsed = timed(lambda: model.forward_features(bank(q[eval_positions]), key_features,
                                                               v, eval_positions, training_safe=False))
            eval_times.append(elapsed)
        eval_peak = torch.cuda.max_memory_allocated()
        reference = model.forward_features(bank(q[eval_positions]), key_features, v, eval_positions,
                                           training_safe=True)
        reference = {name: tensor.detach().cpu().clone() for name, tensor in reference.items()
                     if isinstance(tensor, torch.Tensor)}
    checkpoint = directory / "checkpoints" / f"{mode}-{budget}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "mode": mode, "budget": budget, "steps": args.iterations + 2}, checkpoint)
    restored = SparsePolynomialHead(bank, mode=mode, budget=budget, seed=20261016).cuda().eval()
    saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
    restored.load_state_dict(saved["model"])
    with torch.no_grad():
        actual = restored.forward_features(bank(q[eval_positions]), key_features, v, eval_positions,
                                           training_safe=True)
    for name, tensor in reference.items():
        torch.testing.assert_close(actual[name].cpu(), tensor, rtol=0, atol=0, equal_nan=True,
                                   msg=f"Checkpoint replay differs: {mode}/{budget}/{name}")
    result = {
        "mode": mode, "budget": budget, "iterations": args.iterations,
        "query_batch": len(positions), "evaluation_query_batch": len(eval_positions),
        "keys": len(k), "feature_count": bank.width,
        "forward_backward_seconds": backward_times,
        "median_forward_backward_seconds": statistics.median(backward_times),
        "optimizer_seconds": optimizer_times,
        "median_optimizer_seconds": statistics.median(optimizer_times),
        "evaluation_chunk_seconds": eval_times,
        "median_evaluation_chunk_seconds": statistics.median(eval_times),
        "key_feature_seconds": feature_seconds,
        "peak_training_allocated_bytes": train_peak,
        "peak_evaluation_allocated_bytes": eval_peak,
        "all_gradients_finite": True, "checkpoint_replay_exact": True,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "timing_scope": "one captured head; forward/backward includes Q/K feature construction; eval caches K features",
    }
    del actual, reference, prediction, objective, saved, restored, optimizer, model, q, k, v, teacher, key_features
    torch.cuda.empty_cache()
    return result


def qualify(args, report):
    from scaffolds.state_rank.sparse_polynomial import PolynomialFeatures

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Qualification requires exactly one visible CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    changed = changes(ROOT, args.source_commit, "attention_scaling", "state_rank")
    if changed:
        raise RuntimeError(f"Frozen inputs changed: {changed}")
    if sha256_file(args.corpus / "manifest.json") != args.manifest_sha256:
        raise ValueError("Corpus manifest differs from the submitted checksum")
    windows = load_token_corpus(args.corpus, expected_sha256=args.tokens_sha256)["train"]
    if len(windows) < 2 or windows.shape[1] != 512:
        raise ValueError("Qualification needs at least two 512-token training windows")
    report["hardware"] = {
        "gpu_name": torch.cuda.get_device_name(), "gpu_count": torch.cuda.device_count(),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    report["frozen_execution_sha256"] = frozen_hash(ROOT, args.source_commit, "attention_scaling", "state_rank")
    report["tiny_cuda_capture"] = tiny_cuda_capture()
    report["tiny_cuda_polynomial"] = tiny_cuda_polynomial()
    print("Tiny CUDA capture and polynomial forward/backward parity passed", flush=True)
    model, load_seconds = timed(lambda: load_llama_for_capture())
    captures, capture_times = [], []
    for index in range(2):
        capture, elapsed = timed(lambda: capture_llama_heads(model, windows[index:index+1], DEFAULT_HEADS))
        captures.append(capture)
        capture_times.append(elapsed)
        print(f"Captured train window {index}: {elapsed:.3f}s", flush=True)
    report["capture"] = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "dtype": "float32",
        "load_seconds": load_seconds, "window_seconds": capture_times,
        "median_window_seconds": statistics.median(capture_times), "tokens_per_window": 512,
        "heads": [list(head) for head in DEFAULT_HEADS],
        "native_max_abs_errors": [{f"{layer}:{head}": item.native_max_abs_error
                                    for (layer, head), item in capture.items()} for capture in captures],
        "training_window_indices": [0, 1],
    }
    del model
    torch.cuda.empty_cache()
    bank = PolynomialFeatures(64, degree=3).cuda()
    with torch.no_grad():
        q_features = torch.cat([bank(capture[DEFAULT_HEADS[0]].q[0].cuda()) for capture in captures])
        k_features = torch.cat([bank(capture[DEFAULT_HEADS[0]].k[0].cuda()) for capture in captures])
        calibration = (q_features.double().square().mean(0).sqrt().float(),
                       k_features.double().square().mean(0).sqrt().float())
        del q_features, k_features
    report["timed_head"] = list(DEFAULT_HEADS[0])
    report["benchmarks"] = []
    write_json(args.output / "qualification.json", report)
    variants = [("dense", bank.width)] + [(mode, budget) for mode in ("fixed", "random", "dynamic")
                                         for budget in (128, 512, 2048)]
    for mode, budget in variants:
        result = benchmark_mode(bank, captures[0][DEFAULT_HEADS[0]], calibration, mode, budget, args, args.output)
        report["benchmarks"].append(result)
        write_json(args.output / "qualification.json", report)
        print(f"{mode} {budget}: forward/backward={result['median_forward_backward_seconds']:.6f}s "
              f"eval16={result['median_evaluation_chunk_seconds']:.6f}s", flush=True)
    report["runtime_projection_notes"] = (
        "Multiply each mode's update latency by six heads and its frozen optimizer-step budget. "
        "Add cached-key feature construction per evaluation window and eval-chunk latency per query batch. "
        "These measurements use one head and 512-token inputs; longer-context evaluation, validation "
        "frequency, I/O, checkpointing, scheduling and head-dependent work require separate allowance."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tokens-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--eval-query-batch", type=int, default=16)
    args = parser.parse_args()
    if min(args.iterations, args.query_batch, args.eval_query_batch) < 1:
        parser.error("Iteration and batch counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "kind": "software_hardware_qualification", "source_commit": args.source_commit,
        "started_utc": datetime.now(UTC).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "corpus_manifest_sha256": args.manifest_sha256, "tokens_sha256": args.tokens_sha256,
        "script_sha256": sha256_file(__file__),
    }
    try:
        tests = ["scaffolds/state_rank/tests/test_head_capture.py",
                 "scaffolds/state_rank/tests/test_sparse_polynomial.py"]
        test_command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests,
                        f"--junitxml={args.output / 'tests.xml'}"]
        subprocess.run(test_command, cwd=ROOT, check=True)
        report["cpu_tests"] = {"passed": True, "command": test_command,
                               "junit_sha256": sha256_file(args.output / "tests.xml")}
        qualify(args, report)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(args.output / "qualification.json", report)


if __name__ == "__main__":
    main()
