"""Direct upstream FLA and forced cuDNN benchmarks in an isolated CUDA PyTorch environment.

No local attention/model implementation is imported. Kernel timings cover native
BF16 operators; model timings use the upstream GatedDeltaNetForCausalLM and fused
cross entropy with resident, repeated data. They measure compute, not data loading.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def array_digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(samples, *, tokens):
    result = {}
    for name in ("wall_seconds", "cuda_seconds"):
        values = np.array([sample[name] for sample in samples], dtype=np.float64)
        if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Timing samples must be finite and positive")
        result[name] = {"mean": float(values.mean()), "median": float(np.median(values)),
                        "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}
    result["tokens_per_second"] = tokens * len(samples) / sum(sample["wall_seconds"] for sample in samples)
    result["updates"] = len(samples)
    return result


def execution_provenance(torch):
    import fla
    root = Path(__file__).resolve().parents[3]
    fla_root = Path(fla.__file__).parent
    distribution = metadata.distribution("flash-linear-attention")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    from fla.modules import FusedCrossEntropyLoss, FusedRMSNormGated, GatedMLP, RMSNorm, ShortConvolution
    files = [Path(__file__), fla_root / "layers/gated_deltanet.py"]
    files.extend(Path(inspect.getfile(module)) for module in
                 (FusedCrossEntropyLoss, FusedRMSNormGated, GatedMLP, RMSNorm, ShortConvolution))
    files.extend((fla_root / "modules/activations.py", fla_root / "modules/l2norm.py"))
    for directory in ("ops/gated_delta_rule", "ops/common", "models/gated_deltanet"):
        files.extend(sorted((fla_root / directory).glob("*.py")))
    versions = {}
    for name in ("torch", "triton", "flash-linear-attention", "transformers", "numpy", "nvidia-cublas",
                 "nvidia-cuda-runtime", "nvidia-cuda-nvcc", "nvidia-cudnn-cu13", "nvidia-nccl-cu13"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    properties = torch.cuda.get_device_properties(0)
    lock = Path(__file__).parent / "upstream_runtime/uv.lock"
    source = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "upstream": direct_url, "packages": versions, "torch_cuda": torch.version.cuda,
              "cudnn_version": torch.backends.cudnn.version(),
              "dependency_lock_sha256": digest(lock) if lock.exists() else None,
              "source_sha256": {str(path): digest(path) for path in files},
              "device": {"name": properties.name, "uuid": str(getattr(properties, "uuid", "unavailable")),
                         "compute_capability": list(torch.cuda.get_device_capability(0)),
                         "total_memory_bytes": properties.total_memory},
              "runtime": {name: os.environ.get(name) for name in (
                  "SLURM_JOB_ID", "SLURMD_NODENAME", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "TRITON_PTXAS_PATH",
                  "TRITON_PTXAS_BLACKWELL_PATH", "FLA_USE_CUDA_GRAPH", "FLA_USE_FAST_OPS")}}
    try:
        source["nvidia_smi"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        source["nvidia_smi"] = None
    return source


def measured(torch, function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    start.record()
    result = function()
    end.record()
    end.synchronize()
    finish = time.perf_counter()
    return {"wall_seconds": finish - begin, "cuda_seconds": start.elapsed_time(end) / 1000}, result


def check_tensors(torch, tensors):
    values = []
    for tensor in tensors:
        # Outside measured regions: no per-tensor host synchronization is charged to a kernel.
        finite = bool(torch.isfinite(tensor).all().item())
        if not finite:
            raise FloatingPointError("Nonfinite upstream output or gradient")
        values.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": finite})
    return values


def learning_rate(args, step):
    if step < args.lr_warmup_steps:
        return args.learning_rate * step / args.lr_warmup_steps
    progress = min(1., (step - args.lr_warmup_steps) / (args.total_steps - args.lr_warmup_steps))
    return args.min_learning_rate + .5 * (args.learning_rate - args.min_learning_rate) * (1 + math.cos(math.pi * progress))


def graph_trial(torch, function, args, tokens):
    # Warm all upstream dispatch/autotuning on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = function()
    torch.cuda.current_stream().wait_stream(stream)
    del result
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = function()

    def replay():
        graph.replay()
        return captured

    for _ in range(args.warmup):
        _, result = measured(torch, replay)
        check_tensors(torch, result)
    samples = []
    for _ in range(args.iterations):
        sample, result = measured(torch, replay)
        check_tensors(torch, result)
        samples.append(sample)
    return {"status": "complete", "samples": samples, "summary": summarize(samples, tokens=tokens),
            "scope": "CUDA graph replay; capture, autotuning, and finite checks excluded"}


def kernel_benchmark(torch, args, report):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from scaffolds.transformer.benchmarks.upstream_inputs import kernel_inputs

    if args.key_dim != args.value_dim:
        raise ValueError("The shared fixture requires equal key and value widths")
    fixture = kernel_inputs(args.batch, args.length, args.heads, args.key_dim, args.seed)
    report["input_sha256"] = {name: array_digest(value) for name, value in fixture.items()}
    operands = {name: torch.from_numpy(value).to(device="cuda", dtype=torch.float32 if name == "g" else torch.bfloat16)
                for name, value in fixture.items() if name in ("q", "k", "v", "g", "beta", "do")}
    del fixture
    q, k, v, g, beta, cotangent = (operands[name] for name in ("q", "k", "v", "g", "beta", "do"))
    for value in (q, k, v, g, beta):
        value.requires_grad_(True)
    soft_q = (q.detach().float() * args.key_dim).to(torch.bfloat16).requires_grad_(True)
    scale = args.key_dim ** -.5

    def run_fla(backward):
        output, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                                          initial_state=None, output_final_state=False,
                                          use_qk_l2norm_in_kernel=False, state_v_first=True)
        if backward:
            gradients = torch.autograd.grad(output, (q, k, v, g, beta), cotangent)
            return output, *gradients
        return (output,)

    def run_cudnn(backward):
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            output = torch.nn.functional.scaled_dot_product_attention(
                soft_q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0., is_causal=True, scale=scale).transpose(1, 2)
            if backward:
                gradients = torch.autograd.grad(output, (soft_q, k, v), cotangent)
                return output, *gradients
            return (output,)

    # The forced context raises if unsupported; the profiler also proves actual dispatch.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        proof = run_cudnn(True)
        torch.cuda.synchronize()
    check_tensors(torch, proof)
    del proof
    events = sorted(event.key for event in profile.key_averages() if "cudnn_attention" in event.key.lower())
    if not any("backward" in event for event in events) or not any("cudnn_attention" in event for event in events):
        raise RuntimeError(f"Could not verify cuDNN SDPA forward/backward dispatch: {events}")
    report["cudnn_dispatch_proof"] = events
    report["kernel_settings"] = {"qkv_dtype": "bfloat16", "gate_dtype": "float32", "beta_dtype": "bfloat16",
        "gdn": "upstream standard tied sigmoid write/erase, native BF16 intermediates",
        "gdn_entry_point": "fla.ops.gated_delta_rule.chunk_gated_delta_rule",
        "gdn_state_v_first": True, "gdn_chunk_size": 64,
        "gdn_qk_normalization": "unit-normalized shared fixture, outside timing",
        "softmax_query_transform": "BF16(unit query multiplied by key width), outside timing",
        "softmax_backend": "SDPBackend.CUDNN_ATTENTION forced", "causal": True,
        "scale": scale, "dropout": 0., "initial_state": None, "output_final_state": False,
        "forward_scope": "training forward including autograd context; no backward",
        "backward_scope": "new forward plus all input gradients, excluding fixture preparation and finite checks"}
    report["trials"] = []
    torch.cuda.reset_peak_memory_stats()
    functions = {"upstream_fla": run_fla, "torch_cudnn": run_cudnn}
    for trial in range(args.rounds):
        names = list(functions) if trial % 2 == 0 else list(reversed(functions))
        for name in names:
            for backward in (False, True):
                function = lambda: functions[name](backward)
                warmup = []
                for _ in range(args.warmup):
                    sample, result = measured(torch, function)
                    check_tensors(torch, result)
                    warmup.append(sample)
                    del result
                samples, finite = [], None
                for _ in range(args.iterations):
                    sample, result = measured(torch, function)
                    finite = check_tensors(torch, result)
                    samples.append(sample)
                    del result
                entry = {"round": trial, "implementation": name,
                         "direction": "forward_backward" if backward else "forward",
                         "warmup_samples": warmup, "samples": samples, "finite_outputs_and_gradients": finite,
                         "summary": summarize(samples, tokens=args.batch * args.length)}
                if args.cuda_graph:
                    try:
                        entry["cuda_graph"] = graph_trial(torch, function, args, args.batch * args.length)
                    except RuntimeError as error:
                        entry["cuda_graph"] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
                report["trials"].append(entry)
                write_report(args.output, report)
                print(json.dumps({key: entry[key] for key in ("round", "implementation", "direction", "summary")}), flush=True)


def load_tokens(args):
    count, length = args.global_batch, args.length + 1
    if args.manifest is None:
        tokens = np.random.default_rng(args.seed).integers(0, args.vocab_size, size=(count, length), dtype=np.int64)
        return tokens, {"source": "synthetic", "seed": args.seed, "tokens_sha256": array_digest(tokens)}
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != "ofm-packed-sequences" or manifest.get("version") != 1
            or manifest.get("sequence_length") != length or manifest.get("dtype") not in ("uint16", "int32")):
        raise ValueError("Packed manifest does not match the requested prediction length or dtype")
    batches, remaining, used = [], count, []
    for shard in manifest["shards"]:
        path = (manifest_path.parent / shard["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError("Packed shard escapes the manifest directory")
        checksum = digest(path)
        if shard.get("sha256") is not None and checksum != shard["sha256"]:
            raise ValueError(f"Packed shard checksum mismatch: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (shard["num_sequences"], length) or values.dtype.name != manifest["dtype"]:
            raise ValueError(f"Packed shard shape or dtype mismatch: {path}")
        take = min(remaining, len(values))
        batches.append(np.array(values[:take], dtype=np.int64))
        used.append({"path": str(path), "sha256": checksum, "sequences_read": take})
        remaining -= take
        del values
        if remaining == 0:
            break
    if remaining:
        raise ValueError("Insufficient packed sequences for one global batch")
    tokens = np.concatenate(batches)
    if np.any(tokens < 0) or np.any(tokens >= args.vocab_size):
        raise ValueError("Packed token outside configured vocabulary")
    return tokens, {"source": "packed", "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
                    "shards": used, "tokens_sha256": array_digest(tokens),
                    "sequence_start": 0, "num_sequences": count, "stored_sequence_length": length,
                    "timing_scope": "one resident global batch repeated; no loading or transfers inside timing"}


def model_benchmark(torch, args, report):
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    from fla.modules import FusedCrossEntropyLoss

    if args.global_batch % args.batch:
        raise ValueError("Global batch must be divisible by microbatch")
    config = GatedDeltaNetConfig(hidden_size=args.hidden_size, num_hidden_layers=args.layers,
        intermediate_size=args.intermediate_size, vocab_size=args.vocab_size, head_dim=args.key_dim,
        num_heads=args.heads, num_v_heads=args.heads, expand_v=args.value_dim / args.key_dim,
        attn_mode="chunk", use_gate=True, use_short_conv=True, allow_neg_eigval=False, conv_size=4,
        norm_eps=1e-6, max_position_embeddings=args.length, use_cache=False, tie_word_embeddings=False,
        fuse_norm=True, fuse_swiglu=True, fuse_cross_entropy=True, fuse_linear_cross_entropy=False)
    model = GatedDeltaNetForCausalLM(config).cuda().train()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Upstream model parameters must remain FP32")
    parameters = list(model.parameters())
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        excluded = (parameter.ndim < 2 or name.endswith(".bias") or ".embeddings." in name
                    or getattr(parameter, "_no_weight_decay", False))
        (no_decay if excluded else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .1},
                                   {"params": no_decay, "weight_decay": 0.}],
                                  lr=args.learning_rate, betas=(.9, .95), eps=1e-8, fused=True)
    criterion = FusedCrossEntropyLoss(inplace_backward=True)
    tokens, data_identity = load_tokens(args)
    resident = torch.from_numpy(tokens).to(device="cuda", dtype=torch.long)
    inputs = resident[:, :-1].contiguous()
    targets = resident[:, 1:].contiguous()
    del resident, tokens
    accumulation = args.global_batch // args.batch
    report["data_identity"] = data_identity
    report["model_configuration"] = config.to_dict()
    report["parameter_count"] = sum(parameter.numel() for parameter in parameters)
    report["training_settings"] = {"model_class": "fla.models.gated_deltanet.GatedDeltaNetForCausalLM",
        "initialization": "unmodified upstream model initialization", "parameter_dtype": "float32",
        "autocast_dtype": "bfloat16", "optimizer": "torch.optim.AdamW(fused=True)",
        "learning_rate": args.learning_rate, "min_learning_rate": args.min_learning_rate,
        "schedule": "linear warmup then cosine decay", "warmup_steps": args.lr_warmup_steps,
        "learning_rate_horizon": args.total_steps,
        "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1,
        "weight_decay_exclusions": "vectors, biases, embedding weights, upstream _no_weight_decay parameters",
        "gradient_clip_norm": 1., "global_batch": args.global_batch, "micro_batch": args.batch,
        "accumulation_steps": accumulation, "prediction_positions": args.length,
        "loss": "upstream FusedCrossEntropyLoss(inplace_backward=True) applied to unmodified upstream model logits",
        "target_alignment": "input=tokens[:,:-1], target=tokens[:,1:], exactly length next-token predictions",
        "checkpointing": False, "data": "fixed GPU-resident global batch reused for every update"}

    update_count = 0

    def update():
        nonlocal update_count
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(args, update_count)
        update_count += 1
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
        for start in range(0, args.global_batch, args.batch):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=inputs[start:start + args.batch], use_cache=False, return_dict=True)
                logits = output.logits
                loss = criterion(logits.reshape(-1, args.vocab_size), targets[start:start + args.batch].reshape(-1))
            (loss / accumulation).backward()
            loss_sum.add_(loss.detach().float() / accumulation)
            del output, logits, loss
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=True)
        optimizer.step()
        with torch.no_grad():
            parameter_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(parameters)))
        return loss_sum, norm.detach(), parameter_norm

    def check_update(result):
        check_tensors(torch, result)
        return {"loss": float(result[0].item()), "gradient_norm_pre_clip": float(result[1].item()),
                "parameter_norm": float(result[2].item()), "all_finite": True}

    report["warmup_samples"], report["samples"] = [], []
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup + args.iterations):
        sample, result = measured(torch, update)
        sample.update(check_update(result))
        sample["step"] = index + 1
        sample["learning_rate"] = learning_rate(args, index)
        target = "warmup_samples" if index < args.warmup else "samples"
        report[target].append(sample)
        del result
        print(json.dumps({"phase": target, **sample}), flush=True)
        write_report(args.output, report)
    report["summary"] = summarize(report["samples"], tokens=args.global_batch * args.length)
    report["all_finite"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("kernel", "model"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2, help="Alternating implementation order for kernel trials")
    parser.add_argument("--cuda-graph", action="store_true", help="Also time captured native kernel forward/backward replay")
    parser.add_argument("--batch", type=int, default=64, help="Kernel batch or model microbatch")
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument("--length", type=int, default=2048, help="Prediction positions")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--intermediate-size", type=int, default=3008)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=400)
    parser.add_argument("--total-steps", type=int, default=19074)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.rounds, args.batch, args.global_batch, args.length,
           args.heads, args.key_dim, args.value_dim, args.hidden_size, args.layers, args.intermediate_size,
           args.vocab_size) < 1:
        parser.error("All counts and dimensions must be positive")
    if not 0 < args.lr_warmup_steps < args.total_steps:
        parser.error("Learning-rate warmup must be positive and below the unchanged horizon")
    if args.mode == "model" and args.warmup + args.iterations >= args.total_steps:
        parser.error("Model benchmark must finish before the learning-rate horizon")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run this benchmark with exactly one CUDA GPU and the isolated upstream Torch environment")
    torch.manual_seed(args.seed)
    report = {"format": "direct-upstream-torch-throughput", "schema_version": 1, "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "provenance": execution_provenance(torch),
              "limitations": ["Upstream standard GDN ties sigmoid write and erase gates; it is not the signed independent-gate model.",
                              "Native upstream BF16 intermediates differ from the local JAX GDN's FP32/TF32x3 intermediates.",
                              "Device-resident compute timings exclude data loading, validation, and checkpointing.",
                              "CUDA event timing and synchronized wall timing include the operator's normal upstream dispatch."]}
    write_report(args.output, report)
    started = time.perf_counter()
    try:
        (kernel_benchmark if args.mode == "kernel" else model_benchmark)(torch, args, report)
        report["gpu_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        write_report(args.output, report)


if __name__ == "__main__":
    main()
