"""Audit explicitly selected upstream B300 timing reports and regenerate pooled summaries.

Reads measurements in place. Running/missing trials remain visible as incomplete;
complete reports with invalid numerics, counts, fixture hashes, or GPU identity fail.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


JAX_KERNELS = ("gdn_independent_tanh_jax_bridge", "gdn_tied_sigmoid_jax_bridge",
               "query_polynomial_cubic_mosaic", "softmax_cudnn")
DIRECTIONS = ("forward", "forward_backward")
MODEL_SELECTION = {
    "jax_cudnn": ("cudnn-1.json", "cudnn-2.json"),
    "jax_mosaic_cubic": ("mosaic-1.json", "mosaic-2.json"),
    "upstream_fla_model": ("fla-1.json", "fla-2.json"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uuid(value):
    require(isinstance(value, str) and value not in ("", "unavailable"), "Missing GPU UUID")
    return value.lower().removeprefix("gpu-")


def gpu(report):
    return uuid(report["provenance"]["device"]["uuid"])


def finite(values, *, positive=False):
    require(bool(values), "Missing numerical samples")
    require(all(isinstance(value, (int, float)) and math.isfinite(value)
                and (not positive or value > 0) for value in values), "Invalid numerical samples")


def read_complete(path, pending):
    if not path.exists():
        pending.append({"path": str(path), "status": "missing"})
        return None
    report = json.loads(path.read_text())
    if report.get("status") in ("running", "pending"):
        pending.append({"path": str(path), "status": report["status"]})
        return None
    require(report.get("status") == "complete", f"Unsuccessful report: {path}: {report.get('status')}")
    return report


def pool(values, tokens):
    finite(values, positive=True)
    return {"samples": len(values), "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values), "tokens_per_second": tokens * len(values) / sum(values)}


def kernel_summary(jax, torch, current_gpu):
    require(gpu(jax) == gpu(torch) == current_gpu, "Current kernel GPU UUID mismatch")
    require(jax["input_sha256"] == torch["input_sha256"], "JAX/Torch fixture hashes differ")
    jc, tc = jax["configuration"], torch["arguments"]
    require(jax["shape"] == [tc["batch"], tc["length"], tc["heads"], tc["key_dim"]], "Kernel shape mismatch")
    require(tc["key_dim"] == tc["value_dim"], "Kernel value width mismatch")
    require(all(jc[key] == tc[key] for key in ("iterations", "rounds", "warmup", "seed")), "Kernel timing settings differ")
    require(set(jax["results"]) == set(JAX_KERNELS), "Unexpected JAX kernel selection")
    rows = []
    tokens = tc["batch"] * tc["length"]
    for implementation in JAX_KERNELS:
        result = jax["results"][implementation]
        for direction in DIRECTIONS:
            rounds = result["operations"][direction]["rounds"]
            require([item["round"] for item in rounds] == list(range(1, jc["rounds"] + 1)), "JAX round IDs differ")
            seconds = []
            for item in rounds:
                require(item["warmup_calls"] == jc["warmup"], "JAX warmup count differs")
                require(len(item["samples_ms"]) == jc["iterations"], "JAX sample count differs")
                checks = item["checks"]
                gradients = result["gradient_count"] if direction == "forward_backward" else 0
                require(checks["all_finite"] is True and checks["gradient_count"] == gradients,
                        "JAX output/gradient validation failed")
                require(checks["finite_by_output_and_gradient"] == [True] * (gradients + 1), "JAX finite checks incomplete")
                finite(item["samples_ms"], positive=True)
                seconds.extend(value / 1000 for value in item["samples_ms"])
            rows.append({"implementation": implementation, "direction": direction, "timing": "synchronized_wall",
                         **pool(seconds, tokens)})
    proof = torch.get("cudnn_dispatch_proof", [])
    require(any("backward" in item for item in proof)
            and any("backward" not in item for item in proof), "Missing native cuDNN dispatch proof")
    expected = {(name, direction, index) for name in ("upstream_fla", "torch_cudnn")
                for direction in DIRECTIONS for index in range(tc["rounds"])}
    selected = {(item["implementation"], item["direction"], item["round"]) for item in torch["trials"]}
    require(selected == expected and len(torch["trials"]) == len(expected), "Native kernel trial selection differs")
    samples = defaultdict(list)
    unavailable = []
    for item in torch["trials"]:
        name, direction = item["implementation"], item["direction"]
        require(len(item["samples"]) == tc["iterations"] and len(item["warmup_samples"]) == tc["warmup"],
                "Native kernel sample count differs")
        expected_count = 1 if direction == "forward" else 6 if name == "upstream_fla" else 4
        checks = item["finite_outputs_and_gradients"]
        require(len(checks) == expected_count and all(check["finite"] is True for check in checks),
                "Native output/gradient validation failed")
        for timing_key, timing in (("wall_seconds", "synchronized_wall"), ("cuda_seconds", "cuda_events")):
            values = [sample[timing_key] for sample in item["samples"]]
            finite(values, positive=True)
            samples[name, direction, timing].extend(values)
        graph = item.get("cuda_graph")
        if graph is not None:
            if graph.get("status") != "complete":
                unavailable.append({"implementation": name, "direction": direction, "round": item["round"], **graph})
            else:
                require(len(graph["samples"]) == tc["iterations"], "CUDA graph sample count differs")
                for key, timing in (("wall_seconds", "graph_wall"), ("cuda_seconds", "graph_cuda_events")):
                    values = [sample[key] for sample in graph["samples"]]
                    finite(values, positive=True)
                    samples[name, direction, timing].extend(values)
    for (name, direction, timing), values in sorted(samples.items()):
        rows.append({"implementation": name, "direction": direction, "timing": timing, **pool(values, tokens)})
    return {"status": "complete", "fixture_hashes_match": True, "shape": jax["shape"], "rows": rows,
            "cuda_graph_unavailable": unavailable}


def model_samples(report, native):
    if native:
        settings = report["training_settings"]
        iterations, warmup = report["arguments"]["iterations"], report["arguments"]["warmup"]
        samples, warmups = report["samples"], report["warmup_samples"]
        require(report["all_finite"] is True, "Native model finite checks failed")
        require(len(samples) == iterations and len(warmups) == warmup, "Native model sample count differs")
        require([sample["step"] for sample in warmups + samples] == list(range(1, warmup + iterations + 1)),
                "Native model update sequence differs")
        for sample in warmups + samples:
            require(sample["all_finite"] is True, "Native model update finite check failed")
            finite([sample[key] for key in ("loss", "gradient_norm_pre_clip", "parameter_norm")])
        return [sample["wall_seconds"] for sample in samples], settings["global_batch"] * settings["prediction_positions"]
    warmup, iterations = report["warmup_updates"], report["timed_updates"]
    samples = report["samples"]
    require(report["all_finite"] is True and report["source_recipe_unchanged"] is True, "JAX model validation failed")
    require(len(samples) == warmup + iterations, "JAX model sample count differs")
    require([sample["step"] for sample in samples] == list(range(1, warmup + iterations + 1)), "JAX model update sequence differs")
    for sample in samples:
        require(sample["finite_loss_and_update_numerics"] is True, "JAX model finite check failed")
        finite([sample["loss"], *sample["update_numerics"]])
    # Compute-only is the comparable quantity: native PyTorch reuses resident data.
    return [sample["compute_seconds"] for sample in samples[warmup:]], report["tokens_per_update"]


def summarize(root, historical, *, kernel_only=False):
    pending = []
    record = json.loads((root / "record.json").read_text())
    current_gpu = uuid(record["gpu_uuid"])
    result = {"status": "complete", "current_gpu_uuid": current_gpu, "pending": pending,
              "selected_sources": {}, "models": [],
              "limitations": ["Kernel forward/backward scope differs by mechanism and gradient count.",
                              "Upstream FLA uses native BF16 intermediates and tied sigmoid gates.",
                              "JAX full-model compute_seconds excludes loading; native FLA reuses fixed resident data.",
                              "Historical custom GDN ran on a different physical GPU; report it as historical context."]}
    kernels = []
    for filename in ("jax-kernels.json", "torch-kernels.json"):
        path = root / filename
        result["selected_sources"][filename] = str(path)
        kernels.append(read_complete(path, pending))
    result["kernels"] = (kernel_summary(*kernels, current_gpu) if all(kernels) else {"status": "incomplete"})
    if not kernel_only:
        selection = {**MODEL_SELECTION, "historical_custom_gdn": ("gdn-1.json", "gdn-2.json")}
        for name, filenames in selection.items():
            native, history = name == "upstream_fla_model", name == "historical_custom_gdn"
            values, token_counts, identities, parameters, trials = [], set(), set(), set(), []
            for filename in filenames:
                path = (historical if history else root) / filename
                result["selected_sources"][f"{name}/{filename}"] = str(path)
                report = read_complete(path, pending)
                if report is None:
                    continue
                identity = gpu(report)
                require(history or identity == current_gpu, f"Current full-model GPU mismatch: {path}")
                seconds, tokens = model_samples(report, native)
                finite(seconds, positive=True)
                values.extend(seconds)
                identities.add(identity)
                token_counts.add(tokens)
                parameters.add(report["parameter_count"])
                trials.append({"file": filename, **pool(seconds, tokens)})
            require(len(identities) <= 1 and len(token_counts) <= 1 and len(parameters) <= 1,
                    f"Full-model repeat configuration/identity mismatch: {name}")
            entry = {"implementation": name, "status": "complete" if len(trials) == len(filenames) else "incomplete",
                     "complete_trials": len(trials), "expected_trials": len(filenames), "trials": trials,
                     "scope": "historical_different_gpu" if history else "current_gpu",
                     "sample_key": "wall_seconds" if native else "compute_seconds"}
            if values:
                entry.update(pool(values, next(iter(token_counts))))
                entry.update(gpu_uuid=next(iter(identities)), parameter_count=next(iter(parameters)),
                             tokens_per_update=next(iter(token_counts)))
            result["models"].append(entry)
    if pending:
        result["status"] = "incomplete"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "runs/upstream-b300")
    parser.add_argument("--historical", type=Path, default=Path(__file__).parent / "runs/gdn-mosaic-b300")
    parser.add_argument("--kernel-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full regenerated summary as JSON")
    args = parser.parse_args()
    result = summarize(args.root.resolve(), args.historical.resolve(), kernel_only=args.kernel_only)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return
    print(f"Audit: {result['status']}; current GPU {result['current_gpu_uuid']}")
    grouped = defaultdict(dict)
    for row in result["kernels"].get("rows", []):
        if row["timing"] in ("synchronized_wall", "graph_wall"):
            grouped[row["implementation"], row["timing"]][row["direction"]] = 1000 * row["mean_seconds"]
    print("Kernel pooled means (ms): implementation / timing / forward / forward+backward")
    for (name, timing), values in grouped.items():
        print(f"  {name:37s} {timing:18s} {values.get('forward', float('nan')):8.3f} {values.get('forward_backward', float('nan')):8.3f}")
    for row in result["models"]:
        measurement = (f"{row['mean_seconds']:.4f} sec/update; {row['tokens_per_second']:,.0f} tok/s"
                       if "mean_seconds" in row else "no complete measurements")
        print(f"  {row['implementation']}: {row['status']} ({row['complete_trials']}/2), {measurement}; {row['sample_key']}; {row['scope']}")
    for item in result["pending"]:
        print(f"  pending: {Path(item['path']).name}: {item['status']}")
    for item in result["kernels"].get("cuda_graph_unavailable", []):
        print(f"  graph unavailable: {item['implementation']} {item['direction']} round {item['round']}")


if __name__ == "__main__":
    main()
