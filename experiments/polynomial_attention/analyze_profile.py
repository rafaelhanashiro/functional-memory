"""Summarize actual GPU stream events from benchmark JAX/Perfetto traces.

  python3 -m experiments.polynomial_attention.analyze_profile \
    /path/on/nfs/profiles --output /path/on/nfs/profile-summary.json

Only GPU stream events contribute GPU time; nested host/HLO annotations never
do. Kernel-duration sums can exceed elapsed active time when streams overlap.
Idle means no recorded GPU stream activity inside the annotated host update,
not a measurement of SM utilization. Command-buffer replay may hide operation
scopes; unidentifiable GEMMs remain explicitly unattributed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import re


CATEGORIES = (
    "pallas_attention", "cudnn_attention", "attention_other", "mlp",
    "projections", "cross_entropy", "collectives", "gemm_unattributed", "other",
)
SCOPE_KEYS = ("hlo_op", "tf_op", "op_name", "hlo_op_name", "framework_op")


def kernel_category(event: dict) -> str:
    name = event.get("name", "").lower()
    args = event.get("args", {})
    scope = " ".join(str(args.get(key, "")) for key in SCOPE_KEYS).lower()
    text = name + " " + scope
    if re.search(r"taylor_\d+_(?:forward|dq|dkv)", name):
        return "pallas_attention"
    if "cudnn" in name and any(token in name for token in ("sdpa", "flash", "fmha", "attention")):
        return "cudnn_attention"
    if any(token in text for token in ("nccl", "all-reduce", "all_reduce", "all-gather", "all_gather", "reduce-scatter", "reduce_scatter")):
        return "collectives"
    if "token_cross_entropy" in scope or "cross_entropy" in scope:
        return "cross_entropy"
    if "vocabulary_projection" in scope:
        return "projections"
    if re.search(r"(?:^|[/ :])mlp(?:[/ :]|$)", scope):
        return "mlp"
    is_gemm = any(token in name for token in ("gemm", "nvjet", "cutlass", "mma")) or "dot_general" in scope
    if "attention" in scope:
        return "projections" if is_gemm else "attention_other"
    return "gemm_unattributed" if is_gemm else "other"


def union_duration(intervals: list[tuple[float, float]]) -> float:
    """Measure a union of half-open intervals in the input timestamp units."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def select_events(events: list[dict], step_name: str) -> tuple[dict, dict, list, dict]:
    processes = {
        event["pid"]: event.get("args", {}).get("name", "")
        for event in events if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    gpu_processes = {pid: name for pid, name in processes.items() if re.search(r"(?:/device:)?GPU[: ]\d+", name)}
    stream_tracks = {
        (event["pid"], event["tid"])
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
        and event.get("pid") in gpu_processes
        and event.get("args", {}).get("name", "").startswith("Stream #")
    }
    if not gpu_processes or not stream_tracks:
        raise ValueError("Trace has no named GPU Stream # tracks")
    steps = sorted((
        event for event in events if event.get("ph") == "X" and event.get("name") == step_name
        and event.get("pid") not in gpu_processes and float(event.get("dur", 0)) > 0
    ), key=lambda event: float(event["ts"]))
    if not steps:
        raise ValueError(f"Trace has no host '{step_name}' update annotations")
    if any(float(left["ts"]) + float(left["dur"]) > float(right["ts"]) for left, right in zip(steps, steps[1:])):
        raise ValueError("Host update annotations overlap; cannot assign exclusive update windows")
    selected = defaultdict(list)
    seen = set()
    exclusions = Counter()
    for event in events:
        if event.get("ph") != "X" or (event.get("pid"), event.get("tid")) not in stream_tracks:
            continue
        if float(event.get("dur", 0)) <= 0:
            continue
        identity = (event["pid"], event["tid"], event.get("name"), event["ts"], event["dur"])
        if identity in seen:
            exclusions["duplicate_stream_events"] += 1
            continue
        seen.add(identity)
        name = event.get("name", "").lower()
        is_kernel = "kernel_details" in event.get("args", {})
        is_memory = name.startswith(("memcpy", "memset"))
        if not is_kernel and not is_memory:
            exclusions["stream_events_without_kernel_or_memory_identity"] += 1
            continue
        selected[event["pid"]].append((event, is_kernel))
    return gpu_processes, selected, steps, dict(exclusions)


def summarize_window(selected: list, start: float, end: float, top_kernels: int) -> dict:
    kernels, activity = [], []
    category_us = Counter({category: 0.0 for category in CATEGORIES})
    category_counts = Counter({category: 0 for category in CATEGORIES})
    names_us, names_count = Counter(), Counter()
    memory_us = 0.0
    kernel_count = 0
    hidden_scope_us = 0.0
    for event, is_kernel in selected:
        event_start = max(start, float(event["ts"]))
        event_end = min(end, float(event["ts"]) + float(event["dur"]))
        if event_end <= event_start:
            continue
        duration = event_end - event_start
        activity.append((event_start, event_end))
        if not is_kernel:
            memory_us += duration
            continue
        kernel_count += 1
        kernels.append((event_start, event_end))
        category = kernel_category(event)
        category_us[category] += duration
        category_counts[category] += 1
        names_us[event["name"]] += duration
        names_count[event["name"]] += 1
        if str(event.get("args", {}).get("hlo_op", "")).startswith("command_buffer"):
            hidden_scope_us += duration
    wall_us = end - start
    kernel_sum_us = sum(category_us.values())
    kernel_union_us = union_duration(kernels)
    activity_union_us = union_duration(activity)
    return {
        "wall_ms": wall_us / 1000,
        "kernel_count": kernel_count,
        "kernel_sum_ms": kernel_sum_us / 1000,
        "kernel_active_union_ms": kernel_union_us / 1000,
        "kernel_overlap_excess_ms": max(0.0, kernel_sum_us - kernel_union_us) / 1000,
        "memory_operation_sum_ms": memory_us / 1000,
        "gpu_activity_union_ms": activity_union_us / 1000,
        "no_recorded_gpu_activity_ms": max(0.0, wall_us - activity_union_us) / 1000,
        "gpu_activity_percent_of_wall": 100 * activity_union_us / wall_us,
        "command_buffer_kernel_sum_percent": 100 * hidden_scope_us / kernel_sum_us if kernel_sum_us else 0.0,
        "categories": {
            category: {
                "kernel_count": category_counts[category], "kernel_sum_ms": category_us[category] / 1000,
                "percent_of_kernel_sum": 100 * category_us[category] / kernel_sum_us if kernel_sum_us else 0.0,
            }
            for category in CATEGORIES
        },
        "top_kernels": [
            {"name": name, "kernel_sum_ms": duration / 1000, "count": names_count[name]}
            for name, duration in names_us.most_common(top_kernels)
        ],
    }


def analyze_events(events: list[dict], *, step_name: str = "train", top_kernels: int = 20) -> dict:
    gpu_processes, selected, steps, exclusions = select_events(events, step_name)
    devices = []
    for pid, name in sorted(gpu_processes.items(), key=lambda item: item[1]):
        updates = []
        for step in steps:
            start = float(step["ts"])
            update = summarize_window(selected[pid], start, start + float(step["dur"]), top_kernels)
            update["step_num"] = step.get("args", {}).get("step_num")
            updates.append(update)
        additive_keys = (
            "wall_ms", "kernel_count", "kernel_sum_ms", "kernel_active_union_ms",
            "kernel_overlap_excess_ms", "memory_operation_sum_ms", "gpu_activity_union_ms",
            "no_recorded_gpu_activity_ms",
        )
        mean = {key: sum(update[key] for update in updates) / len(updates) for key in additive_keys}
        mean["gpu_activity_percent_of_wall"] = 100 * mean["gpu_activity_union_ms"] / mean["wall_ms"]
        mean["command_buffer_kernel_sum_percent"] = (
            sum(update["command_buffer_kernel_sum_percent"] * update["kernel_sum_ms"] for update in updates)
            / sum(update["kernel_sum_ms"] for update in updates) if mean["kernel_sum_ms"] else 0.0
        )
        mean["categories"] = {}
        for category in CATEGORIES:
            duration = sum(update["categories"][category]["kernel_sum_ms"] for update in updates) / len(updates)
            mean["categories"][category] = {
                "kernel_sum_ms": duration,
                "kernel_count": sum(update["categories"][category]["kernel_count"] for update in updates) / len(updates),
                "percent_of_kernel_sum": 100 * duration / mean["kernel_sum_ms"] if mean["kernel_sum_ms"] else 0.0,
            }
        devices.append({"pid": pid, "device": name, "mean_per_update": mean, "updates": updates})
    return {
        "step_count": len(steps), "step_annotation": step_name,
        "trial_annotations": [event.get("args", {}) for event in events if event.get("name") == "benchmark_train_step" and event.get("ph") == "X"],
        "excluded_stream_events": exclusions, "devices": devices,
    }


def analyze_file(path: Path, *, step_name: str, top_kernels: int, max_bytes: int) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"{path}: expanded trace exceeds {max_bytes} bytes; process larger traces in a compute allocation")
    data = json.loads(raw)
    result = analyze_events(data["traceEvents"], step_name=step_name, top_kernels=top_kernels)
    return {"trace": str(path.resolve()), "expanded_bytes": len(raw),
            "trace_sha256": hashlib.sha256(raw).hexdigest(), **result}


def discover_traces(paths: list[Path]) -> list[Path]:
    found = set()
    for path in paths:
        if path.is_dir():
            traces = list(path.rglob("perfetto_trace.json.gz"))
            if not traces:
                traces = list(path.rglob("*.trace.json.gz"))
            found.update(trace.resolve() for trace in traces)
        else:
            if not path.is_file():
                raise FileNotFoundError(path)
            found.add(path.resolve())
    if not found:
        raise ValueError("No Perfetto JSON traces found")
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", type=Path, nargs="+", help="Trace JSON(.gz) files or profile directories")
    parser.add_argument("--output", type=Path, help="JSON summary; default prints JSON to stdout")
    parser.add_argument("--step-name", default="train")
    parser.add_argument("--top-kernels", type=int, default=20)
    parser.add_argument("--max-expanded-mib", type=int, default=100)
    args = parser.parse_args()
    if args.top_kernels < 1 or args.max_expanded_mib < 1:
        parser.error("top-kernels and max-expanded-mib must be positive")
    report = {
        "format": "polynomial-attention-perfetto-summary", "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": {
            "timestamp_unit": "Chrome trace ts/dur are microseconds, regardless of displayTimeUnit",
            "selection": "GPU Stream # tracks only; kernel_details identifies kernels; Memcpy/Memset count only toward memory/GPU activity",
            "windows": "clipped to nonoverlapping synchronized host update annotations; gaps between updates excluded",
            "categories": "exclusive name/scope heuristics; categories sum to kernel duration, not elapsed wall time",
            "scope_limit": "CUDA command-buffer replay can erase op scopes; zero attributed MLP/loss time does not mean no MLP/loss work",
            "overlap": "kernel sum minus interval union; not a measure of SM occupancy or critical-path cost",
            "idle": "wall minus union of recorded kernel/memory intervals; includes host delay and trace overhead",
            "profiling": "profiled durations can differ from unprofiled benchmark medians",
        },
        "traces": [
            analyze_file(path, step_name=args.step_name, top_kernels=args.top_kernels,
                         max_bytes=args.max_expanded_mib * 1024 * 1024)
            for path in discover_traces(args.traces)
        ],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized)
        temporary.replace(args.output)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
