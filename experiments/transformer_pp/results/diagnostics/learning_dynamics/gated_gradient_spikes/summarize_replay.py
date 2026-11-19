"""Summarize both explicitly selected late replays after their measurements complete."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").exists())
RUNS = (
    ("re0db53b7b704", "b300_control", "B300 execution control of the retained per-order query-key m4 checkpoint"),
    ("rc1c463ddc494", "h100_replay", "H100 replay of the retained per-order query-key m4 checkpoint"),
)
STUDY = "transformer_pp"
sys.path.insert(0, str(REPOSITORY))

import yaml

from scaffolds.records.selections import run_sources, verify_pinned_runs


def read_json(path):
    return json.loads(path.read_text())


def extra_sources(repository, run):
    return {name: {"path": str((run / name).relative_to(repository)),
                   "sha256": hashlib.sha256((run / name).read_bytes()).hexdigest()}
            for name in ("summary.json",)}


def write_csv(name, rows):
    if not rows:
        raise ValueError(f"No rows for {name}")
    path = DIRECTORY / name
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def formatted(value):
    return format(float(value), ".10g")


def gradient_rows(run_id, report, step):
    groups, layers = [], []
    for probe in ("plain_probe", "instrumented_probe"):
        data = report[probe]
        summaries = [("accumulated", "", data["gradients"])]
        micro = data["microbatches"]["gradients"]
        for index, norm in enumerate(micro["global_norm"]):
            summaries.append(("microbatch", index, {
                "global_norm": norm,
                "group_norms": {name: values[index] for name, values in micro["group_norms"].items()},
                "layer_norms": micro["layer_norms"][index],
            }))
        for scope, index, stats in summaries:
            norm = float(stats["global_norm"])
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError(f"Cannot compute squared-gradient shares for {probe}: {norm}")
            common = {"run_id": run_id, "step": step, "probe": probe,
                      "scope": scope, "microbatch_index": index,
                      "global_gradient_norm": formatted(norm)}
            for name, value in sorted(stats["group_norms"].items()):
                groups.append({**common, "parameter_group": name,
                               "group_gradient_norm": formatted(value),
                               "squared_gradient_share": formatted(float(value) ** 2 / norm ** 2)})
            for layer, value in enumerate(stats["layer_norms"]):
                layers.append({**common, "layer_index": layer,
                               "layer_gradient_norm": formatted(value),
                               "squared_gradient_share_of_global": formatted(float(value) ** 2 / norm ** 2)})
    return groups, layers


def microbatch_rows(run_id, report, step):
    rows = []
    for probe in ("plain_probe", "instrumented_probe"):
        data = report[probe]
        micro = data["microbatches"]
        reductions = data.get("activation_reductions", {})
        for index, norm in enumerate(micro["gradients"]["global_norm"]):
            # Only keys declared 'rms' contain mean squares. The recorded head
            # minimum is already an RMS; the inverse-RMS maximum is already a gain.
            activations = {key: (math.sqrt(float(values[index])) if reductions[key] == "rms"
                                 else float(values[index]))
                           for key, values in micro["activations"].items()}

            def extreme(suffix, operation):
                matches = [(key, value) for key, value in activations.items() if key.endswith(suffix)]
                return operation(matches, key=lambda pair: pair[1]) if matches else ("", "")

            min_key, minimum = extreme("/attention_output_head_rms_min", min)
            max_key, maximum = extreme("/head_rms_norm_gain_max", max)
            _, min_layer_rms = extreme("/attention_output_rms", min)
            _, max_layer_rms = extreme("/attention_output_rms", max)
            rows.append({"run_id": run_id, "step": step, "probe": probe,
                         "microbatch_index": index, "train_nll": formatted(micro["nll"][index]),
                         "gradient_norm": formatted(norm),
                         "minimum_pre_head_rms": formatted(minimum) if minimum != "" else "",
                         "minimum_pre_head_rms_location": min_key.rsplit("/", 1)[0],
                         "maximum_inverse_rms": formatted(maximum) if maximum != "" else "",
                         "maximum_inverse_rms_location": max_key.rsplit("/", 1)[0],
                         "minimum_layer_attention_output_rms": formatted(min_layer_rms) if min_layer_rms != "" else "",
                         "maximum_layer_attention_output_rms": formatted(max_layer_rms) if max_layer_rms != "" else ""})
    return rows


def print_parity(run_id, report, record):
    print(f"Replay {run_id}: {report['device']}")
    print("Native versus historical replay (different hardware/compilation may change the trajectory):")
    print(json.dumps(report["native_replay"], indent=2, sort_keys=True))
    print("Native repeat:")
    print(json.dumps(report["native_repeat"], indent=2, sort_keys=True))
    print("Plain attribution versus native; instrumented versus plain attribution:")
    print(json.dumps(report["probe_parity"], indent=2, sort_keys=True))
    print(f"Run record status: {record['status']}; measurement status: {report['status']}; "
          f"parent files unchanged: {report['source_files_unchanged']}")
    print("Gradient shares are squared norms divided by the corresponding global squared norm. "
          "Microbatch gradients precede averaging; their norms and shares do not add to the accumulated gradient.")
    print("Layer norms exclude model-wide parameters, so their squared shares need not sum to one. "
          "Activation statistics belong to the instrumented graph; no parity threshold is silently assumed.")
    print("RMS reductions have been square-rooted; minimum head RMS and maximum inverse-RMS "
          "were already recorded in their displayed units. The inverse-RMS excludes the learned channel gain.")
    print("Native actual parameter-update norm (not a gradient norm): "
          + formatted(report["native_actual_parameter_update"]["global_norm"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--pin", action="store_true",
                        help="Explicitly create/refresh replay.yaml only after both summaries report completion.")
    args = parser.parse_args()
    repository = args.repository.resolve()
    reports = {}
    for run_id, _, _ in RUNS:
        run = repository / "experiments" / STUDY / "runs" / run_id
        summary = run / "summary.json"
        if not summary.is_file():
            raise SystemExit(f"Replay {run_id} has no summary; existing pins and outputs are unchanged.")
        report = read_json(summary)
        if report["status"] != "complete":
            raise SystemExit(f"Replay {run_id} is {report['status']}; existing pins and outputs are unchanged.")
        reports[run_id] = report
    selection_path = DIRECTORY / "replay.yaml"
    if args.pin:
        selection = {
            "question": "Where do the gradients of the largest event in the retained late replay originate?",
            "study": STUDY,
            "settings": {"interpretation": "Inspect native-history and plain-instrumented parity before attributing a spike.",
                         "gradient_share": "squared group norm divided by squared global norm",
                         "layer_norms": "exclude model-wide parameters",
                         "rms_reductions": "stored mean squares; square root before reporting"},
            "runs": [{"study": STUDY, "run_id": run_id, "role": role,
                      "label": label,
                      "sources": run_sources(repository, STUDY, run_id),
                      "additional_sources": extra_sources(
                          repository, repository / "experiments" / STUDY / "runs" / run_id)}
                     for run_id, role, label in RUNS],
        }
        selection_path.write_text(yaml.safe_dump(selection, sort_keys=False))
    selection = yaml.safe_load(selection_path.read_text())
    entries = selection["runs"]
    if ([(entry["run_id"], entry["study"]) for entry in entries]
            != [(run_id, STUDY) for run_id, _, _ in RUNS]):
        raise ValueError("Unexpected diagnostic selection")
    verify_pinned_runs(repository, entries)
    all_groups, all_layers, all_microbatches = [], [], []
    for entry in entries:
        run_id = entry["run_id"]
        run = repository / "experiments" / STUDY / "runs" / run_id
        report = reports[run_id]
        if entry["additional_sources"] != extra_sources(repository, run):
            raise ValueError(f"Replay {run_id} differs from replay.yaml; refresh pins explicitly after review.")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        first, last = report["interval"]
        if [row["step"] for row in metrics] != list(range(first, last + 1)):
            raise ValueError(f"Incomplete or duplicated native replay updates: {run_id}")
        if len(metrics) != report["native_replay"]["updates"]:
            raise ValueError(f"Summary and native replay update count disagree: {run_id}")
        largest = max(metrics, key=lambda row: row["gradient_norm_pre_clip"])
        if largest["step"] != report["worst_update"]["step"] or largest != report["worst_update"]:
            raise ValueError(f"Summary and native replay largest event disagree: {run_id}")
        print_parity(run_id, report, read_json(run / "record.json"))
        step = largest["step"]
        groups, layers = gradient_rows(run_id, report, step)
        all_groups.extend(groups)
        all_layers.extend(layers)
        all_microbatches.extend(microbatch_rows(run_id, report, step))
    write_csv("replay_group_gradients.csv", all_groups)
    write_csv("replay_layer_gradients.csv", all_layers)
    write_csv("replay_microbatches.csv", all_microbatches)


if __name__ == "__main__":
    main()
