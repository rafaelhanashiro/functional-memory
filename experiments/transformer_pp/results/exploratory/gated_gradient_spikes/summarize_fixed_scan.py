"""Summarize the explicit fixed-state scan after its scan and precision probe finish."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[5]
RUN_ID = "r1736cf7a019c"
STUDY = "transformer_pp"
PARITY_STAGES = ("scan_parity", "microbatch_parity", "capture_parity", "isolated_forward_parity")
FINISHED_PRECISION = {"complete", "skipped", "scan_validation_failed", "local_parity_failed",
                      *(name + "_failed" for name in PARITY_STAGES)}
sys.path.insert(0, str(REPOSITORY))

import yaml

from scaffolds.records.selections import run_sources, verify_pinned_runs
from experiments.transformer_pp.results.exploratory.gated_gradient_spikes.summarize_replay import (
    gradient_rows, microbatch_rows, read_json, write_csv,
)


def extra_sources(repository, run):
    return {name: {"path": str((run / name).relative_to(repository)),
                   "sha256": hashlib.sha256((run / name).read_bytes()).hexdigest()}
            for name in ("summary.json", "precision_probe.json")}


def finished_reports(run):
    for name in ("summary.json", "precision_probe.json"):
        if not (run / name).is_file():
            raise SystemExit(f"{RUN_ID}: {name} is absent; existing pins and outputs are unchanged.")
    report, precision = read_json(run / "summary.json"), read_json(run / "precision_probe.json")
    if report.get("status") != "complete" or precision.get("status") not in FINISHED_PRECISION:
        raise SystemExit(f"{RUN_ID}: scan={report.get('status')}, precision={precision.get('status')}; "
                         "existing pins and outputs are unchanged.")
    if report.get("format") != "frozen_batch_gradient_scan" or precision.get("format") != "frozen_attention_vjp":
        raise ValueError("Unexpected fixed-scan or precision-probe format")
    if not report.get("finished_utc") or not precision.get("finished_utc"):
        raise ValueError("Finished measurements require completion timestamps")
    return report, precision


def verify_measurements(run, report, precision):
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    count, first = report["sampled_count"], report["first_batch_update"]
    if not 1 <= count <= report["max_batches"] or len(metrics) != count:
        raise ValueError("Scan sample count disagrees with native metrics")
    if [row["batch_update"] for row in metrics] != list(range(first, first + count)):
        raise ValueError("Scan batches are incomplete or duplicated")
    for row in metrics:
        cursor = (row["batch_update"] - 1) * report["global_batch_size"]
        if (row["event"] != "frozen_batch_gradient" or row["state_step"] != report["state_step"]
                or row["first_sequence"] != cursor
                or row["next_sequence"] != cursor + report["global_batch_size"]
                or row["exceeds_threshold"] != (row["gradient_norm_pre_clip"] > report["threshold"])):
            raise ValueError("Native batch coordinate or threshold metadata disagrees with summary")
    largest = max(metrics, key=lambda row: row["gradient_norm_pre_clip"])
    if largest != report["largest_batch"] or report["hit_threshold"] != largest["exceeds_threshold"]:
        raise ValueError("Largest native batch or threshold hit disagrees with summary")
    if any(row["exceeds_threshold"] for row in metrics[:-1]):
        raise ValueError("Scan continued past its first threshold hit")
    if not report["hit_threshold"] and count != report["max_batches"]:
        raise ValueError("Scan stopped before its limit without a threshold hit")
    if (report["training_updates_applied"] != 0 or not report["fixed_state_unchanged"]
            or not report["source_files_unchanged"]
            or len({report[name] for name in ("fixed_state_sha256", "fixed_state_after_scan_sha256",
                                             "fixed_state_final_sha256")}) != 1):
        raise ValueError("Fixed-state or protected-source integrity check failed")
    if precision["scan_summary_sha256"] != hashlib.sha256((run / "summary.json").read_bytes()).hexdigest():
        raise ValueError("Precision probe did not use the current completed scan summary")
    status = precision["status"]
    if status == "skipped":
        if report["hit_threshold"]:
            raise ValueError("Precision probe skipped despite a threshold hit")
    else:
        if not report["hit_threshold"]:
            raise ValueError("Precision attribution attempted without a threshold hit")
        if status not in {"scan_validation_failed", "scan_parity_failed"}:
            selected = precision["selection"]
            if (selected["state_step"] != report["state_step"]
                    or selected["batch_update"] != largest["batch_update"]):
                raise ValueError("Precision probe selection disagrees with the fixed scan")
        if status != "scan_validation_failed":
            for name in PARITY_STAGES:
                expected = status != name + "_failed"
                if precision[name]["passed"] is not expected:
                    raise ValueError(f"Precision status {status} disagrees with {name}")
                if not expected:
                    break
        if status in {"complete", "local_parity_failed"}:
            rows = precision["rows"]
            if not rows or not precision["source_files_unchanged"]:
                raise ValueError("Local precision comparisons require rows and source integrity")
            eligible = []
            for row in rows:
                if row["attribution_eligible"] != row["selected_forward_parity"]["passed"]:
                    raise ValueError("Local attribution eligibility disagrees with forward parity")
                for arithmetic in ("bf16", "bf16x3"):
                    if not row["precision"][arithmetic]["gradients"]:
                        raise ValueError("Local precision comparison lacks gradient measurements")
                eligible.append(row["attribution_eligible"])
            expected = status == "complete"
            if precision["attribution_eligible"] is not expected or all(eligible) is not expected:
                raise ValueError("Local precision status disagrees with attribution eligibility")
    if status not in {"complete", "local_parity_failed"} and not precision.get("reason"):
        raise ValueError("Skipped or rejected precision probes require an explicit reason")
    return largest


def fixed_coordinates(rows, state_step):
    # The reused helpers label replay training steps. Here the same integer is
    # only a packed-data/RNG coordinate; all measurements use one optimizer state.
    return [{"run_id": row["run_id"], "state_step": state_step, "batch_update": row["step"],
             **{key: value for key, value in row.items() if key not in {"run_id", "step"}}}
            for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--pin", action="store_true",
                        help="Create/refresh fixed_scan.yaml only after scan and precision measurements finish.")
    args = parser.parse_args()
    repository = args.repository.resolve()
    run = repository / "experiments" / STUDY / "runs" / RUN_ID
    report, precision = finished_reports(run)
    largest = verify_measurements(run, report, precision)
    groups, _ = gradient_rows(RUN_ID, report, largest["batch_update"])
    groups = fixed_coordinates(groups, report["state_step"])
    microbatches = fixed_coordinates(microbatch_rows(RUN_ID, report, largest["batch_update"]), report["state_step"])
    selection_path = DIRECTORY / "fixed_scan.yaml"
    if args.pin:
        selection = {
            "question": "Can data batches produce a native gradient outlier at the fixed retained checkpoint?",
            "study": STUDY,
            "settings": {
                "coordinates": "batch_update identifies packed data/RNGs; state_step remains fixed",
                "gradient_share": "squared group norm divided by squared global norm",
                "rms_reductions": "stored mean squares; square root before reporting",
                "interpretation": "Inspect native-repeat, attribution, and capture parity before assigning a mechanism.",
            },
            "runs": [{"study": STUDY, "run_id": RUN_ID, "role": "fixed_checkpoint_batch_scan",
                      "label": "Fixed checkpoint batch scan of per-order query-key m4 NoPE",
                      "sources": run_sources(repository, STUDY, RUN_ID),
                      "additional_sources": extra_sources(repository, run)}],
        }
        selection_path.write_text(yaml.safe_dump(selection, sort_keys=False))
    selection = yaml.safe_load(selection_path.read_text())
    entries = selection["runs"]
    if [(entry["study"], entry["run_id"]) for entry in entries] != [(STUDY, RUN_ID)]:
        raise ValueError("Unexpected fixed-scan selection")
    verify_pinned_runs(repository, entries)
    if entries[0]["additional_sources"] != extra_sources(repository, run):
        raise ValueError("Scan or precision measurements differ from fixed_scan.yaml; refresh pins after review.")
    print(f"Fixed-state scan {RUN_ID}: {report['device']}; state_step={report['state_step']}")
    print(f"Samples={report['sampled_count']}/{report['max_batches']}; threshold={report['threshold']}; "
          f"hit={report['hit_threshold']}; training updates applied={report['training_updates_applied']}")
    print("Largest native batch: " + json.dumps(largest, sort_keys=True))
    print("Fixed state hashes match before scan, after scan, and after probes: " + report["fixed_state_sha256"])
    print("Native repeat: " + json.dumps(report["native_repeat"], sort_keys=True))
    print("Plain versus native; instrumented versus plain: " + json.dumps(report["probe_parity"], sort_keys=True))
    print(f"Precision probe status: {precision['status']}")
    for name in PARITY_STAGES:
        if name in precision:
            print(name.replace("_", " ").capitalize() + ": " + json.dumps(precision[name], sort_keys=True))
    if "reason" in precision:
        print("Precision probe outcome: " + precision["reason"])
    if precision["status"] == "local_parity_failed":
        print("Precision probe outcome: selected-row forward parity rejected local attribution.")
        for index, row in enumerate(precision["rows"]):
            print(f"Candidate {index} selected forward parity: "
                  + json.dumps(row["selected_forward_parity"], sort_keys=True))
    if precision["status"] == "complete":
        print(f"Local same-input VJP candidate rows measured: {len(precision['rows'])}; "
              "these selected rows do not attribute the entire global gradient.")
    else:
        print("No local VJP conclusion is available from this precision probe.")
    print("Gradient shares use squared norms. Microbatch gradients precede averaging and their norms do not add. "
          "Activation statistics belong to the instrumented graph; inspect parity before interpreting them.")
    print("RMS reductions are square-rooted; minimum head RMS and maximum inverse-RMS are already in displayed units. "
          "Inverse-RMS excludes the learned channel gain. This is not a historical training replay.")
    write_csv("fixed_scan_groups.csv", groups)
    write_csv("fixed_scan_microbatches.csv", microbatches)


if __name__ == "__main__":
    main()
