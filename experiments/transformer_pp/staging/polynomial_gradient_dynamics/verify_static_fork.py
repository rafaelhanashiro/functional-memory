"""Check a completed matched static-coefficient fork and report its held-out loss curve."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


RUNS = Path(__file__).resolve().parents[2] / "runs"
REPLAY_NLL_ATOL = 0.002  # Full-prefix native replay tolerance used by the Taylor reset.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(run, name):
    return json.loads((run / name).read_text())


def validation_rows(run):
    rows = {}
    with (run / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "validation":
                require(row["step"] not in rows, f"Duplicate validation step in {run.name}")
                require(math.isfinite(row["validation_nll"]), f"Nonfinite validation in {run.name}")
                rows[row["step"]] = row
    return rows


def tensor_bytes(row, shape, name="coefficients"):
    """Raw FP32 model parameters round-trip through the validation record's JSON numbers."""
    raw = row["attention_diagnostics"][name]["raw"]
    tensor = np.asarray(raw, dtype="<f4")
    require(tensor.shape == shape and np.isfinite(tensor).all(), f"Invalid stored {name} tensor")
    return tensor.tobytes()


def verify(prefix, whole, frozen):
    runs = (prefix, whole, frozen)
    configs = [yaml.safe_load((run / "config.yaml").read_text()) for run in runs]
    parent, control, intervention = configs
    start, end = parent["optim"]["total_steps"], control["optim"]["total_steps"]
    schedule = parent["optim"]["schedule_steps"]
    require(start < end <= schedule, "Fork steps must lie within the original schedule")
    require(control["continuation"].get("freeze_static_coefficients", False) is False and
            intervention["continuation"]["freeze_static_coefficients"] is True, "Incorrect arm labels")
    identities = [read_json(run, "data_identity.json") for run in runs]
    require(identities[0] == identities[1] == identities[2], "Packed datasets differ")
    require(all(read_json(run, "record.json")["status"] == "complete" for run in runs),
            "All three runs must be complete")

    parent_hash = hashlib.sha256((prefix / "config.yaml").read_bytes()).hexdigest()
    parent_checkpoint = read_json(prefix, "record.json")["selected_checkpoint"]
    require(parent_checkpoint["step"] == start, "Prefix did not retain the fork checkpoint")
    batch = parent["data"]["batch_size"]
    arms = (whole, frozen)
    audits = []
    for run, config in zip(arms, configs[1:], strict=True):
        require(config["data"] == parent["data"] and config["model"] == parent["model"],
                f"Changed data or model in {run.name}")
        require(config["optim"] == dict(parent["optim"], total_steps=end), f"Changed optimizer in {run.name}")
        require(config["continuation"]["preserve_all_parameters"], f"Partial restore in {run.name}")
        for key in ("seed", "dtype", "micro_batch_size", "eval_batches"):
            require(config["train"][key] == parent["train"][key], f"Changed train.{key} in {run.name}")
        audit = read_json(run, "continuation.json")
        require(audit["parent_config_sha256"] == parent_hash and
                audit["parent_checkpoint_sha256"] == parent_checkpoint["sha256"], "Different prefix input")
        require(audit["parent_step"] == audit["backbone_optimizer_count"] == start and
                audit["new_coefficient_optimizer_count"] is None and
                audit["seed"] == parent["train"]["seed"] and audit["data_start_sequence"] == start * batch,
                f"Different optimizer clock, RNG seed or packed-data cursor in {run.name}")
        for field in ("parameters", "optimizer"):
            digest = audit[f"parent_{field}_sha256"]
            require(len(digest) == 64 and audit[f"backbone_{field}_sha256"] == digest,
                    f"{field} restore differs from prefix in {run.name}")
        audits.append(audit)
    for field in ("parent_parameters_sha256", "parent_optimizer_sha256"):
        require(audits[0][field] == audits[1][field], f"Arms started with different {field}")

    source_row = validation_rows(prefix)[start]
    native, locked = (validation_rows(run) for run in arms)
    cadence = control["train"]["eval_every"]
    expected = {s for s in range(start, end + 1) if s % cadence == 0} | {start, start + 1, end}
    require(set(native) == set(locked) == expected, "Incomplete or mismatched validation steps")
    model = parent["model"]
    shape = (model["num_layers"], model["num_heads"], model["attention_degree"] + 1)
    initial = tensor_bytes(source_row, shape)
    require(tensor_bytes(native[start], shape) == initial, "Whole arm baseline coefficients changed")
    require(tensor_bytes(native[start + 1], shape) != initial, "Whole arm coefficients did not update")
    head_shape = (*shape[:2], 1)
    head_scales = tensor_bytes(source_row, head_shape, "qk_norm_head_scales")
    for rows in (native, locked):
        require(tensor_bytes(rows[start], head_shape, "qk_norm_head_scales") == head_scales and
                tensor_bytes(rows[start + 1], head_shape, "qk_norm_head_scales") != head_scales,
                "Expected both arms to restore and update the learned QK head scales")
    baseline = [row["validation_nll"] for row in (source_row, native[start], locked[start])]
    require(max(baseline) - min(baseline) <= REPLAY_NLL_ATOL, "Native baseline NLL replay mismatch")
    targets = batch * (parent["data"]["sequence_length"] - 1)
    for step in sorted(expected):
        left, right = native[step], locked[step]
        require(tensor_bytes(right, shape) == initial, f"Frozen coefficients changed at update {step}")
        require(left["learning_rate"] == right["learning_rate"], f"Learning rates differ at {step}")
        for row in (left, right):
            require(row["training_tokens"] == step * targets and
                    row["validation_tokens"] == parent["train"]["eval_batches"] * targets,
                    f"Token counts differ at {step}")
    return baseline, native, locked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix", "whole", "frozen"):
        parser.add_argument(name, help=f"{name} run ID in the Transformer++ study")
    args = parser.parse_args()
    baseline, native, locked = verify(*(RUNS / rid for rid in (args.prefix, args.whole, args.frozen)))
    print("Baseline NLL (prefix, whole, frozen):", *(f"{v:.9f}" for v in baseline))
    print("step\twhole_nll\tfrozen_nll\tfrozen_minus_whole_nll\twhole_ppl\tfrozen_ppl")
    for step in sorted(native):
        a, b = native[step]["validation_nll"], locked[step]["validation_nll"]
        print(f"{step}\t{a:.9f}\t{b:.9f}\t{b-a:+.9f}\t{math.exp(a):.5f}\t{math.exp(b):.5f}")


if __name__ == "__main__":
    main()
