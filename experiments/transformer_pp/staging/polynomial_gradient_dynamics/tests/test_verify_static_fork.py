import hashlib
import json

import pytest
import yaml

from experiments.transformer_pp.staging.polynomial_gradient_dynamics.verify_static_fork import verify


def make_pair(root):
    parent, whole, frozen = runs = [root / name for name in ("prefix", "whole", "frozen")]
    for run in runs:
        run.mkdir()
        config = {
            "model": {"num_layers": 1, "num_heads": 1, "attention_degree": 1},
            "data": {"batch_size": 2, "sequence_length": 5},
            "optim": {"schedule_steps": 7570, "total_steps": 500 if run == parent else 600},
            "train": {"seed": 42, "dtype": "bfloat16", "micro_batch_size": 1, "eval_batches": 1, "eval_every": 100},
            "continuation": {"preserve_all_parameters": True, "freeze_static_coefficients": run == frozen},
        }
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        (run / "record.json").write_text(json.dumps(
            {"status": "complete", "selected_checkpoint": {"step": 500, "sha256": "a" * 64}}))
        (run / "data_identity.json").write_text('{"source":"same"}')
        if run != parent:
            audit = {"parent_config_sha256": hashlib.sha256((parent / "config.yaml").read_bytes()).hexdigest(),
                     "parent_checkpoint_sha256": "a" * 64, "parent_step": 500, "seed": 42,
                     "backbone_optimizer_count": 500, "new_coefficient_optimizer_count": None,
                     "data_start_sequence": 1000}
            for name in ("parent", "backbone"):
                audit[f"{name}_parameters_sha256"] = "b" * 64
                audit[f"{name}_optimizer_sha256"] = "c" * 64
            (run / "continuation.json").write_text(json.dumps(audit))
        with (run / "metrics.jsonl").open("w") as stream:
            for step in ([500] if run == parent else [500, 501, 600]):
                move = float(step - 500)
                diag = {"coefficients": {"raw": [[[0.0, move if run == whole else 0.0]]]},
                        "qk_norm_head_scales": {"raw": [[[move]]]}}
                row = {"event": "validation", "step": step, "validation_nll": 4 - 0.001 * move,
                       "learning_rate": 0.001, "training_tokens": step * 8, "validation_tokens": 8,
                       "attention_diagnostics": diag}
                print(json.dumps(row), file=stream)
    return parent, whole, frozen


def test_matched_pair_rejects_frozen_coefficient_bit_change(tmp_path):
    parent, whole, frozen = make_pair(tmp_path)
    baseline, full, locked = verify(parent, whole, frozen)
    assert baseline == [4, 4, 4]
    assert sorted(full) == sorted(locked) == [500, 501, 600]
    path = frozen / "metrics.jsonl"
    rows = list(map(json.loads, path.read_text().splitlines()))
    rows[1]["attention_diagnostics"]["coefficients"]["raw"][0][0][0] = -0.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Frozen coefficients changed at update 501"):
        verify(parent, whole, frozen)


def test_matched_pair_rejects_different_restored_adam(tmp_path):
    parent, whole, frozen = make_pair(tmp_path)
    path = frozen / "continuation.json"
    audit = json.loads(path.read_text())
    audit["backbone_optimizer_sha256"] = "d" * 64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="optimizer restore differs"):
        verify(parent, whole, frozen)
