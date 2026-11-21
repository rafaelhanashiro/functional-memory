import csv
from pathlib import Path

import yaml


def test_even_gate_roles_identify_the_three_frozen_recipes():
    topic = Path(__file__).resolve().parents[1]
    study = next(parent for parent in topic.parents if (parent / "sweep.yaml").is_file())
    roles = {
        "cosine_half_dk4": ("rms_norm_cosine", "zero"),
        "qk_half_dk4": ("rms_norm", "zero"),
        "qk_taylor_dk4": ("rms_norm", "logit_taylor"),
    }
    selected = yaml.safe_load((topic / "selection.yaml").read_text())["runs"]
    with (topic / "summary.csv").open() as stream:
        summary = {row["role"]: row["run_id"] for row in csv.DictReader(stream)}
    for role, (score, bias) in roles.items():
        entry, = (entry for entry in selected if entry["role"] == role)
        assert summary[role] == entry["run_id"]
        config = yaml.safe_load((study / "runs" / entry["run_id"] / "config.yaml").read_text())
        model = config["model"]
        assert model["qk_head_dim"] == 4
        assert model["attention_coefficient_orders"] == [2, 4, 6, 8]
        assert model["qk_norm_type"] == score
        assert model["attention_coefficient_sigmoid_bias_initial"] == bias


def test_each_exp_gate_run_is_paired_with_its_degree_eight_control():
    topic = Path(__file__).resolve().parents[1]
    study = next(parent for parent in topic.parents if (parent / "sweep.yaml").is_file())
    selection = yaml.safe_load((topic / "selection.yaml").read_text())
    roles = {row["role"]: row["run_id"] for row in selection["runs"]}
    pairs = selection["settings"]["gated_exp_m16_dk4"]["matched_degree_eight_runs"]
    assert len(pairs) == 3
    for role, control in pairs.items():
        models = [yaml.safe_load((study / "runs" / roles[key] / "config.yaml").read_text())["model"]
                  for key in (role, control)]
        hybrid, parent = models
        assert hybrid["attention_degree"] == 16 and parent["attention_degree"] == 8
        assert hybrid["attention_coefficient_orders"] == list(range(2, 17, 2))
        assert hybrid["attention_coefficient_exp_gate"] and not parent.get("attention_coefficient_exp_gate", False)
        for setting in ("qk_head_dim", "qk_norm_type", "qk_norm_head_scale", "attention_coefficient_sigmoid_bias_initial"):
            assert hybrid[setting] == parent[setting]
