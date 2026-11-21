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
