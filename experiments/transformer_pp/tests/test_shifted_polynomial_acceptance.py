"""The degree-sixteen tanh query and fixed Taylor runs at Q/K width 4: each degree-eight partner's recipe at degree
sixteen, with the row-power shift on its fused kernel and head RMSNorm's max-abs prescale."""
import hashlib
import json

import pytest

from experiments.transformer_pp import shifted_polynomial_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

# Each allocated run: its label and partner.
RUNS = {"raec1c513a51b": ("query-m16-dk4-w-decay", "query"),
        "re125a8977319": ("fixed-m16-head-scale-dk4", "taylor")}


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("key", adapter.PARTNERS)
def test_the_recipe_changes_the_degree_backend_shift_and_prescale_alone(key):
    """The query partner trained with the plain head RMSNorm arithmetic, which the shift cannot take; the Taylor
    partner already has the prescale."""
    config, partner = adapter.recipe(key), adapter.partner_config(key)
    expected = {"model.attention_degree", "model.attention_backend", "model.attention_polynomial_shift"}
    assert changed(config, partner) == expected | ({"model.attention_head_rms_prescale"} if key == "query" else set())
    assert (partner.model.attention_degree, config.model.attention_degree) == (8, 16)
    assert (config.model.qk_head_dim, config.model.attention_score_transform) == (4, "identity")
    assert config.model.attention_head_rms_prescale == "max_abs" and adapter.partner_key(config) == key


@pytest.mark.parametrize("run_id", RUNS)
def test_the_allocated_runs_are_the_recipes(run_id):
    label, key = RUNS[run_id]
    path = adapter.RUNS / run_id / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert adapter.partner_key(config) == key
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("key", adapter.PARTNERS)
def test_another_change_is_refused(key):
    config = adapter.recipe(key)
    config.model.attention_degree = 12
    with pytest.raises(ValueError):
        adapter.check_recipe(config)
