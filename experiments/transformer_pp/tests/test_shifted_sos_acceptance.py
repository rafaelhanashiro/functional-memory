"""The degree-sixteen core z h_t(z) run at Q/K width 4: its parent's recipe with the core degree and the shift alone."""
import hashlib
import json

import pytest

from experiments.transformer_pp import shifted_sos_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

RUN = "r2e7f5617fdf2"
LABEL = "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4"


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


def test_the_recipe_changes_the_core_degree_backend_and_shift_alone():
    config, parent = adapter.recipe(), adapter.parent_config()
    assert changed(config, parent) == {"model.attention_degree", "model.attention_backend",
                                       "model.attention_polynomial_shift"}
    assert (parent.model.attention_degree, config.model.attention_degree) == (9, 17)
    assert config.model.attention_coefficient_input_power == 1 and config.model.qk_head_dim == 4
    assert config.model.attention_score_transform == "softplus"
    assert config.model.attention_head_rms_prescale == "max_abs"


def test_the_allocated_run_is_the_recipe():
    path = adapter.PARENT.parents[1] / RUN / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == RUN and record["label"] == LABEL == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_another_change_is_refused():
    config = adapter.recipe()
    config.model.attention_degree = 13
    with pytest.raises(ValueError):
        adapter.check_recipe(config)
