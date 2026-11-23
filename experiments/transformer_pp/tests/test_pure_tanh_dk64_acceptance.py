"""The pure tanh runs at Q/K width 64 and their affine counterpart: the degree-eight tanh query + W decay partner's
recipe at the run's degree on the row-power shifted kernel with head RMSNorm's max-abs prescale, pure tanh runs with the
Q/K width 4 runs' coefficient settings."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.transformer_pp import pure_tanh_dk64_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

# Each allocated run: its label, coefficient activation and degree.
RUNS = {}
# The repository's byte copies of the dataset manifests, another path to the partner's manifests.
CORPUS = Path(__file__).resolve().parents[3] / "scaffolds/transformer/data/corpora/fineweb_12b"


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("activation, degree", adapter.CELLS)
def test_each_recipe_is_the_partner_on_the_shifted_kernel(activation, degree):
    config, partner = adapter.recipe(activation, degree), adapter.partner_config()
    changes = {**(adapter.PURE_TANH if activation == "pure_tanh" else {}), **adapter.SHIFTED}
    expected = {f"model.{name}" for name, value in changes.items() if getattr(partner.model, name) != value}
    expected |= {"model.attention_degree"} if degree != partner.model.attention_degree else set()
    assert changed(config, partner) == expected
    model = config.model
    assert (model.qk_head_dim, model.attention_score_transform, model.attention_coefficient_activation) == (
        64, "identity", activation)
    assert (model.attention_backend, model.attention_polynomial_shift, model.attention_head_rms_prescale) == (
        "fused", "row_power", "max_abs")
    assert model.attention_coefficient_kernel_weight_decay and config.optim.weight_decay == pytest.approx(0.1)
    assert config.optim.query_coefficient_learning_rate_scale == 1.0 and config.train.seed == partner.train.seed
    if activation == "pure_tanh":
        assert model.attention_coefficient_tanh_bias_initial == pytest.approx(math.atanh(0.5))
        assert (model.attention_coefficient_projection_initial_std, model.attention_coefficient_input_source,
                model.attention_coefficient_input_normalization) == (0.0, "pre_rmsnorm", "l2")
    adapter.check_recipe(config)


@pytest.mark.parametrize("run_id", RUNS)
def test_the_allocated_runs_are_the_recipes(run_id):
    label, activation, degree = RUNS[run_id]
    path = adapter.PARTNER.parents[1] / run_id / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert adapter.recipe_key(config) == (activation, degree)
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_copies_are_accepted_by_their_bytes():
    config = adapter.recipe("pure_tanh", 16)
    config.data.train_manifest = str(CORPUS / "train/manifest.json")
    config.data.eval_manifest = str(CORPUS / "eval/manifest.json")
    adapter.check_recipe(config)
    config.data.eval_manifest = str(CORPUS / "corpus.json")
    with pytest.raises(ValueError, match="eval_manifest"):
        adapter.check_recipe(config)


def test_another_change_or_cell_is_refused():
    config = adapter.recipe("pure_tanh", 8)
    config.model.attention_coefficient_tanh_bias_initial = 0.0
    with pytest.raises(ValueError):
        adapter.check_recipe(config)
    with pytest.raises(ValueError):
        adapter.recipe("tanh", 8)
    with pytest.raises(ValueError):
        adapter.main(["--gpu-type", "B200"])
