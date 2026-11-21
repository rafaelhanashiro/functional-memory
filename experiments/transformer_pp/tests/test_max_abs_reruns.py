"""Earlier runs again under the settings a fresh copy of their recipes now takes (AGENTS.md; scaffolds.records create):
head RMSNorm's max-abs prescale and, for the exponential, its row-maximum shift. Each source run trained with
attention row sums above 2e14, where the plain head RMSNorm arithmetic's FP32 gradient is inexact."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import coefficient_forms_acceptance, softplus_sos_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parents[1] / "runs"
# Execution locations and the monitoring label, which every run has of its own.
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "wandb_run_name")
PRESCALE = frozenset({"model.attention_head_rms_prescale"})
SHIFT = frozenset({"model.attention_exp_shift"})
# Each rerun: label, source run, the settings it changes, and the adapter that pins its recipe (None: the study's
# generic acceptance, which pins none).
RERUNS = {
    "rcb0ee48a6e75": ("exp-alibi-fixed-no-rope-row-max-max-abs-head-rms", "r7820049184ef", PRESCALE | SHIFT, None),
    "r775f6499974f": ("query-m12-sum-of-squares-w-decay-l12-d640-4x-max-abs-head-rms", "r1f935f18fe21", PRESCALE,
                      coefficient_forms_acceptance),
    "rf1fad040d27d": ("fixed-m12-l12-d640-4x-max-abs-head-rms", "rcace22b32f93", PRESCALE, None),
    "r3474c2559dd7": ("query-m12-sos-softplus-prefactor-w-decay-l12-d640-4x-max-abs-head-rms", "r4589172dcd8b",
                      PRESCALE, softplus_sos_acceptance),
}


def recipe(run_id):
    """A run's loaded recipe and its settings apart from the execution keys."""
    config = load_config(RUNS / run_id / "config.yaml")
    values = config_to_dict(config)
    for name in EXECUTION_KEYS:
        values["train"].pop(name)
    return config, values


@pytest.mark.parametrize("run_id", RERUNS)
def test_a_rerun_changes_only_its_evaluation_settings(run_id):
    label, source, changes, adapter = RERUNS[run_id]
    config, values = recipe(run_id)
    original = recipe(source)[1]
    differences = {f"{section}.{name}" for section in set(values) | set(original)
                   for name in set(values.get(section, {})) | set(original.get(section, {}))
                   if values.get(section, {}).get(name) != original.get(section, {}).get(name)}
    assert differences == changes
    assert config.model.attention_head_rms_prescale == "max_abs"
    assert config.model.attention_exp_shift == ("row_max" if SHIFT <= changes else "none")
    assert config.train.wandb_run_name == label
    validate_config(config)
    if adapter is not None:
        adapter.check_recipe(config)


@pytest.mark.parametrize("run_id", RERUNS)
def test_a_rerun_is_recorded_under_its_label_and_recipe(run_id):
    record = json.loads((RUNS / run_id / "record.json").read_text())
    path = RUNS / run_id / "config.yaml"
    assert record["run_id"] == run_id and record["label"] == RERUNS[run_id][0]
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
