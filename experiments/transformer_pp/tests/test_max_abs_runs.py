"""The ALiBi exp run and three 12-layer width-640 4x runs take head RMSNorm's max-abs prescale and, for the
exponential, its row-maximum shift (AGENTS.md; scaffolds.records create): the plain evaluations' functions, exact in
FP32 at the attention row sums above 2e14 that these recipes reach. The 4x recipes' derivations are checked with
their comparisons (results/pretraining/scalings/l12_d640/budgets, tests/test_softplus_sos_acceptance.py)."""
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
# Each run: label, whether it is the exponential, and the adapter that pins its recipe (None: the study's generic
# acceptance, which pins none).
MAX_ABS_RUNS = {
    "rcb0ee48a6e75": ("exp-alibi-fixed-no-rope-row-max-max-abs-head-rms", True, None),
    "r775f6499974f": ("query-m12-sum-of-squares-w-decay-l12-d640-4x-max-abs-head-rms", False,
                      coefficient_forms_acceptance),
    "rf1fad040d27d": ("fixed-m12-l12-d640-4x-max-abs-head-rms", False, None),
    "r3474c2559dd7": ("query-m12-sos-softplus-prefactor-w-decay-l12-d640-4x-max-abs-head-rms", False,
                      softplus_sos_acceptance),
}


def differences(run_id, other):
    """The settings two runs' recipes differ in, apart from the execution keys."""
    first, second = (config_to_dict(load_config(RUNS / run / "config.yaml")) for run in (run_id, other))
    for values in (first, second):
        for name in EXECUTION_KEYS:
            values["train"].pop(name)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("run_id", MAX_ABS_RUNS)
def test_a_run_takes_the_stable_evaluations_and_its_adapter(run_id):
    label, exponential, adapter = MAX_ABS_RUNS[run_id]
    config = load_config(RUNS / run_id / "config.yaml")
    assert config.model.attention_head_rms_prescale == "max_abs"
    assert config.model.attention_exp_shift == ("row_max" if exponential else "none")
    assert config.train.wandb_run_name == label
    validate_config(config)
    if adapter is not None:
        adapter.check_recipe(config)


def test_the_alibi_exp_run_is_the_exp_forget_gate_run_with_fixed_slopes():
    """The no-RoPE exp forget-gate run r183f08346ddc with ALiBi's fixed slopes in place of the gate, on the causal
    strips, with the stable evaluations."""
    assert differences("rcb0ee48a6e75", "r183f08346ddc") == {
        "model.attention_forget_gate_form", "model.attention_forget_gate_constant_rate",
        "model.attention_blocked_implementation", "model.attention_exp_shift", "model.attention_head_rms_prescale"}


@pytest.mark.parametrize("run_id", MAX_ABS_RUNS)
def test_a_run_is_recorded_under_its_label_and_recipe(run_id):
    record = json.loads((RUNS / run_id / "record.json").read_text())
    path = RUNS / run_id / "config.yaml"
    assert record["run_id"] == run_id and record["label"] == MAX_ABS_RUNS[run_id][0]
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
