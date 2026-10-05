"""Each initialization pair repeats its Flax-default recipe and changes only the initialization."""

import collections
from pathlib import Path

import yaml

from scaffolds.records.selections import verify_pinned_runs
from scaffolds.transformer.configs.loader import config_dicts_equal

ROOT = Path(__file__).resolve().parents[3]
SELECTION = ROOT / "results/attention_scaling/initialization.yaml"
# Within a pair, execution identity, each cluster's path to the same data manifests, and a
# pinned attention backend (backends compute the same function) may differ.
EXECUTION = {"train": ("workdir", "checkpoint_dir", "wandb_run_name", "wandb_group", "wandb_dir",
                       "source_commit", "wandb_mode"),
             "data": ("train_manifest", "eval_manifest"),
             "model": ("attention_backend",)}


def _recipe(run_id: str) -> tuple[str, dict]:
    recipe = yaml.safe_load((ROOT / "experiments/attention_scaling/runs" / run_id / "config.yaml").read_text())
    # Recipes from before the option trained with the scaffold's Flax defaults.
    initialization = recipe.setdefault("model", {}).pop("initialization", "flax_default")
    for section, keys in EXECUTION.items():
        for key in keys:
            recipe.setdefault(section, {}).pop(key, None)
    return initialization, recipe


def test_pairs_are_pinned_and_differ_only_in_initialization():
    selection = yaml.safe_load(SELECTION.read_text())
    study = Path(selection["study"]).name
    verify_pinned_runs(ROOT, [dict(entry, study=study) for entry in selection["runs"]])
    pairs = collections.defaultdict(dict)
    for entry in selection["runs"]:
        pairs[entry["label"].rpartition(", ")[0]][entry["role"]] = entry["run_id"]
    assert len(pairs) == 52
    for name, roles in pairs.items():
        assert set(roles) == {"primary", "control"}, name
        reference, primary = _recipe(roles["primary"])
        flax, control = _recipe(roles["control"])
        assert (flax, reference != "flax_default") == ("flax_default", True), name
        assert config_dicts_equal(primary, control), name
