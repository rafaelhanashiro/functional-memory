import csv
from pathlib import Path
import sys

import pytest
import yaml

from scaffolds.records import create_run, regenerate_index
from scaffolds.transformer.native_fla.config import reference_recipe


@pytest.mark.parametrize("label", [None, "GDN FLA reference"])
def test_native_fla_cli_preserves_upstream_configuration_without_jax_defaults(tmp_path, monkeypatch, capsys, label):
    from scaffolds.records.__main__ import main

    recipe = reference_recipe()
    recipe["data"].update(train_manifest="/data/train/manifest.json", eval_manifest="/data/eval/manifest.json")
    recipe["train"].update(workdir="/runs/unallocated", checkpoint_dir="/runs/unallocated/checkpoints",
                            wandb_mode="online", wandb_project="test-project", wandb_entity="test-entity",
                            wandb_run_name="source-reference-name")
    recipe["train"]["wandb_group"] = "copied-group"
    source = tmp_path / "recipe.yaml"
    source.write_text(yaml.safe_dump(recipe, sort_keys=False))
    study = tmp_path / "study"
    # Native allocation must work without importing the JAX recipe/defaults path.
    monkeypatch.setitem(sys.modules, "scaffolds.transformer.configs", None)
    arguments = ["records", "create", "--study", str(study), "--config", str(source)]
    monkeypatch.setattr("sys.argv", arguments + (["--label", label] if label else []))
    main()
    run = Path(capsys.readouterr().out.strip())
    allocated = yaml.safe_load((run / "config.yaml").read_text())
    assert allocated["model"] == recipe["model"]
    assert allocated["optim"] == recipe["optim"]
    assert allocated["reference"] == recipe["reference"]
    assert allocated["implementation"] == "native_fla"
    assert "initialization" not in allocated["model"]
    assert "qk_norm" not in allocated["model"]
    assert allocated["train"]["wandb_group"] == "study"
    assert allocated["train"]["wandb_run_name"] == (label or "source-reference-name")
    assert "wandb_dir" not in allocated["train"]
    assert Path(allocated["train"]["workdir"]) == run


def test_native_fla_index_derives_comparable_fields_without_rewriting_recipe(tmp_path):
    recipe = reference_recipe()
    run = create_run(tmp_path / "study", recipe, "GDN FLA reference")
    frozen = (run / "config.yaml").read_bytes()
    index = regenerate_index(run.parent.parent)
    with index.open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["implementation"] == "native_fla"
    assert row["attention_mechanism"] == row["attention_family"] == "gated_deltanet"
    assert row["qk_head_dim"] == "256"
    assert row["value_head_dim"] == "512"
    assert row["gdn_erase_mode"] == "tied"
    assert row["gdn_write_activation"] == "sigmoid"
    assert row["initialization"] == "fla_reference"
    assert row["qk_norm"] == "False"
    assert row["training_steps"] == str(recipe["optim"]["total_steps"])
    assert int(row["prediction_tokens"]) == (
        recipe["optim"]["total_steps"] * recipe["data"]["batch_size"] * (recipe["data"]["sequence_length"] - 1))
    previous = index.read_bytes()
    assert regenerate_index(run.parent.parent).read_bytes() == previous
    assert (run / "config.yaml").read_bytes() == frozen


def test_historical_implementation_is_unknown_and_value_width_is_preserved(tmp_path):
    run = create_run(tmp_path / "study", {"model": {"attention_mechanism": "gated_deltanet",
                                                   "qk_head_dim": 64, "value_head_dim": 64}})
    with (run.parent.parent / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["implementation"] == ""
    assert row["value_head_dim"] == "64"
