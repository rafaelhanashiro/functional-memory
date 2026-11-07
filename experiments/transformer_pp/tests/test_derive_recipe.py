import pytest
import yaml

from experiments.transformer_pp.derive_recipe import (
    RUNS, _changes, _defined, derive, main, parameter_count, reference_parameter_count,
)

# Run identity and output locations, set by `scaffolds.records create`, not by the recipe rules.
IDENTITY = {"train.workdir", "train.checkpoint_dir", "train.wandb_run_name", "continuation"}
PARTNERS = ("r9040f12f3d80", "r6a76e7fbde45", "r816ab0fbe119", "r4339a245820e", "rca97cc9db8f2",
            "r4a80e28ce0a7", "r183f08346ddc")


def recipe(run):
    return yaml.safe_load((RUNS / run / "config.yaml").read_text())


@pytest.mark.parametrize("run", PARTNERS)
def test_the_reference_shape_and_budget_reproduce_each_partner(run):
    derived, dropped = derive(recipe(run))
    assert _changes(_defined(recipe(run))[0], derived) == {}
    # The query FoX recipe predates the removal of one weight-decay key; nothing else is dropped.
    assert dropped == (["model.attention_coefficient_key_kernel_weight_decay"] if run == "r4a80e28ce0a7" else [])


@pytest.mark.parametrize("partner,run", [
    ("r816ab0fbe119", "r4f1682c1b54d"), ("r6a76e7fbde45", "rfe12e5c36e94"),
    ("r4339a245820e", "raa81f92b6aaf"), ("r2b22f2ff6461", "reb9b5be4ff1b"),
])
def test_the_budget_rule_reproduces_the_existing_2_5x_runs(partner, run):
    derived, _ = derive(recipe(partner), budget=2.5)
    differences = _changes(derived, _defined(recipe(run))[0])
    assert {key for key in differences if key.split(".")[0] != "continuation"} <= IDENTITY
    assert (derived["optim"]["total_steps"], derived["optim"]["warmup_steps"]) == (18925, 1138)
    assert [derived["train"][key] for key in ("eval_early_every", "eval_early_until", "eval_every")] == [250, 3750, 1250]


def test_the_twelve_layer_width_640_shape():
    derived, _ = derive(recipe("rca97cc9db8f2"), layers=12, width=640)
    model, optim = derived["model"], derived["optim"]
    assert [model[key] for key in ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "intermediate_size")] \
        == [12, 640, 10, 10, 1664]
    assert (model["qk_head_dim"], model["value_head_dim"]) == (64, 64)
    assert model["attention_coefficient_projection_initial_std"] == 640 ** -0.5
    assert derived["data"]["batch_size"] == 128
    assert (optim["total_steps"], optim["warmup_steps"]) == (7551, 454)
    assert (optim["learning_rate"], optim["min_learning_rate"]) == (1e-3, 1e-4)
    # The same 4,864 validation sequences as the reference's 19 batches of 256, at the same cadence.
    assert derived["train"]["eval_batches"] * derived["data"]["batch_size"] == 19 * 256
    assert [derived["train"][key] for key in ("eval_early_every", "eval_early_until", "eval_every")] == [100, 1500, 500]
    assert derived["train"]["ckpt_every"] == 500
    assert reference_parameter_count({key: model[key] for key in
                                      ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "intermediate_size")}) \
        == 98_977_016
    assert parameter_count(model) == 99_369_896
    assert reference_parameter_count() == 198_440_146


def test_a_budget_at_the_smaller_shape_holds_its_batch():
    derived, _ = derive(recipe("r816ab0fbe119"), layers=12, width=640, budget=2.5)
    assert derived["data"]["batch_size"] == 128
    assert (derived["optim"]["total_steps"], derived["optim"]["warmup_steps"]) == (18878, 1135)


def test_a_new_peak_learning_rate_keeps_the_tenfold_floor():
    derived, _ = derive(recipe("r816ab0fbe119"), layers=12, width=640, peak_learning_rate=2e-3)
    assert (derived["optim"]["learning_rate"], derived["optim"]["min_learning_rate"]) == (2e-3, 2e-4)


@pytest.mark.parametrize("changes,message", [
    ({"width": 650}, "multiple of the value head width"),
    ({"batch": 96}, "does not divide the 4864 validation sequences"),
    ({"batch": 120}, "not a multiple of micro_batch_size"),
    ({"budget": 0.0}, "must be positive"),
])
def test_inconsistent_requests_fail(changes, message):
    with pytest.raises(ValueError, match=message):
        derive(recipe("r816ab0fbe119"), **{"layers": 12, "width": 640, **changes})


def test_the_cli_writes_a_recipe_that_loads(tmp_path, capsys):
    from scaffolds.transformer.configs import load_config

    output = tmp_path / "config.yaml"
    assert main(["--partner", "r183f08346ddc", "--layers", "12", "--width", "640", "--output", str(output)]) == 0
    config = load_config(output)
    assert (config.model.num_layers, config.model.hidden_size, config.data.batch_size) == (12, 640, 128)
    assert "parameters: 99,061,736" in capsys.readouterr().out
