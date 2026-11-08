import pytest
import yaml

from experiments.transformer_pp.derive_recipe import (
    RUNS, _changes, _defined, budget_of, derive, main, parameter_count, reference_parameter_count,
)

# Run identity and output locations, set by `scaffolds.records create`, not by the recipe rules.
IDENTITY = {"train.workdir", "train.checkpoint_dir", "train.wandb_run_name", "continuation"}
PARTNERS = ("r9040f12f3d80", "r6a76e7fbde45", "r816ab0fbe119", "r4339a245820e", "rca97cc9db8f2",
            "r4a80e28ce0a7", "r183f08346ddc")
CONTEXT_4K = ("r2047d29a0d9f", "r339a31ad8d4b", "r58efc973ff28", "r636f433be44f", "r738b35a8e314")


def recipe(run):
    return yaml.safe_load((RUNS / run / "config.yaml").read_text())


@pytest.mark.parametrize("run", PARTNERS + CONTEXT_4K)
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
    assert reference_parameter_count(12, 640, 1664) == 98_977_016
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
    ({"batch": 120}, "not a positive multiple of micro_batch_size"),
    ({"batch": -16}, "not a positive multiple of micro_batch_size"),
    ({"layers": 18, "width": 1024}, "does not divide the 4864 validation sequences"),
    ({"peak_learning_rate": 0.0}, "must be positive"),
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


def test_rules_count_tokens_so_a_4k_partner_keeps_its_tokens_per_update():
    derived, _ = derive(recipe("r58efc973ff28"), layers=12, width=640)
    assert derived["data"]["sequence_length"] == 4097 and derived["data"]["batch_size"] == 64
    assert (derived["optim"]["total_steps"], derived["optim"]["warmup_steps"]) == (7551, 454)
    # The 4k runs' 2,432 validation sequences, the reference's 9,961,472 tokens.
    assert derived["train"]["eval_batches"] * derived["data"]["batch_size"] == 19 * 128


def test_the_budget_defaults_to_the_partner_s():
    assert budget_of(recipe("r816ab0fbe119")) == 1.0 and budget_of(recipe("r4f1682c1b54d")) == 2.5
    derived, _ = derive(recipe("r4f1682c1b54d"), layers=12, width=640)
    assert derived["data"]["batch_size"] == 128 and derived["optim"]["total_steps"] == 18878


def test_a_budget_change_keeps_the_partner_s_batch():
    partner = recipe("r816ab0fbe119")
    partner["data"]["batch_size"], partner["optim"]["total_steps"] = 128, 15140
    derived, _ = derive(partner, budget=2.5)
    assert derived["data"]["batch_size"] == 128 and derived["train"]["eval_batches"] == 38
    assert derived["optim"]["total_steps"] == 37849  # 2.5 x 20 x 198,440,146 / (128 x 2,048)


def test_derived_values_carry_no_float_noise():
    derived, _ = derive(recipe("r816ab0fbe119"), peak_learning_rate=3e-4)
    assert derived["optim"]["min_learning_rate"] == 3e-5
    partner = recipe("rca97cc9db8f2")
    partner["model"]["attention_coefficient_projection_initial_std"] = 0.02
    assert derive(partner, budget=2.5)[0]["model"]["attention_coefficient_projection_initial_std"] == 0.02


@pytest.mark.parametrize("partner", ["r9c69dd7cfc8d", "r25fffc16c2f3"])
def test_a_coefficient_release_keeps_its_fraction_of_the_updates(partner, tmp_path):
    from scaffolds.transformer.configs import load_config

    by_order, every = recipe(partner), recipe(partner)
    by_order["optim"].update(coefficient_release_schedule="by_order", coefficient_release_step=455,
                             coefficient_release_end_step=1500)
    every["optim"].update(coefficient_release_schedule="all", coefficient_release_step=455,
                          coefficient_release_end_step=None)
    assert _changes(_defined(by_order)[0], derive(by_order)[0]) == {}
    keys = ("total_steps", "warmup_steps", "coefficient_release_step", "coefficient_release_end_step")
    for changes, expected in (({"layers": 12, "width": 640}, (7551, 454, 454, 1496)),
                              ({"budget": 2.5}, (18925, 1138, 1138, 3750))):
        derived, _ = derive(by_order, **changes)
        assert tuple(derived["optim"][key] for key in keys) == expected
        # The derived release still satisfies 0 < step < end < total_steps.
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(derived, sort_keys=False))
        assert load_config(tmp_path / "config.yaml").optim.coefficient_release_end_step == expected[3]
        derived, _ = derive(every, **changes)
        assert tuple(derived["optim"][key] for key in keys) == (*expected[:3], None)
