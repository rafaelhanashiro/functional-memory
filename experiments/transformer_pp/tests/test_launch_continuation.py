from copy import deepcopy
import subprocess

import pytest

from experiments.transformer_pp import launch_continuation
from experiments.transformer_pp.launch_continuation import recipes, submission_command


@pytest.mark.parametrize("conditioning,decay,role", [
    ("static", False, "static_self"), ("query", True, "query_decay_self"),
])
def test_preserved_recipe_keeps_learned_model_and_uses_constant_rate(conditioning, decay, role):
    source = {"model": {"attention_degree": 4, "attention_coefficients": "learned",
                        "attention_coefficient_conditioning": conditioning,
                        "attention_coefficient_kernel_weight_decay": decay},
              "data": {"batch_size": 256}, "train": {}, "optim": {"weight_decay": .1}}
    result = recipes(source, {"parent_step": 7500}, conditioning=conditioning, preserve_all_parameters=True)
    assert set(result) == {role}
    child = result[role]
    assert child["model"] == source["model"]
    assert child["data"] == source["data"]
    assert child["optim"]["learning_rate"] == child["optim"]["min_learning_rate"] == 1e-4
    assert child["optim"]["warmup_steps"] == 0 and child["optim"]["total_steps"] == 9500
    assert child["continuation"]["preserve_all_parameters"] is True
    assert child["continuation"]["new_parameter_learning_rate"] is None


def test_matched_pair_preserves_parent_science_except_new_coefficients():
    source = {
        "model": {"attention_degree": 8, "attention_coefficients": "fixed",
                  "attention_coefficient_sharing": "global", "attention_coefficient_scale": "none"},
        "optim": {"learning_rate": 0.001, "min_learning_rate": 0.0001, "warmup_steps": 455,
                  "total_steps": 7570, "beta1": 0.9, "beta2": 0.95, "weight_decay": 0.1},
        "data": {"batch_size": 256, "shuffle_seed": 0},
        "train": {"seed": 42, "micro_batch_size": 16, "eval_batches": 19},
    }
    original = deepcopy(source)
    parent = {"parent_step": 7500, "parent_checkpoint_sha256": "a" * 64}
    pair = recipes(source, parent)
    assert source == original
    for recipe in pair.values():
        assert recipe["data"] == source["data"]
        assert recipe["train"]["seed"] == 42
        assert recipe["train"]["eval_batches"] == 19
        assert recipe["train"]["micro_batch_size"] == 8
        assert recipe["train"]["eval_every"] == recipe["train"]["ckpt_every"] == 100
        assert recipe["optim"]["total_steps"] - recipe["continuation"]["parent_step"] == 2000
        assert recipe["optim"]["learning_rate"] == recipe["optim"]["min_learning_rate"] == 1e-4
        assert recipe["optim"]["warmup_steps"] == 0
        assert recipe["optim"]["beta1"] == 0.9 and recipe["optim"]["beta2"] == 0.95
    learned = deepcopy(pair["learned_static"])
    learned["model"].update(attention_coefficients="fixed", attention_coefficient_sharing="global",
                            attention_coefficient_scale="none")
    learned["train"]["wandb_run_name"] = pair["fixed_taylor"]["train"]["wandb_run_name"]
    assert learned["continuation"].pop("new_parameter_learning_rate") == 0.001
    assert learned == pair["fixed_taylor"]


def test_independent_single_gpu_submission_uses_shared_acceptance():
    command = submission_command("r0123456789ab", "a" * 40,
                                 partition="standard,long", time_limit="06:00:00")
    assert "--gres=gpu:1" in command
    assert "--partition=standard,long" in command
    assert not any(arg.startswith(("--dependency", "--nice")) for arg in command)
    assert command[-5:] == ["experiments.transformer_pp.acceptance",
                           "experiments/transformer_pp/runs/r0123456789ab/config.yaml",
                           "a" * 40, "1", "B300,H100"]


def test_query_pair_matches_canonical_m4_initialization_and_training():
    source = {
        "model": {"attention_degree": 4, "hidden_size": 896,
                  "attention_coefficients": "fixed", "qk_norm": True,
                  "attention_coefficient_sharing": "global",
                  "attention_coefficient_conditioning": "static",
                  "attention_coefficient_scale": "none"},
        "optim": {"learning_rate": 0.001, "min_learning_rate": 0.0001,
                  "warmup_steps": 455, "total_steps": 7570},
        "data": {"batch_size": 256, "shuffle_seed": 0},
        "train": {"seed": 42, "eval_batches": 19},
    }
    original = deepcopy(source)
    pair = recipes(source, {"parent_step": 7500}, conditioning="query")
    assert source == original
    assert set(pair) == {"fixed_taylor", "learned_query"}
    control, query = pair["fixed_taylor"], pair["learned_query"]
    assert control["model"] == {**source["model"], "qk_norm_token_scale": "none",
                                "qk_norm_token_scale_kernel_weight_decay": False,
                                "attention_forget_gate": False}
    assert query["model"]["attention_degree"] == 4
    assert query["model"]["attention_coefficient_conditioning"] == "query"
    assert query["model"]["attention_coefficient_activation"] == "tanh"
    assert query["model"]["attention_coefficient_amplitude_initial"] == 0.0
    assert query["model"]["attention_coefficient_projection_initial_std"] == 1 / 896**0.5
    assert query["model"]["attention_coefficient_tanh_bias"] is False
    assert query["model"]["attention_coefficient_scale"] == "factorial"
    assert query["model"]["attention_coefficient_amplitude_scale"] == "factorial"
    assert control["data"] == query["data"] == source["data"]
    assert control["optim"] == query["optim"]
    assert query["continuation"] == {**control["continuation"], "new_parameter_learning_rate": 0.001}
    assert query["optim"]["total_steps"] == 9500
    assert query["train"]["wandb_run_name"] == "taylor-m4-continuation-learned_query"
    assert launch_continuation.topic_path("query", "learned_coefficient_continuation") != launch_continuation.topic_path("static")


@pytest.mark.parametrize("conditioning,options,role", [
    ("forget_gate", {"kernel_weight_decay": True, "no_rope": True}, "query_forget_gate_no_rope_decay"),
    ("token_scale", {"token_scale_kernel_weight_decay": True}, "token_query_scale_decay"),
])
def test_decay_variants_preserve_the_matched_control_and_training(conditioning, options, role):
    source = {
        "model": {"attention_degree": 4, "hidden_size": 896, "positional_encoding": "rope",
                  "attention_coefficients": "fixed", "attention_backend": "mosaic",
                  "qk_norm": True, "qk_norm_head_scale": True},
        "optim": {"learning_rate": 0.001, "min_learning_rate": 0.0001, "weight_decay": 0.1},
        "data": {"batch_size": 256, "shuffle_seed": 0},
        "train": {"seed": 42, "eval_batches": 19},
    }
    parent = {"parent_step": 7500, "parent_checkpoint_sha256": "a" * 64}
    original = deepcopy(source)
    control = recipes(source, parent, conditioning=conditioning)["fixed_taylor"]
    pair = recipes(source, parent, conditioning=conditioning, **options)
    assert source == original
    assert pair["fixed_taylor"] == control
    branch = pair[role]
    assert branch["data"] == control["data"]
    assert branch["optim"] == control["optim"]
    assert branch["continuation"] == {**parent, "new_parameter_learning_rate": 0.001}
    assert branch["model"]["qk_norm_head_scale"] is True
    if conditioning == "forget_gate":
        assert branch["model"]["positional_encoding"] == "none"
        assert branch["model"]["attention_forget_gate"] is True
        assert branch["model"]["attention_coefficient_kernel_weight_decay"] is True
        assert branch["model"]["attention_coefficient_scale"] == "factorial"
        assert branch["model"]["attention_coefficient_amplitude_scale"] == "factorial"
    else:
        assert branch["model"]["positional_encoding"] == "rope"
        assert branch["model"]["qk_norm_token_scale"] == "cosine"
        assert branch["model"]["qk_norm_token_scale_kernel_weight_decay"] is True
        assert branch["model"]["attention_coefficients"] == "fixed"


@pytest.mark.parametrize("options,match", [
    ({"no_rope": True}, "Removing RoPE"),
    ({"token_scale_kernel_weight_decay": True}, "Token-scale kernel"),
    ({"kernel_weight_decay": True}, "query coefficients"),
])
def test_variant_flags_reject_unrelated_mechanisms(options, match):
    with pytest.raises(ValueError, match=match):
        recipes({"model": {"attention_degree": 4}}, {"parent_step": 7500}, **options)


def test_default_allocation_leaves_time_after_acceptance_reserve(monkeypatch):
    from scaffolds.transformer.acceptance import require_allocation_budget

    captured = {}
    monkeypatch.setattr(launch_continuation, "submit", lambda run_id, **kwargs: captured.update(kwargs))
    launch_continuation.main(["submit", "r0123456789ab"])
    assert captured["time_limit"] == launch_continuation.DEFAULT_TIME_LIMIT
    hours, minutes, seconds = map(int, captured["time_limit"].split(":"))
    allocation = 3600 * hours + 60 * minutes + seconds
    require_allocation_budget(3600, allocation)


@pytest.mark.parametrize("returncode,stdout,stderr,expected", [
    (0, "RUNNING\n", "", "RUNNING"),
    (0, "", "", ""),
    (1, "", "slurm_load_jobs error: Invalid job id specified", ""),
])
def test_live_job_lookup_accepts_expired_ids(monkeypatch, returncode, stdout, stderr, expected):
    monkeypatch.setattr(launch_continuation.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr=stderr))
    assert launch_continuation.live_job_state("224999") == expected


def test_live_job_lookup_preserves_scheduler_errors(monkeypatch):
    monkeypatch.setattr(launch_continuation.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1, stdout="", stderr="scheduler unavailable"))
    with pytest.raises(RuntimeError, match="scheduler unavailable"):
        launch_continuation.live_job_state("224999")
