from copy import deepcopy

from experiments.transformer_200m_4b.launch_continuation import recipes, submission_command


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
        assert recipe["optim"]["total_steps"] - recipe["continuation"]["parent_step"] == 500
        assert recipe["optim"]["learning_rate"] == recipe["optim"]["min_learning_rate"] == 1e-4
        assert recipe["optim"]["warmup_steps"] == 0
        assert recipe["optim"]["beta1"] == 0.9 and recipe["optim"]["beta2"] == 0.95
    learned = deepcopy(pair["learned_static"])
    learned["model"].update(attention_coefficients="fixed", attention_coefficient_sharing="global",
                            attention_coefficient_scale="none")
    learned["train"]["wandb_run_name"] = pair["fixed_taylor"]["train"]["wandb_run_name"]
    assert learned == pair["fixed_taylor"]


def test_independent_single_gpu_submission_uses_shared_acceptance():
    command = submission_command("r0123456789ab", "a" * 40,
                                 partition="standard,long", time_limit="02:00:00")
    assert "--gres=gpu:1" in command
    assert "--partition=standard,long" in command
    assert not any(arg.startswith(("--dependency", "--nice")) for arg in command)
    assert command[-5:] == ["experiments.transformer_200m_4b.acceptance",
                           "experiments/transformer_200m_4b/runs/r0123456789ab/config.yaml",
                           "a" * 40, "1", "B300,H100"]
