"""CPU checks for reference scheduling, loss alignment and safe resume helpers."""
from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from scaffolds.transformer.native_fla import train
from scaffolds.transformer.native_fla.config import fla_model_kwargs, parameter_count, reference_recipe


@pytest.mark.parametrize("completed_steps,expected", [
    (0, 2.9296875e-7),
    (1022, 0.00029970703125),
    (1023, 0.0003),
    (1024, 0.0003),
    (5888, 0.0002604594154601839),
    (10752, 0.000165),
    (15616, 0.0000695405845398161),
])
def test_learning_rate_matches_pinned_torchtitan_boundaries(completed_steps, expected):
    # The constructor applies LambdaLR(0) before update 1. A one-index shift
    # would alter both the first update and the two-update plateau at the peak.
    assert train.learning_rate(reference_recipe(total_steps=20480), completed_steps) == pytest.approx(expected, rel=1e-12)


def test_last_update_precedes_the_final_scheduler_tick():
    recipe = reference_recipe(total_steps=20480)
    last = train.learning_rate(recipe, 20479)
    assert 3e-5 < last < 3.000001e-5
    with pytest.raises(ValueError, match="horizon"):
        train.learning_rate(recipe, 20480)
    with pytest.raises(ValueError, match="horizon"):
        train.learning_rate(recipe, -1)


@pytest.mark.parametrize("completed_steps,expected", [
    (0, 3e-4 / 379),
    (377, 0.0002992084432717678),
    (378, 0.0003),
    (379, 0.0003),
    (380, 0.0002999999871167648),
    (7569, 0.00003000001288323515),
])
def test_one_chinchilla_learning_rate_boundaries(completed_steps, expected):
    recipe = reference_recipe(total_steps=7570)
    assert recipe["optim"]["warmup_steps"] == 379
    assert train.learning_rate(recipe, completed_steps) == pytest.approx(expected, rel=1e-12)


def test_one_chinchilla_schedule_uses_frozen_horizon_on_resume():
    recipe = reference_recipe(total_steps=7570)
    frozen_recipe = json.dumps(recipe, sort_keys=True)
    # A fresh process can query a checkpoint's completed step directly: neither
    # the number of calls nor the remaining allocation changes the cosine horizon.
    for completed_steps, expected in [(5000, 0.00010652060208669845),
                                      (3785, 0.00017616367923445307),
                                      (0, 3e-4 / 379),
                                      (5000, 0.00010652060208669845)]:
        assert train.learning_rate(recipe, completed_steps) == pytest.approx(expected, rel=1e-12)
    assert json.dumps(recipe, sort_keys=True) == frozen_recipe
    assert 3e-5 < train.learning_rate(recipe, 7569) < 3.000002e-5
    with pytest.raises(ValueError, match="horizon"):
        train.learning_rate(recipe, 7570)
    assert recipe["optim"]["total_steps"] * recipe["data"]["batch_size"] * (recipe["data"]["sequence_length"] - 1) == 3968860160


@pytest.mark.parametrize("total,expected", [
    (7570, set(range(100, 1501, 100)) | set(range(2000, 7501, 500)) | {7570}),
    (18925, set(range(250, 3751, 250)) | set(range(5000, 18751, 1250)) | {18925}),
])
def test_validation_uses_matched_early_late_and_terminal_steps(total, expected):
    config = reference_recipe(total_steps=total)
    assert {step for step in range(1, total + 1) if train.should_evaluate(config, step)} == expected


def test_two_and_half_chinchilla_learning_rate_retains_reference_settings():
    config = reference_recipe(total_steps=18925)
    assert config["optim"]["warmup_steps"] == 946
    assert train.learning_rate(config, 0) == pytest.approx(3e-4 / 946)
    assert train.learning_rate(config, 945) == pytest.approx(3e-4)
    assert train.learning_rate(config, 946) == pytest.approx(3e-4)
    assert 3e-5 < train.learning_rate(config, 18924) < 3.000002e-5
    assert config["optim"]["eps"] == 1e-15


def test_batch_loss_supervises_all_next_tokens_without_double_shift(monkeypatch):
    calls = {}

    def autocast(**kwargs):
        calls["autocast"] = kwargs
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", autocast)

    class TokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Distinct, nonsymmetric scores make a target shift observable.
            self.scores = nn.Parameter(torch.tensor([
                [3., -1., 0., 2., -2., 1.],
                [1., 2., -1., 0., 3., -2.],
                [0., 1., 3., -2., -1., 2.],
                [-1., 3., 2., 1., 0., -2.],
                [2., 0., -2., 3., 1., -1.],
                [-2., -1., 1., 0., 2., 3.],
            ]))

        def forward(self, **kwargs):
            calls["model"] = kwargs
            return SimpleNamespace(logits=self.scores[kwargs["input_ids"]])

    def criterion(logits, targets):
        calls["targets"] = targets.clone()
        calls["shape"] = logits.shape
        return nn.functional.cross_entropy(logits, targets)

    model = TokenModel()
    tokens = torch.tensor([[0, 1, 2, 3, 4], [1, 3, 4, 0, 5]])
    actual = train.batch_loss(model, criterion, tokens)
    expected = nn.functional.cross_entropy(model.scores[tokens[:, :-1]].reshape(-1, 6),
                                           tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(actual, expected)
    assert calls["autocast"] == {"device_type": "cuda", "dtype": torch.bfloat16}
    assert set(calls["model"]) == {"input_ids", "use_cache", "return_dict"}
    assert calls["model"]["use_cache"] is False
    assert calls["model"]["return_dict"] is True
    assert calls["model"]["input_ids"].is_contiguous()
    torch.testing.assert_close(calls["model"]["input_ids"], tokens[:, :-1])
    torch.testing.assert_close(calls["targets"], torch.tensor([1, 2, 3, 4, 3, 4, 0, 5]))
    assert calls["shape"] == (8, 6)
    actual.backward()
    assert model.scores.grad is not None
    assert torch.isfinite(model.scores.grad).all()
    assert torch.count_nonzero(model.scores.grad[0]) == 6


def test_optimizer_uses_native_settings_and_selective_groups(monkeypatch):
    import fla.models.gated_deltanet as upstream

    calls = {}

    class TinyModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.randn(2, 3))
            self.norm = nn.RMSNorm(3)
            self.decay = nn.Parameter(torch.zeros(1))
            self.decay._no_weight_decay = True

        def cuda(self):
            calls["cuda"] = True
            return self

    def config_class(**kwargs):
        calls["config"] = kwargs
        return SimpleNamespace(**kwargs)

    def optimizer_class(parameters, **kwargs):
        calls["groups"] = list(parameters)
        calls["optimizer"] = kwargs
        return SimpleNamespace(param_groups=calls["groups"])

    monkeypatch.setattr(upstream, "GatedDeltaNetConfig", config_class)
    monkeypatch.setattr(upstream, "GatedDeltaNetForCausalLM", TinyModel)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer_class)
    monkeypatch.setattr(train, "parameter_count", lambda config: 10)
    recipe = reference_recipe()
    model, _ = train.build_model_optimizer(recipe)
    assert calls["config"] == recipe["model"]
    assert calls["cuda"]
    decayed, undecayed = calls["groups"]
    assert {id(p) for p in decayed["params"]} == {id(model.weight)}
    assert {id(p) for p in undecayed["params"]} == {id(model.norm.weight), id(model.decay)}
    assert decayed["weight_decay"] == 0.1 and undecayed["weight_decay"] == 0.0
    assert all(p.dtype == torch.float32 for group in calls["groups"] for p in group["params"])
    assert calls["optimizer"] == {"lr": 3e-4, "betas": (0.9, 0.95), "eps": 1e-15,
                                  "weight_decay": 0.1, "fused": True}


def test_decay_policy_uses_roles_and_handles_shared_parameters():
    class Roles(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 3)
            self.alias = nn.Linear(3, 7, bias=False)
            self.alias.weight = self.embedding.weight
            self.norm = nn.LayerNorm((2, 3))
            self.projection = nn.Linear(3, 3)
            self.conv = nn.Conv1d(3, 3, 3)
            self.scalar = nn.Parameter(torch.tensor(2.))
            self.special = nn.Parameter(torch.ones(2, 2))
            self.special._no_weight_decay = True
            self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)

    model = Roles()
    groups = train.optimizer_parameter_groups(model, 0.1)
    decayed = {id(p) for p in groups[0]["params"]}
    assert decayed == {id(model.projection.weight), id(model.conv.weight), id(model.scalar)}
    summary = train.optimizer_policy_summary(model, SimpleNamespace(param_groups=groups))
    assert summary["parameter_tensors"] == len([p for p in model.parameters() if p.requires_grad])
    assert summary["parameter_count"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert {name for g in summary["groups"] for name in g["parameter_names"]} == {
        name for name, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("problem", ["duplicate", "missing", "unknown"])
def test_policy_audit_rejects_incomplete_or_duplicate_optimizer_groups(problem):
    model = nn.Linear(3, 2)
    groups = train.optimizer_parameter_groups(model, 0.1)
    if problem == "duplicate":
        groups[1]["params"].append(model.weight)
    elif problem == "missing":
        groups[0]["params"].clear()
    else:
        groups[0]["params"].append(nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="optimizer|Optimizer"):
        train.optimizer_policy_summary(model, SimpleNamespace(param_groups=groups))


def test_full_native_model_decay_groups_cover_all_parameters_and_native_exemptions():
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    config = reference_recipe()
    with torch.device("meta"):
        model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**fla_model_kwargs(config)))
    groups = train.optimizer_parameter_groups(model, 0.1)
    summary = train.optimizer_policy_summary(model, SimpleNamespace(param_groups=groups))
    assert summary["parameter_count"] == parameter_count(config) == 200_205_878
    decayed = {id(p) for p in groups[0]["params"]}
    assert id(model.model.embeddings.weight) not in decayed
    assert id(model.model.norm.weight) not in decayed
    assert id(model.lm_head.weight) in decayed
    for layer in model.model.layers:
        assert id(layer.attn_norm.weight) not in decayed
        assert id(layer.mlp_norm.weight) not in decayed
        attention = layer.attn
        for parameter in (attention.A_log, attention.dt_bias, attention.o_norm.weight):
            assert id(parameter) not in decayed
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.g_proj,
                           attention.a_proj, attention.b_proj, attention.o_proj,
                           layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            assert id(projection.weight) in decayed
        for convolution in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
            assert id(convolution.weight) in decayed


def test_native_cpu_adamw_update_decays_only_selected_weights():
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    kwargs = fla_model_kwargs(reference_recipe())
    kwargs.update(hidden_size=128, num_hidden_layers=2, num_heads=3, head_dim=32,
                  intermediate_size=512, vocab_size=1024)
    torch.manual_seed(42)
    model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**kwargs))
    groups = train.optimizer_parameter_groups(model, 0.1)
    optimizer = torch.optim.AdamW(groups, lr=0.01, betas=(0.9, 0.95), eps=1e-15, foreach=False)
    train.optimizer_policy_summary(model, optimizer)
    before = {id(p): p.detach().clone() for p in model.parameters()}
    decayed = {id(p) for p in groups[0]["params"]}
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for parameter in model.parameters():
        original = before[id(parameter)]
        expected = original * (1 - 0.01 * 0.1) if id(parameter) in decayed else original
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        if id(parameter) in decayed:
            assert not torch.equal(parameter, original)
    for layer in model.model.layers:
        for parameter in (layer.attn.A_log, layer.attn.dt_bias):
            assert torch.count_nonzero(before[id(parameter)]) > 0
            assert torch.equal(parameter, before[id(parameter)])


def test_rewind_preserves_both_train_and_validation_at_checkpoint(tmp_path):
    path = tmp_path / "metrics.jsonl"
    kept = [{"step": 1, "train_loss": 5.0}, {"step": 4, "train_loss": 4.0},
            {"step": 4, "validation_nll": 4.2}]
    rows = kept + [{"step": 5, "train_loss": 3.9}, {"step": 8, "validation_nll": 4.1}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"step": 9,')
    train.rewind_metrics(path, completed_step=4)
    assert [json.loads(line) for line in path.read_text().splitlines()] == kept
    original = path.read_bytes()
    train.rewind_metrics(path, completed_step=4)
    assert path.read_bytes() == original


def test_rewind_does_not_hide_corruption_before_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "train_loss": 5.0}\ninvalid\n{"step": 2}\n')
    original = path.read_bytes()
    with pytest.raises(json.JSONDecodeError):
        train.rewind_metrics(path, completed_step=1)
    assert path.read_bytes() == original


def test_rewind_missing_metrics_is_a_noop(tmp_path):
    path = tmp_path / "metrics.jsonl"
    train.rewind_metrics(path, completed_step=0)
    assert not path.exists()


@pytest.mark.parametrize("failure_stage", ["prune", "capture"])
def test_completion_can_reconcile_interrupted_closeout(tmp_path, monkeypatch, failure_stage):
    checkpoints = tmp_path / "checkpoints"
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    for step in (1, 3):
        train.save_checkpoint(checkpoints / str(step), model=model, optimizer=optimizer,
                              step=step, data_cursor=step * 256,
                              best_validation={"step": 1, "validation_nll": 2.0},
                              config_sha256="test-recipe")
    summary = {"status": "complete", "final_step": 3, "best_validation_step": 1,
               "validation_nll": 2.1, "validation_perplexity": 8.166169912567652}
    real_retain = train.retain_checkpoints
    first_attempt = True
    records = []

    def retain(*args, **kwargs):
        # Terminal evidence must be durable before deleting the final checkpoint.
        assert json.loads((tmp_path / "completion.json").read_text()) == summary
        if first_attempt and failure_stage == "prune":
            raise OSError("interrupted closeout")
        return real_retain(*args, **kwargs)

    def capture(output, *, checkpoint_directory):
        assert checkpoint_directory == checkpoints / "1"
        if first_attempt and failure_stage == "capture":
            raise OSError("interrupted closeout")
        return {"selected_checkpoint": str(checkpoint_directory)}

    monkeypatch.setattr(train, "retain_checkpoints", retain)
    monkeypatch.setattr(train, "capture_artifacts", capture)
    monkeypatch.setattr(train, "update_record", lambda output, record: records.append(record))
    with pytest.raises(OSError, match="interrupted closeout"):
        train.finalize_completion(tmp_path, checkpoints, summary)
    assert json.loads((tmp_path / "completion.json").read_text()) == summary
    assert sorted(path.name for path in checkpoints.iterdir()) == (["1", "3"] if failure_stage == "prune" else ["1"])
    assert not records
    first_attempt = False
    train.finalize_completion(tmp_path, checkpoints, summary)
    assert sorted(path.name for path in checkpoints.iterdir()) == ["1"]
    assert records[-1]["status"] == "complete"
    assert records[-1]["terminal_step"] == 3
    assert records[-1]["selected_checkpoint"] == str(checkpoints / "1")
    train.finalize_completion(tmp_path, checkpoints, summary)
    assert records[-1] == records[-2]


def test_terminal_checkpoint_finishes_without_repeating_an_update(tmp_path, monkeypatch):
    recipe = reference_recipe()
    recipe["data"].update(train_manifest="train-manifest", eval_manifest="eval-manifest")
    recipe["train"].update(workdir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints"),
                           wandb_mode="disabled")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    step = recipe["optim"]["total_steps"]
    cursor = step * recipe["data"]["batch_size"]
    validation = {"step": step, "validation_nll": 2.1, "validation_perplexity": 8.166169912567652}
    last_train = {"train_loss": 2.2, "train_step_seconds": 1.0}
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(to_dict=lambda: {"test_model": True})
    optimizer = torch.optim.AdamW(model.parameters())
    train.save_checkpoint(tmp_path / "checkpoints" / str(step), model=model, optimizer=optimizer,
                          step=step, data_cursor=cursor, best_validation=validation,
                          config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                          extra={"last_train": last_train, "validation": validation})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(train, "runtime_metadata", lambda config: {"source_commit": "test-commit"})
    monkeypatch.setattr(train, "build_model_optimizer", lambda config: (model, optimizer))
    monkeypatch.setattr(train, "load_packed_manifest", lambda path: (path, {"num_sequences": cursor}))
    monkeypatch.setattr(train, "manifest_fingerprint", lambda path: "test-data-sha")
    monkeypatch.setattr(train, "PackedBatchIterator", lambda path, **kwargs: SimpleNamespace(cursor=kwargs["start_sequence"]))

    def unexpected_update(*args, **kwargs):
        pytest.fail("A terminal checkpoint must finalize without training or evaluation")

    monkeypatch.setattr(train, "train_update", unexpected_update)
    monkeypatch.setattr(train, "evaluate", unexpected_update)
    summary = train.run_training(config_path, qualification=True)
    assert summary["final_step"] == step
    assert summary["data_cursor"] == cursor
    assert summary["last_train"] == last_train
    assert summary["validation_nll"] == validation["validation_nll"]
    assert summary["train_step_seconds"] == []
    assert json.loads((tmp_path / "qualification.json").read_text()) == summary
