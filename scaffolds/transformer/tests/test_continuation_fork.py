"""A fork: a completed parent's terminal weights, a fresh AdamW and schedule, and the parent's next batches.

The parent is a tiny unnormalized exp model trained to completion by the trainer itself on a warmup-and-cosine
schedule. Each child replays that schedule from update zero on the parent's following packed sequences: an
identical exp child (every parameter) and a Taylor-initialized query-polynomial child (the backbone plus new
coefficients).
"""

import copy
from dataclasses import replace
import hashlib
import json
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.records import _tree_descriptor, load_run
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import ContinuationConfig, config_to_dict, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.retention import finalize_run
from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter
from scaffolds.transformer.tests.test_continuation import assert_equal_tree
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import continuation as continuation_module
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import restore_state, state_sha256
from scaffolds.transformer.training.continuation import (
    initialize_continuation_state, packed_batch_sha256,
)
from scaffolds.transformer.training.optim import ContinuationAdamState, split_continuation_parameters
from scaffolds.transformer.training.state import create_state

PARENT_STEPS = 4
BATCH = 4
OFFSET = PARENT_STEPS * BATCH


def _rows(manifest, start, count=BATCH):
    return next(PackedBatchIterator(manifest, batch_size=count, sequence_length=5, start_sequence=start))


@pytest.fixture
def parent_run(tmp_path):
    """A completed exp parent and its terminal checkpoint, trained by the trainer on a warmup-cosine schedule."""
    parent = tiny_config(tmp_path / "parent")
    parent.model = replace(parent.model, attention_degree=None, attention_normalization=False,
                           attention_backend="blocked", attention_output_scaling="head_rms_norm",
                           attention_exp_shift="row_max", attention_head_rms_prescale="max_abs",
                           qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
                           qk_norm_head_scale=True)
    parent.optim.learning_rate, parent.optim.min_learning_rate = 1e-2, 1e-3
    parent.optim.warmup_steps, parent.optim.total_steps = 1, PARENT_STEPS
    parent.optim.grad_clip_norm, parent.optim.weight_decay = 1.0, 0.1
    parent.data.source = "packed"
    for split, count in (("train", 64), ("eval", 4)):
        writer = PackedSequenceWriter(tmp_path / split, sequence_length=5, num_sequences=count, shard_sequences=8)
        rng = np.random.default_rng(1 if split == "train" else 2)
        writer.write(rng.integers(0, 16, size=(count, 5)))
        setattr(parent.data, f"{split}_manifest", str(writer.finish()))
    loop.run_training(parent)
    checkpoint = parent.train.workdir / "checkpoints" / str(PARENT_STEPS)
    assert checkpoint.is_dir()
    return parent, checkpoint


def fork_config(parent_run, path, *, query, total_steps=PARENT_STEPS):
    parent, checkpoint = parent_run
    workdir = parent.train.workdir
    config = copy.deepcopy(parent)
    config.train.workdir, config.train.checkpoint_dir = path, None
    config.train.retain_terminal_checkpoint = True
    config.optim.total_steps = total_steps
    config.continuation = ContinuationConfig(
        parent_config=str(workdir / "config.yaml"),
        parent_config_sha256=hashlib.sha256((workdir / "config.yaml").read_bytes()).hexdigest(),
        parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=_tree_descriptor(checkpoint)["sha256"],
        parent_data_identity_sha256=hashlib.sha256((workdir / "data_identity.json").read_bytes()).hexdigest(),
        parent_step=PARENT_STEPS, preserve_all_parameters=not query, optimizer="fresh", data_start_sequence=OFFSET)
    if query:
        config.model = replace(
            config.model, attention_degree=2, attention_exp_shift="none", attention_coefficients="learned",
            attention_coefficient_sharing="per_head", attention_coefficient_scale="factorial",
            attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
            attention_coefficient_amplitude_scale="factorial", attention_coefficient_amplitude_initial=0.0,
            attention_coefficient_projection_initial_std=1 / math.sqrt(config.model.hidden_size),
            attention_coefficient_kernel_weight_decay=True)
    validate_config(config)
    return config


def _parent_terminal_state(parent_run):
    parent, checkpoint = parent_run
    return restore_state(checkpoint, create_state(parent, jax.random.PRNGKey(parent.train.seed)))


def _metrics(config):
    return [json.loads(line) for line in (config.train.workdir / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("query", (False, True))
def test_fork_takes_the_parents_weights_with_a_fresh_adamw_at_update_zero(parent_run, tmp_path, query):
    config = fork_config(parent_run, tmp_path / "child", query=query)
    parent = _parent_terminal_state(parent_run)
    assert int(parent.step) == PARENT_STEPS and int(parent.opt_state[1][0].count) == PARENT_STEPS
    state, audit = initialize_continuation_state(config)
    assert int(state.step) == 0
    # One plain AdamW, exactly as a new run of the child's recipe initializes it: zero moments and counts.
    assert not isinstance(state.opt_state[1], ContinuationAdamState)
    assert state_sha256(state.opt_state) == state_sha256(create_state(config, jax.random.PRNGKey(0)).opt_state)
    assert audit["optimizer_sha256"] == audit["fresh_optimizer_sha256"] == state_sha256(state.tx.init(state.params))
    assert audit["optimizer_counts"] == [0, 0] and audit["optimizer"] == "fresh" and audit["start_step"] == 0
    assert audit["data_start_sequence"] == OFFSET and "parent_optimizer_sha256" not in audit
    if query:
        introduced, backbone = split_continuation_parameters(state.params)
        assert_equal_tree(backbone, parent.params)
        np.testing.assert_array_equal(np.asarray(introduced["query_polynomial_bias"][...]), 1.0)
        np.testing.assert_array_equal(np.asarray(introduced["query_polynomial_amplitude"][...]), 0.0)
        # The new coefficients are the recipe's own initial draw, the arrays a new run would start from.
        fresh, _ = split_continuation_parameters(create_state(config, jax.random.PRNGKey(config.train.seed)).params)
        assert_equal_tree(introduced, fresh)
    else:
        assert_equal_tree(state.params, parent.params)
    assert audit["parent_parameters_sha256"] == audit["backbone_parameters_sha256"]


@pytest.mark.parametrize("query", (False, True))
def test_fork_trains_from_the_parents_next_batch_on_the_restarted_schedule(parent_run, tmp_path, query):
    parent_config, _ = parent_run
    config = fork_config(parent_run, tmp_path / "child", query=query)
    result = loop.run_training(config, stop_after_step=2, audit_checkpoint=True)
    assert result["start_step"] == 0 and result["data_start_sequence"] == OFFSET
    manifest = parent_config.data.train_manifest
    assert result["fork_batches"] == {
        "parent_last_batch": {"sequence_start": OFFSET - BATCH, "sequence_stop": OFFSET,
                              "sha256": packed_batch_sha256(_rows(manifest, OFFSET - BATCH))},
        "first_batch": {"sequence_start": OFFSET, "sequence_stop": OFFSET + BATCH,
                        "sha256": packed_batch_sha256(_rows(manifest, OFFSET))}}
    # Update k reads sequences OFFSET + (k - 1) x batch: the parent's update PARENT_STEPS + k, micro-batched alike.
    first = np.asarray(_rows(manifest, OFFSET), np.int32).reshape(2, 2, 5)
    assert result["checkpoint_audit"]["first_update"]["batch_sha256"] == state_sha256(jnp.asarray(first))
    # The same two updates by hand, from the forked state: one fresh AdamW whose count is the update clock.
    reference, _ = initialize_continuation_state(fork_config(parent_run, tmp_path / "reference", query=query))
    rng = jax.random.PRNGKey(config.train.seed)
    for step in (1, 2):
        batches = jnp.asarray(np.asarray(_rows(manifest, OFFSET + (step - 1) * BATCH), np.int32).reshape(2, 2, 5))
        reference, *_ = loop.accumulated_train_step(reference, batches,
                                                    jax.random.split(jax.random.fold_in(rng, step), 2))
    trained = restore_state(config.train.workdir / "checkpoints" / "2", create_state(config, rng))
    assert int(trained.step) == 2 and [int(trained.opt_state[1][0].count), int(trained.opt_state[1][2].count)] == [2, 2]
    for a, b in zip(jax.tree.leaves(trained.params), jax.tree.leaves(reference.params), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    # The parent's schedule restarts: the child's update k has the parent's update-k learning rate.
    parent_rates = {row["step"]: row["learning_rate"] for row in _metrics(parent_config)}
    child_rates = {row["step"]: row["learning_rate"] for row in _metrics(config)}
    assert child_rates[2] == parent_rates[2]
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["next_train_sequence"] == OFFSET + 2 * BATCH


def test_both_branches_record_the_same_first_batch_after_the_parents_last(parent_run, tmp_path, capsys):
    parent_config, _ = parent_run
    audits = []
    for query in (False, True):
        config = fork_config(parent_run, tmp_path / f"child-{query}", query=query)
        loop.run_training(config, stop_after_step=1)
        audits.append(json.loads((config.train.workdir / "continuation.json").read_text()))
    exp, polynomial = audits
    assert exp["first_batch"] == polynomial["first_batch"] and exp["parent_last_batch"] == polynomial["parent_last_batch"]
    manifest = parent_config.data.train_manifest
    assert exp["first_batch"] == {"sequence_start": OFFSET, "sequence_stop": OFFSET + BATCH,
                                  "sha256": packed_batch_sha256(_rows(manifest, OFFSET))}
    # The parent's own last update read the batch just before: its cursor ends where the fork starts.
    assert exp["parent_last_batch"] == {"sequence_start": OFFSET - BATCH, "sequence_stop": OFFSET,
                                        "sha256": packed_batch_sha256(_rows(manifest, OFFSET - BATCH))}
    assert json.loads((parent_config.train.workdir / "metadata.json").read_text())["next_train_sequence"] == OFFSET
    assert capsys.readouterr().out.count(f"fork_first_batch_verified sequences={OFFSET}-{OFFSET + BATCH - 1}") == 2


def test_a_fork_refuses_any_first_batch_but_its_recorded_one(parent_run, tmp_path, monkeypatch):
    config = fork_config(parent_run, tmp_path / "child", query=False)
    original = continuation_module.fork_batch_provenance

    def shifted(current, **options):
        provenance = original(current, **options)
        provenance["first_batch"]["sha256"] = provenance["parent_last_batch"]["sha256"]
        return provenance

    monkeypatch.setattr(continuation_module, "fork_batch_provenance", shifted)
    with pytest.raises(RuntimeError, match="recorded first batch"):
        loop.run_training(config, stop_after_step=1)


def test_exp_fork_baseline_reproduces_the_parents_terminal_validation(parent_run, tmp_path):
    parent_config, _ = parent_run
    config = fork_config(parent_run, tmp_path / "child", query=False)
    loop.run_training(config, stop_after_step=1)
    baseline = _metrics(config)[0]
    assert baseline["step"] == 0 and baseline["continuation_updates"] == 0 and baseline["training_tokens"] == 0
    terminal = next(row for row in _metrics(parent_config) if row["step"] == PARENT_STEPS)
    # The same weights on the same validation prefix: the switch cost of the identical branch is zero.
    assert baseline["validation_nll"] == terminal["validation_nll"]


def test_interrupted_fork_resumes_its_own_optimizer_schedule_and_cursor(parent_run, tmp_path):
    full = fork_config(parent_run, tmp_path / "full", query=True)
    resumed = fork_config(parent_run, tmp_path / "resumed", query=True)
    full_result = loop.run_training(full, audit_checkpoint=True)
    paused = loop.run_training(resumed, stop_after_step=2, audit_checkpoint=True)
    resumed_result = loop.run_training(resumed, audit_checkpoint=True)
    assert paused["start_step"] == 0 and paused["data_start_sequence"] == OFFSET
    assert resumed_result["restored_step"] == 2 and resumed_result["data_start_sequence"] == OFFSET + 2 * BATCH
    acceptance.validate_resume_audit(paused, resumed_result)
    assert (full_result["checkpoint_audit"]["final_state_sha256"]
            == resumed_result["checkpoint_audit"]["final_state_sha256"])
    for config in (full, resumed):
        assert [row["step"] for row in _metrics(config)] == [0, 2, 4]
    assert [row["validation_nll"] for row in _metrics(full)] == [row["validation_nll"] for row in _metrics(resumed)]


def test_fork_retains_its_terminal_checkpoint_beside_the_best(parent_run, tmp_path):
    config = fork_config(parent_run, tmp_path / "child", query=True)
    config.train.source_commit = "a" * 40  # Retention verifies the executed commit.
    loop.run_training(config)
    record = finalize_run(config)
    best = json.loads((config.train.workdir / "best_validation.json").read_text())["step"]
    kept = sorted(path.name for path in (config.train.workdir / "checkpoints").iterdir())
    assert kept == sorted({str(best), str(PARENT_STEPS)})
    assert record["terminal_checkpoint_retained"] is True
    run = load_run(config.train.workdir)
    assert run["retained_terminal_checkpoint_step"] == PARENT_STEPS and run["retained_best_checkpoint_step"] == best
    terminal = run["terminal_checkpoint"]
    assert terminal["step"] == PARENT_STEPS and terminal["kind"] == "directory"
    assert terminal["sha256"] == _tree_descriptor(config.train.workdir / "checkpoints" / str(PARENT_STEPS))["sha256"]
    assert run["selected_checkpoint"]["step"] == best
    # A requeue after retention resumes as complete, from the kept terminal state.
    assert loop.run_training(config)["final_step"] == PARENT_STEPS
    metadata = json.loads((config.train.workdir / "metadata.json").read_text())
    assert metadata["retained_terminal_checkpoint_step"] == PARENT_STEPS


def test_without_the_option_retention_keeps_only_the_best(parent_run, tmp_path):
    config = fork_config(parent_run, tmp_path / "child", query=False)
    config.train.retain_terminal_checkpoint = False
    config.train.source_commit = "a" * 40
    loop.run_training(config)
    record = finalize_run(config)
    best = json.loads((config.train.workdir / "best_validation.json").read_text())["step"]
    assert sorted(path.name for path in (config.train.workdir / "checkpoints").iterdir()) == [str(best)]
    assert "terminal_checkpoint_retained" not in record and "terminal_checkpoint" not in load_run(config.train.workdir)


def test_fork_recipe_rules(parent_run, tmp_path):
    config = fork_config(parent_run, tmp_path / "child", query=False)
    for change, message in (
            ({"data_start_sequence": OFFSET - 1}, "data_start_sequence"),
            ({"data_start_sequence": None}, "data_start_sequence"),
            ({"new_parameter_learning_rate": 1e-3}, "new_parameter_learning_rate"),
            ({"optimizer": "adam"}, "parent or fresh"),
            ({"parent_step": 0, "data_start_sequence": 0}, "positive integer parent_step"),
            ({"optimizer": "parent"}, "fork \\(optimizer: fresh\\) only"),
            ({"parent_checkpoint": ""}, "Incomplete continuation specification")):
        broken = copy.deepcopy(config)
        broken.continuation = replace(broken.continuation, **change)
        with pytest.raises(ValueError, match=message):
            validate_config(broken)
    synthetic = copy.deepcopy(config)
    synthetic.data.source = "synthetic"
    with pytest.raises(ValueError, match="packed training stream"):
        validate_config(synthetic)
    beyond = copy.deepcopy(config)
    beyond.continuation = replace(beyond.continuation, parent_step=PARENT_STEPS + 1,
                                  data_start_sequence=(PARENT_STEPS + 1) * BATCH)
    with pytest.raises(ValueError, match="exceeds its training budget"):
        initialize_continuation_state(beyond)
    # An exp child is the identical architecture: it must say so and keep every parameter.
    exp_without_preserve = copy.deepcopy(config)
    exp_without_preserve.continuation = replace(exp_without_preserve.continuation, preserve_all_parameters=False)
    with pytest.raises(ValueError, match="polynomial degree"):
        initialize_continuation_state(exp_without_preserve)
    # The data budget counts the fork's offset.
    long = fork_config(parent_run, tmp_path / "long", query=False, total_steps=13)
    with pytest.raises(ValueError, match="sequences, but 68 are required"):
        loop.run_training(long, stop_after_step=1)


def test_legacy_configs_compare_equal_with_the_fork_defaults(tmp_path):
    current = config_to_dict(tiny_config(tmp_path))
    legacy = copy.deepcopy(current)
    for section, key in (("continuation", "optimizer"), ("continuation", "data_start_sequence"),
                         ("train", "retain_terminal_checkpoint")):
        del legacy[section][key]
    assert config_dicts_equal(legacy, current)
    fork = copy.deepcopy(current)
    fork["continuation"]["optimizer"] = "fresh"
    assert not config_dicts_equal(legacy, fork)


def _acceptance_fork(tmp_path):
    from scaffolds.transformer.configs import Config

    config = Config()
    config.data.source, config.data.batch_size = "packed", 256
    config.data.train_manifest, config.data.eval_manifest = str(tmp_path / "train.json"), str(tmp_path / "eval.json")
    config.continuation = ContinuationConfig(
        parent_config=str(tmp_path / "parent.yaml"), parent_config_sha256="a" * 64,
        parent_checkpoint=str(tmp_path / "parent/7570"), parent_checkpoint_sha256="b" * 64,
        parent_data_identity_sha256="c" * 64, parent_step=7570, optimizer="fresh",
        data_start_sequence=7570 * 256)
    config.optim.total_steps, config.optim.warmup_steps = 7570, 455
    config.train.eval_every, config.train.ckpt_every, config.train.eval_batches = 500, 500, 19
    return config


def test_acceptance_starts_a_fork_at_zero_and_counts_its_baseline(tmp_path):
    config = _acceptance_fork(tmp_path)
    assert acceptance.continuation_start_step(config) == acceptance.continuation_start_step(config_to_dict(config)) == 0
    assert acceptance.is_continuation(config) and acceptance.is_continuation(config_to_dict(config))
    scheduled = acceptance.scheduled_work_counts(config)
    assert acceptance.remaining_work_counts(config) == (scheduled[0] + 1, scheduled[1] + 1)
    spec = acceptance.attention_spec(config, entry_module="unused")
    assert spec.check_initialization is None
    paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=True)
    assert "scaffolds/transformer/tests/test_continuation_fork.py" in paths
    config.continuation = replace(config.continuation, optimizer="parent", data_start_sequence=None)
    paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=True)
    assert "scaffolds/transformer/tests/test_continuation_fork.py" not in paths


@pytest.mark.parametrize("broken", (None, "parameters", "optimizer", "counts", "data", "step"))
def test_fork_initialization_requires_a_fresh_optimizer_at_its_data_position(tmp_path, monkeypatch, broken):
    from scaffolds.transformer import models

    config = _acceptance_fork(tmp_path)
    audit = {"parent_parameters_sha256": "p", "backbone_parameters_sha256": "p", "optimizer": "fresh",
             "optimizer_sha256": "o", "fresh_optimizer_sha256": "o", "optimizer_counts": [0, 0],
             "data_start_sequence": config.continuation.data_start_sequence, "parent_step": 7570, "start_step": 0}
    step = 0
    if broken == "parameters":
        audit["backbone_parameters_sha256"] = "changed"
    elif broken == "optimizer":
        audit["optimizer_sha256"] = "parent history"
    elif broken == "counts":
        audit["optimizer_counts"] = [7570, 7570]
    elif broken == "data":
        audit["data_start_sequence"] = 0
    elif broken == "step":
        step = 7570
    state = SimpleNamespace(params={"weight": jnp.array([2., 3.])}, step=step)
    monkeypatch.setattr(continuation_module, "initialize_continuation_state", lambda current: (state, audit))
    monkeypatch.setattr(models, "model_parameter_count", lambda _: 2)
    if broken is None:
        assert acceptance.initialization_facts(config)["continuation"] == audit
    else:
        with pytest.raises(RuntimeError, match="start step|backbone parameters|fresh AdamW"):
            acceptance.initialization_facts(config)


def test_acceptance_pilot_and_resume_of_a_fork_read_its_offset_cursor(tmp_path, monkeypatch):
    config = _acceptance_fork(tmp_path)
    offset = config.continuation.data_start_sequence
    args = SimpleNamespace(config=tmp_path / "recipe.yaml", directory=tmp_path / "acceptance",
                           gpu_count=1, gpu_type="H100")
    experiment = {"identity": "test", "source_commit": "c" * 40, "config": config_to_dict(config),
                  "allocation_seconds": 48 * 3600}
    spec = acceptance.AcceptanceSpec(entry_module="unused", cpu_tests=(), pilot_steps=20)
    calls = []

    def worker(spec, args, name, output, *, stop, **unused):
        if name == "hardware":
            return {"attention_backend": "fused", "device_kinds": ["NVIDIA H100 80GB HBM3"]}
        calls.append(stop)
        pilot = args.directory / "pilot"
        pilot.mkdir(exist_ok=True)
        (pilot / "metrics.jsonl").write_text(json.dumps({
            "step": 20, "validation_nll": 2.6, "attention_diagnostics": {"valid": True}}) + "\n")
        return {"start_step": 0 if stop == 20 else 20, "final_step": stop,
                "restored_step": None if stop == 20 else 20, "data_parallel_devices": 1,
                "attention_backend": "fused", "data_start_sequence": offset + (0 if stop == 20 else 20) * 256,
                "timing": {"train_step_seconds": [30.0, 2.0], "evaluation_seconds": [3.0],
                           "diagnostic_seconds": [0.5], "evaluation_batch_seconds": [1.0, 1.0, 1.0],
                           "checkpoint_seconds": [4.0], "runtime_seconds": 50.0}}

    monkeypatch.setattr(acceptance, "_run_worker", worker)
    monkeypatch.setattr(acceptance, "load_config", lambda _: config)
    monkeypatch.setattr(acceptance, "passed", lambda path, _: path.stem in {
        "cpu-tests", "cpu-distributed-tests", "gpu-tests", "initialization"})
    monkeypatch.setattr(acceptance, "validate_resume_audit", lambda before, after: None)
    args.directory.mkdir()
    for stage in ("cpu-tests", "cpu-distributed-tests", "gpu-tests"):
        (args.directory / f"{stage}.json").write_text("{}\n")
    acceptance._run(spec, args, experiment)
    assert calls == [20, 21]
    projection = json.loads((args.directory / "pilot.json").read_text())["runtime_projection"]
    assert projection["counts"]["training_updates"] == 7570


def test_a_fork_replays_from_its_update_zero_checkpoint(parent_run, tmp_path, capsys):
    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.validation.replay_checkpoint import prepare

    config = fork_config(parent_run, tmp_path / "child", query=False)
    loop.run_training(config, stop_after_step=1)
    # The replay tool copies the recorded first batch with the baseline checkpoint, and the replay re-checks it.
    replay = load_config(prepare(config.train.workdir, 0, tmp_path / "scratch"))
    capsys.readouterr()
    result = loop.run_training(replay, resume=True, stop_after_step=1)
    assert (result["restored_step"], result["data_start_sequence"]) == (0, OFFSET)
    assert "fork_first_batch_verified" in capsys.readouterr().out
    # Without the record, a replay from update 0 names what it needs instead of training unchecked.
    bare = load_config(prepare(config.train.workdir, 0, tmp_path / "bare"))
    (tmp_path / "bare" / "continuation.json").unlink()
    with pytest.raises(ValueError, match="needs its run's continuation.json"):
        loop.run_training(bare, resume=True, stop_after_step=1)


def test_gradient_events_of_a_fork_name_the_sequences_its_updates_read(parent_run, tmp_path):
    from scaffolds.transformer.data.loaders import make_train_iterator
    from scaffolds.transformer.training.checkpointing import gradient_event_directory

    parent_config, _ = parent_run
    config = fork_config(parent_run, tmp_path / "child", query=False, total_steps=8)
    config.train = replace(config.train, gradient_event_capture=True, gradient_event_factor=1.01,
                           gradient_event_min_norm=0.0, gradient_event_history=2, gradient_event_min_history=1,
                           gradient_event_max_states=2)
    loop.run_training(config)
    rows = [json.loads(line) for line in (config.train.workdir / "gradient_events.jsonl").read_text().splitlines()
            if json.loads(line).get("event") == "gradient_spike"]
    assert rows, "the scenario needs at least one captured update"
    for row in rows:
        update = row["update"]
        assert row["sequences"] == {"first": OFFSET + (update - 1) * BATCH, "end": OFFSET + update * BATCH}
        # The coordinates rebuild the update's batch, as the record hashed it.
        iterator = make_train_iterator(replace(config.data, batch_size=2), start_sequence=row["sequences"]["first"])
        host = np.stack([next(iterator) for _ in range(2)])
        assert state_sha256(jnp.asarray(host)) == row["batch_sha256"]
    assert gradient_event_directory(config).is_dir()
