"""Fixed data selection, pooled diagnostics, and resumable distillation state."""
from copy import deepcopy
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from experiments.transformer_200m_4b import coefficient_distillation as distillation
from scaffolds.transformer.training.coefficient_distillation_regression import (
    make_regression_functions, predict_factorial, regression_metrics,
)


@pytest.fixture
def packed_selection(tmp_path):
    directory = tmp_path / "packed"
    directory.mkdir()
    tokens = np.arange(30, dtype=np.uint16).reshape(6, 5)
    shards = []
    for index in range(3):
        path = directory / f"part-{index}.npy"
        np.save(path, tokens[2 * index:2 * index + 2])
        shards.append({"path": path.name, "sequence_start": 2 * index,
                       "num_sequences": 2, "sha256": distillation.sha256(path)})
    manifest = {"format": "ofm-packed-sequences", "version": 1, "sequence_length": 5,
                "dtype": "uint16", "num_sequences": 6, "shards": shards}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    specification = {"manifest": str(path), "manifest_sha256": distillation.sha256(path),
                     "start_sequence": 1, "sequence_count": 3,
                     "batch_sha256": hashlib.sha256(tokens[1:4].astype(np.int32).tobytes()).hexdigest()}
    return specification, tokens, manifest


def test_selected_tokens_cross_shards_without_reading_unselected_data(packed_selection, monkeypatch):
    specification, tokens, _ = packed_selection
    directory = distillation.Path(specification["manifest"]).parent
    # A missing later shard must not force this selection to inspect unused test
    # data, but both selected shards must be verified before reading their tokens.
    (directory / "part-2.npy").unlink()
    calls = []
    original = distillation.sha256

    def counted(path):
        calls.append(distillation.Path(path).name)
        return original(path)

    monkeypatch.setattr(distillation, "sha256", counted)
    checked_shards = {}
    actual, identity = distillation.read_batch(specification, 5, checked_shards)
    np.testing.assert_array_equal(actual, tokens[1:4])
    assert actual.dtype == np.int32
    assert identity == specification["batch_sha256"]
    assert calls == ["manifest.json", "part-0.npy", "part-1.npy"]
    assert set(checked_shards) == {str(directory / "part-0.npy"), str(directory / "part-1.npy")}


def test_selected_shard_corruption_is_rejected(packed_selection):
    specification, _, _ = packed_selection
    selected = distillation.Path(specification["manifest"]).parent / "part-1.npy"
    changed = np.load(selected)
    changed[0, 0] += 1
    np.save(selected, changed)
    with pytest.raises(ValueError, match="Selected dataset shard changed"):
        distillation.read_batch(specification, 5)


def test_manifest_and_selected_token_identity_are_both_enforced(packed_selection):
    specification, _, _ = packed_selection
    changed = {**specification, "batch_sha256": "0" * 64}
    with pytest.raises(ValueError, match="Selected token batch changed"):
        distillation.read_batch(changed, 5)
    manifest = distillation.Path(specification["manifest"])
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="Dataset manifest changed"):
        distillation.read_batch(specification, 5)


@pytest.mark.parametrize("start,count", [(-1, 1), (6, 1), (4, 3)])
def test_fixed_selection_cannot_wrap_or_read_past_dataset(packed_selection, start, count):
    specification, _, _ = packed_selection
    with pytest.raises(ValueError, match="Selected sequences exceed"):
        distillation.read_batch({**specification, "start_sequence": start, "sequence_count": count}, 5)


def test_selected_shard_cannot_escape_manifest_directory(packed_selection):
    specification, _, manifest = packed_selection
    manifest = deepcopy(manifest)
    manifest["shards"][0]["path"] = "../elsewhere.npy"
    path = distillation.Path(specification["manifest"])
    path.write_text(json.dumps(manifest))
    specification = {**specification, "manifest_sha256": distillation.sha256(path)}
    with pytest.raises(ValueError, match="escapes its directory"):
        distillation.read_batch(specification, 5)


def test_chunked_regression_metrics_equal_pooled_metrics_with_uneven_energy():
    # The chunks have deliberately different target norms and the last is short.
    # Averaging chunk relative-L2 errors is therefore not the pooled relative L2.
    targets = jnp.asarray([[[[1., 2., 3.]], [[2., 1., 1.]], [[80., 60., 40.]],
                            [[100., 150., 200.]], [[.2, .1, .3]]]], jnp.float32)
    prediction = targets + jnp.asarray([[[[.2, .3, -.1]], [[.1, -.1, .4]], [[1., -2., 3.]],
                                        [[-1., 3., -2.]], [[.2, -.1, .3]]]], jnp.float32)
    inputs = prediction[:, :, 0, :]

    def evaluate(_, x, y):
        return regression_metrics(x[:, :, None, :], y)

    actual = distillation.full_regression(evaluate, None, inputs, targets, 2)
    expected = distillation.scalar_metrics(regression_metrics(prediction, targets))
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=3e-6, abs=3e-7), key


def test_checkpoint_restores_controller_candidates_and_exact_next_update(tmp_path):
    rng = np.random.default_rng(7)
    generator = {
        "kernel": jnp.asarray(rng.normal(size=(1, 2, 3, 4)) * .2, jnp.float32),
        "bias": jnp.ones((1, 2, 3)),
        "mixing": jnp.zeros((1, 2, 3, 4)),
    }
    inputs = jnp.asarray(rng.normal(size=(1, 8, 3)), jnp.float32)
    targets = predict_factorial(generator, inputs, 1 / 3) + jnp.asarray([.2, -.1, .3])
    init, step, _ = make_regression_functions(readout_scale=1 / 3)
    fitted, optimizer, _ = step(generator, init(generator), inputs, targets, .01)
    controller = {"step": 1, "best_fit_step": 1, "best_validation_step": 0,
                  "best_fit_mse": .04, "best_validation_nll": 2.7}
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    saved = distillation.save_checkpoint(checkpoints, 1, fitted, optimizer, fitted, generator, controller)
    template = {"generator": fitted, "optimizer": optimizer,
                "fit_best": fitted, "validation_best": generator}
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(saved, item=template)
    assert json.loads((saved / "controller.json").read_text()) == controller
    for expected, actual in zip(jax.tree.leaves(template), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(expected, actual)
    direct = step(fitted, optimizer, inputs, targets, .001)
    resumed = step(restored["generator"], restored["optimizer"], inputs, targets, .001)
    for expected, actual in zip(jax.tree.leaves(direct), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(expected, actual)
    # Saving another resumable checkpoint must keep both selection candidates
    # while retaining only the latest optimizer state.
    controller = {**controller, "step": 2}
    next_path = distillation.save_checkpoint(checkpoints, 2, direct[0], direct[1], fitted, generator, controller)
    assert next_path.is_dir() and not saved.exists()
    assert [path.name for path in checkpoints.iterdir()] == ["2"]
