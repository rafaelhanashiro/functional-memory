"""Small offline integration checks for matched state-rank comparisons."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from scaffolds.state_rank import compare_random, spectrum


@pytest.fixture
def tiny_case():
    config = LlamaConfig(
        vocab_size=31, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=16, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = LlamaModel(config).eval()
    inputs = [
        {"title": "First", "token_ids": [1, 3, 5, 7, 9]},
        {"title": "Second", "token_ids": [1, 2, 4, 8, 16]},
    ]
    return model, inputs


def native_reference(model, inputs, layer):
    """Build an archive from native caches and independent single-head analyses."""
    entries = []
    for record in inputs:
        with torch.inference_mode():
            cache = model(input_ids=torch.tensor([record["token_ids"]]), use_cache=True).past_key_values
        keys, values = (array[0].transpose(0, 1).double().numpy().copy() for array in cache[layer])
        per_head = [
            spectrum.compute_spectra(keys[:, head], values[:, head], scale=model.layers[layer].self_attn.scaling)
            for head in range(model.config.num_key_value_heads)
        ]
        result = {name: np.stack([entry[name] for entry in per_head], axis=1) for name in per_head[0]}
        entries.append(result | {"keys": keys, "values": values, "value_norms": np.linalg.norm(values, axis=-1)})
    return {name: np.stack([entry[name] for entry in entries]) for name in entries[0]}


def test_analyze_model_retains_all_prefix_metrics_and_selected_spectra(tiny_case):
    model, inputs = tiny_case
    layers, selected_prefixes = [1, 0], [1, 3, 5]
    calls = []
    handle = model.register_forward_pre_hook(lambda _model, _args: calls.append(1))
    try:
        arrays, validation = compare_random.analyze_model(model, inputs, layers, selected_prefixes)
    finally:
        handle.remove()
    assert len(calls) == len(inputs)
    np.testing.assert_array_equal(arrays["layers"], layers)
    np.testing.assert_array_equal(arrays["head_indices"], [0, 1])
    np.testing.assert_array_equal(arrays["prefixes"], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(arrays["spectrum_prefixes"], selected_prefixes)
    np.testing.assert_array_equal(arrays["token_ids"], [record["token_ids"] for record in inputs])
    assert arrays["token_ids"].dtype == np.int64
    # Axes are layer, input, prefix, KV head (then mode for spectra).
    for name in compare_random.METRICS:
        assert arrays[name].shape == (2, 2, 5, 2), name
    assert arrays["normalized_eigenvalues"].shape == (2, 2, 3, 2, 4)
    assert set(arrays) == set(compare_random.METRICS) | {
        "normalized_eigenvalues", "layers", "head_indices", "prefixes", "spectrum_prefixes", "token_ids",
    }
    assert not {"keys", "values", "scaled_eigenvalues"}.intersection(arrays)
    for layer_index, layer in enumerate(layers):
        reference = native_reference(model, inputs, layer)
        for name in compare_random.METRICS:
            np.testing.assert_allclose(arrays[name][layer_index], reference[name], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            arrays["normalized_eigenvalues"][layer_index],
            reference["normalized_eigenvalues"][:, np.array(selected_prefixes) - 1],
            rtol=1e-12, atol=1e-12,
        )
    assert validation["baseline_checks"] == []
    assert np.isfinite(validation["maximum_direct_relative_error"])
    assert validation["maximum_direct_relative_error"] < spectrum.DIRECT_RTOL


def test_analyze_model_accepts_matching_baseline_and_rejects_drift(tiny_case):
    model, inputs = tiny_case
    baseline = native_reference(model, inputs, layer=0)
    _, validation = compare_random.analyze_model(
        model, inputs, [1, 0], [1, 3, 5], baseline=baseline, baseline_layer=0,
    )
    assert [check["input_index"] for check in validation["baseline_checks"]] == [0, 1]
    assert all(check["passed"] for check in validation["baseline_checks"])
    baseline["threshold_rank"][0, -1, 0] += 1
    with pytest.raises(ValueError, match="baseline reproduction failed"):
        compare_random.analyze_model(
            model, inputs, [0, 1], [1, 3, 5], baseline=baseline, baseline_layer=0,
        )


@pytest.mark.parametrize("perturbation", [None, "keys", "threshold_rank", "energy_rank90", "energy_rank99"])
def test_compare_baseline_checks_keys_and_each_rank(tiny_case, perturbation):
    model, inputs = tiny_case
    reference = native_reference(model, inputs, layer=0)
    index = 1
    result = {name: value[index].copy() for name, value in reference.items()}
    keys, values = result.pop("keys"), result.pop("values")
    if perturbation == "keys":
        keys *= 1.01
    elif perturbation is not None:
        result[perturbation][-1, 0] += 1
    check = compare_random.compare_baseline(keys, values, result, reference, index)
    assert bool(check["passed"]) is (perturbation is None)
    if perturbation == "keys":
        assert check["keys_relative_error"] > spectrum.CACHE_RTOL
    elif perturbation is not None:
        assert check[f"{perturbation}_mismatches"] == 1


@pytest.fixture
def comparison_project(tmp_path, monkeypatch):
    """An on-disk provenance fixture without a model or dataset download."""
    baseline = tmp_path / "results/retained"
    baseline.mkdir(parents=True)
    inputs = [{"title": "Saved", "token_ids": [1, 2, 3, 4, 5]}]
    (baseline / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    metadata = {"model": "test/tiny-llama", "model_revision": "c" * 40, "layer": 0, "tokens": 5}
    (baseline / "metadata.json").write_text(json.dumps(metadata))
    np.savez_compressed(baseline / "spectra.npz", token_ids=np.array([inputs[0]["token_ids"]]))
    (tmp_path / "uv.lock").write_text("version = 1\nrevision = 3\n")
    config = {
        "schema_version": 1, "model_id": metadata["model"], "model_revision": metadata["model_revision"],
        "baseline_directory": "results/retained", "baseline_layer": 0,
        "layers": [0, 1], "seeds": [0, 2], "tokens": 5, "input_count": 1,
        "spectrum_prefixes": [1, 3, 5], "model_dtype": "float32", "analysis_dtype": "float64",
        "attention_implementation": "eager", "allow_tf32": False,
        "output_directory": "results/state_rank/random_vs_pretrained",
    }
    config_path = tmp_path / "comparison.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(compare_random, "source_commit", lambda: "a" * 40)
    return tmp_path, config_path, config, baseline


def test_prepare_manifest_reuses_identity_and_copies_exact_inputs(comparison_project, monkeypatch):
    root, config_path, config, baseline = comparison_project
    path = compare_random.prepare_manifest(config_path, root)
    before = path.read_bytes()
    manifest = json.loads(before)
    assert path == root / config["output_directory"] / "manifest.json"
    assert (path.parent / "inputs.json").read_bytes() == (baseline / "inputs.json").read_bytes()
    assert manifest["config"] == config
    assert manifest["conditions"] == [
        {"name": "pretrained", "weights": "pretrained", "seed": None},
        {"name": "random-seed0", "weights": "random", "seed": 0},
        {"name": "random-seed2", "weights": "random", "seed": 2},
    ]
    assert manifest["inputs_sha256"] == hashlib.sha256((baseline / "inputs.json").read_bytes()).hexdigest()
    assert manifest["checkpoint_retention"] == "none"
    compare_random.validate_manifest(manifest)
    monkeypatch.setattr(compare_random, "utc_now", lambda: "a later invocation")
    assert compare_random.prepare_manifest(config_path, root) == path
    assert path.read_bytes() == before


def test_prepare_manifest_rejects_incompatible_configuration_reuse(comparison_project):
    root, config_path, config, _ = comparison_project
    path = compare_random.prepare_manifest(config_path, root)
    before = path.read_bytes()
    changed = deepcopy(config)
    changed["seeds"] = [0, 3]
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different source/configuration/input identity"):
        compare_random.prepare_manifest(config_path, root)
    assert path.read_bytes() == before


@pytest.mark.parametrize("drift", ["inputs", "lock", "baseline"])
def test_manifest_validation_rejects_changed_provenance(comparison_project, drift):
    root, config_path, _, baseline = comparison_project
    path = compare_random.prepare_manifest(config_path, root)
    manifest = json.loads(path.read_text())
    changed_path = {
        "inputs": path.parent / "inputs.json",
        "lock": root / "uv.lock",
        "baseline": baseline / "spectra.npz",
    }[drift]
    changed_path.write_bytes(changed_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="provenance changed"):
        compare_random.validate_manifest(manifest)


@pytest.mark.parametrize("field", ["config", "conditions"])
def test_manifest_cannot_change_experiment_under_existing_identity(comparison_project, field):
    root, config_path, _, _ = comparison_project
    path = compare_random.prepare_manifest(config_path, root)
    manifest = json.loads(path.read_text())
    if field == "config":
        manifest["config"]["seeds"] = [17]
    else:
        manifest["conditions"][1]["seed"] = 17
    with pytest.raises(ValueError, match="identity or condition list"):
        compare_random.validate_manifest(manifest)


def test_prepare_requires_nonempty_spectrum_checks(comparison_project):
    root, config_path, config, _ = comparison_project
    config["spectrum_prefixes"] = []
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Spectrum prefixes"):
        compare_random.prepare_manifest(config_path, root)
