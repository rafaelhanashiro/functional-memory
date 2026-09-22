import json
from pathlib import Path
import xml.etree.ElementTree as ET

import jax
import pytest

from scaffolds.transformer.studies.key_dimension import preflight
from scaffolds.transformer.studies.key_dimension.configs import KEY_DIMENSIONS, config_to_dict, load_config, save_config


BASE = Path(__file__).resolve().parents[5] / "results/key_dimension/configs/base.yaml"


def attention_report(path, dimensions, *, skipped=False):
    suite = ET.Element("testsuite")
    for check in preflight.CUDNN_CHECKS:
        for dim in dimensions:
            case = ET.SubElement(suite, "testcase", name=f"{check}[{dim}]")
            if skipped:
                ET.SubElement(case, "skipped", message="cuDNN requires a GPU")
    ET.ElementTree(suite).write(path)


@pytest.mark.parametrize("dimensions", [(256,), KEY_DIMENSIONS])
def test_attention_gate_accepts_exact_manifest_widths(tmp_path, dimensions):
    path = tmp_path / "attention.xml"
    attention_report(path, dimensions)
    assert preflight.validate_attention_report(path, dimensions) == 2 * len(dimensions)


@pytest.mark.parametrize("dimensions,skipped", [((128,), False), ((256,), True)])
def test_attention_gate_rejects_wrong_width_or_skipped_checks(tmp_path, dimensions, skipped):
    path = tmp_path / "attention.xml"
    attention_report(path, dimensions, skipped=skipped)
    with pytest.raises(RuntimeError, match="Expected 2 passing accelerator checks"):
        preflight.validate_attention_report(path, (256,))


def dk256_manifest(tmp_path):
    config = load_config(BASE)
    config.model.qk_head_dim = 256
    path = tmp_path / "dk256.yaml"
    save_config(config, path)
    return {
        "identity": "test-dk256",
        "source_commit": "a" * 40,
        "dependency_lock_sha256": "locked",
        "execution_root": str(tmp_path),
        "preflight_dir": str(tmp_path / "preflight"),
        "runs": [{
            "name": "200m-dk256-seed0",
            "qk_head_dim": 256,
            "config_path": str(path),
            "config": config_to_dict(config),
        }],
    }


def test_dk256_preflight_preserves_production_shape_and_resumes_checkpoint(tmp_path, monkeypatch):
    manifest = dk256_manifest(tmp_path)
    monkeypatch.setattr(preflight, "verify_manifest", lambda _: None)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "local_devices", lambda: ["gpu0", "gpu1"])
    requested_tests = []

    def fake_pytest(command, *, cwd, check):
        assert check and cwd == manifest["execution_root"]
        requested_tests.extend(arg for arg in command if "::test_" in arg)
        path = Path(next(arg.removeprefix("--junitxml=") for arg in command if arg.startswith("--junitxml=")))
        attention_report(path, (256,))

    monkeypatch.setattr(preflight.subprocess, "run", fake_pytest)
    stops = []

    def fake_training(config, *, resume, stop_after_step):
        assert resume
        assert config.model.qk_head_dim == 256
        assert config.model.value_head_dim == 64
        assert config.model.max_seq_len == 2048
        assert config.model.num_layers == 10
        assert config.model.hidden_size == 1024
        assert config.model.attention_backend == "cudnn"
        assert config.data.sequence_length == 2049
        assert config.data.batch_size == 256
        assert config.train.micro_batch_size == 64
        assert config.train.dtype == "bfloat16"
        stops.append(stop_after_step)
        return {"final_step": stop_after_step, "restored_step": 2 if stop_after_step == 3 else None}

    monkeypatch.setattr("scaffolds.transformer.training.loop.run_training", fake_training)
    monkeypatch.setattr("scaffolds.transformer.studies.key_dimension.plot.committed_records", lambda _: [{
        "step": 2, "validation_nll": 10.0, "validation_perplexity": 22026.5,
    }])
    preflight.run_preflight(manifest)
    assert stops == [2, 3]
    assert requested_tests == [
        f"scaffolds/transformer/studies/key_dimension/tests/test_attention.py::{check}[256]"
        for check in preflight.CUDNN_CHECKS
    ]
    root = Path(manifest["preflight_dir"])
    record = json.loads((root / "checks/dk256.json").read_text())
    assert record["outcome"]["restored_step"] == 2
    assert record["outcome"]["final_step"] == 3
    assert preflight.passed_check(root / "success.json", manifest["identity"])


def test_production_preflight_rejects_cpu_before_running_checks(tmp_path, monkeypatch):
    manifest = dk256_manifest(tmp_path)
    monkeypatch.setattr(preflight, "verify_manifest", lambda _: None)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(jax, "local_devices", lambda: ["cpu"])
    with pytest.raises(RuntimeError, match="same two-GPU layout"):
        preflight.run_preflight(manifest)
    assert not (Path(manifest["preflight_dir"]) / "success.json").exists()
