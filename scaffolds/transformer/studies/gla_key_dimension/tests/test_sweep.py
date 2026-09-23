"""Scientific controls and refusal paths for an unqualified experiment."""
from pathlib import Path
import copy
import json

import pytest
import yaml

from scaffolds.transformer.configs import load_config
from scaffolds.transformer.models.registry import model_parameter_count
from scaffolds.transformer.studies.gla_key_dimension import sweep

ROOT = Path(__file__).resolve().parents[5]


def test_recipes_preserve_heatmap_data_optimizer_and_backbone():
    manifest = json.loads((ROOT / sweep.MANIFEST).read_text())
    assert [run["qk_head_dim"] for run in manifest["runs"]] == list(sweep.WIDTHS)
    for run in manifest["runs"]:
        path = ROOT / run["resolved_config"]["path"]
        candidate = yaml.safe_load(path.read_text())
        control = yaml.safe_load((ROOT / f"results/key_dimension_taylor/configs/200m-dk{run['qk_head_dim']:03d}-m004-seed0.yaml").read_text()) if run["qk_head_dim"] != 64 else yaml.safe_load((ROOT / "results/polynomial_attention/configs/200m-m004-seed0.yaml").read_text())
        assert candidate["data"] == control["data"]
        assert candidate["optim"] == control["optim"]
        excluded = {"attention_mechanism", "attention_backend", "attention_degree", "gla_gate_rank",
                    "gla_gate_tau", "gla_chunk_size", "gla_inner_chunk_size", "gla_use_rope"}
        assert {key: value for key, value in candidate["model"].items() if key not in excluded} == {
            key: value for key, value in control["model"].items() if key not in excluded}
        for key in ("seed", "dtype", "micro_batch_size", "log_every", "eval_every", "eval_batches", "ckpt_every"):
            assert candidate["train"][key] == control["train"][key]
        assert model_parameter_count(load_config(path).model) == run["parameter_count"]
        assert sweep.sha256(path) == run["resolved_config"]["sha256"]
        assert run["state_elements_per_layer"] == 16 * run["qk_head_dim"] * 64
    assert manifest["prediction_tokens"] == 10_000_269_312


def test_planning_manifest_cannot_authorize_execution():
    with pytest.raises(ValueError, match="frozen"):
        sweep.verify(ROOT / sweep.MANIFEST)


def test_manifest_identity_binds_recipe_and_execution_choices():
    manifest = json.loads((ROOT / sweep.MANIFEST).read_text())
    original = sweep.identity(manifest)
    for key in ("artifact_root", "source_commit", "home_cluster"):
        changed = copy.deepcopy(manifest)
        changed[key] = "changed"
        assert sweep.identity(changed) != original
    changed = copy.deepcopy(manifest)
    changed["runs"][0]["resolved_config"]["sha256"] = "0" * 64
    assert sweep.identity(changed) != original


def test_submit_refuses_unqualified_production(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / sweep.MANIFEST).read_text())
    manifest.update(artifact_root=str(tmp_path), identity="test", execution_root=str(ROOT))
    monkeypatch.setattr(sweep, "verify", lambda _: manifest)
    with pytest.raises(ValueError, match="Missing qualification"):
        sweep.submit(tmp_path / "manifest.json", "train", None, node="dogo", gpu="h100", dry_run=True)


def test_verification_rejects_code_imported_from_another_checkout(tmp_path):
    manifest = json.loads((ROOT / sweep.MANIFEST).read_text())
    manifest.update(source_status="frozen", execution_root=str(tmp_path), artifact_root=str(tmp_path))
    manifest["identity"] = sweep.identity(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Imported study code"):
        sweep.verify(path)


def test_run_lock_excludes_duplicate_writer_and_releases_after_failure(tmp_path):
    manifest = {"artifact_root": str(tmp_path)}
    with pytest.raises(ValueError, match="interrupted"):
        with sweep.run_lock(manifest, "run", "production"):
            with pytest.raises(RuntimeError, match="Another production process"):
                with sweep.run_lock(manifest, "run", "production"):
                    pytest.fail("Concurrent writer acquired the run lock")
            raise ValueError("interrupted")
    with sweep.run_lock(manifest, "run", "production"):
        pass


def test_preflight_rejects_all_skipped_gpu_tests_before_starting_pilot(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / sweep.MANIFEST).read_text())
    manifest.update(artifact_root=str(tmp_path), identity="test", execution_root=str(ROOT))
    monkeypatch.setattr(sweep, "verify", lambda _: manifest)
    calls = []

    def skipped(command, **kwargs):
        calls.append(command)
        junit = Path(next(arg.removeprefix("--junitxml=") for arg in command if arg.startswith("--junitxml=")))
        junit.write_text('<testsuite><testcase name="gpu"><skipped /></testcase></testsuite>')

    monkeypatch.setattr(sweep.subprocess, "run", skipped)
    with pytest.raises(RuntimeError, match="no skipped tests"):
        sweep.preflight(tmp_path / "manifest.json", manifest["runs"][0]["name"])
    assert len(calls) == 1
    assert not (sweep.evidence_root(manifest) / manifest["runs"][0]["name"] / "success.json").exists()


def test_only_production_jobs_request_requeue(tmp_path, monkeypatch, capsys):
    manifest = json.loads((ROOT / sweep.MANIFEST).read_text())
    manifest.update(artifact_root=str(tmp_path), identity="test", execution_root=str(ROOT))
    monkeypatch.setattr(sweep, "verify", lambda _: manifest)
    name = manifest["runs"][0]["name"]
    sweep.submit(tmp_path / "manifest.json", "preflight", name, node="dogo", gpu="h100", dry_run=True)
    command = capsys.readouterr().out
    assert "--no-requeue" in command and "--requeue" not in command
    success = sweep.evidence_root(manifest) / name / "success.json"
    success.parent.mkdir(parents=True)
    success.write_text("{}")
    sweep.submit(tmp_path / "manifest.json", "train", name, node="dogo", gpu="h100", dry_run=True)
    command = capsys.readouterr().out
    assert "--requeue" in command and "--no-requeue" not in command
