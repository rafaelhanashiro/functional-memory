"""GLA recipes preserve the dK × Taylor controls, and acceptance refuses weak evidence."""
from pathlib import Path
import hashlib
import json

import pytest
import yaml

from scaffolds.transformer.configs import load_config
from scaffolds.transformer.models.registry import model_parameter_count
from scaffolds.transformer.studies.gla_key_dimension import preflight

ROOT = Path(__file__).resolve().parents[5]
RECORD = ROOT / "results/gla_key_dimension"
KEY_DIMENSIONS = (8, 16, 32, 64, 128)
GLA_FIELDS = {"attention_mechanism", "attention_degree", "gla_gate_rank", "gla_gate_tau",
              "gla_chunk_size", "gla_inner_chunk_size", "gla_use_rope"}
OUTPUT_FIELDS = {"workdir", "checkpoint_dir", "wandb_project", "wandb_run_name", "source_commit"}


def taylor_control(key_dimension):
    """The degree-4 Taylor recipe at the same width in the dK × Taylor record."""
    name = "200m-m004-seed0" if key_dimension == 64 else f"200m-dk{key_dimension:03d}-m004-seed0"
    return yaml.safe_load((ROOT / f"results/key_dimension_taylor/configs/{name}.yaml").read_text())


def manifest():
    return json.loads((RECORD / "provenance/manifest.json").read_text())


def test_recipes_preserve_dk_taylor_data_optimizer_backbone_and_cadence():
    runs = manifest()["runs"]
    assert [run["qk_head_dim"] for run in runs] == list(KEY_DIMENSIONS)
    for run in runs:
        path = ROOT / run["resolved_config"]["path"]
        candidate, control = yaml.safe_load(path.read_text()), taylor_control(run["qk_head_dim"])
        assert candidate["data"] == control["data"]
        assert candidate["optim"] == control["optim"]
        assert ({k: v for k, v in candidate["model"].items() if k not in GLA_FIELDS}
                == {k: v for k, v in control["model"].items() if k not in GLA_FIELDS})
        assert ({k: v for k, v in candidate["train"].items() if k not in OUTPUT_FIELDS}
                == {k: v for k, v in control["train"].items() if k not in OUTPUT_FIELDS})
        assert candidate["model"]["attention_mechanism"] == "gla" and candidate["model"]["attention_degree"] is None
        assert candidate["model"]["attention_backend"] == "fused"
        assert candidate["train"]["workdir"].endswith(f"results/gla_key_dimension/runs/{run['name']}")
        assert candidate["train"]["checkpoint_dir"].endswith(f"checkpoints/gla_key_dimension/{run['name']}")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run["resolved_config"]["sha256"]
        config = load_config(path)
        assert model_parameter_count(config.model) == run["parameter_count"]
        assert run["state_elements_per_layer"] == config.model.num_heads * run["qk_head_dim"] * config.model.value_head_dim
        tokens = config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1)
        assert tokens == manifest()["prediction_tokens"] == 10_000_269_312


def test_data_manifests_match_the_recorded_corpus():
    for item in manifest()["data_identity"].values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_comparison_points_reference_retained_key_dimension_taylor_runs():
    for control in manifest()["comparisons"]["runs"]:
        for key in ("resolved_config", "metrics"):
            path = ROOT / control[key]["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == control[key]["sha256"]


def test_acceptance_rejects_skipped_or_empty_test_reports(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text('<testsuite><testcase name="a"/><testcase name="b"><skipped/></testcase></testsuite>')
    with pytest.raises(RuntimeError, match="no skips"):
        preflight.require_passing_tests(report)
    report.write_text("<testsuite/>")
    with pytest.raises(RuntimeError, match="no skips"):
        preflight.require_passing_tests(report)
    report.write_text('<testsuite><testcase name="a"/><testcase name="b"/></testsuite>')
    assert preflight.require_passing_tests(report) == 2


def test_runtime_identity_requires_resolved_transfer_policy_and_records_toolkit(monkeypatch):
    monkeypatch.setenv("TRANSFORMER_HOST_STAGING", "auto")
    with pytest.raises(ValueError, match="resolved"):
        preflight.runtime_identity()
    monkeypatch.setenv("TRANSFORMER_HOST_STAGING", "0")
    monkeypatch.setenv("CUDA_HOME", "/locked/nvidia/cuda_nvcc")
    assert preflight.runtime_identity()["CUDA_HOME"] == "/locked/nvidia/cuda_nvcc"


def test_resume_audit_requires_identical_state_batch_and_rng():
    update = {"step": 21, "batch_sha256": "a" * 64, "rng_sha256": "b" * 64}
    initial = {"checkpoint_audit": {"final_state_sha256": "c" * 64, "checkpoint_state_sha256": "c" * 64,
                                    "next_update": update}}
    resumed = {"checkpoint_audit": {"start_state_sha256": "c" * 64, "final_state_sha256": "d" * 64,
                                    "checkpoint_state_sha256": "d" * 64, "first_update": dict(update)}}
    preflight.validate_resume_audit(initial, resumed)
    resumed["checkpoint_audit"]["first_update"]["batch_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="next batch or RNG"):
        preflight.validate_resume_audit(initial, resumed)
    resumed["checkpoint_audit"]["start_state_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="Restarted state"):
        preflight.validate_resume_audit(initial, resumed)


def test_kennel_launcher_accepts_only_qualified_b300_or_h100():
    launcher = (ROOT / "scaffolds/transformer/studies/gla_key_dimension/launch_kennel.sbatch").read_text()
    assert "#SBATCH --gres=gpu:4" in launcher and "#SBATCH --exclude=collie,samoyed,doodle,terrier" in launcher
    assert 'GPU_TYPE="${6:-auto}"' in launcher and "*B300*) GPU_TYPE=B300" in launcher and "*H100*) GPU_TYPE=H100" in launcher
    assert "--requeue" in launcher
