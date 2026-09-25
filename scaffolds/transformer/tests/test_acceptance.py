"""The shared acceptance module refuses weak or inconsistent evidence."""
import pytest

from scaffolds.transformer import acceptance as preflight


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


@pytest.mark.parametrize("setting,value", [
    ("TRANSFORMER_HOST_STAGING", "1"), ("NCCL_P2P_DISABLE", "1"),
    ("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90"), ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
    ("XLA_PYTHON_CLIENT_ALLOCATOR", "platform"), ("XLA_FLAGS", "--test-runtime-change"), ("CUDA_HOME", "/other/cuda"),
])
def test_passed_acceptance_requires_the_same_runtime(tmp_path, monkeypatch, setting, value):
    from scaffolds.transformer.configs import Config

    config = Config()
    config.train.source_commit = "auto"
    monkeypatch.setattr(preflight, "load_config", lambda path: config)
    monkeypatch.setattr(preflight, "sha256", lambda path: "fixture-sha256")
    monkeypatch.setenv("SHARED_TRANSFORMER_SOURCE_COMMIT", "a" * 40)
    for key in preflight.runtime_identity():
        monkeypatch.delenv(key, raising=False)
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    original = preflight.identity(spec, tmp_path / "config.yaml", 2, "H100")
    monkeypatch.setenv("TRANSFORMER_HOST_STAGING", "0")
    assert preflight.identity(spec, tmp_path / "config.yaml", 2, "H100") == original
    cache = tmp_path / "success.json"
    preflight.record(cache, original)
    assert preflight.passed(cache, original)
    monkeypatch.setenv(setting, value)
    changed = preflight.identity(spec, tmp_path / "config.yaml", 2, "H100")
    assert changed["config"] == original["config"]
    assert not preflight.passed(cache, changed)


def test_acceptance_uses_the_launch_commit_even_after_analysis_commits(tmp_path, monkeypatch):
    from scaffolds.transformer.configs import Config

    monkeypatch.setattr(preflight, "load_config", lambda path: Config())
    monkeypatch.setattr(preflight, "sha256", lambda path: "fixture-sha256")
    monkeypatch.setenv("SHARED_TRANSFORMER_SOURCE_COMMIT", "b" * 40)
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    assert preflight.identity(spec, tmp_path / "config.yaml", 4, "B300")["source_commit"] == "b" * 40
