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


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares", "fixed", "fixed_square"))
@pytest.mark.parametrize("requested_backend", ("auto", "fused"))
@pytest.mark.parametrize("cpu", (False, True))
def test_normalized_fused_acceptance_cannot_omit_required_kernel_tests(parameterization, requested_backend, cpu):
    from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict

    config = Config(model=ModelConfig(attention_degree=2,
                                      attention_coefficients="fixed" if parameterization.startswith("fixed") else "learned",
                                      attention_fixed_coefficients=(0.5, 1.0, 0.5) if parameterization == "fixed" else None,
                                      attention_fixed_square_factors=(1.0, 1.0, 0.0) if parameterization == "fixed_square" else None,
                                      attention_coefficient_parameterization=("direct" if parameterization == "fixed"
                                                                              else "sum_of_squares" if parameterization == "fixed_square"
                                                                              else parameterization),
                                      attention_backend=requested_backend))
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    paths = preflight.test_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=cpu)
    assert paths == (preflight.NORMALIZED_POLYNOMIAL_TESTS,)


def test_required_kernel_tests_follow_resolved_backend_and_keep_adapter_checks():
    from dataclasses import replace
    from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict

    config = Config(model=ModelConfig(attention_degree=4, attention_coefficients="learned"))
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=("cpu.py",),
                                    gpu_tests=("gpu.py", preflight.NORMALIZED_POLYNOMIAL_TESTS))
    assert preflight.test_paths(spec, config_to_dict(config), resolved_backend="fused", cpu=False) == spec.gpu_tests
    assert preflight.test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=True) == spec.cpu_tests
    # Unnormalized and fixed kernels keep their declared acceptance paths.
    for model in (replace(config.model, attention_normalization=False),
                  replace(config.model, attention_coefficients="fixed")):
        other = replace(config, model=model)
        assert preflight.test_paths(spec, config_to_dict(other), resolved_backend="fused", cpu=True) == spec.cpu_tests


def test_production_acceptance_runs_required_gpu_tests_with_an_empty_adapter_list(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict

    config = Config(model=ModelConfig(attention_degree=2, attention_coefficients="learned"))
    experiment = {"identity": "test", "source_commit": "a" * 40, "config": config_to_dict(config)}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "recipe.yaml", gpu_count=1, gpu_type="H100")
    calls = []

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "fused"}
        raise RuntimeError("stop after testing")

    def run(command, **options):
        calls.append((command, options))
        junit = next(value.split("=", 1)[1] for value in command if value.startswith("--junitxml="))
        from pathlib import Path
        Path(junit).write_text('<testsuite><testcase name="kernel"/></testsuite>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight, "_run_worker", worker)
    monkeypatch.setattr(preflight.subprocess, "run", run)
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    with pytest.raises(RuntimeError, match="stop after testing"):
        preflight._run(spec, args, experiment)
    gpu_calls = [command for command, options in calls if options["env"]["JAX_PLATFORMS"] == "cuda"]
    assert len(gpu_calls) == 1 and preflight.NORMALIZED_POLYNOMIAL_TESTS in gpu_calls[0]


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

    config = Config()
    config.train.source_commit = "auto"
    monkeypatch.setattr(preflight, "load_config", lambda path: config)
    monkeypatch.setattr(preflight, "sha256", lambda path: "fixture-sha256")
    monkeypatch.setenv("SHARED_TRANSFORMER_SOURCE_COMMIT", "b" * 40)
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    assert preflight.identity(spec, tmp_path / "config.yaml", 4, "B300")["source_commit"] == "b" * 40


def test_workers_reinvoke_the_adapter_module_when_it_runs_as_main(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    main = ModuleType("__main__")
    main.__spec__ = SimpleNamespace(name="experiments.demo.acceptance")
    monkeypatch.setitem(sys.modules, "__main__", main)
    assert preflight.entry_module(preflight.AcceptanceSpec("__main__", ())) == main.__spec__.name
    assert preflight.entry_module(preflight.AcceptanceSpec("named.module", ())) == "named.module"
