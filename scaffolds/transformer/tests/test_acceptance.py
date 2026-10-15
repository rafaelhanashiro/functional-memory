"""The shared acceptance module refuses weak or inconsistent evidence."""
import pytest

from scaffolds.transformer import acceptance as preflight


@pytest.mark.parametrize("cpu,backend,expected", [
    (True, "blocked", preflight.TANH_WRITE_GDN_TESTS),
    (True, "fused", preflight.TANH_WRITE_GDN_TESTS),
    (False, "fused", preflight.TANH_WRITE_GDN_KERNEL_TESTS),
])
def test_tanh_gdn_acceptance_cannot_omit_signed_write_checks(cpu, backend, expected):
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    config = {"model": {"attention_mechanism": "gated_deltanet", "gdn_write_activation": "tanh"}}
    assert expected in preflight.test_paths(spec, config, resolved_backend=backend, cpu=cpu)


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


@pytest.mark.parametrize("cpu", (False, True))
@pytest.mark.parametrize("backend", ("blocked", "fused"))
@pytest.mark.parametrize("degree", (2, 3, 4))
def test_polynomial_gdn_cannot_omit_its_correctness_checks(cpu, backend, degree):
    config = {"model": {"attention_mechanism": "gated_deltanet", "gdn_polynomial_degree": degree}}
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    paths = preflight.test_paths(spec, config, resolved_backend=backend, cpu=cpu)
    expected = ((preflight.POLYNOMIAL_GDN_TESTS, preflight.POLYNOMIAL_GDN_INTEGRATION_TESTS) if cpu
                else (preflight.POLYNOMIAL_GDN_KERNEL_TESTS,) if backend == "fused" else ())
    if degree >= 3:
        expected += ((preflight.CUBIC_GDN_TESTS,) if cpu
                     else (preflight.CUBIC_GDN_KERNEL_TESTS,) if backend == "fused" else ())
    if degree == 4:
        expected += ((preflight.QUARTIC_GDN_TESTS,) if cpu
                     else (preflight.QUARTIC_GDN_KERNEL_TESTS,) if backend == "fused" else ())
    assert paths == expected
    config["model"]["gdn_polynomial_degree"] = 1
    assert preflight.test_paths(spec, config, resolved_backend=backend, cpu=cpu) == ()


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


@pytest.mark.parametrize("cpu", (False, True))
@pytest.mark.parametrize("backend", ("blocked", "fused"))
def test_query_conditioning_requires_its_checks_even_with_an_empty_adapter(cpu, backend):
    config = {"model": {"attention_degree": 1, "attention_coefficients": "learned",
                        "attention_coefficient_conditioning": "query", "attention_normalization": False}}
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    assert preflight.test_paths(spec, config, resolved_backend=backend, cpu=cpu) == (
        preflight.INPUT_DEPENDENT_COEFFICIENT_TESTS,)
    config["model"]["attention_coefficient_conditioning"] = "static"
    assert preflight.test_paths(spec, config, resolved_backend=backend, cpu=cpu) == ()


def test_production_acceptance_runs_required_gpu_tests_with_an_empty_adapter_list(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict

    config = Config(model=ModelConfig(attention_degree=2, attention_coefficients="learned"))
    experiment = {"identity": "test", "source_commit": "a" * 40, "config": config_to_dict(config)}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "recipe.yaml", gpu_count=1, gpu_type="H100")
    calls = []

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "fused", "device_kinds": ["NVIDIA H100 80GB HBM3"]}
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


def test_runs_selecting_the_same_tests_share_passed_stages(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict

    ran = []
    models = {"H200": "NVIDIA H200", "H100": "NVIDIA H100 80GB HBM3"}
    allocated = {}

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "fused", "device_kinds": [allocated["model"]] * args.gpu_count}
        raise RuntimeError("stop after testing")

    def run(command, **options):
        junit = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--junitxml=")))
        ran.append(junit.stem)
        junit.write_text('<testsuite><testcase name="case"/></testsuite>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight, "_run_worker", worker)
    monkeypatch.setattr(preflight.subprocess, "run", run)
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=("scaffolds/transformer/tests/test_model.py",),
                                    select_tests=preflight.attention_test_paths)
    cache = tmp_path / "cache"

    def acceptance(name, degree, gpu_type, source="a" * 40, model=None):
        allocated["model"] = model or models[gpu_type]
        config = config_to_dict(Config(model=ModelConfig(attention_degree=degree)))
        experiment = {"identity": name, "source_commit": source, "config": config, "gpu_type": gpu_type,
                      "dependency_lock_sha256": "b" * 64, "runtime": {"TRANSFORMER_HOST_STAGING": "0"}}
        args = SimpleNamespace(directory=tmp_path / name, config=tmp_path / "recipe.yaml", gpu_count=2,
                               gpu_type=gpu_type, test_cache=cache)
        ran.clear()
        with pytest.raises(RuntimeError, match="stop after testing"):
            preflight._run(spec, args, experiment)
        return list(ran)

    stages = ["cpu-tests", "cpu-distributed-tests", "gpu-tests"]
    assert acceptance("first", 4, "H200") == stages
    # Another recipe that selects the same tests reruns none of them, and records where they passed.
    assert acceptance("second", 8, "H200") == []
    reused = json.loads((tmp_path / "second" / "gpu-tests.json").read_text())
    assert reused["identity"] == json.loads((tmp_path / "second" / "identity.json").read_text())["identity"]
    assert reused["tests_passed"] == 1
    assert reused["reused_from"]["acceptance_directory"] == str(tmp_path / "first")
    assert (tmp_path / "second" / "gpu-tests.xml").read_text() == (tmp_path / "first" / "gpu-tests.xml").read_text()
    # GPU tests are per GPU model, even under one launcher GPU type; other source or tests
    # rerun everything they change.
    assert acceptance("third", 8, "H100") == ["gpu-tests"]
    assert acceptance("nvl", 8, "H100", model="NVIDIA H100 NVL") == ["gpu-tests"]
    assert acceptance("nvl-again", 4, "H100", model="NVIDIA H100 NVL") == []
    assert acceptance("fourth", 8, "H200", source="c" * 40) == stages
    assert acceptance("odd", 3, "H200") == stages
    # A report that no longer matches its entry is never reused.
    key = preflight.stage_identity({"source_commit": "a" * 40, "dependency_lock_sha256": "b" * 64,
                                   "runtime": {"TRANSFORMER_HOST_STAGING": "0"}, "gpu_type": "H200",
                                   "device_kinds": ["NVIDIA H200"]},
                                  "gpu-tests", preflight.test_paths(spec, config_to_dict(Config(model=ModelConfig(
                                      attention_degree=4))), resolved_backend="fused", cpu=False),
                                  ("-m", "gpu"), cpu=False)
    (cache / f"{key}.xml").write_text('<testsuite><testcase name="other"/></testsuite>')
    assert acceptance("fifth", 4, "H200") == ["gpu-tests"]


def test_a_requeue_onto_another_gpu_model_repeats_the_runs_own_acceptance(tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config, config_to_dict

    allocated = {}

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "fused", "device_kinds": [allocated["model"]] * args.gpu_count}
        raise RuntimeError(f"ran {name}")

    monkeypatch.setattr(preflight, "_run_worker", worker)
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    experiment = {"identity": "launcher", "source_commit": "a" * 40, "config": config_to_dict(Config()),
                  "gpu_type": "H100", "dependency_lock_sha256": "b" * 64, "runtime": {"TRANSFORMER_HOST_STAGING": "0"}}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "recipe.yaml", gpu_count=2, gpu_type="H100")
    allocated["model"] = "NVIDIA H100 80GB HBM3"
    passed_on = preflight.on_devices(experiment, [allocated["model"]] * 2)
    for name in (*preflight.CHECKS, "success"):
        preflight.record(tmp_path / f"{name}.json", passed_on)
    preflight._run(spec, args, experiment)
    assert "matching_acceptance_passed=" in capsys.readouterr().out
    # The launcher's GPU type is unchanged, but a different model has different memory and speed.
    allocated["model"] = "NVIDIA H100 NVL"
    with pytest.raises(RuntimeError, match="ran initialization"):
        preflight._run(spec, args, experiment)
    assert "matching_acceptance_passed=" not in capsys.readouterr().out
    assert json.loads((tmp_path / "identity.json").read_text())["device_kinds"] == ["NVIDIA H100 NVL"]


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


@pytest.mark.parametrize("hours", ("2", "nan", "inf"))
def test_allocation_hours_argument_rejects_invalid_budgets(hours, capsys):
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    with pytest.raises(SystemExit) as error:
        preflight.main(spec, ["--allocation-hours", hours])
    assert error.value.code == 2
    assert "finite and greater than 2" in capsys.readouterr().err


@pytest.mark.parametrize("override,expected", ((None, 48), (24.5, 24.5)))
def test_allocation_override_applies_before_identity_and_execution(tmp_path, monkeypatch, override, expected):
    seen = []
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    monkeypatch.setenv("SLURM_JOB_ID", "test")

    def identify(selected, *args):
        seen.append(("identity", selected.allocation_hours))
        return {"identity": "test"}

    monkeypatch.setattr(preflight, "identity", identify)
    monkeypatch.setattr(preflight, "_run", lambda selected, *args: seen.append(("run", selected.allocation_hours)))
    argv = [str(tmp_path / "recipe.yaml"), "--directory", str(tmp_path)]
    if override is not None:
        argv += ["--allocation-hours", str(override)]
    preflight.main(spec, argv)
    assert seen == [("identity", expected), ("run", expected)]
    assert spec.allocation_hours == 48


def test_worker_receives_the_resolved_allocation_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    output = tmp_path / "worker.json"

    def run(command, **kwargs):
        calls.append(command)
        output.write_text('{"passed": true}')

    monkeypatch.setattr(preflight.subprocess, "run", run)
    spec = preflight.AcceptanceSpec(entry_module="experiments.demo.acceptance", cpu_tests=(), allocation_hours=24.5)
    args = SimpleNamespace(config=tmp_path / "recipe.yaml", directory=tmp_path, gpu_count=4, gpu_type="H100")
    assert preflight._run_worker(spec, args, "pilot", output, stop=20) == {"passed": True}
    assert calls[0][calls[0].index("--allocation-hours") + 1] == "24.5"


def test_shorter_allocation_cannot_reuse_longer_pilot_acceptance(tmp_path, monkeypatch, capsys):
    import json
    from dataclasses import replace
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config

    config = Config()
    config.optim.total_steps = 100
    config.train.source_commit = "auto"
    monkeypatch.setattr(preflight, "load_config", lambda path: config)
    monkeypatch.setattr(preflight, "sha256", lambda path: "fixture-sha256")
    monkeypatch.setenv("SHARED_TRANSFORMER_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("TRANSFORMER_HOST_STAGING", "0")
    args = SimpleNamespace(config=tmp_path / "recipe.yaml", directory=tmp_path, gpu_count=1, gpu_type="H100")
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), allocation_hours=48.)
    device_kinds = ["NVIDIA H100 80GB HBM3"]
    original = preflight.identity(spec, args.config, args.gpu_count, args.gpu_type)
    qualified = preflight.on_devices(original, device_kinds)
    for stage in (*preflight.CHECKS, "success"):
        preflight.record(tmp_path / f"{stage}.json", qualified)

    def worker(selected, arguments, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "fused", "device_kinds": device_kinds}
        if name == "initialization":
            return {}
        assert name == "pilot"
        pilot = tmp_path / "pilot"
        pilot.mkdir()
        (pilot / "metrics.jsonl").write_text(json.dumps({
            "step": 20, "validation_nll": 2., "attention_diagnostics": {"valid": True},
        }) + "\n")
        # A 25-hour projection fits the old 48-hour allocation, but fails the
        # standard partition's 22-hour production guard after a budget change.
        return {"final_step": 20, "data_parallel_devices": 1,
                "timing": {"train_step_seconds": [1., 900.], "runtime_seconds": 0.,
                           "evaluation_seconds": [0.], "checkpoint_seconds": [0.]}}

    monkeypatch.setattr(preflight, "_run_worker", worker)
    preflight._run(spec, args, original)
    assert "matching_acceptance_passed=" in capsys.readouterr().out
    shortened = replace(spec, allocation_hours=24.)
    changed = preflight.identity(shortened, args.config, args.gpu_count, args.gpu_type)
    assert changed["allocation_hours"] == 24.
    assert not preflight.passed(tmp_path / "success.json", preflight.on_devices(changed, device_kinds))
    # CPU/GPU correctness results are still shareable; only run qualification
    # depends on whether the pilot fits this allocation's wall-clock budget.
    assert preflight.stage_identity(qualified, "gpu-tests", ("kernel.py",), ("-m", "gpu"), cpu=False) == (
        preflight.stage_identity(preflight.on_devices(changed, device_kinds), "gpu-tests", ("kernel.py",),
                                 ("-m", "gpu"), cpu=False))
    with pytest.raises(RuntimeError, match="24.0h allocation: 25.00 hours"):
        preflight._run(shortened, args, changed)
    assert json.loads((tmp_path / "failure.json").read_text())["stage"] == "pilot"
