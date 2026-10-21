"""The shared acceptance module refuses weak or inconsistent evidence."""
import pytest

from scaffolds.transformer import acceptance as preflight


def test_long_training_requires_requeue_and_checkpoint_progress():
    from dataclasses import replace
    config = {"optim": {"total_steps": 19074}, "train": {"ckpt_every": 500, "eval_batches": 7}}
    timing = {"runtime_seconds": 300, "train_step_seconds": [100, 12, 13],
              "evaluation_seconds": [2], "checkpoint_seconds": [15]}
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), allocation_hours=24)
    with pytest.raises(RuntimeError, match="beyond a 24h allocation"):
        preflight.validate_allocation_timing(spec, config, timing, 80 * 3600)
    requeue = replace(spec, allow_requeue=True)
    result = preflight.validate_allocation_timing(requeue, config, timing, 80 * 3600)
    assert result["estimated_allocations"] == 4
    assert result["allow_requeue"] is True
    config["train"]["ckpt_every"] = 10000
    with pytest.raises(RuntimeError, match="before durable checkpoint progress"):
        preflight.validate_allocation_timing(requeue, config, timing, 80 * 3600)
    config["train"]["ckpt_every"] = 0
    with pytest.raises(RuntimeError, match="periodic checkpoints"):
        preflight.validate_allocation_timing(requeue, config, timing, 80 * 3600)


@pytest.mark.parametrize("projected", (float("nan"), float("inf"), 0, -1))
def test_allocation_timing_rejects_invalid_estimates(projected):
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), allow_requeue=True)
    with pytest.raises(RuntimeError, match="Invalid allocation budget"):
        preflight.validate_allocation_timing(spec, {}, {}, projected)


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


@pytest.mark.parametrize("conditioning,kernel_tests", (("static", "mosaic_taylor"), ("query", "mosaic_query")))
def test_mosaic_backend_requires_its_kernel_and_repeatability_tests(conditioning, kernel_tests):
    config = {"model": {"attention_degree": 2, "attention_normalization": conditioning == "static",
                        "attention_coefficients": "fixed" if conditioning == "static" else "learned",
                        "attention_coefficient_conditioning": conditioning, "attention_coefficient_activation":
                        "identity" if conditioning == "static" else "tanh"}}
    paths = preflight.attention_test_paths(config, "mosaic", cpu=False)
    assert paths[-2:] == (f"scaffolds/transformer/tests/test_{kernel_tests}.py",
                          "scaffolds/transformer/tests/test_mosaic_repeatability.py")
    assert not any("mosaic" in path for path in preflight.attention_test_paths(config, "mosaic", cpu=True))
    assert not any("mosaic" in path for path in preflight.attention_test_paths(config, "fused", cpu=False))


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


def test_unnormalized_static_mosaic_runs_require_their_kernel_tests():
    config = {"model": {"attention_degree": 3, "attention_normalization": False, "attention_coefficients": "learned",
                        "attention_coefficient_conditioning": "static"}}
    paths = preflight.attention_test_paths(config, "mosaic", cpu=False)
    assert paths[-2:] == ("scaffolds/transformer/tests/test_mosaic_unnormalized.py",
                          "scaffolds/transformer/tests/test_mosaic_repeatability.py")


@pytest.mark.parametrize("model,backend,expected", [
    ({"attention_degree": 3, "attention_normalization": False}, "mosaic", ("--mosaic-precision", "bf16")),
    ({"attention_degree": 2}, "mosaic", ("--mosaic-precision", "bf16")),
    ({"attention_degree": 3}, "mosaic", ("--mosaic-precision", "bf16x3")),
    ({"attention_degree": 3}, "fused", ()),
    ({"attention_degree": None}, "cudnn", ()),
])
def test_mosaic_gpu_tests_cover_only_the_precision_in_use(model, backend, expected):
    assert preflight.mosaic_test_options({"model": model}, backend) == expected
