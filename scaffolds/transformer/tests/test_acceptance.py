"""The shared acceptance module refuses weak or inconsistent evidence."""
import pytest

from scaffolds.transformer import acceptance as preflight


def test_scratch_pilot_overrides_early_evaluation_without_mutating_production_recipe(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config, load_config, save_config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.cadence import evaluation_due

    config = Config()
    config.optim.total_steps = 7570
    config.train.eval_every = 500
    config.train.eval_early_every = 100
    config.train.eval_early_until = 1500
    path = tmp_path / "config.yaml"
    save_config(config, path)
    original = path.read_bytes()
    captured = {}

    def train(scratch, **kwargs):
        captured.update(config=scratch, options=kwargs)
        return {"final_step": kwargs["stop_after_step"]}

    monkeypatch.setattr(loop, "run_training", train)
    args = SimpleNamespace(worker="pilot", config=path, directory=tmp_path / "acceptance", stop_after_step=20)
    spec = preflight.AcceptanceSpec(entry_module="unused", pilot_steps=20, cpu_tests=())
    preflight._worker(spec, args, {"source_commit": "a" * 40})
    scratch = captured["config"]
    assert scratch.train.eval_every == 20 and scratch.train.eval_early_every is None
    assert scratch.train.eval_early_until == 0 and evaluation_due(scratch, 20)
    assert scratch.optim.total_steps == 7570
    assert captured["options"] == {"resume": True, "stop_after_step": 20, "audit_checkpoint": True,
                                   "profile_timing": True}
    assert path.read_bytes() == original
    frozen = load_config(path)
    assert (frozen.train.eval_every, frozen.train.eval_early_every, frozen.train.eval_early_until) == (500, 100, 1500)


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


def timing_profile():
    return {"runtime_seconds": 600.0, "setup_seconds": 80.0,
            "diagnostic_warmup_seconds": 40.0, "evaluation_warmup_seconds": 50.0,
            "train_step_seconds": [100.0, 2.0, 2.0, 2.0, 20.0],
            "diagnostic_seconds": [3.0, 3.0, 9.0],
            "evaluation_batch_seconds": [5.0, 5.0, 15.0],
            "evaluation_seconds": [98.0], "checkpoint_seconds": [6.0, 7.0]}


def projection_config():
    from scaffolds.transformer.configs import Config
    config = Config()
    config.optim.total_steps = 7570
    config.train.eval_every = config.train.ckpt_every = 500
    config.train.eval_early_every, config.train.eval_early_until = 100, 1500
    config.train.eval_batches = 19
    return config


def test_projection_separates_once_only_costs_from_validation_and_batch_work():
    config = projection_config()
    timing = timing_profile()
    result = preflight.project_runtime(timing, config)
    assert result["counts"] == {"training_updates": 7570, "validation_events": 28,
                                "batches_per_validation": 19, "checkpoint_events": 28}
    assert result["components_seconds"] == {
        "pilot_overhead": 600, "training": 2 * 7570, "diagnostics": 3 * 28,
        "evaluation": 5 * 19 * 28, "checkpointing": 7 * 28,
    }
    assert result["projected_seconds"] == 600 + 2 * 7570 + 3 * 28 + 5 * 19 * 28 + 7 * 28
    assert result["reserve_seconds"] == 7200
    # Cold evaluation combines compilation and diagnostics, so it cannot be scaled
    # like warmed batches. Increasing actual once-only pilot cost adds exactly once.
    timing["evaluation_seconds"] = [1e9]
    assert preflight.project_runtime(timing, config) == result
    timing["runtime_seconds"] += 17
    timing["diagnostic_warmup_seconds"] += 17
    assert preflight.project_runtime(timing, config)["projected_seconds"] == result["projected_seconds"] + 17


def test_more_validation_batches_only_scale_warm_batch_work():
    config = projection_config()
    full = preflight.project_runtime(timing_profile(), config)
    config.train.eval_batches = 1
    small = preflight.project_runtime(timing_profile(), config)
    assert full["projected_seconds"] - small["projected_seconds"] == 18 * 28 * 5
    for name in ("pilot_overhead", "training", "diagnostics", "checkpointing"):
        assert full["components_seconds"][name] == small["components_seconds"][name]


@pytest.mark.parametrize("name", ("train_step_seconds", "diagnostic_seconds", "evaluation_batch_seconds",
                                  "checkpoint_seconds"))
@pytest.mark.parametrize("bad", (None, [], [0.0], [-1.0], [float("nan")], [float("inf")], [True], ["2"]))
def test_projection_rejects_missing_or_invalid_measurements(name, bad):
    timing = timing_profile()
    timing[name] = bad
    with pytest.raises(RuntimeError, match=name):
        preflight.project_runtime(timing, projection_config())


def test_projection_rejects_cold_training_sample_without_warm_updates():
    timing = timing_profile()
    timing["train_step_seconds"] = [100.0]
    with pytest.raises(RuntimeError, match="warm training updates"):
        preflight.project_runtime(timing, projection_config())


@pytest.mark.parametrize("count", (1, 2))
def test_projection_rejects_incomplete_warm_batch_profile(count):
    timing = timing_profile()
    timing["evaluation_batch_seconds"] = [1.0] * count
    with pytest.raises(RuntimeError, match="at least three"):
        preflight.project_runtime(timing, projection_config())


@pytest.mark.parametrize("bad", (None, 0, -1, float("nan"), float("inf"), True, "600"))
def test_projection_rejects_invalid_once_only_cost(bad):
    timing = timing_profile()
    timing["runtime_seconds"] = bad
    with pytest.raises(RuntimeError, match="runtime_seconds"):
        preflight.project_runtime(timing, projection_config())


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


@pytest.mark.parametrize("continuation", (False, True))
def test_query_temperature_acceptance_requires_gate_and_decay_checks(continuation):
    config = {"model": {"qk_norm_token_scale": "cosine"},
              "continuation": {"parent_checkpoint": "parent" if continuation else ""}}
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    paths = preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=True)
    assert "scaffolds/transformer/tests/test_token_query_scale.py" in paths
    assert "scaffolds/transformer/tests/test_token_scale_weight_decay.py" in paths
    assert ("scaffolds/transformer/tests/test_token_scale_continuation.py" in paths) == continuation
    assert preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=False) == ()


def test_fox_continuation_requires_backbone_restoration_and_projection_decay_checks():
    config = {"model": {"attention_forget_gate": True, "positional_encoding": "none"},
              "continuation": {"parent_checkpoint": "parent"}}
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    paths = preflight.test_paths(spec, config, resolved_backend="blocked", cpu=True)
    assert "scaffolds/transformer/tests/test_forget_gate_continuation.py" in paths
    assert "scaffolds/transformer/tests/test_continuation.py" in paths
    assert "scaffolds/transformer/tests/test_continuation_acceptance.py" in paths


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
    gpu_calls = [command for command, options in calls if options["env"]["JAX_PLATFORMS"] != "cpu"]
    assert len(gpu_calls) == 1 and preflight.NORMALIZED_POLYNOMIAL_TESTS in gpu_calls[0]
    # The GPU stage runs on CUDA with the host CPU listed after it, for the kernel tests' FP64 references;
    # the CPU stages see the CPU alone.
    assert [options["env"]["JAX_PLATFORMS"] for _, options in calls] == ["cpu"] * (len(calls) - 1) + ["cuda,cpu"]


# This host may have no GPU: the cuda platform gets a CPU client, registered exactly as JAX registers each
# platform JAX_PLATFORMS lists (the GPU test stage's list, or a single "cuda"). The probe then asks for the host
# CPU as test_mosaic_score_bias's FP64 reference does.
_PLATFORM_PROBE = """
import json
from jax._src import hardware_utils, xla_bridge
hardware_utils.has_visible_nvidia_gpu = lambda: True
initialize = xla_bridge._init_backend
xla_bridge._init_backend = lambda platform: initialize("cpu" if platform == "cuda" else platform)
import jax
import jax.numpy as jnp
backends = xla_bridge.backends()
probe = {"platforms": list(backends), "cuda_is_default": xla_bridge.get_backend() is backends.get("cuda")}
try:
    host = jax.devices("cpu")[0]
    with jax.enable_x64(), jax.default_device(host):
        probe["host_dtype"] = str(jax.jit(jnp.sum)(jax.device_put(jnp.arange(3.0, dtype=jnp.float64), host)).dtype)
except RuntimeError as error:
    probe["error"] = str(error)
print(json.dumps(probe))
"""


def test_gpu_test_stage_lists_the_host_cpu_after_cuda_for_float64_references():
    """With JAX_PLATFORMS=cuda alone, jax.devices("cpu") raises "Unknown backend cpu" in the GPU stage, so every
    GPU test whose FP64 reference runs on the host (test_mosaic_score_bias.py) would fail in production
    acceptance; the stage's platform list keeps CUDA the default and gives those references the CPU."""
    import json
    import os
    import subprocess
    import sys

    def probe(platforms):
        result = subprocess.run([sys.executable, "-c", _PLATFORM_PROBE], capture_output=True, text=True, check=True,
                                env=dict(os.environ, JAX_PLATFORMS=platforms))
        return json.loads(result.stdout.strip().splitlines()[-1])

    assert probe(preflight.GPU_TEST_PLATFORMS) == {"platforms": ["cuda", "cpu"], "cuda_is_default": True,
                                                   "host_dtype": "float64"}
    alone = probe("cuda")
    assert alone["platforms"] == ["cuda"] and alone["cuda_is_default"]
    assert "Unknown backend cpu" in alone["error"]


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


def test_returning_to_the_passed_gpu_model_reuses_its_pilot_after_the_checkpoint_was_removed(tmp_path, monkeypatch,
                                                                                             capsys):
    """H200 and H200 NVL share one GPU type. An attempt on the other model re-records the test stages;
    back on the first model, its pilot and resume still pass, and their checkpoint is already gone."""
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config, config_to_dict

    allocated = {"model": "NVIDIA H200"}

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "fused", "device_kinds": [allocated["model"]] * args.gpu_count}
        raise RuntimeError(f"ran {name}")

    monkeypatch.setattr(preflight, "_run_worker", worker)
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    experiment = {"identity": "launcher", "source_commit": "a" * 40, "config": config_to_dict(Config()),
                  "gpu_type": "H200", "dependency_lock_sha256": "b" * 64, "runtime": {"TRANSFORMER_HOST_STAGING": "0"}}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "recipe.yaml", gpu_count=1, gpu_type="H200")
    first = preflight.on_devices(experiment, ["NVIDIA H200"])
    for name in (*preflight.CHECKS, "success"):
        preflight.record(tmp_path / f"{name}.json", first)
    for name, _, _ in preflight.TEST_STAGES:
        preflight.record(tmp_path / f"{name}.json", preflight.on_devices(experiment, ["NVIDIA H200 NVL"]))
    assert not (tmp_path / "pilot").exists()
    preflight._run(spec, args, experiment)
    output = capsys.readouterr().out
    assert "matching_acceptance_passed=" not in output and "acceptance_success=" in output
    assert all(preflight.passed(tmp_path / f"{name}.json", first) for name in (*preflight.CHECKS, "success"))


def test_runtime_identity_requires_resolved_transfer_policy_and_records_toolkit(monkeypatch):
    monkeypatch.setenv("TRANSFORMER_HOST_STAGING", "auto")
    with pytest.raises(ValueError, match="resolved"):
        preflight.runtime_identity()
    monkeypatch.setenv("TRANSFORMER_HOST_STAGING", "0")
    monkeypatch.setenv("CUDA_HOME", "/locked/nvidia/cuda_nvcc")
    assert preflight.runtime_identity()["CUDA_HOME"] == "/locked/nvidia/cuda_nvcc"


def test_runtime_allocation_budget_overrides_the_adapter_fallback(monkeypatch):
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), allocation_hours=48)
    monkeypatch.delenv("TRANSFORMER_ALLOCATION_SECONDS", raising=False)
    assert preflight.allocation_seconds(spec) == 48 * 3600
    monkeypatch.setenv("TRANSFORMER_ALLOCATION_SECONDS", str(7 * 24 * 3600))
    assert preflight.allocation_seconds(spec) == 7 * 24 * 3600
    # The same long single-GPU run fits the requested week, but not the old fallback.
    preflight.require_allocation_budget(65 * 3600, preflight.allocation_seconds(spec))
    monkeypatch.setenv("TRANSFORMER_ALLOCATION_SECONDS", str(48 * 3600))
    with pytest.raises(RuntimeError, match="48h allocation"):
        preflight.require_allocation_budget(65 * 3600, preflight.allocation_seconds(spec))
    # A backfilled allocation must use its actual shorter limit, not the requested one.
    monkeypatch.setenv("TRANSFORMER_ALLOCATION_SECONDS", str(6 * 3600))
    with pytest.raises(RuntimeError, match="6h allocation"):
        preflight.require_allocation_budget(5 * 3600, preflight.allocation_seconds(spec))


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "nan", "inf", "UNLIMITED", " 3600", "٣٦٠٠"])
def test_invalid_runtime_allocation_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv("TRANSFORMER_ALLOCATION_SECONDS", value)
    with pytest.raises(ValueError, match="positive integer"):
        preflight.allocation_seconds(preflight.AcceptanceSpec(entry_module="unused", cpu_tests=()))


def test_acceptance_budget_changes_invalidate_passed_gate_but_preserve_test_cache(tmp_path, monkeypatch):
    import json
    from scaffolds.transformer.configs import Config

    config = Config()
    config.train.source_commit = "auto"
    monkeypatch.setattr(preflight, "load_config", lambda path: config)
    monkeypatch.setattr(preflight, "sha256", lambda path: "fixture-sha256")
    monkeypatch.setenv("SHARED_TRANSFORMER_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("TRANSFORMER_ALLOCATION_SECONDS", str(168 * 3600))
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=())
    original = preflight.on_devices(preflight.identity(spec, tmp_path / "recipe.yaml", 1, "H100"), ["H100"])
    cache = tmp_path / "success.json"
    preflight.record(cache, original)
    assert json.loads(cache.read_text())["allocation_seconds"] == 168 * 3600
    assert preflight.passed(cache, original)
    monkeypatch.setenv("TRANSFORMER_ALLOCATION_SECONDS", str(48 * 3600))
    changed = preflight.on_devices(preflight.identity(spec, tmp_path / "recipe.yaml", 1, "H100"), ["H100"])
    assert changed["allocation_seconds"] == 48 * 3600
    assert changed["config"] == original["config"]
    assert not preflight.passed(cache, changed)
    assert preflight.stage_identity(original, "gpu-tests", ["kernel.py"], ["-m", "gpu"], cpu=False) == (
        preflight.stage_identity(changed, "gpu-tests", ["kernel.py"], ["-m", "gpu"], cpu=False))


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


def test_test_processes_accept_the_mosaic_precision_option(pytestconfig):
    assert pytestconfig.getoption("--mosaic-precision") in (None, "bf16", "bf16x3")


@pytest.mark.parametrize("model,backend,cpu,selected", [
    ({"attention_blocked_implementation": "strips"}, "blocked", True, True),
    ({"attention_blocked_implementation": "strips", "attention_forget_gate": True}, "blocked", False, False),
    ({"attention_blocked_implementation": "strips", "attention_degree": 2}, "fused", True, False),
    ({"attention_blocked_implementation": "scan"}, "blocked", True, False),
    ({}, "blocked", True, False),
])
def test_recipes_training_on_blocked_strips_require_their_cpu_tests(model, backend, cpu, selected):
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    paths = preflight.test_paths(spec, {"model": model}, resolved_backend=backend, cpu=cpu)
    assert (preflight.BLOCKED_STRIPS_TESTS in paths) == selected


@pytest.mark.parametrize("model,cpu,selected", [
    ({"attention_kernel_basis": "bspline", "attention_blocked_implementation": "strips"}, True, True),
    ({"attention_kernel_basis": "bspline", "attention_blocked_implementation": "scan"}, True, True),
    ({"attention_kernel_basis": "bspline", "attention_blocked_implementation": "strips"}, False, True),
    ({"attention_kernel_basis": "polynomial"}, True, False),
    ({"attention_kernel_basis": "polynomial"}, False, False),
    ({}, True, False),
])
def test_spline_recipes_require_their_tests(model, cpu, selected):
    """Both stages: the CPU checks, and the GPU stage's production precision."""
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    paths = preflight.test_paths(spec, {"model": model}, resolved_backend="blocked", cpu=cpu)
    assert (preflight.SPLINE_ATTENTION_TESTS in paths) == selected


def test_spline_production_recipe_selects_spline_strips_and_query_tests():
    """The query spline recipe's shared acceptance runs the spline, strips, and generator tests on
    CPU, and the spline's production precision on the GPU."""
    from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
    from scaffolds.transformer.tests.test_spline_attention import QUERY_CONTROL, spline_model

    config = Config(model=ModelConfig(**spline_model(QUERY_CONTROL)))
    config.data.sequence_length = 2049
    spec = preflight.attention_spec(config, entry_module="unused")
    paths = preflight.test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=True)
    for required in (preflight.SPLINE_ATTENTION_TESTS, preflight.BLOCKED_STRIPS_TESTS,
                     preflight.INPUT_DEPENDENT_COEFFICIENT_TESTS, preflight.QUERY_TANH_POLYNOMIAL_TESTS):
        assert required in paths
    assert spec.required_backend == "blocked"
    gpu = preflight.test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=False)
    assert preflight.SPLINE_ATTENTION_TESTS in gpu and preflight.BLOCKED_STRIPS_TESTS not in gpu


def _score_bias_model(mode, gated):
    model = {"attention_degree": 8, "attention_normalization": False, "attention_coefficients": "learned",
             "attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh",
             "attention_score_bias": mode}
    if gated:
        model.update(attention_forget_gate=True, attention_forget_gate_mode="query_distance",
                     attention_forget_gate_conditioning="query_key", attention_forget_gate_per_order=True,
                     positional_encoding="none")
    return model


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("mode", ("static", "query", "query_key"))
def test_score_bias_recipes_require_their_checks_beside_the_backbones(mode, gated):
    """On Mosaic, CPU: the score-bias reference, model and lowering checks and, with capture on, the
    gradient-event checks; GPU: the H100 kernels and attention_core's dispatch to them. The backbone keeps
    its query and per-order kernel checks."""
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    config = {"model": _score_bias_model(mode, gated), "train": {"gradient_event_capture": True}}
    cpu = preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=True)
    gpu = preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=False)
    assert {preflight.SCORE_BIAS_TESTS, preflight.SCORE_BIAS_LOWERING_TESTS, preflight.GRADIENT_EVENT_TESTS} <= set(cpu)
    assert preflight.MOSAIC_SCORE_BIAS_TESTS not in cpu and preflight.SCORE_BIAS_STRIPS_TESTS not in cpu
    assert {preflight.MOSAIC_SCORE_BIAS_TESTS, preflight.SCORE_BIAS_TESTS} <= set(gpu)
    assert preflight.SCORE_BIAS_LOWERING_TESTS not in gpu and preflight.GRADIENT_EVENT_TESTS not in gpu
    assert preflight.SCORE_BIAS_STRIPS_TESTS not in gpu
    for paths in (cpu, gpu):
        assert preflight.QUERY_TANH_POLYNOMIAL_TESTS in paths
        assert (preflight.PER_ORDER_DECAY_TESTS in paths) is gated
    without = {"model": _score_bias_model("none", gated), "train": {"gradient_event_capture": False}}
    for on_cpu in (True, False):
        for backend in ("mosaic", "blocked"):
            paths = preflight.test_paths(spec, without, resolved_backend=backend, cpu=on_cpu)
            assert not {preflight.SCORE_BIAS_TESTS, preflight.SCORE_BIAS_LOWERING_TESTS,
                        preflight.MOSAIC_SCORE_BIAS_TESTS, preflight.SCORE_BIAS_STRIPS_TESTS,
                        preflight.GRADIENT_EVENT_TESTS} & set(paths)


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("mode", ("static", "query", "query_key"))
def test_blocked_score_bias_recipes_qualify_the_strips_and_no_kernel(mode, gated):
    """On the blocked strips, CPU: the score-bias reference and model checks, the strips' own checks, and
    the gradient-event checks, beside the backbone's CPU checks; no Mosaic module. GPU: the strips with
    score biases at production precision and nothing else, so that no other kernel's GPU case can skip
    or refuse on a GPU that kernel is not qualified on; an adapter's own GPU tests are kept."""
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    config = {"model": {**_score_bias_model(mode, gated), "attention_backend": "blocked",
                        "attention_blocked_implementation": "strips"},
              "train": {"gradient_event_capture": True}}
    cpu = preflight.test_paths(spec, config, resolved_backend="blocked", cpu=True)
    gpu = preflight.test_paths(spec, config, resolved_backend="blocked", cpu=False)
    assert {preflight.SCORE_BIAS_TESTS, preflight.BLOCKED_STRIPS_TESTS, preflight.GRADIENT_EVENT_TESTS} <= set(cpu)
    assert not {preflight.SCORE_BIAS_LOWERING_TESTS, preflight.MOSAIC_SCORE_BIAS_TESTS,
                preflight.SCORE_BIAS_STRIPS_TESTS} & set(cpu)
    assert preflight.QUERY_TANH_POLYNOMIAL_TESTS in cpu and (preflight.PER_ORDER_DECAY_TESTS in cpu) is gated
    assert gpu == (preflight.SCORE_BIAS_STRIPS_TESTS,)
    own = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=("own_gpu_test.py",))
    assert preflight.test_paths(own, config, resolved_backend="blocked", cpu=False) == (
        "own_gpu_test.py", preflight.SCORE_BIAS_STRIPS_TESTS)
    assert preflight.mosaic_test_options(config, "blocked") == ()
    module = (preflight.ROOT / preflight.SCORE_BIAS_STRIPS_TESTS).read_text()
    assert "pytestmark = pytest.mark.gpu" in module and "mosaic" not in module.lower()


@pytest.mark.parametrize("cpu", (True, False), ids=("cpu", "gpu"))
@pytest.mark.parametrize("implementation", ("scan", None), ids=("scan", "unset"))
@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
def test_blocked_score_bias_recipes_on_the_scan_are_refused(cpu, implementation, gated):
    """The scan, the score biases' reference and the field's default, has no GPU qualification for training
    them: both stages refuse such a recipe instead of qualifying the strips, which it would not run."""
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    model = {**_score_bias_model("query_key", gated), "attention_backend": "blocked"}
    if implementation is not None:
        model["attention_blocked_implementation"] = implementation
    with pytest.raises(ValueError, match="train on its causal strips.*attention_blocked_implementation: scan"):
        preflight.test_paths(spec, {"model": model, "train": {"gradient_event_capture": True}},
                             resolved_backend="blocked", cpu=cpu)


@pytest.mark.parametrize("capture", (False, True))
def test_gradient_event_capture_requires_its_cpu_tests(capture):
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    config = {"model": {}, "train": {"gradient_event_capture": capture}}
    cpu, gpu = (preflight.test_paths(spec, config, resolved_backend="blocked", cpu=on_cpu) for on_cpu in (True, False))
    assert (preflight.GRADIENT_EVENT_TESTS in cpu) is capture and preflight.GRADIENT_EVENT_TESTS not in gpu


@pytest.mark.parametrize("gated", (False, True), ids=("ungated", "per_order_gate"))
@pytest.mark.parametrize("mode,shift", (("static", "row"), ("query", "row"), ("query_key", "pair")))
def test_score_bias_gpu_tests_run_at_their_kernels_precision(monkeypatch, mode, shift, gated):
    """The precision comes from the score-bias kernel family that trains (gate and shift), as its entry points
    choose it."""
    from scaffolds.transformer.attention.backends.mosaic import api, score_bias

    config = {"model": _score_bias_model(mode, gated)}
    assert preflight.mosaic_test_options(config, "mosaic") == ("--mosaic-precision", "bf16")
    asked = []
    monkeypatch.setattr(api, "default_precision", lambda family, degree: asked.append((family, degree)) or "bf16x3")
    assert preflight.mosaic_test_options(config, "mosaic") == ("--mosaic-precision", "bf16x3")
    assert asked == [(score_bias.score_bias_family("endpoint" if gated else "none", shift), 8)]


def _score_transform_model(transform, **changes):
    return {"attention_degree": 8, "attention_normalization": False, "attention_coefficients": "learned",
            "attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh",
            "attention_backend": "blocked", "attention_blocked_implementation": "strips",
            "attention_score_transform": transform, **changes}


@pytest.mark.parametrize("start", (None, 0.0))
@pytest.mark.parametrize("transform", ("softplus", "abs"))
def test_score_transform_recipes_qualify_the_strips_and_no_kernel(transform, start):
    """CPU: the transform's FP64 references, start, refusals and diagnostics, the strips' own checks, and the
    gradient-event checks, beside the backbone's CPU checks. GPU: the strips at production precision, with the
    transform and (their existing qualification) with score biases, and nothing else, so that no other kernel's
    GPU case can skip or refuse on a GPU that kernel is not qualified on; an adapter's own GPU tests are kept."""
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    config = {"model": _score_transform_model(transform, attention_coefficient_constant_initial=start),
              "train": {"gradient_event_capture": True}}
    cpu = preflight.test_paths(spec, config, resolved_backend="blocked", cpu=True)
    gpu = preflight.test_paths(spec, config, resolved_backend="blocked", cpu=False)
    assert {preflight.SCORE_TRANSFORM_TESTS, preflight.BLOCKED_STRIPS_TESTS, preflight.GRADIENT_EVENT_TESTS,
            preflight.QUERY_TANH_POLYNOMIAL_TESTS} <= set(cpu)
    assert not {preflight.SCORE_TRANSFORM_STRIPS_TESTS, preflight.SCORE_BIAS_STRIPS_TESTS,
                preflight.SCORE_BIAS_TESTS} & set(cpu)
    assert gpu == (preflight.SCORE_BIAS_STRIPS_TESTS, preflight.SCORE_TRANSFORM_STRIPS_TESTS)
    own = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=("own_gpu_test.py",))
    assert preflight.test_paths(own, config, resolved_backend="blocked", cpu=False) == (
        "own_gpu_test.py", preflight.SCORE_BIAS_STRIPS_TESTS, preflight.SCORE_TRANSFORM_STRIPS_TESTS)
    assert preflight.mosaic_test_options(config, "blocked") == ()
    module = (preflight.ROOT / preflight.SCORE_TRANSFORM_STRIPS_TESTS).read_text()
    assert "pytestmark = pytest.mark.gpu" in module and "mosaic" not in module.lower()
    assert "pytest.mark.gpu" not in (preflight.ROOT / preflight.SCORE_TRANSFORM_TESTS).read_text()


@pytest.mark.parametrize("cpu", (True, False), ids=("cpu", "gpu"))
@pytest.mark.parametrize("backend,implementation", (
    ("blocked", "scan"), ("blocked", None), ("fused", "strips"), ("cudnn", None)))
def test_score_transform_recipes_off_the_strips_and_kernels_are_refused(cpu, backend, implementation):
    """The scan, the transform's reference and the field's default, has no GPU qualification for training it,
    and no backend but the blocked one and Mosaic evaluates it: both stages refuse instead of qualifying
    computations the run would not train on."""
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    model = _score_transform_model("softplus", attention_backend=backend)
    if implementation is None:
        del model["attention_blocked_implementation"]
    else:
        model["attention_blocked_implementation"] = implementation
    with pytest.raises(ValueError, match="score transform trains on the Mosaic transformed query-polynomial kernels "
                                         "or the blocked backend's causal strips"):
        preflight.test_paths(spec, {"model": model}, resolved_backend=backend, cpu=cpu)


@pytest.mark.parametrize("start", (None, 0.0))
@pytest.mark.parametrize("transform", ("softplus", "abs"))
def test_score_transform_recipes_on_mosaic_qualify_the_transformed_kernels(transform, start):
    """CPU: the transform's FP64 references, start, refusals and diagnostics, and the transformed kernels' tile
    mathematics, shared memory and lowering, beside the backbone's CPU checks; never the strips' checks. GPU: the
    transformed kernels against FP64 and the strips, beside the query kernels' own cases (the backbone's
    selection, not part of this spec), at the kernels' precision; an adapter's own GPU tests are kept."""
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=("own_gpu_test.py",))
    model = _score_transform_model(transform, attention_backend="mosaic", attention_blocked_implementation="scan",
                                   attention_coefficient_constant_initial=start)
    config = {"model": model, "train": {"gradient_event_capture": True}}
    cpu = preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=True)
    gpu = preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=False)
    assert {preflight.SCORE_TRANSFORM_TESTS, preflight.SCORE_TRANSFORM_LOWERING_TESTS, preflight.GRADIENT_EVENT_TESTS,
            preflight.QUERY_TANH_POLYNOMIAL_TESTS} <= set(cpu)
    assert not {preflight.MOSAIC_SCORE_TRANSFORM_TESTS, preflight.SCORE_TRANSFORM_STRIPS_TESTS,
                preflight.BLOCKED_STRIPS_TESTS, preflight.SCORE_BIAS_STRIPS_TESTS} & set(cpu)
    assert gpu[0] == "own_gpu_test.py" and preflight.MOSAIC_SCORE_TRANSFORM_TESTS in gpu
    assert not {preflight.SCORE_TRANSFORM_TESTS, preflight.SCORE_TRANSFORM_LOWERING_TESTS,
                preflight.SCORE_TRANSFORM_STRIPS_TESTS, preflight.SCORE_BIAS_STRIPS_TESTS} & set(gpu)
    selected = preflight.attention_test_paths(config, "mosaic", False)
    assert {"scaffolds/transformer/tests/test_mosaic_query.py",
            "scaffolds/transformer/tests/test_mosaic_repeatability.py"} <= set(selected)
    assert preflight.mosaic_test_options(config, "mosaic") == ("--mosaic-precision", "bf16")
    for path in (preflight.MOSAIC_SCORE_TRANSFORM_TESTS, preflight.SCORE_TRANSFORM_LOWERING_TESTS):
        module = (preflight.ROOT / path).read_text()
        assert ("pytestmark = pytest.mark.gpu" in module) is (path == preflight.MOSAIC_SCORE_TRANSFORM_TESTS)


@pytest.mark.parametrize("backend", ("blocked", "mosaic"))
def test_constant_starts_alone_select_their_cpu_checks(backend):
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(), gpu_tests=())
    model = {**_score_transform_model("identity"), "attention_backend": backend}
    for start, selected in ((None, False), (0.0, True)):
        config = {"model": {**model, "attention_coefficient_constant_initial": start}}
        assert (preflight.SCORE_TRANSFORM_TESTS in preflight.test_paths(spec, config, resolved_backend=backend,
                                                                         cpu=True)) is selected
        gpu = preflight.test_paths(spec, config, resolved_backend=backend, cpu=False)
        assert preflight.SCORE_TRANSFORM_TESTS not in gpu and preflight.SCORE_TRANSFORM_STRIPS_TESTS not in gpu


@pytest.mark.parametrize("hook_result", ("absent", "pass", "fail"))
def test_experiment_pilot_check_precedes_success_and_records_its_facts(tmp_path, monkeypatch, hook_result):
    import json
    from types import SimpleNamespace
    from scaffolds.transformer.configs import Config, config_to_dict

    config = Config()
    experiment = {"identity": "launcher", "source_commit": "a" * 40, "config": config_to_dict(config),
                  "gpu_type": "H100", "dependency_lock_sha256": "b" * 64, "runtime": {},
                  "allocation_seconds": 24 * 3600}
    args = SimpleNamespace(directory=tmp_path, config=tmp_path / "recipe.yaml", gpu_count=1, gpu_type="H100")
    on_device = preflight.on_devices(experiment, ["NVIDIA H100"])
    for name in ("cpu-tests", "cpu-distributed-tests", "gpu-tests", "initialization"):
        preflight.record(tmp_path / f"{name}.json", on_device)
    validation = {"step": 20, "validation_nll": 2.0, "attention_diagnostics": {"valid": True,
                  "coefficients": {"effective": [1.001]}}}
    outcome = {"final_step": 20, "data_parallel_devices": 1, "start_step": 0,
               "attention_backend": "blocked", "timing": {}}
    calls = []

    def worker(spec, args, name, output, **unused):
        if name == "hardware":
            return {"attention_backend": "blocked", "device_kinds": ["NVIDIA H100"]}
        if name == "pilot" and output.name == "pilot-result.json":
            (tmp_path / "pilot").mkdir()
            (tmp_path / "pilot/metrics.jsonl").write_text(json.dumps(validation) + "\n")
            return dict(outcome)
        raise RuntimeError("stop at resume")

    def check_pilot(loaded, observed_outcome, observed_validation):
        assert loaded is config
        assert observed_outcome == outcome and observed_validation == validation
        assert not (tmp_path / "pilot.json").exists()
        calls.append("check_pilot")
        if hook_result == "fail":
            raise RuntimeError("coefficient pilot failed")
        return {"coefficient_delta": 0.001}

    monkeypatch.setattr(preflight, "_run_worker", worker)
    monkeypatch.setattr(preflight, "load_config", lambda path: config)
    monkeypatch.setattr(preflight, "project_runtime", lambda timing, recipe: {"projected_seconds": 60})
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=(),
                                    check_pilot=None if hook_result == "absent" else check_pilot)
    with pytest.raises(RuntimeError, match="coefficient pilot failed" if hook_result == "fail" else "stop at resume"):
        preflight._run(spec, args, experiment)
    assert calls == ([] if hook_result == "absent" else ["check_pilot"])
    if hook_result == "fail":
        assert not (tmp_path / "pilot.json").exists()
        assert json.loads((tmp_path / "failure.json").read_text())["stage"] == "pilot"
    else:
        result = json.loads((tmp_path / "pilot.json").read_text())
        assert preflight.passed(tmp_path / "pilot.json", on_device)
        if hook_result == "pass":
            assert result["outcome"]["experiment_check"] == {"coefficient_delta": 0.001}
        else:
            assert "experiment_check" not in result["outcome"]


@pytest.mark.parametrize("transform", ("sigmoid", "softplus"))
def test_fixed_pointwise_mosaic_selects_its_full_qualification_without_polynomial_grids(transform):
    model = {"attention_mechanism": "dot_product", "attention_backend": "mosaic",
             "attention_score_transform": transform, "attention_degree": 1,
             "attention_coefficients": "fixed", "attention_fixed_coefficients": [0.0, 1.0],
             "attention_coefficient_conditioning": "static", "attention_normalization": False,
             "attention_output_scaling": "head_rms_norm", "qk_norm": True, "qk_norm_head_scale": True}
    config = {"model": model}
    spec = preflight.AcceptanceSpec(entry_module="unused", cpu_tests=("common_cpu.py",),
                                   gpu_tests=("own_gpu.py",), select_tests=preflight.attention_test_paths)
    cpu = preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=True)
    assert "common_cpu.py" in cpu
    assert preflight.QK_HEAD_SCALE_TESTS in cpu
    assert preflight.SCORE_TRANSFORM_TESTS in cpu
    gpu = preflight.test_paths(spec, config, resolved_backend="mosaic", cpu=False,
                               runtime_test_paths=("runtime_gpu.py",))
    assert set(gpu) == {"own_gpu.py", "runtime_gpu.py", preflight.POINTWISE_ATTENTION_TESTS,
                        preflight.POINTWISE_SCORE_TRANSFORM_TESTS}
    # Removing the transform keeps the existing fixed-polynomial grid exactly.
    model["attention_score_transform"] = "identity"
    ordinary = preflight.attention_test_paths(config, "mosaic", False)
    assert ordinary == ("scaffolds/transformer/tests/test_mosaic_unnormalized.py",
                        "scaffolds/transformer/tests/test_mosaic_repeatability.py")
