"""Recipe and initialization controls for the narrow-key runs: each partner's recipe with Q/K width 4, 8 or 16."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import key_width_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict

KEYS = tuple(adapter.PARTNERS)
# Every (partner, width) a run takes, and every other width, which the partner refuses.
CASES = tuple((key, width) for key in KEYS for width in adapter.partner_widths(key))
REFUSED = tuple((key, width) for key in KEYS for width in (2, 4, 8, 16, 32, 64, 128)
                if width not in adapter.partner_widths(key))
GPUS = ("H100", "B200", "B300")
TESTS = "scaffolds/transformer/tests"


def variant(key, width, **model):
    config = adapter.partner_config(key)
    config.model.qk_head_dim = width
    for name, value in model.items():
        setattr(config.model, name, value)
    for name, value in adapter.PARTNERS[key]["train"].items():
        setattr(config.train, name, value)
    return config


def first_width(key):
    return adapter.partner_widths(key)[0]


def scientific_differences(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for recipe in (first, second):
        for name in adapter.EXECUTION_KEYS:
            recipe["train"].pop(name, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


def run_config(run_id):
    from scaffolds.transformer.configs import load_config

    return load_config(adapter.RUNS / run_id / "config.yaml")


@pytest.mark.parametrize("key", KEYS)
def test_pinned_partners_are_the_recorded_runs(key):
    partner = adapter.PARTNERS[key]
    path = adapter.RUNS / partner["run_id"] / "config.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == partner["config_sha256"]
    record = json.loads((adapter.RUNS / partner["run_id"] / "record.json").read_text())
    assert record["config"]["sha256"] == partner["config_sha256"]
    config = adapter.partner_config(key)
    assert (config.model.num_layers, config.model.hidden_size, config.optim.total_steps) == (15, 896, 7570)
    assert (config.model.value_head_dim, config.train.seed) == (64, 42)
    assert config.model.qk_head_dim not in adapter.partner_widths(key)
    assert set(adapter.partner_widths(key)) <= set(adapter.WIDTHS)
    assert set(partner["gpu_types"]) == set(adapter.partner_widths(key))
    assert config.model.attention_backend == partner["backend"] and adapter.partner_key(config) == key


def test_the_polynomial_partners_are_the_ladder_on_the_raw_score():
    """Degree eight on the raw score: fixed Taylor, static learned and tanh query coefficients."""
    taylor, static, query = (adapter.partner_config(key).model for key in ("taylor", "static", "query"))
    for model in (taylor, static, query):
        assert (model.attention_degree, model.attention_normalization) == (8, False)
        assert (model.attention_score_transform, model.attention_score_prefactor) == ("identity", "none")
        assert (model.attention_coefficient_parameterization, model.attention_output_scaling) == (
            "direct", "head_rms_norm")
        assert not model.attention_forget_gate and model.attention_coefficient_input_power == 0
    assert (taylor.attention_coefficients, taylor.attention_coefficient_conditioning) == ("fixed", "static")
    assert (static.attention_coefficients, static.attention_coefficient_conditioning) == ("learned", "static")
    assert (query.attention_coefficients, query.attention_coefficient_conditioning) == ("learned", "query")
    assert query.attention_coefficient_activation == "tanh" and query.attention_coefficient_kernel_weight_decay
    assert query.qk_head_dim == 8


@pytest.mark.parametrize("key,width", CASES)
def test_only_the_key_width_and_capture_change(key, width):
    config = variant(key, width)
    adapter.check_recipe(config)
    expected = {"model.qk_head_dim", *(("train.gradient_event_capture",) if key == "exponential" else ())}
    assert scientific_differences(config, adapter.partner_config(key)) == expected
    assert config.model.value_head_dim == 64


@pytest.mark.parametrize("key,width", REFUSED)
def test_other_key_widths_are_refused(key, width):
    config = variant(key, first_width(key))
    config.model.qk_head_dim = width
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("section,name,value", (
    ("model", "value_head_dim", 16), ("model", "num_heads", 7), ("optim", "learning_rate", 0.002),
    ("train", "seed", 43), ("model", "attention_backend", "blocked"), ("data", "batch_size", 128),
))
@pytest.mark.parametrize("key", KEYS)
def test_unrelated_changes_are_refused(key, section, name, value):
    config = variant(key, first_width(key))
    setattr(getattr(config, section), name, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("key,name,value", (
    ("taylor", "attention_coefficients", "learned"), ("static", "attention_coefficients", "fixed"),
    ("static", "attention_coefficient_conditioning", "query"), ("query", "attention_coefficient_conditioning", "static"),
    ("query", "attention_coefficient_kernel_weight_decay", False), ("static", "attention_coefficient_sharing", "global"),
    ("taylor", "attention_degree", 4), ("static", "attention_degree", 12), ("query", "attention_degree", 12),
))
def test_another_polynomial_setting_is_refused(key, name, value):
    """A changed coefficient setting selects another partner, whose recipe then differs, or its own, which refuses it."""
    config = variant(key, 4)
    assert getattr(config.model, name) != value
    setattr(config.model, name, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("width", adapter.WIDTHS)
def test_the_exponential_may_shift_by_its_row_maximum(width):
    config = variant("exponential", width, attention_exp_shift="row_max")
    adapter.check_recipe(config)
    assert scientific_differences(config, adapter.partner_config("exponential")) == {
        "model.qk_head_dim", "train.gradient_event_capture", "model.attention_exp_shift"}


@pytest.mark.parametrize("key", tuple(key for key in KEYS if key != "exponential"))
def test_only_the_exponential_takes_the_shift(key):
    config = variant(key, first_width(key))
    config.model.attention_exp_shift = "row_max"
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_softmax_is_the_normalized_partner_on_cudnn():
    config = variant("softmax", 4)
    assert adapter.partner_key(config) == "softmax" and config.model.attention_normalization
    assert config.model.attention_backend == "cudnn" and config.model.attention_output_scaling == "none"
    normalized_exponential = variant("exponential", 4)
    normalized_exponential.model.attention_normalization = True
    with pytest.raises(ValueError):
        adapter.check_recipe(normalized_exponential)


def test_the_exponential_runs_take_the_capture_policy():
    config = variant("exponential", 8)
    config.train.gradient_event_capture = False
    with pytest.raises(ValueError, match="train.gradient_event_capture"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("key", ("taylor", "static", "query"))
def test_the_polynomial_runs_add_no_capture(key):
    config = variant(key, 4)
    config.train.gradient_event_capture = True
    with pytest.raises(ValueError, match="train.gradient_event_capture"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("key", KEYS)
def test_initialization_differs_only_in_the_reshaped_arrays_on_a_small_backbone(key):
    """Every partner array of unchanged shape is bit-identical at width 4, and Q, K and their gains change."""
    def small(config):
        return replace(config, model=replace(config.model, num_layers=2, hidden_size=64, num_heads=2, num_kv_heads=2,
                                             intermediate_size=128, vocab_size=64))

    facts = adapter.compare_with_partner(small(variant(key, 4)), small(adapter.partner_config(key)))
    assert facts["all_other_partner_arrays_identical"]
    reshaped = facts["reshaped_parameters"]
    assert any("q_proj" in path for path in reshaped) and any("k_proj" in path for path in reshaped)
    assert any("qk_norm_q_gain" in path for path in reshaped) and any("qk_norm_k_gain" in path for path in reshaped)
    assert not any("v_proj" in path or "o_proj" in path for path in reshaped)


@pytest.mark.parametrize("key,width", CASES)
def test_initialization_gate_uses_the_pinned_fingerprint(monkeypatch, key, width):
    count, sha256 = adapter.PARTNERS[key]["initialization"][width]
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda _: {})
    monkeypatch.setattr(adapter, "compare_with_partner", lambda *_: {"all_other_partner_arrays_identical": True})
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {"parameter_count": count,
                                                                    "parameter_sha256": sha256})
    facts = adapter.check_initialization(variant(key, width))
    assert facts["pinned_initialization_matches"] and facts["qk_head_dim"] == width
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {"parameter_count": count,
                                                                    "parameter_sha256": "changed"})
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(variant(key, width))


def test_pinned_counts_are_the_model_formula():
    from scaffolds.transformer.models import model_parameter_count

    for key in KEYS:
        for width, (count, _) in adapter.PARTNERS[key]["initialization"].items():
            assert model_parameter_count(variant(key, width).model) == count


def test_shared_pins_are_the_same_initial_arrays():
    """Fixed Taylor adds no parameter to the exponential's backbone, and at width 4 the query polynomial starts from
    softplus-SoS's arrays, so their pins agree; the static coefficients add one vector per head and order."""
    pins = {key: partner["initialization"] for key, partner in adapter.PARTNERS.items()}
    assert pins["taylor"] == pins["exponential"]
    assert pins["query"][4] == pins["softplus_sos"][4]
    for width in adapter.WIDTHS:
        assert pins["static"][width][0] - pins["taylor"][width][0] == 15 * 14 * 9


@pytest.mark.parametrize("key", KEYS)
def test_acceptance_selects_each_partners_kernels(key):
    config = variant(key, 4)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == adapter.PARTNERS[key]["backend"]
    assert set(adapter.gpu_tests(config)) <= set(spec.gpu_tests)
    assert set(adapter.CPU_TESTS) <= set(spec.cpu_tests)
    assert spec.check_initialization is adapter.check_initialization


def test_gpus_are_the_qualified_ones():
    """Mosaic's query-coefficient kernel: width 4 on H100 and B200, widths 8 and 16 on H100 and B300."""
    every = adapter.every_width
    assert {key: partner["gpu_types"] for key, partner in adapter.PARTNERS.items()} == {
        "softplus_sos": every(("H100", "B200")), "exponential": every(("H100", "B300")),
        "softmax": every(("H100", "B300")),
        "taylor": {4: ("H100", "B200"), 8: ("H100", "B300"), 16: ("H100", "B300")},
        "static": {4: ("H100", "B200"), 8: ("H100", "B300"), 16: ("H100", "B300")},
        "query": {4: ("H100", "B200")},
    }


@pytest.mark.parametrize("key,width", CASES)
def test_each_partners_gpus_are_required(tmp_path, key, width):
    from scaffolds.transformer.configs import save_config

    path = tmp_path / f"{key}-{width}.yaml"
    save_config(variant(key, width), path)
    allowed = adapter.gpu_types(key, width)
    for gpu in GPUS:
        if gpu in allowed:
            adapter.require_gpu_type([str(path), "--gpu-type", gpu])
        else:
            with pytest.raises(ValueError, match="only"):
                adapter.require_gpu_type([str(path), "--gpu-type", gpu])
    adapter.require_gpu_type([str(path), "--runtime-env"])


def test_a_width_the_partner_does_not_take_has_no_gpus(tmp_path):
    from scaffolds.transformer.configs import save_config

    path = tmp_path / "query-8.yaml"
    save_config(adapter.partner_config("query"), path)
    with pytest.raises(ValueError, match="takes model.qk_head_dim"):
        adapter.require_gpu_type([str(path), "--gpu-type", "H100"])


def _test_module(name):
    path = Path(__file__).resolve().parents[3] / TESTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py").replace("test_", "cases_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def test_softmax_gpu_cases_exist_at_the_narrow_widths():
    """The selected cuDNN cases are GPU tests of test_softmax_backends.py at widths 4, 8 and 16 only."""
    path, module = _test_module("test_softmax_backends.py")
    widths = set()
    for test in adapter.GPU_TESTS["softmax"]:
        file, case = test.split("::")
        name, width = case.removesuffix("]").split("[")
        assert Path(file).name == path.name and int(width) in module.ATTENTION_DIMENSIONS
        assert any(mark.name == "gpu" for mark in getattr(module, name).pytestmark)
        widths.add(int(width))
    assert widths == set(adapter.WIDTHS)


@pytest.mark.parametrize("key,width", tuple(case for case in CASES if case[0] in adapter.MOSAIC_EQUAL_COEFFICIENT_KEYS))
def test_mosaic_query_gpu_cases_exist_at_the_runs_width(key, width):
    """Fixed and static runs take GPU tests of test_mosaic_query.py at their degree (eight) and width alone, at both
    weight precisions (acceptance keeps the run's)."""
    path, module = _test_module("test_mosaic_query.py")
    cases = set()
    for test in adapter.gpu_tests(variant(key, width)):
        file, case = test.split("::")
        name, params = case.removesuffix("]").split("[")
        degree, case_width, precision = params.split("-")
        assert Path(file).name == path.name and name == "test_outputs_and_vjp_match_fp64_and_triton"
        assert int(degree) in module.DEGREES and int(case_width) in module.WIDTHS and precision in module.PRECISIONS
        cases.add((int(degree), int(case_width), precision))
    assert module.pytestmark.name == "gpu"
    assert cases == {(8, width, precision) for precision in module.PRECISIONS}


def test_acceptance_stages_take_each_partners_cases():
    """The shift's own suite joins both stages of a shifted recipe; softmax's GPU stage takes the narrow widths only;
    fixed and static coefficients take the query kernel's cases at their own width beside their own Mosaic suite, and
    the query polynomial takes that kernel's whole suite."""
    from scaffolds.transformer.acceptance import test_paths as acceptance_test_paths

    shift = f"{TESTS}/test_exp_row_max_shift.py"
    shifted = variant("exponential", 4, attention_exp_shift="row_max")
    spec = adapter.experiment_spec(shifted, entry_module=adapter.__name__)
    for cpu in (True, False):
        assert shift in acceptance_test_paths(spec, config_to_dict(shifted), resolved_backend="fused", cpu=cpu)
    assert f"{TESTS}/test_unnormalized_fused_attention.py" in acceptance_test_paths(
        spec, config_to_dict(shifted), resolved_backend="fused", cpu=False)
    softmax = variant("softmax", 4)
    spec = adapter.experiment_spec(softmax, entry_module=adapter.__name__)
    gpu = acceptance_test_paths(spec, config_to_dict(softmax), resolved_backend="cudnn", cpu=False)
    assert set(adapter.SOFTMAX_WIDTH_TESTS) <= set(gpu) and shift not in gpu
    assert f"{TESTS}/test_softmax_backends.py" not in gpu
    for key in adapter.MOSAIC_EQUAL_COEFFICIENT_KEYS:
        for width in adapter.partner_widths(key):
            config = variant(key, width)
            spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
            gpu = acceptance_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
            query_cases = {path for path in gpu if path.startswith(f"{TESTS}/test_mosaic_query.py")}
            assert query_cases == set(adapter.mosaic_query_width_tests(8, width))
            assert f"{TESTS}/test_mosaic_unnormalized.py" in gpu and shift not in gpu
    query = variant("query", 4)
    spec = adapter.experiment_spec(query, entry_module=adapter.__name__)
    gpu = acceptance_test_paths(spec, config_to_dict(query), resolved_backend="mosaic", cpu=False)
    assert f"{TESTS}/test_mosaic_query.py" in gpu
    assert not any(path.startswith(f"{TESTS}/test_mosaic_query.py::") for path in gpu)


def test_an_edited_partner_recipe_is_refused(monkeypatch, tmp_path):
    run = adapter.PARTNERS["exponential"]["run_id"]
    edited = tmp_path / run / "config.yaml"
    edited.parent.mkdir(parents=True)
    edited.write_bytes((adapter.RUNS / run / "config.yaml").read_bytes() + b"# edited\n")
    config = deepcopy(variant("exponential", 8))
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    with pytest.raises(ValueError, match=f"Pinned key-width partner recipe {run} changed"):
        adapter.check_recipe(config)


# The allocated runs: label, partner, width and the model settings the run chooses.
SOS_LABEL = "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay"
SHIFTED = {"attention_exp_shift": "row_max"}
PRESCALED = {"attention_head_rms_prescale": "max_abs"}
RUNS = {
    "r5a25f500bfa8": (f"{SOS_LABEL}-dk4", "softplus_sos", 4, {}),
    "r8d6683db38a8": (f"{SOS_LABEL}-dk8", "softplus_sos", 8, {}),
    "r79abb4d7292c": (f"{SOS_LABEL}-dk16", "softplus_sos", 16, {}),
    "ra28b295d81e6": ("exp-head-scale-dk4", "exponential", 4, {}),
    "r535bc84a7772": ("exp-head-scale-dk8", "exponential", 8, {}),
    "r6f84b1db6de6": ("exp-head-scale-dk16", "exponential", 16, {}),
    "r2739c80cc032": ("exp-head-scale-row-max-dk4", "exponential", 4, SHIFTED),
    "reaca8ac9420c": ("softmax-head-scale-dk4", "softmax", 4, {}),
    "r6f65dc73550a": ("softmax-head-scale-dk8", "softmax", 8, {}),
    "r328466965c2b": ("softmax-head-scale-dk16", "softmax", 16, {}),
    "r2a45c9e3d452": ("exp-head-scale-row-max-dk8", "exponential", 8, SHIFTED),
    "rfdc104924c7c": ("exp-head-scale-row-max-dk16", "exponential", 16, SHIFTED),
    "re3515eed3576": ("fixed-m8-head-scale-dk4", "taylor", 4, {}),
    "r858c094cb85f": ("fixed-m8-head-scale-dk8", "taylor", 8, {}),
    "rcc9e0b03e692": ("fixed-m8-head-scale-dk16", "taylor", 16, {}),
    "r0f5e38b0d934": ("static-m8-head-scale-dk4", "static", 4, {}),
    "r161f07127f20": ("static-m8-head-scale-dk8", "static", 8, {}),
    "r576bb8758634": ("static-m8-head-scale-dk16", "static", 16, {}),
    "rd5e3ec5a4f02": ("query-m8-dk4-w-decay", "query", 4, {}),
    "rabfbab32a594": (f"{SOS_LABEL}-dk4-max-abs-head-rms", "softplus_sos", 4, PRESCALED),
    "r1eaa4a4c3c77": ("fixed-m8-head-scale-dk4-max-abs-head-rms", "taylor", 4, PRESCALED),
    "r5e540d8620b1": ("fixed-m8-head-scale-dk8-max-abs-head-rms", "taylor", 8, PRESCALED),
}


@pytest.mark.parametrize("run_id, plain", (("rabfbab32a594", "r5a25f500bfa8"), ("r1eaa4a4c3c77", "re3515eed3576"),
                                            ("r5e540d8620b1", "r858c094cb85f")))
def test_the_prescaled_runs_rerun_the_plain_ones(run_id, plain):
    prescaled = run_config(run_id)
    assert scientific_differences(prescaled, run_config(plain)) == {"model.attention_head_rms_prescale"}
    assert prescaled.model.attention_head_rms_prescale == "max_abs"


@pytest.mark.parametrize("key", ("exponential", "softmax"))
def test_only_the_polynomial_partners_take_the_prescale(key):
    config = variant(key, first_width(key))
    config.model.attention_head_rms_prescale = "max_abs"
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("run_id", RUNS)
def test_allocated_runs_are_their_partner_at_the_key_width(run_id):
    label, key, width, model = RUNS[run_id]
    path = adapter.RUNS / run_id / "config.yaml"
    config = run_config(run_id)
    adapter.check_recipe(config)
    assert adapter.partner_key(config) == key and config.model.qk_head_dim == width
    assert scientific_differences(config, variant(key, width, **model)) == set()
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_shifted_width_4_exponential_changes_only_the_shift():
    """r2739c80cc032 is ra28b295d81e6's recipe, which overflowed FP32 at update 3,954, with the row-maximum shift."""
    shifted, unshifted = run_config("r2739c80cc032"), run_config("ra28b295d81e6")
    assert scientific_differences(shifted, unshifted) == {"model.attention_exp_shift"}
    assert (unshifted.model.attention_exp_shift, shifted.model.attention_exp_shift) == ("none", "row_max")


@pytest.mark.parametrize("run_id,unshifted,width", (("r2a45c9e3d452", "r535bc84a7772", 8),
                                                     ("rfdc104924c7c", "r6f84b1db6de6", 16)))
def test_the_shifted_width_8_and_16_exponentials_rerun_the_unshifted_ones(run_id, unshifted, width):
    """X8 and X16 are E1 (r2739c80cc032) at widths 8 and 16, and the unshifted d_K 8 and 16 runs with the shift."""
    shifted = run_config(run_id)
    assert scientific_differences(shifted, run_config("r2739c80cc032")) == {"model.qk_head_dim"}
    assert scientific_differences(shifted, run_config(unshifted)) == {"model.attention_exp_shift"}
    assert (shifted.model.qk_head_dim, shifted.model.attention_exp_shift) == (width, "row_max")


def test_the_width_4_query_run_extends_the_grid():
    """Q4 and the grid's degree-eight runs at d_K 16, 32 and 64 differ from the d_K 8 partner in the key width alone."""
    partner = adapter.partner_config("query")
    for run_id, width in (("rd5e3ec5a4f02", 4), ("rd10c2ebe1181", 16), ("rf5b51bd384bc", 32), ("r25fffc16c2f3", 64)):
        config = run_config(run_id)
        assert scientific_differences(config, partner) == {"model.qk_head_dim"} and config.model.qk_head_dim == width
