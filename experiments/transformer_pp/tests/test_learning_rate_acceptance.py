"""Learning-rate scalings of completed recipes: the Q/K width 4 exponential and z h_t(z) with a degree-sixteen core,
and the Q/K width 64 degree-eight SoS core under a prefactor of the raw score; each partner's recipe with the peak
and floor rates scaled alone."""
import hashlib
import json

import pytest

from experiments.transformer_pp import learning_rate_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config

# run: (partner key, scale, label)
RUNS = {
    "r03217591054b": ("exponential", 0.5, "exp-head-scale-row-max-dk4-lr-half"),
    "rcc51ed9dbca8": ("exponential", 2.0, "exp-head-scale-row-max-dk4-lr-double"),
    "r1934bca84bd4": ("shifted_sos", 0.5, "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-half"),
    "rd5cc0aec9712": ("shifted_sos", 2.0, "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-double"),
    "r1ff405fe0926": ("exponential", 4.0, "exp-head-scale-row-max-dk4-lr-quadruple"),
    "rdc43b5c9d2d0": ("shifted_sos", 4.0,
                      "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-quadruple"),
    "rda43ee639e0a": ("shifted_sos", 8.0,
                      "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-octuple"),
    "r76b9aa49fe23": ("exp_prefactor_sos", 2.0, "query-sos-m8-exp-prefactor-raw-input-w-decay-lr-double"),
    "rbf850552cbba": ("exp_prefactor_sos", 4.0, "query-sos-m8-exp-prefactor-raw-input-w-decay-lr-quadruple"),
    "rf93d2b71f731": ("softplus_prefactor_sos", 2.0, "query-sos-m8-softplus-prefactor-raw-input-w-decay-lr-double"),
    "rb418110c763c": ("softplus_prefactor_sos", 4.0, "query-sos-m8-softplus-prefactor-raw-input-w-decay-lr-quadruple"),
    "r7d7504df667b": ("sigmoid_prefactor_sos", 2.0, "query-sos-m8-sigmoid-prefactor-raw-input-w-decay-lr-double"),
    "rb6eeed222062": ("sigmoid_prefactor_sos", 4.0, "query-sos-m8-sigmoid-prefactor-raw-input-w-decay-lr-quadruple"),
}
# Every partner's scales.
PARTNER_SCALES = tuple((key, scale) for key in adapter.PARTNERS for scale in adapter.SCALES[key])


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("key, scale", PARTNER_SCALES)
def test_a_recipe_scales_its_partners_peak_and_floor_rates_alone(key, scale):
    config, partner = adapter.recipe(key, scale), adapter.partner_config(key)
    # The exponential's and the prefactor partners predate the max-abs prescale; z h_t(z) has it.
    prescale = ({"model.attention_head_rms_prescale"} if partner.model.attention_head_rms_prescale != "max_abs"
                else set())
    assert changed(config, partner) == {"optim.learning_rate", "optim.min_learning_rate"} | prescale
    assert config.model.attention_head_rms_prescale == "max_abs"
    assert config.optim.learning_rate == pytest.approx(scale * 1e-3, rel=1e-12)
    assert config.optim.min_learning_rate == pytest.approx(scale * 1e-4, rel=1e-12)
    assert (config.optim.warmup_steps, config.optim.total_steps, config.optim.grad_clip_norm) == (455, 7570, 1.0)
    assert config.model.qk_head_dim == (64 if key.endswith("_prefactor_sos") else 4)
    assert config.train.seed == partner.train.seed == 42
    assert adapter.partner_key(config) == key and adapter.scale_of(config) == scale
    adapter.check_recipe(config)


def test_the_partners_are_the_exponential_and_the_sos_cores():
    exponential, sos = adapter.partner_config("exponential"), adapter.partner_config("shifted_sos")
    assert exponential.model.attention_degree is None and not exponential.model.attention_normalization
    assert exponential.model.attention_exp_shift == "row_max"
    assert (sos.model.attention_degree, sos.model.attention_coefficient_input_power) == (17, 1)
    assert sos.model.attention_coefficient_parameterization == "sum_of_squares"
    assert sos.model.attention_score_transform == "softplus" and sos.model.attention_polynomial_shift == "row_power"
    for prefactor in ("exp", "softplus", "sigmoid"):
        model = adapter.partner_config(f"{prefactor}_prefactor_sos").model
        assert (model.attention_score_prefactor, model.attention_score_transform) == (prefactor, "identity")
        assert (model.attention_degree, model.attention_coefficient_input_power, model.qk_head_dim) == (8, 0, 64)
        assert model.attention_coefficient_parameterization == "sum_of_squares"


@pytest.mark.parametrize("run", RUNS)
def test_an_allocated_run_is_its_recipe(run):
    key, scale, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert (adapter.partner_key(config), adapter.scale_of(config)) == (key, scale)
    assert changed(config, adapter.recipe(key, scale)) == set()
    assert config.model.attention_head_rms_prescale == "max_abs"
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_runs_cover_both_partners_at_every_scale():
    assert sorted((key, scale) for key, scale, _ in RUNS.values()) == sorted(PARTNER_SCALES)


def test_other_recipes_are_refused():
    for key, scale in (("exponential", 1.0), ("exponential", 0.25), ("exponential", 3.0), ("exponential", 8.0),
                       ("shifted_sos", 1.0), ("shifted_sos", 16.0), ("exp_prefactor_sos", 0.5),
                       ("softplus_prefactor_sos", 8.0), ("sigmoid_prefactor_sos", 1.0)):
        with pytest.raises(ValueError):
            adapter.recipe(key, scale)
    for key in adapter.PARTNERS:
        with pytest.raises(ValueError):  # the partner itself
            adapter.check_recipe(adapter.partner_config(key))
        config = adapter.recipe(key, adapter.SCALES[key][0])
        config.optim.min_learning_rate = 1e-4  # the floor must scale with the peak
        with pytest.raises(ValueError):
            adapter.check_recipe(config)
        config = adapter.recipe(key, 2.0)
        config.optim.warmup_steps = 910
        with pytest.raises(ValueError):
            adapter.check_recipe(config)
        config = adapter.recipe(key, 2.0)
        config.model.qk_head_dim = 8
        with pytest.raises(ValueError):
            adapter.check_recipe(config)


@pytest.mark.parametrize("scale", adapter.SCALES["exponential"])
def test_the_exponential_takes_either_evaluation_of_head_rms_norm_and_the_shift_only_the_prescale(scale):
    """The shared rule (test_adapters_head_rms_prescale): the comparison leaves the evaluation to the run, and the
    partner's adapter to its kernel."""
    from dataclasses import replace

    def plain(config):
        return replace(config, model=replace(config.model, attention_head_rms_prescale="none"))

    exponential = adapter.recipe("exponential", scale)
    adapter.check_recipe(plain(exponential))
    for config in (exponential, plain(exponential)):
        config = replace(config, optim=replace(config.optim, weight_decay=config.optim.weight_decay + 0.5))
        with pytest.raises(ValueError, match="optim.weight_decay"):
            adapter.check_recipe(config)
    with pytest.raises(ValueError):
        adapter.check_recipe(plain(adapter.recipe("shifted_sos", scale)))


@pytest.mark.parametrize("key", adapter.PARTNERS)
def test_a_run_qualifies_as_its_partner_does(key):
    config = adapter.recipe(key, adapter.SCALES[key][0])
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    partner = adapter.PARTNERS[key][2].experiment_spec(adapter.at_partner_rate(config), entry_module=adapter.__name__)
    assert spec.required_backend == partner.required_backend == ("mosaic" if key.endswith("_prefactor_sos")
                                                                  else "fused")
    assert spec.gpu_tests == partner.gpu_tests and spec.pilot_steps == partner.pilot_steps
    assert spec.cpu_tests == (*partner.cpu_tests, "experiments/transformer_pp/tests/test_learning_rate_acceptance.py")
    assert spec.check_config is adapter.check_recipe
    assert adapter.gpu_types(config) == {"exponential": ("H100", "B300"), "shifted_sos": ("B200", "B300")}.get(
        key, ("H100", "B200"))


def test_the_gpu_type_rule(tmp_path):
    from scaffolds.transformer.configs import save_config

    for key, allowed, refused in (("exponential", "B300", "B200"), ("shifted_sos", "B300", "H100"),
                                  ("sigmoid_prefactor_sos", "B200", "B300")):
        path = tmp_path / f"{key}.yaml"
        save_config(adapter.recipe(key, 2.0), path)
        adapter.require_gpu_type([str(path), "--gpu-type", allowed])
        adapter.require_gpu_type([str(path), "--runtime-env", "--gpu-type", refused])
        with pytest.raises(ValueError):
            adapter.require_gpu_type([str(path), "--gpu-type", refused])
