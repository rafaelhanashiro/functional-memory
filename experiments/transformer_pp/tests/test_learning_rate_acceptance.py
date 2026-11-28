"""Learning-rate scalings of completed recipes: the Q/K width 4 exponential and z h_t(z) with a degree-sixteen or
degree-eight core, and at Q/K width 64 the exponential, softplus(s) exp(s), the tanh query polynomial, z h_t(z) with a
degree-eight core and that core under a prefactor of the raw score on the raw score or on z = softplus(s); each
partner's recipe with the peak and floor rates scaled alone."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import learning_rate_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import fresh_recipe_uses_exp_shift, fresh_recipe_uses_head_rms_prescale

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
    "r57fa34df51d2": ("exponential", 3.0, "exp-head-scale-row-max-dk4-lr-triple"),
    "r73c86839450f": ("exponential", 5.0, "exp-head-scale-row-max-dk4-lr-quintuple"),
    "r9bb10a10547d": ("exponential", 6.0, "exp-head-scale-row-max-dk4-lr-sextuple"),
    "r533793a0833f": ("shifted_sos", 3.0,
                      "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-triple"),
    "r24629147e7fe": ("shifted_sos", 5.0,
                      "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-quintuple"),
    "r60b2eb520c6b": ("shifted_sos", 6.0,
                      "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-sextuple"),
    "r76b9aa49fe23": ("exp_prefactor_sos", 2.0, "query-sos-m8-exp-prefactor-raw-input-w-decay-lr-double"),
    "rbf850552cbba": ("exp_prefactor_sos", 4.0, "query-sos-m8-exp-prefactor-raw-input-w-decay-lr-quadruple"),
    "rf93d2b71f731": ("softplus_prefactor_sos", 2.0, "query-sos-m8-softplus-prefactor-raw-input-w-decay-lr-double"),
    "rb418110c763c": ("softplus_prefactor_sos", 4.0, "query-sos-m8-softplus-prefactor-raw-input-w-decay-lr-quadruple"),
    "r7d7504df667b": ("sigmoid_prefactor_sos", 2.0, "query-sos-m8-sigmoid-prefactor-raw-input-w-decay-lr-double"),
    "rb6eeed222062": ("sigmoid_prefactor_sos", 4.0, "query-sos-m8-sigmoid-prefactor-raw-input-w-decay-lr-quadruple"),
    "rc44dc19a9972": ("exponential_dk64", 0.5, "exp-head-scale-row-max-lr-half"),
    "rd1baa331563c": ("softplus_sos_dk64", 0.5, "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-lr-half"),
    "rfc577a90377d": ("softplus_sos_m8", 2.0,
                      "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-dk4-lr-double"),
    "r9814496cc661": ("softplus_sos_m8", 4.0,
                      "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-dk4-lr-quadruple"),
    "r7cbb2c7ca07d": ("softplus_sos_m8", 8.0,
                      "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-dk4-lr-octuple"),
    # On ORCD, reading byte copies of the partners' manifests.
    "r90776a5d4175": ("query_dk64", 2.0, "query-m8-dk64-w-decay-lr-double"),
    "r9e36ee87aa7e": ("query_dk64", 4.0, "query-m8-dk64-w-decay-lr-quadruple"),
    "r19dfa87cb7ed": ("exponential_dk64", 2.0, "exp-head-scale-row-max-lr-double"),
    "r530a1bce3597": ("exponential_dk64", 4.0, "exp-head-scale-row-max-lr-quadruple"),
    "r752878a0a030": ("softplus_sos_dk64", 2.0,
                      "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-lr-double"),
    "r7924001c42f1": ("softplus_sos_dk64", 4.0,
                      "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-lr-quadruple"),
    "r39d0469cc122": ("exp_prefactor_softplus_sos", 2.0, "query-sos-m8-exp-prefactor-softplus-input-w-decay-lr-double"),
    "r365f0c30f632": ("exp_prefactor_softplus_sos", 4.0,
                      "query-sos-m8-exp-prefactor-softplus-input-w-decay-lr-quadruple"),
    "rf0053d2b3b45": ("sigmoid_prefactor_softplus_sos", 2.0,
                      "query-sos-m8-sigmoid-prefactor-softplus-input-w-decay-lr-double"),
    "r101b6c157a6b": ("sigmoid_prefactor_softplus_sos", 4.0,
                      "query-sos-m8-sigmoid-prefactor-softplus-input-w-decay-lr-quadruple"),
    "r2118e3531d89": ("softplus_exp_dk64", 2.0, "exp-softplus-prefactor-lr-double"),
    "re7d0d4e6a5b7": ("softplus_exp_dk64", 4.0, "exp-softplus-prefactor-lr-quadruple"),
}
# The partners at Q/K width 64, and their backends.
WIDE = ("exp_prefactor_sos", "softplus_prefactor_sos", "sigmoid_prefactor_sos", "exponential_dk64",
        "softplus_sos_dk64", "query_dk64", "softplus_exp_dk64", "exp_prefactor_softplus_sos",
        "sigmoid_prefactor_softplus_sos")
BACKENDS = {"exponential": "fused", "shifted_sos": "fused", "exponential_dk64": "fused", "softplus_exp_dk64": "fused"}
GPUS = {"exponential": ("H100", "B300"), "shifted_sos": ("B200", "B300"), "exponential_dk64": ("H100", "B300"),
        "query_dk64": ("H100",), "softplus_exp_dk64": ("H100",)}
# The repository's byte copies of the dataset manifests: another path to the partners' manifests on every cluster.
CORPUS = Path(__file__).resolve().parents[3] / "scaffolds/transformer/data/corpora/fineweb_12b"
# Every partner's scales.
PARTNER_SCALES = tuple((key, scale) for key in adapter.PARTNERS for scale in adapter.SCALES[key])


def with_manifests(config, train, eval_):
    return replace(config, data=replace(config.data, train_manifest=str(train), eval_manifest=str(eval_)))


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
    newer = {f"model.{name}" for name in adapter.PARTNER_CHANGES.get(key, {})}
    assert changed(config, partner) == {"optim.learning_rate", "optim.min_learning_rate"} | prescale | newer
    assert config.model.attention_head_rms_prescale == "max_abs"
    assert config.optim.learning_rate == pytest.approx(scale * 1e-3, rel=1e-12)
    assert config.optim.min_learning_rate == pytest.approx(scale * 1e-4, rel=1e-12)
    assert (config.optim.warmup_steps, config.optim.total_steps, config.optim.grad_clip_norm) == (455, 7570, 1.0)
    assert config.model.qk_head_dim == (64 if key in WIDE else 4)
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
    narrow_sos = adapter.partner_config("softplus_sos_m8").model
    assert (narrow_sos.attention_degree, narrow_sos.attention_coefficient_input_power,
            narrow_sos.qk_head_dim) == (9, 1, 4)
    assert narrow_sos.attention_score_transform == "softplus" and narrow_sos.attention_backend == "mosaic"
    wide_exp = adapter.partner_config("exponential_dk64").model
    wide_sos = adapter.partner_config("softplus_sos_dk64").model
    assert wide_exp.attention_degree is None and not wide_exp.attention_normalization and wide_exp.qk_head_dim == 64
    assert (wide_sos.attention_degree, wide_sos.attention_coefficient_input_power, wide_sos.qk_head_dim) == (9, 1, 64)
    assert wide_sos.attention_score_transform == "softplus" and wide_sos.attention_score_prefactor == "none"
    for prefactor in ("exp", "sigmoid"):
        model = adapter.partner_config(f"{prefactor}_prefactor_softplus_sos").model
        assert (model.attention_score_prefactor, model.attention_score_transform) == (prefactor, "softplus")
        assert (model.attention_degree, model.attention_coefficient_input_power, model.qk_head_dim) == (8, 0, 64)
        assert model.attention_coefficient_parameterization == "sum_of_squares"
        # The Mosaic softplus-SoS and prefactor kernels qualify through their own adapter's GPU tests.
        assert model.attention_backend == "mosaic"
        assert adapter.PARTNERS[f"{prefactor}_prefactor_softplus_sos"][2] is adapter.softplus_sos_acceptance
    for key in ("query_dk64", "softplus_exp_dk64"):
        shared = adapter.PARTNERS[key][2]
        assert isinstance(shared, adapter.SharedPartner) and shared.run_id == adapter.PARTNERS[key][0]
    query = adapter.partner_config("query_dk64").model
    assert (query.qk_head_dim, query.attention_degree, query.attention_coefficient_parameterization,
            query.attention_coefficient_conditioning) == (64, 8, "direct", "query")
    assert query.attention_coefficient_kernel_weight_decay and query.attention_score_prefactor == "none"
    softplus_exp = adapter.partner_config("softplus_exp_dk64").model
    assert softplus_exp.qk_head_dim == 64 and softplus_exp.attention_degree is None
    assert (softplus_exp.attention_score_prefactor, softplus_exp.attention_score_transform) == ("softplus", "identity")


@pytest.mark.parametrize("run", RUNS)
def test_an_allocated_run_is_its_recipe(run):
    key, scale, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    assert (adapter.partner_key(config), adapter.scale_of(config)) == (key, scale)
    expected = adapter.recipe(key, scale)
    # A run on another cluster reads that cluster's copies of its partner's manifests: compare at its own paths, and
    # check it with the repository's byte copies in their place, which every cluster holds.
    assert changed(config, with_manifests(expected, config.data.train_manifest, config.data.eval_manifest)) == set()
    if config.data.train_manifest != expected.data.train_manifest:
        config = with_manifests(config, CORPUS / "train/manifest.json", CORPUS / "eval/manifest.json")
    adapter.check_recipe(config)
    assert config.model.attention_head_rms_prescale == "max_abs"
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_runs_cover_both_partners_at_every_scale():
    assert sorted((key, scale) for key, scale, _ in RUNS.values()) == sorted(PARTNER_SCALES)


def test_other_recipes_are_refused():
    for key, scale in (("exponential", 1.0), ("exponential", 0.25), ("exponential", 7.0), ("exponential", 8.0),
                       ("shifted_sos", 1.0), ("shifted_sos", 16.0), ("exp_prefactor_sos", 0.5),
                       ("softplus_prefactor_sos", 8.0), ("sigmoid_prefactor_sos", 1.0)):
        with pytest.raises(ValueError):
            adapter.recipe(key, scale)
    # A width-64 family under a prefactor without a partner: sigmoid(s) exp(s), the SoS core under swish, the query
    # polynomial under softplus, and softplus(s) h_t(softplus(s)) through a prefactor (the input power's z h_t(z)).
    for key, prefactor in (("softplus_exp_dk64", "sigmoid"), ("exp_prefactor_softplus_sos", "swish"),
                           ("query_dk64", "softplus"), ("exp_prefactor_softplus_sos", "softplus")):
        config = adapter.recipe(key, 2.0)
        with pytest.raises(ValueError, match="partner"):
            adapter.partner_key(replace(config, model=replace(config.model, attention_score_prefactor=prefactor)))
    exponential = adapter.recipe("exponential_dk64", 2.0)
    with pytest.raises(ValueError, match="attention_exp_shift"):  # the fresh exponential's shift is required
        adapter.check_recipe(replace(exponential, model=replace(exponential.model, attention_exp_shift="none")))
    for key in adapter.PARTNERS:
        with pytest.raises(ValueError):  # the partner itself
            adapter.check_recipe(adapter.partner_config(key))
        config = adapter.recipe(key, adapter.SCALES[key][0])
        config.optim.min_learning_rate = 1e-4  # the floor must scale with the peak
        with pytest.raises(ValueError):
            adapter.check_recipe(config)
        config = adapter.recipe(key, adapter.SCALES[key][-1])
        config.optim.warmup_steps = 910
        with pytest.raises(ValueError):
            adapter.check_recipe(config)
        config = adapter.recipe(key, adapter.SCALES[key][-1])
        config.model.qk_head_dim = 8
        with pytest.raises(ValueError):
            adapter.check_recipe(config)


@pytest.mark.parametrize("key", adapter.PARTNERS)
def test_a_recipe_takes_what_records_create_gives_a_fresh_copy_of_its_partner(key):
    """The max-abs prescale wherever head RMSNorm runs, and the row-maximum shift wherever it applies: the plain
    exponential, not softplus(s) exp(s) on the fused kernel nor a prefactor on the SoS kernels."""
    partner, config = adapter.partner_config(key), adapter.recipe(key, adapter.SCALES[key][-1])
    assert fresh_recipe_uses_head_rms_prescale(partner.model) and config.model.attention_head_rms_prescale == "max_abs"
    shifted = fresh_recipe_uses_exp_shift(partner.model)
    assert shifted == (key in ("exponential", "exponential_dk64"))
    assert config.model.attention_exp_shift == ("row_max" if shifted else partner.model.attention_exp_shift)


def test_manifest_copies_are_accepted_by_their_bytes():
    config = with_manifests(adapter.recipe("sigmoid_prefactor_softplus_sos", 4.0), CORPUS / "train/manifest.json",
                            CORPUS / "eval/manifest.json")
    adapter.check_recipe(config)
    with pytest.raises(ValueError, match="eval_manifest"):
        adapter.check_recipe(with_manifests(config, CORPUS / "train/manifest.json", CORPUS / "corpus.json"))


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
    assert spec.required_backend == partner.required_backend == BACKENDS.get(key, "mosaic")
    assert spec.gpu_tests == partner.gpu_tests and spec.pilot_steps == partner.pilot_steps
    assert spec.cpu_tests == (*partner.cpu_tests, "experiments/transformer_pp/tests/test_learning_rate_acceptance.py")
    assert spec.check_config is adapter.check_recipe
    assert adapter.gpu_types(config) == GPUS.get(key, ("H100", "B200"))


def test_the_gpu_type_rule(tmp_path):
    from scaffolds.transformer.configs import save_config

    for key, allowed, refused in (("exponential", "B300", "B200"), ("shifted_sos", "B300", "H100"),
                                  ("sigmoid_prefactor_sos", "B200", "B300"), ("query_dk64", "H100", "H200"),
                                  ("softplus_exp_dk64", "H100", "B300"),
                                  ("exp_prefactor_softplus_sos", "H100", "H200")):
        path = tmp_path / f"{key}.yaml"
        save_config(adapter.recipe(key, 2.0), path)
        adapter.require_gpu_type([str(path), "--gpu-type", allowed])
        adapter.require_gpu_type([str(path), "--runtime-env", "--gpu-type", refused])
        with pytest.raises(ValueError):
            adapter.require_gpu_type([str(path), "--gpu-type", refused])


def test_the_width_64_exponential_starts_from_its_partners_parameters():
    """The shared acceptance's partner: the half-rate exponential's initial parameters are r6a76e7fbde45's, bit for
    bit (the row-maximum shift and the max-abs prescale add none)."""
    config = adapter.recipe("exponential_dk64", 0.5)
    assert config.model.attention_exp_shift == "row_max"
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_initialization(config)["all_partner_arrays_identical"]
