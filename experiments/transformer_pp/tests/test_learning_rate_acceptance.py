"""Learning-rate runs: each partner's recipe with the peak and floor rates scaled alone. At Q/K width 4, the
exponential and z h_t(z) with a degree-sixteen core at half and double the schedule; at width 64, on ORCD, at two and
four times: the tanh query polynomial m = 8, the exponential, softplus(s) exp(s), and the degree-eight query SoS core
on z = softplus(s) under the softplus (z h_t(z)), exp and sigmoid prefactors."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import learning_rate_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import fresh_recipe_uses_exp_shift
from scaffolds.transformer.configs.loader import save_config

# run: (partner key, scale, label)
RUNS = {
    "r03217591054b": ("exponential", 0.5, "exp-head-scale-row-max-dk4-lr-half"),
    "rcc51ed9dbca8": ("exponential", 2.0, "exp-head-scale-row-max-dk4-lr-double"),
    "r1934bca84bd4": ("shifted_sos", 0.5, "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-half"),
    "rd5cc0aec9712": ("shifted_sos", 2.0, "query-sos-core-m16-softplus-times-input-total-m17-rope-w-decay-dk4-lr-double"),
    "r90776a5d4175": ("query_dk64", 2.0, "query-m8-dk64-w-decay-lr-double"),
    "r9e36ee87aa7e": ("query_dk64", 4.0, "query-m8-dk64-w-decay-lr-quadruple"),
    "r19dfa87cb7ed": ("exponential_dk64", 2.0, "exp-head-scale-row-max-lr-double"),
    "r530a1bce3597": ("exponential_dk64", 4.0, "exp-head-scale-row-max-lr-quadruple"),
    "r752878a0a030": ("softplus_sos_dk64", 2.0, "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-lr-double"),
    "r7924001c42f1": ("softplus_sos_dk64", 4.0,
                      "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay-lr-quadruple"),
    "r39d0469cc122": ("exp_sos_dk64", 2.0, "query-sos-m8-exp-prefactor-softplus-input-w-decay-lr-double"),
    "r365f0c30f632": ("exp_sos_dk64", 4.0, "query-sos-m8-exp-prefactor-softplus-input-w-decay-lr-quadruple"),
    "rf0053d2b3b45": ("sigmoid_sos_dk64", 2.0, "query-sos-m8-sigmoid-prefactor-softplus-input-w-decay-lr-double"),
    "r101b6c157a6b": ("sigmoid_sos_dk64", 4.0, "query-sos-m8-sigmoid-prefactor-softplus-input-w-decay-lr-quadruple"),
    "r2118e3531d89": ("softplus_exp_dk64", 2.0, "exp-softplus-prefactor-lr-double"),
    "re7d0d4e6a5b7": ("softplus_exp_dk64", 4.0, "exp-softplus-prefactor-lr-quadruple"),
}
CELLS = tuple((key, scale) for key, scales in adapter.GRID.items() for scale in scales)
WIDTH_4 = ("exponential", "shifted_sos")
WIDTH_64 = tuple(key for key in adapter.GRID if key not in WIDTH_4)
# The repository's byte copies of the dataset manifests: another path to the partners' manifests on every cluster.
CORPUS = Path(__file__).resolve().parents[3] / "scaffolds/transformer/data/corpora/fineweb_12b"


def changed(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for entry in (first, second):
        for key in adapter.EXECUTION_KEYS:
            entry["train"].pop(key, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


def with_manifests(config, train, eval_):
    return replace(config, data=replace(config.data, train_manifest=str(train), eval_manifest=str(eval_)))


@pytest.mark.parametrize("key, scale", CELLS)
def test_a_recipe_scales_its_partners_peak_and_floor_rates_alone(key, scale):
    config, partner = adapter.recipe(key, scale), adapter.partner_config(key)
    # A fresh recipe's max-abs prescale and the exponential's row-maximum shift, where its partner predates them.
    fresh = {f"model.{name}" for name, value in adapter.fresh_settings(key).items()
             if getattr(partner.model, name) != value}
    assert changed(config, partner) == {"optim.learning_rate", "optim.min_learning_rate"} | fresh
    assert config.model.attention_head_rms_prescale == "max_abs"
    assert config.optim.learning_rate == pytest.approx(scale * 1e-3, rel=1e-12)
    assert config.optim.min_learning_rate == pytest.approx(scale * 1e-4, rel=1e-12)
    assert (config.optim.warmup_steps, config.optim.total_steps, config.optim.grad_clip_norm) == (455, 7570, 1.0)
    assert (config.optim.weight_decay, config.optim.beta1, config.optim.beta2) == (
        partner.optim.weight_decay, partner.optim.beta1, partner.optim.beta2)
    assert config.optim.query_coefficient_learning_rate_scale == partner.optim.query_coefficient_learning_rate_scale
    assert config.model.qk_head_dim == (4 if key in WIDTH_4 else 64) and config.train.seed == partner.train.seed == 42
    assert adapter.partner_key(config) == key and adapter.scale_of(config) == scale
    adapter.check_recipe(config)


def test_the_width_4_partners_are_the_exponential_and_the_degree_sixteen_core():
    exponential, sos = adapter.partner_config("exponential"), adapter.partner_config("shifted_sos")
    assert exponential.model.attention_degree is None and not exponential.model.attention_normalization
    assert exponential.model.attention_exp_shift == "row_max"
    assert (sos.model.attention_degree, sos.model.attention_coefficient_input_power) == (17, 1)
    assert sos.model.attention_coefficient_parameterization == "sum_of_squares"
    assert sos.model.attention_score_transform == "softplus" and sos.model.attention_polynomial_shift == "row_power"


def test_the_width_64_partners_are_query_m8_the_exponential_and_the_degree_eight_core():
    query, exponential, sos = (adapter.partner_config(key)
                               for key in ("query_dk64", "exponential_dk64", "softplus_sos_dk64"))
    assert {query.model.qk_head_dim, exponential.model.qk_head_dim, sos.model.qk_head_dim} == {64}
    assert (query.model.attention_degree, query.model.attention_coefficient_parameterization,
            query.model.attention_coefficient_conditioning) == (8, "direct", "query")
    assert query.model.attention_coefficient_kernel_weight_decay and query.model.positional_encoding == "rope"
    assert exponential.model.attention_degree is None and exponential.model.attention_exp_shift == "none"
    assert adapter.recipe("exponential_dk64", 2.0).model.attention_exp_shift == "row_max"
    assert (sos.model.attention_degree, sos.model.attention_coefficient_input_power) == (9, 1)
    assert sos.model.attention_coefficient_parameterization == "sum_of_squares"
    assert sos.model.attention_score_transform == "softplus"


def test_the_width_64_prefactor_partners_are_f_of_s_on_the_degree_eight_core_of_z_and_softplus_times_exp():
    for key, prefactor in (("exp_sos_dk64", "exp"), ("sigmoid_sos_dk64", "sigmoid")):
        model = adapter.partner_config(key).model
        assert (model.qk_head_dim, model.attention_degree, model.attention_coefficient_input_power) == (64, 8, 0)
        assert model.attention_coefficient_parameterization == "sum_of_squares"
        assert (model.attention_score_prefactor, model.attention_score_transform) == (prefactor, "softplus")
        assert model.attention_backend == "mosaic" and adapter.PARTNERS[key][2] is adapter.softplus_sos_acceptance
    model = adapter.partner_config("softplus_exp_dk64").model
    assert model.qk_head_dim == 64 and model.attention_degree is None
    assert (model.attention_score_prefactor, model.attention_score_transform) == ("softplus", "identity")
    assert model.attention_backend == "fused" and adapter.PARTNERS["softplus_exp_dk64"][2] is None


@pytest.mark.parametrize("key", adapter.GRID)
def test_fresh_settings_are_what_records_create_gives_the_partner(key):
    partner, config = adapter.partner_config(key), adapter.recipe(key, adapter.GRID[key][-1])
    assert config.model.attention_head_rms_prescale == "max_abs"
    # The row-maximum shift applies to the plain exponential alone: not to softplus(s) exp(s) on the fused kernel,
    # nor to a prefactor on the SoS kernels.
    shifted = key in ("exponential", "exponential_dk64")
    assert fresh_recipe_uses_exp_shift(partner.model) == shifted
    assert config.model.attention_exp_shift == ("row_max" if shifted else partner.model.attention_exp_shift)


@pytest.mark.parametrize("run", RUNS)
def test_an_allocated_run_is_its_recipe(run):
    key, scale, label = RUNS[run]
    path = adapter.RUNS / run / "config.yaml"
    config = load_config(path)
    expected = adapter.recipe(key, scale)
    # A run on another cluster reads that cluster's copies of its partner's manifests: compare at its own paths, and
    # check it with the repository's byte copies in their place, which every cluster holds.
    assert changed(config, with_manifests(expected, config.data.train_manifest, config.data.eval_manifest)) == set()
    if config.data.train_manifest != expected.data.train_manifest:
        config = with_manifests(config, CORPUS / "train/manifest.json", CORPUS / "eval/manifest.json")
    adapter.check_recipe(config)
    assert (adapter.partner_key(config), adapter.scale_of(config)) == (key, scale)
    assert config.model.attention_head_rms_prescale == "max_abs"
    record = json.loads((path.parent / "record.json").read_text())
    assert record["run_id"] == run and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_runs_cover_every_partner_at_each_of_its_scales():
    assert sorted((key, scale) for key, scale, _ in RUNS.values()) == sorted(CELLS)


def test_manifest_copies_are_accepted_by_their_bytes():
    config = with_manifests(adapter.recipe("softplus_sos_dk64", 4.0), CORPUS / "train/manifest.json",
                            CORPUS / "eval/manifest.json")
    adapter.check_recipe(config)
    with pytest.raises(ValueError, match="eval_manifest"):
        adapter.check_recipe(with_manifests(config, CORPUS / "train/manifest.json", CORPUS / "corpus.json"))


def test_the_width_64_runs_train_on_orcds_h100s_only(tmp_path):
    path = tmp_path / "config.yaml"
    for key in WIDTH_64:
        save_config(adapter.recipe(key, 4.0), path)
        assert adapter.gpu_types(load_config(path)) == ("H100",)
        adapter.require_gpu_type([str(path), "--gpu-type", "H100"])
        with pytest.raises(ValueError, match="H100"):
            adapter.require_gpu_type([str(path), "--gpu-type", "H200"])


def test_other_recipes_are_refused():
    for scale in (1.0, 0.25, 3.0, 4.0):
        with pytest.raises(ValueError):
            adapter.recipe("exponential", scale)
    for key in WIDTH_64:
        with pytest.raises(ValueError):
            adapter.recipe(key, 0.5)
    # A width-64 family under a prefactor without a partner: sigmoid(s) exp(s), the SoS core under swish.
    for key, prefactor in (("softplus_exp_dk64", "sigmoid"), ("exp_sos_dk64", "swish"), ("query_dk64", "softplus")):
        config = adapter.recipe(key, 2.0)
        with pytest.raises(ValueError, match="partner"):
            adapter.partner_key(replace(config, model=replace(config.model, attention_score_prefactor=prefactor)))
    exponential = adapter.recipe("exponential_dk64", 2.0)
    with pytest.raises(ValueError, match="attention_exp_shift"):  # the fresh exponential's shift is required
        adapter.check_recipe(replace(exponential, model=replace(exponential.model, attention_exp_shift="none")))
    for key in adapter.PARTNERS:
        with pytest.raises(ValueError):  # the partner itself
            adapter.check_recipe(adapter.partner_config(key))
