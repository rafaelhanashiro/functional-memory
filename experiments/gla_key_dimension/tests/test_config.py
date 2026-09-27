"""GLA recipes preserve the dK × Taylor controls; the summary derives from runs/."""
from pathlib import Path
from scaffolds.records import load_study, run_directory
import hashlib
import json

import pytest
import yaml

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models.registry import model_parameter_count
from experiments.gla_key_dimension.acceptance import SPEC

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "experiments/gla_key_dimension"
KEY_DIMENSIONS = (8, 16, 32, 64, 128)
GLA_FIELDS = {"attention_mechanism", "attention_degree", "gla_gate_rank", "gla_gate_tau",
              "gla_chunk_size", "gla_inner_chunk_size", "gla_use_rope"}
OUTPUT_FIELDS = {"workdir", "checkpoint_dir", "wandb_project", "wandb_run_name", "source_commit",
                 "wandb_entity", "wandb_group", "wandb_dir", "wandb_mode"}
# The dK=64 Taylor recipes moved to the scaffold's current defaults (QK-norm and selective weight
# decay) after the GLA runs executed; everything else they share must still match.
CURRENT_DEFAULT_FIELDS = {"model": {"qk_norm", "qk_norm_sharing", "qk_norm_initial_scale", "qk_norm_eps"},
                          "optim": {"weight_decay_policy"}}


def taylor_control(key_dimension):
    """The degree-4 Taylor recipe at the same width in the dK × Taylor record."""
    name = "200m-m004-seed0" if key_dimension == 64 else f"200m-dk{key_dimension:03d}-m004-seed0"
    return config_to_dict(load_config(run_directory(ROOT / "experiments/key_dimension_taylor", name) / "config.yaml"))


def manifest():
    return load_study(RECORD)


def test_recipes_preserve_dk_taylor_data_optimizer_backbone_and_cadence():
    runs = manifest()["runs"]
    assert [run["qk_head_dim"] for run in runs] == list(KEY_DIMENSIONS)
    for run in runs:
        path = ROOT / run["resolved_config"]["path"]
        candidate, control = config_to_dict(load_config(path)), taylor_control(run["qk_head_dim"])
        if control["model"].get("qk_norm"):
            for section, fields in CURRENT_DEFAULT_FIELDS.items():
                for side in (candidate, control):
                    side[section] = {k: v for k, v in side[section].items() if k not in fields}
        assert candidate["data"] == control["data"]
        assert candidate["optim"] == control["optim"]
        assert ({k: v for k, v in candidate["model"].items() if k not in GLA_FIELDS}
                == {k: v for k, v in control["model"].items() if k not in GLA_FIELDS})
        assert ({k: v for k, v in candidate["train"].items() if k not in OUTPUT_FIELDS}
                == {k: v for k, v in control["train"].items() if k not in OUTPUT_FIELDS})
        assert candidate["model"]["attention_mechanism"] == "gla" and candidate["model"]["attention_degree"] is None
        assert candidate["model"]["attention_backend"] == "fused"
        assert candidate["train"]["workdir"].endswith(f"experiments/gla_key_dimension/runs/{run['run_id']}")
        assert candidate["train"]["checkpoint_dir"].endswith(f"experiments/gla_key_dimension/runs/{run['run_id']}/checkpoints")
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


def test_summary_uses_completed_runs_and_width_matched_comparisons(tmp_path):
    from results.gla_key_dimension import plot

    # Result selections match runs by study name, so the copy keeps the study's directory name.
    record = tmp_path / "gla_key_dimension"
    record.mkdir()
    data = manifest()
    (record / "sweep.yaml").write_text(yaml.safe_dump({key: value for key, value in data.items() if key != "runs"}))
    for entry in data["runs"]:
        directory = record / "runs" / entry["run_id"]
        directory.mkdir(parents=True)
        (directory / "record.json").write_text(json.dumps(entry))
    run = run_directory(record, "200m-dk016-gla-seed0")
    (run / "completion.json").write_text(json.dumps({"final_step": 19074, "training_tokens": 10_000_269_312,
                                                     "validation_nll": 2.8, "validation_perplexity": 16.4}))
    (run / "best_validation.json").write_text(json.dumps({"step": 19074, "validation_perplexity": 16.4}))
    rows = {row["run"]: row for row in plot.gla_rows(manifest(), record)}
    assert rows["200m-dk008-gla-seed0"]["status"] == "incomplete"
    completed = rows["200m-dk016-gla-seed0"]
    softmax = next(c for c in manifest()["comparisons"]["runs"] if c["name"] == "200m-dk016-seed0")
    assert completed["softmax_perplexity"] == softmax["terminal"]["validation_perplexity"]
    assert completed["delta_perplexity_vs_softmax"] == pytest.approx(16.4 - completed["softmax_perplexity"])
    (run / "completion.json").write_text(json.dumps({"final_step": 1000, "training_tokens": 524_288_000,
                                                     "validation_nll": 3.0, "validation_perplexity": 20.0}))
    with pytest.raises(ValueError, match="full training budget"):
        plot.gla_rows(manifest(), record)


def test_acceptance_spec_requires_the_fused_kernel_and_declares_existing_tests():
    assert SPEC.entry_module == "experiments.gla_key_dimension.acceptance"
    assert SPEC.required_backend == "fused"
    assert SPEC.runtime_env["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer="
    assert all((ROOT / path).exists() for path in (*SPEC.cpu_tests, *SPEC.gpu_tests))
