"""GLA recipes preserve the dK × Taylor controls; the summary derives from runs/."""
from pathlib import Path
import hashlib
import json

import pytest
import yaml

from scaffolds.transformer.configs import load_config
from scaffolds.transformer.models.registry import model_parameter_count
from experiments.gla_key_dimension.adapter.execution.acceptance import SPEC

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / "experiments/gla_key_dimension"
KEY_DIMENSIONS = (8, 16, 32, 64, 128)
GLA_FIELDS = {"attention_mechanism", "attention_degree", "gla_gate_rank", "gla_gate_tau",
              "gla_chunk_size", "gla_inner_chunk_size", "gla_use_rope"}
OUTPUT_FIELDS = {"workdir", "checkpoint_dir", "wandb_project", "wandb_run_name", "source_commit"}


def taylor_control(key_dimension):
    """The degree-4 Taylor recipe at the same width in the dK × Taylor record."""
    name = "200m-m004-seed0" if key_dimension == 64 else f"200m-dk{key_dimension:03d}-m004-seed0"
    return yaml.safe_load((ROOT / f"experiments/key_dimension_taylor/configs/{name}.yaml").read_text())


def manifest():
    return json.loads((RECORD / "provenance/manifest.json").read_text())


def test_recipes_preserve_dk_taylor_data_optimizer_backbone_and_cadence():
    runs = manifest()["runs"]
    assert [run["qk_head_dim"] for run in runs] == list(KEY_DIMENSIONS)
    for run in runs:
        path = ROOT / run["resolved_config"]["path"]
        candidate, control = yaml.safe_load(path.read_text()), taylor_control(run["qk_head_dim"])
        assert candidate["data"] == control["data"]
        assert candidate["optim"] == control["optim"]
        assert ({k: v for k, v in candidate["model"].items() if k not in GLA_FIELDS}
                == {k: v for k, v in control["model"].items() if k not in GLA_FIELDS})
        assert ({k: v for k, v in candidate["train"].items() if k not in OUTPUT_FIELDS}
                == {k: v for k, v in control["train"].items() if k not in OUTPUT_FIELDS})
        assert candidate["model"]["attention_mechanism"] == "gla" and candidate["model"]["attention_degree"] is None
        assert candidate["model"]["attention_backend"] == "fused"
        assert candidate["train"]["workdir"].endswith(f"experiments/gla_key_dimension/runs/{run['name']}")
        assert candidate["train"]["checkpoint_dir"].endswith(f"experiments/gla_key_dimension/checkpoints/{run['name']}")
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
    from experiments.gla_key_dimension.adapter.analysis import plot

    record = tmp_path / "record"
    (record / "provenance").mkdir(parents=True)
    (record / "provenance/manifest.json").write_text(json.dumps(manifest()))
    run = record / "runs/200m-dk016-gla-seed0"
    run.mkdir(parents=True)
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
    assert SPEC.entry_module == "experiments.gla_key_dimension.adapter.execution.acceptance"
    assert SPEC.required_backend == "fused"
    assert SPEC.runtime_env["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer="
    assert all((ROOT / path).exists() for path in (*SPEC.cpu_tests, *SPEC.gpu_tests))
