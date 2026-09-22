"""Legacy entry points keep their family defaults while using the shared CLI."""

from importlib import import_module
import json

import pytest

from scaffolds.transformer import train as shared_train


@pytest.mark.parametrize("family,flag,value,field,expected,backend", [
    ("key_dimension", "--qk-head-dim", "64", "qk_head_dim", 64, "manual"),
    ("polynomial_attention", "--attention-degree", "3", "attention_degree", 3, "blocked"),
    ("polynomial_attention", "--attention-degree", "softmax", "attention_degree", None, "blocked"),
])
def test_legacy_cli_preserves_overrides_and_source_identity(
    tmp_path, monkeypatch, family, flag, value, field, expected, backend,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train:\n  source_commit: auto\n")
    result_path = tmp_path / "result.json"
    source = "1" * 40
    monkeypatch.setenv(f"{family.upper()}_SOURCE_COMMIT", source)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    observed = []

    def train(config, **kwargs):
        observed.append((config, kwargs))
        return {"final_step": 2}

    monkeypatch.setattr(shared_train, "run_training", train)
    entrypoint = import_module(f"scaffolds.transformer.studies.{family}.train")
    entrypoint.main([
        str(config_path), flag, value, "--workdir", str(tmp_path / "run"),
        "--stop-after-step", "2", "--no-resume", "--result-json", str(result_path),
    ])
    config, options = observed[0]
    assert getattr(config.model, field) == expected
    assert config.model.attention_backend == backend
    assert config.train.source_commit == source
    assert config.train.workdir == tmp_path / "run"
    assert options == {"resume": False, "stop_after_step": 2}
    assert json.loads(result_path.read_text()) == {"final_step": 2}
