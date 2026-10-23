"""Exercise optional Kennel pilot arguments without allocating hardware."""

import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "scaffolds/transformer/launchers/kennel.sbatch"


@pytest.mark.parametrize("stop", [None, "1500"])
def test_pilot_keeps_full_recipe_and_skips_final_retention(tmp_path, stop):
    launcher = LAUNCHER.read_text()
    option = launcher[launcher.index('STOP_AFTER_STEP='):launcher.index('EXECUTION_ROOT=')]
    worker = launcher[launcher.index('run_step "$PYTHON" -m scaffolds.transformer.train'):]
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    fake_python.chmod(0o755)
    calls = tmp_path / "calls"
    script = 'set -euo pipefail\n' + option + '\nrun_step() { "$@"; }\n' + worker
    arguments = ["bash", "-c", script, "test", "adapter", "recipe.yaml", "commit", "4", "H100"]
    if stop is not None:
        arguments.append(stop)
    env = dict(os.environ, PYTHON=str(fake_python), CONFIG="recipe.yaml", RUN_ROOT="run",
               WANDB_MODE_SETTING="offline", CALLS=str(calls))
    subprocess.run(arguments, env=env, check=True, text=True, capture_output=True)
    actual = [shlex.split(line) for line in calls.read_text().splitlines()]
    expected = ["-m", "scaffolds.transformer.train", "recipe.yaml", "--wandb-mode", "offline"]
    if stop is None:
        assert actual == [expected, ["-m", "scaffolds.transformer.retention", "run/config.json"]]
    else:
        assert actual == [[*expected, "--stop-after-step", stop]]


@pytest.mark.parametrize("stop", ["0", "-1", "1.5", "abc", "1; false"])
def test_launcher_rejects_invalid_stop_before_execution(stop):
    launcher = LAUNCHER.read_text()
    option = launcher[launcher.index('STOP_AFTER_STEP='):launcher.index('EXECUTION_ROOT=')]
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + option,
                             "test", "adapter", "recipe", "commit", "4", "H100", stop],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "STOP_AFTER_STEP must be a positive integer" in result.stderr


@pytest.mark.parametrize("stop", ["", "1500"])
def test_attempt_records_the_execution_limit_without_changing_recipe(tmp_path, monkeypatch, stop):
    from scaffolds.launch import frozen
    from scaffolds.records import create_run, load_run
    from scaffolds.transformer.configs import Config, config_to_dict

    run = create_run(tmp_path / "experiments" / "toy", config_to_dict(Config()))
    config = run / "config.yaml"
    original = config.read_bytes()
    source = re.search(r'"\$EXPERIMENT" "\$STOP_AFTER_STEP" <<\'PY\'\n(.*?)\nPY\n',
                       LAUNCHER.read_text(), re.S).group(1)
    monkeypatch.setattr("sys.argv", ["-", str(run), str(config), "launch", "H100", "4", "toy", stop])
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(frozen, "frozen_hash", lambda *args: "frozen")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "JobId=123")
    exec(compile(source, str(LAUNCHER), "exec"), {})
    record = load_run(run)
    assert record["attempts"][-1]["stop_after_step"] == (int(stop) if stop else None)
    assert record["status"] == "acceptance"
    assert config.read_bytes() == original
