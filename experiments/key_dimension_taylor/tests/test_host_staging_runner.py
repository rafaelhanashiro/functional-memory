"""CPU checks for host-staging qualification boundaries and scientific gates."""
import copy
import json
import os
from pathlib import Path
import sys

import pytest

from experiments.key_dimension_taylor.hardware_checks import host_staging as runner


def test_selector_requires_known_distinct_explicit_names(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('original: true\n')
    run = {'name': 'a', 'config_path': str(config), 'config_sha256': runner.common.sha256(config)}
    policy = {'runs': [run]}
    assert runner.resolve_runs(policy, ['a']) == [run]
    for values in ([], ['a,a'], ['unknown']):
        with pytest.raises(ValueError):
            runner.resolve_runs(policy, values)
    config.write_text('changed: true\n')
    with pytest.raises(ValueError, match='config changed'):
        runner.resolve_runs(policy, ['a'])


def test_worker_environment_enforces_runtime_and_transport(tmp_path, monkeypatch):
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setenv('NCCL_P2P_DISABLE', '0')
    env = runner.worker_environment('new-runtime', tmp_path)
    assert env['JAX_PLATFORMS'] == 'cuda'
    assert env['NCCL_P2P_DISABLE'] == env['POLYNOMIAL_ATTENTION_HOST_STAGING'] == '1'
    assert env['POLYNOMIAL_ATTENTION_SOURCE_COMMIT'] == 'new-runtime'
    assert env['PYTHONPATH'] == str(runner.RUNTIME_ROOT)


def test_fresh_continuation_does_not_require_update33_evaluation():
    run = {'qk_head_dim': 8, 'attention_degree': 2, 'parameter_count': 181556224}
    only_record = {**run, 'step': 32, 'source_commit': 'fixed', 'validation_nll': 9.0, 'validation_perplexity': 8103.0}
    assert runner.validate_worker_metrics([only_record], run, 'fixed', 32) == only_record
    assert runner.validate_worker_metrics([only_record], run, 'fixed', 33) == only_record
    with pytest.raises(ValueError, match='identity'):
        runner.validate_worker_metrics([], run, 'fixed', 33)


@pytest.mark.parametrize('field,value', [('validation_nll', float('nan')), ('validation_perplexity', float('inf')), ('source_commit', 'old'), ('parameter_count', 1)])
def test_validation_rejects_nonfinite_or_wrong_identity(field, value):
    run = {'qk_head_dim': 8, 'attention_degree': 2, 'parameter_count': 181556224}
    row = {**run, 'step': 32, 'source_commit': 'fixed', 'validation_nll': 9.0, 'validation_perplexity': 8103.0}
    assert runner.validate_rows([row], run, 'fixed', 32) == row
    row[field] = value
    with pytest.raises((ValueError, FloatingPointError)):
        runner.validate_rows([row], run, 'fixed', 32)


@pytest.mark.parametrize('field', ['start_state_sha256', 'first_update'])
def test_h100_reference_requires_exact_initializer_and_first_input(tmp_path, field):
    manifest = {'project_root': str(tmp_path), 'identity': 'sweep-id', 'source_commit': 'old-scientific'}
    run = {'name': 'run'}
    audit = {'start_state_sha256': 'a' * 64, 'first_update': {'step': 1, 'batch_sha256': 'b' * 64, 'rng_sha256': 'c' * 64}}
    path = tmp_path / 'results/key_dimension_taylor/preflight/sweep-id/checks/run.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'passed', 'identity': 'sweep-id', 'source_commit': 'old-scientific', 'initial': {'checkpoint_audit': audit}}))
    initial = {'checkpoint_audit': copy.deepcopy(audit)}
    assert runner.compare_h100(initial, run, manifest)['status'] == 'passed'
    initial['checkpoint_audit'][field] = 'changed'
    with pytest.raises(ValueError, match='H100 parity failed'):
        runner.compare_h100(initial, run, manifest)


def test_child_timeout_is_bounded_and_writes_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'TIMEOUT_SECONDS', 0.1)
    with (tmp_path / 'lock').open('w') as lock:
        result = runner.run_child([sys.executable, '-c', "import time; print('started',flush=True); time.sleep(60)"], tmp_path / 'child.log', dict(os.environ), lock.fileno())
    assert result['timeout'] is True
    assert result['elapsed_seconds'] < 5
    assert 'started' in (tmp_path / 'child.log').read_text()
