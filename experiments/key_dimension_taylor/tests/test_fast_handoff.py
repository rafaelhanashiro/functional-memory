"""Checkpoint and ownership gates for a live hardware handoff, without Slurm."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from experiments.key_dimension_taylor.concurrent import fast_handoff as handoff


@pytest.fixture
def setup(tmp_path, monkeypatch):
    common = handoff.common
    runs = []
    for name in ('one', 'two'):
        workdir = tmp_path / name
        checkpoint_dir = workdir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True)
        runs.append({'name': name, 'workdir': str(workdir), 'checkpoint_dir': str(checkpoint_dir)})
    old = {'training_identity': 'science', 'source_commit': 'original', 'training_root': '/original',
           'runtime_commit': 'fixed', 'runtime_root': '/fixed', 'assignments': {'one': 'a6000', 'two': 'a6000'}}
    old['identity'] = common.policy_identity(old)
    old_path = tmp_path / 'old.json'
    common.write_json(old_path, old)
    jobs = {'one': ['10', '11'], 'two': ['20', '21']}
    common.write_json(old_path.with_suffix('.jobs.json'), {'identity': old['identity'], 'jobs': {
        job: {'slurm_job_id': job, 'run_name': name, 'profile': 'a6000'} for name, ids in jobs.items() for job in ids}})
    manifest_path = tmp_path / 'manifest.json'
    manifest = {'identity': 'science', 'source_commit': 'original', 'runs': runs,
                'jobs': {**jobs, 'unrelated': ['99']}}
    common.write_json(manifest_path, manifest)
    policy = {'supersedes_policy': {'path': str(old_path), 'sha256': handoff.hashlib.sha256(old_path.read_bytes()).hexdigest(), 'identity': old['identity']},
              'superseded_jobs': jobs, 'assignments': {'one': 'h100_or_better', 'two': 'h100_or_better'},
              'training_manifest': str(manifest_path)}
    policy['identity'] = common.policy_identity(policy)
    policy_path = tmp_path / 'new.json'
    ledger = {'identity': policy['identity'], 'jobs': {}, 'migration': {}}
    states = {'10': 'RUNNING', '11': 'PENDING', '20': 'RUNNING', '21': 'PENDING', '99': 'RUNNING'}
    calls = []
    monkeypatch.setattr(common, 'job_states', lambda ids: {job: states[job] for job in ids if job in states})

    def command(args, **kwargs):
        calls.append(args)
        assert args[-1] != '99'
        if args[0] == 'scancel':
            states[args[-1]] = 'CANCELLED'
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(handoff.subprocess, 'run', command)
    return policy_path, policy, manifest, ledger, states, calls


def _checkpoint(run, step=500, commit=1):
    path = Path(run['checkpoint_dir']) / str(step)
    path.mkdir()
    (path / '_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs': commit}))
    (path / 'payload').write_bytes(b'checkpoint-data')
    return path


def test_no_cancellation_before_all_coordinates_have_committed_checkpoints(setup):
    path, policy, manifest, ledger, states, calls = setup
    _checkpoint(manifest['runs'][0])
    _checkpoint(manifest['runs'][1], commit=None)
    with pytest.raises(handoff.RetriableHandoff, match='finalized checkpoint'):
        handoff.migrate(path, policy, manifest, ledger)
    assert [call[:2] for call in calls] == [['scontrol', 'hold'], ['scontrol', 'hold']]
    assert ledger['migration']['status'] == 'awaiting_checkpoints'
    assert states['10'] == states['20'] == 'RUNNING'


def test_handoff_preserves_progress_and_unrelated_owners_and_is_idempotent(setup):
    path, policy, manifest, ledger, states, calls = setup
    checkpoints = [_checkpoint(run) for run in manifest['runs']]
    result = handoff.migrate(path, policy, manifest, ledger)
    assert result['status'] == 'complete'
    assert calls[:2] == [['scontrol', 'hold', '11'], ['scontrol', 'hold', '21']]
    assert set(result['cancel_requested_jobs']) == {'10', '11', '20', '21'}
    assert manifest['jobs'] == {'one': [], 'two': [], 'unrelated': ['99']}
    assert states['99'] == 'RUNNING'
    assert all(checkpoint.exists() for checkpoint in checkpoints)
    for run in manifest['runs']:
        record = json.loads((Path(run['workdir']) / 'handoff.json').read_text())
        assert record['checkpoint_step'] == 500 and record['old_runtime_commit'] == 'fixed'
        assert record['checkpoint_provenance']['file_count'] == 2
    count = len(calls)
    assert handoff.migrate(path, policy, manifest, ledger) == result
    assert len(calls) == count


def test_async_temporary_and_uncommitted_checkpoints_are_not_eligible(tmp_path):
    (tmp_path / '500.orbax-checkpoint-tmp-1').mkdir()
    (tmp_path / '1000').mkdir()
    (tmp_path / '1000/_CHECKPOINT_METADATA').write_text('{}')
    assert handoff.finalized_checkpoints(tmp_path) == []


def test_accounting_lag_after_cancellation_is_retriable(setup, monkeypatch):
    path, policy, manifest, ledger, states, calls = setup
    for run in manifest['runs']:
        _checkpoint(run)
    def lag(args, **kwargs):
        calls.append(args)
        if args[0] == 'scancel':
            states[args[-1]] = 'COMPLETING'
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(handoff.subprocess, 'run', lag)
    with pytest.raises(handoff.RetriableHandoff, match='terminating'):
        handoff.migrate(path, policy, manifest, ledger)
    assert ledger['migration']['status'] == 'awaiting_terminal'
    assert manifest['jobs']['one'] == ['10', '11']


def test_unknown_owner_state_never_cancels(setup):
    path, policy, manifest, ledger, states, calls = setup
    states.pop('10')
    with pytest.raises(handoff.RetriableHandoff, match='not settled'):
        handoff.migrate(path, policy, manifest, ledger)
    assert calls == []


def test_policy_cannot_sneak_in_unrelated_owner(setup):
    path, policy, manifest, ledger, states, calls = setup
    policy['superseded_jobs']['one'].append('99')
    policy['identity'] = handoff.common.policy_identity(policy)
    ledger['identity'] = policy['identity']
    with pytest.raises(ValueError, match='owner chain'):
        handoff.migrate(path, policy, manifest, ledger)
    assert calls == []
