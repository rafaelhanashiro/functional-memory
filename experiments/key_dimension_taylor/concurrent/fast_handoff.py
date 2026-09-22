"""Preserve durable A6000 progress before transferring selected job ownership."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import UTC, datetime

try:
    from . import common
except ImportError:
    import common


class RetriableHandoff(RuntimeError):
    """Ownership/checkpoint state is not ready; retry without changing science."""


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _ledger_path(policy_path: Path) -> Path:
    return policy_path.with_suffix('.jobs.json')


def finalized_checkpoints(root: Path, minimum: int = 500) -> list[tuple[int, Path]]:
    """Use Orbax's filesystem finalization rule plus its committed metadata.

    These functions inspect paths only and do not initialize a JAX device or
    create a CheckpointManager (which could modify the training directory).
    """
    from orbax.checkpoint import utils
    if not root.exists():
        return []
    finalized = []
    for path in root.iterdir():
        if not path.name.isdigit() or int(path.name) < minimum or not path.is_dir() or path.is_symlink():
            continue
        try:
            if not utils.is_checkpoint_finalized(path):
                continue
            metadata = json.loads((path / '_CHECKPOINT_METADATA').read_text())
            commit = metadata.get('commit_timestamp_nsecs')
            if type(commit) is int and commit > 0:
                finalized.append((int(path.name), path))
        except (FileNotFoundError, json.JSONDecodeError):
            # Async saves and retention may rename/remove paths while listed.
            continue
    return sorted(finalized)


def checkpoint_provenance(path: Path) -> dict:
    """Bounded metadata+inventory fingerprint; payload integrity is restore-audited."""
    inventory = []
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise ValueError('Checkpoint inventory cannot contain a symlink')
        if not item.is_file():
            continue
        stat = item.stat()
        entry = {'path': str(item.relative_to(path)), 'size_bytes': stat.st_size}
        if stat.st_size <= 1024 * 1024:
            entry['sha256'] = hashlib.sha256(item.read_bytes()).hexdigest()
        inventory.append(entry)
    metadata = path / '_CHECKPOINT_METADATA'
    contents = metadata.read_bytes()
    if not inventory or not json.loads(contents).get('commit_timestamp_nsecs'):
        raise ValueError('Checkpoint inventory lacks committed metadata')
    encoded = json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()
    return {'method': 'metadata_and_directory_inventory', 'metadata_sha256': hashlib.sha256(contents).hexdigest(),
            'inventory_sha256': hashlib.sha256(encoded).hexdigest(), 'files': inventory,
            'file_count': len(inventory), 'size_bytes': sum(item['size_bytes'] for item in inventory),
            'payload_validation': 'Independent restored state/data/RNG audit required before training'}


@contextlib.contextmanager
def _run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RetriableHandoff(f'Previous owner still holds its execution lock: {path}') from error
        yield handle


def _old_ownership(policy: dict, manifest: dict) -> tuple[dict, dict, list[str]]:
    reference = policy['supersedes_policy']
    raw = Path(reference['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Previous A6000 policy checksum changed')
    old = json.loads(raw)
    if (common.policy_identity(old) != old['identity'] or reference.get('identity', old['identity']) != old['identity']
            or old['training_identity'] != manifest['identity'] or old['source_commit'] != manifest['source_commit']):
        raise ValueError('Previous A6000 policy provenance changed')
    previous_ledger = json.loads(_ledger_path(Path(reference['path'])).read_text())
    if previous_ledger['identity'] != old['identity']:
        raise ValueError('Previous A6000 ownership ledger identity changed')
    names = list(policy['assignments'])
    if not names or set(policy['assignments'].values()) != {'h100_or_better'} or set(policy['superseded_jobs']) != set(names):
        raise ValueError('Handoff requires the explicit flexible-coordinate set')
    runs = {run['name']: run for run in manifest['runs'] if run['name'] in names}
    if set(runs) != set(names):
        raise ValueError('Handoff contains an unknown coordinate')
    all_jobs = []
    for name in names:
        if old['assignments'].get(name) != 'a6000':
            raise ValueError('Handoff may replace only A6000-assigned coordinates')
        expected = [entry['slurm_job_id'] for entry in previous_ledger['jobs'].values()
                    if entry['run_name'] == name and entry['profile'] == 'a6000']
        jobs = policy['superseded_jobs'][name]
        if not expected or len(jobs) != len(set(jobs)) or set(jobs) != set(expected):
            raise ValueError(f'Handoff does not match the complete previous owner chain: {name}')
        all_jobs.extend(jobs)
    if len(all_jobs) != len(set(all_jobs)):
        raise ValueError('A previous job is assigned to multiple coordinates')
    return old, runs, all_jobs


def migrate(policy_path: Path, policy: dict, manifest: dict, ledger: dict) -> dict:
    """Hold successors, await durable progress, cancel owners, then record handoff.

    Call under the new policy's scheduler lock. This routine never submits jobs
    and only touches explicitly superseded IDs and selected manifest entries.
    """
    if ledger['identity'] != policy['identity'] or common.policy_identity(policy) != policy['identity']:
        raise ValueError('Handoff policy or ledger identity changed')
    if ledger.get('migration', {}).get('status') == 'complete':
        return ledger['migration']
    old, runs, all_jobs = _old_ownership(policy, manifest)
    minimum = policy.get('minimum_checkpoint_step', 500)
    if type(minimum) is not int or minimum < 500:
        raise ValueError('Handoff must preserve at least update 500')
    previous_migration = ledger.get('migration', {})
    migration = {**previous_migration, 'status': 'holding', 'updated_utc': _utc(),
                 'old_policy_identity': old['identity'], 'superseded_jobs': policy['superseded_jobs'],
                 'minimum_checkpoint_step': minimum}

    def save(status, **updates):
        migration.update(status=status, updated_utc=_utc(), **updates)
        ledger['migration'] = migration
        common.write_json(_ledger_path(policy_path), ledger)

    def states_checked():
        states = common.job_states(all_jobs)
        unexpected = {job: states.get(job) for job in all_jobs
                      if states.get(job) not in common.TERMINAL | {'PENDING', 'RUNNING'}}
        if unexpected:
            save('awaiting_owner_state', jobs=states, unexpected_states=unexpected)
            raise RetriableHandoff(f'Previous owner accounting/state has not settled: {unexpected}')
        return states

    states = states_checked()
    for job in all_jobs:
        if states[job] == 'PENDING':
            try:
                subprocess.run(['scontrol', 'hold', job], check=True, capture_output=True, text=True, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                save('awaiting_owner_state', jobs=common.job_states(all_jobs), error=str(error))
                raise RetriableHandoff(f'Could not safely hold previous successor {job}; retry') from error
    states = states_checked()
    save('held', jobs=states)
    ready, missing = {}, []
    for name, run in runs.items():
        checkpoints = finalized_checkpoints(Path(run['checkpoint_dir']), minimum)
        if not checkpoints:
            missing.append(name)
        else:
            step, path = checkpoints[-1]
            ready[name] = {'step': step, 'path': str(path)}
    if missing:
        save('awaiting_checkpoints', checkpoints=ready, missing_checkpoints=missing)
        raise RetriableHandoff(f'Awaiting finalized checkpoint >= {minimum}: {missing}')

    save('cancelling', checkpoints=ready, missing_checkpoints=[])
    cancelled = list(migration.get('cancel_requested_jobs', []))
    # Recheck ownership immediately before each cancellation. Every successor
    # is held, so cancelling a predecessor cannot release another GPU worker.
    for job in reversed(all_jobs):
        state = common.job_states([job]).get(job)
        if state in common.TERMINAL:
            continue
        if state not in {'PENDING', 'RUNNING'}:
            save('awaiting_terminal', jobs=common.job_states(all_jobs), cancel_requested_jobs=cancelled)
            raise RetriableHandoff(f'Previous owner changed state during cancellation: {job}/{state}')
        try:
            subprocess.run(['scancel', job], check=True, capture_output=True, text=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            save('awaiting_terminal', jobs=common.job_states(all_jobs), cancel_requested_jobs=cancelled, error=str(error))
            raise RetriableHandoff(f'Cancellation of previous owner {job} has not settled; retry') from error
        if job not in cancelled:
            cancelled.append(job)
        save('cancelling', cancel_requested_jobs=cancelled)
    final = common.job_states(all_jobs)
    if any(final.get(job) not in common.TERMINAL for job in all_jobs):
        save('awaiting_terminal', jobs=final, cancel_requested_jobs=cancelled)
        raise RetriableHandoff('Previous Slurm owners are still terminating; retry after they become terminal')
    save('recording_handoffs', jobs=final, cancel_requested_jobs=cancelled)
    handoffs = {}
    for name, run in runs.items():
        root = Path(run['workdir'])
        with _run_lock(root / '.concurrent-production.lock'):
            checkpoints = finalized_checkpoints(Path(run['checkpoint_dir']), minimum)
            if not checkpoints:
                raise RetriableHandoff(f'Finalized checkpoint disappeared before handoff: {name}')
            step, checkpoint = checkpoints[-1]
            record = {'format': 'key-dimension-taylor-checkpoint-handoff', 'policy_identity': policy['identity'],
                      'old_policy_identity': old['identity'], 'old_policy_reference': policy['supersedes_policy'],
                      'old_runtime_commit': old.get('runtime_commit', old['source_commit']),
                      'old_runtime_root': old.get('runtime_root', old['training_root']),
                      'source_commit': manifest['source_commit'], 'run_name': name,
                      'superseded_jobs': policy['superseded_jobs'][name],
                      'final_job_states': {job: final[job] for job in policy['superseded_jobs'][name]},
                      'checkpoint_step': step, 'checkpoint_path': str(checkpoint),
                      'checkpoint_provenance': checkpoint_provenance(checkpoint),
                      'checkpoint_preserved': True, 'logs_preserved': True,
                      'required_before_training': 'Independent restore/state/data/RNG audit', 'created_utc': _utc()}
            path = root / 'handoff.json'
            common.write_json(path, record)
            handoffs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'checkpoint_step': step, 'checkpoint_path': str(checkpoint)}
    save('handoffs_recorded', handoffs=handoffs)
    # Reload to preserve unrelated job-list updates made while awaiting Slurm.
    current = json.loads(Path(policy['training_manifest']).read_text())
    if current['identity'] != manifest['identity']:
        raise ValueError('Scientific manifest identity changed during handoff')
    for name in runs:
        current['jobs'][name] = []
    common.write_json(Path(policy['training_manifest']), current)
    manifest.clear()
    manifest.update(current)
    save('complete', completed_utc=_utc(), final_states=final, handoffs=handoffs)
    return migration
