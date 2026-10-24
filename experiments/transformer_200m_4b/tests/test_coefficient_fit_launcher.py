"""Preserve the fitting recipe and selected inputs while adding explicit degrees."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from experiments.transformer_200m_4b import launch_coefficient_fit as launcher
from experiments.transformer_200m_4b.launch_coefficient_fit import fitting_scope


@pytest.mark.parametrize(('count', 'role'), [
    (1, 'static_coefficients_single_sequence'),
    (256, 'static_coefficients_full_batch'),
])
def test_static_scope_requires_explicit_single_or_parent_batch(count, role):
    assert fitting_scope('static_coefficients', 256, count) == (count, role)


@pytest.mark.parametrize(('parent_batch', 'count'), [(256, None), (256, 8), (128, 256)])
def test_static_scope_rejects_missing_partial_or_wrong_parent_batch(parent_batch, count):
    with pytest.raises(ValueError, match='requires --sequence-count'):
        fitting_scope('static_coefficients', parent_batch, count)


@pytest.mark.parametrize(('count', 'role'), [
    (1, 'free_coefficients_single_sequence'),
    (256, 'free_coefficients_full_batch'),
])
def test_free_scope_requires_explicit_single_or_parent_batch(count, role):
    assert fitting_scope('free_per_example_query_coefficients', 256, count) == (count, role)


@pytest.mark.parametrize(('parent_batch', 'count'), [(256, None), (256, 8), (128, 256)])
def test_free_scope_rejects_missing_partial_or_wrong_parent_batch(parent_batch, count):
    with pytest.raises(ValueError, match='requires --sequence-count'):
        fitting_scope('free_per_example_query_coefficients', parent_batch, count)


@pytest.mark.parametrize(('mode', 'expected'), [
    ('query_coefficient_generator', (1, 'query_generator_single_sequence')),
    ('shared_batch_query_coefficients', (256, 'shared_coefficients_full_batch')),
])
def test_existing_scopes_are_preserved(mode, expected):
    assert fitting_scope(mode, 256) == expected
    with pytest.raises(ValueError, match='only supported'):
        fitting_scope(mode, 256, 1)


def test_degree_roles_preserve_existing_selection_names():
    role = 'query_generator_single_sequence'
    assert launcher.fitting_role(role, 8) == role
    assert launcher.fitting_role(role, 4) == role + '_m4'
    for invalid in (None, 0, -1, True, 4.0):
        with pytest.raises(ValueError, match='positive integer'):
            launcher.fitting_role(role, invalid)


def original_query_fit():
    return yaml.safe_load((Path(__file__).parents[1] / 'runs/rd7bf9bc37cb6/config.yaml').read_text())


def test_parent_degree_uses_shared_backend_validation():
    model = original_query_fit()['model']
    for degree in (4, 8, 12):
        assert launcher.parent_degree({**model, 'attention_degree': degree}) == degree
    with pytest.raises(ValueError, match="Backend 'mosaic'"):
        launcher.parent_degree({**model, 'attention_degree': 13})
    with pytest.raises(ValueError, match='query-only parent'):
        launcher.parent_degree({**model, 'attention_coefficient_scale': 'none'})


@pytest.mark.parametrize('parameterization,sequence_count,degree', [
    ('query_coefficient_generator', None, 4),
    ('free_per_example_query_coefficients', 256, 8),
    ('free_per_example_query_coefficients', 1, 4),
])
def test_prepare_preserves_parent_and_allocates_the_selected_fit(
        tmp_path, monkeypatch, parameterization, sequence_count, degree):
    import hashlib
    import numpy as np
    from scaffolds.transformer.data import packed

    prior = original_query_fit()
    source = deepcopy(prior)
    for key in ('oracle', 'experiment_kind', 'condition', 'schema_version'):
        source.pop(key, None)
    source['model']['attention_degree'] = degree
    source['model']['gdn_polynomial_degree'] = 1
    source['data']['batch_size'] = 256
    manifest = tmp_path / 'eval-manifest.json'
    manifest.write_text('{}\n')
    source['data']['eval_manifest'] = str(manifest)
    parent = tmp_path / 'source/experiments/transformer_200m_4b/runs/r49b7943aee1b'
    parent.mkdir(parents=True)
    (parent / 'config.yaml').write_text(yaml.safe_dump(source))
    (parent / 'best_validation.json').write_text(json.dumps({'step': 7500}))
    checkpoint = parent / 'checkpoints/7500'
    checkpoint.mkdir(parents=True)
    descriptor = {'path': str(checkpoint), 'step': 7500, 'sha256': 'c' * 64}
    monkeypatch.setattr(launcher, 'load_run', lambda _: {
        'status': 'complete', 'home_cluster': 'kennel', 'selected_checkpoint': descriptor})
    monkeypatch.setattr(launcher, 'validate_record', lambda *a, **kw: None)
    study = tmp_path / 'child/experiments/transformer_200m_4b'
    topic = study / 'results/exploratory/coefficient_oracle'
    topic.mkdir(parents=True)
    old_entry = {'run_id': 'rd7bf9bc37cb6', 'role': 'query_generator_single_sequence',
                 'parent_run_id': 'r98d5895c02d0', 'sources': {'config.yaml': {'sha256': 'a' * 64}}}
    (topic / 'selection.yaml').write_text(yaml.safe_dump({'settings': {}, 'runs': [old_entry]}))
    reference = study / 'runs/r3c0f21ab7dce'
    reference.mkdir(parents=True)
    sequence_identity = {'manifest_sha256': launcher.sha256(manifest), 'batch_sha256': 'b' * 64}
    (reference / 'data_identity.json').write_text(json.dumps(sequence_identity))
    monkeypatch.setattr(launcher, 'STUDY', study)
    monkeypatch.setattr(launcher, 'TOPIC', topic)
    captured = {}
    def create_run(_, recipe, label):
        captured.update(recipe=recipe, label=label)
        return study / 'runs/r000000000004'
    monkeypatch.setattr(launcher, 'create_run', create_run)
    monkeypatch.setattr(launcher, 'update_record', lambda *a, **kw: None)
    monkeypatch.setattr(launcher, 'refresh_selection', lambda *a: None)
    monkeypatch.setattr(launcher, 'regenerate_index', lambda *a: None)
    count = sequence_count or 1
    batch = np.arange(count * 5, dtype=np.int32).reshape(count, 5)
    def packed_batch_iterator(manifest_path, *, batch_size, sequence_length, start_sequence):
        assert manifest_path == str(manifest)
        assert batch_size == count
        assert sequence_length == source['data']['sequence_length']
        assert start_sequence == 0
        return iter([batch])
    monkeypatch.setattr(packed, 'PackedBatchIterator', packed_batch_iterator)
    launcher.prepare(parent, parameterization=parameterization, sequence_count=sequence_count)
    recipe = captured['recipe']
    assert recipe['model'] == source['model']
    assert recipe['data']['batch_size'] == count
    assert recipe['train']['seed'] == source['train']['seed']
    for key in ('data_start', 'max_updates', 'initial_updates', 'plateau_window',
                'plateau_tolerance', 'confirmation_updates',
                'learning_rate_decay', 'beta1', 'beta2', 'epsilon', 'clip_norm'):
        assert recipe['oracle'][key] == prior['oracle'][key]
    assert recipe['oracle']['parameterization'] == parameterization
    assert recipe['oracle']['sequence_count'] == count
    if parameterization == 'query_coefficient_generator':
        expected_role = 'query_generator_single_sequence_m4'
        assert captured['label'] == 'query-generator-m4-single-sequence'
        assert recipe['oracle']['batch_sha256'] == sequence_identity['batch_sha256']
        for key in ('microbatch_size', 'trainable_parameters', 'learning_rate'):
            assert recipe['oracle'][key] == prior['oracle'][key]
    else:
        suffix = 'single_sequence' if count == 1 else 'full_batch'
        expected_role = launcher.fitting_role(f'free_coefficients_{suffix}', degree)
        assert captured['label'] == f'free-coefficients-m{degree}-' + suffix.replace('_', '-')
        assert recipe['oracle']['batch_sha256'] == hashlib.sha256(batch.tobytes()).hexdigest()
        assert recipe['oracle']['microbatch_size'] == recipe['train']['micro_batch_size'] == min(count, 8)
        assert recipe['oracle']['learning_rate'] == recipe['optim']['learning_rate'] == 0.01
        assert recipe['oracle']['trainable_parameters'] == ['per_example_effective_monomial_coefficients']
        assert recipe['oracle']['initialization'] == "capture actual coefficients on each sequence's original forward pass"
        assert recipe['oracle']['gradient_scaling'] == 'per-sequence mean NLL, clipped independently per sequence'
        assert recipe['oracle']['optimizer_coordinates'] == 'factorial(order)*effective_monomial_coefficient'
        assert recipe['oracle']['frozen_parameters'].startswith('all pretrained parameters;')
    assert recipe['condition'] == f'coefficient_oracle/{expected_role}'
    assert recipe['oracle']['parent_checkpoint_sha256'] == descriptor['sha256']
    assert recipe['oracle']['parent_config_sha256'] == launcher.sha256(parent / 'config.yaml')
    selection = yaml.safe_load((topic / 'selection.yaml').read_text())
    assert selection['runs'][0] == old_entry
    assert selection['runs'][1]['role'] == expected_role


@pytest.mark.parametrize('returncode,stdout,stderr,expected', [
    (0, 'RUNNING\n', '', 'RUNNING'),
    (0, '', '', ''),
    (1, '', 'slurm_load_jobs error: Invalid job id specified', ''),
])
def test_live_job_lookup_accepts_expired_ids(monkeypatch, returncode, stdout, stderr, expected):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr=stderr))
    assert launcher.live_job_state('224999') == expected


def test_live_job_lookup_preserves_scheduler_errors(monkeypatch):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1, stdout='', stderr='scheduler unavailable'))
    with pytest.raises(RuntimeError, match='scheduler unavailable'):
        launcher.live_job_state('224999')


@pytest.mark.parametrize('gpu_type,gres', [(None, 'gpu:1'), ('B300', 'gpu:b300:1'), ('H100', 'gpu:h100:1')])
def test_submit_selects_gpu_without_changing_run_recipe(tmp_path, monkeypatch, gpu_type, gres):
    monkeypatch.setattr(launcher, 'STUDY', tmp_path)
    monkeypatch.setattr(launcher, 'load_run', lambda _: {'status': 'planned'})
    monkeypatch.setattr(launcher, 'validate_record', lambda *a, **kw: None)
    monkeypatch.setattr(launcher, 'changes', lambda *a: [])
    monkeypatch.setattr(launcher, 'frozen_hash', lambda *a: 'f' * 64)
    monkeypatch.setattr(launcher, 'regenerate_index', lambda *a: None)
    captured = {}
    def check_output(command, **kwargs):
        if command[0] == 'git':
            return 'c' * 40 + '\n'
        captured['command'] = command
        return '225100\n'
    monkeypatch.setattr(launcher.subprocess, 'check_output', check_output)
    monkeypatch.setattr(launcher, 'update_record', lambda run, changes: captured.update(record=changes))
    launcher.submit('r000000000008', partition='long', time_limit='06:00:00', gpu_type=gpu_type)
    assert f'--gres={gres}' in captured['command']
    assert '--partition=long' in captured['command']
    assert '--time=06:00:00' in captured['command']
    assert captured['record']['submission']['command'] == captured['command']
    assert captured['record']['submission']['slurm_job_id'] == '225100'
