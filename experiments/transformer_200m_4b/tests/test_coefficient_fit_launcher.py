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


def test_prepare_m4_preserves_parent_model_and_m8_fitting_recipe(tmp_path, monkeypatch):
    prior = original_query_fit()
    source = deepcopy(prior)
    for key in ('oracle', 'experiment_kind', 'condition', 'schema_version'):
        source.pop(key, None)
    source['model']['attention_degree'] = 4
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
    launcher.prepare(parent)
    recipe = captured['recipe']
    assert recipe['model'] == source['model']
    assert recipe['data']['batch_size'] == 1
    assert recipe['train']['seed'] == source['train']['seed']
    assert recipe['condition'] == 'coefficient_oracle/query_generator_single_sequence_m4'
    assert captured['label'] == 'query-generator-m4-single-sequence'
    for key in ('parameterization', 'sequence_count', 'microbatch_size', 'data_start',
                'trainable_parameters', 'max_updates', 'initial_updates', 'plateau_window',
                'plateau_tolerance', 'confirmation_updates', 'learning_rate',
                'learning_rate_decay', 'beta1', 'beta2', 'epsilon', 'clip_norm'):
        assert recipe['oracle'][key] == prior['oracle'][key]
    assert recipe['oracle']['batch_sha256'] == sequence_identity['batch_sha256']
    assert recipe['oracle']['parent_checkpoint_sha256'] == descriptor['sha256']
    selection = yaml.safe_load((topic / 'selection.yaml').read_text())
    assert selection['runs'][0] == old_entry
    assert selection['runs'][1]['role'] == 'query_generator_single_sequence_m4'


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
