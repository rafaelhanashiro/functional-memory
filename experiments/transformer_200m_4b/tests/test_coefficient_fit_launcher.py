"""Selected fitting inputs must remain explicit when coefficients become static."""
import pytest

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
