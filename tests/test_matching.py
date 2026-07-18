"""Field-level matching (spec §7): flattening, inference, comparators, union-of-paths."""

from __future__ import annotations

from opaque.config.field_schema import FieldRule, FieldSchema
from opaque.matching import paths
from opaque.matching.comparators import compare
from opaque.matching.field_match import match_sample
from opaque.matching.inference import infer_type
from opaque.matching.results import Status, status_counts
from opaque.schema.models import Sample


# --- flatten -----------------------------------------------------------------------------

def test_flatten_uses_dot_index_notation():
    obj = {'extractionData': [{'field': [{'sub': 'v'}]}]}
    assert paths.flatten(obj) == {'extractionData.0.field.0.sub': 'v'}


def test_natural_key_orders_indices_numerically():
    ordered = sorted(['items.10', 'items.2', 'items.1'], key=paths.natural_key)
    assert ordered == ['items.1', 'items.2', 'items.10']


# --- inference ---------------------------------------------------------------------------

def test_infer_type_from_shape():
    assert infer_type(42) == 'numeric'
    assert infer_type('1,234.50') == 'numeric'
    assert infer_type('2024-01-02') == 'date'
    assert infer_type(True) == 'boolean'
    assert infer_type('yes') == 'boolean'
    assert infer_type('Acme Corp') == 'string'
    # A numeric-looking id infers numeric — the case the override schema exists to fix.
    assert infer_type('00123') == 'numeric'


# --- comparators -------------------------------------------------------------------------

def test_numeric_comparator_strips_symbols_and_honors_tolerance():
    assert compare('numeric', '$1,234.50', 1234.50)[0] is True
    assert compare('numeric', '1234.51', 1234.50)[0] is False  # near-exact by default
    loosened = FieldRule(numeric_tolerance=0.001)
    assert compare('numeric', '1234.51', 1234.50, loosened)[0] is True


def test_date_comparator_is_format_agnostic():
    assert compare('date', '2024/01/02', '2024-01-02')[0] is True
    assert compare('date', 'Jan 2, 2024', '2024-01-02')[0] is True
    assert compare('date', '2024-01-03', '2024-01-02')[0] is False


def test_boolean_and_string_comparators():
    assert compare('boolean', 'Yes', True)[0] is True
    assert compare('boolean', 'no', False)[0] is True
    assert compare('string', '  Hello ', 'hello')[0] is True
    assert compare('string', 'abc', 'abd')[0] is False
    assert compare('string', None, None)[0] is True
    # comparator_used is reported back for transparency (§7.2)
    assert compare('numeric', 1, 1)[1] == 'numeric'


# --- union-of-paths matching -------------------------------------------------------------

def test_match_sample_classifies_correct_incorrect_missing_extra():
    sample = Sample(
        sample_id='s1',
        raw_file_name='s1.pdf',
        gold={'a': 'x', 'b': 'right', 'c': 'only_gold'},
        prediction={'a': 'x', 'b': 'wrong', 'd': 'only_pred'},
    )
    results = {r.json_path: r for r in match_sample(sample)}
    assert results['a'].status == Status.CORRECT
    assert results['b'].status == Status.INCORRECT
    assert results['c'].status == Status.MISSING
    assert results['d'].status == Status.EXTRA
    assert results['a'].comparator_used == 'string'


def test_no_gold_sample_lists_predictions_as_no_gold():
    sample = Sample(sample_id='s2', gold=None, prediction={'a': '1', 'b': '2'})
    results = match_sample(sample)
    assert {r.status for r in results} == {Status.NO_GOLD}
    assert {r.json_path for r in results} == {'a', 'b'}


def test_override_schema_forces_string_match_on_numeric_id():
    sample = Sample(
        sample_id='s3',
        gold={'extractionData': [{'orderNumber': '00123'}]},
        prediction={'extractionData': [{'orderNumber': '123'}]},
    )
    # Inference alone treats it numerically → 00123 == 123 → falsely "correct".
    assert match_sample(sample)[0].status == Status.CORRECT
    # The override schema forces exact string matching → correctly "incorrect".
    schema = FieldSchema(rules={'extractionData.*.orderNumber': FieldRule(type='string')})
    forced = match_sample(sample, schema)[0]
    assert forced.status == Status.INCORRECT
    assert forced.comparator_used == 'string'


def test_positional_array_shift_cascades(capsys):
    # Model drops the first array item; every later item shifts by one index (§7.3).
    sample = Sample(
        sample_id='s4',
        gold={'items': [{'v': 'a'}, {'v': 'b'}, {'v': 'c'}]},
        prediction={'items': [{'v': 'b'}, {'v': 'c'}]},
    )
    statuses = [r.status for r in match_sample(sample)]
    # A single dropped item surfaces as a run of incorrects + a trailing missing,
    # not a single clean miss — read as "check for an array shift".
    assert statuses == [Status.INCORRECT, Status.INCORRECT, Status.MISSING]


def test_status_counts_helper():
    sample = Sample(
        sample_id='s5',
        gold={'a': '1', 'b': '2'},
        prediction={'a': '1', 'c': '3'},
    )
    counts = status_counts(match_sample(sample))
    assert counts['correct'] == 1 and counts['missing'] == 1 and counts['extra'] == 1
