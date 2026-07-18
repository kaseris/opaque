"""Classification metrics (§6.1)."""

from __future__ import annotations

import pytest

from opaque.metrics import build
from opaque.metrics.classification import ClassificationMetrics
from opaque.schema.models import Sample


def _single_label():
    return [
        Sample(sample_id='1', gold='A', prediction='A'),
        Sample(sample_id='2', gold='A', prediction='B'),
        Sample(sample_id='3', gold='B', prediction='B'),
        Sample(sample_id='4', gold=None, prediction='B'),  # unlabeled
    ]


def test_registry_builds_classification():
    m = build('classification')
    assert isinstance(m, ClassificationMetrics)
    assert m.task_type == 'classification'


def test_single_label_accuracy_and_counts():
    m = build('classification').compute(_single_label())
    assert m['labeled_sample_count'] == 3
    assert m['unlabeled_sample_count'] == 1
    assert m['accuracy'] == pytest.approx(2 / 3)
    assert m['macro_precision'] == pytest.approx(0.75)
    assert m['macro_recall'] == pytest.approx(0.75)
    assert m['macro_f1'] == pytest.approx(2 / 3, abs=1e-3)
    # For single-label multiclass, micro P/R/F1 collapse to accuracy.
    assert m['micro_f1'] == pytest.approx(2 / 3)


def test_no_labeled_samples_returns_only_counts():
    m = build('classification').compute([Sample(sample_id='1', gold=None, prediction='A')])
    assert m['labeled_sample_count'] == 0
    assert 'accuracy' not in m


def test_multi_label_uses_exact_match_accuracy():
    samples = [
        Sample(sample_id='1', gold=['A', 'B'], prediction=['A', 'B']),
        Sample(sample_id='2', gold=['A'], prediction=['A', 'B']),
    ]
    m = build('classification').compute(samples)
    assert m['accuracy'] == pytest.approx(0.5)
    assert m['micro_recall'] == pytest.approx(1.0)  # every gold label was predicted


def test_confusion_matrix_in_artifacts():
    arts = build('classification').artifacts(_single_label())
    cm = arts['confusion_matrix']
    assert cm['labels'] == ['A', 'B']
    # Rows = gold, cols = predicted: A→A once, A→B once, B→B once.
    assert cm['matrix'] == [[1, 1], [0, 1]]
    assert {e['sample_id'] for e in arts['errors']} == {'2'}
