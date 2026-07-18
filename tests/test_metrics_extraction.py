"""Extraction metrics (§6.2)."""

from __future__ import annotations

import pytest

from opaque.matching.results import FieldResult
from opaque.metrics import build
from opaque.schema.models import Sample


def _samples():
    return [
        Sample(sample_id='1', raw_file_name='1.pdf',
               gold={'a': 'x', 'b': 'y'}, prediction={'a': 'x', 'b': 'z'}),   # correct + incorrect
        Sample(sample_id='2', raw_file_name='2.pdf',
               gold={'a': '1'}, prediction={'a': '1', 'c': 'hallucinated'}),   # correct + extra
        Sample(sample_id='3', raw_file_name='3.pdf',
               gold=None, prediction={'a': 'q'}),                              # no gold
    ]


def test_extraction_compute_aggregates():
    m = build('extraction').compute(_samples())
    assert m['labeled_sample_count'] == 2
    assert m['unlabeled_sample_count'] == 1
    # correct=2, incorrect=1, missing=0 → scored=3 → accuracy 2/3.
    assert m['field_accuracy'] == pytest.approx(2 / 3)
    assert m['fields_correct'] == 2
    assert m['fields_extra'] == 1
    # per-sample: s1=0.5, s2=1.0, s3 excluded (no scored fields) → mean 0.75.
    assert m['per_sample_accuracy_mean'] == pytest.approx(0.75)


def test_extraction_artifacts_carry_field_results():
    arts = build('extraction').artifacts(_samples())
    assert all(isinstance(r, FieldResult) for r in arts['field_results'])
    # s1: a,b  s2: a,c  s3: a(no_gold) → 5 rows.
    assert len(arts['field_results']) == 5
    assert arts['status_counts']['no_gold'] == 1
    assert len(arts['per_sample']) == 3
