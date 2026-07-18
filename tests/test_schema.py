"""Per-sample schema + loading (§5)."""

from __future__ import annotations

import json

from opaque.schema.io import load_samples
from opaque.schema.models import Sample


def test_has_gold_reflects_nullable_gold():
    assert Sample(sample_id='1', gold='A', prediction='A').has_gold is True
    assert Sample(sample_id='2', gold=None, prediction='A').has_gold is False


def test_extra_fields_are_preserved():
    s = Sample.model_validate({'sample_id': '1', 'prediction': 'x', 'model_name': 'gpt-4o'})
    assert s.model_dump()['model_name'] == 'gpt-4o'


def test_load_samples_from_directory(tmp_path):
    for i in range(3):
        (tmp_path / f'{i}.json').write_text(
            json.dumps({'sample_id': str(i), 'raw_file_name': f'{i}.pdf', 'prediction': i})
        )
    samples = load_samples(tmp_path)
    assert [s.sample_id for s in samples] == ['0', '1', '2']


def test_load_samples_from_single_list_file(tmp_path):
    f = tmp_path / 'all.json'
    f.write_text(json.dumps([
        {'sample_id': 'a', 'gold': 'x', 'prediction': 'x'},
        {'sample_id': 'b', 'gold': None, 'prediction': 'y'},
    ]))
    samples = load_samples(f)
    assert len(samples) == 2
    assert samples[1].has_gold is False
