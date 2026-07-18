"""Onboarding config (§3) + field-matching schema (§7.1)."""

from __future__ import annotations

import pytest

from opaque.config.field_schema import FieldSchema, FieldRule, load_field_schema
from opaque.config.loader import ConfigNotFound, load_config, scaffold_config


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(ConfigNotFound):
        load_config(tmp_path)


def test_scaffold_then_load_roundtrip(tmp_path):
    scaffold_config(tmp_path, project='demo')
    with pytest.raises(FileExistsError):
        scaffold_config(tmp_path, project='demo')

    cfg = load_config(tmp_path)
    assert cfg.project == 'demo'
    tool = cfg.tool('my_tool')
    assert tool.task_type == 'extraction'
    assert tool.prompts['system'] == 'prompts/system.txt'
    assert '{input}' in tool.invocation.command
    assert '{output_dir}' in tool.invocation.command


def test_tool_lookup_unknown_lists_available(tmp_path):
    scaffold_config(tmp_path, project='demo')
    cfg = load_config(tmp_path)
    with pytest.raises(KeyError, match='my_tool'):
        cfg.tool('nope')


def test_field_schema_wildcard_matches_array_index_only():
    schema = FieldSchema(rules={'extractionData.*.orderNumber': FieldRule(type='string')})
    # '*' matches an array index...
    assert schema.match('extractionData.0.orderNumber') is not None
    assert schema.match('extractionData.5.orderNumber').type == 'string'
    # ...but not a non-numeric segment, and not a different length.
    assert schema.match('extractionData.foo.orderNumber') is None
    assert schema.match('extractionData.0.orderNumber.0') is None
    assert schema.match('extractionData.0.otherField') is None


def test_load_field_schema_parses_tolerances(tmp_path):
    (tmp_path / 'field_schema.yaml').write_text(
        'extractionData.*.total:\n'
        '  type: numeric\n'
        '  numeric_tolerance: 0.01\n'
        'extractionData.*.id:\n'
        '  type: string\n'
    )
    schema = load_field_schema(tmp_path / 'field_schema.yaml')
    total = schema.match('extractionData.2.total')
    assert total.type == 'numeric'
    assert total.numeric_tolerance == 0.01
    assert schema.match('extractionData.2.id').type == 'string'
