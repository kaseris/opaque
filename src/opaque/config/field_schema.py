"""Per-field type/tolerance override schema for extraction matching (spec §7.1).

Keyed by ``json_path`` *pattern* with array indices wildcarded (``*``), so one entry
applies across every array position and sample — e.g. ``extractionData.*.orderNumber``.
Used for fields that would be inferred wrong (a numeric-looking id like ``"00123"`` that
needs exact string matching, not numeric coercion that drops the leading zero), and to
carry numeric tolerance / date-format hints.

The YAML file is a top-level mapping of ``pattern -> rule``:

    extractionData.*.orderNumber:
      type: string
    extractionData.*.totalAmount:
      type: numeric
      numeric_tolerance: 0.01
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

FieldType = Literal['numeric', 'date', 'boolean', 'string']


class FieldRule(BaseModel):
    # Force a comparator type, overriding shape inference (§7.1).
    type: FieldType | None = None
    # Relative tolerance for numeric comparison (§7.2); falls back to the comparator default.
    numeric_tolerance: float | None = None
    numeric_abs_floor: float | None = None
    # Hint for parsing dates in a known non-standard format.
    date_format: str | None = None


class FieldSchema(BaseModel):
    rules: dict[str, FieldRule] = {}

    def match(self, json_path: str) -> FieldRule | None:
        """Return the rule whose pattern matches this concrete ``json_path``, or ``None``.
        A ``*`` segment matches only an array-index segment (all digits), so a wildcard
        never accidentally matches a field name."""
        for pattern, rule in self.rules.items():
            if _pattern_matches(pattern, json_path):
                return rule
        return None


def _pattern_matches(pattern: str, json_path: str) -> bool:
    pat = pattern.split('.')
    concrete = json_path.split('.')
    if len(pat) != len(concrete):
        return False
    for p_seg, c_seg in zip(pat, concrete):
        if p_seg == '*':
            if not c_seg.isdigit():
                return False
        elif p_seg != c_seg:
            return False
    return True


def load_field_schema(path: str | Path) -> FieldSchema:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return FieldSchema(rules={k: FieldRule.model_validate(v or {}) for k, v in data.items()})
