"""Field-type inference from the ground-truth value's shape (spec §7.1, default path).

An override schema (§7.1) takes precedence over inference for fields that would be inferred
wrong — a numeric-looking id like ``"00123"`` that must be matched as an exact string.
"""

from __future__ import annotations

import re
from typing import Any

from dateutil import parser as date_parser

from ..config.field_schema import FieldType

_BOOLEAN_STRINGS = {'true', 'false', 'yes', 'no', 'y', 'n', 't', 'f'}
_NUMERIC_STRIP = re.compile(r'[,$€£¥%\s]')


def infer_type(gold_value: Any) -> FieldType:
    """Infer a comparator type from the gold value: numeric shape → numeric, recognized
    date → date, truthy/falsy word → boolean, else string."""
    if isinstance(gold_value, bool):
        return 'boolean'
    if isinstance(gold_value, (int, float)):
        return 'numeric'
    if isinstance(gold_value, str):
        s = gold_value.strip()
        if s.lower() in _BOOLEAN_STRINGS:
            return 'boolean'
        if _looks_numeric(s):
            return 'numeric'
        if _looks_date(s):
            return 'date'
    return 'string'


def _looks_numeric(s: str) -> bool:
    if not s:
        return False
    try:
        float(_NUMERIC_STRIP.sub('', s))
        return True
    except ValueError:
        return False


def _looks_date(s: str) -> bool:
    # Require at least one digit so bare month names etc. don't get misread as dates.
    if not re.search(r'\d', s):
        return False
    try:
        date_parser.parse(s, fuzzy=False)
        return True
    except (ValueError, OverflowError):
        return False
