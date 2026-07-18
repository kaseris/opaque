"""Field-level value comparators (spec §7.2).

The comparator actually used for each field is returned alongside the result so the report
(§10) can make matches transparent rather than looking like unexplained leniency.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from dateutil import parser as date_parser

from ..config.field_schema import FieldRule, FieldType

# Near-exact by default; the field schema (§7.1) can loosen tolerance per field.
DEFAULT_REL_TOL = 1e-9
DEFAULT_ABS_FLOOR = 1e-9

_NUMERIC_STRIP = re.compile(r'[,$€£¥%\s]')
_TRUTHY = {'true', 'yes', 'y', 't', '1'}
_FALSY = {'false', 'no', 'n', 'f', '0'}


def compare(
    resolved_type: FieldType,
    predicted: Any,
    gold: Any,
    rule: FieldRule | None = None,
) -> tuple[bool, str]:
    """Compare a predicted vs gold value under ``resolved_type``. Returns
    ``(matched, comparator_used)``."""
    if predicted is None and gold is None:
        return True, resolved_type
    fn = _COMPARATORS[resolved_type]
    return fn(predicted, gold, rule), resolved_type


def _cmp_numeric(predicted: Any, gold: Any, rule: FieldRule | None) -> bool:
    try:
        p, g = _to_number(predicted), _to_number(gold)
    except (ValueError, TypeError):
        return False
    rel = rule.numeric_tolerance if rule and rule.numeric_tolerance is not None else DEFAULT_REL_TOL
    floor = rule.numeric_abs_floor if rule and rule.numeric_abs_floor is not None else DEFAULT_ABS_FLOOR
    return math.isclose(p, g, rel_tol=rel, abs_tol=floor)


def _cmp_date(predicted: Any, gold: Any, rule: FieldRule | None) -> bool:
    fmt = rule.date_format if rule else None
    try:
        return _to_date(predicted, fmt) == _to_date(gold, fmt)
    except (ValueError, OverflowError, TypeError):
        return False


def _cmp_boolean(predicted: Any, gold: Any, rule: FieldRule | None) -> bool:
    p, g = _to_bool(predicted), _to_bool(gold)
    if p is None or g is None:
        return _cmp_string(predicted, gold, rule)  # unnormalizable → exact string fallback
    return p == g


def _cmp_string(predicted: Any, gold: Any, rule: FieldRule | None) -> bool:
    return _norm_str(predicted) == _norm_str(gold)


_COMPARATORS = {
    'numeric': _cmp_numeric,
    'date': _cmp_date,
    'boolean': _cmp_boolean,
    'string': _cmp_string,
}


def _to_number(v: Any) -> float:
    if isinstance(v, bool):
        raise ValueError('bool is not numeric')
    if isinstance(v, (int, float)):
        return float(v)
    return float(_NUMERIC_STRIP.sub('', str(v).strip()))


def _to_date(v: Any, date_format: str | None) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    if date_format:
        return datetime.strptime(s, date_format).date()
    return date_parser.parse(s, fuzzy=False).date()


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return None


def _norm_str(v: Any) -> str:
    return str(v).strip().casefold()
