"""Per-(sample, field) match results — the shared currency between the extraction metrics
(§6.2) and the Excel report (§10), so matching logic lives in exactly one place."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class Status(str, Enum):
    CORRECT = 'correct'      # present in both, comparator matched
    INCORRECT = 'incorrect'  # present in both, comparator did not match
    MISSING = 'missing'      # gold has it, prediction does not
    EXTRA = 'extra'          # prediction has it, gold does not (hallucinated)
    NO_GOLD = 'no_gold'      # sample carries no gold at all (§5/§10) — not scored


class FieldResult(BaseModel):
    sample_id: str
    raw_file_name: str | None
    json_path: str
    predicted_value: Any = None
    ground_truth_value: Any = None
    status: Status
    comparator_used: str | None = None


# Statuses that count toward extraction accuracy: a gold value was present to compare against.
SCORED_STATUSES = (Status.CORRECT, Status.INCORRECT, Status.MISSING)


def status_counts(results: list[FieldResult]) -> dict[str, int]:
    counts = {s.value: 0 for s in Status}
    for r in results:
        counts[r.status.value] += 1
    return counts


def field_accuracy(results: list[FieldResult]) -> tuple[int, int, float]:
    """Overall field accuracy (§6.2). Denominator is fields where gold was present
    (correct + incorrect + missing); ``extra`` and ``no_gold`` are excluded. Returns
    ``(correct, scored, accuracy)``."""
    correct = sum(r.status == Status.CORRECT for r in results)
    scored = sum(r.status in SCORED_STATUSES for r in results)
    return correct, scored, (correct / scored if scored else 0.0)


def per_sample_rollup(results: list[FieldResult]) -> list[dict]:
    """One row per document (§10 Sheet 3), preserving first-seen order. Samples with no
    scored fields (e.g. no gold) get ``accuracy_pct = None``."""
    groups: dict[str, list[FieldResult]] = {}
    for r in results:
        groups.setdefault(r.sample_id, []).append(r)
    rows = []
    for sid, rs in groups.items():
        correct, total, _ = field_accuracy(rs)
        rows.append({
            'sample_id': sid,
            'raw_file_name': rs[0].raw_file_name,
            'fields_correct': correct,
            'fields_total': total,
            'accuracy_pct': round(100 * correct / total, 2) if total else None,
        })
    return rows
