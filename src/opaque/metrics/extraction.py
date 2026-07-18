"""Extraction metrics (spec §6.2).

Overall field accuracy + per-sample accuracy as headline metrics. Full field-level detail
is not exploded into MLflow metrics — it lives in the Excel report artifact (§10); only the
aggregates are logged for cross-run comparability. The report and these metrics consume the
same ``FieldResult`` list (§7), so matching logic exists in exactly one place.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from ..matching import field_match
from ..matching.results import (
    Status,
    field_accuracy,
    per_sample_rollup,
    status_counts,
)
from ..schema.models import Sample
from .base import MetricComputer, split_labeled
from .registry import register


@register('extraction')
class ExtractionMetrics(MetricComputer):
    def compute(self, samples: list[Sample]) -> dict[str, float]:
        labeled, unlabeled = split_labeled(samples)
        results = field_match.match_samples(samples, self.field_schema)
        correct, scored, accuracy = field_accuracy(results)
        counts = status_counts(results)

        metrics: dict[str, float] = {
            'labeled_sample_count': float(len(labeled)),
            'unlabeled_sample_count': float(unlabeled),
            'field_accuracy': accuracy,
            'fields_correct': float(counts['correct']),
            'fields_incorrect': float(counts['incorrect']),
            'fields_missing': float(counts['missing']),
            'fields_extra': float(counts['extra']),
            'fields_scored': float(scored),
        }
        per_sample = [
            row['accuracy_pct'] / 100
            for row in per_sample_rollup(results)
            if row['accuracy_pct'] is not None
        ]
        metrics['per_sample_accuracy_mean'] = mean(per_sample) if per_sample else 0.0
        return metrics

    def artifacts(self, samples: list[Sample]) -> dict[str, Any]:
        results = field_match.match_samples(samples, self.field_schema)
        return {
            'field_results': results,  # list[FieldResult] — consumed by the Excel report
            'status_counts': status_counts(results),
            'per_sample': per_sample_rollup(results),
        }
