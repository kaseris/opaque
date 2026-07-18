"""Pluggable metrics interface (spec §6).

One implementation per task type. ``compute`` returns the aggregate metrics logged to
MLflow; ``artifacts`` returns supporting detail (confusion matrix, field-level results)
logged as run artifacts. New task types are added by implementing this interface and
registering it — nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config.field_schema import FieldSchema
from ..schema.models import Sample


class MetricComputer(ABC):
    task_type: str

    def __init__(self, field_schema: FieldSchema | None = None):
        # Extraction uses the schema; classification ignores it (uniform construction
        # from the registry).
        self.field_schema = field_schema

    @abstractmethod
    def compute(self, samples: list[Sample]) -> dict[str, float]:
        """Aggregate metrics, logged to MLflow. Always includes labeled/unlabeled counts."""

    @abstractmethod
    def artifacts(self, samples: list[Sample]) -> dict[str, Any]:
        """Supporting artifacts (confusion matrix, field-level detail, etc.)."""


def split_labeled(samples: list[Sample]) -> tuple[list[Sample], int]:
    """Samples carrying gold, plus the count of those without (§6 partial/absent gold)."""
    labeled = [s for s in samples if s.has_gold]
    return labeled, len(samples) - len(labeled)
