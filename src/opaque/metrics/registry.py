"""Task-type registry (spec §6). Selected via ``--task-type``."""

from __future__ import annotations

from ..config.field_schema import FieldSchema
from .base import MetricComputer

_REGISTRY: dict[str, type[MetricComputer]] = {}


def register(task_type: str):
    def decorator(cls: type[MetricComputer]) -> type[MetricComputer]:
        cls.task_type = task_type
        _REGISTRY[task_type] = cls
        return cls

    return decorator


def build(task_type: str, field_schema: FieldSchema | None = None) -> MetricComputer:
    try:
        cls = _REGISTRY[task_type]
    except KeyError:
        available = ', '.join(sorted(_REGISTRY)) or '(none registered)'
        raise KeyError(f"Unknown task_type '{task_type}'. Registered: {available}")
    return cls(field_schema=field_schema)


def task_types() -> list[str]:
    return sorted(_REGISTRY)
