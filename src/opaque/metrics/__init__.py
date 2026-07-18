"""Metrics system (spec §6). Importing the package registers the built-in computers."""

from .base import MetricComputer
from .registry import build, register, task_types

# Import for side effect: registers 'classification' and 'extraction' in the registry.
from . import classification as classification  # noqa: F401
from . import extraction as extraction  # noqa: F401

__all__ = ['MetricComputer', 'build', 'register', 'task_types']
