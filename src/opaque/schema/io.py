"""Loading per-sample JSON produced by the eval script (spec §5)."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Sample


def load_samples(path: str | Path) -> list[Sample]:
    """Load samples from either a directory of per-sample ``*.json`` files or a single
    ``.json`` file containing a list of samples. Directory files are sorted by name so the
    ordering is stable across runs.
    """
    p = Path(path)
    if p.is_dir():
        return [_load_one(f) for f in sorted(p.glob('*.json'))]
    if not p.exists():
        raise FileNotFoundError(f'No sample data at {p}')
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return [Sample.model_validate(item) for item in data]
    return [Sample.model_validate(data)]


def _load_one(file: Path) -> Sample:
    return Sample.model_validate_json(file.read_text())
