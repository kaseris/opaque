"""Flatten nested JSON into ``{json_path: value}`` using dot/index notation, e.g.
``extractionData.0.fieldName.0.subFieldName`` (spec §7, §10). Arrays are addressed
positionally by index — the basis for positional array alignment (§7.3).
"""

from __future__ import annotations

from typing import Any


def flatten(obj: Any, prefix: str = '') -> dict[str, Any]:
    """Flatten to leaf scalars keyed by dot/index path. Objects and arrays recurse;
    everything else (str/int/float/bool/None) is a leaf. Empty objects/arrays contribute
    no paths."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            out.update(flatten(value, path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            path = f'{prefix}.{i}' if prefix else str(i)
            out.update(flatten(value, path))
    else:
        out[prefix] = obj
    return out


def natural_key(path: str) -> list:
    """Sort key that orders numeric path segments numerically (so ``.2`` sorts before
    ``.10``), keeping flattened paths readable in the report."""
    return [(int(seg), '') if seg.isdigit() else (float('inf'), seg) for seg in path.split('.')]
