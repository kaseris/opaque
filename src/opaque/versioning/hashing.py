"""Content hashing — the *identity* half of versioning (§4).

Two different commits can produce a byte-identical file; the content hash is what decides
whether two runs used the same version, while the git commit is provenance only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HASH_LEN = 12


def content_hash(path: str | Path, length: int = HASH_LEN) -> str:
    """sha256 of a file's content, first ``length`` hex chars (§4.1). For a directory
    (an eval set kept as one-file-per-sample, §4.2) hash the sorted relative paths and
    their bytes together, so the digest covers the whole set deterministically.
    """
    p = Path(path)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(x for x in p.rglob('*') if x.is_file()):
            h.update(str(f.relative_to(p)).encode())
            h.update(b'\0')
            h.update(f.read_bytes())
            h.update(b'\0')
        return h.hexdigest()[:length]
    return hashlib.sha256(p.read_bytes()).hexdigest()[:length]


def prompt_bundle_hash(role_to_content_hash: dict[str, str], length: int = HASH_LEN) -> str:
    """Hash over the sorted ``{role: content_hash}`` mapping (§4.1) — a single value to
    check "did the overall prompt configuration change at all", while each role's hash
    stays independently queryable.
    """
    canonical = json.dumps(role_to_content_hash, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]
