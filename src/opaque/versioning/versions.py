"""Version models and computation (spec §4).

Every artifact that affects "what does correct mean" is versioned the same way: git commit
(provenance) + content hash (identity). A prompt file's tracked version only changes when
its *own* content changes — not on every unrelated commit elsewhere in the repo.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from . import git, hashing


class FileVersion(BaseModel):
    """The version of a single versioned path (a prompt file, the eval set, the field
    schema). ``git_commit`` is provenance; ``content_hash`` is identity (§4)."""

    role: str | None = None
    path: str
    git_commit: str | None
    content_hash: str
    dirty: bool


def path_version(repo: str | Path, rel_path: str | Path, role: str | None = None) -> FileVersion:
    """Compute the version of a repo-relative path (file or directory)."""
    abs_path = Path(repo) / rel_path
    return FileVersion(
        role=role,
        path=str(rel_path),
        git_commit=git.last_commit_for_path(repo, rel_path),
        content_hash=hashing.content_hash(abs_path),
        dirty=git.is_dirty(repo, rel_path),
    )


class PromptBundle(BaseModel):
    """The set of prompt files selected for a run, each versioned individually, plus the
    aggregate ``bundle_hash`` (§4.1)."""

    roles: dict[str, FileVersion]
    bundle_hash: str

    @property
    def any_dirty(self) -> bool:
        return any(v.dirty for v in self.roles.values())

    @property
    def dirty_roles(self) -> list[str]:
        return [role for role, v in self.roles.items() if v.dirty]


def prompt_bundle(repo: str | Path, role_to_path: dict[str, str]) -> PromptBundle:
    """Version each selected prompt file individually and compute the aggregate bundle
    hash over the sorted ``{role: content_hash}`` mapping (§4.1)."""
    roles = {role: path_version(repo, path, role=role) for role, path in role_to_path.items()}
    bundle_hash = hashing.prompt_bundle_hash({r: v.content_hash for r, v in roles.items()})
    return PromptBundle(roles=roles, bundle_hash=bundle_hash)
