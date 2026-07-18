"""Thin subprocess wrapper around git — just what versioning (§4) and the relabeling
commit (§9.2) need. Kept dependency-free (no GitPython) so behaviour stays transparent.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """A git command exited non-zero."""


def _run(repo: str | Path, args: list[str]) -> str:
    result = subprocess.run(
        ['git', *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _relpath(repo: str | Path, path: str | Path) -> str:
    """Path relative to the repo root, which is how we address files to git."""
    return os.path.relpath(Path(repo) / path if not Path(path).is_absolute() else path, repo)


def is_repo(repo: str | Path) -> bool:
    try:
        return _run(repo, ['rev-parse', '--is-inside-work-tree']).strip() == 'true'
    except GitError:
        return False


def last_commit_for_path(repo: str | Path, path: str | Path) -> str | None:
    """Most recent commit that modified *this specific path* (``git log -1 -- <path>``,
    not repo HEAD — §4.1). ``None`` when the path has no commit history yet (newly added
    and never committed).
    """
    out = _run(repo, ['log', '-1', '--format=%H', '--', _relpath(repo, path)]).strip()
    return out or None


def is_dirty(repo: str | Path, path: str | Path) -> bool:
    """True if the path has uncommitted changes — staged, unstaged, or untracked (§4.1)."""
    out = _run(repo, ['status', '--porcelain', '--', _relpath(repo, path)])
    return bool(out.strip())


def current_branch(repo: str | Path) -> str:
    return _run(repo, ['rev-parse', '--abbrev-ref', 'HEAD']).strip()


def head_commit(repo: str | Path) -> str | None:
    try:
        return _run(repo, ['rev-parse', 'HEAD']).strip()
    except GitError:
        return None


def add_and_commit(
    repo: str | Path,
    paths: list[str | Path],
    message: str,
    author: str | None = None,
) -> str:
    """Stage the given paths and create a single commit (the relabeling-session commit,
    §9.2). Returns the new commit hash. Author defaults to the repo's configured identity.
    """
    rels = [_relpath(repo, p) for p in paths]
    _run(repo, ['add', '--', *rels])
    args = ['commit', '-m', message]
    if author:
        args += ['--author', author]
    _run(repo, args)
    commit = head_commit(repo)
    assert commit is not None  # a successful commit always has a HEAD
    return commit
