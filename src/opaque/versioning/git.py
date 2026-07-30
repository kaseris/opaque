"""Thin subprocess wrapper around git — just what versioning (§4) and the relabeling
commit (§9.2) need. Kept dependency-free (no GitPython) so behaviour stays transparent.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """A git command exited non-zero."""


# git uses an all-zero sha to mean "no such object" — a branch being created (no remote
# counterpart yet) or deleted. Both show up on a pre-push hook's stdin.
NULL_SHA = '0' * 40


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


def default_branch(repo: str | Path) -> str:
    """The repo's integration branch — what a PR would target.

    Prefers what the remote advertises (``refs/remotes/origin/HEAD``), falls back to a local
    ``main``/``master``, then to ``main``. Used to pick the merge base for a branch that has
    no remote counterpart yet (§ hook: the push that opens a PR).
    """
    try:
        ref = _run(repo, ['symbolic-ref', '--quiet', 'refs/remotes/origin/HEAD']).strip()
        if ref:
            return ref.rsplit('/', 1)[-1]
    except GitError:
        pass
    for name in ('main', 'master'):
        try:
            _run(repo, ['rev-parse', '--verify', '--quiet', f'refs/heads/{name}'])
            return name
        except GitError:
            continue
    return 'main'


def merge_base(repo: str | Path, a: str, b: str) -> str | None:
    """Common ancestor of two revisions, or ``None`` when they share no history."""
    try:
        return _run(repo, ['merge-base', a, b]).strip() or None
    except GitError:
        return None


def rev_exists(repo: str | Path, rev: str) -> bool:
    try:
        _run(repo, ['rev-parse', '--verify', '--quiet', f'{rev}^{{commit}}'])
        return True
    except GitError:
        return False


def changed_paths(repo: str | Path, base: str | None, head: str) -> list[str]:
    """Repo-relative paths that differ between ``base`` and ``head``.

    With no ``base`` (a branch sharing no history with the integration branch) this falls
    back to the paths touched by ``head`` itself.
    """
    args = ['diff', '--name-only', f'{base}..{head}'] if base else ['show', '--name-only', '--format=', head]
    return sorted({line.strip() for line in _run(repo, args).splitlines() if line.strip()})


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
