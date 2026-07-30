"""Installing the ``pre-push`` hook into a target repo.

The hook is a three-line shell stub that execs opaque; all logic lives in ``hooks.push`` so
an installed hook keeps working across upgrades without reinstalling. It records an absolute
interpreter path because git runs hooks with a minimal environment in which the developer's
virtualenv is usually not on PATH.
"""

from __future__ import annotations

import shlex
import stat
import subprocess
import sys
from pathlib import Path

MARKER = '# opaque-managed-hook v1'
HOOK_NAME = 'pre-push'

# Pushes to anything else (a backup remote, a fork) are not PR events, so they should not
# cost an evaluation. Pass ``remote=None`` to evaluate pushes to every remote.
DEFAULT_REMOTE = 'origin'

_TEMPLATE = """\
#!/bin/sh
{marker}
# Managed by `opaque hook install` — edits here are lost on reinstall.
# Runs the tracked evaluation locally when a push touches a prompt or field-schema file.
# Skip a single push with:  OPAQUE_SKIP=1 git push
if [ -n "${{OPAQUE_SKIP:-}}" ]; then
  exit 0
fi
{remote_guard}exec {python} -m opaque check --repo {repo} --stdin{extra}
"""

# Filtering by remote in the shell rather than in `opaque check` keeps an irrelevant push
# (a backup remote, a fork) from paying interpreter startup just to decide it is irrelevant.
_REMOTE_GUARD = """\
# Only pushes to this remote are evaluated. Git passes the remote *name* as $1; pushing by URL
# (git push git@host:repo.git ...) passes the URL instead, and is therefore skipped.
if [ "$1" != {remote} ]; then
  exit 0
fi
"""


class HookError(RuntimeError):
    """The hook could not be installed or removed."""


def hooks_dir(repo: str | Path) -> Path:
    """Where git will look for hooks in this repo.

    Honours ``core.hooksPath`` and resolves correctly for worktrees and submodules, where
    ``.git`` is a file rather than a directory.
    """
    repo = Path(repo).resolve()

    def _git(args: list[str]) -> str:
        proc = subprocess.run(['git', *args], cwd=str(repo), capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else ''

    configured = _git(['config', '--get', 'core.hooksPath'])
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else repo / path

    relative = _git(['rev-parse', '--git-path', 'hooks'])
    if not relative:
        raise HookError(f'{repo} is not a git repository.')
    path = Path(relative)
    return path if path.is_absolute() else repo / path


def hook_path(repo: str | Path) -> Path:
    return hooks_dir(repo) / HOOK_NAME


def is_ours(path: Path) -> bool:
    return path.exists() and MARKER in path.read_text(errors='replace')


def render_hook(
    repo: str | Path,
    *,
    python: str | None = None,
    gate: bool = False,
    comment: bool = False,
    tracking_uri: str | None = None,
    remote: str | None = DEFAULT_REMOTE,
) -> str:
    extra = ''
    if tracking_uri:
        extra += f' --tracking-uri {shlex.quote(tracking_uri)}'
    if gate:
        extra += ' --gate'
    if comment:
        extra += ' --comment'
    return _TEMPLATE.format(
        marker=MARKER,
        python=shlex.quote(python or sys.executable),
        repo=shlex.quote(str(Path(repo).resolve())),
        extra=extra,
        remote_guard=_REMOTE_GUARD.format(remote=shlex.quote(remote)) if remote else '',
    )


def install(
    repo: str | Path,
    *,
    python: str | None = None,
    gate: bool = False,
    comment: bool = False,
    tracking_uri: str | None = None,
    remote: str | None = DEFAULT_REMOTE,
    force: bool = False,
) -> Path:
    """Write the pre-push hook. Refuses to overwrite a hook opaque did not write."""
    path = hook_path(repo)
    if path.exists() and not is_ours(path) and not force:
        raise HookError(
            f'{path} already exists and was not written by opaque. '
            'Move it aside, or pass --force to replace it.'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_hook(
            repo, python=python, gate=gate, comment=comment,
            tracking_uri=tracking_uri, remote=remote,
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def uninstall(repo: str | Path) -> Path | None:
    """Remove opaque's hook. Leaves a hook opaque did not write alone."""
    path = hook_path(repo)
    if not path.exists():
        return None
    if not is_ours(path):
        raise HookError(f'{path} was not written by opaque — leaving it untouched.')
    path.unlink()
    return path
