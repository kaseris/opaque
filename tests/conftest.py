"""Shared test fixtures — chiefly a throwaway git repo, since versioning (§4) and the
relabeling commit (§9) are defined against real git history."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


class GitRepo:
    """A small helper around a real git repo in a temp dir."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ['git', *args], cwd=self.path, check=True, capture_output=True, text=True
        )
        return result.stdout

    def init(self) -> 'GitRepo':
        self.git('init', '-q', '-b', 'main')
        # Repo-local identity + no signing, so commits work on any host config.
        self.git('config', 'user.email', 'test@opaque.dev')
        self.git('config', 'user.name', 'Opaque Test')
        self.git('config', 'commit.gpgsign', 'false')
        return self

    def write(self, relpath: str, content: str) -> Path:
        p = self.path / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def commit(self, message: str = 'commit', paths: list[str] | None = None) -> str:
        if paths is None:
            self.git('add', '-A')
        else:
            self.git('add', '--', *paths)
        self.git('commit', '-q', '-m', message)
        return self.git('rev-parse', 'HEAD').strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> GitRepo:
    return GitRepo(tmp_path).init()
