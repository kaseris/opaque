"""Versioning (spec §4): git provenance + content-hash identity."""

from __future__ import annotations

from opaque.versioning import hashing, versions


def test_content_hash_is_stable_and_content_sensitive(git_repo):
    git_repo.write('a.txt', 'hello')
    h1 = hashing.content_hash(git_repo.path / 'a.txt')
    assert h1 == hashing.content_hash(git_repo.path / 'a.txt')
    assert len(h1) == hashing.HASH_LEN

    git_repo.write('a.txt', 'hello!')
    assert hashing.content_hash(git_repo.path / 'a.txt') != h1


def test_content_hash_of_directory_covers_all_files(git_repo):
    git_repo.write('data/1.json', '{"a": 1}')
    git_repo.write('data/2.json', '{"a": 2}')
    before = hashing.content_hash(git_repo.path / 'data')
    git_repo.write('data/2.json', '{"a": 3}')
    assert hashing.content_hash(git_repo.path / 'data') != before


def test_prompt_bundle_hash_is_order_independent(git_repo):
    a = {'system': 'aaa', 'extraction': 'bbb'}
    b = {'extraction': 'bbb', 'system': 'aaa'}
    assert hashing.prompt_bundle_hash(a) == hashing.prompt_bundle_hash(b)
    # A changed role hash changes the bundle.
    assert hashing.prompt_bundle_hash(a) != hashing.prompt_bundle_hash({**a, 'system': 'zzz'})


def test_path_version_tracks_only_its_own_file_history(git_repo):
    git_repo.write('a.txt', 'A1')
    git_repo.write('b.txt', 'B1')
    commit1 = git_repo.commit('add a and b')

    git_repo.write('b.txt', 'B2')
    commit2 = git_repo.commit('modify b only')

    va = versions.path_version(git_repo.path, 'a.txt')
    vb = versions.path_version(git_repo.path, 'b.txt')

    # a.txt's tracked commit did NOT move to commit2 — it changed only when a.txt changed.
    assert va.git_commit == commit1
    assert vb.git_commit == commit2
    assert va.dirty is False and vb.dirty is False


def test_path_version_dirty_and_untracked(git_repo):
    git_repo.write('p.txt', 'v1')
    git_repo.commit('add p')

    clean = versions.path_version(git_repo.path, 'p.txt')
    assert clean.dirty is False

    git_repo.write('p.txt', 'v2-uncommitted')
    dirty = versions.path_version(git_repo.path, 'p.txt')
    assert dirty.dirty is True

    git_repo.write('never.txt', 'x')
    untracked = versions.path_version(git_repo.path, 'never.txt')
    assert untracked.git_commit is None
    assert untracked.dirty is True


def test_prompt_bundle_dirty_reporting(git_repo):
    git_repo.write('prompts/system.txt', 'sys')
    git_repo.write('prompts/extract.txt', 'ext')
    git_repo.commit('add prompts')
    git_repo.write('prompts/extract.txt', 'ext-edited')

    bundle = versions.prompt_bundle(
        git_repo.path,
        {'system': 'prompts/system.txt', 'extraction': 'prompts/extract.txt'},
    )
    assert bundle.any_dirty is True
    assert bundle.dirty_roles == ['extraction']
    assert len(bundle.bundle_hash) == hashing.HASH_LEN
    assert set(bundle.roles) == {'system', 'extraction'}
