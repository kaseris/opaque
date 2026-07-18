"""Relabeling backend (spec §9): in-memory edits + one git commit per session."""

from __future__ import annotations

import json
import subprocess

from fastapi.testclient import TestClient

from opaque.relabel.app import create_app

_CONFIG = '''\
project: mini
tools:
  - name: demo
    task_type: extraction
    prompts:
      system: prompts/system.txt
    eval_data: eval_data
    invocation:
      command: python noop.py
'''


def _project(git_repo):
    git_repo.write('prompts/system.txt', 'sys')
    git_repo.write('.opaque/config.yaml', _CONFIG)
    git_repo.write('eval_data/s1.json', json.dumps(
        {'sample_id': 's1', 'raw_file_name': 's1.pdf', 'input': {}, 'gold': {'vendor': 'OLD'}}
    ))
    git_repo.write('eval_data/s2.json', json.dumps(
        {'sample_id': 's2', 'raw_file_name': 's2.pdf', 'input': {}, 'gold': None}
    ))
    git_repo.commit('onboard mini')


def test_relabel_session_edits_then_single_commit(git_repo, tmp_path):
    _project(git_repo)
    client = TestClient(create_app(git_repo.path, 'demo', tracking_uri=str(tmp_path / 'mlruns')))

    listed = client.get('/api/samples').json()
    assert [s['sample_id'] for s in listed['samples']] == ['s1', 's2']
    assert listed['samples'][1]['gold'] is None  # a missing gold to fill in

    # Correct an existing gold and fill in a missing one — accumulates in working state.
    assert client.patch('/api/samples/s1', json={'gold': {'vendor': 'NEW'}}).status_code == 200
    filled = client.patch('/api/samples/s2', json={'gold': 'receipt'}).json()
    assert filled['edited'] is True
    assert set(client.get('/api/info').json()['pending']) == {'s1', 's2'}

    # One commit for the whole session, with the annotation appended (§9.2).
    body = client.post('/api/session/commit', json={'annotation': 'review pass'}).json()
    assert body['committed'] is True
    assert set(body['changed']) == {'s1', 's2'}

    head_msg = subprocess.run(
        ['git', 'log', '-1', '--format=%B'], cwd=git_repo.path, capture_output=True, text=True
    ).stdout
    assert 'Relabel 2 sample(s)' in head_msg
    assert 'review pass' in head_msg

    on_disk = json.loads((git_repo.path / 'eval_data' / 's1.json').read_text())
    assert on_disk['gold'] == {'vendor': 'NEW'}
    assert client.get('/api/info').json()['pending'] == []  # working state cleared


def test_commit_with_no_edits_is_a_noop(git_repo, tmp_path):
    _project(git_repo)
    client = TestClient(create_app(git_repo.path, 'demo', tracking_uri=str(tmp_path / 'mlruns')))
    before = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=git_repo.path, capture_output=True, text=True
    ).stdout.strip()

    assert client.post('/api/session/commit', json={}).json()['committed'] is False

    after = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=git_repo.path, capture_output=True, text=True
    ).stdout.strip()
    assert before == after  # no commit created
