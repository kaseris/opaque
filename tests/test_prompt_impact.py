"""Prompt-change-vs-performance report (§11 comparison report).

Auto-logged on every run (see mlflow_logger); these cover the pure HTML builder and the
end-to-end auto-log behaviour.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from opaque.runner import run
from opaque.tracking import log_run
from opaque.tracking.prompt_impact import build_html

DEMO = Path(__file__).resolve().parents[1] / 'examples' / 'demo_project'


def _mk(run_id, metric, system_text, chash, eval_set='ev1'):
    return {
        'run_id': run_id,
        'short': run_id[:8],
        'metric_name': 'field_accuracy',
        'metric': metric,
        'eval_set': eval_set,
        'roles': {'system': {'text': system_text, 'content_hash': chash, 'git_commit': 'abc1234'}},
    }


def test_build_html_shows_diff_and_regression_delta():
    runs = [
        _mk('r1baseline', 1.0, 'extract all six fields\nbe precise', 'hashAAAA'),
        _mk('r2degraded', 0.4, 'extract only the name\nleave the rest null', 'hashBBBB'),
    ]
    out = build_html('proj/tool', runs)

    # The changed prompt line shows up as a diff (removed + added), and the metric drop is a
    # down-delta chip — the whole point of the report.
    assert 'class="ln del"' in out
    assert 'class="ln add"' in out
    assert 'leave the rest null' in out
    assert 'class="delta down"' in out
    assert '-0.600' in out
    # best/worst badges when more than one scored run.
    assert 'class="tag best"' in out and 'class="tag worst"' in out


def test_build_html_flags_unchanged_prompt_and_eval_set_change():
    same = 'identical prompt text'
    unchanged = build_html('p/t', [_mk('r1', 0.9, same, 'h1'), _mk('r2', 0.7, same, 'h1')])
    # Same content hash -> no diff, explicit "unchanged" note, and no misleading diff lines.
    assert 'nochange' in unchanged
    assert 'class="ln del"' not in unchanged

    crossed = build_html('p/t', [
        _mk('r1', 0.9, 'a', 'h1', eval_set='ev1'),
        _mk('r2', 0.9, 'b', 'h2', eval_set='ev2'),
    ])
    assert 'evalwarn' in crossed  # comparability warning when eval-set version changes


def test_run_auto_logs_prompt_impact_artifact(tmp_path):
    """Every logged run carries a prompt_impact.html artifact — no manual step."""
    store = tmp_path / 'mlruns'
    run_id = log_run(run(DEMO, 'invoices'), tracking_uri=str(store))

    client = mlflow.MlflowClient(tracking_uri=store.resolve().as_uri())
    root = {a.path for a in client.list_artifacts(run_id)}
    assert 'prompt_impact.html' in root

    local = client.download_artifacts(run_id, 'prompt_impact.html')
    assert 'prompt change' in Path(local).read_text().lower()
