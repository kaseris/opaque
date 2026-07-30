"""Rendering a check's outcome — for the terminal, for a PR comment, and the baseline
lookup both depend on.

The delta a developer cares about is "versus what is on the integration branch today", so the
baseline always comes from the canonical ``{project}/{tool}`` experiment even when the run
itself is logged to a branch-scoped one (see ``push.experiment_for``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .push import CheckResult, ToolCheck

# Lets a later version find and edit its own comment instead of stacking new ones.
MARKER = '<!-- opaque:check -->'


def latest_metric(tracking_uri: str, experiment: str) -> tuple[str, float] | None:
    """``(metric_name, value)`` of the most recent run in ``experiment``, or ``None``.

    Never raises: a missing store, a missing experiment, or an unreadable one all mean "no
    baseline yet", which is a normal state on a project's first run.
    """
    from ..tracking import HEADLINE_METRICS, resolve_uri

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=resolve_uri(tracking_uri))
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            return None
        runs = client.search_runs(
            [exp.experiment_id], order_by=['attributes.start_time DESC'], max_results=1
        )
    except Exception:  # noqa: BLE001 — a baseline lookup must never break the push.
        return None
    if not runs:
        return None
    metrics = runs[0].data.metrics
    for name in HEADLINE_METRICS:
        if name in metrics:
            return name, metrics[name]
    return None


def _fmt(value: float | None) -> str:
    return f'{value:.4f}' if value is not None else '—'


def _delta(check: ToolCheck) -> str:
    d = check.delta
    if d is None:
        return 'baseline' if check.metric is not None else '—'
    if abs(d) < 1e-9:
        return 'no change'
    return f"{'▲' if d > 0 else '▼'} {d:+.4f}"


def render_markdown(result: CheckResult) -> str:
    """The PR comment body."""
    lines = [MARKER, '## opaque — evaluation on this branch', '']
    if not result.checks:
        lines += [f'_No run: {result.note}._']
        return '\n'.join(lines)

    lines += [
        '| tool | metric | this run | baseline (`main`) | delta | samples |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for c in result.checks:
        if c.skipped:
            lines.append(f'| `{c.tool_name}` | — | skipped | — | — | — |')
            continue
        lines.append(
            f'| `{c.tool_name}` | `{c.metric_name}` | {_fmt(c.metric)} | {_fmt(c.baseline)} | '
            f'{_delta(c)} | {c.labeled} labeled / {c.unlabeled} unlabeled |'
        )

    lines += ['', '<details><summary>What changed</summary>', '']
    for c in result.checks:
        lines.append(f'**`{c.tool_name}`** — tracked files changed in this push:')
        lines += [f'- `{p}`' for p in c.changed_prompts]
        if c.eval_data_changed:
            lines.append(
                '- ⚠️ eval data also changed in this push, so the delta above reflects both a '
                'prompt edit and moved gold labels — it is not attributable to the prompt alone.'
            )
        if c.skipped:
            lines.append(f'- ⚠️ skipped: {c.skipped}')
        if c.baseline is None and not c.skipped:
            lines.append('- No prior run on the integration branch, so there is no baseline yet.')
        lines.append('')
    lines += ['</details>', '']

    ran = result.ran
    if ran:
        lines.append(
            f'Runs logged locally to `{ran[0].experiment}` — browse with `opaque ui`. '
            'Full prompt-vs-metric history is in each run\'s `prompt_impact.html` artifact.'
        )
    return '\n'.join(lines)


def render_terminal(result: CheckResult) -> str:
    """The compact form printed during the push."""
    if not result.checks:
        return f'opaque: {result.note}.'
    lines = [f'opaque: evaluated {len(result.ran)} tool(s) on {result.branch}']
    for c in result.checks:
        if c.skipped:
            lines.append(f'  • {c.tool_name}: skipped — {c.skipped}')
            continue
        context = (
            'no baseline on the integration branch yet'
            if c.baseline is None
            else f'baseline {_fmt(c.baseline)}, {_delta(c)}'
        )
        lines.append(
            f'  • {c.tool_name}: {c.metric_name}={_fmt(c.metric)} ({context})  {c.labeled} labeled'
        )
        if c.eval_data_changed:
            lines.append('    ⚠ eval data changed too — delta is not prompt-attributable')
    return '\n'.join(lines)


def post_comment(repo: str | Path, body: str, branch: str | None = None) -> str:
    """Post the summary to the branch's PR via ``gh``. Returns a status line.

    Best-effort by design: no ``gh``, no PR, or no auth are all normal (the PR may not exist
    yet on the push that creates it), and none of them should fail a push.
    """
    if shutil.which('gh') is None:
        return 'gh not installed — comment not posted'
    args = ['gh', 'pr', 'comment']
    if branch:
        args.append(branch)
    args += ['--body-file', '-']
    proc = subprocess.run(args, cwd=str(repo), input=body, capture_output=True, text=True)
    if proc.returncode != 0:
        return f'gh comment skipped: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"}'
    return f'comment posted: {proc.stdout.strip()}'
