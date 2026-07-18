"""Prompt-change-vs-performance report (the view MLflow's UI doesn't provide).

MLflow shows the prompt *hash* params moving between runs, but never the prompt *text* that
changed — so you can't see "these words -> this accuracy delta" in one place. Opaque already
logs each run's prompt files (``prompts/<role>.txt`` + ``manifest.json``) and the git commit
as a param, so the data is all there; this stitches an experiment's runs into a timeline that
diffs consecutive prompt versions per role and pins each edit to its headline-metric delta.

It is generated and logged automatically on every ``opaque run`` (see ``mlflow_logger``), so
the latest run always carries a ``prompt_impact.html`` artifact showing history up to itself.
The output is a self-contained HTML fragment (no ``<html>/<head>/<body>``) so it renders as a
standalone file, an MLflow artifact, and a claude.ai Artifact alike.
"""

from __future__ import annotations

import difflib
import html
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# Headline metric per task type — first one present on a run wins.
HEADLINE_METRICS = ('field_accuracy', 'accuracy')

ARTIFACT_NAME = 'prompt_impact.html'


def log_impact_report(tracking_uri: str, experiment: str, run_id: str) -> str | None:
    """Build the impact report for ``experiment`` and log it into ``run_id``.

    Returns the artifact name on success, or ``None`` if there was nothing to report.
    Reads runs (including the one just logged) via an MlflowClient, so ``run_id``'s own
    prompt artifacts must already be on the store before this is called.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None
    runs = client.search_runs([exp.experiment_id], order_by=['attributes.start_time ASC'])
    if not runs:
        return None

    loaded = [_load_run(r) for r in runs]
    report_html = build_html(experiment, loaded)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ARTIFACT_NAME
        path.write_text(report_html)
        client.log_artifact(run_id, str(path))
    return ARTIFACT_NAME


def _artifacts_dir(run) -> Path:
    uri = run.info.artifact_uri
    parsed = urlparse(uri)
    return Path(parsed.path if parsed.scheme == 'file' else uri)


def _load_run(run) -> dict:
    """Pull the headline metric + per-role prompt {text, hash, commit} for one run."""
    metric = next((m for m in HEADLINE_METRICS if m in run.data.metrics), None)
    prompts_dir = _artifacts_dir(run) / 'prompts'
    manifest = {}
    manifest_file = prompts_dir / 'manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
    roles = {}
    for role, meta in manifest.items():
        txt = prompts_dir / f'{role}.txt'
        roles[role] = {
            'text': txt.read_text() if txt.exists() else '',
            'content_hash': meta.get('content_hash', ''),
            'git_commit': (meta.get('git_commit') or '')[:7],
        }
    return {
        'run_id': run.info.run_id,
        'short': run.info.run_id[:8],
        'metric_name': metric,
        'metric': run.data.metrics.get(metric) if metric else None,
        'eval_set': run.data.params.get('eval_set_content_hash', ''),
        'roles': roles,
    }


def _diff_html(old: str, new: str) -> str:
    lines = []
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=2):
        if line.startswith('+++') or line.startswith('---'):
            continue
        esc = html.escape(line)
        if line.startswith('@@'):
            cls = 'hunk'
        elif line.startswith('+'):
            cls = 'add'
        elif line.startswith('-'):
            cls = 'del'
        else:
            cls = 'ctx'
        lines.append(f'<span class="ln {cls}">{esc or "&nbsp;"}</span>')
    return '<div class="diff">' + ''.join(lines) + '</div>' if lines else ''


def _delta_chip(delta: float | None) -> str:
    if delta is None:
        return '<span class="delta base">baseline</span>'
    if abs(delta) < 1e-9:
        return '<span class="delta flat">no change</span>'
    cls, arrow = ('up', '▲') if delta > 0 else ('down', '▼')
    return f'<span class="delta {cls}">{arrow} {delta:+.3f}</span>'


def _fmt_metric(v: float | None) -> str:
    return f'{v:.4f}' if v is not None else '—'


def build_html(experiment: str, runs: list[dict]) -> str:
    """Render the impact timeline. ``runs`` is oldest-first (see ``_load_run``)."""
    metric_name = next((r['metric_name'] for r in runs if r['metric_name']), 'metric')
    gen = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    scored = [r for r in runs if r['metric'] is not None]
    best = max(scored, key=lambda r: r['metric'], default=None)
    worst = min(scored, key=lambda r: r['metric'], default=None)

    nodes = []
    for i, run in enumerate(runs):
        prev = runs[i - 1] if i else None
        delta = None
        if prev and run['metric'] is not None and prev['metric'] is not None:
            delta = run['metric'] - prev['metric']

        transition = ''
        if prev:
            blocks = []
            for role in sorted(set(prev['roles']) | set(run['roles'])):
                p = prev['roles'].get(role, {})
                c = run['roles'].get(role, {})
                if p.get('content_hash') == c.get('content_hash'):
                    continue
                head = (
                    f'<div class="role-head"><span class="role">{html.escape(role)}</span>'
                    f'<span class="hashjump"><code>{p.get("content_hash", "∅")}</code>'
                    f'<span class="arr">→</span><code>{c.get("content_hash", "∅")}</code>'
                    f'<span class="commit">git {p.get("git_commit", "∅")} → {c.get("git_commit", "∅")}</span>'
                    f'</span></div>'
                )
                blocks.append(head + _diff_html(p.get('text', ''), c.get('text', '')))
            if blocks:
                inner = ''.join(blocks)
            else:
                inner = ('<div class="nochange">Prompt unchanged — metric moved for another '
                         'reason (data, model, or nondeterminism).</div>')
            evalwarn = ''
            if prev['eval_set'] != run['eval_set']:
                evalwarn = ('<div class="evalwarn">⚠ eval-set version changed '
                            f'({prev["eval_set"]} → {run["eval_set"]}) — this Δ is NOT '
                            'apples-to-apples.</div>')
            transition = (
                f'<div class="transition"><div class="tlabel">changed'
                f'<span class="dwrap">{_delta_chip(delta)}</span></div>{evalwarn}{inner}</div>'
            )

        badge = ''
        if len(scored) > 1 and best and run['run_id'] == best['run_id']:
            badge = '<span class="tag best">best</span>'
        elif len(scored) > 1 and worst and run['run_id'] == worst['run_id']:
            badge = '<span class="tag worst">worst</span>'

        rmeta = ''.join(
            f'<span>{html.escape(role)} <code>{meta["content_hash"]}</code></span>'
            for role, meta in run['roles'].items()
        )
        nodes.append(
            f'{transition}'
            f'<div class="run"><div class="dot"></div><div class="rcard">'
            f'<div class="rtop"><code class="rid">{run["short"]}</code>{badge}'
            f'<span class="metricbig">{_fmt_metric(run["metric"])}'
            f'<span class="mname">{html.escape(metric_name)}</span></span></div>'
            f'<div class="rmeta"><span>eval_set <code>{run["eval_set"] or "—"}</code></span>'
            f'{rmeta}</div></div></div>'
        )

    summary = ''
    if best and worst and best['run_id'] != worst['run_id']:
        summary = (
            '<div class="summary">'
            f'<div class="stat"><span class="sv">{len(runs)}</span><span class="sl">runs</span></div>'
            f'<div class="stat"><span class="sv good">{_fmt_metric(best["metric"])}</span>'
            f'<span class="sl">best · {best["short"]}</span></div>'
            f'<div class="stat"><span class="sv bad">{_fmt_metric(worst["metric"])}</span>'
            f'<span class="sl">worst · {worst["short"]}</span></div>'
            f'<div class="stat"><span class="sv">{best["metric"] - worst["metric"]:.3f}</span>'
            f'<span class="sl">spread</span></div></div>'
        )

    return f"""<title>Prompt Impact · {html.escape(experiment)}</title>
{_CSS}
<div class="wrap">
  <header class="phead">
    <div class="eyebrow">prompt change → performance impact</div>
    <h1>{html.escape(experiment)}</h1>
    <div class="sub">Diff of every prompt edit against its <code>{html.escape(metric_name)}</code> delta · generated {gen}</div>
  </header>
  {summary}
  <div class="timeline">
    {''.join(nodes)}
  </div>
  <footer class="pfoot">Same <code>eval_set_content_hash</code> across runs means deltas are comparable (opaque §4.2). Prompt text + git commit come straight from each run's logged artifacts.</footer>
</div>"""


_CSS = """<style>
  :root{--font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --font-mono:"SF Mono","JetBrains Mono","Roboto Mono",ui-monospace,Menlo,Consolas,monospace;
    --bg:#eef2f6;--surface:#fff;--surface-2:#f6f9fc;--surface-3:#eef3f8;--border:#dbe3ec;--border-strong:#c4d0dc;
    --text:#16202e;--text-2:#566476;--text-3:#8593a3;--accent:#0d7de6;--accent-weak:#e5f1fc;--accent-ink:#0a5fb0;
    --good:#12855a;--good-weak:#e2f4ec;--bad:#d1453b;--bad-weak:#fbe7e5;--warn:#b7791f;--warn-weak:#f8eed6;}
  @media(prefers-color-scheme:dark){:root{--bg:#0b111a;--surface:#131c28;--surface-2:#0f1722;--surface-3:#1a2634;
    --border:#26313f;--border-strong:#33414f;--text:#e5eaf0;--text-2:#9aa7b5;--text-3:#6b7888;--accent:#4098e8;
    --accent-weak:#14263a;--accent-ink:#7bb8f0;--good:#35b37e;--good-weak:#12261e;--bad:#e8635a;--bad-weak:#2a1512;
    --warn:#d99a3a;--warn-weak:#291f10;}}
  :root[data-theme=light]{--bg:#eef2f6;--surface:#fff;--surface-2:#f6f9fc;--surface-3:#eef3f8;--border:#dbe3ec;
    --border-strong:#c4d0dc;--text:#16202e;--text-2:#566476;--text-3:#8593a3;--accent:#0d7de6;--accent-weak:#e5f1fc;
    --accent-ink:#0a5fb0;--good:#12855a;--good-weak:#e2f4ec;--bad:#d1453b;--bad-weak:#fbe7e5;--warn:#b7791f;--warn-weak:#f8eed6;}
  :root[data-theme=dark]{--bg:#0b111a;--surface:#131c28;--surface-2:#0f1722;--surface-3:#1a2634;--border:#26313f;
    --border-strong:#33414f;--text:#e5eaf0;--text-2:#9aa7b5;--text-3:#6b7888;--accent:#4098e8;--accent-weak:#14263a;
    --accent-ink:#7bb8f0;--good:#35b37e;--good-weak:#12261e;--bad:#e8635a;--bad-weak:#2a1512;--warn:#d99a3a;--warn-weak:#291f10;}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:var(--font-sans);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
  code{font-family:var(--font-mono)}
  .wrap{max-width:920px;margin:0 auto;padding:28px 20px 60px}
  .phead{margin-bottom:20px}
  .eyebrow{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent-ink);font-weight:700}
  .phead h1{font-family:var(--font-mono);font-size:24px;font-weight:700;letter-spacing:-.01em;margin:6px 0 4px}
  .phead .sub{color:var(--text-2);font-size:13px}
  .phead .sub code,.pfoot code{background:var(--surface-3);padding:1px 5px;border-radius:4px;font-size:12px}
  .summary{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}
  .stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;min-width:110px;box-shadow:0 1px 2px rgba(22,32,46,.05)}
  .stat .sv{display:block;font-family:var(--font-mono);font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
  .stat .sv.good{color:var(--good)}.stat .sv.bad{color:var(--bad)}
  .stat .sl{font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em}
  .timeline{position:relative;padding-left:26px}
  .timeline:before{content:"";position:absolute;left:6px;top:8px;bottom:8px;width:2px;background:var(--border)}
  .run{position:relative;margin:0 0 6px}
  .dot{position:absolute;left:-26px;top:16px;width:13px;height:13px;border-radius:50%;background:var(--accent);border:3px solid var(--bg)}
  .rcard{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;box-shadow:0 1px 2px rgba(22,32,46,.05)}
  .rtop{display:flex;align-items:center;gap:10px}
  .rid{font-size:13px;color:var(--accent-ink);font-weight:600}
  .metricbig{margin-left:auto;font-family:var(--font-mono);font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
  .metricbig .mname{font-family:var(--font-sans);font-size:10px;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;margin-left:8px}
  .rmeta{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:8px;font-size:11.5px;color:var(--text-2)}
  .rmeta code{background:var(--surface-3);padding:1px 5px;border-radius:4px}
  .tag{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;padding:2px 7px;border-radius:999px}
  .tag.best{color:var(--good);background:var(--good-weak)}.tag.worst{color:var(--bad);background:var(--bad-weak)}
  .transition{position:relative;margin:6px 0 6px}
  .tlabel{font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.06em;font-weight:700;display:flex;align-items:center;gap:10px;margin:10px 0 6px}
  .delta{font-family:var(--font-mono);font-size:12px;font-weight:700;padding:2px 8px;border-radius:5px;font-variant-numeric:tabular-nums;text-transform:none;letter-spacing:0}
  .delta.up{color:var(--good);background:var(--good-weak)}.delta.down{color:var(--bad);background:var(--bad-weak)}
  .delta.flat,.delta.base{color:var(--text-3);background:var(--surface-3)}
  .role-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:2px 0 4px;font-size:12px}
  .role{font-family:var(--font-mono);font-weight:700;color:var(--text)}
  .hashjump{display:flex;align-items:center;gap:6px;color:var(--text-2);font-size:11.5px}
  .hashjump code{background:var(--surface-3);padding:1px 5px;border-radius:4px}
  .hashjump .arr{color:var(--text-3)}.hashjump .commit{color:var(--text-3);margin-left:4px}
  .diff{font-family:var(--font-mono);font-size:12px;border:1px solid var(--border);border-radius:7px;overflow-x:auto;background:var(--surface-2);margin-bottom:8px}
  .diff .ln{display:block;padding:1px 12px;white-space:pre;border-left:3px solid transparent}
  .diff .add{background:var(--good-weak);color:var(--good);border-left-color:var(--good)}
  .diff .del{background:var(--bad-weak);color:var(--bad);border-left-color:var(--bad)}
  .diff .ctx{color:var(--text-2)}
  .diff .hunk{color:var(--accent-ink);background:var(--surface-3)}
  .nochange{font-size:12.5px;color:var(--text-2);background:var(--surface-2);border:1px dashed var(--border-strong);border-radius:7px;padding:8px 12px}
  .evalwarn{font-size:12px;color:var(--warn);background:var(--warn-weak);border:1px solid var(--warn);border-radius:6px;padding:6px 10px;margin-bottom:8px}
  .pfoot{margin-top:24px;font-size:12px;color:var(--text-3);line-height:1.7}
</style>"""
