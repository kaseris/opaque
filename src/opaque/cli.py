"""Opaque command-line interface (Typer).

Commands:
  onboard   scaffold or inspect a project's .opaque/config.yaml (§3)
  run       run + score + report + log an evaluation (§2)
  report    on-demand Excel report for an extraction run (§10, §11)
  ui        launch MLflow's built-in UI over the tracking store (§8.3)
  relabel   launch the relabeling / data-cleaning UI (§9)

Heavy dependencies (MLflow, FastAPI) are imported lazily inside commands so `--help` is fast.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help='LLM evaluation tracking tool (see CLAUDE.md).')


def _parse_prompts(items: list[str] | None) -> dict[str, str] | None:
    if not items:
        return None
    prompts: dict[str, str] = {}
    for item in items:
        if '=' not in item:
            raise typer.BadParameter(f"expected role=path, got '{item}'")
        role, path = item.split('=', 1)
        prompts[role.strip()] = path.strip()
    return prompts


@app.command()
def onboard(
    repo: str = typer.Argument(..., help='path to the target project repo'),
    project: str = typer.Option(None, help='project id (defaults to repo folder name)'),
):
    """Create a starter config if absent, or summarize the existing one."""
    from .config.loader import config_path, load_config, scaffold_config

    if config_path(repo).exists():
        cfg = load_config(repo)
        typer.echo(f"Project '{cfg.project}' is onboarded with {len(cfg.tools)} tool(s):")
        for tool in cfg.tools:
            typer.echo(f'  • {tool.name}  ({tool.task_type})  eval_data={tool.eval_data}')
        return
    project = project or Path(repo).resolve().name
    path = scaffold_config(repo, project)
    typer.echo(f'Created {path}. Edit it, commit it, then run: opaque run {repo} --tool <name>')


@app.command()
def run(
    repo: str = typer.Argument(..., help='path to the target project repo'),
    tool: str = typer.Option(..., '--tool', '-t', help='tool name from the config'),
    prompt: list[str] = typer.Option(None, '--prompt', '-p', help='role=path (repeatable)'),
    task_type: str = typer.Option(None, '--task-type', help='override the tool task type'),
    model: str = typer.Option(None, help='override model name'),
    temperature: float = typer.Option(None, help='override temperature'),
    eval_data: str = typer.Option(None, help='override eval-data path'),
    allow_dirty: bool = typer.Option(False, '--allow-dirty', help='permit uncommitted prompts (§4.1)'),
    tracking_uri: str = typer.Option('./mlruns', '--tracking-uri', help='MLflow store'),
    no_report: bool = typer.Option(False, '--no-report', help='skip Excel report'),
    no_log: bool = typer.Option(False, '--no-log', help='skip MLflow logging'),
):
    """Run an evaluation: predict + score, log to MLflow, and (for extraction) write the report."""
    from . import pipeline
    from .runner import RunnerError

    try:
        out = pipeline.evaluate(
            repo, tool,
            prompts=_parse_prompts(prompt), task_type=task_type, model=model,
            temperature=temperature, eval_data=eval_data, allow_dirty=allow_dirty,
            tracking_uri=tracking_uri, report=not no_report, log=not no_log,
        )
    except (RunnerError, FileNotFoundError, KeyError) as exc:
        typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    _print_run_summary(out)


@app.command()
def report(
    repo: str = typer.Argument(...),
    tool: str = typer.Option(..., '--tool', '-t'),
    prompt: list[str] = typer.Option(None, '--prompt', '-p', help='role=path (repeatable)'),
    out: str = typer.Option('report.xlsx', '--out', '-o', help='output .xlsx path'),
    allow_dirty: bool = typer.Option(False, '--allow-dirty'),
):
    """Generate an extraction report on demand (no MLflow logging)."""
    from . import pipeline
    from .report import build_report
    from .runner import RunnerError

    try:
        out_result = pipeline.evaluate(
            repo, tool, prompts=_parse_prompts(prompt),
            allow_dirty=allow_dirty, report=False, log=False,
        )
    except (RunnerError, FileNotFoundError, KeyError) as exc:
        typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if out_result.result.task_type != 'extraction':
        typer.secho('The Excel report is defined for extraction tools only (§10).', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    path = build_report(out_result.result, out)
    typer.echo(f'Wrote {path}')


@app.command()
def ui(
    tracking_uri: str = typer.Option('./mlruns', '--tracking-uri'),
    host: str = typer.Option('127.0.0.1'),
    port: int = typer.Option(5000),
):
    """Launch MLflow's built-in run browser (§8.3)."""
    import os
    import subprocess

    env = {**os.environ, 'MLFLOW_ALLOW_FILE_STORE': 'true'}
    typer.echo(f'MLflow UI → http://{host}:{port}  (store: {tracking_uri})')
    subprocess.run(
        ['mlflow', 'ui', '--backend-store-uri', tracking_uri, '--host', host, '--port', str(port)],
        env=env,
    )


@app.command()
def relabel(
    repo: str = typer.Argument(...),
    tool: str = typer.Option(..., '--tool', '-t'),
    tracking_uri: str = typer.Option('./mlruns', '--tracking-uri'),
    host: str = typer.Option('127.0.0.1'),
    port: int = typer.Option(8000),
    no_browser: bool = typer.Option(False, '--no-browser'),
):
    """Launch the relabeling / data-cleaning UI over the project's eval data (§9)."""
    from .relabel.app import serve

    serve(repo, tool, tracking_uri=tracking_uri, host=host, port=port, open_browser=not no_browser)


@app.command()
def check(
    repo: str = typer.Option('.', '--repo', help='path to the target project repo'),
    stdin: bool = typer.Option(False, '--stdin', help='read pre-push ref lines from stdin'),
    base: str = typer.Option(None, '--base', help='compare from this rev (default: merge-base)'),
    head: str = typer.Option(None, '--head', help='compare to this rev (default: HEAD)'),
    tool: list[str] = typer.Option(None, '--tool', '-t', help='limit to these tools (repeatable)'),
    tracking_uri: str = typer.Option('./mlruns', '--tracking-uri', help='MLflow store'),
    gate: bool = typer.Option(False, '--gate', help='exit non-zero on regression (blocks the push)'),
    tolerance: float = typer.Option(0.0, '--tolerance', help='allowed drop before --gate fails'),
    comment: bool = typer.Option(False, '--comment', help="post the summary to the branch's PR via gh"),
    out: str = typer.Option(None, '--out', help='write the markdown summary to this path'),
):
    """Evaluate the tools whose prompts changed in a push — the pre-push hook's entry point.

    Runs locally against the developer's own eval data, credentials, and MLflow store, since
    none of those survive an ephemeral CI runner. Exits 0 even when a tool regresses unless
    --gate is set: a push should not be held hostage to an LLM eval by default.
    """
    from .config.loader import ConfigNotFound
    from .hooks import push as hook_push
    from .hooks import summary as hook_summary

    refs = hook_push.parse_push_refs(_read_stdin()) if stdin else None

    try:
        result = hook_push.check(
            repo, refs=refs, base=base, head=head, tools=list(tool) if tool else None,
            tracking_uri=tracking_uri,
        )
    except ConfigNotFound:
        return  # Not an onboarded repo — the hook stays silent rather than nagging.
    except Exception as exc:  # noqa: BLE001 — an eval problem must not strand a push.
        typer.secho(f'opaque check failed: {exc}', fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1 if gate else 0)

    typer.echo(hook_summary.render_terminal(result))

    body = hook_summary.render_markdown(result)
    if out:
        Path(out).write_text(body)
        typer.echo(f'  summary: {out}')
    if comment and result.ran:
        typer.echo(f'  {hook_summary.post_comment(repo, body, result.branch)}')

    worst = result.worst_delta()
    if gate and worst is not None and worst < -tolerance:
        typer.secho(
            f'opaque: blocking push — headline metric dropped {worst:+.4f} '
            f'(tolerance {tolerance:.4f}). Push anyway with OPAQUE_SKIP=1 git push.',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


hook_app = typer.Typer(add_completion=False, help="Manage the pre-push hook that triggers local evals.")
app.add_typer(hook_app, name='hook')


@hook_app.command('install')
def hook_install(
    repo: str = typer.Argument('.', help='path to the target project repo'),
    gate: bool = typer.Option(False, '--gate', help='block the push when the metric regresses'),
    comment: bool = typer.Option(False, '--comment', help='post results to the PR via gh'),
    tracking_uri: str = typer.Option(None, '--tracking-uri', help='pin the MLflow store (absolute path recommended)'),
    python: str = typer.Option(None, '--python', help='interpreter to run opaque with (default: current)'),
    force: bool = typer.Option(False, '--force', help='replace an existing non-opaque hook'),
):
    """Install the pre-push hook into a repo."""
    from .hooks.installer import HookError, install

    try:
        path = install(
            repo, python=python, gate=gate, comment=comment,
            tracking_uri=tracking_uri, force=force,
        )
    except HookError as exc:
        typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f'Installed {path}')
    typer.echo('  Pushes that touch a tracked prompt or field-schema file now run an eval locally.')
    typer.echo('  Skip one push with: OPAQUE_SKIP=1 git push')


@hook_app.command('uninstall')
def hook_uninstall(repo: str = typer.Argument('.', help='path to the target project repo')):
    """Remove opaque's pre-push hook."""
    from .hooks.installer import HookError, uninstall

    try:
        path = uninstall(repo)
    except HookError as exc:
        typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f'Removed {path}' if path else 'No hook installed.')


@hook_app.command('status')
def hook_status(repo: str = typer.Argument('.', help='path to the target project repo')):
    """Report whether the pre-push hook is installed."""
    from .hooks.installer import HookError, hook_path, is_ours

    try:
        path = hook_path(repo)
    except HookError as exc:
        typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not path.exists():
        typer.echo(f'Not installed ({path} absent).')
    elif is_ours(path):
        typer.echo(f'Installed: {path}')
    else:
        typer.secho(f'A non-opaque {path} exists — install with --force to replace it.', fg=typer.colors.YELLOW)


def _read_stdin() -> str:
    import sys

    return '' if sys.stdin.isatty() else sys.stdin.read()


def _print_run_summary(out) -> None:
    r = out.result
    typer.secho(f'{r.project}/{r.tool.name}', bold=True, nl=False)
    typer.echo(f'  ({r.task_type}, model={r.model})')
    headline = 'field_accuracy' if r.task_type == 'extraction' else 'accuracy'
    if headline in r.metrics:
        typer.echo(f'  {headline}: {r.metrics[headline]:.4f}')
    typer.echo(
        f"  samples: {int(r.metrics.get('labeled_sample_count', 0))} labeled / "
        f"{int(r.metrics.get('unlabeled_sample_count', 0))} unlabeled"
    )
    if r.prompt_bundle.any_dirty:
        typer.secho(f"  ⚠ dirty prompts: {', '.join(r.prompt_bundle.dirty_roles)}", fg=typer.colors.YELLOW)
    if out.report_path:
        typer.echo(f'  report: {out.report_path}')
    if out.run_id:
        typer.echo(f'  mlflow run: {out.run_id}  (browse with: opaque ui)')


def main() -> None:
    app()


if __name__ == '__main__':
    main()
