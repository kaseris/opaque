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
