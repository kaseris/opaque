"""MLflow tracking (spec §8).

One experiment per (project, tool) pair — named ``{project}/{tool}`` — with ``project_id``
also tagged on every run so runs stay filterable across the whole store. Flat runs for v1.
Params, tags, metrics, and artifacts follow §8.2.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from .prompt_impact import log_impact_report

if TYPE_CHECKING:
    from ..runner.runner import RunResult

DEFAULT_TRACKING_URI = './mlruns'


def log_run(
    result: 'RunResult',
    tracking_uri: str = DEFAULT_TRACKING_URI,
    report_path: str | Path | None = None,
) -> str:
    """Log a completed run to MLflow. Returns the MLflow run id."""
    mlflow.set_tracking_uri(_resolve_uri(tracking_uri))
    mlflow.set_experiment(f'{result.project}/{result.tool.name}')

    experiment = f'{result.project}/{result.tool.name}'
    run_name = f'{result.prompt_bundle.bundle_hash}-{result.model}-{result.timestamp}'
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_params(result))
        mlflow.set_tags(_tags(result))
        mlflow.log_metrics(result.metrics)
        _log_artifacts(result, report_path)
        run_id = run.info.run_id

    # Prompt-change-vs-performance report (§11 comparison report): regenerated across the
    # whole experiment and logged into this run, so the latest run always carries the full
    # prompt history. Runs after the current one is finished so it is included. Never allowed
    # to fail the eval — the run is already logged above.
    try:
        log_impact_report(_resolve_uri(tracking_uri), experiment, run_id)
    except Exception as exc:  # noqa: BLE001 — a report failure must not sink a logged run.
        warnings.warn(f'prompt-impact report skipped: {exc}', RuntimeWarning, stacklevel=2)

    return run_id


def _params(result: 'RunResult') -> dict:
    params = {
        'prompt_bundle_hash': result.prompt_bundle.bundle_hash,
        'model_name': result.model,
        'temperature': result.temperature,
        'task_type': result.task_type,
        'eval_set_git_commit': result.eval_set_version.git_commit or 'none',
        'eval_set_content_hash': result.eval_set_version.content_hash,
    }
    for role, fv in result.prompt_bundle.roles.items():
        params[f'prompt.{role}.git_commit'] = fv.git_commit or 'none'
        params[f'prompt.{role}.content_hash'] = fv.content_hash
    if result.field_schema_version is not None:
        params['field_schema_git_commit'] = result.field_schema_version.git_commit or 'none'
        params['field_schema_content_hash'] = result.field_schema_version.content_hash
    return params


def _tags(result: 'RunResult') -> dict:
    return {
        'project_id': result.project,
        'prompt_dirty': str(result.prompt_bundle.any_dirty),
        'git_branch': result.git_branch,
        'tool_name': result.tool.name,
    }


def _log_artifacts(result: 'RunResult', report_path: str | Path | None) -> None:
    # Per-sample raw outputs (§5).
    mlflow.log_artifacts(str(result.output_dir), artifact_path='raw_outputs')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prompts_dir = tmp / 'prompts'
        prompts_dir.mkdir()
        manifest = {}
        for role, rel in result.prompt_paths.items():
            (prompts_dir / f'{role}.txt').write_text((result.repo / rel).read_text())
            fv = result.prompt_bundle.roles[role]
            manifest[role] = {
                'path': rel,
                'git_commit': fv.git_commit,
                'content_hash': fv.content_hash,
                'dirty': fv.dirty,
            }
        (prompts_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        mlflow.log_artifacts(str(prompts_dir), artifact_path='prompts')

        if result.tool.field_schema_path:
            mlflow.log_artifact(str(result.repo / result.tool.field_schema_path))

        if result.task_type == 'classification':
            _log_classification_artifacts(tmp, result)

    if report_path is not None:
        mlflow.log_artifact(str(report_path))


def _log_classification_artifacts(tmp: Path, result: 'RunResult') -> None:
    cdir = tmp / 'classification'
    cdir.mkdir()
    arts = result.artifacts
    if arts.get('confusion_matrix') is not None:
        (cdir / 'confusion_matrix.json').write_text(json.dumps(arts['confusion_matrix'], indent=2))
    (cdir / 'per_class.json').write_text(json.dumps(arts.get('per_class', []), indent=2))
    (cdir / 'errors.json').write_text(json.dumps(arts.get('errors', []), indent=2, default=str))
    mlflow.log_artifacts(str(cdir), artifact_path='classification')


def _resolve_uri(tracking_uri: str) -> str:
    # A bare local path becomes a file:// store; explicit schemes (http, file, etc.) pass through.
    uri = tracking_uri if '://' in tracking_uri else Path(tracking_uri).resolve().as_uri()
    if uri.startswith('file:'):
        # MLflow 3 gates the filesystem store behind this opt-in; the spec (§8.3) uses ./mlruns.
        # A prototype wants the zero-config, directory-browsable file store, so opt in here.
        os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
    return uri
