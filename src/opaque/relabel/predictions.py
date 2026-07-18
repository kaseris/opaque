"""Fetch the model's predictions for the current prompt version, to show alongside gold
during relabeling (spec §9.1) — turning relabeling into a review pass rather than blind entry.

Predictions come from the most recent MLflow run whose ``prompt_bundle_hash`` matches the
prompt files currently checked out. Best-effort: any failure (no store, no matching run)
yields an empty mapping, and the UI falls back to plain gold editing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.models import ToolConfig
from ..tracking.mlflow_logger import _resolve_uri
from ..versioning.versions import prompt_bundle


def latest_predictions(
    repo: str | Path,
    project: str,
    tool: ToolConfig,
    tracking_uri: str,
) -> dict[str, Any]:
    try:
        return _latest_predictions(repo, project, tool, tracking_uri)
    except Exception:
        return {}


def _latest_predictions(repo, project, tool, tracking_uri) -> dict[str, Any]:
    import mlflow

    bundle = prompt_bundle(repo, tool.prompts)
    uri = _resolve_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=uri)

    experiment = client.get_experiment_by_name(f'{project}/{tool.name}')
    if experiment is None:
        return {}
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"params.prompt_bundle_hash = '{bundle.bundle_hash}'",
        order_by=['attributes.start_time DESC'],
        max_results=1,
    )
    if not runs:
        return {}

    local = mlflow.artifacts.download_artifacts(
        run_id=runs[0].info.run_id, artifact_path='raw_outputs', tracking_uri=uri
    )
    predictions: dict[str, Any] = {}
    for f in sorted(Path(local).glob('*.json')):
        record = json.loads(f.read_text())
        predictions[record['sample_id']] = record.get('prediction')
    return predictions
