"""MLflow tracking (spec §8)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from opaque.report import build_report
from opaque.runner import run
from opaque.tracking import log_run

DEMO = Path(__file__).resolve().parents[1] / 'examples' / 'demo_project'


def test_extraction_run_logged_with_params_metrics_artifacts(tmp_path):
    result = run(DEMO, 'invoices')
    report = build_report(result, tmp_path / 'report.xlsx')
    store = tmp_path / 'mlruns'
    run_id = log_run(result, tracking_uri=str(store), report_path=report)

    client = mlflow.MlflowClient(tracking_uri=store.resolve().as_uri())
    data = client.get_run(run_id)

    # Params (§8.2): bundle hash, per-role hashes, eval-set + field-schema versions.
    assert data.data.params['task_type'] == 'extraction'
    assert data.data.params['prompt_bundle_hash'] == result.prompt_bundle.bundle_hash
    assert data.data.params['prompt.system.content_hash'] == result.prompt_bundle.roles['system'].content_hash
    assert 'eval_set_content_hash' in data.data.params
    assert 'field_schema_content_hash' in data.data.params

    # Tags + metrics.
    assert data.data.tags['project_id'] == 'acme_demo'
    assert data.data.tags['tool_name'] == 'invoices'
    assert float(data.data.metrics['field_accuracy']) == pytest.approx(16 / 27)

    # Artifacts (§8.2).
    root = {a.path for a in client.list_artifacts(run_id)}
    assert {'raw_outputs', 'prompts', 'field_schema.yaml', 'report.xlsx'} <= root
    prompt_arts = {a.path for a in client.list_artifacts(run_id, 'prompts')}
    assert 'prompts/manifest.json' in prompt_arts
    # raw_outputs holds only the per-sample JSON — the report is not duplicated in there.
    raw = {a.path for a in client.list_artifacts(run_id, 'raw_outputs')}
    assert raw and all(p.endswith('.json') for p in raw)


def test_experiment_named_project_slash_tool(tmp_path):
    result = run(DEMO, 'doc_type')
    store = tmp_path / 'mlruns'
    run_id = log_run(result, tracking_uri=str(store))

    client = mlflow.MlflowClient(tracking_uri=store.resolve().as_uri())
    data = client.get_run(run_id)
    experiment = client.get_experiment(data.info.experiment_id)
    assert experiment.name == 'acme_demo/doc_type'
    # Classification logs its confusion matrix as an artifact.
    classif = {a.path for a in client.list_artifacts(run_id, 'classification')}
    assert 'classification/confusion_matrix.json' in classif
