"""CLI integration (spec §2 end-to-end through the Typer app)."""

from __future__ import annotations

from pathlib import Path

import mlflow
from typer.testing import CliRunner

from opaque.cli import app

DEMO = Path(__file__).resolve().parents[1] / 'examples' / 'demo_project'
runner = CliRunner()


def test_run_command_logs_run_and_report(tmp_path):
    store = tmp_path / 'mlruns'
    result = runner.invoke(
        app, ['run', str(DEMO), '--tool', 'invoices', '--tracking-uri', str(store)]
    )
    assert result.exit_code == 0, result.output
    assert 'field_accuracy' in result.output

    client = mlflow.MlflowClient(tracking_uri=store.resolve().as_uri())
    experiment = client.get_experiment_by_name('acme_demo/invoices')
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    artifacts = {a.path for a in client.list_artifacts(runs[0].info.run_id)}
    assert 'report.xlsx' in artifacts


def test_onboard_summarizes_existing_config():
    result = runner.invoke(app, ['onboard', str(DEMO)])
    assert result.exit_code == 0, result.output
    assert 'invoices' in result.output
    assert 'doc_type' in result.output


def test_report_command_writes_xlsx(tmp_path):
    out = tmp_path / 'r.xlsx'
    result = runner.invoke(app, ['report', str(DEMO), '--tool', 'invoices', '--out', str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_unknown_tool_exits_nonzero():
    result = runner.invoke(app, ['run', str(DEMO), '--tool', 'nope', '--no-log', '--no-report'])
    assert result.exit_code == 1
    assert 'Error' in result.output
