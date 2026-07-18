"""End-to-end orchestration: run (predict + score) → Excel report → MLflow log.

Runs are combined (§1), so this is the single high-level entry point the CLI and tests share.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .report import build_report
from .runner import RunResult, run
from .tracking import DEFAULT_TRACKING_URI, log_run


@dataclass
class PipelineResult:
    result: RunResult
    run_id: str | None
    report_path: Path | None


def evaluate(
    repo: str | Path,
    tool: str,
    *,
    prompts: dict[str, str] | None = None,
    task_type: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    eval_data: str | None = None,
    allow_dirty: bool = False,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    report: bool = True,
    log: bool = True,
) -> PipelineResult:
    result = run(
        repo, tool,
        prompts=prompts, task_type=task_type, model=model,
        temperature=temperature, eval_data=eval_data, allow_dirty=allow_dirty,
    )

    report_path: Path | None = None
    # The Excel report is an extraction deliverable (§10). Write it outside output_dir so the
    # logged raw_outputs artifact stays limited to the per-sample JSON.
    if report and result.task_type == 'extraction':
        report_dir = Path(tempfile.mkdtemp(prefix='opaque-report-'))
        report_path = build_report(result, report_dir / 'report.xlsx')

    run_id = log_run(result, tracking_uri=tracking_uri, report_path=report_path) if log else None
    return PipelineResult(result=result, run_id=run_id, report_path=report_path)
