#!/usr/bin/env python3
"""Regenerate the prompt-change-vs-performance report for an experiment on demand.

This is normally unnecessary — opaque logs the report into every run automatically (see
``opaque.tracking.prompt_impact``). Use this to re-render across all runs and optionally
re-log into the latest one, e.g. after pulling runs from another machine.

Usage:
  uv run python scripts/prompt_impact.py --experiment "eml-extract-model/id_card"
      [--tracking-uri ./mlruns] [--out prompt_impact.html] [--log]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlflow.tracking import MlflowClient

from opaque.tracking.prompt_impact import ARTIFACT_NAME, _load_run, build_html, log_impact_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True)
    ap.add_argument('--tracking-uri', default='./mlruns')
    ap.add_argument('--out', default='prompt_impact.html')
    ap.add_argument('--log', action='store_true', help='also re-log into the latest run')
    args = ap.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f'No experiment named {args.experiment!r} in {args.tracking_uri}')
    runs = client.search_runs([exp.experiment_id], order_by=['attributes.start_time ASC'])
    if not runs:
        raise SystemExit('Experiment has no runs.')

    Path(args.out).write_text(build_html(args.experiment, [_load_run(r) for r in runs]))
    print(f'Wrote {args.out}  ({len(runs)} runs)')

    if args.log:
        latest = runs[-1].info.run_id
        log_impact_report(args.tracking_uri, args.experiment, latest)
        print(f'Logged {ARTIFACT_NAME} into run {latest[:8]}')


if __name__ == '__main__':
    main()
