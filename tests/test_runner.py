"""Evaluation runner (spec §2) — against the bundled demo project and the dirty-prompt
reproducibility policy (§4.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opaque.runner import DirtyPromptError, run

DEMO = Path(__file__).resolve().parents[1] / 'examples' / 'demo_project'

_MINIMAL_SCRIPT = '''\
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--input'); p.add_argument('--out'); p.add_argument('--system')
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for f in sorted(Path(a.input).glob('*.json')):
    s = json.loads(f.read_text())
    s['prediction'] = s.get('gold')   # perfect predictor
    (out / f.name).write_text(json.dumps(s))
'''

_MINIMAL_CONFIG = '''\
project: mini
tools:
  - name: demo
    task_type: extraction
    prompts:
      system: prompts/system.txt
    eval_data: eval_data
    invocation:
      command: python eval_script.py --input {input} --out {output_dir} --system {prompt.system}
'''


def test_extraction_run_against_demo():
    r = run(DEMO, 'invoices')
    assert r.project == 'acme_demo'
    assert r.metrics['labeled_sample_count'] == 3
    assert r.metrics['unlabeled_sample_count'] == 1
    assert r.metrics['fields_correct'] == 16
    assert r.metrics['fields_extra'] == 1
    assert r.metrics['field_accuracy'] == pytest.approx(16 / 27)
    # The dropped array item shows as a run of incorrects + a hallucinated extra (§7.3).
    inv3 = {(fr.json_path, fr.status.value) for fr in r.field_results if fr.sample_id == 'inv-003'}
    assert ('lineItems.0.sku', 'incorrect') in inv3
    assert ('poNumber', 'extra') in inv3


def test_classification_run_against_demo():
    r = run(DEMO, 'doc_type')
    assert r.metrics['accuracy'] == pytest.approx(0.75)
    assert r.artifacts['confusion_matrix']['labels'] == ['contract', 'invoice', 'receipt']


def _minimal_project(git_repo):
    git_repo.write('prompts/system.txt', 'system prompt v1')
    git_repo.write('eval_script.py', _MINIMAL_SCRIPT)
    git_repo.write('.opaque/config.yaml', _MINIMAL_CONFIG)
    git_repo.write('eval_data/s1.json', json.dumps(
        {'sample_id': 's1', 'raw_file_name': 's1.pdf', 'input': {}, 'gold': {'a': 'x'}}
    ))
    git_repo.commit('onboard mini project')


def test_dirty_prompt_blocks_then_overridable(git_repo):
    _minimal_project(git_repo)

    clean = run(git_repo.path, 'demo')
    assert clean.metrics['field_accuracy'] == pytest.approx(1.0)
    assert clean.prompt_bundle.any_dirty is False

    # Uncommitted edit to a selected prompt file blocks the run (§4.1)...
    git_repo.write('prompts/system.txt', 'edited but not committed')
    with pytest.raises(DirtyPromptError):
        run(git_repo.path, 'demo')

    # ...unless explicitly overridden, in which case dirtiness is recorded.
    overridden = run(git_repo.path, 'demo', allow_dirty=True)
    assert overridden.prompt_bundle.any_dirty is True
    assert 'system' in overridden.prompt_bundle.dirty_roles
